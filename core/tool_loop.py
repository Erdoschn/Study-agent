from typing import Any
from .__debug__ import debug
from .state import AgentStep


class ToolExecutor:
    def __init__(self, search_router=None):
        self.search_router = search_router

    def execute(self, tool: str, arguments: dict[str, Any]) -> Any:
        debug.log("ToolExecutor", f"EXECUTE → {tool}")
        debug.log("ToolExecutor", f"ARGS → {arguments}")
        if tool == "search": return self._search(arguments)
        if tool == "calculate": return self._calculate(arguments)
        if tool == "verify": return self._verify(arguments)
        raise ValueError(f"未知工具：{tool}")

    def _search(self, arguments):
        if self.search_router is None: raise RuntimeError("SearchRouter 尚未配置。")
        from tools.search import SearchQuery
        query = SearchQuery(query=str(arguments.get("query", "")), source=str(arguments.get("source", "arxiv")), categories=list(arguments.get("categories", [])), max_results=int(arguments.get("max_results", 5)))
        return [{"source": item.source, "source_type": item.source_type, "title": item.title, "url": item.url, "abstract": item.abstract, "authors": item.authors, "published": item.published, "updated": item.updated, "identifier": item.identifier} for item in self.search_router.search(query)]

    def _calculate(self, arguments):
        expression = str(arguments.get("expression", ""))
        if not expression: raise ValueError("calculate 缺少 expression。")
        return self._safe_calculate(expression)

    def _verify(self, arguments):
        return {"claim": str(arguments.get("claim", "")), "evidence": arguments.get("evidence", []), "verification_status": "REQUIRES_MODEL_REVIEW"}

    @staticmethod
    def _safe_calculate(expression):
        import ast, operator
        operators = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
        def evaluate(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)): return node.value
                raise ValueError("只允许数字常量。")
            if isinstance(node, ast.UnaryOp):
                fn = operators.get(type(node.op))
                if fn is None: raise ValueError("不支持的运算符。")
                return fn(evaluate(node.operand))
            if isinstance(node, ast.BinOp):
                fn = operators.get(type(node.op))
                if fn is None: raise ValueError("不支持的运算符。")
                return fn(evaluate(node.left), evaluate(node.right))
            raise ValueError("表达式包含不允许的内容。")
        return evaluate(ast.parse(expression, mode="eval").body)


class AgentToolLoop:
    """Closed loop: decide → act → observe → feed observation into the next decision."""

    def __init__(self, reasoner, executor: ToolExecutor):
        self.reasoner, self.executor = reasoner, executor

    @staticmethod
    def _fingerprint(action, tool, arguments):
        import json
        return (action, tool, json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str))

    def _repeated_tool(self, state, decision):
        if decision.action not in {"SEARCH", "CALCULATE", "VERIFY"}: return False
        fp = self._fingerprint(decision.action, decision.tool or decision.action.lower(), decision.arguments)
        return any(self._fingerprint(s.action, s.tool or s.action.lower(), s.arguments) == fp for s in state.steps)

    def run(self, state):
        with debug.scope("AgentToolLoop", "RUN"):
            while not state.finished and state.step_count < state.max_steps:
                debug.log("AgentToolLoop", f"REASON → step={state.step_count + 1}")
                decision = self.reasoner.decide(state)
                debug.log("AgentToolLoop", f"DECISION → {decision.action}")
                if decision.goal: state.goal = decision.goal
                if decision.task_type: state.task_type = decision.task_type
                if decision.domain: state.domain = decision.domain
                if decision.claims: state.claims = decision.claims
                step_id = state.step_count + 1

                if decision.action in {"ANSWER", "STOP"}:
                    if decision.action == "STOP":
                        state.error = decision.finish_reason or decision.reasoning_summary
                    state.add_step(AgentStep(step_id=step_id, action=decision.action, model=decision.model, reasoning_summary=decision.reasoning_summary, success=decision.action != "STOP", error=state.error or ""))
                    state.finished = True
                    break

                tool = decision.tool or decision.action.lower()
                if self._repeated_tool(state, decision):
                    state.add_step(AgentStep(step_id=step_id, action="STOP", model=decision.model, tool=tool, arguments=decision.arguments or {}, reasoning_summary="检测到完全相同的工具调用，停止以避免无意义循环。", success=False, error="重复工具调用"))
                    state.error = "Agent 检测到重复工具调用，已停止。"
                    state.finished = True
                    break

                debug.log("AgentToolLoop", f"ACT → tool={tool}")
                try:
                    observation = self.executor.execute(tool, decision.arguments or {})
                    success, error = True, ""
                    debug.log("AgentToolLoop", "OBSERVE → success")
                except Exception as exc:
                    observation, success = None, False
                    error = f"{type(exc).__name__}: {exc}"
                    debug.log("AgentToolLoop", f"OBSERVE → ERROR: {error}")

                state.add_step(AgentStep(step_id=step_id, action=decision.action, model=decision.model, tool=decision.tool, arguments=decision.arguments or {}, reasoning_summary=decision.reasoning_summary, observation=observation, success=success, error=error))
                if decision.action == "SEARCH" and success and observation:
                    state.evidence.extend(observation)
                state.current_plan_step = self._next_plan_step(state, decision.action)
                debug.log("AgentToolLoop", f"OBSERVE → fed back; next_plan_step={state.current_plan_step}")
            return state

    @staticmethod
    def _next_plan_step(state, action):
        if not state.plan: return state.current_plan_step
        for i, step in enumerate(state.plan.steps):
            if i > state.current_plan_step and step.action == action: return i
        for i, step in enumerate(state.plan.steps):
            if step.action == action: return i
        return state.current_plan_step
