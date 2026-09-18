from typing import Any
from .__debug__ import debug
from .state import AgentStep
from .reasoner import ReasoningDecision


class ToolExecutor:
    def __init__(self, search_router=None):
        self.search_router = search_router

    def execute(
        self,
        tool: str,
        arguments: dict[str, Any],
        state=None,
    ) -> Any:
        debug.log("ToolExecutor", f"EXECUTE → {tool}")
        debug.log("ToolExecutor", f"ARGS → {arguments}")
        if tool == "search":
            return self._search(arguments, state)
        if tool == "calculate":
            return self._calculate(arguments)
        if tool == "verify":
            return self._verify(arguments)
        raise ValueError(f"未知工具：{tool}")

    def _search(self, arguments, state=None):
        if self.search_router is None:
            raise RuntimeError("SearchRouter 尚未配置。")

        from tools.search import SearchQuery

        explicit_source = str(arguments.get("source", "")).strip().lower()
        if explicit_source:
            source = explicit_source
            preferences = []
        else:
            source = "auto"
            preferences = list(getattr(state, "search_sources", []) or [])

        categories = arguments.get("categories", [])
        if not isinstance(categories, list):
            categories = []

        sort_by = str(arguments.get("sort_by", "")).strip()
        if not sort_by:
            sort_by = str(getattr(state, "search_sort_by", "relevance"))

        query = SearchQuery(
            query=str(arguments.get("query", "")),
            source=source,
            source_preferences=preferences,
            categories=[str(x) for x in categories],
            max_results=int(arguments.get("max_results", 5)),
            sort_by=sort_by,
            sort_order=str(arguments.get("sort_order", "descending")),
        )

        results = self.search_router.search(query)
        debug.log(
            "ToolExecutor",
            (
                f"SEARCH RESULT → source={source} "
                f"count={len(results)}"
            ),
        )
        return [
            {
                "source": item.source,
                "source_type": item.source_type,
                "title": item.title,
                "url": item.url,
                "abstract": item.abstract,
                "authors": item.authors,
                "published": item.published,
                "updated": item.updated,
                "identifier": item.identifier,
            }
            for item in results
        ]

    def _calculate(self, arguments):
        expression = str(arguments.get("expression", ""))
        if not expression:
            raise ValueError("calculate 缺少 expression。")
        return self._safe_calculate(expression)

    def _verify(self, arguments):
        claim = str(arguments.get("claim", "")).strip()
        evidence = arguments.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        return {
            "claim": claim,
            "evidence": evidence,
            "verification_status": (
                "READY_FOR_REVIEW"
                if claim or evidence
                else "INSUFFICIENT_INPUT"
            ),
            "verification_note": (
                "这是结构化验证入口；当前不对事实真伪给出自动量化结论。"
            ),
        }

    @staticmethod
    def _safe_calculate(expression):
        import ast
        import operator

        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        def evaluate(node):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("只允许数字常量。")
            if isinstance(node, ast.UnaryOp):
                fn = operators.get(type(node.op))
                if fn is None:
                    raise ValueError("不支持的运算符。")
                return fn(evaluate(node.operand))
            if isinstance(node, ast.BinOp):
                fn = operators.get(type(node.op))
                if fn is None:
                    raise ValueError("不支持的运算符。")
                return fn(evaluate(node.left), evaluate(node.right))
            raise ValueError("表达式包含不允许的内容。")

        return evaluate(ast.parse(expression, mode="eval").body)


class AgentToolLoop:
    """Closed loop: decide → act → observe → feed observation into next decision."""

    def __init__(self, reasoner, executor: ToolExecutor):
        self.reasoner, self.executor = reasoner, executor

    def _execute_tool(self, tool, arguments, state):
        """Call both new and legacy executor interfaces safely."""
        import inspect

        execute = self.executor.execute
        try:
            signature = inspect.signature(execute)
            parameters = signature.parameters
            accepts_state = (
                "state" in parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in parameters.values()
                )
            )
        except (TypeError, ValueError):
            accepts_state = False

        if accepts_state:
            return execute(tool, arguments, state=state)
        return execute(tool, arguments)

    @staticmethod
    def _fingerprint(action, tool, arguments):
        import json

        return (
            action,
            tool,
            json.dumps(
                arguments or {},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        )

    def _repeated_tool(self, state, decision):
        if decision.action not in {"SEARCH", "CALCULATE", "VERIFY"}:
            return False
        fp = self._fingerprint(
            decision.action,
            decision.tool or decision.action.lower(),
            decision.arguments,
        )
        return any(
            self._fingerprint(
                s.action,
                s.tool or s.action.lower(),
                s.arguments,
            )
            == fp
            for s in state.steps
        )

    def _verify_pending(self, state) -> bool:
        if not state.plan or not state.evidence:
            return False
        verify_index = next(
            (
                i
                for i, step in enumerate(state.plan.steps)
                if step.action == "VERIFY"
            ),
            None,
        )
        if verify_index is None:
            return False
        return not any(
            step.action == "VERIFY" and step.success
            for step in state.steps
        )

    def _force_verify(self, state, decision):
        if decision.action != "ANSWER" or not self._verify_pending(state):
            return decision
        debug.log(
            "AgentToolLoop",
            "VERIFY GATE → plan requires verification before ANSWER",
        )
        return ReasoningDecision(
            action="VERIFY",
            reasoning_summary=(
                "计划包含未执行的 VERIFY，当前已有外部证据；"
                "先执行验证入口，再决定是否回答。"
            ),
            tool="verify",
            arguments={
                "claim": state.goal or state.question,
                "evidence": list(state.evidence),
            },
            goal=decision.goal,
            task_type=decision.task_type,
            domain=decision.domain,
            claims=state.claims,
            model=decision.model,
        )

    def run(self, state):
        with debug.scope("AgentToolLoop", "RUN"):
            while (
                not state.finished
                and state.step_count < state.max_steps
            ):
                debug.log(
                    "AgentToolLoop",
                    f"REASON → step={state.step_count + 1}",
                )
                decision = self.reasoner.decide(state)
                decision = self._force_verify(state, decision)
                debug.log(
                    "AgentToolLoop",
                    f"DECISION → {decision.action}",
                )

                if decision.goal:
                    state.goal = decision.goal
                if decision.task_type:
                    state.task_type = decision.task_type
                if decision.domain:
                    state.domain = decision.domain
                if decision.claims:
                    state.claims = decision.claims

                step_id = state.step_count + 1

                if decision.action in {"ANSWER", "STOP"}:
                    if decision.action == "STOP":
                        state.error = (
                            decision.finish_reason
                            or decision.reasoning_summary
                        )
                    state.add_step(
                        AgentStep(
                            step_id=step_id,
                            action=decision.action,
                            model=decision.model,
                            reasoning_summary=decision.reasoning_summary,
                            success=decision.action != "STOP",
                            error=state.error or "",
                        )
                    )
                    state.finished = True
                    break

                tool = decision.tool or decision.action.lower()
                if self._repeated_tool(state, decision):
                    state.add_step(
                        AgentStep(
                            step_id=step_id,
                            action="STOP",
                            model=decision.model,
                            tool=tool,
                            arguments=decision.arguments or {},
                            reasoning_summary=(
                                "检测到完全相同的工具调用，"
                                "停止以避免无意义循环。"
                            ),
                            success=False,
                            error="重复工具调用",
                        )
                    )
                    state.error = (
                        "Agent 检测到重复工具调用，已停止。"
                    )
                    state.finished = True
                    break

                debug.log("AgentToolLoop", f"ACT → tool={tool}")
                try:
                    observation = self._execute_tool(
                        tool,
                        decision.arguments or {},
                        state,
                    )
                    success = True
                    error = ""

                    if decision.action == "SEARCH" and not observation:
                        success = False
                        error = "SEARCH_EMPTY: 搜索请求成功，但没有返回结果。"
                        debug.log(
                            "AgentToolLoop",
                            "OBSERVE → SEARCH_EMPTY",
                        )
                    else:
                        debug.log(
                            "AgentToolLoop",
                            "OBSERVE → success",
                        )
                except Exception as exc:
                    observation, success = None, False
                    error = f"{type(exc).__name__}: {exc}"
                    debug.log(
                        "AgentToolLoop",
                        f"OBSERVE → ERROR: {error}",
                    )

                state.add_step(
                    AgentStep(
                        step_id=step_id,
                        action=decision.action,
                        model=decision.model,
                        tool=decision.tool,
                        arguments=decision.arguments or {},
                        reasoning_summary=decision.reasoning_summary,
                        observation=observation,
                        success=success,
                        error=error,
                    )
                )

                if (
                    decision.action == "SEARCH"
                    and success
                    and observation
                ):
                    state.evidence.extend(observation)

                state.current_plan_step = self._next_plan_step(
                    state,
                    decision.action,
                )
                debug.log(
                    "AgentToolLoop",
                    (
                        "OBSERVE → fed back; "
                        f"next_plan_step={state.current_plan_step}"
                    ),
                )
            return state

    @staticmethod
    def _next_plan_step(state, action):
        if not state.plan:
            return state.current_plan_step

        for i in range(
            max(0, state.current_plan_step),
            len(state.plan.steps),
        ):
            if state.plan.steps[i].action == action:
                return min(i + 1, len(state.plan.steps))

        return state.current_plan_step
