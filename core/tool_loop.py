from typing import Any, Callable

from .__debug__ import debug
from .state import AgentStep
from .reasoner import ReasoningDecision
from .evidence import EvidenceEngine, EvidenceStore


class ToolSpec:
    """Harness 中注册给 LLM 的工具定义。"""

    def __init__(self, name: str, description: str, parameters: dict[str, Any], handler: Callable[..., Any] | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler


class SearchObservation(list):
    """List-compatible search observation with dict-style metadata for the Harness."""

    def __init__(self, results, coverage):
        super().__init__(results)
        self.coverage = coverage

    def get(self, key, default=None):
        if key == "results":
            return list(self)
        if key == "coverage":
            return self.coverage
        return default


class ToolExecutor:
    """Tool harness: LLM 只提出 action，Harness 负责真正执行。"""

    DEFAULT_SEARCH_RESULTS = 10

    def __init__(self, search_router=None, evidence_engine=None):
        self.search_router = search_router
        self.evidence_engine = evidence_engine or EvidenceEngine()
        self._tools: dict[str, tuple[ToolSpec, Callable[..., Any]]] = {}
        self.register(
            ToolSpec(
                "search",
                "Search configured knowledge sources and return normalized evidence.",
                {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "source": {"type": "string", "description": "Optional explicit source."},
                        "categories": {"type": "array", "items": {"type": "string"}},
                        "max_results": {"type": "integer"},
                        "sort_by": {"type": "string", "enum": ["relevance", "submittedDate"]},
                        "sort_order": {"type": "string", "enum": ["ascending", "descending"]},
                    },
                    "required": ["query"],
                },
                self._search,
            )
        )
        self.register(
            ToolSpec(
                "calculate",
                "Safely evaluate an arithmetic expression.",
                {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
                self._calculate,
            )
        )
        self.register(
            ToolSpec(
                "verify",
                "Review a claim against supplied evidence; if evidence is omitted, Harness uses current state evidence.",
                {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence": {"type": "array"},
                    },
                    "required": ["claim"],
                },
                self._verify,
            )
        )

    def register(self, spec: ToolSpec, handler: Callable[..., Any] | None = None) -> None:
        handler = handler or spec.handler
        if handler is None:
            raise ValueError("register 需要 handler。")
        self._tools[spec.name] = (spec, handler)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec, _ in self._tools.values()
        ]

    def execute(self, tool: str, arguments: dict[str, Any], state=None) -> Any:
        debug.log("ToolExecutor", f"EXECUTE → {tool}")
        debug.log("ToolExecutor", f"ARGS → {arguments}")
        entry = self._tools.get(tool)
        if entry is None:
            raise ValueError(f"未知工具：{tool}")
        _, handler = entry
        if tool in {"search", "verify"}:
            return handler(arguments, state)
        return handler(arguments)

    def _search(self, arguments, state=None):
        if self.search_router is None:
            raise RuntimeError("SearchRouter 尚未配置。")
        from tools.search import SearchQuery

        explicit_source = str(arguments.get("source", "")).strip().lower()
        source = explicit_source or "auto"
        preferences = [] if explicit_source else list(getattr(state, "search_sources", []) or [])
        categories = arguments.get("categories", [])
        if not isinstance(categories, list):
            categories = []
        sort_by = str(arguments.get("sort_by", "")).strip() or str(getattr(state, "search_sort_by", "relevance"))
        query = SearchQuery(
            query=str(arguments.get("query", "")),
            source=source,
            source_preferences=preferences,
            categories=[str(x) for x in categories],
            max_results=int(arguments.get("max_results", self.DEFAULT_SEARCH_RESULTS)),
            sort_by=sort_by,
            sort_order=str(arguments.get("sort_order", "descending")),
        )
        results = self.search_router.search(query)
        raw = [
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
        normalized = self.evidence_engine.normalize(query.query, raw)
        coverage = self.evidence_engine.coverage(query.query, normalized)
        debug.log(
            "ToolExecutor",
            f"SEARCH RESULT → source={source} raw={len(raw)} unique={len(normalized)} coverage={coverage['status']}",
        )
        return SearchObservation(normalized, coverage)

    def _calculate(self, arguments):
        expression = str(arguments.get("expression", ""))
        if not expression:
            raise ValueError("calculate 缺少 expression。")
        return self._safe_calculate(expression)

    def _verify(self, arguments, state=None):
        claim = str(arguments.get("claim", "")).strip()
        evidence = arguments.get("evidence")
        if evidence is None and state is not None:
            evidence = state.evidence
        if not isinstance(evidence, list):
            evidence = []
        return self.evidence_engine.verify(claim, evidence)

    @staticmethod
    def _safe_calculate(expression):
        import ast
        import operator
        operators = {
            ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
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
    """真正的 LLM ↔ Harness 闭环：Decide → Act → Observe → Decide。"""

    def __init__(self, reasoner, executor: ToolExecutor):
        self.reasoner, self.executor = reasoner, executor

    @staticmethod
    def _fingerprint(action, tool, arguments):
        import json
        return action, tool, json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True, default=str)

    def _repeated_tool(self, state, decision):
        if decision.action not in {"SEARCH", "CALCULATE", "VERIFY"}:
            return False
        fp = self._fingerprint(decision.action, decision.tool or decision.action.lower(), decision.arguments)
        return any(
            self._fingerprint(s.action, s.tool or s.action.lower(), s.arguments) == fp
            for s in state.steps
        )

    def _decide(self, state):
        """Support current Reasoner and simple one-argument custom/test adapters."""
        import inspect
        decide = self.reasoner.decide
        try:
            params = list(inspect.signature(decide).parameters.values())
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(params) >= 2:
                specs = self.executor.tool_specs() if hasattr(self.executor, "tool_specs") else []
                return decide(state, specs)
        except (TypeError, ValueError):
            pass
        return decide(state)

    def _execute(self, tool, arguments, state):
        """Use state-aware execution when supported; keep simple test adapters working."""
        import inspect
        execute = self.executor.execute
        try:
            signature = inspect.signature(execute)
            params = signature.parameters
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) or "state" in params:
                return execute(tool, arguments, state=state)
        except (TypeError, ValueError):
            pass
        return execute(tool, arguments)

    def _force_verify(self, state, decision):
        """Compatibility gate for callers that still expose a pre-planned VERIFY step."""
        if decision.action == "ANSWER" and state.plan and state.current_plan_step < len(state.plan.steps):
            if any(step.action == "VERIFY" for step in state.plan.steps[state.current_plan_step:]):
                return ReasoningDecision(action="VERIFY", reasoning_summary="在回答前完成计划要求的证据核查。", tool="verify", arguments={"claim": state.question, "evidence": state.evidence}, model=decision.model)
        return decision

    def run(self, state):
        evidence_store = EvidenceStore(state.evidence)
        with debug.scope("AgentToolLoop", "RUN"):
            while not state.finished:
                debug.log("AgentToolLoop", f"REASON → step={state.step_count + 1}")
                decision = self._decide(state)
                debug.log("AgentToolLoop", f"DECISION → {decision.action}")

                if decision.goal:
                    state.goal = decision.goal
                if decision.task_type:
                    state.task_type = decision.task_type
                if decision.domain:
                    state.domain = decision.domain
                if decision.claims:
                    state.claims = decision.claims
                if decision.evidence_relevance:
                    state.evidence_relevance = decision.evidence_relevance
                if decision.student_model_update:
                    state.student.apply_mind_update(decision.student_model_update)

                step_id = state.step_count + 1
                if decision.action in {"ANSWER", "STOP"}:
                    if decision.action == "ANSWER":
                        state.final_answer = decision.answer
                    else:
                        state.error = decision.finish_reason or decision.reasoning_summary
                    state.add_step(AgentStep(
                        step_id=step_id, action=decision.action, model=decision.model,
                        reasoning_summary=decision.reasoning_summary,
                        success=decision.action == "ANSWER",
                        error=state.error or "",
                    ))
                    state.finished = True
                    break

                tool = decision.tool or decision.action.lower()
                if self._repeated_tool(state, decision):
                    state.error = "Agent 检测到重复工具调用，已停止。"
                    state.add_step(AgentStep(
                        step_id=step_id, action="STOP", model=decision.model, tool=tool,
                        arguments=decision.arguments or {},
                        reasoning_summary="检测到完全相同的工具调用，停止以避免无意义循环。",
                        success=False, error=state.error,
                    ))
                    state.finished = True
                    break

                try:
                    observation = self._execute(tool, decision.arguments or {}, state)
                    search_results = observation.get("results", []) if isinstance(observation, dict) else observation
                    success = not (decision.action == "SEARCH" and not search_results)
                    error = "" if success else "SEARCH_EMPTY: 搜索请求成功，但没有返回结果。"
                    error_type = "" if success else "SEARCH_EMPTY"
                except Exception as exc:
                    observation, success = None, False
                    error_type = type(exc).__name__
                    error = f"{error_type}: {exc}"

                state.add_step(AgentStep(
                    step_id=step_id, action=decision.action, model=decision.model,
                    tool=tool, arguments=decision.arguments or {},
                    reasoning_summary=decision.reasoning_summary,
                    observation=observation, success=success, error=error,
                ))

                if decision.action == "SEARCH" and success and observation:
                    search_results = observation.get("results", []) if isinstance(observation, dict) else observation
                    evidence_store.add_many(search_results)
                    state.evidence = evidence_store.items

                if decision.action == "VERIFY" and success and isinstance(observation, dict):
                    state.claims.append({"claim": observation.get("claim", ""), "verification_status": observation.get("verification_status", "UNCERTAIN"), "matched_evidence": observation.get("matched_evidence", [])})
                state.current_plan_step = self._next_plan_step(state, decision.action)
                state.last_observation = observation
                state.last_error_type = error_type if not success else ""
                if not success:
                    state.recovery_count += 1
                debug.log("AgentToolLoop", "OBSERVE → fed back to LLM")

            return state

    @staticmethod
    def _next_plan_step(state, action):
        if not state.plan:
            return state.current_plan_step
        for i in range(max(0, state.current_plan_step), len(state.plan.steps)):
            if state.plan.steps[i].action == action:
                return min(i + 1, len(state.plan.steps))
        return state.current_plan_step
