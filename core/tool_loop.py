from typing import Any, Callable

from .__debug__ import debug
from .state import AgentStep
from .reasoner import ReasoningDecision
from .evidence import EvidenceEngine, EvidenceStore
from .goal import GoalMatcher
from .search_strategy import SearchStrategy


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

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "results":
                return list(self)
            if key == "coverage":
                return self.coverage
            raise KeyError(key)
        return super().__getitem__(key)


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
                "assess",
                "Create a scorable assessment and store it as pending for the student.",
                {
                    "type": "object",
                    "properties": {
                        "concepts": {"type": "array", "items": {"type": "string"}},
                        "relations": {"type": "array"},
                        "difficulty": {"type": "string", "enum": ["basic", "undergraduate", "graduate", "postgraduate", "postgraduate_plus"]},
                        "question_type": {"type": "string"},
                        "question": {"type": "string"},
                        "expected_answer": {"type": "string"},
                        "rubric": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["concepts", "question"],
                },
                self._assess,
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
            {"name": spec.name, "description": spec.description, "parameters": spec.parameters}
            for spec, _ in self._tools.values()
        ]

    def execute(self, tool: str, arguments: dict[str, Any], state=None) -> Any:
        debug.log("ToolExecutor", f"EXECUTE → {tool}")
        debug.log("ToolExecutor", f"ARGS → {arguments}")
        entry = self._tools.get(tool)
        if entry is None:
            raise ValueError(f"未知工具：{tool}")
        _, handler = entry
        if tool in {"search", "verify", "assess"}:
            return handler(arguments, state)
        return handler(arguments)

    def _search(self, arguments, state=None):
        if self.search_router is None:
            raise RuntimeError("SearchRouter 尚未配置。")
        from tools.search import SearchQuery

        explicit_source = str(arguments.get("source", "")).strip().lower()
        source = explicit_source or "auto"
        if source != "auto":
            available = {str(x).strip().lower() for x in self.search_router.available_sources()}
            if source not in available:
                names = ", ".join(sorted(available))
                raise ValueError(f"未知搜索源：{source}。可用搜索源：{names}")
        preferences = []
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
                "source": item.source, "source_type": item.source_type, "title": item.title,
                "url": item.url, "abstract": item.abstract, "authors": item.authors,
                "published": item.published, "updated": item.updated, "identifier": item.identifier,
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

    def _assess(self, arguments, state=None):
        from .knowledge_graph import normalize_difficulty
        concepts = arguments.get("concepts", [])
        question = str(arguments.get("question", "")).strip()
        if not isinstance(concepts, list) or not concepts or not question:
            raise ValueError("assess 需要 concepts 和 question。")
        level, score = normalize_difficulty(arguments.get("difficulty", "graduate"))
        rubric = arguments.get("rubric", [])
        pending = {
            "concepts": [str(x).strip() for x in concepts if str(x).strip()][:8],
            "relations": arguments.get("relations", []),
            "difficulty": score,
            "difficulty_level": level,
            "question_type": str(arguments.get("question_type", "open_ended")),
            "question": question,
            "expected_answer": str(arguments.get("expected_answer", "")).strip(),
            "rubric": [str(x).strip() for x in rubric if str(x).strip()][:8] if isinstance(rubric, list) else [],
        }
        if state is not None:
            state.pending_assessment = pending
        return {"status": "ASSESSMENT_PENDING", "assessment": pending}

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
        if decision.action not in {"SEARCH", "CALCULATE", "VERIFY", "ASSESS"}:
            return False
        fp = self._fingerprint(decision.action, decision.tool or decision.action.lower(), decision.arguments)
        return any(
            self._fingerprint(s.action, s.tool or s.action.lower(), s.arguments) == fp
            for s in state.steps
        )

    def _decide(self, state):
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

    def run(self, state):
        evidence_store = EvidenceStore(state.evidence)
        with debug.scope("AgentToolLoop", "RUN"):
            while not state.finished:
                if state.max_steps is not None and state.step_count >= state.max_steps:
                    state.error = f"达到 Agent 最大安全步数上限：{state.max_steps}。"
                    state.add_step(AgentStep(
                        step_id=state.step_count + 1, action="STOP",
                        reasoning_summary="达到安全步数上限，停止继续调用。",
                        success=False, error=state.error,
                    ))
                    state.finished = True
                    break

                saved_goals = list(state.student.mind.long_term.desires) + list(state.student.mind.short_term.desires)
                state.goal_context = GoalMatcher.context_for(state.question, state.goal, saved_goals)
                debug.log("AgentToolLoop", f"GOAL CONTEXT → matched={len(state.goal_context)}")
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
                if decision.belief_revisions:
                    state.student.mind.revise_beliefs(decision.belief_revisions, state.evidence)

                step_id = state.step_count + 1
                if decision.action in {"ANSWER", "STOP"}:
                    if decision.action == "ANSWER":
                        verified = any(
                            step.action == "VERIFY"
                            and step.success
                            and isinstance(step.observation, dict)
                            and step.observation.get("verification_status") == "MATCHED"
                            for step in state.steps
                        )
                        if state.evidence and not verified:
                            state.last_error_type = "VERIFY_REQUIRED"
                            state.recovery_count += 1
                            state.add_step(AgentStep(
                                step_id=step_id, action="ANSWER_BLOCKED", model=decision.model,
                                reasoning_summary="已有外部证据但尚未完成有效 VERIFY，Harness 阻止直接回答。",
                                success=False,
                                error="VERIFY_REQUIRED: 有外部证据时必须先完成至少一次 MATCHED VERIFY。",
                            ))
                            continue
                        state.final_answer = decision.answer
                    else:
                        state.error = decision.finish_reason or decision.reasoning_summary
                    state.add_step(AgentStep(
                        step_id=step_id, action=decision.action, model=decision.model,
                        reasoning_summary=decision.reasoning_summary,
                        success=decision.action == "ANSWER", error=state.error or "",
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
                    error_type = type(exc).__name__
                    error = f"{error_type}: {exc}"
                    observation = {"status": "ERROR", "error_type": error_type, "error": str(exc)}
                    success = False

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
                    if state.knowledge_graph is not None:
                        state.knowledge_graph.learn_from_search(
                            str((decision.arguments or {}).get("query", "")),
                            search_results,
                        )
                elif decision.action == "SEARCH" and not success:
                    if isinstance(observation, dict):
                        observation["search_strategy"] = SearchStrategy.guidance(
                            state.steps,
                            error_type=error_type,
                        )

                if decision.action == "VERIFY" and success and isinstance(observation, dict):
                    state.claims.append({
                        "claim": observation.get("claim", ""),
                        "verification_status": observation.get("verification_status", "UNCERTAIN"),
                        "matched_evidence": observation.get("matched_evidence", []),
                    })
                state.last_observation = observation
                state.last_error_type = error_type if not success else ""
                if not success:
                    state.recovery_count += 1
                if decision.action == "ASSESS" and success:
                    state.finished = True
                    debug.log("AgentToolLoop", "ASSESSMENT → waiting for student answer")
                    break
                debug.log("AgentToolLoop", "OBSERVE → fed back to LLM")

            return state
