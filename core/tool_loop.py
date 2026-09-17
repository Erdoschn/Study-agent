from typing import Any
from .__debug__ import debug
from .state import AgentStep


class ToolExecutor:

    def __init__(
        self,
        search_router=None,
    ):
        self.search_router = search_router

    def execute(
        self,
        tool: str,
        arguments: dict[str, Any],
    ) -> Any:

        debug.log(
            "ToolExecutor",
            f"EXECUTE → {tool}",
        )

        debug.log(
            "ToolExecutor",
            f"ARGS → {arguments}",
        )

        if tool == "search":
            return self._search(arguments)

        if tool == "calculate":
            return self._calculate(arguments)

        if tool == "verify":
            return self._verify(arguments)

        raise ValueError(
            f"未知工具：{tool}"
        )

    def _search(self, arguments):

        if self.search_router is None:
            raise RuntimeError(
                "SearchRouter 尚未配置。"
            )

        from tools.search import SearchQuery

        query = SearchQuery(
            query=str(
                arguments.get(
                    "query",
                    "",
                )
            ),
            source=str(
                arguments.get(
                    "source",
                    "arxiv",
                )
            ),
            categories=list(
                arguments.get(
                    "categories",
                    [],
                )
            ),
            max_results=int(
                arguments.get(
                    "max_results",
                    5,
                )
            ),
        )

        results = (
            self.search_router.search(
                query
            )
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

    def _calculate(
        self,
        arguments,
    ):

        expression = str(
            arguments.get(
                "expression",
                "",
            )
        )

        if not expression:
            raise ValueError(
                "calculate 缺少 expression。"
            )

        return self._safe_calculate(
            expression
        )

    def _verify(
        self,
        arguments,
    ):
        return {
            "claim": str(
                arguments.get(
                    "claim",
                    "",
                )
            ),
            "evidence": arguments.get(
                "evidence",
                [],
            ),
            "verification_status":
                "REQUIRES_MODEL_REVIEW",
        }

    @staticmethod
    def _safe_calculate(
        expression,
    ):
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

            if isinstance(
                node,
                ast.Constant,
            ):
                if isinstance(
                    node.value,
                    (int, float),
                ):
                    return node.value

                raise ValueError(
                    "只允许数字常量。"
                )

            if isinstance(
                node,
                ast.UnaryOp,
            ):
                fn = operators.get(
                    type(node.op)
                )

                if fn is None:
                    raise ValueError(
                        "不支持的运算符。"
                    )

                return fn(
                    evaluate(
                        node.operand
                    )
                )

            if isinstance(
                node,
                ast.BinOp,
            ):
                fn = operators.get(
                    type(node.op)
                )

                if fn is None:
                    raise ValueError(
                        "不支持的运算符。"
                    )

                return fn(
                    evaluate(node.left),
                    evaluate(node.right),
                )

            raise ValueError(
                "表达式包含不允许的内容。"
            )

        tree = ast.parse(
            expression,
            mode="eval",
        )

        return evaluate(tree.body)


class AgentToolLoop:

    def __init__(
        self,
        reasoner,
        executor: ToolExecutor,
    ):
        self.reasoner = reasoner
        self.executor = executor

    def run(self, state):

        with debug.scope(
            "AgentToolLoop",
            "RUN",
        ):

            while (
                not state.finished
                and state.step_count
                < state.max_steps
            ):

                debug.log(
                    "AgentToolLoop",
                    f"REASON → "
                    f"step={state.step_count + 1}",
                )

                decision = (
                    self.reasoner.decide(
                        state
                    )
                )

                debug.log(
                    "AgentToolLoop",
                    f"DECISION → "
                    f"{decision.action}",
                )

                if decision.goal:
                    state.goal = decision.goal

                if decision.task_type:
                    state.task_type = (
                        decision.task_type
                    )

                if decision.domain:
                    state.domain = (
                        decision.domain
                    )

                if decision.claims:
                    state.claims = (
                        decision.claims
                    )

                step_id = (
                    state.step_count + 1
                )

                if decision.action == "ANSWER":

                    debug.log(
                        "AgentToolLoop",
                        "ANSWER → stop loop",
                    )

                    state.add_step(
                        AgentStep(
                            step_id=step_id,
                            action="ANSWER",
                            model=decision.model,
                            reasoning_summary=(
                                decision.reasoning_summary
                            ),
                        )
                    )

                    state.finished = True
                    break

                if decision.action == "STOP":

                    state.error = (
                        decision.finish_reason
                        or decision.reasoning_summary
                    )

                    state.add_step(
                        AgentStep(
                            step_id=step_id,
                            action="STOP",
                            model=decision.model,
                            reasoning_summary=(
                                decision.reasoning_summary
                            ),
                            success=False,
                            error=(
                                state.error or ""
                            ),
                        )
                    )

                    state.finished = True
                    break

                tool = (
                    decision.tool
                    or decision.action.lower()
                )

                debug.log(
                    "AgentToolLoop",
                    f"ACT → tool={tool}",
                )

                try:
                    observation = (
                        self.executor.execute(
                            tool,
                            decision.arguments
                            or {},
                        )
                    )

                    success = True
                    error = ""

                    debug.log(
                        "AgentToolLoop",
                        "OBSERVE → success",
                    )

                except Exception as exc:

                    observation = None
                    success = False

                    error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                    debug.log(
                        "AgentToolLoop",
                        f"OBSERVE → ERROR: "
                        f"{error}",
                    )

                state.add_step(
                    AgentStep(
                        step_id=step_id,
                        action=decision.action,
                        model=decision.model,
                        tool=decision.tool,
                        arguments=(
                            decision.arguments
                            or {}
                        ),
                        reasoning_summary=(
                            decision.reasoning_summary
                        ),
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
                    state.evidence.extend(
                        observation
                    )

            return state