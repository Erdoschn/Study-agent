from typing import Any

from .state import AgentStep


class ToolExecutor:
    """
    Agent 工具执行层。

    这里不做推理，只执行已经决定的 Action。
    """

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

        if tool == "search":
            return self._search(arguments)

        if tool == "calculate":
            return self._calculate(arguments)

        if tool == "verify":
            return self._verify(arguments)

        raise ValueError(
            f"未知工具：{tool}"
        )

    def _search(
        self,
        arguments: dict[str, Any],
    ) -> list[dict[str, Any]]:

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

        results = self.search_router.search(query)

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
        arguments: dict[str, Any],
    ) -> Any:

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

        return self._safe_calculate(expression)

    def _verify(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        claim = str(
            arguments.get(
                "claim",
                "",
            )
        )

        evidence = arguments.get(
            "evidence",
            [],
        )

        return {
            "claim": claim,
            "evidence": evidence,
            "verification_status": (
                "REQUIRES_MODEL_REVIEW"
            ),
        }

    @staticmethod
    def _safe_calculate(
        expression: str,
    ) -> Any:
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
                    evaluate(node.operand)
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
    """
    Reason → Act → Observe 循环。

    注意：
    Reasoner 决定做什么。
    Executor 负责执行。
    """

    def __init__(
        self,
        reasoner,
        executor: ToolExecutor,
    ):
        self.reasoner = reasoner
        self.executor = executor

    def run(self, state):
        while (
            not state.finished
            and state.step_count
            < state.max_steps
        ):
            decision = self.reasoner.decide(
                state
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
                state.claims = decision.claims

            step_id = state.step_count + 1

            if decision.action == "ANSWER":
                state.final_answer = (
                    decision.reasoning_summary
                )

                state.add_step(
                    AgentStep(
                        step_id=step_id,
                        action="ANSWER",
                        reasoning_summary=(
                            decision.reasoning_summary
                        ),
                        success=True,
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
                        reasoning_summary=(
                            decision.reasoning_summary
                        ),
                        success=False,
                        error=state.error or "",
                    )
                )

                state.finished = True
                break

            try:
                observation = self.executor.execute(
                    decision.tool or decision.action.lower(),
                    decision.arguments or {},
                )

                success = True
                error = ""

            except Exception as exc:
                observation = None
                success = False
                error = (
                    f"{type(exc).__name__}: {exc}"
                )

            state.add_step(
                AgentStep(
                    step_id=step_id,
                    action=decision.action,
                    tool=decision.tool,
                    arguments=(
                        decision.arguments or {}
                    ),
                    reasoning_summary=(
                        decision.reasoning_summary
                    ),
                    observation=observation,
                    success=success,
                    error=error,
                )
            )

            if decision.action == "SEARCH":
                if success and observation:
                    state.evidence.extend(
                        observation
                    )

        if not state.finished:
            state.finished = True
            state.error = (
                "达到 Agent 最大行动步数。"
            )

        return state