import argparse

from config.loader import (
    load_config,
    setup_debug,
)

from core import (
    AgentReasoner,
    ModelClientFactory,
    ModelRegistry,
    ModelRouter,
    StudyAgent,
    Teacher,
    ToolExecutor,
)

from tools.search import (
    ArxivSearchProvider,
    SearchRouter,
    WikipediaSearchProvider,
)


def build_search_router() -> SearchRouter:
    """
    创建搜索工具路由器。
    """

    router = SearchRouter()

    router.register(
        ArxivSearchProvider()
    )

    router.register(
        WikipediaSearchProvider()
    )

    return router


def build_agent(
    config: dict,
    *,
    allow_paid: bool = False,
) -> StudyAgent:
    """
    根据配置创建完整 Study Agent。
    """

    # -----------------------------
    # Model Registry
    # -----------------------------
    registry = ModelRegistry(
        config
    )

    # -----------------------------
    # Model Router
    # -----------------------------
    model_router = ModelRouter(
        registry
    )

    # -----------------------------
    # Model Factory
    # -----------------------------
    model_factory = ModelClientFactory(
        config
    )

    # -----------------------------
    # Reasoner
    #
    # 默认不允许付费模型。
    # -----------------------------
    reasoner = AgentReasoner(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=allow_paid,  # 仅显式 --allow-paid 时允许付费模型
    )

    # -----------------------------
    # Teacher
    #
    # 与 Reasoner 使用相同的付费模型策略。
    # -----------------------------
    teacher = Teacher(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=allow_paid,
    )

    # -----------------------------
    # Search Router
    # -----------------------------
    search_router = (
        build_search_router()
    )

    # -----------------------------
    # Tool Executor
    # -----------------------------
    executor = ToolExecutor(
        search_router=search_router
    )

    # -----------------------------
    # Study Agent
    # -----------------------------
    return StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=executor,
        max_steps=None,
    )


def print_trace(
    result,
) -> None:
    """
    输出 Agent 本次运行的结构化步骤。

    注意：
    Debug 模式已经会输出实时调用链。
    这里是最终结果汇总。
    """

    print(
        "\n========== Agent Trace ==========\n"
    )

    for step in result.steps:

        print(
            f"[Step {step.step_id}] "
            f"{step.action}"
        )

        if step.model:
            print(
                f"  Model: {step.model}"
            )

        if step.tool:
            print(
                f"  Tool: {step.tool}"
            )

        if step.reasoning_summary:
            print(
                "  Decision: "
                f"{step.reasoning_summary}"
            )

        if step.arguments:
            print(
                f"  Arguments: "
                f"{step.arguments}"
            )

        if step.observation is not None:

            if isinstance(
                step.observation,
                list,
            ):
                print(
                    "  Observation: "
                    f"{len(step.observation)} "
                    "items"
                )
            else:
                print(
                    "  Observation: "
                    f"{step.observation}"
                )

        if step.error:
            print(
                f"  ERROR: "
                f"{step.error}"
            )

        print()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Study Agent CLI"
    )
    parser.add_argument(
        "--allow-paid",
        action="store_true",
        help="显式允许付费模型。本开关默认关闭。",
    )
    args = parser.parse_args()

    # -----------------------------
    # 读取配置
    # -----------------------------
    try:
        config = load_config()

    except Exception as exc:

        print(
            "❌ 配置加载失败："
            f"{type(exc).__name__}: {exc}"
        )

        return

    # -----------------------------
    # 启用 Debug
    #
    # providers.json:
    # "debug": true
    # -----------------------------
    setup_debug(config)

    # -----------------------------
    # 创建 Agent
    # -----------------------------
    try:
        agent = build_agent(
            config,
            allow_paid=args.allow_paid,
        )

    except Exception as exc:

        print(
            "❌ Agent 初始化失败："
            f"{type(exc).__name__}: {exc}"
        )

        return

    print(
        "\n========== Study Agent ==========\n"
    )

    print(
        "Debug 模式："
        + (
            "ON"
            if config.get(
                "debug",
                False,
            )
            else "OFF"
        )
    )

    print(
        "付费模型："
        + ("允许（--allow-paid）" if args.allow_paid else "禁止")
    )

    print(
        "\n可输入学习问题。"
    )

    print(
        "输入 exit 或 quit 退出。\n"
    )

    # -----------------------------
    # 交互循环
    # -----------------------------
    while True:

        try:
            question = input(
                "User > "
            ).strip()

        except (
            EOFError,
            KeyboardInterrupt,
        ):

            print(
                "\n退出。"
            )

            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
        }:

            print(
                "退出。"
            )

            break

        print(
            "\n========== Agent Run ==========\n"
        )

        try:

            result = agent.run(
                question
            )

        except Exception as exc:

            print(
                "❌ Agent 执行失败："
                f"{type(exc).__name__}: {exc}"
            )

            continue

        # -------------------------
        # 输出 Trace
        # -------------------------
        print_trace(
            result
        )

        # -------------------------
        # 输出最终答案
        # -------------------------
        print(
            "========== Final Answer ==========\n"
        )

        print(
            result.final_answer
            or ""
        )

        print(
            "\n========== Student BDI =========="
        )
        mind = result.student.mind.as_dict()
        for horizon, label in (("short_term", "短期"), ("long_term", "长期")):
            data = mind[horizon]
            print(f"[{label}] B: {data['beliefs']}")
            print(f"[{label}] D: {data['desires']}")
            print(f"[{label}] I: {data['intentions']}")
        print(f"[决定] {mind['recent_decisions']}")

        print(
            "\n=================================\n"
        )


if __name__ == "__main__":
    main()