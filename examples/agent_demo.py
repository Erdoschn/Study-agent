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
        allow_paid=True,  # 允许付费模型，便于测试
    )

    # -----------------------------
    # Teacher
    #
    # 默认不允许付费模型。
    # -----------------------------
    teacher = Teacher(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=True,  # 允许付费模型，便于测试
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
        max_steps=8,
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
            config
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
        "默认禁止"
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
            "\n=================================\n"
        )


if __name__ == "__main__":
    main()