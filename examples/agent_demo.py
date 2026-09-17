import os

from core import (
    AgentReasoner,
    OpenAICompatibleClient,
    StudyAgent,
    Teacher,
    ToolExecutor,
)

from tools.search import (
    ArxivSearchProvider,
    SearchRouter,
    WikipediaSearchProvider,
)


def build_search_router():
    router = SearchRouter()

    router.register(
        ArxivSearchProvider()
    )

    router.register(
        WikipediaSearchProvider()
    )

    return router


def main():
    base_url = os.getenv(
        "STUDY_AGENT_BASE_URL"
    )
    api_key = os.getenv(
        "STUDY_AGENT_API_KEY"
    )
    model = os.getenv(
        "STUDY_AGENT_MODEL"
    )

    if not base_url or not api_key or not model:
        print(
            "缺少 LLM 配置：\n"
            "STUDY_AGENT_BASE_URL\n"
            "STUDY_AGENT_API_KEY\n"
            "STUDY_AGENT_MODEL"
        )
        return

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
    )

    reasoner = AgentReasoner(client)

    search_router = build_search_router()

    executor = ToolExecutor(
        search_router=search_router
    )

    teacher = Teacher(client)

    agent = StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=executor,
    )

    question = input(
        "请输入学习问题："
    ).strip()

    if not question:
        return

    print("\n========== Study Agent ==========\n")

    try:
        result = agent.run(question)

    except Exception as exc:
        print(
            f"Agent 执行失败："
            f"{type(exc).__name__}: {exc}"
        )
        return

    for step in result.steps:
        print(
            f"\n[Step {step.step_id}] "
            f"{step.action}"
        )

        if step.tool:
            print(
                f"Tool: {step.tool}"
            )

        if step.reasoning_summary:
            print(
                f"Decision: "
                f"{step.reasoning_summary}"
            )

        if step.error:
            print(
                f"ERROR: {step.error}"
            )

    print("\n========== Final Answer ==========\n")
    print(result.final_answer or "")


if __name__ == "__main__":
    main()