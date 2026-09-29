import argparse
import sys
from pathlib import Path

# 允许直接运行：python main.py
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWLEDGE_GRAPH_PATH = PROJECT_ROOT / "data" / "knowledge_graph.sqlite3"

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
        knowledge_graph_path=KNOWLEDGE_GRAPH_PATH,
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

    print(
        "测试：可直接输入“出题 / 测试我 / 检验理解”，也可用 /test <知识点> [难度]；查看学习状态：/learner [知识点]\n"
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

        # Explicit formal assessment session:
        # /test <concept> [difficulty]
        # This path never exposes shell, file, or arbitrary code tools.
        if question.lower().startswith("/test "):
            parts = question.split(maxsplit=2)
            concept = parts[1].strip() if len(parts) > 1 else ""
            difficulty = parts[2].strip() if len(parts) > 2 else "postgraduate_plus"
            if not concept:
                print("用法：/test <知识点> [难度]\n")
                continue
            try:
                assessment = agent.start_assessment(concept, difficulty)
                print("\n========== Formal Assessment ==========\n")
                print(f"Concept: {assessment['primary_concept']}")
                print(f"Difficulty: {assessment['difficulty_level']}")
                print(f"\n{assessment['question']}\n")
                answer = input("Student Answer > ").strip()
                result = agent.submit_assessment_answer(answer)
                learner = result.get("learner_state", {}).get(
                    assessment["primary_concept"], {}
                )
                print("\n========== Assessment Result ==========\n")
                print(f"Score: {result['score']:.3f}")
                print(f"Correct: {result['correct']}")
                print(f"Learning Stage: {learner.get('learning_stage', 'unknown')}")
                print(f"Familiarity: {learner.get('familiarity', 0):.3f}")
                print(f"Confidence: {learner.get('confidence', 0):.3f}")
                print()
            except Exception as exc:
                print(f"❌ 正式测评失败：{type(exc).__name__}: {exc}\n")
            continue

        if question.lower().startswith("/learner"):
            parts = question.split(maxsplit=1)
            concept = parts[1].strip() if len(parts) > 1 else None
            try:
                learner = agent.learner_state(concept)
                print("\n========== Learner State ==========\n")
                print(learner)
                print()
            except Exception as exc:
                print(f"❌ 学习状态读取失败：{type(exc).__name__}: {exc}\n")
            continue

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
        if result.pending_assessment:
            assessment = result.pending_assessment
            print("\n========== Formal Assessment ==========\n")
            print(f"Concept: {assessment.get('primary_concept', '')}")
            print(f"Difficulty: {assessment.get('difficulty_level', '')}")
            print(f"\n{assessment.get('question', '')}\n")
            try:
                answer = input("Student Answer > ").strip()
                result = agent.submit_assessment_answer(answer)
                primary = assessment.get("primary_concept", "")
                learner = result.get("learner_state", {}).get(primary, {})
                print("\n========== Assessment Result ==========\n")
                print(f"Score: {result['score']:.3f}")
                print(f"Correct: {result['correct']}")
                print(f"Learning Stage: {learner.get('learning_stage', 'unknown')}")
                print(f"Familiarity: {learner.get('familiarity', 0):.3f}")
                print(f"Confidence: {learner.get('confidence', 0):.3f}")
                print()
            except Exception as exc:
                print(f"❌ 测评失败：{type(exc).__name__}: {exc}\n")
            continue

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

        if result.metrics.get("assessment_offer"):
            print("\n📝 要检验一下刚才的理解吗？输入“出题”或“测试我”，我会等你明确要求后再出题。")

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