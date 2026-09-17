from dataclasses import dataclass, field


@dataclass
class ToolReasoning:
    """
    Agent 对“是否使用工具以及下一步做什么”的结构化判断。
    """

    goal: str
    current_state: str
    information_needed: list[str] = field(default_factory=list)

    need_tool: bool = False
    tool: str = ""
    query: str = ""

    reason: str = ""
    next_action: str = "answer"

    confidence_note: str = ""


class ToolReasoner:
    """
    Tool Reasoner

    不负责真正执行工具。
    只负责根据当前任务状态决定：
        1. 当前目标是什么
        2. 缺什么信息
        3. 是否需要工具
        4. 使用什么工具
        5. 下一步做什么
    """

    def reason(self, question: str) -> ToolReasoning:
        question = question.strip()

        if not question:
            return ToolReasoning(
                goal="无明确学习任务",
                current_state="输入为空",
                reason="没有可分析的问题。",
                next_action="stop",
            )

        # --------------------------------------------------
        # Mock reasoning
        #
        # 这里只是为了先把 Agent 的“思考 -> 行动”接口跑起来。
        # 后面会替换成真实 LLM reasoning。
        # --------------------------------------------------

        lower = question.lower()

        research_signals = [
            "论文",
            "paper",
            "research",
            "arxiv",
            "实验结果",
            "benchmark",
        ]

        realtime_signals = [
            "最新",
            "现在",
            "今天",
            "目前",
            "新闻",
            "latest",
            "today",
            "current",
            "news",
        ]

        concept_signals = [
            "什么是",
            "是什么",
            "原理",
            "定义",
            "概念",
            "介绍",
            "解释",
            "what is",
            "definition",
            "explain",
        ]

        if any(x in lower for x in research_signals):
            return ToolReasoning(
                goal="获得研究资料以支持回答",
                current_state="用户明确需要论文或研究证据",
                information_needed=[
                    "相关研究论文",
                    "论文核心结论",
                    "实验或证据",
                ],
                need_tool=True,
                tool="arxiv",
                query=question,
                reason=(
                    "问题包含明确的论文/研究需求，"
                    "直接搜索研究资料比仅凭模型记忆回答更合适。"
                ),
                next_action="search",
                confidence_note="明确工具需求",
            )

        if any(x in lower for x in realtime_signals):
            return ToolReasoning(
                goal="获得当前信息",
                current_state="用户要求最新或实时信息",
                information_needed=[
                    "当前资料",
                    "最新变化",
                ],
                need_tool=True,
                tool="web",
                query=question,
                reason=(
                    "当前信息可能超过模型的静态知识范围，"
                    "需要外部资料。"
                ),
                next_action="search",
                confidence_note="明确需要外部信息",
            )

        if any(x in lower for x in concept_signals):
            return ToolReasoning(
                goal="解释概念并建立学习理解",
                current_state="用户正在询问概念或定义",
                information_needed=[
                    "概念定义",
                    "核心机制",
                ],
                need_tool=True,
                tool="wikipedia",
                query=question,
                reason=(
                    "可以通过百科资料补充基础定义，"
                    "再结合模型解释。"
                ),
                next_action="search",
                confidence_note="可用基础资料辅助",
            )

        return ToolReasoning(
            goal="直接分析并回答学习问题",
            current_state="没有发现必须依赖外部资料的明确条件",
            information_needed=[],
            need_tool=False,
            reason=(
                "当前问题不包含明显的实时、论文或外部资料要求，"
                "可以先进行内部推理。"
            ),
            next_action="answer",
            confidence_note="暂不需要工具",
        )