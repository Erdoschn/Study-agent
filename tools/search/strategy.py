from dataclasses import dataclass


@dataclass
class SearchDecision:
    """
    搜索策略的决定结果。
    """

    need_search: bool
    source: str = ""
    reason: str = ""
    confidence: float = 0.0


class SearchStrategy:
    """
    第一版搜索策略。

    只负责：
    1. 判断是否明显需要联网
    2. 选择首选搜索源

    不负责：
    - 真正执行搜索
    - 判断搜索结果是否正确
    - 判断 Claim 是否被证实
    """

    REALTIME_KEYWORDS = {
        "今天",
        "现在",
        "最新",
        "目前",
        "最近",
        "实时",
        "刚刚",
        "新闻",
        "current",
        "latest",
        "today",
        "now",
        "recent",
        "news",
    }

    RESEARCH_KEYWORDS = {
        "论文",
        "研究",
        "paper",
        "papers",
        "research",
        "arxiv",
        "实验结果",
        "实验",
        "benchmark",
        "state of the art",
        "sota",
    }

    CONCEPT_KEYWORDS = {
        "什么是",
        "是什么",
        "定义",
        "概念",
        "原理",
        "介绍",
        "解释",
        "meaning",
        "definition",
        "concept",
        "what is",
        "explain",
    }

    def decide(self, question: str) -> SearchDecision:
        question = question.strip()

        if not question:
            return SearchDecision(
                need_search=False,
                reason="问题为空。",
                confidence=1.0,
            )

        text = question.lower()

        # 1. 明显的论文/研究问题 → arXiv。
        if self._contains_any(text, self.RESEARCH_KEYWORDS):
            return SearchDecision(
                need_search=True,
                source="arxiv",
                reason="检测到论文或研究资料需求。",
                confidence=0.95,
            )
        
        # 2. 明显的实时问题 → 优先联网。
        if self._contains_any(text, self.REALTIME_KEYWORDS):
            return SearchDecision(
                need_search=True,
                source="wikipedia",
                reason="检测到实时/最新信息需求。",
                confidence=0.95,
            )

        

        # 3. 概念解释类问题：
        #    第一版默认 Wikipedia。
        if self._contains_any(text, self.CONCEPT_KEYWORDS):
            return SearchDecision(
                need_search=True,
                source="wikipedia",
                reason="检测到概念/定义类问题，优先查询百科资料。",
                confidence=0.85,
            )

        # 4. 第一版无法判断时，不强制联网。
        return SearchDecision(
            need_search=False,
            reason="未检测到明确的外部资料需求。",
            confidence=0.65,
        )

    @staticmethod
    def _contains_any(text: str, keywords: set[str]) -> bool:
        return any(keyword in text for keyword in keywords)