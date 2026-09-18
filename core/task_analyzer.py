import json
from dataclasses import dataclass, field
from typing import Any

from .__debug__ import debug


@dataclass
class TaskAnalysis:
    task_type: str = "general"
    domain: str = "general"
    goal: str = ""
    issues: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    external_facts_needed: bool = False
    answer_strategy: str = ""
    search_sources: list[str] = field(default_factory=list)


class TaskAnalyzer:
    """Use a reasoning-capable model to understand the task before acting."""

    SYSTEM_PROMPT = """
你是 Study Agent 的任务分析器，不是最终回答器。
请把用户问题转成结构化任务分析，帮助后续 Agent 决定如何行动。

分析：
- task_type：math / coding / conceptual / research / factual / comparison / troubleshooting / general
- domain：尽可能具体的知识领域
- goal：用户真正要解决的目标
- issues：问题中可能存在的概念、逻辑、前提或范围问题；没有则为空
- knowledge_gaps：为了可靠回答仍缺少的关键知识
- required_tools：只填写真正需要的工具，可选 search / calculate / verify
- external_facts_needed：是否需要外部事实、最新信息、论文或网页证据
- answer_strategy：给后续 Reasoner 的简短行动建议
- search_sources：搜索来源的定性优先顺序，只能使用 wikipedia / arxiv。
  基础概念、定义、术语解释优先 wikipedia；
  专业研究、论文、研究进展优先 arxiv；
  同时包含概念与研究的问题可以给出 [wikipedia, arxiv]；
  不需要搜索时填 []。

search_sources 是行动策略，不是相关度分数，也不要填写任何数值评分。

不要因为关键词出现就机械判断需要工具。
不要编造用户没有表达的背景。
不要输出隐藏思维链，只输出简洁、可审计的分析摘要。
必须只输出 JSON。
"""

    def __init__(self, reasoner):
        self.model_router = reasoner.model_router
        self.model_factory = reasoner.model_factory
        self.allow_paid = reasoner.allow_paid

    def analyze(self, question: str, student_state=None) -> TaskAnalysis:
        with debug.scope("TaskAnalyzer", "ANALYZE"):
            prompt = json.dumps(
                {
                    "question": question,
                    "student_state": {
                        "known_topics": sorted(
                            getattr(student_state, "known_topics", set())
                        ),
                        "weak_topics": sorted(
                            getattr(student_state, "weak_topics", set())
                        ),
                        "misconceptions": list(
                            getattr(student_state, "misconceptions", [])
                        ),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            candidates = self.model_router.select_candidates(
                capability="reasoning",
                allow_paid=self.allow_paid,
            )
            if not candidates:
                raise RuntimeError("没有可用于 Task Analysis 的模型。")

            errors = []
            for model in candidates:
                try:
                    debug.log("TaskAnalyzer", f"TRY MODEL → {model.name}")
                    client = self.model_factory.create(model)
                    raw = client.generate(
                        self.SYSTEM_PROMPT,
                        prompt,
                        json_mode=True,
                    )
                    analysis = self._parse(raw)
                    self.model_router.registry.record_success(
                        model.name,
                        "reasoning",
                    )
                    debug.log(
                        "TaskAnalyzer",
                        f"SUCCESS → {model.name}",
                    )
                    return analysis
                except Exception as exc:
                    self.model_router.registry.record_failure(
                        model.name,
                        "reasoning",
                    )
                    errors.append(
                        f"{model.name}: {type(exc).__name__}: {exc}"
                    )
                    debug.log(
                        "TaskAnalyzer",
                        f"MODEL FAILED → {model.name}",
                    )

            raise RuntimeError(
                "所有 Task Analyzer 候选模型均调用失败：
"
                + "
".join(errors)
            )

    @staticmethod
    def _parse(raw: str) -> TaskAnalysis:
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Task Analysis JSON 解析失败：{exc}
原始输出：{raw}"
            ) from exc

        tools = data.get("required_tools", [])
        if not isinstance(tools, list):
            tools = []
        tools = [
            str(x).lower()
            for x in tools
            if str(x).lower() in {"search", "calculate", "verify"}
        ]

        issues = data.get("issues", [])
        gaps = data.get("knowledge_gaps", [])
        if not isinstance(issues, list):
            issues = []
        if not isinstance(gaps, list):
            gaps = []

        sources = data.get("search_sources", [])
        if not isinstance(sources, list):
            sources = []
        sources = [
            str(x).strip().lower()
            for x in sources
            if str(x).strip().lower() in {"wikipedia", "arxiv"}
        ]
        sources = list(dict.fromkeys(sources))

        return TaskAnalysis(
            task_type=str(data.get("task_type", "general")),
            domain=str(data.get("domain", "general")),
            goal=str(data.get("goal", "")),
            issues=[str(x) for x in issues],
            knowledge_gaps=[str(x) for x in gaps],
            required_tools=tools,
            external_facts_needed=bool(
                data.get("external_facts_needed", False)
            ),
            answer_strategy=str(data.get("answer_strategy", "")),
            search_sources=sources,
        )
