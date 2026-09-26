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
不要指定具体搜索来源、搜索排序或工具调用顺序；这些由后续 Reasoner 根据当前证据动态决定。\n不要因为关键词出现就机械判断需要工具。
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
                    debug.log(
                        "TaskAnalyzer",
                        f"RESULT → type={analysis.task_type}, domain={analysis.domain}, tools={analysis.required_tools}, external_facts={analysis.external_facts_needed}",
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
                "所有 Task Analyzer 候选模型均调用失败：\n"
                + "\n".join(errors)
            )

    @staticmethod
    def _strip_think(raw: str) -> str:
        import re
        return re.sub(r"<think>.*?</think>", "", str(raw or ""), flags=re.IGNORECASE | re.DOTALL).strip()

    @staticmethod
    def _extract_json(raw: str) -> str:
        text = TaskAnalyzer._strip_think(raw)
        if not text:
            return text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return text

    @staticmethod
    def _parse(raw: str) -> TaskAnalysis:
        cleaned = TaskAnalyzer._extract_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Task Analysis JSON 解析失败：{exc}\n原始输出：{raw}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError("Task Analysis JSON 解析失败：顶层结果必须是对象。")

        allowed_task_types = {
            "math", "coding", "conceptual", "research", "factual",
            "comparison", "troubleshooting", "explanation", "general",
        }
        task_type = str(data.get("task_type", "general")).strip().lower()
        if task_type not in allowed_task_types:
            task_type = "general"

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

        def clean_list(items, limit=12):
            cleaned = []
            for item in items:
                text = str(item).strip()
                if text and text not in cleaned:
                    cleaned.append(text)
            return cleaned[:limit]

        raw_external = data.get("external_facts_needed", False)
        if isinstance(raw_external, bool):
            external_facts_needed = raw_external
        elif isinstance(raw_external, (int, float)):
            external_facts_needed = bool(raw_external)
        else:
            external_text = str(raw_external).strip().lower()
            external_facts_needed = external_text in {"true", "1", "yes", "y", "on"}

        return TaskAnalysis(
            task_type=task_type,
            domain=str(data.get("domain", "general")).strip() or "general",
            goal=str(data.get("goal", "")).strip(),
            issues=clean_list(issues),
            knowledge_gaps=clean_list(gaps),
            required_tools=tools,
            external_facts_needed=external_facts_needed,
            answer_strategy=str(data.get("answer_strategy", "")).strip(),
        )
