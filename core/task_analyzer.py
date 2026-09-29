import json
from dataclasses import dataclass, field
from typing import Any

from .__debug__ import debug
from .model_router import get_model_choices, call_model_with_effort


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
    difficulty: int = 3
    assessment_requested: bool = False
    assessment_concept: str = ""
    assessment_difficulty: str = "graduate"


def is_explicit_assessment_request(question: str) -> bool:
    text = str(question or "").strip().lower()
    markers = (
        "出题", "给我一道题", "给我一题", "来一道题", "来道题",
        "测试我", "测试一下", "测测我", "测一下", "考考我", "检查", "创建测试",
        "检验一下", "检验我的理解", "做题", "quiz", "test me",
        "give me a question", "assess me",
    )
    return any(marker in text for marker in markers)


class TaskAnalyzer:
    """Use a reasoning-capable model to understand the task before acting."""

    SYSTEM_PROMPT = """
你是 Study Agent 的任务分析器，不是最终回答器。
请把用户问题转成结构化任务分析，帮助后续 Agent 决定如何行动。

分析：
- task_type：math / coding / conceptual / research / factual / comparison / troubleshooting / general
- domain：尽可能具体的知识领域。注意：上下文缓存 / context caching / prompt caching / KV cache 默认属于“大语言模型 / LLM 推理 / 模型服务”，只有明确出现 CPU cache、缓存行、L1/L2/L3、缓存一致性等术语时才归入计算机体系结构。
- goal：用户真正要解决的目标
- issues：问题中可能存在的概念、逻辑、前提或范围问题；没有则为空
- knowledge_gaps：为了可靠回答仍缺少的关键知识
- required_tools：只填写真正需要的工具，可选 search / calculate / verify
- external_facts_needed：是否需要外部事实、最新信息、论文或网页证据
- answer_strategy：给后续 Reasoner 的简短行动建议
- difficulty：任务难度 1-5；1=直接事实/简单解释，3=需要工具或多步推理，5=复杂研究、多轮证据整合或高难度推理
- assessment_requested：只有用户明确要求“出题/测试/测测我/检验理解”等时才为 true；普通教学回答不要主动设置为 true。
- assessment_concept：用户要检验的主要知识点；若用户明确给出则原样保留，否则尽量从问题中提取，不要凭空创造。
- assessment_difficulty：若用户明确指定难度，使用 basic/undergraduate/graduate/postgraduate/postgraduate_plus；否则使用 graduate。
- 不要决定执行路径。不要输出 execution_mode；后续 ModelRouter 会根据任务信号自动决定 direct / direct_verified / reasoner。
不要指定具体搜索来源、搜索排序或工具调用顺序；这些由后续 Reasoner 根据当前证据动态决定。
不要因为关键词出现就机械判断需要工具。
输出中若 assessment_requested=true，必须同时填写 assessment_concept；普通问题必须为 false。
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
            choices = get_model_choices(
                self.model_router,
                "reasoning",
                allow_paid=self.allow_paid,
            )
            if not choices:
                raise RuntimeError("没有可用于 Task Analysis 的模型。")

            errors = []
            for choice in choices:
                model = choice.model
                debug.log(
                    "TaskAnalyzer",
                    f"TRY MODEL → {model.name} effort={choice.effort or 'default'}",
                )
                try:
                    client = self.model_factory.create(model)
                    raw = call_model_with_effort(
                        self.model_router,
                        client,
                        self.SYSTEM_PROMPT,
                        prompt,
                        json_mode=True,
                        reasoning_effort=choice.effort,
                        reasoning_effort_param=getattr(choice.model, "reasoning_effort_param", None),
                    )
                    analysis = self._parse(raw)
                    analysis.domain = self._normalize_domain(question, analysis.domain)
                    analysis.assessment_requested = (
                        analysis.assessment_requested
                        or self._is_explicit_assessment_request(question)
                    )
                    if analysis.assessment_requested and not analysis.assessment_concept:
                        analysis.assessment_concept = self._extract_assessment_concept(
                            question,
                            analysis.domain,
                        )
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
                        f"RESULT → type={analysis.task_type}, domain={analysis.domain}, difficulty={analysis.difficulty}, tools={analysis.required_tools}, external_facts={analysis.external_facts_needed}, assessment={analysis.assessment_requested}",
                    )
                    return analysis
                except Exception as exc:
                    self.model_router.registry.record_failure(
                        model.name,
                        "reasoning",
                        provider_level=self.model_router.registry.is_provider_level_failure(exc),
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
    def _is_explicit_assessment_request(question: str) -> bool:
        text = str(question or "").strip().lower()
        markers = (
            "出题", "给我一道题", "给我一题", "来一道题", "来道题",
            "测试我", "测试一下", "测测我", "测一下", "考考我", "检查", "创建测试", "检验一下", "检验我的理解",
            "做题", "quiz", "test me", "give me a question", "assess me",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _extract_assessment_concept(question: str, domain: str) -> str:
        import re
        text = str(question or "").strip()
        # Prefer explicit quoted/topic phrases, then strip common assessment language.
        quoted = re.findall(r'[“”"]([^“”"]+)[“”"]', text)
        for candidate in quoted:
            candidate = candidate.strip()
            if candidate:
                return candidate[:120]
        candidate = re.sub(
            r"(请|帮我|给我|来|出|一道|一题|个|题目|题|测试|测测|考考|检验|一下|我的理解|quiz|test me|give me a question|assess me)",
            " ",
            text,
            flags=re.I,
        )
        candidate = re.sub(r"\s+", " ", candidate).strip(" ：:，,。！？?!")
        if candidate:
            return candidate[:120]
        return str(domain or "").strip()

    @staticmethod
    def _normalize_domain(question: str, domain: str) -> str:
        """Normalize only explicit cache terminology in the user query.

        The model's domain is the primary task scope. Semantic anchors such as
        "transformer" must not be reclassified merely because the question
        happens to be processed by the LLM/cache normalization rules.
        """
        text = str(question or "").lower()
        llm_cache_terms = (
            "上下文缓存", "context caching", "context cache",
            "prompt caching", "prompt cache", "kv cache", "kv-cache",
            "kvcache", "kv缓存",
        )
        cpu_terms = (
            "cpu缓存", "cpu cache", "cache line", "缓存行",
            "l1 cache", "l2 cache", "l3 cache",
            "一级缓存", "二级缓存", "三级缓存",
            "cache coherence", "缓存一致性",
            "计算机体系结构", "computer architecture",
        )
        has_llm_cache = any(term in text for term in llm_cache_terms)
        has_cpu = any(term in text for term in cpu_terms)
        if has_llm_cache and not has_cpu:
            if any(term in text for term in ("kv cache", "kv-cache", "kvcache", "kv缓存")):
                return "LLM推理-KV Cache"
            return "LLM推理-上下文缓存"
        if has_cpu and not has_llm_cache:
            return "计算机体系结构-CPU缓存"
        if has_llm_cache and has_cpu:
            return domain.strip() or "缓存机制-跨领域比较"
        return domain.strip() or "general"

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

        raw_difficulty = data.get("difficulty", 3)
        try:
            difficulty = int(raw_difficulty)
        except (TypeError, ValueError):
            difficulty = 3
        difficulty = max(1, min(5, difficulty))

        raw_external = data.get("external_facts_needed", False)
        if isinstance(raw_external, bool):
            external_facts_needed = raw_external
        elif isinstance(raw_external, (int, float)):
            external_facts_needed = bool(raw_external)
        else:
            external_text = str(raw_external).strip().lower()
            external_facts_needed = external_text in {"true", "1", "yes", "y", "on"}

        requested_raw = data.get("assessment_requested", False)
        if isinstance(requested_raw, bool):
            assessment_requested = requested_raw
        else:
            assessment_requested = str(requested_raw).strip().lower() in {"true", "1", "yes", "y", "on"}

        assessment_concept = str(data.get("assessment_concept", "")).strip()
        assessment_difficulty = str(data.get("assessment_difficulty", "graduate")).strip().lower()
        allowed_assessment_levels = {"basic", "undergraduate", "graduate", "postgraduate", "postgraduate_plus"}
        if assessment_difficulty not in allowed_assessment_levels:
            assessment_difficulty = "graduate"

        return TaskAnalysis(
            task_type=task_type,
            domain=str(data.get("domain", "general")).strip() or "general",
            goal=str(data.get("goal", "")).strip(),
            issues=clean_list(issues),
            knowledge_gaps=clean_list(gaps),
            required_tools=tools,
            external_facts_needed=external_facts_needed,
            answer_strategy=str(data.get("answer_strategy", "")).strip(),
            difficulty=difficulty,
            assessment_requested=assessment_requested,
            assessment_concept=assessment_concept,
            assessment_difficulty=assessment_difficulty,
        )
