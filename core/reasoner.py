import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .__debug__ import debug
from .evidence import EvidenceStore
from .state import StudentMind
from .search_strategy import SearchStrategy


@dataclass
class ReasoningDecision:
    action: str
    reasoning_summary: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None
    goal: str = ""
    task_type: str = ""
    domain: str = ""
    claims: list[dict[str, Any]] | None = None
    evidence_relevance: list[dict[str, Any]] | None = None
    finish_reason: str = ""
    model: str | None = None
    student_model_update: dict[str, Any] | None = None
    belief_revisions: list[dict[str, Any]] | None = None
    knowledge_relations: list[dict[str, Any]] | None = None


class ModelTimeoutError(TimeoutError):
    """A model request exceeded its configured timeout."""


class ModelClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ModelClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.headers = dict(headers or {})

    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", **self.headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
        except TimeoutError as exc:
            raise ModelTimeoutError(f"LLM 请求超时：{self.timeout}s，model={self.model}") from exc
        except Exception as exc:
            raise RuntimeError(f"LLM 请求失败：{type(exc).__name__}: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"LLM 返回格式异常：{exc}") from exc


class AgentReasoner:
    SYSTEM_PROMPT = """
你是 Study Agent 的决策器。每轮根据当前状态和最近观察选择下一步行动。
循环：OBSERVE → DECIDE → ACT → OBSERVE → …

原则：
- 当前状态优先；不要机械执行旧计划。
- 只选择一个下一步行动。
- 工具由 Harness 执行；不要假设工具成功。
- SEARCH 的 HTTP 成功不代表证据有效。优先参考 Harness 提供的 relevance、recency 和 coverage。
- 证据不足或存在关键缺口时继续行动；证据足够时 ANSWER。
- 不要重复任何已经执行过的完全相同工具调用；失败后必须真正改变 query、source 或参数。
- SEARCH 的 source 由你在每轮决定；TaskAnalyzer 的 search_sources 只是参考，不是强制路由。
- 搜索失败后的策略由 Harness 提供 search_strategy。必须遵守 required_change：查询过长时缩短；中文连续无结果时改用英文核心关键词；连续失败后只用 1~2 个核心词并可更换来源。
- SEARCH 的 source 必须是可用搜索源（通常为 arxiv、wikipedia 或 auto）；不要输出“学术数据库”等自然语言来源名。
- query 必须是搜索关键词，而不是把用户问题整句复制进去。
- VERIFY 只表示结构化文本核查结果，不表示事实概率或证明。
- 不输出隐藏思维链；reasoning_summary 只写简短、可审计的行动理由。
- knowledge_relations 用来显式记录概念之间的知识关系；只有当前问题、证据或已有知识图谱直接支持的关系才填写，不要凭关键词臆测层级。

行动：
SEARCH：搜索知识源。query 应直接服务于当前未解决的问题；必要时下一轮换查询或来源。
CALCULATE：计算需要精确数值结果的表达式。
VERIFY：用当前证据核查一个具体 claim。
ASSESS：生成可评分测试题，必须提供 concepts、difficulty、question_type、question、expected_answer、rubric；difficulty 只能是 basic/undergraduate/graduate/postgraduate/postgraduate_plus。只有 postgraduate 或 postgraduate_plus 的高质量正确表现才可能支持 mastered，低难度题不能证明高级掌握。
ANSWER：回答用户。外部证据被使用时引用 observation 中的来源；证据不足时明确说明。
STOP：无法继续时停止并说明原因。

教学：
- 目标是帮助学生理解，而不只是给出结论。
- 对明确的问题直接解释；需要诊断时再追问。
- 指出错误的具体位置、原因和修正方式。
- 数学、代码、概念题在合适时让学生自己完成关键一步。
- 学生模型只是工作假设；长期信息必须有明确表达或稳定证据支持。
- 不推测隐私、人格、情绪或其他心理事实。

目标上下文：
- goal_context 是 Harness 从历史学习目标中匹配出的上下文，不是用户本轮新说的内容。
- 可以用它辅助教学，但不得把它当作新的用户事实。

学生模型：
- D=学习目标，I=行动计划；recent_decisions 单独记录。
- short_term 记录近期工作假设；long_term 仅记录稳定、重复或明确表达的信息。
- 只记录有对话依据的学习信息。
- belief_revisions 仅在已有 belief 与明确证据或用户明确纠正冲突时提出。
- revision 字段：old/new/horizon/status/reason/evidence_refs；status 只能为 REVISED/CONFIRMED/RETRACTED/UNCERTAIN。

必须只输出 JSON，且 JSON 中包含单词 JSON：
{"action":"SEARCH|CALCULATE|VERIFY|ASSESS|ANSWER|STOP","reasoning_summary":"简短行动理由","tool":null,"arguments":{},"answer":null,"goal":"","task_type":"","domain":"","claims":[],"evidence_relevance":[],"finish_reason":"","student_model_update":{"short_term":{"beliefs":[],"desires":[],"intentions":[]},"long_term":{"beliefs":[],"desires":[],"intentions":[]},"recent_decisions":[]},"belief_revisions":[],"knowledge_relations":[{"source":"","target":"","relation":"related_to","confidence":0.0}]}
"""
    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def decide(self, state, tool_specs=None):
        with debug.scope("AgentReasoner", f"DECIDE → step={state.step_count + 1}"):
            prompt = self._build_prompt(state, tool_specs or [])
            candidates = self.model_router.select_candidates(
                capability="reasoning",
                allow_paid=self.allow_paid,
                task_analysis=state.task_analysis,
                plan=state.plan,
                exclude=set(),
            )
            if not candidates:
                raise RuntimeError("没有可用于 Reasoning 的模型。")
            errors = []
            for model in candidates:
                debug.log("AgentReasoner", f"TRY MODEL → {model.name}")
                try:
                    client = self.model_factory.create(model)
                    decision = self._parse(
                        client.generate(self.SYSTEM_PROMPT, prompt, json_mode=True)
                    )
                    self.model_router.registry.record_success(model.name, "reasoning")
                    decision.model = model.name
                    debug.log("AgentReasoner", f"ACTION → {decision.action}")
                    return decision
                except ModelTimeoutError as exc:
                    self.model_router.registry.record_failure(model.name, "reasoning")
                    errors.append(f"{model.name}: TIMEOUT: {exc}")
                    debug.log("AgentReasoner", f"MODEL TIMEOUT → {model.name}")
                except Exception as exc:
                    self.model_router.registry.record_failure(model.name, "reasoning")
                    errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
                    debug.log("AgentReasoner", f"MODEL FAILED → {model.name}")
            raise RuntimeError(
                "所有 Reasoner 候选模型均调用失败：\n" + "\n".join(errors)
            )

    @staticmethod
    def _serialize_observation(observation):
        """Compact tool observations before they enter the Reasoner prompt."""
        if hasattr(observation, "get") and hasattr(observation, "coverage"):
            raw_results = observation.get("results", []) or []
            compact_results = []
            for item in raw_results[:8]:
                if not isinstance(item, dict):
                    continue
                compact_results.append({
                    "source": item.get("source"),
                    "title": item.get("title"),
                    "identifier": item.get("identifier"),
                    "abstract": str(item.get("abstract", ""))[:500],
                    "harness_relevance": item.get("harness_relevance", "UNCERTAIN"),
                    "harness_recency": item.get("harness_recency", "UNKNOWN"),
                })
            return {
                "results": compact_results,
                "coverage": observation.get("coverage", {}),
            }
        if isinstance(observation, dict):
            return {
                key: (str(value)[:1000] if isinstance(value, str) else value)
                for key, value in observation.items()
            }
        return observation

    def _build_prompt(self, state, tool_specs):
        recent_steps = state.steps[-8:]
        observations = [
            {"step": s.step_id, "action": s.action, "model": s.model, "tool": s.tool,
             "arguments": s.arguments, "reasoning_summary": s.reasoning_summary,
             "observation": self._serialize_observation(s.observation),
             "success": s.success, "error": s.error}
            for s in recent_steps
        ]
        analysis = state.task_analysis
        payload = {
            "question": state.question,
            "task_analysis": analysis.__dict__ if analysis else None,
            "plan": [{"action": s.action, "purpose": s.purpose, "tool": s.tool} for s in state.plan.steps] if state.plan else [],
            "current_plan_step": state.current_plan_step,
            "goal": state.goal, "goal_context": list(state.goal_context), "task_type": state.task_type, "domain": state.domain,
            "search_strategy": {**SearchStrategy.guidance(state.steps, state.last_error_type), "sources_hint": state.search_sources, "sort_by_hint": state.search_sort_by, "routing_authority": "Reasoner"},
            "knowledge_graph": state.knowledge_graph.context_for(state.question) if state.knowledge_graph is not None else {},
            "student_state": {
                "known_topics": sorted(state.student.known_topics),
                "weak_topics": sorted(state.student.weak_topics),
                "misconceptions": state.student.misconceptions,
                "mind_bdi": state.student.mind.as_dict(),
            },
            "available_tools": tool_specs,
            "previous_steps": observations,
            "evidence": EvidenceStore(state.evidence).prompt_view(),
            "claims": state.claims,
            "evidence_relevance": state.evidence_relevance,
            "action_counts": state.action_counts,
            "last_action": state.last_action,
            "last_observation": self._serialize_observation(state.last_observation),
            "step_count": state.step_count,
            "pending_assessment": getattr(state, "pending_assessment", None),
        }
        prompt = json.dumps(payload, ensure_ascii=False, indent=2)
        debug.log(
            "AgentReasoner",
            f"PROMPT → chars={len(prompt)}, recent_steps={len(recent_steps)}/{len(state.steps)}, evidence={len(state.evidence)}, claims={len(state.claims)}",
        )
        return prompt

    @staticmethod
    def _strip_think(raw: str) -> str:
        """Remove model-private <think> blocks before parsing structured output."""
        text = str(raw or "")
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> str:
        """Accept JSON surrounded by harmless whitespace or wrapper text."""
        text = text.strip()
        if not text:
            return text
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
        return text

    @staticmethod
    def _parse(raw):
        raw = AgentReasoner._extract_json(AgentReasoner._strip_think(raw))
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Reasoner JSON 解析失败：{exc}\n原始输出：{raw}") from exc
        action = str(data.get("action", "")).upper()
        if action not in {"SEARCH", "CALCULATE", "VERIFY", "ASSESS", "ANSWER", "STOP"}:
            raise RuntimeError(f"未知 action：{action}")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        claims = AgentReasoner._normalize_claims(data.get("claims", []))
        evidence_relevance = data.get("evidence_relevance", [])
        if not isinstance(evidence_relevance, list):
            evidence_relevance = []

        student_model_update = AgentReasoner._normalize_student_model_update(
            data.get("student_model_update", {})
        )
        belief_revisions = AgentReasoner._normalize_belief_revisions(
            data.get("belief_revisions", [])
        )
        knowledge_relations = AgentReasoner._normalize_knowledge_relations(
            data.get("knowledge_relations", [])
        )
        allowed_relevance = {"DIRECT", "PARTIAL", "TANGENTIAL", "IRRELEVANT", "UNCERTAIN"}
        allowed_recency = {"DATED", "UNDATED", "UNKNOWN", "NEWER", "OLDER", "SAME"}
        normalized = []
        for item in evidence_relevance:
            if not isinstance(item, dict):
                continue
            relevance = str(item.get("relevance", "UNCERTAIN")).upper()
            recency = str(item.get("recency", "UNKNOWN")).upper()
            normalized.append({
                "step": item.get("step"), "index": item.get("index"),
                "relevance": relevance if relevance in allowed_relevance else "UNCERTAIN",
                "recency": recency if recency in allowed_recency else "UNKNOWN",
                "use": bool(item.get("use", False)), "reason": str(item.get("reason", "")),
            })
        return ReasoningDecision(
            action=action,
            reasoning_summary=str(data.get("reasoning_summary", "")),
            tool=str(data["tool"]) if data.get("tool") is not None else None,
            arguments=arguments,
            answer=str(data["answer"]) if data.get("answer") is not None else None,
            goal=str(data.get("goal", "")),
            task_type=str(data.get("task_type", "")),
            domain=str(data.get("domain", "")),
            claims=claims,
            evidence_relevance=normalized,
            finish_reason=str(data.get("finish_reason", "")),
            student_model_update=student_model_update,
            belief_revisions=belief_revisions,
            knowledge_relations=knowledge_relations,
        )

    @staticmethod
    def _normalize_claims(raw):
        if not isinstance(raw, list):
            return []
        normalized = []
        for item in raw:
            if isinstance(item, str):
                claim = item.strip()
                if claim:
                    normalized.append({"claim": claim})
                continue
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            normalized_item = {"claim": claim}
            for key in ("reason", "evidence_refs"):
                if key not in item:
                    continue
                if key == "reason":
                    normalized_item[key] = str(item.get(key, "")).strip()
                elif isinstance(item.get(key), list):
                    normalized_item[key] = [
                        x for x in item[key]
                        if isinstance(x, int) and x >= 0
                    ][:8]
            normalized.append(normalized_item)
        debug.log(
            "AgentReasoner",
            f"CLAIMS → accepted={len(normalized)}",
        )
        return normalized[:12]

    @staticmethod
    def _normalize_belief_revisions(raw):
        """Validate explicit belief revision proposals without deciding truth."""
        if not isinstance(raw, list):
            return []
        allowed = {"REVISED", "CONFIRMED", "RETRACTED", "UNCERTAIN"}
        normalized = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            old = str(item.get("old", "")).strip()
            new = str(item.get("new", "")).strip()
            horizon = str(item.get("horizon", "short_term")).strip()
            status = str(item.get("status", "UNCERTAIN")).upper()
            reason = str(item.get("reason", "")).strip()
            evidence_refs = item.get("evidence_refs", [])
            if not isinstance(evidence_refs, list):
                evidence_refs = []
            evidence_refs = [x for x in evidence_refs if isinstance(x, int) and x >= 0][:8]
            if not old or horizon not in {"short_term", "long_term"}:
                continue
            normalized_item = {
                "old": old,
                "new": new,
                "horizon": horizon,
                "status": status if status in allowed else "UNCERTAIN",
                "reason": reason,
            }
            if "evidence_refs" in item:
                normalized_item["evidence_refs"] = evidence_refs
            normalized.append(normalized_item)
        return normalized[:8]

    @staticmethod
    def _normalize_knowledge_relations(raw):
        if not isinstance(raw, list):
            return []
        from .knowledge_graph import RELATIONS

        normalized = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "related_to")).strip().lower()
            try:
                candidate = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, candidate)) if math.isfinite(candidate) else 0.5
            except (TypeError, ValueError):
                confidence = 0.5
            if not source or not target or source == target or relation not in RELATIONS:
                continue
            normalized.append({
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": round(confidence, 3),
            })
        return normalized[:12]

    @staticmethod
    def _normalize_student_model_update(raw):
        """Bound and sanitize ToM output; it remains a hypothesis, not a fact."""
        if not isinstance(raw, dict):
            return {}
        normalized = {}
        for horizon in ("short_term", "long_term"):
            source = raw.get(horizon, {})
            if not isinstance(source, dict):
                continue
            target = {}
            for category in ("beliefs", "desires", "intentions"):
                items = source.get(category, [])
                if not isinstance(items, list):
                    items = []
                cleaned = []
                for item in items:
                    text = str(item).strip()
                    if text and text not in cleaned:
                        cleaned.append(text)
                target[category] = cleaned[:8 if horizon == "short_term" else 12]
            normalized[horizon] = target
        decisions = raw.get("recent_decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        normalized["recent_decisions"] = [
            str(x).strip() for x in decisions if str(x).strip()
        ][:6]
        return normalized
