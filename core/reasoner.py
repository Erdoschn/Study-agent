import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .__debug__ import debug
from .evidence import EvidenceStore
from .state import StudentMind


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
- 不要重复已经成功的完全相同工具调用。
- VERIFY 只表示结构化文本核查结果，不表示事实概率或证明。
- 不输出隐藏思维链；reasoning_summary 只写简短、可审计的行动理由。

行动：
SEARCH：搜索知识源。query 应直接服务于当前未解决的问题；必要时下一轮换查询或来源。
CALCULATE：计算需要精确数值结果的表达式。
VERIFY：用当前证据核查一个具体 claim。
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
{"action":"SEARCH|CALCULATE|VERIFY|ANSWER|STOP","reasoning_summary":"简短行动理由","tool":null,"arguments":{},"answer":null,"goal":"","task_type":"","domain":"","claims":[],"evidence_relevance":[],"finish_reason":"","student_model_update":{"short_term":{"beliefs":[],"desires":[],"intentions":[]},"long_term":{"beliefs":[],"desires":[],"intentions":[]},"recent_decisions":[]},"belief_revisions":[]}
    @staticmethod
    def _serialize_observation(observation):
        """Preserve Harness metadata when list-compatible observations enter JSON prompts."""
        if hasattr(observation, "get") and hasattr(observation, "coverage"):
            return {
                "results": observation.get("results", []),
                "coverage": observation.get("coverage", {}),
            }
        return observation

    def _build_prompt(self, state, tool_specs):
        observations = [
            {"step": s.step_id, "action": s.action, "model": s.model, "tool": s.tool,
             "arguments": s.arguments, "reasoning_summary": s.reasoning_summary,
             "observation": self._serialize_observation(s.observation),
             "success": s.success, "error": s.error}
            for s in state.steps
        ]
        analysis = state.task_analysis
        payload = {
            "question": state.question,
            "task_analysis": analysis.__dict__ if analysis else None,
            "plan": [{"action": s.action, "purpose": s.purpose, "tool": s.tool} for s in state.plan.steps] if state.plan else [],
            "current_plan_step": state.current_plan_step,
            "goal": state.goal, "goal_context": list(state.goal_context), "task_type": state.task_type, "domain": state.domain,
            "search_strategy": {"sources": state.search_sources, "sort_by": state.search_sort_by},
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
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

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
        if action not in {"SEARCH", "CALCULATE", "VERIFY", "ANSWER", "STOP"}:
            raise RuntimeError(f"未知 action：{action}")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        claims = data.get("claims", [])
        if not isinstance(claims, list):
            claims = []
        evidence_relevance = data.get("evidence_relevance", [])
        if not isinstance(evidence_relevance, list):
            evidence_relevance = []

        student_model_update = AgentReasoner._normalize_student_model_update(
            data.get("student_model_update", {})
        )
        belief_revisions = AgentReasoner._normalize_belief_revisions(
            data.get("belief_revisions", [])
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
        )

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
