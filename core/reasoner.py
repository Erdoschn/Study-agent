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
from .goal import GoalMatcher
from .model_router import get_model_choices, call_model_with_effort
from .prompt_config import get_prompt


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
    effort: str | None = None
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

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
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
    SYSTEM_PROMPT = get_prompt("study_agent.reasoner")


    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def decide(self, state, tool_specs=None):
        with debug.scope("AgentReasoner", f"DECIDE → step={state.step_count + 1}"):
            prompt = self._build_prompt(state, tool_specs or [])
            choices = get_model_choices(
                self.model_router,
                "reasoning",
                allow_paid=self.allow_paid,
                task_analysis=state.task_analysis,
                plan=state.plan,
                exclude=set(),
            )
            if not choices:
                raise RuntimeError("没有可用于 Reasoning 的模型。")
            errors = []
            for choice in choices:
                model = choice.model
                debug.log(
                    "AgentReasoner",
                    f"TRY MODEL → {model.name} effort={choice.effort or 'default'}",
                )
                try:
                    client = self.model_factory.create(model)
                    decision = self._parse(
                        call_model_with_effort(
                            self.model_router,
                            client,
                            self.SYSTEM_PROMPT,
                            prompt,
                            json_mode=True,
                            reasoning_effort=choice.effort,
                            reasoning_effort_param=getattr(choice.model, "reasoning_effort_param", None),
                        )
                    )
                    self.model_router.registry.record_success(model.name, "reasoning")
                    decision.model = model.name
                    decision.effort = choice.effort
                    debug.log("AgentReasoner", f"ACTION → {decision.action}")
                    return decision
                except ModelTimeoutError as exc:
                    self.model_router.registry.record_failure(
                        model.name, "reasoning",
                        provider_level=self.model_router.registry.is_provider_level_failure(exc),
                    )
                    errors.append(f"{model.name}: TIMEOUT: {exc}")
                    debug.log("AgentReasoner", f"MODEL TIMEOUT → {model.name}")
                except Exception as exc:
                    self.model_router.registry.record_failure(
                        model.name, "reasoning",
                        provider_level=self.model_router.registry.is_provider_level_failure(exc),
                    )
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
            for item in raw_results[:12]:
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
            "goal": state.goal, "goal_context": list(state.goal_context), "task_goals": GoalMatcher.extract_goals(state.question),
            "task_type": state.task_type, "domain": state.domain,
            "search_strategy": {**SearchStrategy.guidance(state.steps, state.last_error_type), "sources_hint": state.search_sources, "sort_by_hint": state.search_sort_by, "routing_authority": "Reasoner"},
            "knowledge_graph": state.knowledge_graph.context_for(state.question) if state.knowledge_graph is not None else {},
            "student_state": {
                "known_topics": sorted(state.student.known_topics),
                "weak_topics": sorted(state.student.weak_topics),
                "learning_topics": sorted(state.student.learning_topics),
                "misconceptions": state.student.misconceptions,
                "mind_bdi": state.student.mind.as_dict(),
            },
            "available_tools": tool_specs,
            "previous_steps": observations,
            "evidence": EvidenceStore(state.evidence).prompt_view(max_items=12),
            "claims": state.claims,
            "evidence_relevance": state.evidence_relevance,
            "action_counts": state.action_counts,
            "last_action": state.last_action,
            "last_observation": self._serialize_observation(state.last_observation),
            "step_count": state.step_count,
            "pending_assessment": getattr(state, "pending_assessment", None),
        }
        # Compact JSON preserves every field while substantially reducing
        # prompt bytes/tokens and therefore model prefill latency.
        serialized_prompt = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        debug.log(
            "AgentReasoner",
            f"PROMPT → chars={len(serialized_prompt)}, recent_steps={len(recent_steps)}/{len(state.steps)}, evidence={len(state.evidence)}, claims={len(state.claims)}",
        )
        return serialized_prompt

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
        if not isinstance(data, dict):
            raise RuntimeError("Reasoner JSON 解析失败：顶层结果必须是对象。")
        action = str(data.get("action", "")).upper()
        if action not in {"SEARCH", "CALCULATE", "VERIFY", "ASSESS", "ANSWER", "STOP"}:
            raise RuntimeError(f"未知 action：{action}")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        tool = str(data.get("tool", "")).strip().lower() if data.get("tool") is not None else None
        expected_tools = {
            "SEARCH": "search",
            "CALCULATE": "calculate",
            "VERIFY": "verify",
            "ASSESS": "assess",
        }
        expected_tool = expected_tools.get(action)
        if expected_tool and tool and tool != expected_tool:
            raise RuntimeError(
                f"action={action} 与 tool={tool} 不一致，应为 {expected_tool}。"
            )
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
        answer = str(data.get("answer", "")).strip() if data.get("answer") is not None else ""
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
            tool=tool,
            arguments=arguments,
            answer=answer or None,
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
            normalized_item = {
                "source": source,
                "target": target,
                "relation": relation,
                "confidence": round(confidence, 3),
            }
            refs = item.get("evidence_refs")
            if isinstance(refs, list):
                normalized_item["evidence_refs"] = [
                    x for x in refs if isinstance(x, int) and x >= 0
                ][:8]
            normalized.append(normalized_item)
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
