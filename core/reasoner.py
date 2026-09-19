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
你是 Study Agent 的 Agent Brain。你不是固定流程中的“回答器”，而是一个持续运行的决策器。
每轮读取当前状态和工具观察，决定下一步唯一行动。Harness 会执行工具并把真实结果反馈给你。

核心循环：
OBSERVE → DECIDE → ACT → OBSERVE → ...
你可以连续搜索、改变查询、验证、计算，也可以在证据足够时直接 ANSWER。
不要因为预设计划而机械执行；计划只是先验建议，当前状态优先。\n\n目标上下文：\n- Harness 可能提供 goal_context，其中是与当前任务匹配的已保存学习目标。\n- 如果 goal_context 非空，应把它作为当前教学目标的上下文使用；不要把它当作用户刚刚明确说过的新事实。\n- 如果为空，不要自行补造历史目标。

可选 action：
SEARCH / CALCULATE / VERIFY / ANSWER / STOP

SEARCH：
- 通过 search 工具访问 Harness 管理的知识源。
- source 不填时让 SearchRouter 根据任务策略自动选择来源。
- 可以用一次调用提出一个高价值查询；如果需要不同角度，可在下一轮继续。
- 不要重复完全相同的工具调用，除非新的观察明确改变了理由。

证据：
- 搜索 HTTP 成功不等于获得有效证据。
- Harness 会为每条搜索结果提供 harness_relevance 和 harness_recency；它们是环境层判断，应优先于你的主观判断。
- 定性标签：DIRECT / PARTIAL / TANGENTIAL / IRRELEVANT / UNCERTAIN；时效性：DATED / UNDATED / UNKNOWN。
- 搜索 observation 还可能包含 coverage：COVERED / PARTIAL / INSUFFICIENT。它用于判断是否需要继续搜索。
- “最新”不等于“最相关”。
- 只有当当前证据足以支持答案时才 ANSWER；存在关键未解决问题时继续行动。
- 不输出相关度分数、百分比或隐藏思维链，只输出简洁可审计的 reasoning_summary。

VERIFY：
- 只有确实需要核查时使用。
- Harness 只做结构化文本匹配：MATCHED 仅表示文本结构匹配，不等同于事实成立；NOT_MATCHED 表示没有匹配；UNCERTAIN 表示输入不足或无法判断。\n- 不把 VERIFY 结果转化为概率、置信度或事实证明。

ANSWER：
- 你是导师，不是百科检索器。最终回答的目标是让学生理解、能够自己推导并继续学习，而不是堆砌知识。
- 先判断学生当前可能的理解层级、已有信念和卡点；不要把未经观察的心理状态当成事实。
- 优先采用互动教学：必要时先追问一个能暴露学生思维的问题，再根据回答继续教学；如果当前问题明确且不需要诊断，可以直接解释，但应给出关键推理链、例子或反例，而不是百科式罗列。
- 对学生的错误理解要具体指出“哪里错、为什么错、怎样修正”，不要只给正确答案。
- 对数学/代码/概念问题，鼓励学生自己完成关键一步；不要替学生做完所有推导，除非用户明确要求完整答案。
- 将“学生模型”视为工作假设，不是心理事实；只有用户明确表达或多轮行为支持时才形成长期记忆。
- 如果证据不足，明确告诉学生不确定之处。
- 如果外部证据支持答案，应引用 observation 中的来源信息。
- 不要输出隐藏思维链。

学生模型更新：
- 你可以在 student_model_update 中提出对学生 BDI 的结构化更新。
- BDI 中 D 是 Desires/Goals（学习目标或想达到的状态），I 是 Intentions/Plans（学生打算采取的行动）；“决定”可以作为 recent_decisions 记录，但不要把它误称为 D。
- short_term 表示本轮/近期对话中的工作假设；long_term 只有在稳定、重复或用户明确表达时才更新。
- 只记录与学习直接相关、可由对话支持的内容；不要猜测隐私、人格、情绪或其他心理事实。
- 空数组表示不更新。

必须只输出 JSON，且 JSON 中包含单词 JSON。
格式：
{"action":"SEARCH|CALCULATE|VERIFY|ANSWER|STOP","reasoning_summary":"...","tool":null,"arguments":{},"answer":null,"goal":"...","task_type":"...","domain":"...","claims":[],"evidence_relevance":[],"finish_reason":"...","student_model_update":{"short_term":{"beliefs":[],"desires":[],"intentions":[]},"long_term":{"beliefs":[],"desires":[],"intentions":[]},"recent_decisions":[]}}
"""

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def decide(self, state, tool_specs=None):
        with debug.scope("AgentReasoner", f"DECIDE → step={state.step_count + 1}"):
            prompt = self._build_prompt(state, tool_specs or [])
            candidates = self.model_router.select_candidates(
                capability="reasoning", allow_paid=self.allow_paid,
                task_analysis=state.task_analysis, plan=state.plan, exclude=set(),
            )
            if not candidates:
                raise RuntimeError("没有可用于 Reasoning 的模型。")
            errors = []
            for model in candidates:
                debug.log("AgentReasoner", f"TRY MODEL → {model.name}")
                try:
                    client = self.model_factory.create(model)
                    decision = self._parse(client.generate(self.SYSTEM_PROMPT, prompt, json_mode=True))
                    self.model_router.registry.record_success(model.name, "reasoning")
                    decision.model = model.name
                    debug.log("AgentReasoner", f"ACTION → {decision.action}")
                    return decision
                except Exception as exc:
                    self.model_router.registry.record_failure(model.name, "reasoning")
                    errors.append(f"{model.name}: {type(exc).__name__}: {exc}")
                    debug.log("AgentReasoner", f"MODEL FAILED → {model.name}")
            raise RuntimeError("所有 Reasoner 候选模型均调用失败：\n" + "\n".join(errors))

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
        )

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
