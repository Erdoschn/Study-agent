import json
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from .__debug__ import debug


@dataclass
class ReasoningDecision:
    action: str
    reasoning_summary: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    goal: str = ""
    task_type: str = ""
    domain: str = ""
    claims: list[dict[str, Any]] | None = None
    finish_reason: str = ""
    model: str | None = None


class ModelClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(ModelClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
    ) -> str:
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
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"LLM HTTP {exc.code}: {body}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"LLM 请求失败：{type(exc).__name__}: {exc}"
            ) from exc

        try:
            return json.loads(raw.decode("utf-8"))["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"LLM 返回格式异常：{exc}") from exc


class AgentReasoner:
    SYSTEM_PROMPT = """
你是 Study Agent 的核心决策器，不是最终回答器。
根据任务分析、计划和当前 Agent 状态决定下一步行动。

可选 action：SEARCH / CALCULATE / VERIFY / ANSWER / STOP

计划不是死板流程，但必须遵守两个结构性约束：
1. 已有外部证据且计划包含 VERIFY 时，在 VERIFY 完成前不要直接 ANSWER。
2. 搜索时优先遵守 task_analysis / state 中给出的定性 search_sources 和 search_sort_by。
   除非有明确理由，不要在 SEARCH arguments 中覆盖 source，让 SearchRouter 使用自动来源策略。
   如果 source 被明确指定，则按指定来源执行。

搜索来源只表达定性策略：
- wikipedia：基础概念、定义、术语解释
- arxiv：专业研究、论文、研究进展
- 混合问题可以依次使用多个来源。
如果当前来源没有结果，应根据已有策略继续寻找下一来源，而不是把空结果当成有效证据。

搜索排序只表达定性策略：
- relevance：一般知识/定义
- submittedDate：用户明确要求最新、近期、最近研究或研究进展

工具结果是新的环境观察：先理解观察，再决定下一步。
搜索 HTTP 成功不等于获得有效证据；空结果要继续调整策略。
不要重复完全相同的工具调用，除非最新观察明确改变了调用依据。
不确定时不要编造。
不输出隐藏思维链，只输出简洁、可审计的 reasoning_summary。
必须只输出 JSON，并且 JSON 中包含单词 JSON。

格式：
{"action":"SEARCH|CALCULATE|VERIFY|ANSWER|STOP","reasoning_summary":"...","tool":null,"arguments":{},"goal":"...","task_type":"...","domain":"...","claims":[],"finish_reason":"..."}
"""

    def __init__(self, model_router, model_factory, allow_paid: bool = False):
        self.model_router = model_router
        self.model_factory = model_factory
        self.allow_paid = allow_paid

    def decide(self, state):
        with debug.scope(
            "AgentReasoner",
            f"DECIDE → step={state.step_count + 1}",
        ):
            prompt = self._build_prompt(state)
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
                debug.log(
                    "AgentReasoner",
                    f"TRY MODEL → {model.name}",
                )
                try:
                    client = self.model_factory.create(model)
                    raw = client.generate(
                        self.SYSTEM_PROMPT,
                        prompt,
                        json_mode=True,
                    )
                    decision = self._parse(raw)
                    self.model_router.registry.record_success(
                        model.name,
                        "reasoning",
                    )
                    decision.model = model.name
                    debug.log(
                        "AgentReasoner",
                        f"ACTION → {decision.action}",
                    )
                    return decision
                except Exception as exc:
                    self.model_router.registry.record_failure(
                        model.name,
                        "reasoning",
                    )
                    errors.append(
                        f"{model.name}: {type(exc).__name__}: {exc}"
                    )
                    debug.log(
                        "AgentReasoner",
                        f"MODEL FAILED → {model.name}",
                    )

            raise RuntimeError(
                "所有 Reasoner 候选模型均调用失败：\n"
                + "\n".join(errors)
            )

    def _build_prompt(self, state):
        observations = [
            {
                "step": s.step_id,
                "action": s.action,
                "model": s.model,
                "tool": s.tool,
                "arguments": s.arguments,
                "reasoning_summary": s.reasoning_summary,
                "observation": s.observation,
                "success": s.success,
                "error": s.error,
            }
            for s in state.steps
        ]
        analysis = state.task_analysis
        plan = state.plan
        payload = {
            "question": state.question,
            "task_analysis": analysis.__dict__ if analysis else None,
            "plan": [
                {
                    "action": s.action,
                    "purpose": s.purpose,
                    "tool": s.tool,
                }
                for s in plan.steps
            ] if plan else [],
            "current_plan_step": state.current_plan_step,
            "search_strategy": {
                "sources": state.search_sources,
                "sort_by": state.search_sort_by,
            },
            "goal": state.goal,
            "task_type": state.task_type,
            "domain": state.domain,
            "student_state": {
                "known_topics": sorted(state.student.known_topics),
                "weak_topics": sorted(state.student.weak_topics),
                "misconceptions": state.student.misconceptions,
            },
            "available_tools": ["search", "calculate", "verify"],
            "previous_steps": observations,
            "evidence": state.evidence,
            "claims": state.claims,
            "action_counts": state.action_counts,
            "last_action": state.last_action,
            "last_observation": state.last_observation,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def _parse(raw):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Reasoner JSON 解析失败：{exc}
原始输出：{raw}"
            ) from exc

        action = str(data.get("action", "")).upper()
        if action not in {
            "SEARCH",
            "CALCULATE",
            "VERIFY",
            "ANSWER",
            "STOP",
        }:
            raise RuntimeError(f"未知 action：{action}")

        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        claims = data.get("claims", [])
        if not isinstance(claims, list):
            claims = []

        return ReasoningDecision(
            action=action,
            reasoning_summary=str(
                data.get("reasoning_summary", "")
            ),
            tool=(
                str(data["tool"])
                if data.get("tool") is not None
                else None
            ),
            arguments=arguments,
            goal=str(data.get("goal", "")),
            task_type=str(data.get("task_type", "")),
            domain=str(data.get("domain", "")),
            claims=claims,
            finish_reason=str(data.get("finish_reason", "")),
        )
