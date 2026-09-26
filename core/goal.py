"""Deterministic matching of the current learning goal against saved goals."""

import re

from .__debug__ import debug


class GoalMatcher:
    """Match current task text to saved learning goals without LLM self-scoring."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        text = str(text or "").lower()
        words = re.findall(r"[A-Za-z0-9]+", text)
        chinese_runs = re.findall(r"[\u4e00-\u9fff]+", text)
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
            "with", "is", "are", "how", "what", "why", "do", "does", "i",
        }
        chinese_stop = {
            "学习", "理解", "掌握", "解决", "当前", "问题", "相关",
            "研究", "知识", "概念", "内容", "方法", "一下", "这个",
        }
        tokens = {x for x in words if x not in stop and len(x) > 1}
        for run in chinese_runs:
            if len(run) >= 2:
                if run not in chinese_stop:
                    tokens.add(run)
                for pair in (run[i:i + 2] for i in range(len(run) - 1)):
                    if pair not in chinese_stop:
                        tokens.add(pair)
        return tokens

    @classmethod
    def match(cls, current: str, saved_goals: list[str]) -> list[str]:
        """Return only directly relevant saved goals, preserving stored order."""
        current_tokens = cls._tokens(current)
        if not current_tokens:
            return []
        matched = []
        for goal in saved_goals or []:
            goal_text = str(goal).strip()
            if not goal_text:
                continue
            goal_tokens = cls._tokens(goal_text)
            overlap = current_tokens & goal_tokens
            if any(len(token) >= 2 for token in overlap):
                matched.append(goal_text)
        debug.log(
            "GoalMatcher",
            f"MATCH → current={current!r}, candidates={len(saved_goals or [])}, matched={len(matched)}",
        )
        return matched

    @classmethod
    def context_for(cls, question: str, current_goal: str, saved_goals: list[str]) -> list[str]:
        query = " ".join(x for x in (current_goal, question) if str(x).strip())
        return cls.match(query, saved_goals)
