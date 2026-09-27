from __future__ import annotations

import math
import re
from typing import Any

from .__debug__ import debug


class AssessmentEvaluator:
    """Conservative evaluator for structured assessments."""

    def evaluate(self, assessment: dict[str, Any], answer: str, confidence: float | None = None) -> dict[str, Any]:
        expected = str(assessment.get("expected_answer", "")).strip()
        rubric = assessment.get("rubric", [])
        answer = str(answer).strip()
        if not answer:
            raise ValueError("测试答案不能为空。")
        if not expected:
            raise ValueError("assessment 缺少 expected_answer，无法进行安全评分。")
        score = self._score(answer, expected, rubric)
        correct = score >= 0.8
        confidence_value = score
        if confidence is not None:
            try:
                candidate = float(confidence)
                confidence_value = max(0.0, min(1.0, candidate)) if math.isfinite(candidate) else score
            except (TypeError, ValueError):
                confidence_value = score
        debug.log(
            "AssessmentEvaluator",
            f"EVALUATE → expected={expected[:80]!r}, score={score:.3f}, correct={correct}, confidence={confidence_value:.3f}",
        )
        return {"correct": correct, "score": score, "confidence": confidence_value, "evaluation_reason": "答案满足核心评分要求。" if correct else "答案未满足全部核心评分要求。"}

    NEGATION_RE = re.compile(
        r"(?:does\s+not|doesn't|do\s+not|don't|did\s+not|didn't|"
        r"is\s+not|isn't|are\s+not|aren't|was\s+not|wasn't|"
        r"were\s+not|weren't|cannot|can't|never|not)\b"
        r"|(?:不是|并非|不会|不能|没有|未|无|非)"
    )

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return [
            token for token in re.findall(
                r"[a-z0-9_\u4e00-\u9fff]+",
                str(value).lower(),
            )
            if token not in {"a", "i"}
        ]

    @classmethod
    def _negation_conflict(cls, answer: str, expected: str) -> bool:
        """Detect direct polarity reversal instead of treating shared words as proof."""
        expected_tokens = cls._tokens(expected)
        if not expected_tokens:
            return False

        expected_lower = expected.lower()
        # Do not treat a negative expected answer as a positive claim.
        if cls.NEGATION_RE.search(expected_lower):
            return False

        answer_lower = answer.lower()
        for match in cls.NEGATION_RE.finditer(answer_lower):
            before = answer_lower[max(0, match.start() - 80):match.start()]
            after = answer_lower[match.end():match.end() + 120]

            first = expected_tokens[0]
            if len(expected_tokens) == 1:
                if first in after[:80] or first in before[-40:]:
                    return True
                continue

            second = expected_tokens[1]
            if first in before and second in after:
                return True

        return False

    @classmethod
    def _criterion_matches(cls, criterion: str, answer: str) -> bool:
        if not criterion:
            return False
        criterion_tokens = cls._tokens(criterion)
        answer_tokens = cls._tokens(answer)
        if not criterion_tokens:
            return False
        if len(criterion_tokens) == 1:
            token = criterion_tokens[0]
            return token in answer_tokens or token in {"，", "。"}
        answer_set = set(answer_tokens)
        return all(token in answer_set for token in criterion_tokens)

    @classmethod
    def _score(cls, answer: str, expected: str, rubric: list[Any]) -> float:
        a, e = cls._normalize(answer), cls._normalize(expected)
        if cls._negation_conflict(answer, expected):
            return 0.0
        if a == e:
            return 1.0

        criteria = [str(x).strip() for x in rubric if str(x).strip()]
        if criteria:
            return sum(1 for item in criteria if cls._criterion_matches(item, answer)) / len(criteria)

        tokens = cls._tokens(expected)
        normalized_answer_tokens = set(cls._tokens(answer))
        return (
            sum(1 for token in tokens if token in normalized_answer_tokens) / len(tokens)
            if tokens else 0.0
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", str(value).strip().lower())
