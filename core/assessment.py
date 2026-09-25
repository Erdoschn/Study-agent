from __future__ import annotations

import re
from typing import Any


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
        confidence_value = score if confidence is None else max(0.0, min(1.0, float(confidence)))
        return {"correct": correct, "score": round(score, 3), "confidence": round(confidence_value, 3), "evaluation_reason": "答案满足核心评分要求。" if correct else "答案未满足全部核心评分要求。"}

    @classmethod
    def _score(cls, answer: str, expected: str, rubric: list[Any]) -> float:
        a, e = cls._normalize(answer), cls._normalize(expected)
        if a == e or (len(e) >= 4 and e in a):
            return 1.0
        criteria = [cls._normalize(x) for x in rubric if str(x).strip()]
        if criteria:
            return sum(1 for item in criteria if item and item in a) / len(criteria)
        tokens = [x for x in re.findall(r"[a-z0-9\u4e00-\u9fff]+", e) if len(x) > 1]
        return sum(1 for token in tokens if token in a) / len(tokens) if tokens else 0.0

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", "", str(value).strip().lower())
