from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


class EvidenceEngine:
    """Harness-side evidence normalization, deduplication, and qualitative assessment.

    This layer does not ask the LLM to score its own evidence. It derives labels
    from the query/result content and records the basis for later decisions.
    """

    RELEVANCE = {"DIRECT", "PARTIAL", "TANGENTIAL", "IRRELEVANT", "UNCERTAIN"}
    RECENCY = {"NEWER", "OLDER", "UNKNOWN"}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text or "")
            if len(token) > 1
        }

    @classmethod
    def assess(cls, query: str, result: dict[str, Any]) -> dict[str, str]:
        q = cls._tokens(query)
        title = cls._tokens(str(result.get("title", "")))
        body = cls._tokens(
            f"{result.get('abstract', '')} {result.get('notes', '')}"
        )
        if not q or not (title or body):
            relevance = "UNCERTAIN"
        elif q.issubset(title | body):
            relevance = "DIRECT"
        elif q & title:
            relevance = "PARTIAL"
        elif q & body:
            relevance = "TANGENTIAL"
        else:
            relevance = "IRRELEVANT"

        published = str(result.get("published") or result.get("updated") or "").strip()
        recency = "UNKNOWN"
        if published:
            try:
                value = datetime.fromisoformat(published.replace("Z", "+00:00"))
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - value).days
                recency = "NEWER" if age_days <= 365 else "OLDER"
            except ValueError:
                recency = "UNKNOWN"

        return {"relevance": relevance, "recency": recency}

    @classmethod
    def normalize(cls, query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            key = (
                str(result.get("source", "")),
                str(result.get("identifier") or result.get("url") or result.get("title") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            item = dict(result)
            assessment = cls.assess(query, item)
            item["harness_relevance"] = assessment["relevance"]
            item["harness_recency"] = assessment["recency"]
            normalized.append(item)
        return normalized

    @classmethod
    def coverage(cls, query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        q = cls._tokens(query)
        covered: set[str] = set()
        relevant = []
        for item in evidence:
            if item.get("harness_relevance") in {"DIRECT", "PARTIAL"}:
                relevant.append(item)
                covered |= q & cls._tokens(
                    f"{item.get('title', '')} {item.get('abstract', '')}"
                )
        if not q:
            status = "UNKNOWN"
        elif not relevant:
            status = "INSUFFICIENT"
        elif covered >= q:
            status = "COVERED"
        else:
            status = "PARTIAL"
        return {
            "status": status,
            "relevant_count": len(relevant),
            "uncovered_terms": sorted(q - covered),
        }
