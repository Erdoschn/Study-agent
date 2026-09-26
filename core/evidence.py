from __future__ import annotations

import re
from datetime import datetime
from typing import Any


class EvidenceStore:
    """Persistent evidence collection used by the Harness."""

    def __init__(self, items: list[dict[str, Any]] | None = None):
        self.items: list[dict[str, Any]] = []
        self.add_many(items or [])

    @staticmethod
    def _key(item: dict[str, Any]) -> tuple[str, str]:
        return (str(item.get("source", "")), str(item.get("identifier") or item.get("url") or item.get("title") or ""))

    def add_many(self, items: list[dict[str, Any]]) -> int:
        seen = {self._key(item) for item in self.items}
        added = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            key = self._key(item)
            if key in seen:
                continue
            self.items.append(dict(item))
            seen.add(key)
            added += 1
        return added

    def prompt_view(self, max_items: int = 20) -> list[dict[str, Any]]:
        view = []
        start = max(0, len(self.items) - max_items)
        for index in range(start, len(self.items)):
            item = self.items[index]
            view.append({
                "index": index,
                "source": item.get("source"), "title": item.get("title"),
                "url": item.get("url"), "identifier": item.get("identifier"),
                "abstract": item.get("abstract"), "published": item.get("published"),
                "harness_relevance": item.get("harness_relevance", "UNCERTAIN"),
                "harness_recency": item.get("harness_recency", "UNKNOWN"),
            })
        return view


class EvidenceEngine:
    """Harness-side evidence normalization, deduplication, and qualitative assessment."""

    RELEVANCE = {"DIRECT", "PARTIAL", "TANGENTIAL", "IRRELEVANT", "UNCERTAIN"}
    RECENCY = {"DATED", "UNDATED", "UNKNOWN"}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "and", "or", "in", "on", "for", "with", "uses", "use", "used", "by"}
        tokens = set()
        for token in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", text or ""):
            token = token.lower()
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                if len(token) >= 2:
                    tokens.add(token)
                    tokens.update(token[i:i + 2] for i in range(len(token) - 1))
                continue
            if len(token) <= 1 or token in stopwords:
                continue
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3:
                token = token[:-1]
            tokens.add(token)
        return tokens

    @classmethod
    def assess(cls, query: str, result: dict[str, Any]) -> dict[str, str]:
        q = cls._tokens(query)
        title = cls._tokens(str(result.get("title", "")))
        body = cls._tokens(f"{result.get('abstract', '')} {result.get('notes', '')}")
        combined_text = f"{result.get('title', '')} {result.get('abstract', '')} {result.get('notes', '')}".lower()
        query_text = str(query or "").strip().lower()
        chinese_direct = bool(
            query_text
            and re.search(r"[\u4e00-\u9fff]", query_text)
            and query_text in combined_text
        )
        if not q or not (title or body):
            relevance = "UNCERTAIN"
        elif chinese_direct or q.issubset(title | body):
            relevance = "DIRECT"
        elif q & title:
            relevance = "PARTIAL"
        elif q & body:
            relevance = "TANGENTIAL"
        else:
            relevance = "IRRELEVANT"

        published = str(result.get("published") or result.get("updated") or "").strip()
        recency = "UNDATED"
        if published:
            try:
                datetime.fromisoformat(published.replace("Z", "+00:00"))
                recency = "DATED"
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
    def verify(cls, claim: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        """Structural text matching only; this does not establish factual truth."""
        claim_tokens = cls._tokens(claim)
        # VERIFY should check a concrete statement, not a bare topic label.
        # Requiring at least two substantive tokens prevents a single keyword
        # such as "attention" from being treated as a verified claim.
        if len(claim_tokens) < 2 or not evidence:
            return {
                "claim": claim, "verification_status": "UNCERTAIN",
                "matched_evidence": [],
                "verification_note": "claim 过于简短或缺少证据，Harness 无法进行有意义的结构化文本匹配。",
            }
        matched = []
        for index, item in enumerate(evidence):
            if not isinstance(item, dict) or item.get("harness_relevance") not in {"DIRECT", "PARTIAL"}:
                continue
            text = f"{item.get('title', '')} {item.get('abstract', '')} {item.get('notes', '')}"
            if claim_tokens.issubset(cls._tokens(text)):
                matched.append(index)
        status = "MATCHED" if matched else "NOT_MATCHED"
        return {
            "claim": claim, "verification_status": status,
            "matched_evidence": matched,
            "verification_note": "Harness 仅进行了结构化文本匹配；MATCHED 不等同于事实成立，也不构成概率或置信度判断。",
        }

    @classmethod
    def coverage(cls, query: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        q = cls._tokens(query)
        covered: set[str] = set()
        relevant = []
        for item in evidence:
            if item.get("harness_relevance") in {"DIRECT", "PARTIAL"}:
                relevant.append(item)
                covered |= q & cls._tokens(f"{item.get('title', '')} {item.get('abstract', '')}")
        if not q:
            status = "UNKNOWN"
        elif not relevant:
            status = "INSUFFICIENT"
        elif covered >= q:
            status = "COVERED"
        else:
            status = "PARTIAL"
        return {"status": status, "relevant_count": len(relevant), "uncovered_terms": sorted(q - covered)}