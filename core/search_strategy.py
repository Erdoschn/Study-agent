from __future__ import annotations

import re
from typing import Any


class SearchStrategy:
    """Deterministic search-recovery policy used to guide the Reasoner after weak searches."""

    MAX_LONG_QUERY_CHARS = 20
    MAX_CORE_TERMS = 3

    _CN_STOP = {
        "的", "和", "与", "及", "相关", "研究", "现状", "目的", "结论",
        "定义", "是什么", "有哪些", "如何", "为什么", "以及", "进行", "了解",
    }

    @classmethod
    def guidance(cls, steps: list[Any]) -> dict[str, Any]:
        searches = [s for s in steps if getattr(s, "action", "") == "SEARCH"]
        empty = [s for s in searches if not getattr(s, "success", True)
                 and "SEARCH_EMPTY" in str(getattr(s, "error", ""))]
        if not empty:
            return {"stage": "initial", "required_change": "none",
                    "instruction": "先使用最直接的核心查询。"}

        query = str((getattr(empty[-1], "arguments", {}) or {}).get("query", "")).strip()
        count = len(empty)

        if count == 1 and len(query) > cls.MAX_LONG_QUERY_CHARS:
            return {"stage": "shorten", "required_change": "shorter_query",
                    "suggested_query": cls.core_query(query),
                    "instruction": "上一轮无结果且查询过长；下一轮必须明显缩短，只保留核心概念。"}

        if count <= 2 and cls._has_cjk(query):
            return {"stage": "english", "required_change": "english_query",
                    "suggested_query": None,
                    "instruction": "中文查询连续无结果；下一轮必须改用英文核心关键词，不要继续提交中文长句。"}

        if count == 3:
            return {"stage": "compact", "required_change": "one_or_two_terms",
                    "suggested_query": None,
                    "instruction": "连续三次无结果；下一轮只使用1~2个最核心的英文关键词，并可更换搜索源。"}

        return {"stage": "alternate", "required_change": "new_query_or_source",
                "suggested_query": None,
                "instruction": "搜索策略已失败多次；必须改变查询词或搜索源，禁止重复历史搜索。"}

    @classmethod
    def core_query(cls, query: str) -> str:
        text = re.sub(r"[，。！？；：、,.!?;:]+", " ", str(query or "")).strip()
        m = re.match(r"^(.{2,20}?)的(?:定义|研究|现状|目的|结论|应用|成果)", text)
        if m:
            return m.group(1).strip()

        parts = [p.strip() for p in text.split() if p.strip()]
        if not parts:
            parts = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{1,}", text)

        cleaned: list[str] = []
        for part in parts:
            part = re.sub(r"(是什么|有哪些|如何|为什么)$", "", part)
            part = re.sub(r"(的定义|相关研究|研究现状|研究成果|应用目的|研究目的|研究结论|结论)$", "", part)
            if part and part not in cls._CN_STOP and part not in cleaned:
                cleaned.append(part)
        return " ".join(cleaned[:cls.MAX_CORE_TERMS]) or text[:18]

    @classmethod
    def query_signature(cls, query: str) -> str:
        text = re.sub(r"[^A-Za-z0-9\\u4e00-\\u9fff]+", " ", str(query or "").lower())
        text = re.sub(
            r"(是什么|有哪些|如何|为什么|的定义|相关研究|研究现状|研究成果|研究结论|应用目的|研究目的|结论|研究|现状|目的|定义|成果|应用|相关)",
            " ",
            text,
        )
        return re.sub(r"\\s+", " ", text).strip()

    @classmethod
    def is_ineffective_rewrite(cls, query: str, history: list[str]) -> bool:
        current = cls.query_signature(query)
        if not current:
            return False
        for previous in history:
            old = cls.query_signature(previous)
            if old and (current == old or current in old or old in current):
                return True
        return False

    @staticmethod
    def _has_cjk(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text or ""))
