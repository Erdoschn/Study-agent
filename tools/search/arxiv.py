from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from core.__debug__ import debug
from .base import SearchProvider
from .models import SearchQuery, SearchResult


class ArxivSearchProvider(SearchProvider):
    """arXiv Atom API 搜索提供器。"""

    name = "arxiv"
    API_URL = "https://export.arxiv.org/api/query"
    MIN_REQUEST_INTERVAL = 3.0
    REQUEST_TIMEOUT = 30
    _last_request_time = 0.0

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def search(self, query: SearchQuery) -> list[SearchResult]:
        text = query.query.strip()
        if not text:
            raise ValueError("arXiv query 不能为空。")

        search_query = self._build_search_query(text, query.categories)
        params = {
            "search_query": search_query,
            "start": "0",
            "max_results": str(max(1, min(query.max_results, 50))),
            "sortBy": query.sort_by or "relevance",
            "sortOrder": query.sort_order or "descending",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"

        debug.log("ArxivSearchProvider", f"SEARCH QUERY → {search_query}")
        debug.log("ArxivSearchProvider", f"REQUEST URL → {url}")

        self._wait_for_rate_limit()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "StudyAgent/2.0 (educational research client; arXiv API search)",
                "Accept": "application/atom+xml",
            },
        )

        try:
            with urllib.request.urlopen(request, timeout=self.REQUEST_TIMEOUT) as response:
                if response.status != 200:
                    raise RuntimeError(f"arXiv API HTTP {response.status}")
                data = response.read()
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read(2048).decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            debug.log(
                "ArxivSearchProvider",
                f"HTTP ERROR → {exc.code} {exc.reason}; body={body or '<empty>'}",
            )
            raise RuntimeError(
                f"arXiv API 请求失败：HTTPError: HTTP Error {exc.code}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"arXiv API 网络请求失败：{exc.reason}") from exc

        return self._parse_atom(data)

    @staticmethod
    def _build_search_query(text: str, categories: list[str]) -> str:
        """只负责生成 arXiv search_query；不混入 HTTP 层逻辑。"""
        if text.startswith('"') and text.endswith('"'):
            phrase = text[1:-1].strip()
            expression = f'all:"{phrase}"'
        elif any(op in text.upper().split() for op in ("AND", "OR", "ANDNOT")):
            expression = text
        else:
            expression = f"all:{text}"

        cats = [f"cat:{c.strip()}" for c in categories if c.strip()]
        if cats:
            expression += " AND (" + " OR ".join(cats) + ")"
        return expression

    @classmethod
    def _wait_for_rate_limit(cls) -> None:
        now = time.monotonic()
        delay = cls.MIN_REQUEST_INTERVAL - (now - cls._last_request_time)
        if delay > 0:
            time.sleep(delay)
        cls._last_request_time = time.monotonic()

    def _parse_atom(self, data: bytes) -> list[SearchResult]:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            raise RuntimeError(f"arXiv XML 解析失败：{exc}") from exc

        results: list[SearchResult] = []
        for entry in root.findall("atom:entry", self.NS):
            identifier = self._text(entry.find("atom:id", self.NS))
            results.append(
                SearchResult(
                    source=self.name,
                    source_type="paper",
                    title=self._normalize(self._text(entry.find("atom:title", self.NS))),
                    url=self._find_html_url(entry) or identifier,
                    abstract=self._normalize(self._text(entry.find("atom:summary", self.NS))),
                    authors=[
                        self._text(author.find("atom:name", self.NS))
                        for author in entry.findall("atom:author", self.NS)
                        if self._text(author.find("atom:name", self.NS))
                    ],
                    published=self._text(entry.find("atom:published", self.NS)),
                    updated=self._text(entry.find("atom:updated", self.NS)),
                    identifier=self._extract_id(identifier),
                )
            )

        debug.log("ArxivSearchProvider", f"RESULTS → {len(results)}")
        return results

    def _find_html_url(self, entry) -> str:
        for link in entry.findall("atom:link", self.NS):
            if link.attrib.get("type") == "text/html" and link.attrib.get("href"):
                return link.attrib["href"]
        return ""

    @staticmethod
    def _text(node) -> str:
        return "" if node is None else "".join(node.itertext()).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _extract_id(identifier: str) -> str:
        return identifier.rsplit("/abs/", 1)[-1] if "/abs/" in identifier else identifier
