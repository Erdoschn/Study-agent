from __future__ import annotations

from core.__debug__ import debug
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import SearchProvider
from .models import SearchQuery, SearchResult


class ArxivSearchProvider(SearchProvider):
    """arXiv Public API 搜索适配器。"""

    name = "arxiv"

    API_URLS = (
        "https://export.arxiv.org/api/query",
        "https://arxiv.org/api/query",
    )

    MIN_REQUEST_INTERVAL = 3.0
    REQUEST_TIMEOUT = 30
    _last_request_time = 0.0

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def search(self, query: SearchQuery) -> list[SearchResult]:
        debug.log("ArxivSearchProvider", f"SEARCH → {query.query}")

        if not query.query.strip():
            raise ValueError("arXiv query 不能为空。")

        max_results = max(1, min(query.max_results, 50))
        search_expression = query.query.strip()

        if search_expression.startswith('"') and search_expression.endswith('"'):
            search_expression = f"all:{search_expression}"

        categories = [c.strip() for c in query.categories if c.strip()]
        if categories:
            category_expression = " OR ".join(
                f"cat:{category}" for category in categories
            )
            # arXiv API 对 OR 分类表达式支持括号；保留查询字段和分类字段
            # 的标准布尔结构，不对整个 search expression 再包一层括号。
            search_expression = f"{search_expression} AND ({category_expression})"

        debug.log(
            "ArxivSearchProvider",
            f"FINAL SEARCH EXPRESSION → {search_expression}",
        )

        params = {
            "search_query": search_expression,
            "start": 0,
            "max_results": max_results,
            "sortBy": query.sort_by,
            "sortOrder": query.sort_order,
        }

        self._wait_for_rate_limit()

        last_error: Exception | None = None
        for index, api_url in enumerate(self.API_URLS):
            url = api_url + "?" + urllib.parse.urlencode(params)
            request = self._build_request(url)

            debug.log("ArxivSearchProvider", f"REQUEST URL → {url}")
            debug.log(
                "ArxivSearchProvider",
                f"REQUEST HEADERS → {dict(request.header_items())}",
            )
            debug.log(
                "ArxivSearchProvider",
                f"API CALL → {api_url}"
                + (" (406 fallback)" if index else ""),
            )

            try:
                xml_data = self._request(request)
                return self._parse_atom(xml_data)
            except urllib.error.HTTPError as exc:
                last_error = exc
                self._log_http_error(exc)
                if exc.code != 406 or index == len(self.API_URLS) - 1:
                    break
            except Exception as exc:
                last_error = exc
                debug.log(
                    "ArxivSearchProvider",
                    f"REQUEST ERROR → {type(exc).__name__}: {exc}",
                )
                break

            self._wait_for_rate_limit()

        raise RuntimeError(
            "arXiv API 请求失败："
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    @classmethod
    def _wait_for_rate_limit(cls) -> None:
        now = time.monotonic()
        elapsed = now - cls._last_request_time
        if elapsed < cls.MIN_REQUEST_INTERVAL:
            time.sleep(cls.MIN_REQUEST_INTERVAL - elapsed)
        cls._last_request_time = time.monotonic()

    @staticmethod
    def _build_request(url: str) -> urllib.request.Request:
        return urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "User-Agent": (
                    "StudyAgent/2.0 (educational research client; "
                    "arXiv API search)"
                ),
                "Accept": "*/*",
                "Connection": "keep-alive",
            },
        )

    def _request(self, request: urllib.request.Request) -> bytes:
        with urllib.request.urlopen(
            request,
            timeout=self.REQUEST_TIMEOUT,
        ) as response:
            debug.log("ArxivSearchProvider", f"HTTP STATUS → {response.status}")
            debug.log(
                "ArxivSearchProvider",
                f"RESPONSE HEADERS → {dict(response.headers.items())}",
            )
            return response.read()

    @staticmethod
    def _log_http_error(exc: urllib.error.HTTPError) -> None:
        debug.log(
            "ArxivSearchProvider",
            f"HTTP STATUS → {exc.code} {exc.reason}",
        )
        debug.log(
            "ArxivSearchProvider",
            f"RESPONSE HEADERS → {dict(exc.headers.items()) if exc.headers else {}}",
        )
        try:
            body = exc.read(2048)
        except Exception as read_exc:
            debug.log(
                "ArxivSearchProvider",
                f"RESPONSE BODY → <unreadable: {type(read_exc).__name__}: {read_exc}>",
            )
            return
        text = body.decode("utf-8", errors="replace").strip()
        if len(text) > 1000:
            text = text[:1000] + "..."
        debug.log(
            "ArxivSearchProvider",
            f"RESPONSE BODY → {text or '<empty>'}",
        )

    def _parse_atom(self, xml_data: bytes) -> list[SearchResult]:
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as exc:
            raise RuntimeError(f"arXiv XML 解析失败：{exc}") from exc

        results: list[SearchResult] = []
        for entry in root.findall("atom:entry", self.NS):
            identifier_url = self._text(entry.find("atom:id", self.NS))
            title = self._normalize(self._text(entry.find("atom:title", self.NS)))
            abstract = self._normalize(
                self._text(entry.find("atom:summary", self.NS))
            )
            published = self._text(entry.find("atom:published", self.NS))
            updated = self._text(entry.find("atom:updated", self.NS))
            html_url = self._find_html_url(entry) or identifier_url

            authors = []
            for author in entry.findall("atom:author", self.NS):
                name = self._text(author.find("atom:name", self.NS))
                if name:
                    authors.append(name)

            results.append(
                SearchResult(
                    source=self.name,
                    source_type="paper",
                    title=title,
                    url=html_url,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    updated=updated,
                    identifier=self._extract_id(identifier_url),
                )
            )

        debug.log("ArxivSearchProvider", f"RESULTS → {len(results)}")
        return results

    def _find_html_url(self, entry) -> str:
        for link in entry.findall("atom:link", self.NS):
            href = link.attrib.get("href", "")
            if href and link.attrib.get("type", "") == "text/html":
                return href
        return ""

    @staticmethod
    def _text(node) -> str:
        return "" if node is None else "".join(node.itertext()).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _extract_id(identifier: str) -> str:
        marker = "/abs/"
        if marker in identifier:
            return identifier.rsplit(marker, 1)[1]
        return identifier
