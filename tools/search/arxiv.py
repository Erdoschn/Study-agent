from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from core.__debug__ import debug
from .base import SearchProvider
from .http import HttpClient, HttpRequestError
from .models import SearchError, SearchQuery, SearchResponse, SearchResult


class ArxivSearchProvider(SearchProvider):
    """arXiv Atom API provider with transport isolation and endpoint fallback."""

    name = "arxiv"
    API_URL = "https://export.arxiv.org/api/query"
    API_URLS = (
        "https://export.arxiv.org/api/query",
        "https://arxiv.org/api/query",
    )
    MIN_REQUEST_INTERVAL = 3.0
    REQUEST_TIMEOUT = 30.0
    MAX_RESULTS = 50

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def __init__(
        self,
        *,
        http_client: HttpClient | None = None,
        api_url: str | None = None,
    ) -> None:
        self.http = http_client or HttpClient(timeout=self.REQUEST_TIMEOUT)
        self.api_urls = (api_url,) if api_url else self.API_URLS
        self._last_request_time = 0.0

    def search(self, query: SearchQuery) -> list[SearchResult]:
        response = self.search_detailed(query)
        if not response.success:
            assert response.error is not None
            raise RuntimeError(response.error.message)
        return response.results

    def search_detailed(self, query: SearchQuery) -> SearchResponse:
        started = time.monotonic()
        try:
            self._validate_query(query)
            search_query = self._build_search_query(
                query.query.strip(), query.categories
            )
            self._wait_for_rate_limit()

            last_error: HttpRequestError | None = None
            for endpoint in self.api_urls:
                url = self._build_url(search_query, query, endpoint)
                debug.log("ArxivSearchProvider", f"SEARCH QUERY → {search_query}")
                debug.log("ArxivSearchProvider", f"REQUEST URL → {url}")

                try:
                    response = self.http.get(url, headers=self._headers())
                except HttpRequestError as exc:
                    last_error = exc
                    if exc.status_code == 406 and endpoint != self.api_urls[-1]:
                        debug.log(
                            "ArxivSearchProvider",
                            f"ENDPOINT FALLBACK → {endpoint} → next endpoint",
                        )
                        continue
                    raise

                results = self._parse_atom(response.body)
                return SearchResponse(
                    query=query,
                    provider=self.name,
                    results=results,
                    success=True,
                    elapsed_seconds=time.monotonic() - started,
                    attempts=response.attempts,
                    metadata={
                        "status_code": response.status,
                        "url": url,
                        "endpoint": endpoint,
                    },
                )

            if last_error is not None:
                raise last_error
            raise RuntimeError("没有可用的 arXiv API endpoint。")

        except ValueError as exc:
            return self._failure(query, started, "validation", str(exc))
        except HttpRequestError as exc:
            return self._failure(
                query,
                started,
                "http",
                str(exc),
                status_code=exc.status_code,
                reason=exc.reason,
                response_body=exc.response_body,
                retryable=exc.retryable,
                attempts=exc.attempts,
            )
        except RuntimeError as exc:
            return self._failure(query, started, "parse", str(exc))

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": "StudyAgent/2.0 (educational research client; arXiv API search)",
            "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.1",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    @classmethod
    def _validate_query(cls, query: SearchQuery) -> None:
        if not query.query.strip():
            raise ValueError("arXiv query 不能为空。")
        if query.max_results < 1:
            raise ValueError("arXiv max_results 必须大于 0。")

    @classmethod
    def _build_url(
        cls,
        search_query: str,
        query: SearchQuery,
        endpoint: str | None = None,
    ) -> str:
        params = {
            "search_query": search_query,
            "start": "0",
            "max_results": str(max(1, min(query.max_results, cls.MAX_RESULTS))),
            "sortBy": query.sort_by or "relevance",
            "sortOrder": query.sort_order or "descending",
        }
        return f"{endpoint or cls.API_URL}?{urllib.parse.urlencode(params)}"

    @staticmethod
    def _build_search_query(text: str, categories: list[str]) -> str:
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

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        delay = self.MIN_REQUEST_INTERVAL - elapsed
        if delay > 0:
            time.sleep(delay)
        self._last_request_time = time.monotonic()

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
                    raw=entry,
                )
            )

        debug.log("ArxivSearchProvider", f"RESULTS → {len(results)}")
        return results

    def _failure(
        self,
        query: SearchQuery,
        started: float,
        stage: str,
        message: str,
        **kwargs,
    ) -> SearchResponse:
        error = SearchError(
            provider=self.name,
            stage=stage,
            message=message if stage == "validation" else f"arXiv API 请求失败：{message}",
            **kwargs,
        )
        debug.log("ArxivSearchProvider", f"FAILED → {error.message}")
        return SearchResponse(
            query=query,
            provider=self.name,
            success=False,
            elapsed_seconds=time.monotonic() - started,
            attempts=error.attempts,
            error=error,
        )

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
