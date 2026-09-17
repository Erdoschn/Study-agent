from __future__ import annotations

import json
import time
import urllib.parse

from core.__debug__ import debug
from .base import SearchProvider
from .http import HttpClient, HttpRequestError
from .models import SearchError, SearchQuery, SearchResponse, SearchResult


class WikipediaSearchProvider(SearchProvider):
    """Wikipedia search provider using the shared HTTP transport."""

    name = "wikipedia"
    MIN_REQUEST_INTERVAL = 0.2
    REQUEST_TIMEOUT = 30.0
    MAX_RESULTS = 20

    def __init__(
        self,
        language: str = "en",
        *,
        http_client: HttpClient | None = None,
    ):
        self.language = language
        self.http = http_client or HttpClient(timeout=self.REQUEST_TIMEOUT)
        self._last_request_time = 0.0

        self.API_URL = f"https://{language}.wikipedia.org/w/api.php"
        self.SUMMARY_URL = (
            f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
        )

    def search(self, query: SearchQuery) -> list[SearchResult]:
        response = self.search_detailed(query)
        if not response.success:
            assert response.error is not None
            if response.error.stage == "validation":
                raise ValueError(response.error.message)
            raise RuntimeError(response.error.message)
        return response.results

    def search_detailed(self, query: SearchQuery) -> SearchResponse:
        started = time.monotonic()
        try:
            self._validate_query(query)
            title_results = self._search_titles(query)

            results: list[SearchResult] = []
            for title, url in title_results:
                try:
                    summary = self._get_summary(title)
                except HttpRequestError as exc:
                    debug.log(
                        "WikipediaSearchProvider",
                        f"SUMMARY FAILED → {title}: {exc}",
                    )
                    summary = {}

                results.append(self._to_result(title, url, summary))

            debug.log("WikipediaSearchProvider", f"RESULTS → {len(results)}")
            return SearchResponse(
                query=query,
                provider=self.name,
                results=results,
                success=True,
                elapsed_seconds=time.monotonic() - started,
                attempts=self._attempts,
                metadata={
                    "language": self.language,
                    "title_count": len(title_results),
                },
            )

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

    @classmethod
    def _validate_query(cls, query: SearchQuery) -> None:
        if not query.query.strip():
            raise ValueError("Wikipedia query 不能为空。")
        if query.max_results < 1:
            raise ValueError("Wikipedia max_results 必须大于 0。")

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        delay = self.MIN_REQUEST_INTERVAL - elapsed
        if delay > 0:
            time.sleep(delay)
        self._last_request_time = time.monotonic()

    def _request_json(self, url: str) -> dict | list:
        self._wait()
        debug.log("WikipediaSearchProvider", f"API CALL → {url}")
        response = self.http.get(url, headers=self._headers())
        self._attempts = response.attempts

        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Wikipedia JSON 解析失败：{exc}") from exc

    def _search_titles(self, query: SearchQuery) -> list[tuple[str, str]]:
        params = {
            "action": "opensearch",
            "namespace": 0,
            "search": query.query.strip(),
            "limit": max(1, min(query.max_results, self.MAX_RESULTS)),
            "format": "json",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        data = self._request_json(url)

        if not isinstance(data, list) or len(data) < 4:
            raise RuntimeError("Wikipedia OpenSearch 返回格式异常。")

        titles = data[1] if isinstance(data[1], list) else []
        urls = data[3] if isinstance(data[3], list) else []
        return [(str(title), str(url)) for title, url in zip(titles, urls)]

    def _get_summary(self, title: str) -> dict:
        encoded_title = urllib.parse.quote(title, safe="")
        return self._request_json(self.SUMMARY_URL + encoded_title)

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": "StudyAgent/2.0 (educational research client)",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _to_result(self, title: str, url: str, summary: dict) -> SearchResult:
        content_urls = summary.get("content_urls", {})
        desktop = content_urls.get("desktop", {}) if isinstance(content_urls, dict) else {}
        page_url = desktop.get("page", url) if isinstance(desktop, dict) else url

        return SearchResult(
            source=self.name,
            source_type="encyclopedia",
            title=str(summary.get("title", title)),
            url=str(page_url),
            abstract=str(summary.get("extract", "")),
            authors=[],
            published="",
            updated=str(summary.get("timestamp", "")),
            identifier=str(summary.get("wikibase_item", "")),
            raw=summary,
        )

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
            message=message if stage == "validation" else f"Wikipedia API 请求失败：{message}",
            **kwargs,
        )
        debug.log("WikipediaSearchProvider", f"FAILED → {error.message}")
        return SearchResponse(
            query=query,
            provider=self.name,
            success=False,
            elapsed_seconds=time.monotonic() - started,
            attempts=error.attempts,
            error=error,
        )

    @property
    def _attempts(self) -> int:
        return getattr(self, "__attempts", 1)

    @_attempts.setter
    def _attempts(self, value: int) -> None:
        self.__attempts = value
