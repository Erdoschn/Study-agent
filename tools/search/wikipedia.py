from __future__ import annotations

import json
import time
import urllib.parse

from core.__debug__ import debug
from .base import SearchProvider
from .http import HttpClient, HttpRequestError, SearchTimeoutError
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
            if response.error.stage == "timeout":
                raise SearchTimeoutError(response.error.message, attempts=response.error.attempts)
            raise RuntimeError(response.error.message)
        return response.results

    def search_detailed(self, query: SearchQuery) -> SearchResponse:
        started = time.monotonic()
        self._attempts = 0
        try:
            self._validate_query(query)
            title_results = self._search_titles(query)

            summaries = self._get_summaries(
                [title for title, _ in title_results]
            )
            results = [
                self._to_result(
                    title,
                    url,
                    summaries.get(title, {}),
                )
                for title, url in title_results
            ]

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
                    "search_engine": "mediawiki_fulltext",
                    "summary_fetch": "batch",
                },
            )

        except ValueError as exc:
            return self._failure(query, started, "validation", str(exc), attempts=0)
        except SearchTimeoutError as exc:
            return self._failure(query, started, "timeout", str(exc), reason="timeout", retryable=True, attempts=max(1, self._attempts))
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
                attempts=max(1, self._attempts),
            )
        except RuntimeError as exc:
            return self._failure(query, started, "parse", str(exc), attempts=max(1, self._attempts))

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
        try:
            response = self.http.get(url, headers=self._headers())
        except SearchTimeoutError as exc:
            self._attempts += exc.attempts
            raise
        except HttpRequestError as exc:
            self._attempts += exc.attempts
            raise

        self._attempts += response.attempts

        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Wikipedia JSON 解析失败：{exc}") from exc

    def _search_titles(self, query: SearchQuery) -> list[tuple[str, str]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query.query.strip(),
            "srnamespace": 0,
            "srlimit": max(1, min(query.max_results, self.MAX_RESULTS)),
            "srprop": "snippet|timestamp",
            "format": "json",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        data = self._request_json(url)

        if not isinstance(data, dict):
            raise RuntimeError("Wikipedia search 返回格式异常。")
        query_data = data.get("query", {})
        if not isinstance(query_data, dict):
            raise RuntimeError("Wikipedia search 缺少 query。")
        search_results = query_data.get("search", [])
        if not isinstance(search_results, list):
            raise RuntimeError("Wikipedia search 结果格式异常。")

        results: list[tuple[str, str]] = []
        for item in search_results:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            title = str(item["title"])
            url = f"https://{self.language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'), safe='()/,:;-')}"
            results.append((title, url))
        return results

    def _get_summaries(self, titles: list[str]) -> dict[str, dict]:
        if not titles:
            return {}

        params = {
            "action": "query",
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "redirects": 1,
            "titles": "|".join(titles),
            "format": "json",
        }
        url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
        data = self._request_json(url)

        if not isinstance(data, dict):
            raise RuntimeError("Wikipedia batch summary 返回格式异常。")

        pages = (
            data.get("query", {}).get("pages", {})
            if isinstance(data.get("query"), dict)
            else {}
        )
        if not isinstance(pages, dict):
            raise RuntimeError("Wikipedia batch summary 缺少 pages。")

        summaries: dict[str, dict] = {}
        for page in pages.values():
            if not isinstance(page, dict):
                continue
            title = str(page.get("title", ""))
            if not title or "missing" in page:
                continue
            summaries[title] = page

        return summaries

    @staticmethod
    def _headers() -> dict[str, str]:
        return {
            "User-Agent": "StudyAgent/2.0 (educational research client)",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

    def _to_result(self, title: str, url: str, summary: dict) -> SearchResult:
        page_url = str(summary.get("fullurl", url))
        return SearchResult(
            source=self.name,
            source_type="encyclopedia",
            title=str(summary.get("title", title)),
            url=page_url,
            abstract=str(summary.get("extract", "")),
            authors=[],
            published="",
            updated=str(summary.get("timestamp", "")),
            identifier=str(summary.get("pageid", "")),
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
