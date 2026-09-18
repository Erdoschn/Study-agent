from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import arxiv as arxiv_api
import requests

from core.__debug__ import debug
from .base import SearchProvider
from .models import SearchError, SearchQuery, SearchResponse, SearchResult


class ArxivSearchProvider(SearchProvider):
    """arXiv provider backed by the maintained arxiv Python client.

    The official client uses requests for HTTP and lxml for Atom parsing.
    Endpoint selection remains in this provider so the Study Agent can
    survive a single endpoint or network path failing.
    """

    name = "arxiv"

    API_URL = "https://export.arxiv.org/api/query"
    API_URLS = (
        "https://export.arxiv.org/api/query",
        "https://arxiv.org/api/query",
    )
    FALLBACK_API_URLS = (
        "http://export.arxiv.org/api/query",
        "http://arxiv.org/api/query",
    )

    # arXiv documents the HTTP export endpoint; it is first so a broken
    # HTTPS path does not prevent the provider from working.
    TRANSPORT_ENDPOINTS = (
        "http://export.arxiv.org/api/query",
        "https://export.arxiv.org/api/query",
        "http://arxiv.org/api/query",
        "https://arxiv.org/api/query",
    )

    MIN_REQUEST_INTERVAL = 3.0
    REQUEST_TIMEOUT = 30.0
    MAX_RESULTS = 50

    FALLBACK_HTTP_STATUSES = {403, 406, 408, 429, 500, 502, 503, 504}

    _SORT_BY = {
        "relevance": arxiv_api.SortCriterion.Relevance,
        "lastupdateddate": arxiv_api.SortCriterion.LastUpdatedDate,
        "submitteddate": arxiv_api.SortCriterion.SubmittedDate,
    }
    _SORT_ORDER = {
        "ascending": arxiv_api.SortOrder.Ascending,
        "descending": arxiv_api.SortOrder.Descending,
    }

    def __init__(
        self,
        *,
        client_factory: Callable[[str, int], arxiv_api.Client] | None = None,
        search_factory: Callable[..., arxiv_api.Search] | None = None,
    ) -> None:
        self._client_factory = client_factory or self._make_client
        self._search_factory = search_factory or arxiv_api.Search
        self._last_request_time = 0.0

    @property
    def endpoints(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.TRANSPORT_ENDPOINTS))

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
                query.query.strip(),
                query.categories,
            )
            api_search = self._build_api_search(search_query, query)

            last_error: Exception | None = None
            attempts = 0

            for index, endpoint in enumerate(self.endpoints):
                self._wait_for_rate_limit()
                debug.log(
                    "ArxivSearchProvider",
                    f"SEARCH → endpoint={endpoint} query={search_query}",
                )

                client = self._client_factory(
                    endpoint,
                    max(1, min(query.max_results, self.MAX_RESULTS)),
                )

                try:
                    api_results = list(client.results(api_search))
                    results = [self._to_search_result(item) for item in api_results]

                    debug.log(
                        "ArxivSearchProvider",
                        f"RESULTS → {len(results)} via {endpoint}",
                    )

                    return SearchResponse(
                        query=query,
                        provider=self.name,
                        results=results,
                        success=True,
                        elapsed_seconds=time.monotonic() - started,
                        attempts=max(1, attempts + 1),
                        metadata={
                            "transport": "arxiv.py + requests",
                            "endpoint": endpoint,
                            "endpoint_index": index,
                            "endpoint_count": len(self.endpoints),
                        },
                    )

                except arxiv_api.HTTPError as exc:
                    attempts += max(1, exc.retry + 1)
                    last_error = exc
                    debug.log(
                        "ArxivSearchProvider",
                        f"HTTP ERROR → {exc.status} on {endpoint}",
                    )
                    if (
                        exc.status not in self.FALLBACK_HTTP_STATUSES
                        or index >= len(self.endpoints) - 1
                    ):
                        raise

                except requests.RequestException as exc:
                    attempts += 1
                    last_error = exc
                    debug.log(
                        "ArxivSearchProvider",
                        f"NETWORK ERROR → {type(exc).__name__}: {exc}",
                    )
                    if index >= len(self.endpoints) - 1:
                        raise

                except arxiv_api.UnexpectedEmptyPageError as exc:
                    attempts += max(1, exc.retry + 1)
                    last_error = exc
                    debug.log(
                        "ArxivSearchProvider",
                        f"EMPTY PAGE → fallback from {endpoint}",
                    )
                    if index >= len(self.endpoints) - 1:
                        raise

                except Exception as exc:
                    last_error = exc
                    debug.log(
                        "ArxivSearchProvider",
                        f"UNEXPECTED ERROR → {type(exc).__name__}: {exc}",
                    )
                    raise RuntimeError(
                        f"arXiv 客户端失败：{type(exc).__name__}: {exc}"
                    ) from exc

                if index < len(self.endpoints) - 1:
                    debug.log(
                        "ArxivSearchProvider",
                        f"FALLBACK → {endpoint} → {self.endpoints[index + 1]}",
                    )

            if last_error is not None:
                raise last_error
            raise RuntimeError("没有可用的 arXiv API endpoint。")

        except ValueError as exc:
            return self._failure(query, started, "validation", str(exc))

        except arxiv_api.HTTPError as exc:
            return self._failure(
                query,
                started,
                "http",
                str(exc),
                status_code=exc.status,
                reason=f"HTTP {exc.status}",
                retryable=exc.status in self.FALLBACK_HTTP_STATUSES,
            )

        except requests.RequestException as exc:
            return self._failure(
                query,
                started,
                "network",
                str(exc),
                reason=type(exc).__name__,
                retryable=True,
            )

        except arxiv_api.UnexpectedEmptyPageError as exc:
            return self._failure(
                query,
                started,
                "api",
                str(exc),
                reason="unexpected_empty_page",
                retryable=True,
            )

        except RuntimeError as exc:
            return self._failure(query, started, "client", str(exc))

    @classmethod
    def _make_client(cls, endpoint: str, max_results: int) -> arxiv_api.Client:
        client = arxiv_api.Client(
            page_size=max_results,
            delay_seconds=0.0,
            num_retries=0,
        )
        client.query_url_format = f"{endpoint}?{{}}"
        return client

    def _build_api_search(
        self,
        search_query: str,
        query: SearchQuery,
    ) -> arxiv_api.Search:
        sort_by = self._SORT_BY.get(
            (query.sort_by or "relevance").strip().lower(),
        )
        if sort_by is None:
            raise ValueError(
                "arXiv sort_by 必须是 relevance、lastUpdatedDate 或 submittedDate。"
            )

        sort_order = self._SORT_ORDER.get(
            (query.sort_order or "descending").strip().lower(),
        )
        if sort_order is None:
            raise ValueError(
                "arXiv sort_order 必须是 ascending 或 descending。"
            )

        return self._search_factory(
            query=search_query,
            max_results=max(1, min(query.max_results, self.MAX_RESULTS)),
            sort_by=sort_by,
            sort_order=sort_order,
        )

    @classmethod
    def _validate_query(cls, query: SearchQuery) -> None:
        if not query.query.strip():
            raise ValueError("arXiv query 不能为空。")
        if query.max_results < 1:
            raise ValueError("arXiv max_results 必须大于 0。")

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

    @staticmethod
    def _to_search_result(item: arxiv_api.Result) -> SearchResult:
        identifier = item.get_short_id()
        return SearchResult(
            source="arxiv",
            source_type="paper",
            title=" ".join(item.title.split()),
            url=item.entry_id or f"https://arxiv.org/abs/{identifier}",
            abstract=" ".join(item.summary.split()),
            authors=[author.name for author in item.authors if author.name],
            published=ArxivSearchProvider._iso_datetime(item.published),
            updated=ArxivSearchProvider._iso_datetime(item.updated),
            identifier=identifier,
            raw=item,
        )

    @staticmethod
    def _iso_datetime(value: datetime | None) -> str:
        if value is None:
            return ""
        return value.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _failure(
        query: SearchQuery,
        started: float,
        stage: str,
        message: str,
        **kwargs,
    ) -> SearchResponse:
        error = SearchError(
            provider="arxiv",
            stage=stage,
            message=(
                message
                if stage == "validation"
                else f"arXiv 搜索失败：{message}"
            ),
            **kwargs,
        )
        debug.log("ArxivSearchProvider", f"FAILED → {error.message}")
        return SearchResponse(
            query=query,
            provider="arxiv",
            success=False,
            elapsed_seconds=time.monotonic() - started,
            attempts=error.attempts,
            error=error,
        )
