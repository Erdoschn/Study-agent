import json
import time
import urllib.parse
import urllib.request
from core.__debug__ import debug
from .base import SearchProvider
from .models import SearchQuery, SearchResult


class WikipediaSearchProvider(SearchProvider):
    """
    Wikipedia 搜索 Provider。

    流程：
        OpenSearch
            ↓
        获取候选条目
            ↓
        Summary API
            ↓
        转换成统一 SearchResult
    """

    name = "wikipedia"

    MIN_REQUEST_INTERVAL = 0.2

    def __init__(
        self,
        language: str = "en",
    ):
        self.language = language
        self._last_request_time = 0.0

        if language == "en":
            self.API_URL = (
                "https://en.wikipedia.org/w/api.php"
            )
            self.SUMMARY_URL = (
                "https://en.wikipedia.org/"
                "api/rest_v1/page/summary/"
            )
        else:
            self.API_URL = (
                f"https://{language}.wikipedia.org/"
                "w/api.php"
            )
            self.SUMMARY_URL = (
                f"https://{language}.wikipedia.org/"
                "api/rest_v1/page/summary/"
            )

    def _wait(self) -> None:
        elapsed = (
            time.monotonic()
            - self._last_request_time
        )

        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(
                self.MIN_REQUEST_INTERVAL
                - elapsed
            )

    def _request_json(
        self,
        url: str,
    ) -> dict | list:

        self._wait()

        debug.log(
            "WikipediaSearchProvider",
            f"API CALL → {url}",
        )

        request = urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "User-Agent": (
                    "StudyAgent/2.0 "
                    "(educational research client)"
                ),
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=30,
            ) as response:

                data = response.read()

            self._last_request_time = (
                time.monotonic()
            )

            return json.loads(
                data.decode("utf-8")
            )

        except Exception as exc:

            debug.log(
                "WikipediaSearchProvider",
                (
                    "API ERROR → "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            )

            raise RuntimeError(
                "Wikipedia API 请求失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _search_titles(
        self,
        query: SearchQuery,
    ) -> list[tuple[str, str]]:

        params = {
            "action": "opensearch",
            "namespace": 0,
            "search": query.query.strip(),
            "limit": max(
                1,
                min(
                    query.max_results,
                    20,
                ),
            ),
            "format": "json",
        }

        url = (
            self.API_URL
            + "?"
            + urllib.parse.urlencode(params)
        )

        data = self._request_json(url)

        if (
            not isinstance(data, list)
            or len(data) < 4
        ):
            raise RuntimeError(
                "Wikipedia OpenSearch "
                "返回格式异常。"
            )

        titles = data[1]
        urls = data[3]

        if not isinstance(
            titles,
            list,
        ):
            titles = []

        if not isinstance(
            urls,
            list,
        ):
            urls = []

        results = []

        for title, url in zip(
            titles,
            urls,
        ):
            results.append(
                (
                    str(title),
                    str(url),
                )
            )

        return results

    def _get_summary(
        self,
        title: str,
    ) -> dict:

        encoded_title = urllib.parse.quote(
            title,
            safe="",
        )

        url = (
            self.SUMMARY_URL
            + encoded_title
        )

        return self._request_json(url)

    def search(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:

        debug.log(
            "WikipediaSearchProvider",
            (
                "SEARCH → "
                f"query={query.query}, "
                f"max_results={query.max_results}, "
                f"language={self.language}"
            ),
        )

        if not query.query.strip():
            raise ValueError(
                "Wikipedia query 不能为空。"
            )

        title_results = self._search_titles(
            query
        )

        debug.log(
            "WikipediaSearchProvider",
            (
                "OPENSEARCH RESULTS → "
                f"{len(title_results)}"
            ),
        )

        results: list[SearchResult] = []

        for title, url in title_results:

            try:
                summary = self._get_summary(
                    title
                )

            except RuntimeError as exc:

                debug.log(
                    "WikipediaSearchProvider",
                    (
                        "SUMMARY FAILED → "
                        f"{title}: {exc}"
                    ),
                )

                summary = {}

            content_urls = summary.get(
                "content_urls",
                {},
            )

            desktop = (
                content_urls.get(
                    "desktop",
                    {},
                )
                if isinstance(
                    content_urls,
                    dict,
                )
                else {}
            )

            page_url = (
                desktop.get(
                    "page",
                    url,
                )
                if isinstance(
                    desktop,
                    dict,
                )
                else url
            )

            result = SearchResult(
                source=self.name,
                source_type="encyclopedia",
                title=str(
                    summary.get(
                        "title",
                        title,
                    )
                ),
                url=str(page_url),
                abstract=str(
                    summary.get(
                        "extract",
                        "",
                    )
                ),
                authors=[],
                published="",
                updated=str(
                    summary.get(
                        "timestamp",
                        "",
                    )
                ),
                identifier=str(
                    summary.get(
                        "wikibase_item",
                        "",
                    )
                ),
                raw=summary,
            )

            results.append(result)

        debug.log(
            "WikipediaSearchProvider",
            (
                "RESULTS → "
                f"{len(results)}"
            ),
        )

        return results