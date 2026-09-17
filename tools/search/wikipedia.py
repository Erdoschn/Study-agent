import json
import time
import urllib.parse
import urllib.request

from .base import SearchProvider
from .models import SearchQuery, SearchResult


class WikipediaSearchProvider(SearchProvider):
    """
    Wikipedia 搜索 Provider。

    第一阶段：
    1. 使用 MediaWiki OpenSearch 搜索条目
    2. 再读取每个条目的 summary
    3. 转换成统一的 SearchResult

    注意：
    - 本模块只负责搜索和获取资料
    - 不负责判断资料是否正确
    - 不负责判断资料能否证明某个 Claim
    """

    name = "wikipedia"

    API_URL = "https://en.wikipedia.org/w/api.php"
    SUMMARY_URL = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/"
    )

    MIN_REQUEST_INTERVAL = 0.2

    def __init__(self, language: str = "en"):
        self.language = language
        self._last_request_time = 0.0

        if language == "en":
            self.API_URL = "https://en.wikipedia.org/w/api.php"
            self.SUMMARY_URL = (
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
            )
        else:
            self.API_URL = f"https://{language}.wikipedia.org/w/api.php"
            self.SUMMARY_URL = (
                f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
            )

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_time

        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

    def _request_json(self, url: str) -> dict | list:
        self._wait()

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
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()

            self._last_request_time = time.monotonic()
            return json.loads(data.decode("utf-8"))

        except Exception as exc:
            raise RuntimeError(
                f"Wikipedia API 请求失败："
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
            "limit": max(1, min(query.max_results, 20)),
            "format": "json",
        }

        url = self.API_URL + "?" + urllib.parse.urlencode(params)
        data = self._request_json(url)

        if not isinstance(data, list) or len(data) < 4:
            raise RuntimeError("Wikipedia OpenSearch 返回格式异常。")

        titles = data[1]
        urls = data[3]

        return list(zip(titles, urls))

    def _get_summary(self, title: str) -> dict:
        encoded_title = urllib.parse.quote(title, safe="")
        url = self.SUMMARY_URL + encoded_title

        return self._request_json(url)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        if not query.query.strip():
            raise ValueError("Wikipedia query 不能为空。")

        title_results = self._search_titles(query)
        results: list[SearchResult] = []

        for title, url in title_results:
            try:
                summary = self._get_summary(title)
            except RuntimeError:
                summary = {}

            results.append(
                SearchResult(
                    source=self.name,
                    source_type="encyclopedia",
                    title=summary.get("title", title),
                    url=summary.get("content_urls", {})
                    .get("desktop", {})
                    .get("page", url),
                    abstract=summary.get(
                        "extract",
                        "",
                    ),
                    authors=[],
                    published="",
                    updated=summary.get(
                        "timestamp",
                        "",
                    ),
                    identifier=str(
                        summary.get("wikibase_item", "")
                    ),
                    raw=summary,
                )
            )

        return results