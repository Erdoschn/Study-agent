from __future__ import annotations
from core.__debug__ import debug
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import SearchProvider
from .models import SearchQuery, SearchResult


class ArxivSearchProvider(SearchProvider):
    """
    arXiv Public API 搜索适配器。

    当前职责只有：

        搜索论文
        ↓
        获取 metadata / abstract
        ↓
        转换为 SearchResult

    当前不负责：

        判断论文是否正确
        判断论文是否支持某个 Claim
        判断论文质量
        自动读取全文

    这些应该属于后面的 Evidence / Verification 层。
    """

    name = "arxiv"

    API_URL = "https://export.arxiv.org/api/query"

    # 公共 API 请求间隔保护。
    MIN_REQUEST_INTERVAL = 3.0

    _last_request_time = 0.0

    # Atom XML 命名空间。
    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """
        执行一次 arXiv 搜索。
        """
        debug.log(
            "ArxivSearchProvider",
            f"SEARCH → {query.query}",
        )

        if not query.query.strip():
            raise ValueError("arXiv query 不能为空。")

        # 限制单次最大结果数。
        max_results = max(
            1,
            min(query.max_results, 50),
        )

        search_expression = query.query.strip()

        # ------------------------------------------------------
        # 如果指定 arXiv 分类，则加入分类条件。
        # ------------------------------------------------------
        categories = [
            category.strip()
            for category in query.categories
            if category.strip()
        ]

        if categories:
            category_expression = " OR ".join(
                f"cat:{category}"
                for category in categories
            )

            search_expression = (
                f"({search_expression}) "
                f"AND ({category_expression})"
            )

        params = {
            "search_query": search_expression,
            "start": 0,
            "max_results": max_results,
            "sortBy": query.sort_by,
            "sortOrder": query.sort_order,
        }

        url = (
            self.API_URL
            + "?"
            + urllib.parse.urlencode(params)
        )

        # ------------------------------------------------------
        # API 请求间隔控制。
        # ------------------------------------------------------
        now = time.monotonic()
        elapsed = now - self._last_request_time

        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(
                self.MIN_REQUEST_INTERVAL - elapsed
            )

        req = urllib.request.Request(
            url=url,
            method="GET",
            headers={
                "User-Agent": (
                    "StudyAgent/2.0 "
                    "(educational research client)"
                )
            },
        )
        debug.log(
        "ArxivSearchProvider",
        "API CALL",
        )
        try:
            with urllib.request.urlopen(
                req,
                timeout=30,
            ) as response:
                xml_data = response.read()

            self._last_request_time = time.monotonic()

        except Exception as exc:
            raise RuntimeError(
                f"arXiv API 请求失败："
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return self._parse_atom(xml_data)

    def _parse_atom(
        self,
        xml_data: bytes,
    ) -> list[SearchResult]:
        """
        把 arXiv Atom XML 转换成统一 SearchResult。
        """

        try:
            root = ET.fromstring(xml_data)

        except ET.ParseError as exc:
            raise RuntimeError(
                f"arXiv XML 解析失败：{exc}"
            ) from exc

        results: list[SearchResult] = []

        for entry in root.findall(
            "atom:entry",
            self.NS,
        ):
            identifier_url = self._text(
                entry.find(
                    "atom:id",
                    self.NS,
                )
            )

            title = self._normalize(
                self._text(
                    entry.find(
                        "atom:title",
                        self.NS,
                    )
                )
            )

            abstract = self._normalize(
                self._text(
                    entry.find(
                        "atom:summary",
                        self.NS,
                    )
                )
            )

            published = self._text(
                entry.find(
                    "atom:published",
                    self.NS,
                )
            )

            updated = self._text(
                entry.find(
                    "atom:updated",
                    self.NS,
                )
            )

            # 尝试寻找 HTML 页面。
            html_url = self._find_html_url(entry)

            # 没找到时退回 arXiv ID URL。
            if not html_url:
                html_url = identifier_url

            # --------------------------------------------------
            # 作者
            # --------------------------------------------------
            authors: list[str] = []

            for author in entry.findall(
                "atom:author",
                self.NS,
            ):
                name = self._text(
                    author.find(
                        "atom:name",
                        self.NS,
                    )
                )

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
                    identifier=self._extract_id(
                        identifier_url
                    ),
                )
            )
        debug.log(
            "ArxivSearchProvider",
            f"RESULTS → {len(results)}",
        )
        return results

    def _find_html_url(self, entry) -> str:
        """寻找论文 HTML 页面链接。"""

        for link in entry.findall(
            "atom:link",
            self.NS,
        ):
            href = link.attrib.get("href", "")
            link_type = link.attrib.get("type", "")

            if href and link_type == "text/html":
                return href

        return ""

    @staticmethod
    def _text(node) -> str:
        """获取 XML 节点文本。"""

        if node is None:
            return ""

        return "".join(
            node.itertext()
        ).strip()

    @staticmethod
    def _normalize(text: str) -> str:
        """清理换行和多余空格。"""
        return " ".join(text.split())

    @staticmethod
    def _extract_id(identifier: str) -> str:
        """从 arXiv URL 中提取论文 ID。"""

        marker = "/abs/"

        if marker in identifier:
            return identifier.rsplit(
                marker,
                1,
            )[1]

        return identifier