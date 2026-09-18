from .base import SearchProvider
from .models import SearchQuery, SearchResult
from core.__debug__ import debug


class SearchRouter:
    """
    搜索源路由器。

    source="arxiv" / "wikipedia"
        → 明确指定搜索源

    source="auto"
        → 按 source_preferences 顺序尝试已注册搜索源。
          某个来源返回空结果时，继续尝试下一个来源。
    """

    AUTO_SOURCES = ("wikipedia", "arxiv")

    def __init__(self):
        self.providers: dict[str, SearchProvider] = {}

    def register(self, provider: SearchProvider) -> None:
        self.providers[provider.name] = provider

    def available_sources(self) -> list[str]:
        return sorted(self.providers)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        source = query.source.strip().lower()
        debug.log(
            "SearchRouter",
            f"ROUTE → source={source}, query={query.query}",
        )

        if source == "auto":
            return self._search_auto(query)

        if not source:
            raise ValueError("SearchQuery.source 不能为空。")

        provider = self.providers.get(source)
        if provider is None:
            available = ", ".join(self.available_sources())
            raise ValueError(
                f"未知搜索源：{source}。可用搜索源：{available}"
            )

        debug.log("SearchRouter", f"PROVIDER → {provider.name}")
        return provider.search(query)

    def _search_auto(self, query: SearchQuery) -> list[SearchResult]:
        preferences = query.source_preferences or list(self.AUTO_SOURCES)
        ordered = []
        for source in preferences:
            normalized = str(source).strip().lower()
            if normalized and normalized not in ordered:
                ordered.append(normalized)

        if not ordered:
            raise ValueError("auto 搜索没有可用的 source_preferences。")

        attempted: list[str] = []
        for source in ordered:
            provider = self.providers.get(source)
            if provider is None:
                debug.log(
                    "SearchRouter",
                    f"AUTO SKIP → source={source} 未注册",
                )
                continue

            attempted.append(source)
            debug.log(
                "SearchRouter",
                f"AUTO PROVIDER → {source}",
            )
            results = provider.search(
                SearchQuery(
                    query=query.query,
                    source=source,
                    categories=list(query.categories),
                    max_results=query.max_results,
                    sort_by=query.sort_by,
                    sort_order=query.sort_order,
                )
            )
            if results:
                return results

            debug.log(
                "SearchRouter",
                f"AUTO EMPTY → {source}; try next source",
            )

        if not attempted:
            available = ", ".join(self.available_sources())
            raise ValueError(
                f"auto 搜索没有可用 provider。可用搜索源：{available}"
            )

        return []
