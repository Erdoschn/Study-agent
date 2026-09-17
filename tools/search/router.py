from .base import SearchProvider
from .models import SearchQuery, SearchResult


class SearchRouter:
    """
    搜索源路由器。

    source="arxiv"
        → ArxivSearchProvider

    source="wikipedia"
        → WikipediaSearchProvider
    """

    def __init__(self):
        self.providers: dict[str, SearchProvider] = {}

    def register(self, provider: SearchProvider) -> None:
        self.providers[provider.name] = provider

    def available_sources(self) -> list[str]:
        return sorted(self.providers)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        source = query.source.strip().lower()

        if not source:
            raise ValueError("SearchQuery.source 不能为空。")

        provider = self.providers.get(source)

        if provider is None:
            available = ", ".join(self.available_sources())
            raise ValueError(
                f"未知搜索源：{source}。"
                f"可用搜索源：{available}"
            )

        return provider.search(query)