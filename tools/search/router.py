from __future__ import annotations

from .base import SearchProvider
from .models import SearchQuery, SearchResult


class SearchRouter:
    """
    搜索源路由器。

    当前只注册 arXiv。

    以后可以：

        router.register(
            WebSearchProvider()
        )

        router.register(
            OfficialDocsProvider()
        )

    最终 Agent 只需要：

        router.search(query)

    不需要知道具体搜索网站。
    """

    def __init__(self):
        # provider.name -> provider object
        self.providers: dict[str, SearchProvider] = {}

    def register(self, provider: SearchProvider):
        """
        注册一个搜索 Provider。
        """

        self.providers[provider.name] = provider

    def available_sources(self) -> list[str]:
        """
        返回当前可用的搜索源。
        """

        return sorted(
            self.providers.keys()
        )

    def search(
        self,
        query: SearchQuery,
    ) -> list[SearchResult]:
        """
        根据 SearchQuery 选择搜索 Provider。
        """

        # ------------------------------------------------------
        # 如果明确指定 source：
        # 只使用指定搜索源。
        # ------------------------------------------------------
        if query.source:
            provider = self.providers.get(
                query.source
            )

            if provider is None:
                raise ValueError(
                    f"未注册搜索源：{query.source}"
                )

            return provider.search(query)

        # ------------------------------------------------------
        # 当前没有指定 source：
        # 第一版默认使用 arXiv。
        #
        # 后面这里会真正变成：
        #
        # Task
        # ↓
        # Search Policy
        # ↓
        # Web / arXiv / Official / ...
        # ------------------------------------------------------
        provider = self.providers.get("arxiv")

        if provider is None:
            raise RuntimeError(
                "当前没有可用的默认搜索源。"
            )

        return provider.search(query)