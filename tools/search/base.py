from __future__ import annotations

from abc import ABC, abstractmethod

from .models import SearchQuery, SearchResult


class SearchProvider(ABC):
    """
    所有搜索后端必须遵守的统一接口。

    例如以后可以有：

        ArxivSearchProvider
        WebSearchProvider
        OfficialDocsProvider
        SemanticScholarProvider

    SearchRouter 不需要知道它们内部怎么实现，
    只需要调用：

        provider.search(query)
    """

    name = "base"

    @abstractmethod
    def search(self, query: SearchQuery) -> list[SearchResult]:
        """执行搜索并返回统一 SearchResult 列表。"""
        raise NotImplementedError