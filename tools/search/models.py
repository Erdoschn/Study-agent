from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchQuery:
    query: str
    source: str = "arxiv"
    categories: list[str] = field(default_factory=list)
    max_results: int = 10
    sort_by: str = "relevance"
    sort_order: str = "descending"


@dataclass
class SearchResult:
    source: str
    title: str
    url: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    identifier: str = ""
    raw: Any = None
    relevance_score: float | None = None
    source_type: str = "unknown"
    notes: list[str] = field(default_factory=list)