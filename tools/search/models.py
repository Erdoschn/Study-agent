from __future__ import annotations

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
    source_preferences: list[str] = field(default_factory=list)


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


@dataclass
class SearchError:
    """Structured search failure information."""

    provider: str
    stage: str
    message: str
    status_code: int | None = None
    reason: str = ""
    response_body: str = ""
    retryable: bool = False
    attempts: int = 1


@dataclass
class SearchResponse:
    """Unified result returned by the modern router API."""

    query: SearchQuery
    provider: str
    results: list[SearchResult] = field(default_factory=list)
    success: bool = True
    elapsed_seconds: float = 0.0
    attempts: int = 1
    error: SearchError | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
