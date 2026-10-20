"""Composition root for building the legacy-compatible Study Agent graph.

Concrete implementations are wired here only. CLI, HTTP and MCP entry points
should reuse this factory rather than constructing competing dependency graphs.
"""
from pathlib import Path
from typing import Any

from core import (
    AgentReasoner,
    ModelClientFactory,
    ModelRegistry,
    ModelRouter,
    StudyAgent,
    Teacher,
    ToolExecutor,
)
from tools.search import ArxivSearchProvider, SearchRouter, WikipediaSearchProvider

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_GRAPH_PATH = PROJECT_ROOT / "data" / "knowledge_graph.sqlite3"


def build_search_router() -> SearchRouter:
    """Create the configured search router used by the default composition."""
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    return router


def build_agent(
    config: dict[str, Any],
    *,
    allow_paid: bool = False,
    knowledge_graph_path: str | Path | None = None,
) -> StudyAgent:
    """Build one Study Agent dependency graph from validated runtime settings."""
    registry = ModelRegistry(config)
    model_router = ModelRouter(registry)
    model_factory = ModelClientFactory(config)

    reasoner = AgentReasoner(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=allow_paid,
    )
    teacher = Teacher(
        model_router=model_router,
        model_factory=model_factory,
        allow_paid=allow_paid,
    )
    executor = ToolExecutor(search_router=build_search_router())

    return StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=executor,
        max_steps=None,
        knowledge_graph_path=knowledge_graph_path or KNOWLEDGE_GRAPH_PATH,
    )
