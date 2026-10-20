"""Composition root for building the legacy-compatible Study Agent graph.

Concrete implementations are wired here only. CLI, HTTP and MCP entry points
should reuse this factory rather than constructing competing dependency graphs.
"""
from __future__ import annotations

from collections.abc import Callable
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

RegistryConfigurator = Callable[[ModelRegistry], None]
BeforeAgentHook = Callable[
    [dict[str, Any], ModelRegistry, ModelRouter, ModelClientFactory],
    None,
]


def build_search_router() -> SearchRouter:
    """Create the default search router used by standard agent compositions."""
    router = SearchRouter()
    router.register(ArxivSearchProvider())
    router.register(WikipediaSearchProvider())
    return router


def build_agent(
    config: dict[str, Any],
    *,
    allow_paid: bool = False,
    knowledge_graph_path: str | Path | None = None,
    configure_registry: RegistryConfigurator | None = None,
    before_agent: BeforeAgentHook | None = None,
) -> StudyAgent:
    """Build one shared dependency graph with narrowly scoped extension hooks.

    configure_registry is for deliberate runtime policy differences (for
    example, a browser endpoint with one model candidate). before_agent is
    for setup that must happen after the model stack exists but before serving
    requests (for example, browser prewarming). Hooks may raise; initialization
    must then fail visibly rather than return a partially configured agent.
    """
    registry = ModelRegistry(config)
    if configure_registry is not None:
        configure_registry(registry)

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
    if before_agent is not None:
        before_agent(config, registry, model_router, model_factory)

    executor = ToolExecutor(search_router=build_search_router())
    return StudyAgent(
        reasoner=reasoner,
        teacher=teacher,
        tool_executor=executor,
        max_steps=None,
        knowledge_graph_path=knowledge_graph_path or KNOWLEDGE_GRAPH_PATH,
    )
