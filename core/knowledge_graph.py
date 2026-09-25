from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KnowledgeNode:
    id: str
    name: str
    node_type: str = "concept"
    aliases: list[str] = field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class KnowledgeEdge:
    source: str
    target: str
    relation: str = "related_to"
    confidence: float = 0.0
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)


class KnowledgeGraph:
    """Small persistent, evidence-backed concept graph for search guidance and learning."""

    MAX_NODES = 500
    MAX_EDGES = 1000

    def __init__(self):
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: dict[tuple[str, str, str], KnowledgeEdge] = {}

    @staticmethod
    def _id(text: str) -> str:
        import re
        value = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", str(text or "").strip().lower()).strip("_")
        return value[:120]

    def add_concept(self, name: str, node_type: str = "concept", alias: str | None = None,
                    evidence: dict[str, Any] | None = None) -> str:
        name = str(name or "").strip()
        node_id = self._id(name)
        if not node_id:
            return ""
        node = self.nodes.get(node_id)
        if node is None:
            if len(self.nodes) >= self.MAX_NODES:
                return node_id
            node = KnowledgeNode(id=node_id, name=name, node_type=node_type)
            self.nodes[node_id] = node
        if alias and alias.strip() and alias.strip() not in node.aliases and alias.strip() != node.name:
            node.aliases.append(alias.strip())
        if evidence and evidence not in node.evidence_refs:
            node.evidence_refs.append(dict(evidence))
            del node.evidence_refs[:-10]
        return node_id

    def add_relation(self, source: str, target: str, relation: str = "related_to",
                     confidence: float = 0.0, evidence: dict[str, Any] | None = None) -> None:
        source_id = self.add_concept(source)
        target_id = self.add_concept(target)
        if not source_id or not target_id or source_id == target_id:
            return
        key = (source_id, target_id, str(relation))
        edge = self.edges.get(key)
        if edge is None:
            if len(self.edges) >= self.MAX_EDGES:
                return
            edge = KnowledgeEdge(source_id, target_id, str(relation), float(confidence))
            self.edges[key] = edge
        else:
            edge.confidence = max(edge.confidence, float(confidence))
        if evidence and evidence not in edge.evidence_refs:
            edge.evidence_refs.append(dict(evidence))
            del edge.evidence_refs[:-10]

    def learn_from_search(self, query: str, results: list[dict[str, Any]]) -> None:
        """Record only explicit search concepts and result titles; never invent a hierarchy."""
        query = str(query or "").strip()
        if not query:
            return
        query_id = self.add_concept(query, alias=query)
        for item in results[:20]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            evidence = {
                "source": item.get("source"),
                "title": title,
                "identifier": item.get("identifier"),
                "url": item.get("url"),
            }
            title_id = self.add_concept(title, node_type="document", evidence=evidence)
            if query_id and title_id:
                self.add_relation(query, title, "supported_by_search", 0.5, evidence)

    def neighbors(self, concept: str, limit: int = 12) -> list[dict[str, Any]]:
        node_id = self._id(concept)
        items = []
        for edge in self.edges.values():
            if edge.source == node_id or edge.target == node_id:
                other_id = edge.target if edge.source == node_id else edge.source
                node = self.nodes.get(other_id)
                if node:
                    items.append({
                        "name": node.name,
                        "relation": edge.relation,
                        "direction": "out" if edge.source == node_id else "in",
                        "confidence": edge.confidence,
                    })
        items.sort(key=lambda x: x["confidence"], reverse=True)
        return items[:limit]

    def context_for(self, query: str, limit: int = 12) -> dict[str, Any]:
        return {
            "query": query,
            "known_concepts": [
                {"name": self.nodes[n].name, "aliases": self.nodes[n].aliases}
                for n in [self._id(query)] if n in self.nodes
            ],
            "neighbors": self.neighbors(query, limit=limit),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }
