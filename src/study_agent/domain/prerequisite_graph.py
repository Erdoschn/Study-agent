"""Versioned DAG of pedagogical prerequisites.

Edges are always oriented prerequisite -> target concept. The DAG models the
curriculum dependency relation only; it deliberately does not store learner
mastery or arbitrary semantic links such as "related_to".
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping


def _required_text(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


@dataclass(frozen=True)
class KnowledgePoint:
    id: str
    name: str
    description: str = ""
    aliases: tuple[str, ...] = ()
    source_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "knowledge point id"))
        object.__setattr__(self, "name", _required_text(self.name, "knowledge point name"))
        aliases = tuple(_required_text(item, "knowledge point alias") for item in self.aliases)
        refs = tuple(_required_text(item, "knowledge point source reference") for item in self.source_refs)
        if len(aliases) != len(set(aliases)):
            raise ValueError("knowledge point aliases must be unique")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "source_refs", refs)


@dataclass(frozen=True)
class PrerequisiteEdge:
    prerequisite_id: str
    target_id: str
    confidence: float = 1.0
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        prerequisite_id = _required_text(self.prerequisite_id, "prerequisite_id")
        target_id = _required_text(self.target_id, "target_id")
        if prerequisite_id == target_id:
            raise ValueError("a knowledge point cannot be its own prerequisite")
        try:
            confidence = float(self.confidence)
        except (TypeError, ValueError) as exc:
            raise ValueError("prerequisite confidence must be a finite number between 0 and 1") from exc
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("prerequisite confidence must be a finite number between 0 and 1")
        refs = tuple(_required_text(item, "prerequisite evidence reference") for item in self.evidence_refs)
        object.__setattr__(self, "prerequisite_id", prerequisite_id)
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class PrerequisiteGap:
    target_point_id: str
    prerequisite_point_id: str
    prerequisite_name: str
    learner_stage: str

    def __post_init__(self) -> None:
        for name in ("target_point_id", "prerequisite_point_id", "prerequisite_name"):
            _required_text(getattr(self, name), name)
        stage = str(self.learner_stage or "unknown").strip().lower() or "unknown"
        object.__setattr__(self, "learner_stage", stage)


@dataclass(frozen=True)
class PrerequisiteGraph:
    MAX_POINTS = 10_000
    MAX_EDGES = 50_000

    graph_id: str
    version: int
    points: tuple[KnowledgePoint, ...] = ()
    edges: tuple[PrerequisiteEdge, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_id", _required_text(self.graph_id, "graph_id"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("graph version must be a positive integer")
        if self.schema_version != 1:
            raise ValueError(f"unsupported prerequisite graph schema_version: {self.schema_version}")
        points = tuple(self.points)
        edges = tuple(self.edges)
        if any(not isinstance(item, KnowledgePoint) for item in points):
            raise TypeError("points must contain KnowledgePoint records")
        if any(not isinstance(item, PrerequisiteEdge) for item in edges):
            raise TypeError("edges must contain PrerequisiteEdge records")
        if len(points) > self.MAX_POINTS:
            raise ValueError(f"prerequisite graph exceeds the {self.MAX_POINTS} point limit")
        if len(edges) > self.MAX_EDGES:
            raise ValueError(f"prerequisite graph exceeds the {self.MAX_EDGES} edge limit")
        point_ids = tuple(item.id for item in points)
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("prerequisite graph contains duplicate point IDs")
        point_id_set = set(point_ids)
        edge_keys = tuple((item.prerequisite_id, item.target_id) for item in edges)
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("prerequisite graph contains duplicate edges")
        for edge in edges:
            missing = {edge.prerequisite_id, edge.target_id} - point_id_set
            if missing:
                raise ValueError("prerequisite edge references unknown point IDs: " + ", ".join(sorted(missing)))
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "edges", edges)
        # Validate acyclicity at construction and deserialization boundaries.
        self.topological_order()

    @property
    def points_by_id(self) -> dict[str, KnowledgePoint]:
        return {item.id: item for item in self.points}

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic topological ordering of point IDs."""
        adjacency = {item.id: [] for item in self.points}
        indegree = {item.id: 0 for item in self.points}
        for edge in self.edges:
            adjacency[edge.prerequisite_id].append(edge.target_id)
            indegree[edge.target_id] += 1
        for targets in adjacency.values():
            targets.sort()
        ready = [point_id for point_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            current = heapq.heappop(ready)
            ordered.append(current)
            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    heapq.heappush(ready, target)
        if len(ordered) != len(indegree):
            raise ValueError("prerequisite graph contains a cycle; prerequisite relations must form a DAG")
        return tuple(ordered)

    def prerequisites_for(self, target_id: str, *, transitive: bool = True) -> tuple[KnowledgePoint, ...]:
        """Return prerequisites in learning order; prerequisites precede dependent concepts."""
        target_id = _required_text(target_id, "target_id")
        points = self.points_by_id
        if target_id not in points:
            raise KeyError(f"unknown target knowledge point: {target_id}")
        incoming: dict[str, list[str]] = {point_id: [] for point_id in points}
        for edge in self.edges:
            incoming[edge.target_id].append(edge.prerequisite_id)
        if not transitive:
            found = set(incoming[target_id])
        else:
            found: set[str] = set()
            pending = list(incoming[target_id])
            while pending:
                current = pending.pop()
                if current in found:
                    continue
                found.add(current)
                pending.extend(incoming[current])
        order = self.topological_order()
        return tuple(points[item] for item in order if item in found)

    def prerequisite_gaps(
        self,
        target_id: str,
        learner_stages: Mapping[str, str],
        *,
        mastered_stages: tuple[str, ...] = ("familiar", "mastered"),
    ) -> tuple[PrerequisiteGap, ...]:
        """Find unmastered transitive prerequisites in the order they should be learned."""
        points = self.points_by_id
        target = _required_text(target_id, "target_id")
        if target not in points:
            raise KeyError(f"unknown target knowledge point: {target}")
        normalized_mastered = {str(item).strip().lower() for item in mastered_stages}
        if not normalized_mastered:
            raise ValueError("mastered_stages must contain at least one stage")
        gaps = []
        for point in self.prerequisites_for(target, transitive=True):
            stage = str(learner_stages.get(point.id, "unknown") or "unknown").strip().lower()
            if stage not in {"unknown", "new", "learning", "familiar", "mastered", "weak"}:
                stage = "unknown"
            if stage not in normalized_mastered:
                gaps.append(PrerequisiteGap(
                    target_point_id=target,
                    prerequisite_point_id=point.id,
                    prerequisite_name=point.name,
                    learner_stage=stage or "unknown",
                ))
        return tuple(gaps)

    def to_dict(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "version": self.version,
            "schema_version": self.schema_version,
            "points": [
                {
                    "id": point.id,
                    "name": point.name,
                    "description": point.description,
                    "aliases": list(point.aliases),
                    "source_refs": list(point.source_refs),
                }
                for point in self.points
            ],
            "edges": [
                {
                    "prerequisite_id": edge.prerequisite_id,
                    "target_id": edge.target_id,
                    "confidence": edge.confidence,
                    "evidence_refs": list(edge.evidence_refs),
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PrerequisiteGraph":
        if not isinstance(data, dict):
            raise TypeError("PrerequisiteGraph data must be a dictionary")
        return cls(
            graph_id=data.get("graph_id", ""),
            version=data.get("version", 1),
            schema_version=data.get("schema_version", 1),
            points=tuple(KnowledgePoint(
                id=item["id"],
                name=item["name"],
                description=item.get("description", ""),
                aliases=tuple(item.get("aliases", [])),
                source_refs=tuple(item.get("source_refs", [])),
            ) for item in data.get("points", [])),
            edges=tuple(PrerequisiteEdge(
                prerequisite_id=item["prerequisite_id"],
                target_id=item["target_id"],
                confidence=item.get("confidence", 1.0),
                evidence_refs=tuple(item.get("evidence_refs", [])),
            ) for item in data.get("edges", [])),
        )
