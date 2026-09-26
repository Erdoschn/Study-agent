from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re
import time

from .__debug__ import debug

LEARNING_STAGES = {"unknown", "new", "learning", "familiar", "mastered", "weak"}
RELATIONS = {"is_a", "part_of", "related_to", "used_in", "depends_on", "alias_of"}

DIFFICULTY_LEVELS = {
    "basic": 0.25,
    "undergraduate": 0.50,
    "graduate": 0.70,
    "postgraduate": 0.90,
    "postgraduate_plus": 1.00,
}
POSTGRADUATE_THRESHOLD = DIFFICULTY_LEVELS["postgraduate"]

def normalize_difficulty(value: str | float | int | None) -> tuple[str, float]:
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in DIFFICULTY_LEVELS:
            return key, DIFFICULTY_LEVELS[key]
        try:
            value = float(value)
        except ValueError:
            return "graduate", DIFFICULTY_LEVELS["graduate"]
    try:
        candidate = float(value)
        score = max(0.0, min(1.0, candidate)) if math.isfinite(candidate) else DIFFICULTY_LEVELS["graduate"]
    except (TypeError, ValueError):
        score = DIFFICULTY_LEVELS["graduate"]
    nearest = min(DIFFICULTY_LEVELS, key=lambda name: abs(DIFFICULTY_LEVELS[name] - score))
    return nearest, score


@dataclass
class KnowledgeNode:
    id: str
    name: str
    node_type: str = "concept"
    aliases: list[str] = field(default_factory=list)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    learner: "LearnerState" = field(default_factory=lambda: LearnerState())


@dataclass
class LearnerState:
    familiarity: float = 0.0
    confidence: float = 0.0
    exposure_count: int = 0
    successful_count: int = 0
    last_seen: float | None = None
    learning_stage: str = "unknown"
    assessment_history: list[dict[str, Any]] = field(default_factory=list)
    max_familiarity: float = 0.2
    highest_assessment_level: float = 0.0

    def recent_accuracy(self) -> float:
        if not self.assessment_history:
            return 0.0
        return sum(1 for x in self.assessment_history if x.get("correct")) / len(self.assessment_history)

    def as_dict(self) -> dict[str, Any]:
        return {"familiarity": round(self.familiarity, 3), "confidence": round(self.confidence, 3), "exposure_count": self.exposure_count, "successful_count": self.successful_count, "last_seen": self.last_seen, "learning_stage": self.learning_stage, "recent_accuracy": self.recent_accuracy(), "assessment_history": list(self.assessment_history[-8:]), "max_familiarity": round(self.max_familiarity, 3), "highest_assessment_level": round(self.highest_assessment_level, 3)}


@dataclass
class KnowledgeEdge:
    source: str
    target: str
    relation: str = "related_to"
    confidence: float = 0.0
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    learner: LearnerState = field(default_factory=LearnerState)


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
                debug.log("KnowledgeGraph", f"NODE LIMIT → rejected concept={name!r}")
                return ""
            node = KnowledgeNode(id=node_id, name=name, node_type=node_type)
            self.nodes[node_id] = node
        if alias and alias.strip() and alias.strip() not in node.aliases and alias.strip() != node.name:
            node.aliases.append(alias.strip())
        if evidence and evidence not in node.evidence_refs:
            node.evidence_refs.append(dict(evidence))
            del node.evidence_refs[:-10]
        return node_id

    def _apply_assessment(self, state: LearnerState, correct: bool, confidence: float | None = None, difficulty: float = 1.0) -> None:
        level, difficulty = normalize_difficulty(difficulty)
        state.exposure_count += 1
        if correct:
            state.successful_count += 1
        safe_confidence = None
        if confidence is not None:
            try:
                candidate = float(confidence)
                if math.isfinite(candidate):
                    safe_confidence = max(0.0, min(1.0, candidate))
            except (TypeError, ValueError):
                safe_confidence = None
        state.assessment_history.append({
            "correct": bool(correct),
            "confidence": safe_confidence,
            "timestamp": time.time(),
            "difficulty": difficulty,
            "difficulty_level": level,
        })
        del state.assessment_history[:-8]
        accuracy = state.recent_accuracy()
        state.highest_assessment_level = max(state.highest_assessment_level, difficulty)
        # Low-difficulty questions provide evidence of basics, but cannot certify advanced mastery.
        evidence_cap = max(0.2, difficulty)
        if difficulty < POSTGRADUATE_THRESHOLD:
            evidence_cap = min(evidence_cap, 0.89)
        state.max_familiarity = max(state.max_familiarity, evidence_cap)
        raw = 0.75 * state.familiarity + 0.25 * accuracy
        state.familiarity = max(0.0, min(state.max_familiarity, raw))
        observed_conf = safe_confidence if safe_confidence is not None else accuracy
        state.confidence = max(0.0, min(1.0, 0.75 * state.confidence + 0.25 * float(observed_conf)))
        state.last_seen = time.time()
        n = len(state.assessment_history)
        if n < 2:
            state.learning_stage = "new"
        elif n < 3:
            state.learning_stage = "learning" if accuracy >= 0.5 else "new"
        elif n >= 5 and difficulty >= POSTGRADUATE_THRESHOLD and accuracy >= 0.8 and all(x.get("correct") for x in state.assessment_history[-3:]) and state.highest_assessment_level >= 0.9:
            state.learning_stage = "mastered"
        elif n >= 3 and accuracy < 0.5 and sum(1 for x in state.assessment_history[-3:] if x.get("correct")) <= 1:
            state.learning_stage = "weak"
        elif accuracy >= 0.7:
            state.learning_stage = "familiar"
        else:
            state.learning_stage = "learning"
        debug.log(
            "KnowledgeGraph",
            f"ASSESS → stage={state.learning_stage}, correct={correct}, difficulty={level}/{difficulty:.2f}, accuracy={accuracy:.2f}, familiarity={state.familiarity:.2f}",
        )

    @staticmethod
    def difficulty_policy() -> dict[str, float]:
        return dict(DIFFICULTY_LEVELS)

    def update_learner(self, concept: str, correct: bool, confidence: float | None = None, difficulty: float = 1.0) -> None:
        node_id = self.add_concept(concept)
        if node_id:
            debug.log(
                "KnowledgeGraph",
                f"LEARNER UPDATE → concept={concept!r}, correct={bool(correct)}, difficulty={difficulty!r}",
            )
            self._apply_assessment(self.nodes[node_id].learner, correct, confidence, difficulty)

    def record_assessment(self, concepts: list[str], correct: bool, confidence: float | None = None, difficulty: float = 1.0) -> None:
        seen = set()
        for concept in concepts:
            name = str(concept or "").strip()
            node_id = self._id(name)
            if not name or not node_id or node_id in seen:
                continue
            seen.add(node_id)
            self.update_learner(name, correct, confidence, difficulty)
        debug.log(
            "KnowledgeGraph",
            f"RECORD ASSESSMENT → concepts={len(seen)}, correct={bool(correct)}, difficulty={difficulty!r}",
        )

    def update_relation_learner(self, source: str, target: str, relation: str, correct: bool, confidence: float | None = None, difficulty: float = 1.0) -> None:
        key = (self._id(source), self._id(target), relation)
        if key not in self.edges:
            self.add_relation(source, target, relation)
        edge = self.edges.get(key)
        if edge:
            self._apply_assessment(edge.learner, correct, confidence, difficulty)

    def learner_context(self, query: str, limit: int = 12) -> dict[str, Any]:
        node = self.nodes.get(self._id(query))
        if not node: return {"status": "unknown", "weak_concepts": []}
        related = [
            item for item in self.neighbors(query, limit * 2)
            if item.get("relation") != "supported_by_search"
        ]
        return {
            "status": node.learner.as_dict(),
            "weak_concepts": [
                x["name"]
                for x in related
                if x.get("learner", {}).get("learning_stage")
                in {"unknown", "new", "weak", "learning"}
            ][:limit],
        }


    def add_relation(self, source: str, target: str, relation: str = "related_to",
                     confidence: float = 0.0, evidence: dict[str, Any] | None = None) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"未知知识关系：{relation}")
        try:
            candidate = float(confidence)
            safe_confidence = max(0.0, min(1.0, candidate)) if math.isfinite(candidate) else 0.0
        except (TypeError, ValueError):
            safe_confidence = 0.0

        source_id = self.add_concept(source)
        target_id = self.add_concept(target)
        if not source_id or not target_id or source_id == target_id:
            return
        key = (source_id, target_id, str(relation))
        edge = self.edges.get(key)
        if edge is None:
            if len(self.edges) >= self.MAX_EDGES:
                debug.log("KnowledgeGraph", f"EDGE LIMIT → rejected relation={source!r} -[{relation}]-> {target!r}")
                return
            edge = KnowledgeEdge(source_id, target_id, str(relation), safe_confidence)
            self.edges[key] = edge
        else:
            edge.confidence = max(edge.confidence, safe_confidence)
        if evidence and evidence not in edge.evidence_refs:
            edge.evidence_refs.append(dict(evidence))
            del edge.evidence_refs[:-10]
        debug.log(
            "KnowledgeGraph",
            f"RELATION → {source!r} -[{relation}]-> {target!r} confidence={edge.confidence:.2f}",
        )

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
                self._add_provenance_edge(query_id, title_id, evidence)

    def _add_provenance_edge(self, source_id: str, target_id: str, evidence: dict[str, Any]) -> None:
        key = (source_id, target_id, "supported_by_search")
        edge = self.edges.get(key)
        if edge is None and len(self.edges) < self.MAX_EDGES:
            edge = KnowledgeEdge(source_id, target_id, "supported_by_search", 0.5)
            self.edges[key] = edge
        if edge and evidence not in edge.evidence_refs:
            edge.evidence_refs.append(dict(evidence))
            del edge.evidence_refs[:-10]

    @staticmethod
    def _concept_tokens(text: str) -> set[str]:
        value = str(text or "").lower()
        tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", value))
        expanded = set(tokens)
        for token in list(tokens):
            if re.fullmatch(r"[\u4e00-\u9fff]+", token):
                expanded.update(token[i:i + 2] for i in range(len(token) - 1))
        return {token for token in expanded if len(token) > 1}

    def relevant_concepts(self, query: str, limit: int = 6) -> list[str]:
        query_id = self._id(query)
        if query_id in self.nodes:
            return [self.nodes[query_id].name]

        query_tokens = self._concept_tokens(query)
        if not query_tokens:
            return []

        ranked = []
        for node in self.nodes.values():
            if node.node_type != "concept":
                continue
            candidate_text = " ".join([node.name, *node.aliases])
            overlap = query_tokens & self._concept_tokens(candidate_text)
            if not overlap:
                continue
            score = len(overlap) / max(1, len(query_tokens))
            ranked.append((score, len(overlap), node.name))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        result = [name for _, _, name in ranked[:limit]]
        debug.log(
            "KnowledgeGraph",
            f"RESOLVE → query={query!r}, matched={result}",
        )
        return result

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
                        "learner": edge.learner.as_dict(),
                    })
        items.sort(key=lambda x: x["confidence"], reverse=True)
        return items[:limit]

    def related_concepts(self, query: str, limit: int = 8) -> list[str]:
        node_id = self._id(query)
        if node_id not in self.nodes:
            return []
        candidates = []
        for item in self.neighbors(query, limit=limit * 2):
            name = str(item.get("name", "")).strip()
            if name and item.get("relation") != "supported_by_search":
                candidates.append(name)
        return candidates[:limit]

    def search_candidates(self, query: str, limit: int = 8) -> list[str]:
        candidates = []
        node = self.nodes.get(self._id(query))
        if node:
            candidates.extend(node.aliases)
        candidates.extend(self.related_concepts(query, limit=limit))
        seen = set()
        result = []
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result[:limit]

    def context_for(self, query: str, limit: int = 12) -> dict[str, Any]:
        matched_names = self.relevant_concepts(query, limit=6)
        matched_ids = [self._id(name) for name in matched_names]
        neighbors = []
        seen_neighbors = set()
        for node_name in matched_names:
            for item in self.neighbors(node_name, limit=max(4, limit)):
                key = (item.get("name"), item.get("relation"), item.get("direction"))
                if key in seen_neighbors:
                    continue
                seen_neighbors.add(key)
                neighbors.append(item)
        neighbors.sort(key=lambda item: item.get("confidence", 0.0), reverse=True)

        known_concepts = []
        for node_id in matched_ids:
            node = self.nodes.get(node_id)
            if node:
                known_concepts.append({
                    "name": node.name,
                    "aliases": list(node.aliases),
                    "learner": node.learner.as_dict(),
                })

        weak = []
        for item in neighbors:
            if item.get("relation") == "supported_by_search":
                continue
            if item.get("learner", {}).get("learning_stage") in {"unknown", "new", "weak", "learning"}:
                weak.append(item["name"])

        search_candidates = []
        seen_candidates = set()
        for node_name in matched_names:
            for candidate in self.search_candidates(node_name, limit=min(8, limit)):
                if candidate and candidate not in seen_candidates:
                    seen_candidates.add(candidate)
                    search_candidates.append(candidate)

        context = {
            "query": query,
            "matched_concepts": matched_names,
            "known_concepts": known_concepts,
            "neighbors": neighbors[:limit],
            "learner_context": {
                "matched_concepts": matched_names,
                "weak_concepts": weak[:limit],
            },
            "search_candidates": search_candidates[:8],
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }
        debug.log(
            "KnowledgeGraph",
            f"CONTEXT → query={query!r}, matched={matched_names}, neighbors={len(context['neighbors'])}",
        )
        return context
