from dataclasses import dataclass, field
from typing import Any

from .__debug__ import debug


@dataclass
class BDIState:
    """Student mental-state model. BDI means Beliefs, Desires, Intentions."""

    beliefs: list[str] = field(default_factory=list)
    desires: list[str] = field(default_factory=list)
    intentions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "beliefs": list(self.beliefs),
            "desires": list(self.desires),
            "intentions": list(self.intentions),
        }

    def add(self, category: str, items: list[str], limit: int) -> None:
        target = getattr(self, category)
        for item in items:
            text = str(item).strip()
            if text and text not in target:
                target.append(text)
        del target[:-limit]


@dataclass
class StudentMind:
    """Two-timescale student model: short-term and long-term BDI."""

    short_term: BDIState = field(default_factory=BDIState)
    long_term: BDIState = field(default_factory=BDIState)
    recent_decisions: list[str] = field(default_factory=list)
    belief_history: list[dict[str, Any]] = field(default_factory=list)
    belief_support: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "short_term": self.short_term.as_dict(),
            "long_term": self.long_term.as_dict(),
            "recent_decisions": list(self.recent_decisions),
            "belief_history": [dict(item) for item in self.belief_history],
            "belief_support": {key: [dict(ref) for ref in refs] for key, refs in self.belief_support.items()},
        }

    def apply_update(self, update: dict[str, Any]) -> None:
        """Apply an LLM hypothesis after Harness-side schema validation."""
        if not isinstance(update, dict):
            return

        for horizon, target, limit in (
            ("short_term", self.short_term, 8),
            ("long_term", self.long_term, 30),
        ):
            data = update.get(horizon)
            if not isinstance(data, dict):
                continue
            for category in ("beliefs", "desires", "intentions"):
                items = data.get(category, [])
                if isinstance(items, list):
                    target.add(category, [str(x) for x in items], limit)

        decisions = update.get("recent_decisions", [])
        if isinstance(decisions, list):
            for item in decisions:
                text = str(item).strip()
                if text and text not in self.recent_decisions:
                    self.recent_decisions.append(text)
        del self.recent_decisions[:-12]

    def revise_beliefs(self, revisions: list[dict[str, Any]], evidence: list[dict[str, Any]] | None = None) -> None:
        """Apply explicit belief replacements/retractions with a small audit trail."""
        if not isinstance(revisions, list):
            return
        for item in revisions:
            if not isinstance(item, dict):
                continue
            old = str(item.get("old", "")).strip()
            new = str(item.get("new", "")).strip()
            horizon = str(item.get("horizon", "short_term")).strip()
            status = str(item.get("status", "REVISED")).upper()
            reason = str(item.get("reason", "")).strip()
            evidence_refs = item.get("evidence_refs", [])
            if not old or horizon not in {"short_term", "long_term"}:
                continue
            target = getattr(self, horizon).beliefs
            if old in target:
                target.remove(old)
            if new and status in {"REVISED", "CONFIRMED"} and new not in target:
                target.append(new)
            valid_refs = []
            for ref in evidence_refs if isinstance(evidence_refs, list) else []:
                if not isinstance(ref, int) or evidence is None or ref < 0 or ref >= len(evidence):
                    continue
                item_ref = evidence[ref]
                if not isinstance(item_ref, dict):
                    continue
                valid_refs.append({
                    "index": ref, "source": item_ref.get("source"),
                    "title": item_ref.get("title"), "identifier": item_ref.get("identifier"),
                    "harness_relevance": item_ref.get("harness_relevance", "UNCERTAIN"),
                })
            if new and status in {"REVISED", "CONFIRMED"} and valid_refs:
                self.belief_support[new] = valid_refs
            elif new and status in {"REVISED", "CONFIRMED"} and new not in self.belief_support:
                self.belief_support[new] = []
            if status == "RETRACTED":
                self.belief_support.pop(old, None)
            self.belief_history.append({
                "horizon": horizon, "old": old, "new": new,
                "status": status if status in {"REVISED", "CONFIRMED", "RETRACTED", "UNCERTAIN"} else "UNCERTAIN",
                "reason": reason, "evidence_refs": valid_refs,
            })
        del self.belief_history[:-20]


@dataclass
class StudentState:
    known_topics: set[str] = field(default_factory=set)
    weak_topics: set[str] = field(default_factory=set)
    misconceptions: list[str] = field(default_factory=list)
    mind: StudentMind = field(default_factory=StudentMind)

    def apply_mind_update(self, update: dict[str, Any]) -> None:
        self.mind.apply_update(update)

    def sync_from_knowledge_graph(self, graph) -> None:
        """Synchronize explicit learner evidence into the fast teaching-facing state."""
        if graph is None:
            return
        known = set()
        weak = set()
        for node in getattr(graph, "nodes", {}).values():
            # Search-result/document nodes are evidence provenance, not learner concepts.
            if getattr(node, "node_type", "concept") != "concept":
                continue
            stage = getattr(getattr(node, "learner", None), "learning_stage", "unknown")
            if stage in {"familiar", "mastered"}:
                known.add(node.name)
            elif stage in {"weak", "new", "learning"}:
                weak.add(node.name)
        self.known_topics = known
        self.weak_topics = weak - known
        debug.log(
            "StudentState",
            f"SYNC → known={sorted(self.known_topics)}, weak={sorted(self.weak_topics)}",
        )


@dataclass
class AgentStep:
    step_id: int
    action: str
    model: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning_summary: str = ""
    observation: Any = None
    success: bool = True
    error: str = ""


@dataclass
class AgentState:
    question: str
    student: StudentState = field(default_factory=StudentState)
    goal: str = ""
    goal_context: list[str] = field(default_factory=list)
    task_type: str = ""
    domain: str = ""
    task_analysis: Any = None
    plan: Any = None
    search_sources: list[str] = field(default_factory=list)
    search_sort_by: str = "relevance"
    current_plan_step: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    evidence_relevance: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    finished: bool = False
    error: str | None = None
    max_steps: int | None = None
    action_counts: dict[str, int] = field(default_factory=dict)
    last_action: str = ""
    last_observation: Any = None
    last_error_type: str = ""
    recovery_count: int = 0
    knowledge_graph: Any = None
    pending_assessment: dict[str, Any] | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.last_action = step.action
        self.last_observation = step.observation
        self.action_counts[step.action] = self.action_counts.get(step.action, 0) + 1

    def observations(self) -> list[Any]:
        return [step.observation for step in self.steps if step.observation is not None]
