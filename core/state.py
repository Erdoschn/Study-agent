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
    """Two-timescale student model: short-term hypotheses and promoted long-term memory."""

    LONG_TERM_PROMOTION_CONFIRMATIONS = 3
    MAX_LONG_TERM_CANDIDATES = 64

    short_term: BDIState = field(default_factory=BDIState)
    long_term: BDIState = field(default_factory=BDIState)
    recent_decisions: list[str] = field(default_factory=list)
    belief_history: list[dict[str, Any]] = field(default_factory=list)
    belief_support: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    long_term_candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    interaction_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "short_term": self.short_term.as_dict(),
            "long_term": self.long_term.as_dict(),
            "recent_decisions": list(self.recent_decisions),
            "belief_history": [dict(item) for item in self.belief_history],
            "belief_support": {key: [dict(ref) for ref in refs] for key, refs in self.belief_support.items()},
            "long_term_candidates": {
                key: {
                    "category": value.get("category"),
                    "text": value.get("text"),
                    "confirmations": value.get("confirmations", 0),
                }
                for key, value in self.long_term_candidates.items()
            },
        }

    def begin_interaction(self) -> None:
        """Advance the interaction epoch used to count independent confirmations."""
        self.interaction_count += 1

    @staticmethod
    def _candidate_key(category: str, text: str) -> str:
        return f"{category}:{text.casefold()}"

    def _observe_long_term_candidate(self, category: str, text: str) -> None:
        """Count a candidate at most once per interaction before promotion."""
        if not text:
            return
        target = getattr(self.long_term, category)
        if text in target:
            return

        key = self._candidate_key(category, text)
        record = self.long_term_candidates.get(key)
        if record is None:
            record = {
                "category": category,
                "text": text,
                "confirmations": 0,
                "last_interaction": None,
            }
            self.long_term_candidates[key] = record

        if record.get("last_interaction") != self.interaction_count:
            record["confirmations"] = int(record.get("confirmations", 0)) + 1
            record["last_interaction"] = self.interaction_count

        if record["confirmations"] >= self.LONG_TERM_PROMOTION_CONFIRMATIONS:
            target.append(text)
            del self.long_term_candidates[key]
            del target[:-30]
            debug.log(
                "StudentMind",
                f"LONG_TERM PROMOTED → category={category}, text={text!r}, confirmations={self.LONG_TERM_PROMOTION_CONFIRMATIONS}",
            )

        if len(self.long_term_candidates) > self.MAX_LONG_TERM_CANDIDATES:
            oldest_key = min(
                self.long_term_candidates,
                key=lambda item: int(self.long_term_candidates[item].get("last_interaction") or -1),
            )
            self.long_term_candidates.pop(oldest_key, None)

    def confirm_long_term(self, category: str, items: list[str]) -> None:
        """Explicitly promote user/application-confirmed stable memory."""
        if category not in {"beliefs", "desires", "intentions"}:
            return
        target = getattr(self.long_term, category)
        for item in items if isinstance(items, list) else []:
            text = str(item).strip()
            if not text:
                continue
            target.append(text)
            del target[:-30]
            self.long_term_candidates.pop(self._candidate_key(category, text), None)

    def apply_update(self, update: dict[str, Any]) -> None:
        """Apply model hypotheses; long-term entries require repeated interaction-level confirmation."""
        if not isinstance(update, dict):
            return

        short_term = update.get("short_term")
        if isinstance(short_term, dict):
            for category in ("beliefs", "desires", "intentions"):
                items = short_term.get(category, [])
                if isinstance(items, list):
                    self.short_term.add(category, [str(x) for x in items], 8)

        long_term = update.get("long_term")
        if isinstance(long_term, dict):
            for category in ("beliefs", "desires", "intentions"):
                items = long_term.get(category, [])
                if isinstance(items, list):
                    for item in items:
                        text = str(item).strip()
                        if text:
                            self._observe_long_term_candidate(category, text)

        decisions = update.get("recent_decisions", [])
        if isinstance(decisions, list):
            for item in decisions:
                text = str(item).strip()
                if text and text not in self.recent_decisions:
                    self.recent_decisions.append(text)
        del self.recent_decisions[:-12]

    def revise_beliefs(self, revisions: list[dict[str, Any]], evidence: list[dict[str, Any]] | None = None) -> None:
        """Apply belief revisions with evidence gates for long-term memory."""
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

            valid_refs = []
            for ref in evidence_refs if isinstance(evidence_refs, list) else []:
                if not isinstance(ref, int) or evidence is None or ref < 0 or ref >= len(evidence):
                    continue
                item_ref = evidence[ref]
                if not isinstance(item_ref, dict):
                    continue
                relevance = str(item_ref.get("harness_relevance", "UNCERTAIN")).upper()
                if relevance not in {"DIRECT", "PARTIAL"}:
                    debug.log(
                        "StudentMind",
                        f"BELIEF SUPPORT SKIP → index={ref}, relevance={relevance}",
                    )
                    continue
                valid_refs.append({
                    "index": ref,
                    "source": item_ref.get("source"),
                    "title": item_ref.get("title"),
                    "identifier": item_ref.get("identifier"),
                    "harness_relevance": relevance,
                })

            # Long-term memory cannot be mutated by an unsupported LLM proposal.
            long_term_requires_support = horizon == "long_term" and status in {
                "REVISED", "CONFIRMED", "RETRACTED"
            }
            blocked = long_term_requires_support and not valid_refs

            target = getattr(self, horizon).beliefs
            applied_status = status if status in {"REVISED", "CONFIRMED", "RETRACTED", "UNCERTAIN"} else "UNCERTAIN"

            if blocked:
                debug.log(
                    "StudentMind",
                    f"BELIEF REVISION BLOCKED → horizon={horizon}, status={status}, reason=no DIRECT/PARTIAL evidence",
                )
                applied_status = "UNCERTAIN"
            elif old in target:
                target.remove(old)
                if new and status in {"REVISED", "CONFIRMED"} and new not in target:
                    target.append(new)
            elif new and status in {"REVISED", "CONFIRMED"}:
                target.append(new)

            if new and status in {"REVISED", "CONFIRMED"}:
                # Support metadata is useful for both horizons. For long-term
                # memory, an empty list explicitly means the proposal lacked
                # acceptable evidence and therefore was not applied.
                self.belief_support[new] = valid_refs if not blocked else []
            elif horizon == "long_term" and status == "RETRACTED" and not blocked:
                self.belief_support.pop(old, None)

            debug.log(
                "StudentMind",
                f"BELIEF REVISION → status={applied_status}, horizon={horizon}, evidence_refs={len(valid_refs)}",
            )
            self.belief_history.append({
                "horizon": horizon,
                "old": old,
                "new": new,
                "status": applied_status,
                "reason": reason,
                "evidence_refs": valid_refs,
                "applied": not blocked,
            })

        del self.belief_history[:-20]

@dataclass
class StudentState:
    known_topics: set[str] = field(default_factory=set)
    weak_topics: set[str] = field(default_factory=set)
    learning_topics: set[str] = field(default_factory=set)
    misconceptions: list[str] = field(default_factory=list)
    mind: StudentMind = field(default_factory=StudentMind)

    def apply_mind_update(self, update: dict[str, Any]) -> None:
        self.mind.apply_update(update)

    def confirm_long_term_memory(self, category: str, items: list[str]) -> None:
        self.mind.confirm_long_term(category, items)

    def sync_from_knowledge_graph(self, graph) -> None:
        """Synchronize explicit learner evidence into the fast teaching-facing state."""
        if graph is None:
            return
        graph_known = set()
        graph_weak = set()
        graph_learning = set()
        for node in getattr(graph, "nodes", {}).values():
            # Search-result/document nodes are evidence provenance, not learner concepts.
            if getattr(node, "node_type", "concept") != "concept":
                continue
            stage = getattr(getattr(node, "learner", None), "learning_stage", "unknown")
            if stage in {"familiar", "mastered"}:
                graph_known.add(node.name)
            elif stage == "weak":
                graph_weak.add(node.name)
            elif stage in {"new", "learning"}:
                graph_learning.add(node.name)

        # Preserve externally supplied learner topics that have no conflicting
        # graph assessment; explicit graph assessment wins on conflicts.
        self.known_topics = (set(self.known_topics) - graph_weak - graph_learning) | graph_known
        self.weak_topics = (set(self.weak_topics) - graph_known - graph_learning) | graph_weak
        self.learning_topics = (set(self.learning_topics) - graph_known - graph_weak) | graph_learning
        self.weak_topics -= self.known_topics
        self.learning_topics -= self.known_topics
        self.learning_topics -= self.weak_topics
        debug.log(
            "StudentState",
            f"SYNC → graph_known={sorted(graph_known)}, graph_weak={sorted(graph_weak)}, "
            f"graph_learning={sorted(graph_learning)}, known={sorted(self.known_topics)}, "
            f"weak={sorted(self.weak_topics)}, learning={sorted(self.learning_topics)}",
        )


@dataclass
class AgentStep:
    step_id: int
    action: str
    model: str | None = None
    effort: str | None = None
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
    assessment_requested: bool | None = None
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
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
        self.last_action = step.action
        self.last_observation = step.observation
        self.action_counts[step.action] = self.action_counts.get(step.action, 0) + 1
        debug.log(
            "AgentState",
            f"STEP → id={step.step_id}, action={step.action}, tool={step.tool or 'none'}, success={step.success}, error={step.error or 'none'}",
        )

    def observations(self) -> list[Any]:
        return [step.observation for step in self.steps if step.observation is not None]
