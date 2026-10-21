"""Build learner-specific prerequisite-gap diagnostics for an answer revision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from study_agent.domain.answer_revision import AnswerRevision
from study_agent.domain.prerequisite_graph import PrerequisiteGap, PrerequisiteGraph


@dataclass(frozen=True)
class PrerequisiteDiagnostic:
    user_id: str
    task_id: str
    answer_revision_id: str
    graph_id: str
    graph_version: int
    target_point_ids: tuple[str, ...]
    gaps: tuple[PrerequisiteGap, ...]

    def __post_init__(self) -> None:
        for name in ("user_id", "task_id", "answer_revision_id", "graph_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if (
            not isinstance(self.graph_version, int)
            or isinstance(self.graph_version, bool)
            or self.graph_version < 1
        ):
            raise ValueError("graph_version must be a positive integer")
        target_ids = tuple(str(item or "").strip() for item in self.target_point_ids)
        if any(not item for item in target_ids) or len(target_ids) != len(set(target_ids)):
            raise ValueError("target_point_ids must be unique non-empty IDs")
        gaps = tuple(self.gaps)
        if any(item.target_point_id not in target_ids for item in gaps):
            raise ValueError("prerequisite gap references an unrequested target point")
        object.__setattr__(self, "target_point_ids", target_ids)
        object.__setattr__(self, "gaps", gaps)


def diagnose_prerequisite_gaps(
    revision: AnswerRevision,
    graph: PrerequisiteGraph,
    learner_stages: Mapping[str, str],
) -> PrerequisiteDiagnostic:
    """Map an answer's linked knowledge points to the learner's missing foundations.

    The curriculum graph is shared knowledge structure; learner stages are passed
    per request and are never persisted into that shared graph.
    """
    point_ids = revision.knowledge_point_ids
    unknown = sorted(set(point_ids) - set(graph.points_by_id))
    if unknown:
        raise ValueError("answer revision references unknown knowledge points: " + ", ".join(unknown))

    gaps: list[PrerequisiteGap] = []
    seen: set[tuple[str, str]] = set()
    for target_id in point_ids:
        for gap in graph.prerequisite_gaps(target_id, learner_stages):
            key = (gap.target_point_id, gap.prerequisite_point_id)
            if key not in seen:
                seen.add(key)
                gaps.append(gap)

    return PrerequisiteDiagnostic(
        user_id=revision.user_id,
        task_id=revision.task_id,
        answer_revision_id=revision.revision_id,
        graph_id=graph.graph_id,
        graph_version=graph.version,
        target_point_ids=point_ids,
        gaps=tuple(gaps),
    )
