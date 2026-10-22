"""Shadow-mode integration of versioned dossiers, answer coverage and prerequisite DAGs.

The runtime persists traceability, but deliberately leaves requirement coverage
UNRESOLVED until a dedicated requirement-by-requirement verifier is wired in.
An unavailable or unassessed prerequisite stage is never described as a proven
learning weakness.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.storage_security import ensure_private_directory
from core.user_identity import normalize_username, user_graph_path

from study_agent.application.prerequisite_diagnostics import (
    PrerequisiteDiagnostic,
    diagnose_prerequisite_gaps,
)
from study_agent.domain.answer_revision import (
    AnswerRevision,
    CoverageEntry,
    CoverageLedger,
    CoverageOutcome,
)
from study_agent.domain.prerequisite_graph import PrerequisiteGraph
from study_agent.domain.task_dossier import (
    AnswerClaim,
    DossierFact,
    FactOrigin,
    RequirementStatus,
    SourceReference,
    TaskDossier,
    TaskRequirement,
)
from study_agent.infrastructure.answer_revision_sqlite import (
    SQLiteAnswerRevisionRepository,
    SQLiteCoverageLedgerRepository,
)
from study_agent.infrastructure.dossier_sqlite import SQLiteTaskDossierRepository
from study_agent.infrastructure.prerequisite_graph_sqlite import SQLitePrerequisiteGraphRepository


_REQUIREMENT_HINT = re.compile(
    r"(?:必须|需要|请|要求|只给|只需|只要|不得|不要|不能|禁止|保留|"
    r"有效数字|四舍五入|单位|输出|列出|包括|分别|分开|用中文|用英文|"
    r"must|must not|should|only|do not|include|exclude|provide|express|round|state)",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class RuntimeEpistemicResult:
    dossier_version: int
    answer_revision_id: str
    revision_number: int
    coverage_count: int
    coverage_unresolved_count: int
    prerequisite_diagnostic: PrerequisiteDiagnostic | None = None
    prerequisite_note: str = ""


def _request_text(value: object) -> str:
    text = str(value or "").strip()
    if "【原始文件附件】" in text:
        text = text.split("【原始文件附件】", 1)[0].rstrip()
    text = _CONTROL.sub("", text)
    return text[:12_000]


def _extract_requirements(question: str) -> tuple[str, ...]:
    found: list[str] = []
    for raw_line in question.splitlines():
        line = _BULLET.sub("", raw_line).strip(" \t：:;；")
        if not line or len(line) > 400 or not _REQUIREMENT_HINT.search(line):
            continue
        if line.startswith(("【系统】", "【助手】", "以下是当前 StudyAgent 对话上下文")):
            continue
        normalized = " ".join(line.split())
        if normalized not in found:
            found.append(normalized)
        if len(found) >= 32:
            break
    return tuple(found)


def _model_claims(value: object) -> tuple[AnswerClaim, ...]:
    records = value if isinstance(value, (list, tuple)) else ()
    claims: list[AnswerClaim] = []
    seen: set[str] = set()
    for index, item in enumerate(records[:30]):
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or item.get("claim") or item.get("text") or "").strip()
        if not statement:
            continue
        identity = "claim-" + hashlib.sha256(statement.encode("utf-8")).hexdigest()[:16]
        if identity in seen:
            continue
        seen.add(identity)
        kind = str(item.get("claim_type") or item.get("type") or "model_claim").strip() or "model_claim"
        claims.append(AnswerClaim(
            id=identity,
            statement=statement[:2_000],
            claim_type=kind[:80],
            answer_span=str(item.get("answer_span") or "")[:2_000],
        ))
    return tuple(claims)


def _normalize_label(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _matching_knowledge_points(
    graph: PrerequisiteGraph,
    question: str,
    domain: str = "",
) -> tuple[str, ...]:
    text = _normalize_label(question)
    hits: list[tuple[int, str]] = []
    for point in graph.points:
        labels = (point.name, *point.aliases)
        lengths = [
            len(_normalize_label(label))
            for label in labels
            if len(_normalize_label(label)) >= 2
            and _normalize_label(label) in text
        ]
        if lengths:
            hits.append((max(lengths), point.id))
    if not hits and _normalize_label(domain):
        for point in graph.points:
            if any(_normalize_label(label) == _normalize_label(domain) for label in (point.name, *point.aliases)):
                hits.append((len(_normalize_label(point.name)), point.id))
    # Longer explicit labels outrank broader substrings. Keep graph order stable for ties.
    hits.sort(key=lambda item: (-item[0], item[1]))
    return tuple(point_id for _length, point_id in hits[:12])


def _learner_stages(graph: PrerequisiteGraph, result: Any) -> dict[str, str]:
    student = getattr(result, "student", None)
    known = {_normalize_label(value) for value in (getattr(student, "known_topics", ()) or ())}
    weak = {_normalize_label(value) for value in (getattr(student, "weak_topics", ()) or ())}
    stages: dict[str, str] = {}
    for point in graph.points:
        labels = {_normalize_label(point.name), *(_normalize_label(alias) for alias in point.aliases)}
        if labels & weak:
            stages[point.id] = "weak"
        elif labels & known:
            stages[point.id] = "familiar"
        else:
            stages[point.id] = "unknown"
    return stages


def _curriculum_graph(
    curriculum_db_path: str | Path | None,
) -> PrerequisiteGraph | None:
    configured = curriculum_db_path or os.getenv("STUDY_AGENT_CURRICULUM_DB", "")
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    if not path.exists() or path.is_symlink() or not path.is_file():
        return None
    repository = SQLitePrerequisiteGraphRepository(path)
    return repository.load_latest(graph_id="curriculum")


def persist_runtime_epistemic_artifacts(
    *,
    username: object,
    task_id: object,
    question: str,
    result: Any,
    data_root: str | Path,
    max_profiles: int | None = None,
    curriculum_db_path: str | Path | None = None,
) -> RuntimeEpistemicResult:
    """Persist one immutable answer revision and a conservative coverage ledger."""
    user = normalize_username(username)
    task = str(task_id or "").strip()
    if not task or len(task) > 128 or any(ord(char) < 32 for char in task):
        raise ValueError("task_id 格式无效。")
    answer = str(getattr(result, "final_answer", "") or "").strip()
    if not answer:
        raise ValueError("不能为没有最终答案的任务建立 AnswerRevision。")

    graph: PrerequisiteGraph | None = None
    try:
        graph = _curriculum_graph(curriculum_db_path)
    except Exception:
        # A missing or invalid curriculum graph must not make answering unavailable.
        graph = None

    graph_targets: tuple[str, ...] = ()
    domain = str(getattr(result, "domain", "") or "")
    if graph is not None:
        graph_targets = _matching_knowledge_points(graph, _request_text(question), domain)

    profile_graph_path = user_graph_path(
        user,
        data_root=data_root,
        max_profiles=max_profiles,
    )
    database_path = ensure_private_directory(profile_graph_path.parent) / "epistemic.sqlite3"
    dossier_repository = SQLiteTaskDossierRepository(database_path)
    revision_repository = SQLiteAnswerRevisionRepository(database_path)
    coverage_repository = SQLiteCoverageLedgerRepository(database_path)

    previous_dossier = dossier_repository.load_latest(user_id=user, task_id=task)
    dossier_version = 1 if previous_dossier is None else previous_dossier.version + 1
    clean_question = _request_text(question) or "原始用户请求不可用；仅保留本轮答案版本。"
    digest = hashlib.sha256(
        f"{user}\0{task}\0{dossier_version}\0{clean_question}".encode("utf-8")
    ).hexdigest()
    source_id = "source-" + digest[:20]
    source = SourceReference(
        id=source_id,
        kind="user_request",
        locator=f"task:{task}:request:v{dossier_version}",
        content_hash="sha256:" + hashlib.sha256(clean_question.encode("utf-8")).hexdigest(),
    )
    requirements = tuple(
        TaskRequirement(
            id="requirement-" + hashlib.sha256(f"{index}\0{line}".encode("utf-8")).hexdigest()[:16],
            statement=line,
            source_refs=(source_id,),
            critical=True,
            status=RequirementStatus.OPEN,
            notes="Detected from explicit request phrasing; semantic satisfaction has not yet been verified.",
        )
        for index, line in enumerate(_extract_requirements(clean_question), start=1)
    )
    dossier = TaskDossier(
        user_id=user,
        task_id=task,
        version=dossier_version,
        sources=(source,),
        facts=(DossierFact(
            id="user-request-context",
            statement=clean_question,
            origin=FactOrigin.USER_STATED,
            source_refs=(source_id,),
        ),),
        requirements=requirements,
    )
    dossier_repository.save(dossier)

    previous_revision = revision_repository.load_latest(user_id=user, task_id=task)
    revision_number = 1 if previous_revision is None else previous_revision.revision_number + 1
    parent_revision_id = previous_revision.revision_id if previous_revision else None
    revision_id = uuid.uuid4().hex
    claims = _model_claims(getattr(result, "claims", ()))

    # Build a provisional in-memory revision so the diagnostic can bind to the
    # exact revision ID. The diagnostic itself becomes part of the final stored
    # answer; it must not be appended only after persistence.
    revision = AnswerRevision(
        user_id=user,
        task_id=task,
        revision_id=revision_id,
        dossier_version=dossier.version,
        answer_text=answer,
        claims=claims,
        knowledge_point_ids=graph_targets,
        parent_revision_id=parent_revision_id,
        revision_number=revision_number,
    )
    revision.validate_against(dossier)

    diagnostic = None
    diagnostic_note = ""
    if graph is not None and graph_targets:
        diagnostic = diagnose_prerequisite_gaps(
            revision,
            graph,
            _learner_stages(graph, result),
        )
        if diagnostic.gaps:
            recommendations = []
            for gap in diagnostic.gaps:
                stage = {
                    "unknown": "尚未评估",
                    "new": "新接触/待确认",
                    "learning": "正在学习",
                    "weak": "需要巩固",
                }.get(gap.learner_stage, "尚未评估")
                recommendations.append(f"- **{gap.prerequisite_name}**（{stage}）")
            diagnostic_note = (
                "\n\n---\n\n### 前置知识查缺补漏建议\n\n"
                "以下前置知识按先修 DAG 的学习顺序列出。这里的“尚未评估”表示当前没有足够的学习状态证据，并不代表你一定不会。\n\n"
                + "\n".join(recommendations)
                + f"\n\n<sub>课程图：{diagnostic.graph_id} v{diagnostic.graph_version}；"
                f"基于答案版本 {diagnostic.answer_revision_id[:8]} 的非阻断建议。</sub>"
            )

    final_answer = answer.rstrip() + diagnostic_note
    if diagnostic_note:
        revision = AnswerRevision(
            user_id=user,
            task_id=task,
            revision_id=revision_id,
            dossier_version=dossier.version,
            answer_text=final_answer,
            claims=claims,
            knowledge_point_ids=graph_targets,
            parent_revision_id=parent_revision_id,
            revision_number=revision_number,
        )
        revision.validate_against(dossier)

    # Persist only the final answer revision, then its coverage ledger.
    revision_repository.save(revision)
    entries = tuple(
        CoverageEntry(
            user_id=user,
            task_id=task,
            answer_revision_id=revision.revision_id,
            dossier_version=dossier.version,
            requirement_id=requirement.id,
            outcome=CoverageOutcome.UNRESOLVED,
            rationale="尚未完成针对本条要求的逐项核查；不能将此记录视为已满足。",
        )
        for requirement in dossier.requirements
    )
    ledger = CoverageLedger(
        user_id=user,
        task_id=task,
        answer_revision_id=revision.revision_id,
        dossier_version=dossier.version,
        entries=entries,
    )
    ledger.validate_against(dossier, revision)
    coverage_repository.save(ledger)

    unresolved = sum(item.outcome is CoverageOutcome.UNRESOLVED for item in entries)
    return RuntimeEpistemicResult(
        dossier_version=dossier.version,
        answer_revision_id=revision.revision_id,
        revision_number=revision.revision_number,
        coverage_count=len(entries),
        coverage_unresolved_count=unresolved,
        prerequisite_diagnostic=diagnostic,
        prerequisite_note=diagnostic_note,
    )
