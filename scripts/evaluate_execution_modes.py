"""Evaluate StudyAgent under adaptive or forced execution modes.

Input JSONL:
  {"id": "q1", "question": "什么是 Transformer？"}
  {"id": "q2", "question": "2025 年 Transformer attention 有哪些优化？"}

Run:
  python scripts/evaluate_execution_modes.py dataset.jsonl --mode adaptive --output results.jsonl
  python scripts/evaluate_execution_modes.py dataset.jsonl --mode all --output results.jsonl

The script records execution mode, answer, errors, timing, step count,
evidence count, tool calls, and routing metadata. It does not judge answer
quality automatically; reference answers may be stored in the input dataset
and scored separately.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from config.loader import load_config
from examples.study_agent_api import build_agent


MODES = {
    "adaptive": None,
    "chat": "chat",
    "knowledge_direct": "knowledge_direct",
    "knowledge_agent": "knowledge_agent",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} 不是有效 JSON：{exc}") from exc

        if isinstance(item, str):
            item = {"question": item}
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_no} 顶层必须是字符串或对象。")

        question = str(item.get("question", "")).strip()
        if not question:
            raise ValueError(f"{path}:{line_no} 缺少 question。")

        case_id = str(item.get("id", line_no)).strip() or str(line_no)
        cases.append({"id": case_id, "question": question, "reference": item.get("reference")})
    return cases


def run_condition(config: dict[str, Any], cases: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    agent = build_agent(config)
    agent.execution_mode_override = MODES[mode]

    rows: list[dict[str, Any]] = []
    for case in cases:
        started = time.perf_counter()
        state = agent.run(case["question"])
        wall_ms = round((time.perf_counter() - started) * 1000, 2)

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "reference": case["reference"],
            "condition": mode,
            "execution_mode": state.execution_mode,
            "analyzer_execution_mode": state.metrics.get("analyzer_execution_mode"),
            "route_overridden": state.metrics.get("route_overridden", False),
            "route_fallback": state.metrics.get("route_fallback", False),
            "task_type": state.task_type,
            "domain": state.domain,
            "finished": state.finished,
            "answer": state.final_answer,
            "error": state.error,
            "step_count": state.step_count,
            "evidence_count": len(state.evidence),
            "tool_calls": state.metrics.get("tool_calls", 0),
            "reasoner_steps": state.metrics.get("reasoner_steps", 0),
            "task_analysis_ms": state.metrics.get("task_analysis_ms"),
            "execution_ms": state.metrics.get("execution_ms"),
            "total_ms": state.metrics.get("total_ms"),
            "wall_ms": wall_ms,
        })
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StudyAgent execution routing.")
    parser.add_argument("dataset", help="JSONL dataset containing question fields.")
    parser.add_argument(
        "--mode",
        choices=list(MODES) + ["all"],
        default="adaptive",
        help="adaptive uses the analyzer route; other modes force a controlled condition.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if not cases:
        raise ValueError("数据集为空。")

    config = load_config()
    modes = list(MODES) if args.mode == "all" else [args.mode]

    all_rows: list[dict[str, Any]] = []
    for mode in modes:
        print(f"[evaluate] condition={mode}, cases={len(cases)}", flush=True)
        all_rows.extend(run_condition(config, cases, mode))

    write_jsonl(args.output, all_rows)

    print(f"[evaluate] wrote {len(all_rows)} rows → {args.output}", flush=True)
    print("[evaluate] modes:", ", ".join(modes), flush=True)


if __name__ == "__main__":
    main()
