"""Print an aggregate report from Study Agent token-usage JSONL logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.token_usage import summarize_usage_log

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "observability.json"


def _load_settings() -> dict:
    if not DEFAULT_CONFIG.is_file():
        raise FileNotFoundError(f"找不到可观测性配置文件：{DEFAULT_CONFIG}")
    with DEFAULT_CONFIG.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("observability.json 顶层必须是 JSON 对象。")
    settings = data.get("token_usage", {})
    if not isinstance(settings, dict):
        raise ValueError("token_usage 配置必须是对象。")
    return settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Study Agent API token usage and cache hit-rate report")
    parser.add_argument("--input", type=Path, help="JSONL log path (overrides config)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()

    settings = _load_settings()
    configured_path = settings.get("path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("observability.json must configure token_usage.path")
    path = args.input or Path(configured_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    backup_count = settings.get("backup_count", 3)
    if isinstance(backup_count, bool) or not isinstance(backup_count, int) or backup_count < 0:
        raise ValueError("token_usage.backup_count must be a non-negative integer")

    report = summarize_usage_log(path, backup_count=backup_count)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    totals = report["totals"]

    def rate(value):
        return "n/a" if value is None else f"{value * 100:.1f}%"

    print(f"Log: {report['path']}")
    print(
        "ALL MODELS: "
        f"requests={totals['requests']} successful={totals['successful_requests']} "
        f"failed={totals['failed_requests']} usage_available={totals['usage_available_requests']} "
        f"input_tokens={totals['input_tokens']} output_tokens={totals['output_tokens']} "
        f"total_tokens={totals['total_tokens']} cache_hit_tokens={totals['cache_hit_tokens']} "
        f"cache_miss_tokens={totals['cache_miss_tokens']} "
        f"cache_hit_rate={rate(totals['cache_hit_rate'])}"
    )
    if totals["cost_estimate_requests"]:
        print(
            f"Estimated cost total: USD {totals['estimated_cost_usd']:.8f} "
            f"(priced requests={totals['cost_estimate_requests']})"
        )
    if report["malformed_lines"]:
        print(f"Warning: skipped {report['malformed_lines']} unreadable log line(s).")

    groups = report["by_provider_model"]
    if not groups:
        print("No token-usage events recorded yet.")
        return 0
    print()
    print("BY PROVIDER / MODEL")
    for name, item in sorted(groups.items()):
        print(
            f"{name}: requests={item['requests']} usage_available={item['usage_available_requests']} "
            f"input={item['input_tokens']} output={item['output_tokens']} "
            f"cache_hit={item['cache_hit_tokens']} cache_miss={item['cache_miss_tokens']} "
            f"cache_hit_rate={rate(item['cache_hit_rate'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
