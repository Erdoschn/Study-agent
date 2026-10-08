from __future__ import annotations

import argparse

from core.__debug__ import debug
from coder import CoderAgent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the browser-backed Coder Agent."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="保留 Edge 窗口并开启详细调试输出。",
    )
    parser.add_argument(
        "request",
        nargs="+",
        help="要交给 Coder Agent 的任务。",
    )
    args = parser.parse_args()

    debug.set_enabled(args.debug)
    request = " ".join(args.request).strip()
    result = CoderAgent(debug_mode=args.debug).run(request)

    status = (
        f"Coder finished={result.finished}, "
        f"verified={result.goal_verified}, "
        f"steps={result.step_count}, "
        f"modified={len(result.modified_files)}, "
        f"tests={result.metrics.get('test_runs', 0)}"
    )
    print(status)
    if result.error:
        print(f"Coder error: {result.error}")
        return 1
    return 0 if result.finished and result.goal_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
