from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from coder import CoderAgent


BROKEN_SOURCE = '''from __future__ import annotations


def median(values):
    """Return the median of a non-empty numeric sequence."""
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle]
'''


CODER_REQUEST = """修复这个小型 Python 项目的 median 实现，并完成验证。

目标：
1. 保持公开函数名 median(values) 不变。
2. 接受任意可排序的数字序列；不得修改调用者传入的原序列。
3. 奇数长度返回排序后的中间值。
4. 偶数长度返回中间两个值的算术平均值。
5. 空序列继续抛出 ValueError。
6. 自己创建 pytest 测试，至少覆盖：
   - 奇数长度
   - 偶数长度
   - 乱序输入
   - 输入序列不会被修改
   - 空序列
7. 修改后运行 pytest，并确保测试通过。
8. 不要修改与这个任务无关的文件。

这是一个刻意带 bug 的 demo。你需要自己读取 median.py，定位问题，修改实现，创建测试，再运行测试和 Goal 验证。
"""


def prepare_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / "median.py").write_text(BROKEN_SOURCE, encoding="utf-8")
    (workspace / "tests").mkdir()


def on_event(event: dict) -> None:
    event_type = event.get("type")

    if event_type == "started":
        print("▶ Coder 开始执行")
        return

    if event_type == "step":
        step = event.get("step")
        action = getattr(step, "action", "?")
        step_id = getattr(step, "step_id", "?")
        success = getattr(step, "success", True)
        marker = "✓" if success else "!"
        print(f"  {marker} STEP {step_id}: {action}")
        return

    if event_type == "finished":
        state = event.get("state")
        print("■ Coder 已完成并通过事件回调")
        summary = getattr(state, "summary", "")
        if summary:
            print(summary)
        return

    if event_type == "error":
        state = event.get("state")
        print(f"✗ Coder error: {getattr(state, 'error', 'unknown error')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end demo for the browser-backed Coder Agent."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="指定一个不存在的独立 workspace；不传则创建临时 workspace 并保留。",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="开启 Coder / BrowserModel 调试输出。",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="使用临时 workspace 时，结束后删除它。",
    )
    args = parser.parse_args()

    temporary = args.workspace is None
    if temporary:
        workspace = Path(tempfile.mkdtemp(prefix="study-agent-coder-demo-"))
        # mkdtemp 已经创建目录，因此直接填充。
        (workspace / "median.py").write_text(BROKEN_SOURCE, encoding="utf-8")
        (workspace / "tests").mkdir()
    else:
        workspace = args.workspace.resolve()
        prepare_workspace(workspace)

    print(f"Workspace: {workspace}")
    print("Challenge: 修复 median，并由 Coder 自己创建、运行 pytest。")
    print("提示：这个 demo 故意把偶数长度 median 写错了。")

    try:
        agent = CoderAgent(
            workspace=workspace,
            debug_mode=args.debug,
            reuse_chat=True,
            min_send_interval_seconds=5.0,
            max_runtime_seconds=1800,
        )
        result = agent.run(CODER_REQUEST, event_hook=on_event)

        print("\n=== RESULT ===")
        print(
            f"finished={result.finished}, "
            f"verified={result.goal_verified}, "
            f"steps={result.step_count}, "
            f"modified={sorted(result.modified_files)}, "
            f"created_tests={sorted(result.created_tests)}, "
            f"test_runs={result.metrics.get('test_runs', 0)}, "
            f"chat_resets={result.metrics.get('chat_resets', 0)}"
        )

        if result.last_test_result:
            test = result.last_test_result
            print(
                f"last_test: kind={test.get('kind')}, "
                f"passed={test.get('passed')}, "
                f"returncode={test.get('returncode')}"
            )

        if result.summary:
            print("\n=== SUMMARY ===")
            print(result.summary)

        if result.error:
            print(f"\n=== ERROR ===\n{result.error}")

        print("\n=== FILES ===")
        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                print(path.relative_to(workspace))

        return 0 if result.finished and result.goal_verified else 1
    finally:
        if temporary and args.cleanup:
            shutil.rmtree(workspace, ignore_errors=True)
        elif temporary:
            print(f"\nTemporary workspace kept at: {workspace}")


if __name__ == "__main__":
    raise SystemExit(main())
