from __future__ import annotations

import sys

from coder import CoderAgent


def main() -> int:
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        print('用法：python examples/coder_agent.py "修复 xxx，并添加 pytest 回归测试"')
        return 2

    result = CoderAgent().run(request)
    print(f"Coder finished={result.finished}, verified={result.goal_verified}, steps={result.step_count}")
    if result.modified_files:
        print("Modified files:")
        for path in sorted(result.modified_files):
            print("  " + path)
    if result.last_test_result:
        print("Last test result:")
        print(result.last_test_result)
    if result.error:
        print(result.error)
        return 1
    return 0 if result.finished and result.goal_verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
