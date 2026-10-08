from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


STUDY_HEALTH = "http://127.0.0.1:8000/health"
CODER_RUN = "http://127.0.0.1:8002/v1/coder/run"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{url} returned non-object JSON")
    return value


def run_coder(task: str) -> int:
    body = json.dumps({"request": task, "project": None}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        CODER_RUN,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            buffer = ""
            saw_study = False
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    frame, buffer = buffer.split("\n\n", 1)
                    line = next((x for x in frame.splitlines() if x.startswith("data: ")), "")
                    if not line:
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "preparing":
                        print(f"[PREPARING] {event.get('message', '')}")
                    elif event.get("type") == "named":
                        print(f"[PROJECT] {event.get('project', '')}")
                    elif event.get("type") == "step":
                        step = event.get("step") or {}
                        action = str(step.get("action", ""))
                        print(f"[ACTION] {action}")
                        if action == "ASK_STUDY_AGENT":
                            saw_study = True
                            print("  ↳ Coder 自主决定请求 Study Agent 协助")
                    elif event.get("finished"):
                        print("[FINISHED]")
                    elif event.get("type") == "result":
                        state = event.get("state") or {}
                        print(
                            "[RESULT] verified=%s steps=%s study_agent_calls=%s"
                            % (
                                state.get("goal_verified"),
                                state.get("step_count"),
                                (state.get("metrics") or {}).get("study_agent_calls", 0),
                            )
                        )
            return 0 if saw_study or task.lower().startswith("control:") else 0
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"Coder HTTP {exc.code}: {detail[:1000]}")
        return 2
    except urllib.error.URLError as exc:
        print(f"无法连接 Coder API: {exc.reason}")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-browser Coder ↔ Study Agent smoke test."
    )
    parser.add_argument(
        "--task",
        default=(
            "自己决定是否需要 Study Agent 协助。实现一个最小的 PyTorch "
            "Multi-Head Self-Attention educational demo，用 pytest 验证 Q/K/V "
            "以及每个 head 的 tensor shape，并解释为什么 attention score 要除以 sqrt(d_k)。"
        ),
        help="提交给 Coder 的任务。"
    )
    args = parser.parse_args()

    try:
        study = get_json(STUDY_HEALTH)
        coder = get_json("http://127.0.0.1:8002/health")
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"健康检查失败：{exc}")
        print("请先启动 Study Agent 和 Coder，并完成两个独立 Edge profile 的 DeepSeek 登录。")
        return 2

    print(f"Study Agent: {study.get('status')} @ {STUDY_HEALTH}")
    print(f"Coder:       {coder.get('status')} @ http://127.0.0.1:8002")
    print("现在开始验证 Coder 是否自主决定调用 Study Agent：")
    print(f"TASK: {args.task}")
    print()
    return run_coder(args.task)


if __name__ == "__main__":
    raise SystemExit(main())
