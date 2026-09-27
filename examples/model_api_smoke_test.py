"""Minimal LLM API smoke test.

Each model receives exactly the same prompt: "1+1=?"
The script uses the project's existing model configuration and OpenAI-compatible
client, so it tests the real configured endpoint/API key/model ID.

Examples:
    python examples/model_api_smoke_test.py
    python examples/model_api_smoke_test.py --models llama33-70b qwen3-32b
    python examples/model_api_smoke_test.py --models agents-a1-free coding-glm-5.3-free
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import get_model_config, load_config, setup_debug
from core.reasoner import ModelTimeoutError, OpenAICompatibleClient


DEFAULT_MODELS = [
    # Models that failed in the latest Study Agent debug run.
    "gpt-4o-free",
    "gpt-5.5-free",
    "gpt-4.1-free",
    "gemini-3.8-flash-free",
    "gemini-3.7-flash-free",
    "gemini-3.6-flash-free",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "deepseek-r1-distill",
    "intern-s2-free",
    "minimax-m2.7-free",
    "qwen3.6-plus-preview-free",
    # Models that succeeded in the same debug run.
    "llama33-70b",
    "qwen3-32b",
    "agents-a1-free",
    "coding-glm-5.3-free",
    "nemotron-3-super-120b-a12b-free",
    "nemotron-3-ultra-550b-a55b-free",
]


PROMPT = "1+1=?"


def test_model(config: dict, model_name: str) -> tuple[str, float, str]:
    started = time.perf_counter()

    try:
        model = get_model_config(config, model_name)
        client = OpenAICompatibleClient(
            base_url=model["base_url"],
            api_key=model["api_key"],
            model=model["model"],
            timeout=model["timeout"],
            headers=model["headers"],
        )
        answer = client.generate(
            "You are a minimal API connectivity test. Answer the user's question directly.",
            PROMPT,
            json_mode=False,
        )
        elapsed = time.perf_counter() - started
        answer = " ".join(str(answer).split())
        return "PASS", elapsed, answer[:200]
    except ModelTimeoutError as exc:
        elapsed = time.perf_counter() - started
        return "TIMEOUT", elapsed, str(exc)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return "FAIL", elapsed, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用最简单的 1+1=? 检查配置中的 LLM API 是否可调用。"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="要测试的模型名；默认测试最近 Debug 中涉及的模型。",
    )
    args = parser.parse_args()

    try:
        config = load_config()
        setup_debug(config)
    except Exception as exc:
        print(f"配置加载失败：{type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print("=" * 78)
    print("Study Agent · Minimal Model API Smoke Test")
    print(f'Prompt: "{PROMPT}"')
    print(f"Models: {len(args.models)}")
    print("=" * 78)

    results = []

    for index, model_name in enumerate(args.models, 1):
        print(f"\n[{index}/{len(args.models)}] {model_name}")
        status, elapsed, detail = test_model(config, model_name)
        results.append((model_name, status, elapsed, detail))

        if status == "PASS":
            print(f"  PASS    {elapsed:6.2f}s    answer={detail}")
        elif status == "TIMEOUT":
            print(f"  TIMEOUT {elapsed:6.2f}s    {detail}")
        else:
            print(f"  FAIL    {elapsed:6.2f}s    {detail}")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)

    for model_name, status, elapsed, detail in results:
        print(f"{status:7} {elapsed:7.2f}s  {model_name}")

    passed = sum(status == "PASS" for _, status, _, _ in results)
    failed = sum(status == "FAIL" for _, status, _, _ in results)
    timed_out = sum(status == "TIMEOUT" for _, status, _, _ in results)

    print("\n" + "-" * 78)
    print(f"PASS={passed}  FAIL={failed}  TIMEOUT={timed_out}  TOTAL={len(results)}")
    print("-" * 78)


if __name__ == "__main__":
    main()
