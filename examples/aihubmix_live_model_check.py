"""Query AIHubMix /v1/models and optionally probe every returned model.

This separates two questions:
1. Is the model currently listed by AIHubMix?
2. Can the configured API actually serve it?

Run:
    python examples/aihubmix_live_model_check.py
    python examples/aihubmix_live_model_check.py --probe
    python examples/aihubmix_live_model_check.py --free-only --probe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.loader import get_model_config, load_config, setup_debug


PROMPT = "1+1=?"
PROBE_TIMEOUT = 20


def get_provider(config: dict) -> dict:
    providers = config.get("providers", {})
    for name, provider in providers.items():
        if str(name).strip().lower() == "aihubmix":
            if isinstance(provider, dict):
                return provider
    raise KeyError("找不到 AIHubMix provider 配置。")


def list_models(config: dict) -> list[dict]:
    provider = get_provider(config)
    base_url = str(provider["base_url"]).rstrip("/")
    headers = {
        "Accept": "application/json",
        **dict(provider.get("headers", {})),
    }
    if "Authorization" not in headers and provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    request = urllib.request.Request(
        f"{base_url}/models",
        method="GET",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    data = payload.get("data", []) if isinstance(payload, dict) else []
    return [
        item for item in data
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]


def probe_model(config: dict, model_name: str) -> tuple[str, float, str]:
    started = time.perf_counter()
    try:
        model = get_model_config(config, model_name)
    except Exception as exc:
        return "NOT_CONFIGURED", time.perf_counter() - started, str(exc)

    payload = {
        "model": model["model"],
        "messages": [
            {"role": "user", "content": PROMPT},
        ],
        "temperature": 0,
    }
    request = urllib.request.Request(
        f"{str(model['base_url']).rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {model['api_key']}",
            **model.get("headers", {}),
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as response:
            raw = response.read()
        data = json.loads(raw.decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        return "PASS", time.perf_counter() - started, " ".join(str(answer).split())[:120]
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return "FAIL", time.perf_counter() - started, f"HTTP {exc.code}: {body[:300]}"
    except TimeoutError:
        return "TIMEOUT", time.perf_counter() - started, f">{PROBE_TIMEOUT}s"
    except Exception as exc:
        return "FAIL", time.perf_counter() - started, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="实时查询 AIHubMix 模型列表，并可用 1+1=? 探测实际可调用性。"
    )
    parser.add_argument("--probe", action="store_true", help="对模型逐个发送最小请求。")
    parser.add_argument("--free-only", action="store_true", help="只显示模型 ID 以 -free 结尾的模型。")
    args = parser.parse_args()

    config = load_config()
    setup_debug(config)

    try:
        models = list_models(config)
    except Exception as exc:
        print(f"获取 AIHubMix /v1/models 失败: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    if args.free_only:
        models = [
            item for item in models
            if str(item["id"]).strip().lower().endswith("-free")
        ]

    print("=" * 88)
    print("AIHubMix Live Model Check")
    print("API: GET /v1/models")
    print(f"Prompt: {PROMPT}")
    print(f"Listed models: {len(models)}")
    print("=" * 88)

    if not args.probe:
        print("\n仅查询服务端当前模型列表，不代表每个模型都一定可以成功调用。\n")
        for i, item in enumerate(models, 1):
            model_id = str(item["id"]).strip()
            owner = str(item.get("owned_by", "")).strip()
            print(f"{i:>4}. {model_id}" + (f"    owner={owner}" if owner else ""))
        print("\n提示：加 --probe 会进一步逐个调用 1+1=?。")
        return

    results = []
    for i, item in enumerate(models, 1):
        model_id = str(item["id"]).strip()
        print(f"\n[{i}/{len(models)}] {model_id}")
        status, elapsed, detail = probe_model(config, model_id)
        results.append((model_id, status, elapsed, detail))
        print(f"  {status:<13} {elapsed:6.2f}s  {detail}")

    print("\n" + "=" * 88)
    print("Probe Summary")
    print("=" * 88)
    for model_id, status, elapsed, _ in results:
        print(f"{status:<13} {elapsed:7.2f}s  {model_id}")

    print("\n说明：LISTED = AIHubMix 当前列出；PASS = 你的当前配置实际调用成功；")
    print("FAIL/TIMEOUT = 本次探测失败，不等于永久下线。")


if __name__ == "__main__":
    main()
