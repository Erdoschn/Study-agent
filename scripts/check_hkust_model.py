"""Check HKUST GenAI model availability without printing API credentials.

Usage:
    python scripts/check_hkust_model.py
    python scripts/check_hkust_model.py --smoke-test hkust-qwen3-32b

The smoke test sends a tiny prompt and may consume a small amount of credit.
The official OpenAI-compatible API documentation does not document a balance endpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "providers.json"
PORTAL_URL = "https://genai.ust.hk"


def _load_hkust_config() -> tuple[str, dict[str, str], dict[str, str], dict[str, Any]]:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"未找到 {CONFIG_PATH}。请先创建本机 config/providers.json，勿提交 API Key。"
        )
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 providers.json：{type(exc).__name__}: {exc}") from exc
    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    models = config.get("models", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict) or not isinstance(models, dict):
        raise ValueError("providers.json 的 providers/models 必须是对象。")

    matches = [
        (name, item) for name, item in providers.items()
        if isinstance(item, dict)
        and ("hkust" in str(name).casefold()
             or "hkust" in str(item.get("base_url", "")).casefold())
    ]
    if not matches:
        raise ValueError("providers.json 中没有找到 HKUST Provider。")
    provider_name, provider = next(
        ((name, item) for name, item in matches if item.get("enabled", True)),
        matches[0],
    )
    base_url = str(provider.get("base_url", "")).rstrip("/")
    if not base_url:
        raise ValueError("HKUST Provider 缺少 base_url。")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    raw_headers = provider.get("headers", {})
    if isinstance(raw_headers, dict):
        headers.update({str(k): str(v) for k, v in raw_headers.items()})
    if "Authorization" not in headers and provider.get("api_key"):
        headers["Authorization"] = "Bearer " + str(provider["api_key"])
    configured = {
        str(name): item for name, item in models.items()
        if isinstance(item, dict) and str(item.get("provider", "")) == str(provider_name)
        and item.get("enabled", True)
    }
    return base_url, headers, configured, provider


def _request(url: str, headers: dict[str, str], body: dict[str, Any] | None = None) -> tuple[int, dict[str, str], bytes]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, dict(response.headers.items()), response.read(2 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        # Read only a bounded response for diagnostics; never echo request headers.
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"网络或连接错误：{type(exc).__name__}: {exc}") from None


def _report_credit_headers(headers: dict[str, str]) -> None:
    found = False
    for name, value in headers.items():
        lowered = name.casefold()
        if any(term in lowered for term in ("credit", "balance", "quota-remaining", "remaining-quota")):
            print(f"额度响应头 {name}: {value}")
            found = True
    if not found:
        print(
            "剩余 credit：HKUST OpenAI-compatible API 响应未提供可识别的余额字段。"
            f"请登录 {PORTAL_URL} 查看账户余额；不要用 token usage 估算余额。"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 HKUST GenAI API 与模型可用性")
    parser.add_argument(
        "--smoke-test",
        metavar="MODEL",
        help="对指定的本地模型配置名或上游 model ID 发起最小聊天请求（可能消耗少量 credit）",
    )
    args = parser.parse_args()
    try:
        base_url, headers, configured, _provider = _load_hkust_config()
        print(f"HKUST API endpoint: {base_url}")
        status, response_headers, raw = _request(base_url + "/models", headers)
        payload = json.loads(raw.decode("utf-8"))
        live_ids = {
            str(item.get("id", "")).strip()
            for item in payload.get("data", [])
            if isinstance(item, dict) and str(item.get("id", "")).strip()
        }
        print(f"Provider connection: OK (HTTP {status})")
        if not configured:
            print("本地没有启用的 HKUST 模型配置。")
        for name, item in configured.items():
            upstream_id = str(item.get("model", name)).strip()
            print(f"{name}: {'AVAILABLE in /models' if upstream_id in live_ids else 'NOT FOUND in /models'} (upstream={upstream_id})")
        if args.smoke_test:
            chosen = args.smoke_test.strip()
            entry = configured.get(chosen)
            if entry is None:
                entry = next(
                    (item for item in configured.values()
                     if str(item.get("model", "")).strip() == chosen),
                    None,
                )
            if not isinstance(entry, dict):
                print(f"SMOKE TEST FAILED: {chosen} 不在本地启用的 HKUST 模型配置中。")
                return 2
            upstream_id = str(entry.get("model", chosen)).strip()
            if upstream_id not in live_ids:
                print(f"SMOKE TEST FAILED: {upstream_id} 不在 HKUST /models 返回列表中。")
                return 2
            try:
                smoke_headers = dict(headers)
                smoke_headers["Content-Type"] = "application/json"
                code, result_headers, result_raw = _request(
                    base_url + "/chat/completions",
                    smoke_headers,
                    {
                        "model": upstream_id,
                        "messages": [{"role": "user", "content": "Reply with OK."}],
                        "max_tokens": 1,
                        "stream": False,
                    },
                )
                result = json.loads(result_raw.decode("utf-8"))
                choices = result.get("choices", [])
                if code < 200 or code >= 300 or not choices:
                    print(f"SMOKE TEST FAILED: HTTP {code}, response did not contain choices.")
                    _report_credit_headers(result_headers)
                    return 2
                print(f"SMOKE TEST: OK (HTTP {code}, model={upstream_id})")
                _report_credit_headers(result_headers)
            except (RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
                print(f"SMOKE TEST FAILED: {exc}")
                return 2
        else:
            print("提示：/models 只能确认模型出现在目录中；要验证实际推理，请追加 --smoke-test <模型配置名>。")
            _report_credit_headers(response_headers)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"HKUST CHECK FAILED: {exc}", file=sys.stderr)
        print("请确认本机 providers.json 中 HKUST Provider 已启用、API Key 正确且网络可访问。", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
