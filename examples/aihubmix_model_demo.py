import argparse
import json
import urllib.error
import urllib.request

from config.loader import load_config, setup_debug


def get_aihubmix_provider(config: dict) -> dict:
    providers = config.get("providers", {})
    for name, provider in providers.items():
        if str(name).strip().lower() == "aihubmix":
            if not isinstance(provider, dict):
                raise RuntimeError("AIHubMix provider 配置格式异常。")
            return provider
    raise KeyError("找不到 AIHubMix provider 配置。")


def list_aihubmix_models(config: dict) -> list[dict]:
    provider = get_aihubmix_provider(config)

    base_url = str(provider["base_url"]).rstrip("/")
    url = f"{base_url}/models"

    headers = {
        "Accept": "application/json",
        **dict(provider.get("headers", {})),
    }

    if "Authorization" not in headers and provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    request = urllib.request.Request(
        url,
        method="GET",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"AIHubMix /models HTTP {exc.code}: {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"AIHubMix /models 请求失败：{type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"AIHubMix /models 返回不是有效 JSON：{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("AIHubMix /models 返回格式异常：不是 JSON 对象。")

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError("AIHubMix /models 返回格式异常：data 不是列表。")

    return [
        item for item in data
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    ]


def print_models(models: list[dict]) -> None:
    print("\n========== AIHubMix Models ==========")
    print(f"共返回 {len(models)} 个模型。\n")

    for index, item in enumerate(models, 1):
        model_id = str(item.get("id", "")).strip()
        owned_by = str(item.get("owned_by", "")).strip()

        print(f"{index:>4}. {model_id}", end="")
        if owned_by:
            print(f"    owner={owned_by}")
        else:
            print()


def print_free_models_once(models: list[dict]) -> None:
    free_models = [
        str(item.get("id", "")).strip()
        for item in models
        if str(item.get("id", "")).strip().lower().endswith("-free")
    ]

    print("\n========== AIHubMix *-free Models ==========")

    if not free_models:
        print("没有找到以 -free 结尾的模型。")
        return

    print(f"共 {len(free_models)} 个：")
    for index, model_id in enumerate(free_models, 1):
        print(f"{index:>4}. {model_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check current AIHubMix models"
    )
    args = parser.parse_args()
    _ = args

    try:
        config = load_config()
        setup_debug(config)
        models = list_aihubmix_models(config)
    except Exception as exc:
        print("状态   : ❌ 失败")
        print(f"错误   : {type(exc).__name__}: {exc}")
        return

    print("状态   : ✅ 成功")
    print("来源   : AIHubMix /v1/models")

    # 先输出全部模型，再统一输出一次 -free 后缀模型。
    print_models(models)
    print_free_models_once(models)

    print("\n========== Check Finished ==========")


if __name__ == "__main__":
    main()
