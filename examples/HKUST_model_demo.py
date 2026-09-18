import argparse
import json
import urllib.error
import urllib.request

from config.loader import load_config, setup_debug
from core import ModelClientFactory, ModelRegistry


def list_hkust_models(config: dict) -> list[dict]:
    provider = config["providers"]["HKUST_GEN_AI"]
    base_url = provider["base_url"].rstrip("/")
    headers = {
        "Accept": "application/json",
        **dict(provider.get("headers", {})),
    }

    # HKUST 使用 api-key Header；如果配置中没有，则保留兼容性的
    # Authorization Bearer Header。
    if "api-key" not in headers and provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    request = urllib.request.Request(
        f"{base_url}/models",
        method="GET",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HKUST /models HTTP {exc.code}: {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"HKUST /models 请求失败：{type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"HKUST /models 返回不是有效 JSON：{exc}"
        ) from exc

    data = payload.get("data", [])
    if not isinstance(data, list):
        raise RuntimeError(
            "HKUST /models 返回格式异常：data 不是列表。"
        )

    return [
        item for item in data
        if isinstance(item, dict)
    ]


def get_hkust_balance(config: dict) -> dict:
    """查询 HKUST 账户余额；余额接口不是 OpenAI 标准接口，因此独立处理。"""
    provider = config["providers"]["HKUST_GEN_AI"]
    base_url = provider["base_url"].rstrip("/")
    headers = {
        "Accept": "application/json",
        **dict(provider.get("headers", {})),
    }
    if "api-key" not in headers and provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    request = urllib.request.Request(
        f"{base_url}/balance",
        method="GET",
        headers=headers,
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HKUST /balance HTTP {exc.code}: {body}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"HKUST /balance 请求失败：{type(exc).__name__}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"HKUST /balance 返回不是有效 JSON：{exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("HKUST /balance 返回格式异常：不是 JSON 对象。")
    return payload


def print_hkust_balance(config: dict) -> None:
    print("\n========== HKUST Balance ==========")
    try:
        balance = get_hkust_balance(config)
        print("状态   : ✅ 成功")
        print(f"余额   : {json.dumps(balance, ensure_ascii=False)}")
    except Exception as exc:
        print("状态   : ⚠️ 无法查询")
        print(f"原因   : {type(exc).__name__}: {exc}")
        print("说明   : HKUST 官方公开文档确认余额存在，但未公开 /balance 的接口说明。")


def test_configured_models(config: dict, only_model: str | None = None) -> None:
    registry = ModelRegistry(config)
    factory = ModelClientFactory(config)

    models = [
        model
        for model in registry.models.values()
        if model.provider == "HKUST_GEN_AI"
        and model.enabled
        and not model.paid
        and (only_model is None or model.name == only_model)
    ]

    print("\n========== Configured HKUST Free Models Test ==========\n")

    if not models:
        print("没有找到符合条件的 HKUST 免费模型。")
        return

    for model in models:
        print("----------------------------------------")
        print(f"配置名 : {model.name}")
        print(f"模型ID : {model.model}")

        try:
            answer = factory.create(model).generate(
                "你是模型连通性测试助手，只需简短回答。",
                "请回答：1+1等于多少？只输出结果和一句简短说明。",
            )
            registry.record_success(model.name)
            print("状态   : ✅ 成功")
            print(f"回答   : {answer}")
        except Exception as exc:
            registry.record_failure(model.name)
            print("状态   : ❌ 失败")
            print(f"错误   : {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and test HKUST OpenAI-compatible models"
    )
    parser.add_argument(
        "--model",
        help="只测试指定的 providers.json 模型配置名。",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="在列出 HKUST 可用模型后，再测试 providers.json 中配置的免费模型。",
    )
    args = parser.parse_args()

    config = load_config()
    setup_debug(config)

    # 普通模式也查询余额；--test 只额外执行模型连通性测试。
    print_hkust_balance(config)

    print("\n========== HKUST Model Discovery ==========\n")
    print("正在查询 HKUST /models ...")

    try:
        models = list_hkust_models(config)
    except Exception as exc:
        print(f"状态   : ❌ 失败")
        print(f"错误   : {type(exc).__name__}: {exc}")
        return

    print(f"状态   : ✅ 成功")
    print(f"HKUST 返回 {len(models)} 个可见模型。\n")

    if not models:
        print("HKUST 没有返回可用模型。")
    else:
        for index, item in enumerate(models, 1):
            model_id = item.get("id", "<unknown>")
            owned_by = item.get("owned_by", "")
            print(f"{index:>3}. {model_id}", end="")
            if owned_by:
                print(f"    owner={owned_by}")
            else:
                print()

    print("\n提示：这里列出的模型来自 HKUST 当前 API 的 /models 响应，")
    print("比直接猜模型 ID 更可靠。你的 Team 仍可能对具体模型施加额外权限。")

    if args.test:
        test_configured_models(config, args.model)

    print("\n========== Test Finished ==========")


if __name__ == "__main__":
    main()
