import argparse

from config.loader import load_config, setup_debug
from core import ModelClientFactory, ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test HKUST OpenAI-compatible models"
    )
    parser.add_argument(
        "--model",
        help="只测试指定的模型配置名；默认测试 HKUST_GEN_AI 下全部免费模型。",
    )
    args = parser.parse_args()

    config = load_config()
    setup_debug(config)

    registry = ModelRegistry(config)
    factory = ModelClientFactory(config)

    hkust_models = [
        model
        for model in registry.models.values()
        if model.provider == "HKUST_GEN_AI"
        and model.enabled
        and not model.paid
    ]

    if args.model:
        hkust_models = [
            model for model in hkust_models
            if model.name == args.model
        ]

    if not hkust_models:
        print("没有找到可测试的 HKUST_GEN_AI 免费模型。")
        print("请检查 providers.json 中是否已添加 models，并设置 paid=false。")
        return

    print("\n========== HKUST Model Demo ==========\n")
    print(f"检测到 {len(hkust_models)} 个 HKUST 免费模型。")
    print("本测试只调用 paid=false 的模型，不会调用付费模型。\n")

    system_prompt = (
        "你是模型连通性测试助手。"
        "只需简短回答用户的问题。"
    )
    user_prompt = "请回答：1+1等于多少？只输出结果和一句简短说明。"

    for model in hkust_models:
        print("----------------------------------------")
        print(f"配置名 : {model.name}")
        print(f"Provider: {model.provider}")
        print(f"模型ID : {model.model}")

        try:
            client = factory.create(model)
            answer = client.generate(
                system_prompt,
                user_prompt,
            )

            registry.record_success(model.name)
            print("状态   : ✅ 成功")
            print(f"回答   : {answer}")

        except Exception as exc:
            registry.record_failure(model.name)
            print("状态   : ❌ 失败")
            print(f"错误   : {type(exc).__name__}: {exc}")

    print("\n========== Test Finished ==========")


if __name__ == "__main__":
    main()
