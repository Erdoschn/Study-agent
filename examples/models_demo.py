from config.loader import get_model_config, load_config
from core.reasoner import OpenAICompatibleClient


def main():
    try:
        config = load_config()

        model_name = "gpt-4.1-free"

        model_config = get_model_config(
            config,
            model_name,
        )

    except Exception as exc:
        print(
            f"❌ 配置读取失败："
            f"{type(exc).__name__}: {exc}"
        )
        return

    print("========== API Test ==========")
    print(f"Model: {model_name}")
    print(f"Provider: {model_config['provider']}")
    print(f"Base URL: {model_config['base_url']}")
    print("API Key: 已读取")
    print()

    client = OpenAICompatibleClient(
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
        model=model_config["model"],
    )

    try:
        result = client.generate(
            system_prompt=(
                "你是 API 连通性测试助手。"
                "请确认你能正确接收请求并返回响应。"
            ),
            user_prompt=(
                "请回复：API 测试成功"
            ),
        )

        print("✅ API 调用成功")
        print()
        print("模型返回：")
        print(result)

    except Exception as exc:
        print("❌ API 调用失败")
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()