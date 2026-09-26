from config.loader import get_model_config


def test_model_provider_timeout_is_propagated():
    config = {
        "providers": {
            "p": {
                "enabled": True,
                "base_url": "http://test",
                "api_key": "key",
                "timeout": 17,
            }
        },
        "models": {
            "m": {
                "provider": "p",
                "model": "model",
                "enabled": True,
                "paid": False,
            }
        },
    }

    result = get_model_config(config, "m")
    assert result["timeout"] == 17
