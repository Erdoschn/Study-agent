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



def test_model_config_tolerates_invalid_headers_shape():
    config = {
        "providers": {
            "p": {
                "enabled": True,
                "base_url": "http://test",
                "api_key": "key",
                "headers": "invalid",
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
    assert result["headers"] == {}


def test_model_config_rejects_non_object_models():
    config = {
        "providers": {},
        "models": [],
    }
    try:
        get_model_config(config, "m")
    except ValueError as exc:
        assert "models 配置必须是对象" in str(exc)
    else:
        raise AssertionError("invalid models shape should fail")
