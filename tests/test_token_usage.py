from __future__ import annotations

import json
from pathlib import Path

from core.reasoner import OpenAICompatibleClient
from core.token_usage import TokenUsageRecorder, summarize_usage_log


def test_recorder_tracks_deepseek_cache_usage_without_prompt_content(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )

    recorder.record(
        provider="deepseek",
        model="deepseek-flash",
        status="success",
        usage={
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": 60,
        },
        elapsed_seconds=0.25,
    )

    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert event["input_tokens"] == 100
    assert event["output_tokens"] == 25
    assert event["total_tokens"] == 125
    assert event["cache_hit_tokens"] == 40
    assert event["cache_miss_tokens"] == 60
    assert event["cache_hit_rate"] == 0.4
    assert "prompt" not in event
    assert "completion" not in event


def test_missing_usage_is_reported_as_unavailable_not_zero_tokens(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )
    recorder.record(
        provider="provider",
        model="model",
        status="success",
        usage=None,
        elapsed_seconds=0.1,
    )

    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert event["usage_available"] is False
    assert event["input_tokens"] is None
    assert event["cache_stats_available"] is False
    assert event["cache_hit_rate"] is None


def test_recorder_calculates_cost_only_from_configured_prices(tmp_path):
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(tmp_path / "usage.jsonl"), "console_log": False},
        project_root=tmp_path,
    )
    event = recorder.record(
        provider="deepseek",
        model="configured-model",
        status="success",
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_cache_hit_tokens": 600,
            "prompt_cache_miss_tokens": 400,
        },
        elapsed_seconds=1.0,
        pricing_usd_per_million={
            "input": 0.15,
            "cache_hit_input": 0.003,
            "output": 0.28,
        },
    )
    assert event is not None
    expected = (400 * 0.15 + 600 * 0.003 + 200 * 0.28) / 1_000_000
    assert event["estimated_cost_usd"] == round(expected, 10)

    without_prices = recorder.record(
        provider="deepseek",
        model="unpriced-model",
        status="success",
        usage={"prompt_tokens": 100, "completion_tokens": 10},
        elapsed_seconds=1.0,
    )
    assert without_prices is not None
    assert without_prices["estimated_cost_usd"] is None


def test_summary_uses_token_weighted_cache_hit_rate_and_groups_models(tmp_path):
    log_path = tmp_path / "usage.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )
    for model, total, hit in (
        ("short", 100, 10),
        ("long", 900, 450),
    ):
        recorder.record(
            provider="deepseek",
            model=model,
            status="success",
            usage={
                "prompt_tokens": total,
                "completion_tokens": 1,
                "prompt_cache_hit_tokens": hit,
                "prompt_cache_miss_tokens": total - hit,
            },
            elapsed_seconds=0.1,
        )

    report = summarize_usage_log(log_path, backup_count=0)
    totals = report["totals"]
    assert totals["input_tokens"] == 1000
    assert totals["cache_hit_tokens"] == 460
    assert totals["cache_hit_rate"] == 0.46
    assert set(report["by_provider_model"]) == {"deepseek / short", "deepseek / long"}


def test_recorder_disabled_does_not_create_a_file(tmp_path):
    log_path = tmp_path / "disabled.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": False, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )
    assert recorder.record(
        provider="provider",
        model="model",
        status="success",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        elapsed_seconds=0,
    ) is None
    assert not log_path.exists()


def test_openai_compatible_client_records_usage_without_storing_payload(monkeypatch, tmp_path):
    log_path = tmp_path / "client-usage.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": "model answer"}}],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 12,
                    "total_tokens": 62,
                    "prompt_cache_hit_tokens": 20,
                    "prompt_cache_miss_tokens": 30,
                },
            }).encode("utf-8")

    monkeypatch.setattr("core.reasoner.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OpenAICompatibleClient(
        base_url="https://api.example.invalid/v1",
        api_key="not-recorded",
        model="test-model",
        provider_name="test-provider",
        usage_recorder=recorder,
    )

    assert client.generate("private system prompt", "private user prompt") == "model answer"
    event_text = log_path.read_text(encoding="utf-8")
    event = json.loads(event_text.strip())
    assert event["provider"] == "test-provider"
    assert event["model"] == "test-model"
    assert "private system prompt" not in event_text
    assert "private user prompt" not in event_text
    assert "not-recorded" not in event_text


def test_load_config_merges_observability_defaults_and_local_overrides(monkeypatch, tmp_path):
    import config.loader as loader

    providers_path = tmp_path / "providers.json"
    providers_path.write_text(
        json.dumps({
            "providers": {},
            "models": {},
            "observability": {"token_usage": {"console_log": False}},
        }),
        encoding="utf-8",
    )
    observability_path = tmp_path / "observability.json"
    observability_path.write_text(
        json.dumps({
            "token_usage": {
                "enabled": True,
                "path": "data/custom.jsonl",
                "console_log": True,
                "backup_count": 2,
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIG_PATH", providers_path)
    monkeypatch.setattr(loader, "OBSERVABILITY_CONFIG_PATH", observability_path)
    monkeypatch.setattr(loader, "_merge_model_profiles", lambda _config: None)
    monkeypatch.setattr(loader, "_refresh_provider_models", lambda _config: None)

    config = loader.load_config()
    assert config["observability"]["token_usage"] == {
        "enabled": True,
        "path": "data/custom.jsonl",
        "console_log": False,
        "backup_count": 2,
    }



def test_invalid_completion_envelope_is_not_recorded_as_success(monkeypatch, tmp_path):
    log_path = tmp_path / "invalid-response.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }).encode("utf-8")

    monkeypatch.setattr("core.reasoner.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    client = OpenAICompatibleClient(
        base_url="https://api.example.invalid/v1",
        api_key="not-recorded",
        model="test-model",
        usage_recorder=recorder,
    )

    import pytest
    with pytest.raises(RuntimeError, match="格式异常"):
        client.generate("", "test prompt")

    event = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert event["status"] == "invalid_response"
    assert event["usage_available"] is False


def test_invalid_or_inconsistent_cache_counters_are_not_reported(tmp_path):
    log_path = tmp_path / "bad-cache-metrics.jsonl"
    recorder = TokenUsageRecorder(
        {"enabled": True, "path": str(log_path), "console_log": False},
        project_root=tmp_path,
    )
    event = recorder.record(
        provider="provider",
        model="model",
        status="success",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 7,
            "prompt_cache_miss_tokens": 8,
        },
        elapsed_seconds=0.1,
    )
    assert event is not None
    assert event["cache_stats_available"] is False
    assert event["cache_hit_tokens"] is None
    assert event["cache_miss_tokens"] is None
    assert event["cache_hit_rate"] is None


def test_report_settings_include_local_provider_overrides(monkeypatch, tmp_path):
    import config.loader as loader
    from scripts.token_usage_report import _load_settings

    providers_path = tmp_path / "providers.json"
    providers_path.write_text(
        json.dumps({
            "providers": {},
            "models": {},
            "observability": {
                "token_usage": {
                    "path": "data/local-usage.jsonl",
                    "backup_count": 1,
                    "console_log": False,
                }
            },
        }),
        encoding="utf-8",
    )
    observability_path = tmp_path / "observability.json"
    observability_path.write_text(
        json.dumps({
            "token_usage": {
                "enabled": True,
                "path": "data/default-usage.jsonl",
                "backup_count": 3,
                "console_log": True,
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(loader, "CONFIG_PATH", providers_path)
    monkeypatch.setattr(loader, "OBSERVABILITY_CONFIG_PATH", observability_path)

    settings = _load_settings()
    assert settings["enabled"] is True
    assert settings["path"] == "data/local-usage.jsonl"
    assert settings["backup_count"] == 1
    assert settings["console_log"] is False


def test_non_integer_token_counters_are_rejected():
    from core.token_usage import _non_negative_int

    assert _non_negative_int(4) == 4
    assert _non_negative_int("4") == 4
    assert _non_negative_int(4.0) is None
    assert _non_negative_int(4.5) is None
    assert _non_negative_int(True) is None
