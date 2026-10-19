from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.browser_environment import write_browser_environment_report
from core.web_model import BrowserModel
from core import web_model as web_model_module


def test_environment_report_persists_local_installation_id_and_redacts_paths(tmp_path):
    profile = tmp_path / "private-profile-name"
    profile.mkdir()
    report = write_browser_environment_report(
        user_data_dir=str(profile),
        browser_channel="msedge",
        target_url="https://example.invalid/private/path?token=must-not-leak",
        browser_version="136.0",
        chat_input_ready=True,
        project_root=tmp_path,
    )
    report_path = tmp_path / "data" / "browser-environment" / "report.json"
    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert str(uuid.UUID(report["installation_id"])) == report["installation_id"]
    assert saved["installation_id"] == report["installation_id"]
    assert saved["profile"]["label"] == "private-profile-name"
    assert saved["target_origin"] == "https://example.invalid"
    assert saved["browser"]["version"] == "136.0"
    assert saved["chat_input_ready"] is True
    serialized = json.dumps(saved)
    assert str(tmp_path) not in serialized
    assert "private/path" not in serialized
    assert "must-not-leak" not in serialized
    assert "cookie" not in serialized.casefold()
    assert report_path.stat().st_mode & 0o077 == 0


def test_settings_signature_matches_for_same_settings_but_runtime_signature_is_separate(tmp_path):
    profile_a = tmp_path / "machine-a" / ".study-agent-browser"
    profile_b = tmp_path / "machine-b" / ".study-agent-browser"
    profile_a.mkdir(parents=True)
    profile_b.mkdir(parents=True)

    report_a = write_browser_environment_report(
        user_data_dir=str(profile_a),
        browser_channel="msedge",
        target_url="https://example.invalid/",
        browser_version="136.0",
        project_root=tmp_path / "machine-a",
    )
    report_b = write_browser_environment_report(
        user_data_dir=str(profile_b),
        browser_channel="msedge",
        target_url="https://example.invalid/",
        browser_version="136.0",
        project_root=tmp_path / "machine-b",
    )
    other_settings = write_browser_environment_report(
        user_data_dir=str(profile_b),
        browser_channel="chromium",
        target_url="https://example.invalid/",
        browser_version="136.0",
        project_root=tmp_path / "machine-b",
    )

    assert report_a["settings_signature"] == report_b["settings_signature"]
    assert report_a["installation_id"] != report_b["installation_id"]
    assert report_a["settings_signature"] != other_settings["settings_signature"]


def test_invalid_existing_installation_id_fails_closed(tmp_path):
    directory = tmp_path / "data" / "browser-environment"
    directory.mkdir(parents=True)
    (directory / "installation_id").write_text("not-a-uuid\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="installation ID"):
        write_browser_environment_report(
            user_data_dir=str(tmp_path / ".study-agent-browser"),
            browser_channel="msedge",
            target_url="https://example.invalid/",
            project_root=tmp_path,
        )


def test_browser_model_diagnostics_can_be_disabled(monkeypatch):
    model = BrowserModel(environment_config={"diagnostics_enabled": False})
    model._context = SimpleNamespace(browser=None)

    def fail_if_called(**_kwargs):
        raise AssertionError("disabled diagnostics must not write a report")

    monkeypatch.setattr(web_model_module, "write_browser_environment_report", fail_if_called)
    model._record_environment_diagnostics()


def test_browser_model_rejects_non_boolean_diagnostics_setting():
    with pytest.raises(ValueError, match="diagnostics_enabled"):
        BrowserModel(environment_config={"diagnostics_enabled": "false"})
