from __future__ import annotations

import pytest

from core.file_ingestion import extract_attachments_to_text


def test_api_attachment_extraction_includes_text_and_untrusted_context():
    result = extract_attachments_to_text(
        "请总结文件",
        [{"name": "notes.txt", "buffer": "学习目标是理解高斯定律。".encode("utf-8")}],
    )
    assert "请总结文件" in result
    assert "学习目标是理解高斯定律。" in result
    assert "不可信" in result
    assert "notes.txt" in result


def test_api_attachment_extraction_supports_json_and_csv():
    json_result = extract_attachments_to_text(
        "检查配置",
        [{"name": "settings.json", "buffer": b'{"enabled":true,"count":3}'}],
    )
    csv_result = extract_attachments_to_text(
        "分析表格",
        [{"name": "scores.csv", "buffer": b"name,score\\nAlice,95\\n"}],
    )
    assert '"enabled": true' in json_result
    assert "Alice,95" in csv_result


def test_api_attachment_rejects_images_with_clear_actionable_error():
    with pytest.raises(ValueError, match="请切换到 Web Model"):
        extract_attachments_to_text(
            "识别图片",
            [{"name": "diagram.png", "buffer": b"not really an image"}],
        )


def test_api_attachment_rejects_unsupported_binary():
    with pytest.raises(ValueError, match="暂不支持解析"):
        extract_attachments_to_text(
            "阅读压缩包",
            [{"name": "archive.zip", "buffer": b"PK\\x03\\x04"}],
        )
