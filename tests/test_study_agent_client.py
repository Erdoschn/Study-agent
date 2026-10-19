from __future__ import annotations

import unittest
from unittest.mock import Mock

from integrations.study_agent_client import (
    StudyAgentClient,
    StudyAgentClientError,
    validate_api_url,
)


class StudyAgentClientTests(unittest.TestCase):
    def test_rejects_non_loopback_http(self):
        with self.assertRaises(ValueError):
            validate_api_url("http://example.com/v1/chat/completions")

    def test_rejects_credentials_and_query(self):
        urls = (
            "http://user:pass@127.0.0.1/v1/chat/completions",
            "http://127.0.0.1/v1/chat/completions?token=secret",
        )
        for url in urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                validate_api_url(url)

    def test_accepts_loopback_http_and_remote_https(self):
        self.assertTrue(validate_api_url("http://127.0.0.1:8001/v1/chat/completions"))
        self.assertTrue(validate_api_url("https://study.example.com/v1/chat/completions"))

    def test_ask_normalizes_chat_completion_response(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        client._request_json = Mock(return_value={
            "model": "test-model",
            "choices": [{"message": {"content": "  The answer is 42.  "}}],
        })
        result = client.ask("Explain the answer", username="Alice")
        self.assertEqual(result, {"answer": "The answer is 42.", "model": "test-model"})
        payload = client._request_json.call_args.kwargs["payload"]
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["messages"][0]["role"], "user")
        self.assertEqual(payload["username"], "alice")

    def test_ask_includes_context_as_reference(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        client._request_json = Mock(return_value={
            "choices": [{"message": {"content": "answer"}}],
        })
        client.ask("question", username="alice", context="Relevant notes")
        prompt = client._request_json.call_args.kwargs["payload"]["messages"][0]["content"]
        self.assertIn("Relevant notes", prompt)
        self.assertIn("question", prompt)
        self.assertIn("<reference_context>", prompt)

    def test_ask_rejects_invalid_username(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        for username in ("", "../alice", "bad name", "x" * 65):
            with self.subTest(username=username), self.assertRaises(ValueError):
                client.ask("question", username=username)

    def test_ask_rejects_empty_question(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        with self.assertRaises(ValueError):
            client.ask("   ", username="alice")

    def test_ask_rejects_malformed_response(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        client._request_json = Mock(return_value={"choices": []})
        with self.assertRaises(StudyAgentClientError):
            client.ask("question", username="alice")

    def test_health_uses_same_origin_health_path(self):
        client = StudyAgentClient(api_url="http://127.0.0.1:8001/v1/chat/completions")
        client._request_json = Mock(return_value={"status": "ok"})
        self.assertEqual(client.health(), {"status": "ok"})
        self.assertEqual(
            client._request_json.call_args.args[0],
            "http://127.0.0.1:8001/health",
        )


if __name__ == "__main__":
    unittest.main()
