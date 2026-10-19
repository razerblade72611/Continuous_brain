from types import SimpleNamespace

import agent_core.llm_client as module
from agent_core.llm_client import SyncLLMClient


class FakeResponse:
    def __init__(self, status_code, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    @property
    def is_error(self):
        return self.status_code >= 400

    def json(self):
        return self._data


def test_retries_without_response_format_on_400(monkeypatch):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        if len(calls) == 1:
            return FakeResponse(400, text='{"error":{"message":"response_format rejected"}}')
        return FakeResponse(
            200,
            data={"choices": [{"message": {"content": '{"ok": true}'}}]},
        )

    monkeypatch.setattr(module.httpx, "post", fake_post)
    client = SyncLLMClient("http://localhost:8080/v1", "qwen")
    out = client.chat(
        "system", "user",
        response_format={"type": "json_object"},
    )
    assert out == '{"ok": true}'
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_error_contains_server_body(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(400, text='{"error":{"message":"bad model alias"}}')

    monkeypatch.setattr(module.httpx, "post", fake_post)
    client = SyncLLMClient("http://localhost:8080/v1", "qwen")
    try:
        client.chat("system", "user")
    except RuntimeError as exc:
        assert "bad model alias" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
