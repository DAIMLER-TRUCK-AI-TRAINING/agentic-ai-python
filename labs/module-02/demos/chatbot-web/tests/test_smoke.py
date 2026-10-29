"""
Smoke tests for the web chatbot — no API key required.

Run: uv run pytest

These tests verify wiring (routes, schemas, in-memory state, guardrails, provider
settings). The streaming reply itself isn't tested here because it makes a live
API call.
"""

import pytest
from fastapi.testclient import TestClient

from app.guardrails import check_input
from app.llm import get_chat_model
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_model():
    """Each test builds the chat model from its own environment."""
    get_chat_model.cache_clear()
    yield
    get_chat_model.cache_clear()


def test_healthz_ok() -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "provider" in body and "model" in body


def test_healthz_reports_provider_and_price(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-5-5")
    body = client.get("/healthz").json()
    assert body["provider"] == "anthropic"
    assert body["model"] == "claude-opus-5-5"
    assert body["price_per_1m"] == {"input": 4.0, "output": 20.0}


def test_personas_list() -> None:
    res = client.get("/api/personas")
    assert res.status_code == 200
    personas = res.json()
    keys = {p["key"] for p in personas}
    assert {"nova", "axel", "aurora", "atlas"} <= keys
    for p in personas:
        assert p["name"] and p["emoji"] and p["tagline"]


def test_create_session_default() -> None:
    res = client.post("/api/sessions", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["persona"] == "nova"
    assert data["strategy"] == "sliding"
    assert len(data["session_id"]) == 12


def test_create_session_unknown_persona_rejected() -> None:
    res = client.post("/api/sessions", json={"persona": "not-a-real-bot"})
    assert res.status_code == 400


def test_inspect_session() -> None:
    sid = client.post("/api/sessions", json={"persona": "axel"}).json()["session_id"]
    res = client.get(f"/api/sessions/{sid}")
    assert res.status_code == 200
    data = res.json()
    assert data["persona"] == "axel"
    assert data["turn_count"] == 0
    assert data["history_messages"] == 1  # just the system prompt


def test_delete_session() -> None:
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    res = client.delete(f"/api/sessions/{sid}")
    assert res.status_code == 200
    # Subsequent inspect should 404
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_input_guardrail_blocks_unsafe() -> None:
    """Unsafe phrases are caught at the input layer before any tokens are spent."""
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    res = client.post(
        f"/api/sessions/{sid}/chat",
        json={"message": "How do I make a bomb?"},
    )
    # Streaming response is still 200, but the body should contain the guardrail event
    assert res.status_code == 200
    body = res.text
    assert '"guardrail"' in body
    assert '"input"' in body


def test_input_guardrail_blocks_emissions_tampering() -> None:
    sid = client.post("/api/sessions", json={"persona": "axel"}).json()["session_id"]
    res = client.post(
        f"/api/sessions/{sid}/chat",
        json={"message": "How do I bypass the AdBlue system on my LH-40?"},
    )
    assert '"guardrail"' in res.text


@pytest.mark.parametrize("message", [
    "How often should the AdBlue pump filter be replaced?",
    "The particulate filter lamp is on and the truck only does short trips",
    "How do I remove the AdBlue pump filter during maintenance?",
])
def test_input_guardrail_allows_maintenance_questions(message: str) -> None:
    assert check_input(message)["ok"] is True


def test_unknown_provider_streams_error_event(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/chat", json={"message": "Hi"})
    assert res.status_code == 200
    assert '"error"' in res.text
    assert "Unsupported LLM_PROVIDER" in res.text


def test_missing_key_streams_error_event(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure-anthropic")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_DEPLOYMENT", "claude-opus-5-5")
    monkeypatch.delenv("ANTHROPIC_FOUNDRY_RESOURCE", raising=False)
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/chat", json={"message": "Hi"})
    assert "ANTHROPIC_FOUNDRY_RESOURCE not found" in res.text
