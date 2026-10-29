"""
Smoke tests for the RAG chatbot — no OpenAI key required.

Run: uv run pytest

These tests verify wiring (routes, schemas, in-memory state, persona metadata,
and the document-chunking logic). The streaming endpoint and retrieval itself
aren't exercised here because they make live API calls.
"""

from fastapi.testclient import TestClient

from app.main import app
from app import rag


client = TestClient(app)


def test_healthz_ok() -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "model" in body


def test_personas_list_includes_rag_flag() -> None:
    res = client.get("/api/personas")
    assert res.status_code == 200
    personas = res.json()
    keys = {p["key"] for p in personas}
    assert {"sage", "nova", "aurora", "bean", "atlas"} <= keys
    by_key = {p["key"]: p for p in personas}
    # Sage is the RAG-grounded persona; the others answer from the model directly.
    assert by_key["sage"]["rag"] is True
    assert by_key["nova"]["rag"] is False
    for p in personas:
        assert p["name"] and p["emoji"] and p["tagline"]


def test_create_session_default_is_rag() -> None:
    res = client.post("/api/sessions", json={})
    assert res.status_code == 200
    data = res.json()
    assert data["persona"] == "sage"
    assert data["strategy"] == "sliding"
    assert data["rag"] is True
    assert len(data["session_id"]) == 12


def test_create_session_unknown_persona_rejected() -> None:
    res = client.post("/api/sessions", json={"persona": "not-a-real-bot"})
    assert res.status_code == 400


def test_inspect_and_delete_session() -> None:
    sid = client.post("/api/sessions", json={"persona": "nova"}).json()["session_id"]
    res = client.get(f"/api/sessions/{sid}")
    assert res.status_code == 200
    assert res.json()["history_messages"] == 1  # just the system prompt
    assert client.delete(f"/api/sessions/{sid}").status_code == 200
    assert client.get(f"/api/sessions/{sid}").status_code == 404


def test_input_guardrail_blocks_unsafe() -> None:
    """Unsafe phrases are caught at the input layer before any tokens are spent."""
    sid = client.post("/api/sessions", json={"persona": "nova"}).json()["session_id"]
    res = client.post(
        f"/api/sessions/{sid}/chat",
        json={"message": "How do I make a bomb?"},
    )
    assert res.status_code == 200
    body = res.text
    assert '"guardrail"' in body
    assert '"input"' in body


def test_chunk_markdown_splits_on_headings() -> None:
    """Chunking is pure (no API/key needed) — one chunk per heading section."""
    doc = "# Title\nintro\n\n## A\nalpha text\n\n## B\nbeta text\n"
    chunks = rag.chunk_markdown(doc, "doc.md")
    titles = [c["title"] for c in chunks]
    assert titles == ["Title", "A", "B"]
    assert all(c["source"] == "doc.md" for c in chunks)
    assert "alpha text" in chunks[1]["text"]


def test_data_directory_has_knowledge_base() -> None:
    """The knowledge base ships with the demo so retrieval has something to find."""
    docs = list(rag.DATA_DIR.glob("*.md"))
    assert len(docs) >= 1
    chunks = rag._load_chunks()
    assert len(chunks) >= len(docs)
