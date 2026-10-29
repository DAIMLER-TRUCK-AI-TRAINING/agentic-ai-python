"""
Smoke tests for the RAG chatbot — no API key required.

Run: uv run pytest

These tests verify wiring (routes, schemas, in-memory state, persona metadata,
provider settings, guardrails) and the document-chunking logic. A full RAG turn
and a full index build also run, with LangChain's fake chat and embedding models
standing in for the real ones, so nothing calls an API or downloads a model.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app import main, rag
from app.guardrails import check_input
from app.llm import DEFAULT_LOCAL_EMBEDDING, _build_embeddings, get_chat_model
from app.main import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_models(monkeypatch):
    """Each test builds its models and its index from its own environment."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("ST_MODEL", raising=False)
    monkeypatch.setattr(rag, "_collection", None)
    get_chat_model.cache_clear()
    _build_embeddings.cache_clear()
    yield
    get_chat_model.cache_clear()
    _build_embeddings.cache_clear()


def _events(body: str) -> list[dict]:
    return [json.loads(part[6:]) for part in body.split("\n\n") if part.startswith("data: ")]


def test_healthz_reports_models() -> None:
    res = client.get("/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert "provider" in body and "model" in body
    assert body["embeddings"] == {"provider": "local", "model": DEFAULT_LOCAL_EMBEDDING}


def test_healthz_reports_configured_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "azure-anthropic")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_DEPLOYMENT", "claude-opus-5-5")
    body = client.get("/healthz").json()
    assert body["provider"] == "azure-anthropic"
    assert body["model"] == "claude-opus-5-5"


def test_personas_list_includes_rag_flag() -> None:
    res = client.get("/api/personas")
    assert res.status_code == 200
    personas = res.json()
    keys = {p["key"] for p in personas}
    assert {"sage", "nova", "axel", "aurora", "atlas"} <= keys
    by_key = {p["key"]: p for p in personas}
    # Sage is the RAG-grounded persona; the others answer from the model directly.
    assert by_key["sage"]["rag"] is True
    assert by_key["axel"]["rag"] is False
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


def test_input_guardrail_blocks_emissions_tampering() -> None:
    sid = client.post("/api/sessions", json={}).json()["session_id"]
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
    assert "Unsupported LLM_PROVIDER" in res.text


def test_unknown_embedding_provider_streams_retrieval_error(monkeypatch) -> None:
    monkeypatch.setattr(main, "get_chat_model", lambda: GenericFakeChatModel(messages=iter([])))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere")
    sid = client.post("/api/sessions", json={}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/chat", json={"message": "Hi"})
    assert "Retrieval failed: Unsupported EMBEDDING_PROVIDER" in res.text


def test_rag_turn_streams_sources_and_keeps_context_out_of_history(monkeypatch) -> None:
    """A whole RAG turn with a fake model: sources first, then the reply, and the
    retrieved context is not stored in the session (even with full history)."""
    reply = "Dosing pressure stayed below 3.5 bar [1]."
    monkeypatch.setattr(
        main, "get_chat_model",
        lambda: GenericFakeChatModel(messages=iter([AIMessage(content=reply)])),
    )
    hit = {"text": "...below 3.5 bar...", "source": "KB-0610.md", "title": "What the code means", "score": 0.66}
    monkeypatch.setattr(rag, "retrieve", lambda question, top_k=3: [hit])

    sid = client.post("/api/sessions", json={"persona": "sage", "strategy": "full"}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/chat", json={"message": "What does fault code FC-ADB-0412 mean?"})
    events = _events(res.text)

    assert [e["event"] for e in events][:2] == ["sources", "meta"]
    assert events[0]["hits"][0]["source"] == "KB-0610.md"
    assert events[1]["messages_in_prompt"] == 3  # persona, retrieved context, question
    assert "".join(e["delta"] for e in events if e["event"] == "delta") == reply
    assert events[-1]["event"] == "done"
    assert client.get(f"/api/sessions/{sid}").json()["history_messages"] == 3  # system, user, assistant


def test_non_rag_persona_skips_retrieval(monkeypatch) -> None:
    monkeypatch.setattr(
        main, "get_chat_model",
        lambda: GenericFakeChatModel(messages=iter([AIMessage(content="Check it in DiagLink.")])),
    )
    monkeypatch.setattr(rag, "retrieve", lambda *a, **k: pytest.fail("Axel must not retrieve"))
    sid = client.post("/api/sessions", json={"persona": "axel"}).json()["session_id"]
    res = client.post(f"/api/sessions/{sid}/chat", json={"message": "What does fault code FC-ADB-0412 mean?"})
    assert '"sources"' not in res.text
    assert '"done"' in res.text


def test_chunk_markdown_splits_on_headings() -> None:
    """Chunking is pure (no API/key needed) — one chunk per heading section."""
    doc = "# Title\nintro\n\n## A\nalpha text\n\n## B\nbeta text\n"
    chunks = rag.chunk_markdown(doc, "doc.md")
    titles = [c["title"] for c in chunks]
    assert titles == ["Title", "A", "B"]
    assert all(c["source"] == "doc.md" for c in chunks)
    assert "alpha text" in chunks[1]["text"]


def test_chunk_markdown_drops_front_matter_and_keeps_doc_title() -> None:
    doc = (
        "---\ndoc_id: KB-0610\ntitle: \"Fault code FC-ADB-0412: AdBlue dosing pressure low\"\n---\n\n"
        "# Fault code FC-ADB-0412: AdBlue dosing pressure low\n\n## What the code means\n\nBelow 3.5 bar.\n"
    )
    chunks = rag.chunk_markdown(doc, "KB-0610.md")
    # The title alone has no text, so it gets no chunk of its own
    assert [c["title"] for c in chunks] == ["What the code means"]
    assert not any("doc_id:" in c["text"] for c in chunks)
    # A section alone does not name the fault code; the document title does
    assert chunks[0]["text"].startswith("Fault code FC-ADB-0412")


def test_data_directory_has_knowledge_base() -> None:
    """The knowledge base ships with the demo so retrieval has something to find."""
    docs = rag._doc_paths()
    assert len(docs) == 20
    chunks = rag._load_chunks()
    assert len(chunks) >= len(docs)


def test_index_builds_once_per_embedding_model(tmp_path, monkeypatch) -> None:
    """Index the real documents with fake embeddings, in a temporary Chroma store."""
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path)
    monkeypatch.setattr(rag, "get_embeddings", lambda: DeterministicFakeEmbedding(size=16))

    hits = rag.retrieve("What does fault code FC-ADB-0412 mean?", top_k=3)
    assert len(hits) == 3
    assert all(h["source"].endswith(".md") and -1 <= h["score"] <= 1 for h in hits)

    col = rag.get_collection()
    assert col.name == "kb-local-paraphrase-multilingual-minilm-l12-v2"
    assert col.count() == len(rag._load_chunks())

    # A restart with the same documents reuses the index instead of embedding again
    monkeypatch.setattr(rag, "_collection", None)
    monkeypatch.setattr(rag, "get_embeddings", lambda: pytest.fail("must not re-embed"))
    assert rag.get_collection().count() == col.count()
