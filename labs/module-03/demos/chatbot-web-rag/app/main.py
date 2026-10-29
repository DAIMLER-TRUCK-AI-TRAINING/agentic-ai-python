"""
Module 3 RAG chatbot — FastAPI backend on LangChain.

This is the Module 2 streaming chatbot enhanced with Retrieval-Augmented
Generation. When the active persona is RAG-enabled (Sage), each turn embeds the
user's question, retrieves the top-k most relevant chunks from a Chroma vector
store, and injects them as grounding context so the model answers from the
documents instead of its parametric memory.

Endpoints:
- GET  /api/personas               → metadata for the persona picker (incl. `rag`)
- POST /api/sessions               → create a new session
- POST /api/sessions/{sid}/chat    → send a message, stream the reply (SSE)
- DELETE /api/sessions/{sid}       → end a session
- GET  /api/sessions/{sid}         → inspect session state (debugging)
- GET  /healthz                    → liveness probe, with the configured models

The chat and embedding models come from app/llm.py: LLM_PROVIDER picks OpenAI,
Anthropic Claude, Azure OpenAI or Claude on Microsoft Foundry, EMBEDDING_PROVIDER
picks local, OpenAI or Azure OpenAI embeddings, and nothing below changes.

Static files (the chat UI) are served from /static.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anthropic
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import rag
from .guardrails import SAFE_DECLINE, check_input, check_output
from .llm import ProviderConfigError, describe, get_chat_model
from .memory import (
    Strategy,
    apply_strategy,
    estimate_tokens,
    messages_token_estimate,
    should_rotate_summary,
)
from .personas import PERSONAS, get_system_prompt, persona_meta, persona_uses_rag
from .sessions import store

load_dotenv()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

app = FastAPI(title="Module 3 RAG Chatbot")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateSessionReq(BaseModel):
    persona: str = Field(default="sage")
    strategy: Strategy = Field(default="sliding")


class ChatReq(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, **describe()}


@app.get("/api/personas")
def list_personas() -> list[dict]:
    return persona_meta()


@app.post("/api/sessions")
def create_session(req: CreateSessionReq) -> dict:
    if req.persona not in PERSONAS:
        raise HTTPException(400, f"unknown persona: {req.persona}")
    sess = store.create(req.persona, req.strategy, get_system_prompt(req.persona))
    return {
        "session_id": sess.sid,
        "persona": req.persona,
        "strategy": req.strategy,
        "rag": persona_uses_rag(req.persona),
    }


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    sess = store.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found")
    return {
        "session_id": sess.sid,
        "persona": sess.persona_key,
        "strategy": sess.strategy,
        "turn_count": sess.turn_count,
        "history_messages": len(sess.history),
        "total_input_tokens": sess.total_input_tokens,
        "total_output_tokens": sess.total_output_tokens,
    }


@app.delete("/api/sessions/{sid}")
def end_session(sid: str) -> dict:
    store.delete(sid)
    return {"ended": sid}


@app.post("/api/sessions/{sid}/chat")
async def chat(sid: str, req: ChatReq) -> StreamingResponse:
    sess = store.get(sid)
    if sess is None:
        raise HTTPException(404, "session not found — start a new one")

    # ── Input guardrails ──
    guard = check_input(req.message)
    if not guard["ok"]:
        def safe_stream():
            yield _sse({"event": "guardrail", "stage": "input", "reason": guard["reason"]})
            for word in SAFE_DECLINE.split():
                yield _sse({"event": "delta", "delta": word + " "})
            yield _sse({"event": "done", "input_tokens": 0, "output_tokens": 0})
        return StreamingResponse(safe_stream(), media_type="text/event-stream")

    user_msg = guard["redacted"]
    sess.history.append({"role": "user", "content": user_msg})

    use_rag = persona_uses_rag(sess.persona_key)

    async def event_stream():
        # Errors become an `error` event: once streaming starts, the HTTP status is already sent
        try:
            model = get_chat_model()

            # ── Memory strategy ──
            if sess.strategy == "summary" and should_rotate_summary(sess.history, summarize_every_turns=4):
                await _rotate_summary(sess, model)

            # A copy: the retrieved context is for this turn only, not for the session history
            prompt_messages = list(apply_strategy(sess.history, sess.strategy, window_pairs=4))
        except Exception as exc:
            yield _sse({"event": "error", "error": _error_text(exc, "LLM")})
            return

        # ── Retrieval (RAG) — embed the question, pull top-k grounding chunks ──
        if use_rag:
            try:
                # In a worker thread: embedding and Chroma calls block
                hits = await run_in_threadpool(rag.retrieve, user_msg, RAG_TOP_K)
            except Exception as exc:
                yield _sse({"event": "error", "error": f"Retrieval failed: {_error_text(exc, 'Embedding')}"})
                return

            # Show the user exactly what grounded the answer.
            yield _sse({
                "event": "sources",
                "hits": [
                    {
                        "n": i,
                        "source": h["source"],
                        "title": h["title"],
                        "score": h["score"],
                        "preview": h["text"][:260],
                    }
                    for i, h in enumerate(hits, 1)
                ],
            })

            if hits:
                context_block = rag.build_context_block(hits)
                context_msg = {
                    "role": "system",
                    "content": (
                        "Use ONLY the following retrieved context to answer the "
                        "user's question. Cite the passages you use as [n]. If the "
                        "answer is not in the context, say you don't know.\n\n"
                        f"{context_block}"
                    ),
                }
                # Inject right after the persona system prompt.
                prompt_messages.insert(1, context_msg)

        in_tok_est = messages_token_estimate(prompt_messages)

        yield _sse({
            "event": "meta",
            "persona": sess.persona_key,
            "strategy": sess.strategy,
            "rag": use_rag,
            "messages_in_prompt": len(prompt_messages),
            "input_tokens_est": in_tok_est,
        })

        # ── LLM call ── LangChain takes the role/content dicts as they are; each
        # chunk's .text is the new text on every provider. Adding the chunks up
        # gives the full message, with usage_metadata from the last chunk.
        full = None
        try:
            async for chunk in model.astream(prompt_messages):
                full = chunk if full is None else full + chunk
                if chunk.text:
                    yield _sse({"event": "delta", "delta": chunk.text})
        except Exception as exc:
            yield _sse({"event": "error", "error": _error_text(exc, "LLM")})
            return

        reply = full.text if full is not None else ""
        # Real token counts when the provider reports them, otherwise the estimates
        usage = (full.usage_metadata if full is not None else None) or {}
        in_tok = usage.get("input_tokens") or in_tok_est
        out_tok = usage.get("output_tokens") or estimate_tokens(reply)

        # ── Output guardrails ──
        sys_prompt = get_system_prompt(sess.persona_key)
        og = check_output(reply, sys_prompt)
        if not og["ok"]:
            yield _sse({"event": "guardrail", "stage": "output", "reason": og["reason"]})
            reply = SAFE_DECLINE

        sess.history.append({"role": "assistant", "content": reply})
        sess.turn_count += 1
        sess.total_input_tokens += in_tok
        sess.total_output_tokens += out_tok

        print(
            f"[turn] sid={sess.sid} persona={sess.persona_key} "
            f"strategy={sess.strategy} rag={use_rag} in={in_tok} out={out_tok}"
        )

        yield _sse({
            "event": "done",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "turn_count": sess.turn_count,
            "total_input_tokens": sess.total_input_tokens,
            "total_output_tokens": sess.total_output_tokens,
        })

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _error_text(exc: Exception, kind: str) -> str:
    """A short message for the `error` event; `kind` is 'LLM' or 'Embedding'."""
    if isinstance(exc, ProviderConfigError):
        return str(exc)
    if isinstance(exc, (openai.APIStatusError, anthropic.APIStatusError)):
        # 401: wrong key · 404: wrong model or deployment name · 429: rate limit
        return f"{kind} provider error {exc.status_code}: {exc.message}"
    if isinstance(exc, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return f"Could not reach the {kind.lower()} provider: check the endpoint or resource name"
    return str(exc)


async def _rotate_summary(sess, model) -> None:
    """
    Replace the oldest mid-conversation turns with a model-generated summary.
    Keeps the system prompt and the last 4 messages verbatim.
    """
    keep_recent = 4
    rest = [m for m in sess.history if m["role"] != "system"]
    if len(rest) <= keep_recent:
        return
    middle = rest[: -keep_recent]
    recent = rest[-keep_recent:]

    body = "\n".join(f"{m['role']}: {m['content']}" for m in middle)
    resp = await model.ainvoke([
        {"role": "system", "content":
            "Summarize the conversation below in 4–6 bullet points. "
            "Preserve vehicle models, fault codes, part numbers, IDs, dates, decisions, "
            "and unresolved questions. This summary will replace the verbatim turns."},
        {"role": "user", "content": body},
    ])
    summary = resp.text

    system_msg = next(m for m in sess.history if m["role"] == "system")
    sess.history = [
        system_msg,
        {"role": "system", "content": f"[Earlier conversation summary]\n{summary}"},
        *recent,
    ]
