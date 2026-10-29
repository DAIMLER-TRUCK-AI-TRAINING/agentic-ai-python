# Request / Response Flow

Sequence diagrams for the web chatbot, from page load to a streamed reply. Each diagram follows the code as written, so you can read a diagram next to the file it names.

| Participant | File | Role |
| --- | --- | --- |
| Browser | `static/index.html` | Chat UI. Reads the reply with `fetch` and a stream reader (not `EventSource`, because the chat call is a `POST`). |
| FastAPI | `app/main.py` | Routes, and the order of steps in one chat turn. |
| guardrails | `app/guardrails.py` | `check_input` before the model, `check_output` after it. |
| SessionStore | `app/sessions.py` | In-memory `dict` of sessions. Each holds the history and the token counters. |
| memory | `app/memory.py` | Memory strategies and token estimates. |
| personas | `app/personas.py` | System prompts and the picker metadata. |
| llm.py / LangChain model | `app/llm.py` | `get_chat_model()` builds a `ChatOpenAI` or `ChatAnthropic` from `LLM_PROVIDER`. |
| LLM provider | — | OpenAI, Anthropic, Azure OpenAI, or Claude on Microsoft Foundry. |

Contents: [1. Page load](#1-page-load) · [2. Session lifecycle](#2-session-lifecycle) · [3. A chat turn](#3-a-chat-turn-happy-path) · [4. Input blocked](#4-input-guardrail-blocks-the-message) · [5. Errors](#5-provider-and-configuration-errors) · [6. Cancel](#6-cancel-mid-stream) · [7. Summary rotation](#7-running-summary-rotation) · [Behaviours worth knowing](#behaviours-worth-knowing)

---

## 1. Page load

Nothing here needs an API key. The model is not built until the first chat turn.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant B as Browser<br/>index.html
    participant API as FastAPI<br/>main.py
    participant L as llm.py
    participant PE as personas.py

    U->>B: Open http://localhost:8000
    B->>API: GET /
    API-->>B: index.html (FileResponse)
    Note over B: The script calls loadConfig() and loadPersonas()<br/>without awaiting, so both requests go out together

    par loadConfig()
        B->>API: GET /healthz
        API->>L: describe()
        L-->>API: provider, model, price_per_1m<br/>(or config_error, never a key)
        API-->>B: {ok, provider, model, price_per_1m}
        Note over B: Top-bar chip shows provider · model.<br/>price_per_1m feeds the cost estimate
    and loadPersonas()
        B->>API: GET /api/personas
        API->>PE: persona_meta()
        PE-->>API: key, name, emoji, tagline for each persona
        API-->>B: JSON list
        Note over B: Fills the persona picker, the persona card<br/>and the example prompts
    end

    Note over B,API: No session exists yet
```

---

## 2. Session lifecycle

A session is created lazily on the first message, and dropped when the user changes persona or memory strategy.

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant API as FastAPI<br/>main.py
    participant PE as personas.py
    participant S as SessionStore<br/>sessions.py

    rect rgba(100, 150, 255, 0.12)
    Note over B,S: Create: ensureSession() runs when SESSION.sid is null
    B->>API: POST /api/sessions {persona, strategy}
    API->>API: Pydantic checks the body<br/>(strategy must be full, sliding or summary, else 422)
    API->>API: persona in PERSONAS? (else 400)
    API->>PE: get_system_prompt(persona)
    PE-->>API: system prompt
    API->>S: create(persona, strategy, system_prompt)
    S-->>API: Session with a 12-character sid<br/>and history = [system prompt]
    API-->>B: {session_id, persona, strategy}
    Note over B: Keeps SESSION.sid in memory
    end

    rect rgba(255, 170, 60, 0.12)
    Note over B,S: Reset: the user changes persona or strategy, or clicks Reset
    B->>API: DELETE /api/sessions/{sid}
    API->>S: delete(sid)
    Note right of S: pop() with a default, so a missing sid is not an error
    API-->>B: {ended: sid}
    Note over B: SESSION.sid = null, chat cleared, stats back to 0.<br/>The next message creates a new session
    end
```

`GET /api/sessions/{sid}` is a debugging endpoint. It returns the persona, strategy, turn count, history length and token totals, or 404 for an unknown `sid`. The UI does not call it.

---

## 3. A chat turn (happy path)

The reply is a Server-Sent Events stream. The seven-layer labels match the comments in `main.py`.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant B as Browser<br/>index.html
    participant API as FastAPI<br/>chat()
    participant G as guardrails.py
    participant S as SessionStore
    participant M as memory.py
    participant LC as LangChain model<br/>llm.py
    participant P as LLM provider

    U->>B: Type a message, press Enter
    B->>B: ensureSession() (see section 2)<br/>show the user bubble and a typing-dots bot bubble<br/>disable the composer, show ⏹
    B->>API: POST /api/sessions/{sid}/chat {message}

    rect rgba(100, 150, 255, 0.12)
    Note over API,S: Request checks
    API->>API: Pydantic: message is 1 to 4000 characters (else 422)
    API->>S: get(sid)
    S-->>API: Session (also refreshes last_active)
    Note right of API: Unknown sid: 404 JSON, no stream
    end

    rect rgba(255, 170, 60, 0.12)
    Note over API,G: Layer 3: input guardrails
    API->>G: check_input(message)
    G-->>API: ok, with PII masked in "redacted"<br/>(card numbers, SSNs, API keys)
    API->>S: history.append(user, redacted text)
    end

    API-->>B: 200 text/event-stream
    Note over B: t0 starts when the headers arrive.<br/>TTFT is measured from here

    rect rgba(120, 200, 120, 0.12)
    Note over API,M: Layer 4: memory strategy, inside event_stream()
    API->>LC: get_chat_model()
    Note right of LC: Built from LLM_PROVIDER on the first call,<br/>then reused (lru_cache)
    opt strategy is summary and a rotation is due
        API->>LC: _rotate_summary(): one extra LLM call (see section 7)
    end
    API->>M: apply_strategy(history, strategy, window_pairs=4)
    M-->>API: prompt_messages
    API->>M: messages_token_estimate(prompt_messages)
    M-->>API: input_tokens_est
    end

    API-->>B: SSE meta {persona, strategy, messages_in_prompt, input_tokens_est}
    Note over B: Shows "↪ N messages · ~X in-tok · strategy"<br/>under the bot bubble

    rect rgba(190, 120, 230, 0.12)
    Note over API,P: Layer 5: LLM call, streamed
    API->>LC: astream(prompt_messages)
    LC->>P: Streaming request
    loop every chunk
        P-->>LC: token chunk
        LC-->>API: chunk
        API->>API: full = full + chunk
        API-->>B: SSE delta {delta: chunk.text}
        Note over B: First delta: TTFT = now - t0, typing dots removed.<br/>Text is appended with a blinking cursor
    end
    P-->>LC: last chunk carries usage
    end

    rect rgba(240, 100, 100, 0.12)
    Note over API,G: Layer 6: output guardrails
    API->>API: reply = full.text<br/>tokens from full.usage_metadata, else the estimates
    API->>G: check_output(reply, system_prompt)
    alt reply is clean
        G-->>API: ok
    else system prompt leaked into the reply
        G-->>API: not ok, reason "system-prompt-leak"
        API-->>B: SSE guardrail {stage: output, reason}
        Note over API: reply becomes SAFE_DECLINE,<br/>for the stored history only
    end
    API->>S: history.append(assistant, reply)<br/>turn_count += 1, token totals += in / out
    end

    rect rgba(150, 150, 150, 0.15)
    Note over API: Layer 7: telemetry
    API->>API: print("[turn] sid=... persona=... strategy=... in=... out=...")
    end

    API-->>B: SSE done {input_tokens, output_tokens, turn_count, totals}
    Note over B: Final bubble and the line<br/>"TTFT ... · total ... · N in / M out".<br/>Sidebar stats and cost update, composer re-enabled
```

### The SSE contract

Every event is one line, `data: {json}`, followed by a blank line. The client splits the byte stream on `\n\n`.

| `event` | Sent when | Client reaction |
| --- | --- | --- |
| `meta` | Once, before the model call | Shows the prompt size under the bubble |
| `delta` | Once per chunk | Appends `delta` to the bubble. The first one records TTFT |
| `guardrail` | Input blocked (before the deltas) or output blocked (after them) | Sets an "Input blocked" or "Output blocked" badge |
| `done` | Last event of a finished turn | Final bubble, timings, sidebar stats |
| `error` | Config or provider failure | Replaces the bubble with `Error: ...` |

---

## 4. Input guardrail blocks the message

The cheapest path. The model is never built and no tokens are spent. The decline text is streamed word by word so the UI behaves like a normal reply.

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant API as FastAPI<br/>chat()
    participant S as SessionStore
    participant G as guardrails.py

    B->>API: POST /api/sessions/{sid}/chat {message}
    API->>S: get(sid)
    S-->>API: Session
    API->>G: check_input(message)
    G-->>API: not ok, reason "unsafe"<br/>(a DENY_PATTERNS regex matched)

    Note over API: Return early. The history is not changed,<br/>get_chat_model() is never called

    API-->>B: 200 text/event-stream
    API-->>B: SSE guardrail {stage: input, reason: unsafe}
    loop each word of SAFE_DECLINE
        API-->>B: SSE delta {delta: "word "}
    end
    API-->>B: SSE done {input_tokens: 0, output_tokens: 0}
    Note over B: Bubble shows the "Input blocked (unsafe)" badge and the decline.<br/>done has no turn_count, so updateStats() leaves the sidebar alone
```

---

## 5. Provider and configuration errors

Once streaming has started the HTTP status is already sent, so a failure cannot become a 500. It becomes an `error` event instead.

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant API as FastAPI<br/>event_stream()
    participant S as SessionStore
    participant LC as llm.py / LangChain
    participant P as LLM provider

    B->>API: POST /api/sessions/{sid}/chat {message}
    API->>S: history.append(user)
    API-->>B: 200 text/event-stream

    alt LLM_PROVIDER unknown, or a variable it needs is missing
        API->>LC: get_chat_model()
        LC-->>API: raises ProviderConfigError
        API-->>B: SSE error {"OPENAI_API_KEY not found in environment variables!"}
    else the provider rejects the call
        API->>LC: astream(prompt_messages)
        LC->>P: Streaming request
        P-->>LC: HTTP 401 wrong key, 404 wrong model, 429 rate limit
        LC-->>API: raises APIStatusError
        API-->>B: SSE error {"LLM provider error 401: ..."}
    else the provider cannot be reached
        API->>LC: astream(prompt_messages)
        LC--xP: Connection fails
        LC-->>API: raises APIConnectionError
        API-->>B: SSE error {"Could not reach the LLM provider: ..."}
    end

    Note over API: Return. No assistant message is stored<br/>and the turn counters do not change
    Note over B: Bubble text becomes "Error: ...".<br/>There is no done event, so the stream just ends and the composer is re-enabled
```

A provider error from `astream` arrives after the `meta` event, and can arrive after some `delta` events. The error replaces whatever text was already on screen.

---

## 6. Cancel mid-stream

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant B as Browser
    participant API as FastAPI<br/>event_stream()
    participant S as SessionStore
    participant P as LLM provider

    P-->>API: chunks keep arriving
    API-->>B: SSE delta ...
    U->>B: Click ⏹
    B->>B: cancelStream(): AbortController.abort()
    B--xAPI: Connection closed, fetch rejects with AbortError

    Note over API,P: Starlette stops the response. The generator is cancelled<br/>at its next await, and the provider stream goes down with it
    Note over API,S: The code after the loop never runs:<br/>no assistant message, turn_count and totals unchanged, no [turn] log

    Note over B: Cursor removed, "[cancelled]" added,<br/>composer re-enabled. No done event, so the sidebar is not updated
```

---

## 7. Running-summary rotation

An extra, non-streaming model call that runs before the `meta` event when the strategy is `summary`. The user waits for it, so it adds to time-to-first-token on that turn.

```mermaid
sequenceDiagram
    autonumber
    participant API as FastAPI<br/>event_stream()
    participant M as memory.py
    participant S as Session.history
    participant LC as LangChain model
    participant P as LLM provider

    API->>M: should_rotate_summary(history, summarize_every_turns=4)
    M-->>API: true when (non-system messages // 2) is a multiple of 4<br/>The first time is the 5th message of the session

    API->>S: split the non-system messages<br/>middle = all but the last 4, recent = the last 4
    API->>LC: ainvoke([system: "Summarize in 4-6 bullets...",<br/>user: middle as "role: content" lines])
    LC->>P: Non-streaming request
    P-->>LC: summary
    LC-->>API: resp.text

    API->>S: history = [persona system prompt,<br/>system "[Earlier conversation summary] ...",<br/>...recent]

    Note over API: apply_strategy() then windows the tail and the normal turn continues
```

---

## Behaviours worth knowing

These follow from the order of steps in the diagrams.

- **The user message is stored before the model runs.** If the call fails (section 5) or the user cancels (section 6), the history ends with a user message that has no reply. The next turn sends both user messages to the model.
- **A blocked input is not stored.** Section 4 never touches the history or the counters, so a blocked message does not count as a turn.
- **The output guardrail cannot take back streamed text.** `check_output` runs after the last `delta`, so the client has already shown the reply. The `guardrail` event adds a badge, and `SAFE_DECLINE` replaces the reply only in the stored history, so the model never sees it on later turns.
- **Summary rotation counts messages, not turns.** After a rotation the history is shorter, so the next rotation does not land four turns later. A rotation also rebuilds the history from the persona prompt, the new summary and the last four messages, so the previous summary is not carried into the next one.
- **Sessions live in the server process.** After a restart (including a `--reload`), the browser still holds the old `sid`. Every message gets a 404 and the bubble shows the error until the user clicks Reset or changes persona.
