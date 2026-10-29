# Demo 13: Streamlit Integration with FastAPI Streaming

This project demonstrates how to build a **Streamlit chatbot frontend** with **real-time streaming responses** that integrates with the **FastAPI streaming backend** from `demo-08-llm-stream-endpoint`. The chatbot uses Server-Sent Events (SSE) to display AI responses token-by-token as they're generated, creating a ChatGPT-like typing effect.

---

## Features

- **Real-Time Streaming**: Token-by-token display of AI responses as they arrive
- **Server-Sent Events (SSE)**: Standard protocol for server-to-client streaming
- **ChatGPT-Like Experience**: Typing effect with cursor animation during streaming
- **FastAPI Integration**: Connects to demo-08's `/query/stream` endpoint
- **Interactive Chat UI**: Clean Streamlit interface with message history
- **Error Handling**: Graceful handling of connection errors and timeouts
- **Session Management**: Maintains chat history within session
- **Instant Feedback**: Responses start appearing immediately without waiting for completion

---

## Architecture

```
┌─────────────────────┐    HTTP POST (SSE)      ┌─────────────────────┐
│                     │    /query/stream         │                     │
│  Streamlit Frontend │ ─────────────────────>  │  FastAPI Backend    │
│  (This Demo)        │                          │  (demo-08)          │
│  Port: 8501         │ <──────────────────────  │  Port: 8000         │
│                     │   Token Stream (SSE)     │                     │
└─────────────────────┘                          └─────────────────────┘
         │                                                 │
         │ Parse SSE                                      │ Stream tokens
         │ Display tokens                                 │
         ▼                                                 ▼
   ┌─────────────┐                               ┌─────────────────┐
   │ Chat UI     │                               │   LLM API       │
   │ (Streaming) │                               │   (Gemini)      │
   └─────────────┘                               └─────────────────┘
```

---

## Project Structure

```bash
demo-13-streamlit-integration-with-fastapi-streaming/
├── main.py                      # Streamlit streaming application
├── pyproject.toml              # Project dependencies
├── .gitignore                  # Ignore sensitive/config files
└── README.md                   # Documentation
```

---

## Prerequisites

Before running this demo, ensure that **demo-08-llm-stream-endpoint** is set up and running:

1. Navigate to demo-08:

   ```bash
   cd ../demo-08-llm-stream-endpoint
   ```

2. Make sure the `.env` file is configured with API keys

3. Start the FastAPI streaming server:

   ```bash
   uv run fastapi dev main.py
   ```

4. Verify the API is running at `http://localhost:8000`

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-13-streamlit-integration-with-fastapi-streaming
cd demo-13-streamlit-integration-with-fastapi-streaming
```

---

### 2. Install Dependencies

```bash
uv add streamlit requests
```

---

### 3. Run the Application

Start the Streamlit app:

```bash
uv run streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

---

## Usage

### Starting Both Services

**Terminal 1 - FastAPI Streaming Backend:**

```bash
cd demo-08-llm-stream-endpoint
uv run fastapi dev main.py
```

**Terminal 2 - Streamlit Frontend:**

```bash
cd demo-13-streamlit-integration-with-fastapi-streaming
uv run streamlit run main.py
```

### Using the Streaming Chat Interface

1. **Type Your Question**: Enter your question in the chat input at the bottom
2. **Watch It Stream**: See the response appear word-by-word in real-time
3. **Clear History**: Use the "Clear Chat History" button to start fresh
4. **View History**: Scroll up to see previous conversations

---

## Example Interaction

**User Input:**

```
Explain quantum computing in simple terms
```

**SSE Stream from demo-08:**

```
data: Quantum
data:  computing
data:  is
data:  a
data:  revolutionary
data:  technology
...
```

**Displayed in Streamlit:**

```
🤖 Assistant:
Quantum computing is a revolutionary technology▌
```

_(Updates in real-time as tokens arrive)_

---

## How Streaming Works

### 1. Client Makes Request

```python
response = requests.post(
    STREAM_ENDPOINT,
    json={"prompt": prompt},
    stream=True  # Enable streaming mode
)
```

### 2. Parse SSE Format

```python
for line in response.iter_lines():
    if line.startswith('data: '):
        token = line[6:]  # Extract token
        full_response += token
        display(full_response + "▌")  # Show cursor
```

### 3. Display Updates

- Each token updates the message placeholder
- Cursor (`▌`) shows active streaming
- Final response saves to chat history

---

## Configuration

### API Endpoints

Edit the constants in `main.py` if your FastAPI backend runs on a different host/port:

```python
API_BASE_URL = "http://localhost:8000"
STREAM_ENDPOINT = f"{API_BASE_URL}/query/stream"
```

### Streaming Timeout

Adjust the timeout value for longer responses:

```python
response = requests.post(
    STREAM_ENDPOINT,
    json={"prompt": prompt},
    stream=True,
    timeout=60  # Increase if needed
)
```

---

## Troubleshooting

### "Cannot connect to API" Error

**Problem**: Streamlit cannot reach the FastAPI backend.

**Solutions**:

- Ensure demo-08 FastAPI server is running: `uv run fastapi dev main.py`
- Check if port 8000 is available: `lsof -i :8000`
- Verify the API base URL in `main.py`

### Streaming Stops Mid-Response

**Problem**: Response streaming cuts off unexpectedly.

**Solutions**:

- Check demo-08 logs for errors
- Verify LLM API key is valid
- Increase timeout value in `main.py`
- Check network stability

### No Streaming Effect (Full Response at Once)

**Problem**: Response appears all at once instead of streaming.

**Solutions**:

- Verify demo-08 is returning SSE format with `stream=True`
- Check if `parse_sse_stream()` is correctly parsing the response
- Test the API directly with curl:
  ```bash
  curl -X POST http://localhost:8000/query/stream \
    -H "Content-Type: application/json" \
    -d '{"prompt":"test"}' \
    --no-buffer
  ```

### Response Shows with "[ERROR: ...]"

**Problem**: API error is being streamed.

**Solutions**:

- Check demo-08's `.env` file for valid API keys
- Review FastAPI logs for detailed error messages
- Verify the LLM service is accessible

---

## Key Differences from Demo-12

| Feature         | Demo-12 (Standard)        | Demo-13 (Streaming)        |
| --------------- | ------------------------- | -------------------------- |
| Endpoint        | `/query`                  | `/query/stream`            |
| Protocol        | HTTP Request/Response     | Server-Sent Events (SSE)   |
| Response        | Complete response at once | Token-by-token streaming   |
| User Experience | Wait for full response    | Immediate, gradual display |
| Implementation  | `response.json()`         | `response.iter_lines()`    |
| Visual Feedback | Loading spinner           | Typing cursor animation    |

---

## Key Concepts

| Component     | Technology             | Purpose                                |
| ------------- | ---------------------- | -------------------------------------- |
| Frontend      | Streamlit              | Interactive streaming chat UI          |
| Backend       | FastAPI                | REST API with SSE streaming            |
| Protocol      | Server-Sent Events     | Unidirectional server-to-client stream |
| Streaming     | requests (stream=True) | Parse SSE chunks in real-time          |
| LLM           | Gemini                 | AI text generation with streaming      |
| Session State | Streamlit              | Maintain chat history                  |

---

## Extending the Demo

### Add Stop Button

Allow users to stop streaming mid-response:

- Add a stop button that appears during streaming
- Implement request cancellation logic

### Show Token Rate

Display streaming speed:

```python
start_time = time.time()
token_count = 0
# ... after streaming ...
tokens_per_second = token_count / (time.time() - start_time)
st.caption(f"⚡ {tokens_per_second:.1f} tokens/sec")
```

### Add Response Metadata

Show model and timing information:

```python
st.caption(f"Model: {model} • Response time: {elapsed:.2f}s")
```

### Add Typing Sound Effect

Play subtle typing sounds during streaming:

```python
import streamlit.components.v1 as components
components.html('<audio autoplay><source src="typing.mp3"></audio>')
```

### Multi-Model Support

Allow users to select different AI models:

- Add model selector in sidebar
- Pass model parameter to streaming endpoint

---

## Performance Tips

1. **Buffer Size**: The default buffer size works well for most cases
2. **Timeout Values**: Set reasonable timeouts (30-60s) for long responses
3. **Error Recovery**: Implement retry logic for transient failures
4. **Connection Pooling**: Reuse HTTP connections for better performance

---

## Dependencies

- **streamlit** (>=1.40.0): Web UI framework with streaming support
- **requests** (>=2.32.0): HTTP client with SSE streaming capabilities

---

## Related Demos

- **demo-08-llm-stream-endpoint**: The FastAPI streaming backend this demo connects to
- **demo-12-streamlit-integration-with-fastapi**: Non-streaming version
- **demo-05-first-interactive-chat-app-with-streamlit**: Basic Streamlit chat (echo bot)
- **demo-04-console-llm-app-to-rest-api**: Standard (non-streaming) FastAPI backend

---

## Server-Sent Events (SSE) Format

SSE messages follow this format:

```
data: First token
data: Second token
data: Third token

```

Each line starts with `data: ` followed by the content. Double newline (`\n\n`) signals the end of the stream.

---

## License

MIT License - Feel free to modify and use in your projects!
