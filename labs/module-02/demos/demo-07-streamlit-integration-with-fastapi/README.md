# Demo 12: Streamlit Integration with FastAPI

This project demonstrates how to build a **Streamlit chatbot frontend** that integrates with the **FastAPI backend** from `demo-04-console-llm-app-to-rest-api`. The chatbot sends user queries to the FastAPI `/query` endpoint and displays AI-generated responses in an interactive chat interface.

---

## Features

- **Interactive Chat UI**: Clean Streamlit chat interface with message history
- **FastAPI Integration**: Connects to demo-04's REST API `/query` endpoint
- **Health Check**: Monitor backend API status from the sidebar
- **Error Handling**: Graceful handling of connection errors and timeouts
- **Session Management**: Maintains chat history within session
- **Real-time Responses**: Displays loading indicators while waiting for API responses
- **Clear Chat**: Button to reset conversation history

---

## Architecture

```
┌─────────────────────┐         HTTP POST          ┌─────────────────────┐
│                     │     /query endpoint        │                     │
│  Streamlit Frontend │ ─────────────────────────> │  FastAPI Backend    │
│  (This Demo)        │                            │  (demo-04)          │
│  Port: 8501         │ <───────────────────────── │  Port: 8000         │
│                     │     JSON Response          │                     │
└─────────────────────┘                            └─────────────────────┘
                                                             │
                                                             │
                                                             ▼
                                                    ┌─────────────────┐
                                                    │   LLM API       │
                                                    │   (Gemini)      │
                                                    └─────────────────┘
```

---

## Project Structure

```bash
demo-12-streamlit-integration-with-fastapi/
├── main.py                      # Streamlit application
├── pyproject.toml              # Project dependencies
├── .gitignore                  # Ignore sensitive/config files
└── README.md                   # Documentation
```

---

## Prerequisites

Before running this demo, ensure that **demo-04-console-llm-app-to-rest-api** is set up and running:

1. Navigate to demo-04:

   ```bash
   cd ../demo-04-console-llm-app-to-rest-api
   ```

2. Make sure the `.env` file is configured with API keys

3. Start the FastAPI server:

   ```bash
   uv run fastapi dev main.py
   ```

4. Verify the API is running at `http://localhost:8000`

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-12-streamlit-integration-with-fastapi
cd demo-12-streamlit-integration-with-fastapi
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

**Terminal 1 - FastAPI Backend:**

```bash
cd demo-04-console-llm-app-to-rest-api
uv run fastapi dev main.py
```

**Terminal 2 - Streamlit Frontend:**

```bash
cd demo-12-streamlit-integration-with-fastapi
uv run streamlit run main.py
```

### Using the Chat Interface

1. **Check API Status**: Click "Check API Status" in the sidebar to verify the backend is running
2. **Ask Questions**: Type your question in the chat input at the bottom
3. **View Responses**: AI responses will appear in the chat with model information
4. **Clear History**: Use the "Clear Chat History" button to start a new conversation

---

## Example Interaction

**User Input:**

```
What is Python programming?
```

**API Request to demo-04:**

```json
POST http://localhost:8000/query
{
  "prompt": "What is Python programming?"
}
```

**API Response:**

```json
{
  "model": "gemini-2.5-flash",
  "answer": "Python is a high-level, interpreted programming language known for its simplicity and readability..."
}
```

**Displayed in Streamlit:**

```
🤖 Assistant:
Python is a high-level, interpreted programming language known for its simplicity and readability...

Model: gemini-2.5-flash
```

---

## Configuration

### API Endpoints

Edit the constants in `main.py` if your FastAPI backend runs on a different host/port:

```python
API_BASE_URL = "http://localhost:8000"  # Change if needed
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
```

---

## Troubleshooting

### "Cannot connect to API" Error

**Problem**: Streamlit cannot reach the FastAPI backend.

**Solutions**:

- Ensure demo-04 FastAPI server is running: `uv run fastapi dev main.py`
- Check if port 8000 is available: `lsof -i :8000`
- Verify the API base URL in `main.py` matches your setup

### "API returned status code 500" Error

**Problem**: The backend encountered an error processing the request.

**Solutions**:

- Check demo-04's `.env` file has valid API keys
- Review FastAPI logs in the terminal for error details
- Test the API directly: `curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"prompt":"test"}'`

### Request Timeout

**Problem**: API takes too long to respond.

**Solutions**:

- Increase timeout in `main.py`: `timeout=60` (default is 30 seconds)
- Check your LLM API rate limits
- Verify network connectivity

---

## Key Concepts

| Component     | Technology | Purpose                  |
| ------------- | ---------- | ------------------------ |
| Frontend      | Streamlit  | Interactive chat UI      |
| Backend       | FastAPI    | REST API for LLM queries |
| Communication | HTTP/REST  | Request-response pattern |
| LLM           | Gemini     | AI text generation       |
| Session State | Streamlit  | Maintain chat history    |

---

## Extending the Demo

### Add Streaming Responses

Modify to use Server-Sent Events (SSE) for real-time streaming:

- Update demo-04 to add a streaming endpoint
- Use `st.write_stream()` in Streamlit

### Add Authentication

Implement API key authentication:

- Add API key header to requests
- Protect FastAPI endpoints with dependencies

### Add Conversation Memory

Store chat history in a database:

- Use session_id to track conversations
- Persist messages beyond browser session

### Multi-Model Support

Allow users to select different AI models:

- Add model selector in Streamlit sidebar
- Pass model parameter to FastAPI endpoint

---

## Dependencies

- **streamlit** (>=1.40.0): Web UI framework
- **requests** (>=2.32.0): HTTP client for API calls

---

## Related Demos

- **demo-04-console-llm-app-to-rest-api**: The FastAPI backend this demo connects to
- **demo-05-first-interactive-chat-app-with-streamlit**: Basic Streamlit chat (echo bot)
- **demo-08-llm-stream-endpoint**: Streaming responses example

---

## License

MIT License - Feel free to modify and use in your projects!
