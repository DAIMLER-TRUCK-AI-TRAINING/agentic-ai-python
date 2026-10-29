# FastAPI Application Health Check

This project demonstrates how to **create a minimal FastAPI application with a health check endpoint** for API monitoring.
It is the foundation for building production-ready services that can be monitored by load balancers, orchestrators (Kubernetes), and monitoring tools.

---

## Features

- **Health Check Endpoint:** `/health` endpoint for service monitoring
- **FastAPI Metadata:** Configured with title, description, and version
- **CORS Middleware:** Enable browser-based access
- **Auto-Generated Documentation:** Interactive Swagger UI at `/docs`
- **Production-Ready Pattern:** Standard pattern used in microservices
- **Minimal Dependencies:** Only FastAPI and Uvicorn required

---

## Project Structure

```bash
demo-03-fastapi-application-health-check/
├── main.py                     # FastAPI application
├── .env                        # Environment variables (if needed)
├── .gitignore                  # Ignore sensitive/config files
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Lock file
└── README.md                   # Documentation
```

---

## Setup

### 1. Create and Initialize Project

```bash
uv init demo-03-fastapi-application-health-check
cd demo-03-fastapi-application-health-check
```

---

### 2. Create Virtual Environment

```bash
uv venv
```

Activate the virtual environment:

**Linux/macOS:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
uv add fastapi uvicorn
```

**Dependencies:**

- `fastapi` - Web framework
- `uvicorn` - ASGI server

---

## Usage

Run the script using `uv`:

```bash
uv run uvicorn main:app --reload --port 8000
```

The server will start at: http://localhost:8000

---

## Example Output

### `/health` Response (200 OK):

```json
{
  "status": "healthy",
  "service": "llm-query-api",
  "version": "1.0.0"
}
```

---

## Key Concepts

| Step | Concept            | Description                               |
| ---- | ------------------ | ----------------------------------------- |
| 1    | FastAPI Setup      | Create application instance with metadata |
| 2    | CORS Middleware    | Enable cross-origin resource sharing      |
| 3    | Health Check       | Add monitoring endpoint                   |
| 4    | Auto Documentation | Interactive Swagger UI at `/docs`         |
| 5    | ASGI Server        | Run with Uvicorn for async support        |

---

## Summary

This project demonstrates how to implement a **basic FastAPI application with health check endpoint** using **FastAPI** and **Uvicorn**.
It provides the **foundation for production-ready APIs** with monitoring capabilities and auto-generated documentation.
