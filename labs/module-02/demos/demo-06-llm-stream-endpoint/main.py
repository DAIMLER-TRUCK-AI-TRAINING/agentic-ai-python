"""
Demo 08: LLM Streaming Endpoint with Server-Sent Events (SSE)

This example demonstrates how to implement streaming AI responses
using FastAPI and Server-Sent Events for real-time token-by-token output.
Supports multiple LLM providers (OpenAI, Gemini, etc.)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

def initialize_llm_client() -> tuple[OpenAI, str, str]:
    """Initialize and return OpenAI client with model config.
    
    Supports multiple LLM providers via environment variables.
    
    Returns:
        tuple: (OpenAI client, model name, provider name)
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    api_key = os.getenv(f"{provider.upper()}_API_KEY")
    model_name = os.getenv(f"{provider.upper()}_MODEL_NAME")
    base_url = os.getenv(f"{provider.upper()}_BASE_URL")
    
    if not api_key:
        raise ValueError(f"{provider.upper()}_API_KEY environment variable is required.")
    
    if not model_name:
        raise ValueError(f"{provider.upper()}_MODEL_NAME environment variable is required.")
    
    if not base_url:
        raise ValueError(f"{provider.upper()}_BASE_URL environment variable is required.")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, model_name, provider

# Initialize client
client, model, provider = initialize_llm_client()

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Multi-Provider AI Streaming API",
    description=f"Streaming LLM responses using Server-Sent Events. Currently using {provider.upper()} provider",
    version="2.0.0"
)

# Configure CORS middleware
# This allows browser-based clients from any origin ("*") to access the API
# WARNING: For production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define Request and Response Models
class QueryRequest(BaseModel):
    """Request model for user prompts"""
    prompt: str = "What is a list in Python?"
Define Streaming Function
def stream_ai_response(user_prompt: str):
    """
    Stream AI response word-by-word using Server-Sent Events.
    
    Workflow:
    1. Create streaming request to LLM
    2. Iterate through response chunks
    3. Yield each chunk in SSE format
    4. Signal completion
    5. Handle errors gracefully
    
    Args:
        user_prompt: User's question or prompt
        
    Yields:
        str: SSE-formatted response chunks
    """
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            stream=True
        )
        
        for chunk in stream:
            word = chunk.choices[0].delta.content or ""
            if word:
                yield f"data: {word}\n\n"
        
        yield "\n\n"
        
    except Exception as error:
        yield f"data: [ERROR: {str(error)}]\n\n"


# Define Non-Streaming Endpoint (for comparison)
@app.post("/query", response_model=QueryResponse)
def ask_ai(request: QueryRequest) -> QueryResponse:
    """
    Standard non-streaming endpoint for comparison.
    
    Args:
        request: Request object containing user prompt
        
    Returns:
        QueryResponse: Complete AI response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        content = response.choices[0].message.content
        
        return QueryResponse(
            response=content,
            model=model,
            provider=provider
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Define Streaming Endpoint
@app.post("/query/stream")
def ask_ai_streaming(request: QueryRequest):
    """
    Stream AI response word-by-word in real-time using Server-Sent Events.
    
    Workflow:
    1. Receive user prompt
    2. Initialize streaming connection to LLM
    3. Stream response chunks as they arrive
    4. Return StreamingResponse with SSE format
    
    Benefits:
    - Real-time feedback: See response as it's generated
    - Better UX: Reduced perceived latency
    - Progressive rendering: Display content incrementally
    
    Args:
        request: Request object containing user prompt
        
    Returns:
        StreamingResponse: SSE stream of AI response
    """
    return StreamingResponse(
        stream_ai_response(request.prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Provider AI Streaming API",
        "provider": provider,
        "model": model,
        "endpoints": {
            "query": "/query (POST) - Non-streaming response",
            "stream": "/query/stream (POST) - Streaming response"
        }
    }   request: Request object containing user prompt
        
    Returns:
        StreamingResponse: SSE stream of AI response
    """
    return StreamingResponse(
        stream_ai_response(request.prompt),
        media_type="text/event-stream"
    )
