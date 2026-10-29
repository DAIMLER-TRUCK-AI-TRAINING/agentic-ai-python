"""
Demo 3: FastAPI Application Health Check

This project demonstrates how to **create a minimal FastAPI application with a health check endpoint** for API monitoring.
"""

# 1. Import dependencies: FastAPI, CORS middleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 2. Initialize FastAPI application with metadata
app = FastAPI(
    title="LLM Query API",
    description="REST API for querying Gemini LLM",
    version="1.0.0"
)

# 3. Configure CORS middleware
# This allows browser-based clients from any origin ("*") to access the API
# WARNING: For production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Define Health Check Endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API is running and accessible.
    
    Returns:
        dict: Service status information
    """
    return {
        "status": "healthy",
        "service": "llm-query-api",
        "version": "1.0.0"
    }
