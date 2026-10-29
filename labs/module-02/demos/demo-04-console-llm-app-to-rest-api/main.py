"""
Demo 4: Converting Console LLM App to REST API, with LangChain

This project demonstrates how to **transform a console-based AI application into a REST API** using **FastAPI**.
The Study Buddy answers through a LangChain LCEL chain (prompt | model | parser). LLM_PROVIDER in .env picks
the model: OpenAI, Anthropic Claude, Azure OpenAI or Claude on Microsoft Foundry; the endpoints never change.
"""

# 1. Import dependencies: FastAPI, CORS middleware, Pydantic, LangChain, dotenv,
#    and both SDKs for their error types
import os
import logging
import anthropic
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def require_env(name):
    """Return an environment variable or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} not found in environment variables!")
    return value


def build_model(provider):
    """
    Build a LangChain chat model for the provider in LLM_PROVIDER.
    init_chat_model returns a ChatOpenAI or a ChatAnthropic; only the model name,
    endpoint and key differ, so the chain and the endpoints stay the same.
    Returns the model and its name (the deployment name on Azure).
    """
    base_url = None

    if provider == "openai":
        # OpenAI's own API, with an OpenAI API key
        model_name = require_env("OPENAI_MODEL")
        model = init_chat_model(
            model_name,
            model_provider="openai",
            api_key=require_env("OPENAI_API_KEY")
        )
    elif provider == "anthropic":
        # Anthropic's own Claude API, with an Anthropic API key
        model_name = require_env("ANTHROPIC_MODEL")
        model = init_chat_model(
            model_name,
            model_provider="anthropic",
            api_key=require_env("ANTHROPIC_API_KEY"),
            max_tokens=4096  # otherwise LangChain asks for the model's maximum
        )
    elif provider == "azure-openai":
        # Azure OpenAI v1 API: the regular OpenAI client, base_url points at your Azure resource.
        # "model" is the deployment name you created in Azure, not the base model name.
        base_url = f"{require_env('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"
        model_name = require_env("AZURE_OPENAI_DEPLOYMENT")
        model = init_chat_model(
            model_name,
            model_provider="openai",
            base_url=base_url,
            api_key=require_env("AZURE_OPENAI_API_KEY")
        )
    elif provider == "azure-anthropic":
        # Claude on Microsoft Foundry: the Anthropic Messages API under /anthropic on your resource.
        # "model" is the deployment name you created in Foundry.
        base_url = f"https://{require_env('ANTHROPIC_FOUNDRY_RESOURCE')}.services.ai.azure.com/anthropic"
        model_name = require_env("ANTHROPIC_FOUNDRY_DEPLOYMENT")
        model = init_chat_model(
            model_name,
            model_provider="anthropic",
            base_url=base_url,
            api_key=require_env("ANTHROPIC_FOUNDRY_API_KEY"),
            max_tokens=4096
        )
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. "
                         "Use 'openai', 'anthropic', 'azure-openai' or 'azure-anthropic'.")

    logger.info(f"Provider: {provider} | model: {model_name}" + (f" | base URL: {base_url}" if base_url else ""))
    return model, model_name


# 2. Load environment variables from .env file and build the chat model for LLM_PROVIDER
load_dotenv()
provider = (os.getenv("LLM_PROVIDER") or "openai").strip().lower()
model, model_name = build_model(provider)

# 3. Build the Study Buddy chain: prompt template | chat model | output parser
# The question is a template variable, so braces in user input are never read as template syntax.
# StrOutputParser keeps only the text, so the answer is a plain string on every provider.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Study Buddy, a patient tutor. Explain concepts in plain language, "
               "give one short example, and keep answers under 150 words."),
    ("human", "{question}")
])
chain = prompt | model | StrOutputParser()

# 4. Initialize FastAPI application with metadata
app = FastAPI(
    title="Console to REST API",
    description="A REST API version of the Study Buddy console app using FastAPI and LangChain.",
    version="1.0.0"
)

# 5. Configure CORS middleware
# This allows browser-based clients from any origin ("*") to access the API
# WARNING: For production, replace ["*"] with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 6. Define Request and Response Models
class QueryRequest(BaseModel):
    """Request model for user prompts"""
    prompt: str = Field(
        ...,
        min_length=1,
        description="The question for the Study Buddy",
        examples=["Explain the theory of relativity in simple terms."]
    )


class AnswerResponse(BaseModel):
    """Response model for AI answers"""
    provider: str
    model: str
    answer: str


# 7. Define Health Check Endpoint
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify API is running and accessible.

    Returns:
        dict: Service status information and the configured model
    """
    return {
        "status": "healthy",
        "service": "llm-query-api",
        "version": "1.0.0",
        "provider": provider,
        "model": model_name
    }


# 8. Define Query Endpoint
@app.post("/query", response_model=AnswerResponse, summary="Ask your AI")
async def ask_study_buddy(request: QueryRequest):
    """
    Accepts a study prompt, runs the LangChain chain, and returns the generated answer.

    Args:
        request: QueryRequest object containing user prompt

    Returns:
        AnswerResponse: AI-generated answer with provider and model information
    """
    try:
        # ainvoke awaits the model call without blocking the server for other requests
        answer = await chain.ainvoke({"question": request.prompt})
        return AnswerResponse(provider=provider, model=model_name, answer=answer)
    except (openai.APIStatusError, anthropic.APIStatusError) as error:
        # The LLM is an upstream service, so its errors are 502 Bad Gateway
        # 401: wrong key · 404: wrong model or deployment name · 429: rate limit
        raise HTTPException(
            status_code=502,
            detail=f"LLM provider error {error.status_code}: {error.message}"
        )
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        raise HTTPException(
            status_code=503,
            detail="Could not reach the LLM provider: check the endpoint or resource name"
        )
