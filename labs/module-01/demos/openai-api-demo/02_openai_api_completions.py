from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure the API key is set in the environment
MODEL_NAME = os.getenv("MODEL_NAME")  # Ensure the model name is set in the environment


client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "explain diff b/w openai chat completion vs responses api"}],
    # temperature=0.5,
    # top_p=0.9,
    # max_tokens=500
)

print(f"Full Response: {response}")

print(response.choices[0].message.content)

# Token usage (important for cost tracking)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Total tokens: {response.usage.total_tokens}")