from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure the API key is set in the environment
MODEL_NAME = os.getenv("MODEL_NAME")  # Ensure the model name is set in the environment


client = OpenAI(api_key=OPENAI_API_KEY)

response = client.responses.create(
    model=MODEL_NAME,
    input="Latest AI trends in 2026"
)

print(response.output_text)