from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # Azure portal > your resource > Keys and Endpoint
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # e.g. https://<your-resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # Your deployment name, not the base model name


# Azure OpenAI v1 API: same OpenAI client, just point base_url at your Azure resource
client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/v1/",
)

response = client.responses.create(
    model=AZURE_OPENAI_DEPLOYMENT,
    input="Latest AI trends in 2026"
)

print(response.output_text)
