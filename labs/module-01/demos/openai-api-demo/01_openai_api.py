from llm_client import get_client

# Provider (OpenAI or Azure OpenAI) is selected by LLM_PROVIDER in .env
client, model = get_client()

response = client.responses.create(
    model=model,
    input="Latest AI trends in 2026"
)

print(response.output_text)
print(response.usage)

