from llm_client import get_client

# Provider (OpenAI or Azure OpenAI) is selected by LLM_PROVIDER in .env
client, model = get_client()

response = client.chat.completions.create(
    model=model,
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