"""
Demo 4: Private Document Analyzer
Uses the OpenAI-compatible client to call the local Ollama model.
"""


import logging
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# 1. Configure for local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

def analyze_document_privately(file_path, analysis_type="summary"):
    """
    Analyze a document using a local LLM for complete privacy.

    Args:
        file_path: Path to document file
        analysis_type: Type of analysis ('summary', 'key_points', 'sentiment')
    Returns:
        Analysis results as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    # 2. Add the prompt to the messages
    '''Prompt is the instruction for the model to follow'''

    prompts = {
        "summary": "Provide a concise 3-sentence summary of this document.",
    }

    '''The messages is the list of messages to send to the model'''
    messages = [
        {"role": "user", "content": f"{prompts[analysis_type]}\n\nDocument:\n{content}"}
    ]

    logger.info(f"Analyzing: {analysis_type}...")

    # 3. API invocation: call chat.completions.create()
    response = client.chat.completions.create(
        model="llama3.2:1b",
        messages=messages
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    logger.info("Private Document Analyzer\n")
    
    # Use existing document file
    doc_file = "sample._doc.txt"
    
    result = analyze_document_privately(doc_file, "summary")
    logger.info(f"\nSUMMARY:")
    logger.info(result)
    logger.info("\nAnalysis complete!")
