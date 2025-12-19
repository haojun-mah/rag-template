from ollama import Client
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class ChunkMetadata(BaseModel):
    """Structured metadata for a document chunk."""
    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: List[str] = Field(description="A list of 5-7 key topics or entities mentioned.")
    hypothetical_questions: List[str] = Field(description="A list of 3-5 questions this chunk could answer.")
    table_summary: Optional[str] = Field(description="If the chunk is a table, a natural language summary of its key insights.")

def chunk_enricher(chunk_text: str, is_table: bool, expert_role: str) -> str:
    """Generates a prompt for LLM to enrich the chunk with metadata."""
    if is_table:
        table_instruction = "This chunk is a TABLE. Your summary should describe the main data points and trends."
    else:
        table_instruction = "This chunk is NOT a table. You MUST set the 'table_summary' field to null."

    prompt = f"""
    You are an expert {expert_role}. Please analyze the following document chunk and generate the specified metadata.
    {table_instruction}
    Chunk Content:
    ---
    {chunk_text}
    ---
    """
    return prompt

def generate_enriched_chunk(metadata_prompt: str) -> ChunkMetadata:
    """Uses the Ollama Qwen model to generate enriched metadata for a chunk."""
    
    # Provide a clear JSON example structure instead of the complex JSON schema
    example_structure = {
        "summary": "A concise 1-2 sentence summary of the chunk.",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "hypothetical_questions": ["Question 1?", "Question 2?", "Question 3?"],
        "table_summary": "Summary of table data or null if not a table"
    }
    
    full_prompt = f"{metadata_prompt}\n\nRespond strictly in JSON format. The output must be a single JSON object matching this structure:\n{json.dumps(example_structure, indent=2)}"

    client = Client(host='http://192.168.88.17:11434') # Default host

    response = client.chat(
        model='qwen2.5:1.5b',
        messages=[{'role': 'user', 'content': full_prompt}],
        format='json'
    )

    # Parse the JSON response
    try:
        response_data = json.loads(response['message']['content'])
    except json.JSONDecodeError:
        # Fallback or empty if JSON is invalid
        print(f"Error decoding JSON from LLM: {response['message']['content']}")
        response_data = {}
    
    # Ensure table_summary is present (set to None if missing)
    if 'table_summary' not in response_data:
        response_data['table_summary'] = None
    
    # Validate and return
    return ChunkMetadata.model_validate(response_data)
