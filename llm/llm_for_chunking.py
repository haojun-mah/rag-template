import google.generativeai as genai
import os
from google.generativeai import types
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import json

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") 
genai.configure(api_key=api_key)

class ChunkMetadata(BaseModel):
    """Structured metadata for a document chunk."""
    summary: str = Field(description="A concise 1-2 sentence summary of the chunk.")
    keywords: List[str] = Field(description="A list of 5-7 key topics or entities mentioned.")
    hypothetical_questions: List[str] = Field(description="A list of 3-5 questions this chunk could answer.")
    table_summary: Optional[str] = Field(description="If the chunk is a table, a natural language summary of its key insights.")

model = genai.GenerativeModel(
    "gemini-2.0-flash-lite",
    generation_config={
        "response_mime_type": "application/json",
        "response_schema": ChunkMetadata,
    }
)


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
    """Uses the Gemini LLM to generate enriched metadata for a chunk."""
    
    response = model.generate_content(metadata_prompt)
    
    # Parse the JSON response
    response_data = json.loads(response.text)
    
    # Ensure table_summary is present (set to None if missing)
    if 'table_summary' not in response_data:
        response_data['table_summary'] = None
    
    # Validate and return
    return ChunkMetadata.model_validate(response_data)
