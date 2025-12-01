from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from llm.llm_for_chunking import chunk_enricher, generate_enriched_chunk


def enrich_chunk(chunks) -> List[Dict[str, Any]]:
    """Generates chunk metadata using LLM"""
    enriched_chunks = []
    
    for chunk in chunks: 
        # Have to change chunker for different file types
        is_table = 'text_as_html' in chunk.metadata.to_dict()
        content = chunk.metadata.text_as_html if is_table else chunk.text
        expert = "financial analyst"

        # Adjust this value to avoid flooding long chunks to LLM
        truncated_content = content[:3000]

        prompt = chunk_enricher(truncated_content, is_table, expert)    

        try:
            metadata = generate_enriched_chunk(prompt)
            enriched_chunks.append(metadata.model_dump())
        except Exception as e:
            print(f"Error enriching chunk: {e}")
            enriched_chunks.append({})
    
    return enriched_chunks