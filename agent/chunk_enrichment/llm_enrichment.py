from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from llm.qwen_ollama_chunking import chunk_enricher, generate_enriched_chunk


def _process_single_chunk(chunk) -> Dict[str, Any]:
    """Helper to process a single chunk for parallel execution."""
    # Have to change chunker for different file types
    is_table = 'text_as_html' in chunk.metadata.to_dict()
    content = chunk.metadata.text_as_html if is_table else chunk.text
    expert = "financial analyst"

    # Adjust this value to avoid flooding long chunks to LLM
    truncated_content = content[:3000]

    prompt = chunk_enricher(truncated_content, is_table, expert)

    try:
        metadata = generate_enriched_chunk(prompt)
        return metadata.model_dump()
    except Exception as e:
        print(f"Error enriching chunk: {e}")
        return {}


def enrich_chunk(chunks) -> List[Dict[str, Any]]:
    """Generates chunk metadata using LLM in parallel"""
    
    # Adjust max_workers based on your Ollama server's capacity
    # Too many workers might cause timeouts or OOM errors on the server
    with ThreadPoolExecutor(max_workers=4) as executor:
        # map ensures the results are returned in the same order as the input chunks
        enriched_chunks = list(executor.map(_process_single_chunk, chunks))
    
    return enriched_chunks