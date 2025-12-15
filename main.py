import os
from pathlib import Path

from dotenv import load_dotenv

from agent.chunk_enrichment.chunk_aggregator import (
    collect_enriched_chunks,
)
from agent.chunk_enrichment.llm_enrichment import enrich_chunk
from agent.chunk_generation.title_chunker import title_chunker
from agent.parser.parse_html import parse_html
from agent.embedder.fast_embed_qdrant import create_embedding_from_chunks
from utils.sqlite_db import create_table_from_input


def main():
    """Main orchestrator for Agentic Rag workflow"""

    load_dotenv()
    
    # Extract environment variables
    folder = os.getenv("TARGET_FILE_LOCATION")
    if not folder:
        print("Error: TARGET_FILE_LOCATION environment variable not set.")
        return

    enriched_chunks_path = os.getenv("ENRICHED_CHUNKS_PATH")
    if not enriched_chunks_path:
        print("Error: ENRICHED_CHUNKS_PATH environment variable not set.")
        return

    # Parse files into text
    parsed_elements = parse_html(folder)
    if not parsed_elements:
        print(f"No elements were parsed from {folder}. Nothing to chunk.")
        return
    print(f"\nParsed {len(parsed_elements)} elements from {folder}")

    # Chunk text into smaller text chunks
    chunked_elements = title_chunker(parsed_elements)
    print(f"\nGenerated {len(chunked_elements)} chunks from parsed elements.")

    # Enrich chunks with LLM-generated metadata
    enriched_chunks = enrich_chunk(chunked_elements)
    print(f"\nEnriched {len(enriched_chunks)} chunks with metadata.")
    print("\nSample enriched chunk metadata:")
    for i, enriched in enumerate(enriched_chunks[:3]):
        print(f"\nEnriched Chunk {i+1}: {enriched}")

    # Aggregate and persist enriched chunks in JSON
    enriched_chunks = collect_enriched_chunks(
        file_path=folder,
        chunks=chunked_elements,
        enriched_metadata=enriched_chunks,
        output_path=enriched_chunks_path,
        filings_root=folder,
    )

    # Create embeddings from enriched chunks and store in Qdrant
    create_embedding_from_chunks(enriched_chunks)

    # Create SQLite table from data
    create_table_from_input(
        input_path="sec-edgar-filings/revenue_summary.csv",
        table_name="revenue_summary",
    )


if __name__ == "__main__":
    main()