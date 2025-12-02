import os
from pathlib import Path

from dotenv import load_dotenv

from agent.chunk_enrichment.chunk_aggregator import (
    collect_enriched_chunks,
)
from agent.chunk_enrichment.llm_enrichment import enrich_chunk
from agent.chunk_generation.title_chunker import title_chunker
from agent.parser.parse_html import parse_html

TARGET_FILE_LOCATION = os.getenv("TARGET_FILE_LOCATION")
ENRICHED_CHUNKS_PATH = Path("enriched/enriched_chunks.json")


def main():
    """Main orchestrator for Agentic Rag workflow"""

    load_dotenv()

    folder = TARGET_FILE_LOCATION

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
    collect_enriched_chunks(
        file_path=folder,
        chunks=chunked_elements,
        enriched_metadata=enriched_chunks,
        output_path=ENRICHED_CHUNKS_PATH,
        filings_root=TARGET_FILE_LOCATION,
    )


if __name__ == "__main__":
    main()
