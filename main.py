from dotenv import load_dotenv
from ingestion.chunk_enrichment.llm_enrichment import enrich_chunk
from ingestion.chunk_generation.title_chunker import title_chunker
from ingestion.parser.parse_html import parse_html

# Load environment variables from .env file
load_dotenv()

TARGET_FILE_LOCATION = "sec-edgar-filings/MSFT/8-K"

def main():
    """Main orchestrator for Agentic Rag workflow"""
   
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


if __name__ == "__main__":
    main()
