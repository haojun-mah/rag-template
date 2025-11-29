from ingestion.chunk_generation.title_chunker import title_chunker
from ingestion.parser.parse_html import parse_html

TARGET_FILE_LOCATION = "sec-edgar-filings/MSFT/8-K"

def main():
    """Main orchestrator for Agentic Rag workflow"""
   
    folder = TARGET_FILE_LOCATION

    parsed_elements = parse_html(folder)
    if not parsed_elements:
        print(f"No elements were parsed from {folder}. Nothing to chunk.")
        return

    print(f"\nParsed {len(parsed_elements)} elements from {folder}")

    chunked_elements = title_chunker(parsed_elements)
    print(f"\nGenerated {len(chunked_elements)} chunks from parsed elements.")
    
    
    
    
    
    


if __name__ == "__main__":
    main()
