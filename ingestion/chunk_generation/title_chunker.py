from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element
from typing import List

def title_chunker(parsed_elements: List[Element]):
    """Chunks parsed elements based on titles."""

    chunks = chunk_by_title(
        parsed_elements,
        max_characters=2048,
        combine_text_under_n_chars=256,
        new_after_n_chars=1800
    )
    
    print(f"Document chunked into {len(chunks)} sections.")

    text_chunk_sample = None
    table_chunk_sample = None

    print("\n--- Sample Chunks ---")
    for chunk in chunks:
        if 'text_as_html' not in chunk.metadata.to_dict() and text_chunk_sample is None and len(chunk.text) > 500:
            text_chunk_sample = chunk
        if 'text_as_html' in chunk.metadata.to_dict() and table_chunk_sample is None:
            table_chunk_sample = chunk
        if text_chunk_sample and table_chunk_sample:
            break

# Print details of the text chunk sample
    if text_chunk_sample:
        print("** Sample Text Chunk **")
        print(f"Content: {text_chunk_sample.text[:500]}...")  # Preview first 500 chars
        print(f"Metadata: {text_chunk_sample.metadata.to_dict()}")
# Print details of the table chunk sample

    if table_chunk_sample:
        print("\n** Sample Table Chunk **")
        html_preview = (table_chunk_sample.metadata.text_as_html or "")[:500]
        print(f"HTML Content: {html_preview}...")  # Preview HTML
        print(f"Metadata: {table_chunk_sample.metadata.to_dict()}")

    return chunks
