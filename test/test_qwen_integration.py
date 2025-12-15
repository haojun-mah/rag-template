import pytest
from llm.qwen_ollama_chunking import generate_enriched_chunk, ChunkMetadata

def test_qwen_json_structure():
    """
    Tests that the Qwen model returns valid JSON that matches the ChunkMetadata schema.
    This is an integration test that requires a running Ollama instance.
    """
    
    # Simulating a prompt for a financial text chunk
    sample_prompt = """
    You are an expert financial analyst. Please analyze the following document chunk and generate the specified metadata.
    This chunk is NOT a table. You MUST set the 'table_summary' field to null.
    Chunk Content:
    ---
    Microsoft Corporation reported a 10% increase in revenue for the fiscal year 2023, driven by strong growth in its Intelligent Cloud segment. 
    Operating income also saw a significant rise, reflecting operational efficiency.
    ---
    """

    try:
        # Call the function that hits the LLM
        result = generate_enriched_chunk(sample_prompt)
        
        # Verify the result is of the correct type
        assert isinstance(result, ChunkMetadata)
        
        # Verify fields are populated reasonably
        assert result.summary is not None
        assert len(result.summary) > 0
        assert isinstance(result.keywords, list)
        assert len(result.keywords) > 0
        assert isinstance(result.hypothetical_questions, list)
        assert len(result.hypothetical_questions) > 0
        assert result.table_summary is None
        
        print("\n\n--- LLM Output ---")
        print(f"Summary: {result.summary}")
        print(f"Keywords: {result.keywords}")
        print("------------------\n")

    except Exception as e:
        pytest.fail(f"LLM Integration failed: {e}")
