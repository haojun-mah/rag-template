import os
import json
import pytest
import main as main_module
from agent.embedder.fast_embed_qdrant import client as qdrant_client, COLLECTION_NAME

def test_workflow_skip_enrichment(tmp_path, monkeypatch):
    """
    Runs the workflow with SKIP_ENRICHMENT=true.
    Verifies that the workflow completes without calling the LLM and uses default metadata.
    """
    
    # 1. Setup Sample Data
    sample_text = "Quarterly Update\n\nRevenue increased by 10% year-over-year while expenses declined.\n"
    sample_file = tmp_path / "sample_small.txt"
    sample_file.write_text(sample_text, encoding="utf-8")

    # Setup CSV for SQLite part
    csv_dir = tmp_path / "sec-edgar-filings"
    csv_dir.mkdir()
    csv_file = csv_dir / "revenue_summary.csv"
    csv_file.write_text("year,revenue\n2023,100\n2022,90", encoding="utf-8")

    output_path = tmp_path / "enriched_chunks.json"
    
    # 2. Set Environment Variables
    monkeypatch.setenv("TARGET_FILE_LOCATION", str(sample_file))
    monkeypatch.setenv("ENRICHED_CHUNKS_PATH", str(output_path))
    monkeypatch.setenv("SKIP_ENRICHMENT", "true") # Enable skip mode
    monkeypatch.chdir(tmp_path)

    # 3. Run the Main Workflow
    import importlib
    importlib.reload(main_module)

    # Mock load_dotenv to prevent it from overwriting our monkeypatched env vars
    # We must patch AFTER reload, because reload re-imports the real function
    monkeypatch.setattr(main_module, "load_dotenv", lambda **kwargs: None)
    
    # Clear Qdrant collection
    try:
        qdrant_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
        
    # Re-create collection
    from qdrant_client import models
    from agent.embedder.fast_embed_qdrant import embedding_model
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_model.embedding_size,
            distance=models.Distance.COSINE
        )
    )

    # Execute main
    print("\nRunning main workflow with SKIP_ENRICHMENT=true...")
    main_module.main()

    # 4. Verifications
    
    # Verify Output File
    assert output_path.exists(), "Enriched chunks JSON file should exist"
    data = json.loads(output_path.read_text())
    assert len(data) > 0, "Should have at least one chunk"
    
    # Check structure of the first chunk
    first_chunk = data[0]
    print(f"\nOutput for Chunk 1: {first_chunk}")
    
    assert first_chunk["summary"] == "No summary available (enrichment skipped)"
    assert first_chunk["keywords"] == []

    # Verify Qdrant Embeddings
    count_result = qdrant_client.count(COLLECTION_NAME)
    assert count_result.count > 0, "Qdrant should have stored vectors even without enrichment"
    
    # Verify SQLite
    import sqlite3
    db_path = tmp_path / "financials.db"
    
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM revenue_summary")
        count = cursor.fetchone()[0]
        assert count == 2, "SQLite table should have 2 rows"
        conn.close()
