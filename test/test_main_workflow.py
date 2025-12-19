import importlib
import json
import sqlite3
from utils.sqlite_db import create_table_from_input as real_create_table

import main as main_module

def test_main_workflow_with_small_file(monkeypatch, tmp_path):
    # 1. Setup Text File
    sample_file = tmp_path / "sample_small.txt"
    sample_file.write_text(
        "Quarterly Update\n\nRevenue increased by 10% year-over-year while expenses declined.\n",
        encoding="utf-8",
    )

    # 2. Setup CSV File for SQLite
    csv_dir = tmp_path / "sec-edgar-filings"
    csv_dir.mkdir()
    csv_file = csv_dir / "revenue_summary.csv"
    csv_file.write_text("year,revenue\n2023,100\n2022,90", encoding="utf-8")

    output_path = tmp_path / "enriched.json"
    db_path = tmp_path / "financials.db"

    # 3. Environment & Path Mocking
    monkeypatch.setenv("TARGET_FILE_LOCATION", str(sample_file))
    monkeypatch.setenv("ENRICHED_CHUNKS_PATH", str(output_path))
    monkeypatch.chdir(tmp_path) # Run in tmp_path so relative paths work

    main = importlib.reload(main_module)

    # 4. Mock Enrichment (to skip LLM)
    def fake_enrich(chunks):
        enriched = []
        for idx, _ in enumerate(chunks):
            enriched.append(
                {
                    "summary": f"Chunk {idx} summary",
                    "keywords": [f"keyword-{idx}"],
                    "hypothetical_questions": [f"question-{idx}"],
                    "table_summary": None,
                }
            )
        return enriched

    monkeypatch.setattr(main, "enrich_chunk", fake_enrich)
    # monkeypatch.setattr(main, "ENRICHED_CHUNKS_PATH", output_path) # Removed as it is now an env var

    # 5. Mock SQLite creation to use temp DB
    def patched_create_table(csv_path, table_name, db_path=str(db_path)):
        # Pass the temp db_path to the real function
        return real_create_table(csv_path, table_name, db_path=db_path)
    
    monkeypatch.setattr(main, "create_table_from_input", patched_create_table)

    # 6. Run Main
    main.main()

    # 7. Verify JSON Output
    assert output_path.exists()
    stored_records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(stored_records) == 1
    assert stored_records[0]["summary"] == "Chunk 0 summary"
    assert stored_records[0]["keywords"] == ["keyword-0"]

    # 8. Verify Qdrant Insertion
    from agent.embedder.fast_embed_qdrant import client, COLLECTION_NAME
    
    # Force a refresh or check count directly
    count_result = client.count(COLLECTION_NAME)
    assert count_result.count == 1, f"Expected 1 document in Qdrant, found {count_result.count}"

    # Retrieve the point to check vector dimension
    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[0],
        with_vectors=True
    )
    vector = points[0].vector
    print(f"\n\n--- Verification ---")
    print(f"Number of vectors (chunks): {len(points)}")
    print(f"Dimension of the vector: {len(vector)}")
    print(f"First 5 values of vector: {vector[:5]}...")
    
    assert len(vector) == 384, f"Expected vector dimension 384, got {len(vector)}"

    # 9. Verify SQLite Database
    assert db_path.exists(), "SQLite database file was not created"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='revenue_summary'")
    assert cursor.fetchone() is not None, "Table 'revenue_summary' not found in DB"
    
    cursor.execute("SELECT * FROM revenue_summary")
    rows = cursor.fetchall()
    assert len(rows) == 2, f"Expected 2 rows in SQLite, found {len(rows)}"
    assert rows[0] == (2023, 100), "Data mismatch in SQLite"
    conn.close()
    print("SQLite verification passed.")
