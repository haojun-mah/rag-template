import importlib
import json

import main as main_module


def test_main_workflow_with_small_file(monkeypatch, tmp_path):
    sample_file = tmp_path / "sample_small.txt"
    sample_file.write_text(
        "Quarterly Update\n\nRevenue increased by 10% year-over-year while expenses declined.\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "enriched.json"

    monkeypatch.setenv("TARGET_FILE_LOCATION", str(sample_file))

    main = importlib.reload(main_module)

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
    monkeypatch.setattr(main, "ENRICHED_CHUNKS_PATH", output_path)

    main.main()

    assert output_path.exists()
    stored_records = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(stored_records) == 1
    assert stored_records[0]["summary"] == "Chunk 0 summary"
    assert stored_records[0]["keywords"] == ["keyword-0"]
