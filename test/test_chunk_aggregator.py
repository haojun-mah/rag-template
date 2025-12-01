from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ingestion.chunk_enrichment.chunk_aggregator import collect_enriched_chunks


class DummyMetadata:
    def __init__(self, data: dict[str, Any], html: str | None = None):
        self._data = data
        self.text_as_html = html

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self._data)
        if self.text_as_html is not None:
            payload["text_as_html"] = self.text_as_html
        return payload


class DummyChunk:
    def __init__(self, text: str, metadata: DummyMetadata):
        self.text = text
        self.metadata = metadata


def test_collect_enriched_chunks_writes_combined_json(tmp_path):
    filings_root = tmp_path / "sec"
    file_path = filings_root / "MSFT" / "8-K" / "0001" / "full-submission.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("dummy", encoding="utf-8")

    chunks = [
        DummyChunk("plain content", DummyMetadata({"page": 1})),
        DummyChunk("table content", DummyMetadata({}, html="<table>rows</table>")),
    ]

    enriched_metadata = [
        {"summary": "plain summary", "keywords": ["plain"]},
        {"summary": "table summary", "keywords": ["table"]},
    ]

    output_path = tmp_path / "enriched" / "chunks.json"

    results = collect_enriched_chunks(
        file_path=file_path,
        chunks=chunks,
        enriched_metadata=enriched_metadata,
        output_path=output_path,
        filings_root=filings_root,
        append=False,
    )

    assert len(results) == 2
    assert results[0]["source"] == "8-K/0001"
    assert results[0]["is_table"] is False
    assert results[0]["content"] == "plain content"
    assert results[0]["summary"] == "plain summary"

    assert results[1]["is_table"] is True
    assert results[1]["content"].startswith("<table>")
    assert results[1]["summary"] == "table summary"

    with output_path.open("r", encoding="utf-8") as saved:
        persisted = json.load(saved)

    assert persisted == results
