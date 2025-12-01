"""Combine chunk content with metadata and persist aggregated results to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence


def collect_enriched_chunks(
    file_path: str | Path,
    chunks: Sequence[Any],
    enriched_metadata: Sequence[dict[str, Any]],
    output_path: str | Path,
    *,
    filings_root: str | Path | None = None,
    append: bool = True,
) -> List[dict[str, Any]]:
    """Combine chunk content with metadata and store the result in a JSON file."""

    file_path = Path(file_path)
    filings_root = Path(filings_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    persisted_data: List[dict[str, Any]] = []
    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing_file:
            persisted_data = json.load(existing_file)

    new_records: List[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        enrichment_data = (
            enriched_metadata[index] if index < len(enriched_metadata) else {}
        )
        final_chunk = _finalize_chunk_data(file_path, filings_root, chunk, enrichment_data)
        if final_chunk:
            new_records.append(final_chunk)

    combined_records = persisted_data + new_records if append else new_records

    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(combined_records, outfile, indent=2)

    print(
        f"Persisted {len(new_records)} enriched chunks from {file_path} to '{output_path}'. "
        f"Total stored: {len(combined_records)}"
    )

    return combined_records


def _finalize_chunk_data(
    file_path: Path, filings_root: Path, chunk: Any, enrichment_data: dict[str, Any]
):
    if not enrichment_data:
        return None

    chunk_metadata = chunk.metadata.to_dict()
    is_table = "text_as_html" in chunk_metadata
    content = getattr(chunk.metadata, "text_as_html", None) if is_table else chunk.text

    source = _derive_source_label(file_path, filings_root)

    return {
        "source": source,
        "content": (content or ""),
        "is_table": is_table,
        **enrichment_data,
    }


def _derive_source_label(file_path: Path, filings_root: Path) -> str:
    try:
        relative_parts = file_path.relative_to(filings_root).parts
    except ValueError:
        relative_parts = file_path.parts

    if len(relative_parts) >= 3:
        return "/".join(relative_parts[1:3])
    if len(relative_parts) >= 2:
        return "/".join(relative_parts[:2])
    if relative_parts:
        return relative_parts[0]
    return file_path.name