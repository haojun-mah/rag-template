from pathlib import Path
from typing import List

from unstructured.documents.elements import Element
from unstructured.partition.auto import partition


def _parse_file(file_path: Path) -> List[Element]:
    """Parse a single document into unstructured Element objects."""

    try:
        print(f"Parsing file: {file_path}")
        return partition(
            filename=str(file_path),
            strategy="fast",
            infer_table_structure=True,
            include_page_breaks=True,
            extract_images_in_pdf=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        print(f"Error parsing file {file_path}: {exc}")
        return []


def parse_html_file(file_path: str | Path) -> List[Element]:
    """Parse a single file into Element objects (public helper wrapper)."""

    return _parse_file(Path(file_path))


def parse_html(path: str) -> List[Element]:
    """Parse a file or directory of filings and return Element objects."""

    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    if path_obj.is_file():
        return _parse_file(path_obj)

    elements: List[Element] = []
    for file_path in sorted(p for p in path_obj.rglob("*") if p.is_file()):
        elements.extend(_parse_file(file_path))

    return elements
