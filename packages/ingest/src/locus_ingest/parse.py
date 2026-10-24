"""Parse: Docling JSON with page and bbox provenance, cached by doc_hash and parser version."""

import sqlite3
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from docling_core.types.doc import DoclingDocument, ImageRefMode

PARSED_DIR = "parsed"


class DocumentParser(Protocol):
    version: str

    def parse(self, path: Path) -> DoclingDocument: ...


class DoclingParser:
    def __init__(self) -> None:
        self.version = f"docling/{version('docling')}"
        self._converter = None

    def parse(self, path: Path) -> DoclingDocument:
        if self._converter is None:
            # Docling is slow to import and loads layout models, so wait until a parse needs it.
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter.convert(path).document


def parse(
    conn: sqlite3.Connection, parser: DocumentParser, data_dir: Path, doc_hash: str, path: Path
) -> tuple[DoclingDocument, bool]:
    """Return the document's Docling JSON and whether it came from the cache."""
    row = conn.execute(
        "SELECT json_path FROM parse_cache WHERE doc_hash = ? AND parser_version = ?",
        (doc_hash, parser.version),
    ).fetchone()
    if row and (data_dir / row[0]).is_file():
        return DoclingDocument.load_from_json(data_dir / row[0]), True

    doc = parser.parse(path)
    # json_path is relative to the data dir, so the data dir can move.
    json_path = Path(PARSED_DIR, doc_hash, f"{parser.version.replace('/', '-')}.json")
    target = data_dir / json_path
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    # Placeholders keep page images out of the cache; only the structure is stored.
    doc.save_as_json(partial, image_mode=ImageRefMode.PLACEHOLDER)
    partial.replace(target)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO parse_cache (doc_hash, parser_version, json_path)"
            " VALUES (?, ?, ?)",
            (doc_hash, parser.version, str(json_path)),
        )
    return doc, False
