"""Title resolution: PDF metadata, then the first Docling heading, then the filename."""

from pathlib import Path

import pypdfium2
from docling_core.types.doc import DoclingDocument, SectionHeaderItem, TitleItem


def metadata_title(path: Path) -> str | None:
    with pypdfium2.PdfDocument(path) as pdf:
        title = pdf.get_metadata_dict(skip_empty=True).get("Title", "").strip()
    return title or None


def first_heading(doc: DoclingDocument) -> str | None:
    for item, _level in doc.iterate_items():
        if isinstance(item, TitleItem | SectionHeaderItem) and item.text.strip():
            return item.text.strip()
    return None


def resolve_title(path: Path, doc: DoclingDocument) -> tuple[str, str]:
    """Return the title and its source: metadata, heading or filename."""
    if title := metadata_title(path):
        return title, "metadata"
    if title := first_heading(doc):
        return title, "heading"
    return path.stem, "filename"
