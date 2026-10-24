"""Chunk: split a parsed document into token-bounded chunks with character offsets.

Offsets point into the document text (see docs/adr/0001-chunk-offsets.md), so they
don't depend on the chunker and stay comparable across indexes.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument
from locus_core.indexes import Index

ITEM_SEPARATOR = "\n\n"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    char_start: int
    char_end: int
    page: int | None
    bbox: dict | None
    modality: str
    # The string that gets embedded, heading prefix included.
    text: str
    # Tokens in text, counting the special tokens the model adds.
    token_count: int


def item_texts(doc: DoclingDocument, chunker: HybridChunker) -> dict[str, str]:
    """Each item as the chunker renders it (list markers, table triplets), keyed by self_ref.

    Joined with ITEM_SEPARATOR, in order, these make up the document text.
    """
    serializer = chunker.serializer_provider.get_serializer(doc=doc)
    texts = {}
    for item, _level in doc.iterate_items():
        if text := serializer.serialize(item=item).text:
            texts[item.self_ref] = text
    return texts


def item_spans(texts: dict[str, str]) -> dict[str, tuple[int, int]]:
    """Each item's span in the document text."""
    spans = {}
    offset = 0
    for ref, text in texts.items():
        spans[ref] = (offset, offset + len(text))
        offset += len(text) + len(ITEM_SEPARATOR)
    return spans


class UnlocatableChunkError(Exception):
    def __init__(self, ordinal: int):
        super().__init__(f"chunk {ordinal} could not be located in the document text")
        self.ordinal = ordinal


def chunk_document(
    doc: DoclingDocument, tokenizer: BaseTokenizer, special_tokens: int = 0
) -> list[Chunk]:
    """Chunk doc within tokenizer's max_tokens, which excludes the model's special tokens."""
    chunker = HybridChunker(tokenizer=tokenizer)
    texts = item_texts(doc, chunker)
    spans = item_spans(texts)
    # Where the last chunk ended inside each item, so repeated text isn't matched twice.
    cursors = dict.fromkeys(spans, 0)
    chunks = []
    for ordinal, doc_chunk in enumerate(chunker.chunk(doc)):
        refs = [item.self_ref for item in doc_chunk.meta.doc_items if item.self_ref in spans]
        located = _locate(
            doc_chunk.text, [texts[ref] for ref in refs], cursors[refs[0]], chunker.delim
        )
        if located is None:
            raise UnlocatableChunkError(ordinal)
        start, end = located
        cursors[refs[-1]] = end
        first = doc_chunk.meta.doc_items[0]
        text = chunker.contextualize(doc_chunk)
        token_count = tokenizer.count_tokens(text) + special_tokens
        if token_count > tokenizer.get_max_tokens() + special_tokens:
            # The embedder truncates what doesn't fit, so the tail is lost from search.
            logger.error("chunk %d has %d tokens and will be truncated", ordinal, token_count)
        prov = next((p for item in doc_chunk.meta.doc_items for p in item.prov), None)
        chunks.append(
            Chunk(
                ordinal=ordinal,
                char_start=spans[refs[0]][0] + start,
                char_end=spans[refs[-1]][0] + end,
                page=prov.page_no if prov else None,
                bbox=prov.bbox.model_dump(mode="json") if prov else None,
                modality="table" if first.label == DocItemLabel.TABLE else "text",
                text=text,
                token_count=token_count,
            )
        )
    return chunks


def _locate(text: str, item_texts: list[str], cursor: int, delim: str) -> tuple[int, int] | None:
    """Where text starts in the first item and ends in the last, relative to each item.

    The chunker joins items with delim; a chunk can begin partway through its first
    item and end partway through its last when an item was split.
    """
    if len(item_texts) == 1:
        start = item_texts[0].find(text, cursor)
        return None if start < 0 else (start, start + len(text))
    first, last = item_texts[0], item_texts[-1]
    start = next((k for k in range(cursor, len(first)) if text.startswith(first[k:] + delim)), None)
    end = next((e for e in range(len(last), 0, -1) if text.endswith(delim + last[:e])), None)
    return None if start is None or end is None else (start, end)


# Fixed so chunk IDs are the same on every machine.
CHUNK_NAMESPACE = uuid.UUID("6f0d1c4e-3b9a-4a57-9f53-2c1e8d7b5a10")


def config_hash(index: Index, tokenizer: BaseTokenizer) -> str:
    """Everything that changes chunk boundaries, so two configs never share chunk IDs."""
    dense = index.bundle.dense
    config = [
        index.chunker_version,
        f"{dense.repo_id}@{dense.revision}",
        tokenizer.get_max_tokens(),
    ]
    return hashlib.sha256(json.dumps(config).encode()).hexdigest()


def chunk_id(doc_hash: str, config: str, char_start: int, char_end: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_hash}:{config}:{char_start}:{char_end}"))


def write_chunks(
    conn: sqlite3.Connection,
    index: Index,
    tokenizer: BaseTokenizer,
    doc_hash: str,
    doc: DoclingDocument,
    special_tokens: int = 0,
) -> int:
    """Chunk doc into the index's chunks rows, replacing any from an earlier attempt."""
    config = config_hash(index, tokenizer)
    chunks = chunk_document(doc, tokenizer, special_tokens)
    rows = [
        (
            chunk_id(doc_hash, config, c.char_start, c.char_end),
            index.index_id,
            doc_hash,
            c.ordinal,
            c.char_start,
            c.char_end,
            c.page,
            json.dumps(c.bbox) if c.bbox else None,
            c.modality,
            c.text,
            json.dumps({index.bundle.dense.repo_id: c.token_count}),
        )
        for c in chunks
    ]
    with conn:
        conn.execute(
            "DELETE FROM chunks WHERE index_id = ? AND doc_hash = ?", (index.index_id, doc_hash)
        )
        conn.executemany(
            "INSERT INTO chunks (chunk_id, index_id, doc_hash, ordinal, char_start, char_end,"
            " page, bbox, modality, text, token_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)
