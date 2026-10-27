"""Chunk: split a parsed document into token-bounded chunks with character offsets.

Offsets point into the document text, each item as the chunker's serializer renders it
(docs/adr/0001-chunk-offsets.md). Plain text offsets are the same in every index; list,
table and heading offsets can shift with a Docling upgrade.
"""

import hashlib
import json
import logging
import sqlite3
import uuid
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version

from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DocItemLabel, DoclingDocument, NodeItem, TextItem
from locus_core.indexes import Index

ITEM_SEPARATOR = "\n\n"
MERGE_PEERS = True
# How chunk spans are worked out (docs/adr/0002-item-level-chunk-spans.md).
OFFSETS_SCHEME = "item-level"
# Fixed so chunk IDs are the same on every machine.
CHUNK_NAMESPACE = uuid.UUID("6f0d1c4e-3b9a-4a57-9f53-2c1e8d7b5a10")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    char_start: int
    char_end: int
    # Earlier chunks of the document with the same span; >0 only for a split table's pieces.
    piece: int
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
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=MERGE_PEERS)
    texts = item_texts(doc, chunker)
    spans = item_spans(texts)
    # Where the last chunk ended inside each item, so repeated text isn't matched twice.
    cursors = dict.fromkeys(spans, 0)
    # Chunks seen per span, so the pieces of a split table get distinct IDs.
    pieces: Counter[tuple[int, int]] = Counter()
    chunks = []
    for ordinal, doc_chunk in enumerate(chunker.chunk(doc)):
        refs = [_resolve(doc, item, spans) for item in doc_chunk.meta.doc_items]
        if not refs or None in refs:
            raise UnlocatableChunkError(ordinal)
        refs = list(dict.fromkeys(refs))
        span = (spans[refs[0]][0], spans[refs[-1]][1])
        first = doc_chunk.meta.doc_items[0]
        if refs == [first.self_ref] and isinstance(first, TextItem):
            # Part of one text item, maybe a split paragraph: narrow to the chunk's text.
            [ref] = refs
            start = texts[ref].find(doc_chunk.text, cursors[ref])
            if start < 0:
                raise UnlocatableChunkError(ordinal)
            cursors[ref] = start + len(doc_chunk.text)
            span = (span[0] + start, span[0] + cursors[ref])
        piece = pieces[span]
        pieces[span] += 1
        text = chunker.contextualize(doc_chunk)
        token_count = tokenizer.count_tokens(text) + special_tokens
        prov = next((p for item in doc_chunk.meta.doc_items for p in item.prov), None)
        chunks.append(
            Chunk(
                ordinal=ordinal,
                char_start=span[0],
                char_end=span[1],
                piece=piece,
                page=prov.page_no if prov else None,
                bbox=prov.bbox.model_dump(mode="json") if prov else None,
                modality="table" if first.label == DocItemLabel.TABLE else "text",
                text=text,
                token_count=token_count,
            )
        )
    return chunks


def _resolve(doc: DoclingDocument, item: NodeItem, spans: dict[str, tuple[int, int]]) -> str | None:
    """The item, or the nearest ancestor, that has a span: a caption renders inside its parent."""
    while item.self_ref not in spans:
        if item.parent is None:
            return None
        item = item.parent.resolve(doc)
    return item.self_ref


def config_hash(index: Index, tokenizer: BaseTokenizer) -> str:
    """Everything that changes chunk boundaries, so two configs never share chunk IDs."""
    dense = index.bundle.dense
    config = [
        index.chunker_version,
        # The serializer shapes the document text, so its version shapes the offsets.
        f"docling-core/{version('docling-core')}",
        f"{dense.repo_id}@{dense.revision}",
        tokenizer.get_max_tokens(),
        MERGE_PEERS,
        OFFSETS_SCHEME,
    ]
    return hashlib.sha256(json.dumps(config).encode()).hexdigest()


def chunk_id(doc_hash: str, config: str, char_start: int, char_end: int, piece: int) -> str:
    return str(uuid.uuid5(CHUNK_NAMESPACE, f"{doc_hash}:{config}:{char_start}:{char_end}:{piece}"))


def write_chunks(
    conn: sqlite3.Connection,
    index: Index,
    tokenizer: BaseTokenizer,
    doc_hash: str,
    doc: DoclingDocument,
    special_tokens: int,
) -> int:
    """Chunk doc into the index's chunks rows, replacing any from an earlier attempt."""
    config = config_hash(index, tokenizer)
    chunks = chunk_document(doc, tokenizer, special_tokens)
    for c in chunks:
        if c.token_count > index.chunk_ceiling:
            # The embedder truncates what doesn't fit, so the tail is lost from search.
            logger.error("chunk %d has %d tokens and will be truncated", c.ordinal, c.token_count)
    rows = [
        (
            chunk_id(doc_hash, config, c.char_start, c.char_end, c.piece),
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
