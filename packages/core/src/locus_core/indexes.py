"""Indexes: immutable bundles of models and chunk settings chosen at creation."""

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass

from locus_core.models import ModelResolver, ResolvedModel
from locus_core.qdrant_store import QdrantStore

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
CHUNKER_VERSION = "docling-hybrid/1"
DEFAULT_TARGET = 512

INDEX_COLUMNS = (
    "index_id, name, model_bundle, chunk_target, chunk_ceiling, chunker_version,"
    " collection, alias, status"
)


class IndexExistsError(Exception):
    def __init__(self, name: str):
        super().__init__(f"an index named {name!r} already exists")
        self.name = name


class IndexNotFoundError(Exception):
    def __init__(self, name: str):
        super().__init__(f"no index named {name!r}; create it with `locus index create {name}`")
        self.name = name


@dataclass(frozen=True)
class ModelBundle:
    dense: ResolvedModel


@dataclass(frozen=True)
class Index:
    index_id: str
    name: str
    bundle: ModelBundle
    chunk_target: int
    chunk_ceiling: int
    chunker_version: str
    collection: str
    alias: str
    status: str


async def create_index(
    conn: sqlite3.Connection,
    store: QdrantStore,
    resolver: ModelResolver,
    name: str,
    target: int = DEFAULT_TARGET,
) -> Index:
    # Indexes are immutable, so a taken name is refused before anything is created.
    if conn.execute("SELECT 1 FROM indexes WHERE name = ?", (name,)).fetchone():
        raise IndexExistsError(name)
    bundle = ModelBundle(dense=resolver.resolve(DENSE_MODEL))
    # The ceiling is the smallest max sequence length across the bundle.
    ceiling = min(model.max_seq_length for model in (bundle.dense,))
    index_id = uuid.uuid4().hex
    index = Index(
        index_id=index_id,
        name=name,
        bundle=bundle,
        chunk_target=min(target, ceiling),
        chunk_ceiling=ceiling,
        chunker_version=CHUNKER_VERSION,
        collection=f"locus_{index_id}",
        alias=name,
        status="ready",
    )
    await store.create_collection(index.collection, index.alias, bundle.dense.dim)
    with conn:
        conn.execute(
            f"INSERT INTO indexes ({INDEX_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                index.index_id,
                index.name,
                json.dumps(asdict(index.bundle)),
                index.chunk_target,
                index.chunk_ceiling,
                index.chunker_version,
                index.collection,
                index.alias,
                index.status,
            ),
        )
    return index


def get_index(conn: sqlite3.Connection, name: str) -> Index:
    row = conn.execute(f"SELECT {INDEX_COLUMNS} FROM indexes WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise IndexNotFoundError(name)
    return _index_from_row(row)


def list_indexes(conn: sqlite3.Connection) -> list[Index]:
    rows = conn.execute(f"SELECT {INDEX_COLUMNS} FROM indexes ORDER BY name").fetchall()
    return [_index_from_row(row) for row in rows]


def _index_from_row(row: tuple) -> Index:
    index_id, name, bundle_json, *rest = row
    bundle = json.loads(bundle_json)
    return Index(index_id, name, ModelBundle(dense=ResolvedModel(**bundle["dense"])), *rest)
