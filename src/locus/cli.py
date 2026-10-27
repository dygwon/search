import asyncio
import os
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Annotated

import typer
from locus_core.db import connect
from locus_core.indexes import (
    DEFAULT_TARGET,
    Index,
    IndexExistsError,
    IndexNotFoundError,
    create_index,
    get_index,
    list_indexes,
)
from locus_core.models import ModelResolver
from locus_core.qdrant_store import QdrantStore
from locus_core.settings import Settings
from locus_ingest.pipeline import Backends, ingest_file
from locus_search.search import CANDIDATES, DEFAULT_LIMIT, Encoders, Hit, search

app = typer.Typer(no_args_is_help=True, help="Local, search-first RAG for sensitive documents.")
index_app = typer.Typer(no_args_is_help=True, help="Create and inspect indexes.")
app.add_typer(index_app, name="index")

COMPOSE_FILE = files("locus") / "compose.yaml"


@dataclass
class Services:
    conn: sqlite3.Connection
    store: QdrantStore
    resolver: ModelResolver
    # The tokenizer and embedders follow the index's pinned bundle.
    backends: Callable[[Index], Backends]
    encoders: Callable[[Index], Encoders]


def build_services(settings: Settings) -> Services:
    """The one place real backends are constructed; tests replace it."""
    from locus_core.models import HfModelResolver
    from qdrant_client import AsyncQdrantClient

    from locus.backends import real_backends, real_encoders

    client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, prefer_grpc=True
    )
    return Services(
        conn=connect(settings.data_dir),
        store=QdrantStore(client),
        resolver=HfModelResolver(),
        backends=real_backends,
        encoders=real_encoders,
    )


@app.command()
def up() -> None:
    """Start Qdrant."""
    settings = Settings()
    env = {**os.environ, "LOCUS_DATA_DIR": str(settings.data_dir)}
    result = subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"], env=env)
    if result.returncode != 0:
        typer.echo("docker compose failed; is the Docker daemon running?", err=True)
        raise typer.Exit(result.returncode)


@index_app.command("create")
def index_create(
    name: str,
    target: Annotated[int, typer.Option(help="Chunk target in tokens.")] = DEFAULT_TARGET,
) -> None:
    """Create an index with the default model bundle."""
    services = build_services(Settings())
    try:
        index = asyncio.run(
            create_index(services.conn, services.store, services.resolver, name, target)
        )
    except IndexExistsError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    if index.chunk_target < target:
        typer.echo(f"Target {target} exceeds the ceiling; capped to {index.chunk_target}")
    typer.echo(
        f"Created index {name!r}: chunk target {index.chunk_target} tokens"
        f" (ceiling {index.chunk_ceiling})"
    )


@index_app.command("list")
def index_list() -> None:
    """List indexes with their dense model and chunk target."""
    indexes = list_indexes(build_services(Settings()).conn)
    width = max((len(index.name) for index in indexes), default=0)
    for index in indexes:
        dense = index.bundle.dense
        typer.echo(
            f"{index.name:<{width}}  {dense.repo_id}@{dense.revision[:7]}"
            f"  target {index.chunk_target}"
        )


@app.command()
def ingest(
    paths: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, readable=True, help="PDFs to add.")
    ],
    index: Annotated[str, typer.Option(help="Index to add the files to.")],
) -> None:
    """Chunk, embed and index PDFs. Files are read in place, never copied."""
    settings = Settings()
    services = build_services(settings)
    try:
        target = get_index(services.conn, index)
    except IndexNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    backends = services.backends(target)

    # One event loop for the whole batch: the gRPC client is bound to the loop it starts on.
    async def ingest_all() -> bool:
        failed = False
        for path in paths:
            try:
                result = await ingest_file(
                    services.conn, services.store, backends, settings.data_dir, target, path
                )
            # One bad file shouldn't stop the batch; its jobs row keeps the error.
            except Exception as exc:
                typer.echo(f"Error: {path}: {exc}", err=True)
                failed = True
                continue
            chunks = f"{result.chunks} chunk{'s' if result.chunks != 1 else ''}"
            typer.echo(
                f"{result.status:<9}  {result.doc_hash[:12]}  {result.title}"
                f"  ({chunks}, {path.name})"
            )
        return failed

    if asyncio.run(ingest_all()):
        raise typer.Exit(1)


@app.command("search")
def search_command(
    query: str,
    index: Annotated[str, typer.Option(help="Index to search.")],
    limit: Annotated[int, typer.Option(min=1, help="Documents to show.")] = DEFAULT_LIMIT,
    candidates: Annotated[
        int, typer.Option(min=1, help="Candidates retrieved per list before fusion.")
    ] = CANDIDATES,
) -> None:
    """Search an index: dense and BM25 retrieval fused with RRF, best chunk per document."""
    services = build_services(Settings())
    try:
        target = get_index(services.conn, index)
    except IndexNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    hits = asyncio.run(
        search(services.store, services.encoders(target), target, query, limit, candidates)
    )
    if not hits:
        typer.echo("No results.")
    for rank, hit in enumerate(hits, 1):
        typer.echo(_format_hit(rank, hit))


SNIPPET_CHARS = 240


def _format_hit(rank: int, hit: Hit) -> str:
    page = f"  p. {hit.page}" if hit.page is not None else ""
    scores = "  ".join(f"{name} {_score(hit.scores.get(name))}" for name in ("dense", "bm25"))
    text = " ".join(hit.text.split())
    snippet = text if len(text) <= SNIPPET_CHARS else text[: SNIPPET_CHARS - 1].rstrip() + "…"
    return f"{rank}. {hit.title}{page}\n   rrf {hit.score:.4f}  {scores}\n   {snippet}\n"


def _score(value: float | None) -> str:
    # A list that didn't return the chunk has no score for it.
    return "-" if value is None else f"{value:.3f}"


@app.callback()
def main() -> None:
    pass
