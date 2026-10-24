import asyncio
import os
import sqlite3
import subprocess
from dataclasses import dataclass
from importlib.resources import files
from typing import Annotated

import typer
from locus_core.db import connect
from locus_core.indexes import DEFAULT_TARGET, IndexExistsError, create_index, list_indexes
from locus_core.models import ModelResolver
from locus_core.qdrant_store import QdrantStore
from locus_core.settings import Settings

app = typer.Typer(no_args_is_help=True, help="Local, search-first RAG for sensitive documents.")
index_app = typer.Typer(no_args_is_help=True, help="Create and inspect indexes.")
app.add_typer(index_app, name="index")

COMPOSE_FILE = files("locus") / "compose.yaml"


@dataclass
class Services:
    conn: sqlite3.Connection
    store: QdrantStore
    resolver: ModelResolver


def build_services(settings: Settings) -> Services:
    """The one place real backends are constructed; tests replace it."""
    from locus_core.models import HfModelResolver
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, prefer_grpc=True
    )
    return Services(
        conn=connect(settings.data_dir),
        store=QdrantStore(client),
        resolver=HfModelResolver(),
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


@app.callback()
def main() -> None:
    pass
