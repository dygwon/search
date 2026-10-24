import os
import subprocess
from importlib.resources import files

import typer
from locus_core.settings import Settings

app = typer.Typer(no_args_is_help=True, help="Local, search-first RAG for sensitive documents.")

COMPOSE_FILE = files("locus") / "compose.yaml"


@app.command()
def up() -> None:
    """Start Qdrant."""
    settings = Settings()
    env = {**os.environ, "LOCUS_DATA_DIR": str(settings.data_dir)}
    result = subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"], env=env)
    if result.returncode != 0:
        typer.echo("docker compose failed; is the Docker daemon running?", err=True)
        raise typer.Exit(result.returncode)


@app.callback()
def main() -> None:
    pass
