"""Model identity: every model in a bundle is pinned to a Hugging Face commit."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str
    max_seq_length: int
    dim: int


class ModelResolver(Protocol):
    def resolve(self, repo_id: str) -> ResolvedModel: ...


# Tokenizers without a real limit report a huge sentinel (often 1e30).
_NO_LIMIT = 1_000_000


class HfModelResolver:
    """Pins a Hub model to its current commit and reads limits from its config files."""

    def resolve(self, repo_id: str) -> ResolvedModel:
        revision = HfApi().model_info(repo_id).sha

        def config(filename: str) -> dict:
            try:
                path = hf_hub_download(repo_id, filename, revision=revision)
            except EntryNotFoundError:
                return {}
            return json.loads(Path(path).read_text())

        limits = [
            config("sentence_bert_config.json").get("max_seq_length"),
            config("tokenizer_config.json").get("model_max_length"),
        ]
        limits = [limit for limit in limits if limit is not None and limit < _NO_LIMIT]
        if not limits:
            raise ValueError(f"{repo_id} declares no max sequence length")
        return ResolvedModel(
            repo_id=repo_id,
            revision=revision,
            max_seq_length=min(limits),
            # Assumes no projection layer after pooling, true for BERT-style encoders.
            dim=config("config.json")["hidden_size"],
        )
