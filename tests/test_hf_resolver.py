"""Resolves real Hub metadata. Needs network: run with `pytest -m network`."""

import re

import pytest
from locus_core.models import HfModelResolver

pytestmark = pytest.mark.network


def test_resolves_bge_small_to_a_pinned_commit_with_its_limits():
    model = HfModelResolver().resolve("BAAI/bge-small-en-v1.5")

    assert re.fullmatch(r"[0-9a-f]{40}", model.revision)
    assert (model.max_seq_length, model.dim) == (512, 384)


def test_falls_back_to_the_tokenizer_limit_without_a_sentence_transformers_config():
    model = HfModelResolver().resolve("google-bert/bert-base-uncased")

    assert (model.max_seq_length, model.dim) == (512, 768)
