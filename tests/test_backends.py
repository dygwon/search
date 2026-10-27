import math

import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument
from locus_core.embedders import FastEmbedBm25, SentenceTransformerEmbedder
from locus_ingest.chunk import chunk_document
from locus_ingest.tokenizer import hf_tokenizer
from transformers import AutoTokenizer

from tests.fakes import BGE_SMALL

# Models download from the Hugging Face Hub on first use.
pytestmark = pytest.mark.network


def test_no_chunk_exceeds_the_bge_ceiling_counting_special_tokens():
    doc = DoclingDocument(name="long")
    for section in range(5):
        doc.add_heading(f"Section {section} on quarterly revenue and operating costs")
        for para in range(4):
            words = " ".join(f"figure{section}x{para}x{i} rose sharply." for i in range(150))
            doc.add_text(label=DocItemLabel.TEXT, text=words)
    tokenizer, special_tokens = hf_tokenizer(BGE_SMALL, budget=512)
    reference = AutoTokenizer.from_pretrained(BGE_SMALL.repo_id, revision=BGE_SMALL.revision)

    chunks = chunk_document(doc, tokenizer, special_tokens)

    assert len(chunks) > 5
    lengths = [len(reference(chunk.text)["input_ids"]) for chunk in chunks]
    assert max(lengths) <= 512
    assert [c.token_count for c in chunks] == lengths


def test_bge_embeds_unit_vectors_with_the_index_dimension():
    embedder = SentenceTransformerEmbedder(BGE_SMALL)

    [doc_vector] = embedder.embed_documents(["Revenue rose in the third quarter."])
    query_vector = embedder.embed_query("how did revenue change")

    assert len(doc_vector) == len(query_vector) == 384
    assert math.isclose(math.sqrt(sum(x * x for x in doc_vector)), 1.0, rel_tol=1e-4)


def test_bge_ranks_a_relevant_passage_above_an_unrelated_one():
    embedder = SentenceTransformerEmbedder(BGE_SMALL)
    relevant, unrelated = embedder.embed_documents(
        ["Revenue rose 12% in the third quarter.", "The cat slept on the warm windowsill."]
    )
    query = embedder.embed_query("quarterly revenue growth")

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, relevant) > dot(query, unrelated)


def test_bm25_gives_shared_terms_shared_indices():
    bm25 = FastEmbedBm25()

    [doc] = bm25.embed_documents(["Revenue rose in the third quarter."])
    query = bm25.embed_query("revenue")

    assert len(doc.indices) == len(doc.values) > 0
    assert set(query.indices) <= set(doc.indices)
