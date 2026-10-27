import math

from locus_core.protocols import ScoredChunk
from locus_search.fuse import RRFFuser


def ranked(*chunk_ids):
    # Raw scores don't matter to RRF; only the order does.
    return [ScoredChunk(chunk_id, 1.0 - i / 10) for i, chunk_id in enumerate(chunk_ids)]


def test_rrf_sums_reciprocal_ranks_across_lists():
    fused = RRFFuser(k=60).fuse({"dense": ranked("a", "b", "c"), "bm25": ranked("c", "a", "b")})

    assert [hit.chunk_id for hit in fused] == ["a", "c", "b"]
    assert math.isclose(fused[0].score, 1 / 61 + 1 / 62)
    assert math.isclose(fused[1].score, 1 / 63 + 1 / 61)
    assert math.isclose(fused[2].score, 1 / 62 + 1 / 63)


def test_a_chunk_in_only_one_list_gets_the_missing_score_from_the_other():
    fused = RRFFuser(k=60).fuse({"dense": ranked("a", "b"), "bm25": ranked("b", "z")})

    assert [hit.chunk_id for hit in fused] == ["b", "a", "z"]
    scores = {hit.chunk_id: hit.score for hit in fused}
    assert math.isclose(scores["a"], 1 / 61)
    assert math.isclose(scores["z"], 1 / 62)


def test_the_missing_score_is_explicit_and_configurable():
    fused = RRFFuser(k=60, missing=0.01).fuse({"dense": ranked("a"), "bm25": ranked("b")})

    assert {hit.chunk_id: hit.score for hit in fused} == {
        "a": 1 / 61 + 0.01,
        "b": 0.01 + 1 / 61,
    }


def test_ties_keep_the_earlier_lists_order():
    fused = RRFFuser().fuse({"dense": ranked("a"), "bm25": ranked("b")})

    assert [hit.chunk_id for hit in fused] == ["a", "b"]


def test_empty_lists_fuse_to_nothing():
    assert RRFFuser().fuse({"dense": [], "bm25": []}) == []
