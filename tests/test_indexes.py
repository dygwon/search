import pytest
from locus_core.db import connect
from locus_core.indexes import IndexExistsError, create_index, get_index, list_indexes
from locus_core.qdrant_store import QdrantStore
from qdrant_client import AsyncQdrantClient, models

from tests.fakes import BGE_SMALL, FakeResolver

pytestmark = pytest.mark.anyio


@pytest.fixture
def conn(tmp_path):
    return connect(tmp_path)


@pytest.fixture
def client():
    return AsyncQdrantClient(":memory:")


@pytest.fixture
def store(client):
    return QdrantStore(client)


async def test_created_index_pins_the_dense_model_revision(conn, store):
    await create_index(conn, store, FakeResolver(BGE_SMALL), "notes")

    index = get_index(conn, "notes")

    assert index.bundle.dense == BGE_SMALL


async def test_target_above_the_ceiling_is_capped(conn, store):
    index = await create_index(conn, store, FakeResolver(BGE_SMALL), "notes", target=2000)

    assert (index.chunk_target, index.chunk_ceiling) == (512, 512)


async def test_target_below_the_ceiling_is_kept(conn, store):
    index = await create_index(conn, store, FakeResolver(BGE_SMALL), "notes", target=256)

    assert (index.chunk_target, index.chunk_ceiling) == (256, 512)


async def test_index_alias_names_a_collection_with_dense_and_bm25_vectors(conn, store, client):
    await create_index(conn, store, FakeResolver(BGE_SMALL), "notes")

    [alias] = (await client.get_aliases()).aliases
    params = (await client.get_collection(alias.collection_name)).config.params

    assert alias.alias_name == "notes"
    assert params.vectors == {
        "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
    }
    assert params.sparse_vectors == {
        "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
    }


async def test_duplicate_name_is_refused_and_changes_nothing(conn, store, client):
    resolver = FakeResolver(BGE_SMALL)
    original = await create_index(conn, store, resolver, "notes")

    with pytest.raises(IndexExistsError):
        await create_index(conn, store, resolver, "notes")

    assert [index.index_id for index in list_indexes(conn)] == [original.index_id]
    assert [c.name for c in (await client.get_collections()).collections] == [original.collection]
    assert [a.collection_name for a in (await client.get_aliases()).aliases] == [
        original.collection
    ]
