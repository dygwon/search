"""Index collections on a real Qdrant. Needs Docker: run with `pytest -m docker`."""

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from qdrant_client import AsyncQdrantClient, models

from tests.fakes import BGE_SMALL, FakeResolver

pytestmark = [pytest.mark.docker, pytest.mark.anyio]


async def test_index_collection_has_payload_indexes_for_filters(qdrant, tmp_path):
    client = AsyncQdrantClient(url="http://127.0.0.1:6333", prefer_grpc=True)
    index = await create_index(
        connect(tmp_path), QdrantStore(client), FakeResolver(BGE_SMALL), "notes"
    )

    schema = (await client.get_collection(index.collection)).payload_schema

    assert {field: info.data_type for field, info in schema.items()} == {
        "doc_hash": models.PayloadSchemaType.KEYWORD,
        "modality": models.PayloadSchemaType.KEYWORD,
        "title": models.PayloadSchemaType.TEXT,
        "added_at": models.PayloadSchemaType.DATETIME,
    }
