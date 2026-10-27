"""VectorStore backed by Qdrant."""

from qdrant_client import AsyncQdrantClient, models

UPSERT_BATCH = 256

# Fields search filters on. In-memory Qdrant ignores these; only a server applies them.
PAYLOAD_INDEXES = {
    "doc_hash": models.PayloadSchemaType.KEYWORD,
    "modality": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "added_at": models.PayloadSchemaType.DATETIME,
}


class QdrantStore:
    def __init__(self, client: AsyncQdrantClient):
        self.client = client

    async def create_collection(self, collection: str, alias: str, dense_dim: int) -> None:
        """Create one index's collection and point the alias (the index name) at it."""
        await self.client.create_collection(
            collection,
            vectors_config={
                "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
            },
            # Qdrant applies IDF at query time; FastEmbed supplies term frequencies.
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            },
        )
        for field, schema in PAYLOAD_INDEXES.items():
            await self.client.create_payload_index(collection, field, schema, wait=True)
        await self.client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)
                )
            ]
        )

    async def upsert(self, collection: str, points: list[models.PointStruct]) -> None:
        for start in range(0, len(points), UPSERT_BATCH):
            await self.client.upsert(collection, points[start : start + UPSERT_BATCH], wait=True)
