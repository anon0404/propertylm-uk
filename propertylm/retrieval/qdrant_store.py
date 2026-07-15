from __future__ import annotations
from qdrant_client import QdrantClient, models

PAYLOAD_INDEXES = ["postcode", "sector", "district", "number"]

class QdrantStore:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url, timeout=120)
        self.collection = collection

    def create(self, dimension: int, recreate: bool = False):
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE, on_disk=True),
                hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=128),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=50000),
            )
            for field in PAYLOAD_INDEXES:
                self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)

    def search(self, vector: list[float], field: str | None, value: str | None, limit: int):
        query_filter = None
        if field and value:
            query_filter = models.Filter(must=[models.FieldCondition(key=field, match=models.MatchValue(value=value))])
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points
