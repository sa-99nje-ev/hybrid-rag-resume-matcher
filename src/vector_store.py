"""Persistent server-backed Qdrant chunk index."""
import uuid
from qdrant_client import QdrantClient, models

class Store:
    def __init__(self, url='http://127.0.0.1:6333', collection='financial_passages'):
        if url.startswith('http://localhost:'):
            url = url.replace('http://localhost:', 'http://127.0.0.1:', 1)
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection

    def ensure(self, dimension):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(collection_name=self.collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, full_scan_threshold=10),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1))
            self.client.create_payload_index(self.collection, field_name='kind', field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
        info = self.client.get_collection(self.collection)
        if info.config.params.vectors.size != dimension or info.config.params.vectors.distance != models.Distance.COSINE:
            raise ValueError('Existing collection dimension or metric differs; use a different collection')
        return info

    def upsert(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError('chunks and vectors differ in length')
        points = [models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c['chunk_id'])),
                                    vector=v.tolist(), payload=c) for c, v in zip(chunks, vectors)]
        for start in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[start:start+128], wait=True)

    def search(self, vector, limit=30, kind=None, exact=False):
        f = models.Filter(must=[models.FieldCondition(key='kind', match=models.MatchValue(value=kind))]) if kind else None
        return self.client.query_points(self.collection, query=vector.tolist(), query_filter=f,
            search_params=models.SearchParams(exact=exact, hnsw_ef=128), limit=limit, with_payload=True).points
