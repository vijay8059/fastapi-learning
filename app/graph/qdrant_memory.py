import asyncio
import uuid
from functools import lru_cache

import truststore
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import AsyncQdrantClient, models

from app.core.config import Settings

# Semantic long-term memory — the third leg alongside pg_memory's exact-key
# facts and InMemorySaver's thread-scoped checkpoints. Where recall_fact
# needs an exact key, this answers "what does the user's uploaded content
# say about X", ranked by meaning *and* keyword overlap (see the hybrid
# dense + BM25 setup below) rather than an exact match.

_DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_DENSE_DIM = 384
_DENSE_VECTOR_NAME = "dense"

_BM25_MODEL_NAME = "Qdrant/bm25"
_BM25_VECTOR_NAME = "bm25"

# huggingface_hub's first-run model download otherwise fails behind this
# network's TLS-inspecting proxy (self-signed cert in the chain) — this
# patches the stdlib ssl module to trust the OS certificate store instead.
truststore.inject_into_ssl()


@lru_cache
def _dense_model() -> TextEmbedding:
    # Built once and reused — reloading the ONNX model on every call would
    # be as wasteful as rebuilding docling's DocumentConverter per request.
    return TextEmbedding(model_name=_DENSE_MODEL_NAME)


@lru_cache
def _sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=_BM25_MODEL_NAME)


async def _embed_dense(texts: list[str]) -> list[list[float]]:
    # FastEmbed's .embed() is a synchronous, CPU-bound ONNX call — run it off
    # the event loop the same way a blocking docling conversion would need to be.
    def _run() -> list[list[float]]:
        return [vec.tolist() for vec in _dense_model().embed(texts)]

    return await asyncio.to_thread(_run)


async def _embed_sparse(texts: list[str]) -> list[SparseEmbedding]:
    def _run() -> list[SparseEmbedding]:
        return list(_sparse_model().embed(texts))

    return await asyncio.to_thread(_run)


def _sparse_vector(embedding: SparseEmbedding) -> models.SparseVector:
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def _client(settings: Settings) -> AsyncQdrantClient:
    return AsyncQdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _has_expected_schema(info: models.CollectionInfo) -> bool:
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors
    return (
        isinstance(vectors, dict)
        and _DENSE_VECTOR_NAME in vectors
        and bool(sparse_vectors)
        and _BM25_VECTOR_NAME in sparse_vectors
    )


async def ensure_collection(settings: Settings) -> None:
    """Idempotent — safe to call on every app startup.

    Recreates the collection if it exists with an incompatible schema (e.g.
    the single-unnamed-dense-vector layout from before BM25 support was
    added) — acceptable here since this is local dev/learning data, not a
    migration a production system could get away with.
    """

    client = _client(settings)
    try:
        exists = await client.collection_exists(settings.qdrant_collection)
        if exists and not _has_expected_schema(await client.get_collection(settings.qdrant_collection)):
            await client.delete_collection(settings.qdrant_collection)
            exists = False

        if not exists:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config={
                    _DENSE_VECTOR_NAME: models.VectorParams(
                        size=_DENSE_DIM, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    _BM25_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
    finally:
        await client.close()


async def index_chunks(
    settings: Settings, attachment_id: int, username: str, filename: str, thread_id: str, chunks: list[str]
) -> int:
    if not chunks:
        return 0

    dense_vectors = await _embed_dense(chunks)
    sparse_vectors = await _embed_sparse(chunks)
    points = [
        models.PointStruct(
            id=str(uuid.uuid4()),
            vector={_DENSE_VECTOR_NAME: dense, _BM25_VECTOR_NAME: _sparse_vector(sparse)},
            payload={
                "attachment_id": attachment_id,
                "username": username,
                "filename": filename,
                "thread_id": thread_id,
                "chunk_index": index,
                "text": chunk,
            },
        )
        for index, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors))
    ]

    client = _client(settings)
    try:
        await client.upsert(collection_name=settings.qdrant_collection, points=points)
    finally:
        await client.close()
    return len(points)


async def search(settings: Settings, username: str, query: str, limit: int = 4) -> list[dict]:
    """Hybrid retrieval: dense (semantic) and BM25 (keyword) candidates are
    fetched in parallel, then fused with Reciprocal Rank Fusion — catches
    both "means the same thing" and "uses the same rare term" matches,
    which neither method alone reliably does.

    Scoped to `username` on both prefetch legs — search_documents only ever
    surfaces the caller's own uploads, matching learning.attachments (see
    pg_memory.list_attachments), not every document anyone's ever indexed.
    """

    (dense_vector,) = await _embed_dense([query])
    (sparse_vector,) = await _embed_sparse([query])
    owner_filter = models.Filter(
        must=[models.FieldCondition(key="username", match=models.MatchValue(value=username))]
    )

    client = _client(settings)
    try:
        results = await client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector, using=_DENSE_VECTOR_NAME, limit=20, filter=owner_filter
                ),
                models.Prefetch(
                    query=_sparse_vector(sparse_vector),
                    using=_BM25_VECTOR_NAME,
                    limit=20,
                    filter=owner_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
    finally:
        await client.close()

    return [
        {
            "filename": point.payload["filename"],
            "text": point.payload["text"],
            "score": point.score,
        }
        for point in results.points
    ]
