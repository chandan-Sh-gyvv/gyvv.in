"""
shsrs_rag/grpc_server.py
========================
gRPC server — mirrors REST API endpoints for high-throughput clients.
Generate stubs first:
  python -m grpc_tools.protoc -I proto --python_out=shsrs_rag/proto_gen
      --grpc_python_out=shsrs_rag/proto_gen proto/shsrs_rag.proto
"""

from __future__ import annotations
import asyncio
import logging
import grpc
from pathlib import Path

from shsrs_rag.config import settings
from shsrs_rag.embedder import embed, embed_one
from shsrs_rag.chunker import load_and_chunk
from shsrs_rag.index_manager import CollectionIndex, registry
from shsrs_rag.auth import get_user_by_username
from shsrs_rag.database import AsyncSessionLocal, Collection, Document, Chunk
from sqlalchemy import select

logger = logging.getLogger(__name__)

# Proto-generated stubs — generate once with: python generate_proto.py
_proto_gen_dir = Path(__file__).parent / "proto_gen"


def _ensure_stubs():
    """Check stubs exist — run generate_proto.py once to create them."""
    if not (_proto_gen_dir / "shsrs_rag_pb2.py").exists():
        raise RuntimeError(
            "gRPC stubs not found. Run once: python generate_proto.py"
        )


class SHSRSRagServicer:
    """gRPC service implementation."""

    async def Health(self, request, context):
        from shsrs_rag.proto_gen import shsrs_rag_pb2
        return shsrs_rag_pb2.HealthResponse(
            status  = "ok",
            version = settings.app_version,
        )

    async def Ingest(self, request, context):
        from shsrs_rag.proto_gen import shsrs_rag_pb2
        from jose import jwt, JWTError

        # Auth
        try:
            payload  = jwt.decode(request.token, settings.secret_key,
                                  algorithms=[settings.algorithm])
            username = payload.get("sub")
        except JWTError:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
            return

        async with AsyncSessionLocal() as db:
            col_result = await db.execute(
                select(Collection).where(Collection.id == request.collection_id)
            )
            col = col_result.scalar_one_or_none()
            if not col:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Collection not found")
                return

            chunks = load_and_chunk(
                request.content,
                request.filename,
                request.chunk_size or None,
                request.chunk_overlap or None,
            )
            if not chunks:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No content found")
                return

            import hashlib
            content_hash = hashlib.sha256(request.content).hexdigest()
            doc = Document(
                collection_id = request.collection_id,
                filename      = request.filename,
                content_hash  = content_hash,
                n_chunks      = len(chunks),
            )
            db.add(doc)
            await db.commit()
            await db.refresh(doc)

            texts      = [c.text for c in chunks]
            embeddings = embed(texts, model=col.model)
            base_id    = col.n_vectors
            vector_ids = list(range(base_id, base_id + len(chunks)))

            for chunk, vid in zip(chunks, vector_ids):
                db.add(Chunk(
                    document_id = doc.id,
                    vector_id   = vid,
                    text        = chunk.text,
                    chunk_index = chunk.chunk_index,
                    token_count = chunk.token_count,
                ))

            col.n_vectors += len(chunks)
            await db.commit()

            asyncio.create_task(_index_collection_grpc(request.collection_id, col.model))

        return shsrs_rag_pb2.IngestResponse(
            document_id = doc.id,
            filename    = request.filename,
            n_chunks    = len(chunks),
            status      = "ingested",
        )

    async def Query(self, request, context):
        from shsrs_rag.proto_gen import shsrs_rag_pb2
        from jose import jwt, JWTError

        # Auth
        try:
            payload = jwt.decode(request.token, settings.secret_key,
                                 algorithms=[settings.algorithm])
        except JWTError:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
            return

        async with AsyncSessionLocal() as db:
            col_result = await db.execute(
                select(Collection).where(Collection.id == request.collection_id)
            )
            col = col_result.scalar_one_or_none()
            if not col:
                await context.abort(grpc.StatusCode.NOT_FOUND, "Collection not found")
                return

            idx = await registry.get_or_load(request.collection_id, col.model)
            if not idx.is_ready:
                await context.abort(grpc.StatusCode.UNAVAILABLE, "Index not ready")
                return

            model_to_use = request.model or col.model
            query_vec    = embed_one(request.query, model=model_to_use)
            probe        = request.probe or settings.default_probe
            raw_results  = await idx.search(query_vec, k=request.k or 5, probe=probe)

            if not raw_results:
                return shsrs_rag_pb2.QueryResponse(
                    query=request.query, results=[], model_used=model_to_use)

            vector_ids = [r[0] for r in raw_results]
            scores_map = {r[0]: r[1] for r in raw_results}

            chunks_result = await db.execute(
                select(Chunk, Document).join(Document).where(
                    Document.collection_id == request.collection_id,
                    Chunk.vector_id.in_(vector_ids),
                )
            )
            rows = chunks_result.all()

        results = []
        for chunk, doc in rows:
            results.append(shsrs_rag_pb2.QueryResult(
                chunk_id    = chunk.id,
                vector_id   = chunk.vector_id,
                score       = scores_map.get(chunk.vector_id, 0.0),
                text        = chunk.text,
                document_id = doc.id,
                filename    = doc.filename,
                chunk_index = chunk.chunk_index,
            ))
        results.sort(key=lambda r: -r.score)

        return shsrs_rag_pb2.QueryResponse(
            query      = request.query,
            results    = results,
            model_used = model_to_use,
        )


async def _index_collection_grpc(collection_id: int, model: str):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Chunk).join(Document).where(
                Document.collection_id == collection_id
            ).order_by(Chunk.vector_id)
        )
        chunks = result.scalars().all()

    if not chunks:
        return

    texts      = [c.text for c in chunks]
    vector_ids = [c.vector_id for c in chunks]
    embeddings = embed(texts, model=model)
    idx        = CollectionIndex(collection_id, model)
    await idx.build(embeddings, vector_ids)
    registry.register(idx)

    async with AsyncSessionLocal() as db:
        col_result = await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )
        col = col_result.scalar_one_or_none()
        if col:
            col.is_indexed = True
            await db.commit()


async def serve_grpc():
    """Start the gRPC server."""
    _ensure_stubs()
    from shsrs_rag.proto_gen import shsrs_rag_pb2_grpc

    server = grpc.aio.server()
    shsrs_rag_pb2_grpc.add_SHSRSRagServicer_to_server(SHSRSRagServicer(), server)
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    await server.start()
    logger.info(f"gRPC server started on port {settings.grpc_port}")
    await server.wait_for_termination()
