"""
shsrs_rag/rest_api.py
=====================
FastAPI REST routes:
  POST   /auth/register
  POST   /auth/token
  POST   /collections
  GET    /collections
  GET    /collections/{id}/stats
  DELETE /collections/{id}                     — deletes DB + index files
  POST   /collections/{id}/ingest
  GET    /collections/{id}/documents           — list documents
  DELETE /collections/{id}/documents/{doc_id} — remove one document + rebuild
  POST   /collections/{id}/query
"""

from __future__ import annotations
import hashlib
import logging
import shutil
import numpy as np
from datetime import timedelta
from pathlib import Path

from fastapi import (APIRouter, Depends, HTTPException, UploadFile,
                     File, Form, status, BackgroundTasks)
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import Optional

from shsrs_rag.config import settings
from shsrs_rag.database import get_db, User, Collection, Document, Chunk
from shsrs_rag.auth import (hash_password, create_access_token,
                              authenticate_user, get_current_user)
from shsrs_rag.embedder import embed, embed_one
from shsrs_rag.chunker import load_and_chunk
from shsrs_rag.index_manager import CollectionIndex, registry

logger = logging.getLogger(__name__)
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email:    str
    password: str = Field(min_length=8)

class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"

class CollectionCreate(BaseModel):
    name:        str = Field(min_length=1, max_length=128)
    description: str = ""
    model:       str = "local"

class CollectionResponse(BaseModel):
    id:          int
    name:        str
    description: str
    model:       str
    n_vectors:   int
    is_indexed:  bool

class IngestResponse(BaseModel):
    document_id: int
    filename:    str
    n_chunks:    int
    status:      str

class DocumentResponse(BaseModel):
    id:         int
    filename:   str
    n_chunks:   int
    created_at: str

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k:     int = Field(default=5, ge=1, le=50)
    probe: Optional[int] = None
    model: Optional[str] = None

class QueryResult(BaseModel):
    chunk_id:    int
    vector_id:   int
    score:       float
    text:        str
    document_id: int
    filename:    str
    chunk_index: int

class QueryResponse(BaseModel):
    query:      str
    results:    list[QueryResult]
    model_used: str

class StatsResponse(BaseModel):
    collection_id: int
    name:          str
    n_documents:   int
    n_chunks:      int
    n_vectors:     int
    is_indexed:    bool
    model:         str


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/auth/register", status_code=201)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(select(User).where(User.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(400, "Username already taken")
    user = User(
        username        = req.username,
        email           = req.email,
        hashed_password = hash_password(req.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": user.id, "username": user.username, "message": "User created"}


@router.post("/auth/token", response_model=TokenResponse)
async def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db:   AsyncSession = Depends(get_db),
):
    user = await authenticate_user(db, form.username, form.password)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Incorrect username or password",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token({"sub": user.username}))


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTIONS
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/collections", response_model=CollectionResponse, status_code=201)
async def create_collection(
    req:          CollectionCreate,
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    if req.model not in ("local", "openai"):
        raise HTTPException(400, "model must be 'local' or 'openai'")
    col = Collection(
        name        = req.name,
        description = req.description,
        owner_id    = current_user.id,
        model       = req.model,
    )
    db.add(col)
    await db.commit()
    await db.refresh(col)
    return CollectionResponse(
        id=col.id, name=col.name, description=col.description,
        model=col.model, n_vectors=0, is_indexed=False
    )


@router.get("/collections", response_model=list[CollectionResponse])
async def list_collections(
    db:           AsyncSession = Depends(get_db),
    current_user: User         = Depends(get_current_user),
):
    result = await db.execute(
        select(Collection).where(Collection.owner_id == current_user.id)
    )
    return [
        CollectionResponse(id=c.id, name=c.name, description=c.description,
                           model=c.model, n_vectors=c.n_vectors, is_indexed=c.is_indexed)
        for c in result.scalars().all()
    ]


@router.get("/collections/{collection_id}/stats", response_model=StatsResponse)
async def collection_stats(
    collection_id: int,
    db:            AsyncSession = Depends(get_db),
    current_user:  User         = Depends(get_current_user),
):
    col = await _get_collection(db, collection_id, current_user.id)

    docs   = (await db.execute(
        select(Document).where(Document.collection_id == collection_id)
    )).scalars().all()
    chunks = (await db.execute(
        select(Chunk).join(Document).where(Document.collection_id == collection_id)
    )).scalars().all()

    return StatsResponse(
        collection_id = col.id,
        name          = col.name,
        n_documents   = len(docs),
        n_chunks      = len(chunks),
        n_vectors     = col.n_vectors,
        is_indexed    = col.is_indexed,
        model         = col.model,
    )


@router.delete("/collections/{collection_id}", status_code=204)
async def delete_collection(
    collection_id: int,
    db:            AsyncSession = Depends(get_db),
    current_user:  User         = Depends(get_current_user),
):
    col = await _get_collection(db, collection_id, current_user.id)

    # Wipe index files from disk
    index_dir = settings.index_base_dir / f"col_{collection_id}"
    if index_dir.exists():
        shutil.rmtree(index_dir)
        logger.info(f"Deleted index dir: {index_dir}")

    # Remove from in-memory registry
    registry._indexes.pop(collection_id, None)

    # Delete children with raw SQL in correct order (bypass ORM cascade confusion)
    doc_ids = (await db.execute(
        select(Document.id).where(Document.collection_id == collection_id)
    )).scalars().all()
    if doc_ids:
        await db.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))
        await db.flush()
    await db.execute(delete(Document).where(Document.collection_id == collection_id))
    await db.flush()
    await db.execute(delete(Collection).where(Collection.id == collection_id))
    await db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/collections/{collection_id}/documents",
            response_model=list[DocumentResponse])
async def list_documents(
    collection_id: int,
    db:            AsyncSession = Depends(get_db),
    current_user:  User         = Depends(get_current_user),
):
    await _get_collection(db, collection_id, current_user.id)

    result = await db.execute(
        select(Document).where(Document.collection_id == collection_id)
        .order_by(Document.created_at)
    )
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id         = d.id,
            filename   = d.filename,
            n_chunks   = d.n_chunks,
            created_at = d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.delete("/collections/{collection_id}/documents/{document_id}",
               status_code=200)
async def delete_document(
    collection_id:    int,
    document_id:      int,
    background_tasks: BackgroundTasks,
    db:               AsyncSession = Depends(get_db),
    current_user:     User         = Depends(get_current_user),
):
    col = await _get_collection(db, collection_id, current_user.id)

    # Load document
    doc_result = await db.execute(
        select(Document).where(
            Document.id            == document_id,
            Document.collection_id == collection_id,
        )
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "Document not found")

    n_chunks_removed = doc.n_chunks

    # Delete chunks then document
    await db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await db.delete(doc)

    # Update collection vector count
    col.n_vectors  = max(0, col.n_vectors - n_chunks_removed)
    col.is_indexed = False
    await db.commit()

    # Rebuild index in background with remaining documents
    background_tasks.add_task(_rebuild_collection, collection_id, col.model)

    return {
        "message":          f"Document '{doc.filename}' deleted",
        "chunks_removed":   n_chunks_removed,
        "status":           "index rebuilding in background",
    }


# ══════════════════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/collections/{collection_id}/ingest",
             response_model=IngestResponse, status_code=201)
async def ingest_document(
    collection_id:    int,
    background_tasks: BackgroundTasks,
    file:             UploadFile   = File(...),
    chunk_size:       int          = Form(default=None),
    chunk_overlap:    int          = Form(default=None),
    db:               AsyncSession = Depends(get_db),
    current_user:     User         = Depends(get_current_user),
):
    col = await _get_collection(db, collection_id, current_user.id)

    filename = file.filename or "upload.txt"
    if not filename.lower().endswith(('.txt', '.md', '.markdown')):
        raise HTTPException(400, "Only .txt and .md files supported in v1")

    content      = await file.read()
    content_hash = hashlib.sha256(content).hexdigest()

    # Reject duplicates
    dup = await db.execute(
        select(Document).where(
            Document.collection_id == collection_id,
            Document.content_hash  == content_hash,
        )
    )
    if dup.scalar_one_or_none():
        raise HTTPException(409, "Document already ingested (duplicate content)")

    chunks = load_and_chunk(content, filename, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(422, "No text content found in document")

    doc = Document(
        collection_id = collection_id,
        filename      = filename,
        content_hash  = content_hash,
        n_chunks      = len(chunks),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Embed + assign vector IDs
    texts      = [c.text for c in chunks]
    embeddings = embed(texts, model=col.model)
    base_id    = col.n_vectors
    vector_ids = list(range(base_id, base_id + len(chunks)))

    db.add_all([
        Chunk(
            document_id = doc.id,
            vector_id   = vid,
            text        = chunk.text,
            chunk_index = chunk.chunk_index,
            token_count = chunk.token_count,
        )
        for chunk, vid in zip(chunks, vector_ids)
    ])

    col.n_vectors  += len(chunks)
    col.is_indexed  = False
    await db.commit()

    background_tasks.add_task(_index_collection, collection_id, col.model)

    return IngestResponse(
        document_id = doc.id,
        filename    = filename,
        n_chunks    = len(chunks),
        status      = "ingested — indexing in background",
    )


# ══════════════════════════════════════════════════════════════════════════════
# QUERY
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/collections/{collection_id}/query", response_model=QueryResponse)
async def query_collection(
    collection_id: int,
    req:           QueryRequest,
    db:            AsyncSession = Depends(get_db),
    current_user:  User         = Depends(get_current_user),
):
    col = await _get_collection(db, collection_id, current_user.id)

    if not col.is_indexed:
        raise HTTPException(503, "Collection is still indexing — try again shortly")

    idx = await registry.get_or_load(collection_id, col.model)
    if not idx.is_ready:
        raise HTTPException(503, "Index not ready")

    model_to_use = req.model or col.model
    query_vec    = embed_one(req.query, model=model_to_use)
    raw_results  = await idx.search(query_vec, k=req.k, probe=req.probe)

    if not raw_results:
        return QueryResponse(query=req.query, results=[], model_used=model_to_use)

    vector_ids = [r[0] for r in raw_results]
    scores_map = {r[0]: r[1] for r in raw_results}

    rows = (await db.execute(
        select(Chunk, Document).join(Document).where(
            Document.collection_id == collection_id,
            Chunk.vector_id.in_(vector_ids),
        )
    )).all()

    results = sorted([
        QueryResult(
            chunk_id    = chunk.id,
            vector_id   = chunk.vector_id,
            score       = scores_map.get(chunk.vector_id, 0.0),
            text        = chunk.text,
            document_id = doc.id,
            filename    = doc.filename,
            chunk_index = chunk.chunk_index,
        )
        for chunk, doc in rows
    ], key=lambda r: -r.score)

    return QueryResponse(query=req.query, results=results, model_used=model_to_use)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

async def _get_collection(db: AsyncSession, collection_id: int,
                          owner_id: int) -> Collection:
    result = await db.execute(
        select(Collection).where(
            Collection.id       == collection_id,
            Collection.owner_id == owner_id,
        )
    )
    col = result.scalar_one_or_none()
    if not col:
        raise HTTPException(404, "Collection not found")
    return col


async def _index_collection(collection_id: int, model: str):
    """Background task: build SHSRS index from all chunks in collection."""
    from shsrs_rag.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        chunks = (await db.execute(
            select(Chunk).join(Document).where(
                Document.collection_id == collection_id
            ).order_by(Chunk.vector_id)
        )).scalars().all()

    if not chunks:
        return

    embeddings = embed([c.text for c in chunks], model=model)
    vector_ids = [c.vector_id for c in chunks]

    idx = CollectionIndex(collection_id, model)
    await idx.build(embeddings, vector_ids)
    registry.register(idx)

    from shsrs_rag.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        col = (await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )).scalar_one_or_none()
        if col:
            col.is_indexed = True
            await db.commit()

    logger.info(f"Index complete for collection {collection_id} "
                f"({len(chunks)} chunks)")


async def _rebuild_collection(collection_id: int, model: str):
    """
    Full rebuild after document deletion.
    Re-embeds all remaining chunks and reassigns contiguous vector IDs.
    """
    from shsrs_rag.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        chunks = (await db.execute(
            select(Chunk).join(Document).where(
                Document.collection_id == collection_id
            ).order_by(Chunk.id)
        )).scalars().all()

        if not chunks:
            # No documents left — wipe index
            index_dir = settings.index_base_dir / f"col_{collection_id}"
            if index_dir.exists():
                shutil.rmtree(index_dir)
            registry._indexes.pop(collection_id, None)
            col = (await db.execute(
                select(Collection).where(Collection.id == collection_id)
            )).scalar_one_or_none()
            if col:
                col.n_vectors  = 0
                col.is_indexed = False
                await db.commit()
            logger.info(f"Collection {collection_id} is now empty — index cleared")
            return

        # Reassign contiguous vector IDs
        for new_vid, chunk in enumerate(chunks):
            chunk.vector_id = new_vid

        col = (await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )).scalar_one_or_none()
        if col:
            col.n_vectors = len(chunks)
        await db.commit()

    # Rebuild index
    embeddings = embed([c.text for c in chunks], model=model)
    vector_ids = list(range(len(chunks)))

    idx = CollectionIndex(collection_id, model)
    await idx.build(embeddings, vector_ids)
    registry.register(idx)

    async with AsyncSessionLocal() as db:
        col = (await db.execute(
            select(Collection).where(Collection.id == collection_id)
        )).scalar_one_or_none()
        if col:
            col.is_indexed = True
            await db.commit()

    logger.info(f"Rebuild complete for collection {collection_id} "
                f"({len(chunks)} chunks remaining)")
