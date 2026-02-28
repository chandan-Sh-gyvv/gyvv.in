"""
shsrs_rag/database.py
=====================
SQLAlchemy async database setup and models.
"""

from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from shsrs_rag.config import settings


# ── Engine + session ──────────────────────────────────────────────────────────
engine = create_async_engine(str(settings.database_url), echo=settings.debug)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── Models ────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True)
    username:     Mapped[str]      = mapped_column(String(64), unique=True, index=True)
    email:        Mapped[str]      = mapped_column(String(128), unique=True, index=True)
    hashed_password: Mapped[str]   = mapped_column(String(256))
    is_active:    Mapped[bool]     = mapped_column(Boolean, default=True)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    collections:  Mapped[list[Collection]] = relationship("Collection", back_populates="owner")


class Collection(Base):
    """A named group of documents with its own SHSRS index."""
    __tablename__ = "collections"

    id:           Mapped[int]      = mapped_column(Integer, primary_key=True)
    name:         Mapped[str]      = mapped_column(String(128), index=True)
    description:  Mapped[str]      = mapped_column(Text, default="")
    owner_id:     Mapped[int]      = mapped_column(ForeignKey("users.id"))
    model:        Mapped[str]      = mapped_column(String(64), default="local")
    index_dir:    Mapped[str]      = mapped_column(String(256), default="")
    n_vectors:    Mapped[int]      = mapped_column(Integer, default=0)
    is_indexed:   Mapped[bool]     = mapped_column(Boolean, default=False)
    created_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at:   Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow,
                                                    onupdate=datetime.utcnow)

    owner:        Mapped[User]           = relationship("User", back_populates="collections")
    documents:    Mapped[list[Document]] = relationship("Document", back_populates="collection")


class Document(Base):
    """A single ingested document and its chunks."""
    __tablename__ = "documents"

    id:            Mapped[int]      = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[int]      = mapped_column(ForeignKey("collections.id"))
    filename:      Mapped[str]      = mapped_column(String(256))
    content_hash:  Mapped[str]      = mapped_column(String(64), index=True)
    n_chunks:      Mapped[int]      = mapped_column(Integer, default=0)
    created_at:    Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    collection:    Mapped[Collection]  = relationship("Collection", back_populates="documents")
    chunks:        Mapped[list[Chunk]] = relationship("Chunk", back_populates="document")


class Chunk(Base):
    """A single text chunk with its vector index ID."""
    __tablename__ = "chunks"

    id:            Mapped[int]   = mapped_column(Integer, primary_key=True)
    document_id:   Mapped[int]   = mapped_column(ForeignKey("documents.id"))
    vector_id:     Mapped[int]   = mapped_column(Integer, index=True)  # SHSRS global_id
    text:          Mapped[str]   = mapped_column(Text)
    chunk_index:   Mapped[int]   = mapped_column(Integer)  # position in document
    token_count:   Mapped[int]   = mapped_column(Integer, default=0)

    document:      Mapped[Document] = relationship("Document", back_populates="chunks")
