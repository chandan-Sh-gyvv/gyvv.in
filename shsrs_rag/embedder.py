"""
shsrs_rag/embedder.py
=====================
Embedding service — supports local (sentence-transformers) and OpenAI models.
Model is selected per collection at ingest time and reused at query time.
"""

from __future__ import annotations
import numpy as np
from typing import Literal
from functools import lru_cache

from shsrs_rag.config import settings

EmbedModel = Literal["local", "openai"]


# ── Local model (sentence-transformers) ───────────────────────────────────────
@lru_cache(maxsize=2)
def _get_local_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_local(texts: list[str], model_name: str = None) -> np.ndarray:
    """Embed texts using a local sentence-transformers model."""
    model_name = model_name or settings.local_model_name
    model      = _get_local_model(model_name)
    embeddings = model.encode(texts, batch_size=64,
                               normalize_embeddings=True,
                               show_progress_bar=False)
    return embeddings.astype(np.float32)


# ── OpenAI model ──────────────────────────────────────────────────────────────
def embed_openai(texts: list[str], model_name: str = None) -> np.ndarray:
    """Embed texts using OpenAI embeddings API."""
    from openai import OpenAI
    import numpy as np

    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    model_name = model_name or settings.openai_model
    client     = OpenAI(api_key=settings.openai_api_key)

    # OpenAI API accepts max 2048 inputs per call — batch if needed
    all_embeddings = []
    batch_size     = 512

    for i in range(0, len(texts), batch_size):
        batch    = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2 normalize for cosine search
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1.0, norms)

    return embeddings


# ── Unified interface ─────────────────────────────────────────────────────────
def embed(texts: list[str],
          model: EmbedModel = None,
          model_name: str = None) -> np.ndarray:
    """
    Embed a list of texts.

    Parameters
    ----------
    texts      : list of strings to embed
    model      : 'local' or 'openai'
    model_name : override default model name

    Returns
    -------
    float32 array [N, D], L2-normalised
    """
    model = model or settings.default_model

    if model == "local":
        return embed_local(texts, model_name)
    elif model == "openai":
        return embed_openai(texts, model_name)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'local' or 'openai'")


def embed_one(text: str, model: EmbedModel = None, model_name: str = None) -> np.ndarray:
    """Embed a single string. Returns 1D array [D]."""
    return embed([text], model=model, model_name=model_name)[0]


def get_model_dim(model: EmbedModel = None, model_name: str = None) -> int:
    """Return embedding dimension for a given model."""
    model = model or settings.default_model
    if model == "local":
        name  = model_name or settings.local_model_name
        m     = _get_local_model(name)
        return m.get_sentence_embedding_dimension()
    elif model == "openai":
        dims  = {"text-embedding-3-small": 1536,
                 "text-embedding-3-large": 3072,
                 "text-embedding-ada-002": 1536}
        name  = model_name or settings.openai_model
        return dims.get(name, 1536)
    raise ValueError(f"Unknown model: {model}")
