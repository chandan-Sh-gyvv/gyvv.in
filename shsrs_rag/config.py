"""
shsrs_rag/config.py
===================
Central configuration via environment variables.
Copy .env.example to .env and fill in values.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    # ── API ───────────────────────────────────────────────────────────────────
    app_name:        str  = "SHSRS RAG API"
    app_version:     str  = "0.1.0"
    debug:           bool = False
    host:            str  = "0.0.0.0"
    rest_port:       int  = 8000
    grpc_port:       int  = 50051

    # ── Auth ──────────────────────────────────────────────────────────────────
    secret_key:      str  = "change-me-in-production-use-openssl-rand-hex-32"
    algorithm:       str  = "HS256"
    token_expire_minutes: int = 60 * 24  # 24 hours

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_ingest:  str = "10/minute"
    rate_limit_query:   str = "60/minute"

    # ── Embedding models ──────────────────────────────────────────────────────
    default_model:   Literal["local", "openai"] = "local"
    local_model_name: str = "all-MiniLM-L6-v2"   # 384D, fast
    openai_api_key:  str  = ""
    openai_model:    str  = "text-embedding-3-small"  # 1536D

    # ── SHSRS index ───────────────────────────────────────────────────────────
    index_base_dir:  Path = Path("indexes")
    default_n_clusters: int  = 100
    default_m:          int  = 16
    default_ef_construction: int = 200
    default_probe:      int  = 16

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size:      int  = 512    # tokens
    chunk_overlap:   int  = 64     # tokens

    # ── Database ──────────────────────────────────────────────────────────────
    database_url:    str  = "sqlite+aiosqlite:///./shsrs_rag.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
