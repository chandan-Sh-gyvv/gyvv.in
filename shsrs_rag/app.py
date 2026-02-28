"""
shsrs_rag/app.py
================
FastAPI application entry point.
Starts REST server + gRPC server concurrently.

Run:
    uvicorn shsrs_rag.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from shsrs_rag.config import settings
from shsrs_rag.database import init_db
from shsrs_rag.rest_api import router
from shsrs_rag.grpc_server import serve_grpc

logging.basicConfig(
    level   = logging.DEBUG if settings.debug else logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title       = settings.app_name,
    version     = settings.app_version,
    description = (
        "SHSRS RAG API — High-recall vector search for document retrieval. "
        "99.1% Recall@10 on SIFT1M at 2,506 QPS."
    ),
    docs_url  = "/docs",
    redoc_url = "/redoc",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(router)


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    settings.index_base_dir.mkdir(parents=True, exist_ok=True)
    await init_db()
    logger.info("Database initialised")

    # Pre-load embedding model so first request is instant
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: __import__('shsrs_rag.embedder', fromlist=['embed']).embed(
                ["warmup"], model="local"
            )
        )
        logger.info("Embedding model warmed up")
    except Exception as e:
        logger.warning(f"Embedding warmup failed (non-fatal): {e}")

    # Start gRPC server
    try:
        asyncio.create_task(serve_grpc())
        logger.info(f"gRPC server launching on port {settings.grpc_port}")
    except Exception as e:
        logger.warning(f"gRPC not started: {e}. Run: python generate_proto.py")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": settings.app_version, "engine": "SHSRS"}


@app.get("/")
async def root():
    return {
        "name":    settings.app_name,
        "version": settings.app_version,
        "docs":    "/docs",
        "health":  "/health",
    }
