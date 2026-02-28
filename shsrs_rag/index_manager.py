"""
shsrs_rag/index_manager.py
==========================
Manages one SHSRS index per collection.
Handles build, load, search, and incremental adds.
"""

from __future__ import annotations
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from shsrs_rag.config import settings

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)


class CollectionIndex:
    def __init__(self, collection_id: int, model: str):
        self.collection_id = collection_id
        self.model         = model
        self.index_dir     = settings.index_base_dir / f"col_{collection_id}"
        self._engine       = None
        self._buffer_vecs  : list[np.ndarray] = []
        self._buffer_ids   : list[int]         = []
        self._lock         = asyncio.Lock()

    async def build(self, vectors: np.ndarray, vector_ids: list[int]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._build_sync, vectors, vector_ids)

    def _build_sync(self, vectors: np.ndarray, vector_ids: list[int]):
        from shsrs import SHSRSEngine

        self.index_dir.mkdir(parents=True, exist_ok=True)
        N = len(vectors)

        # n_clusters must be <= n_samples — scale safely
        n_clusters = max(2, min(int(N ** 0.5), N // 2, settings.default_n_clusters))
        logger.info(f"Building SHSRS index: {N} vectors, {n_clusters} clusters")

        self._engine = SHSRSEngine.build(
            vectors         = vectors,
            index_dir       = str(self.index_dir),
            n_clusters      = n_clusters,
            M               = settings.default_m,
            ef_construction = settings.default_ef_construction,
        )
        logger.info(f"Index built for collection {self.collection_id}")

    async def load(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, self._load_sync)

    def _load_sync(self):
        from shsrs import SHSRSEngine
        if self.index_dir.exists():
            self._engine = SHSRSEngine.load(str(self.index_dir))
            logger.info(f"Loaded index for collection {self.collection_id}")

    async def search(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        probe: int = None,
    ) -> list[tuple[int, float]]:
        probe = probe or settings.default_probe
        loop  = asyncio.get_event_loop()
        results = []

        if self._engine is not None:
            main_results = await loop.run_in_executor(
                _executor,
                lambda: self._engine.search(query_vec, k=k, probe=probe)
            )
            results.extend(main_results)

        # Buffer brute force search
        if self._buffer_vecs:
            buf_vecs = np.stack(self._buffer_vecs)
            scores   = buf_vecs @ query_vec
            top_idx  = np.argsort(-scores)[:k]
            for i in top_idx:
                results.append((self._buffer_ids[i], float(scores[i])))

        results.sort(key=lambda x: -x[1])
        return results[:k]

    async def add_vectors(self, vectors: np.ndarray, vector_ids: list[int]):
        async with self._lock:
            for vec, vid in zip(vectors, vector_ids):
                self._buffer_vecs.append(vec)
                self._buffer_ids.append(vid)

        if len(self._buffer_vecs) >= 5000:
            asyncio.create_task(self._merge_buffer())

    async def _merge_buffer(self):
        async with self._lock:
            if not self._buffer_vecs:
                return
            buf_vecs = np.stack(self._buffer_vecs)
            self._buffer_vecs.clear()
            self._buffer_ids.clear()

        if self._engine is not None:
            existing = self._engine._data_norm
            all_vecs = np.concatenate([existing, buf_vecs])
        else:
            all_vecs = buf_vecs

        await self.build(all_vecs, list(range(len(all_vecs))))

    @property
    def is_ready(self) -> bool:
        return self._engine is not None or len(self._buffer_vecs) > 0

    @property
    def n_vectors(self) -> int:
        base = self._engine._n_vectors if self._engine else 0
        return base + len(self._buffer_vecs)


class IndexRegistry:
    def __init__(self):
        self._indexes: dict[int, CollectionIndex] = {}

    def get(self, collection_id: int) -> Optional[CollectionIndex]:
        return self._indexes.get(collection_id)

    def register(self, idx: CollectionIndex):
        self._indexes[idx.collection_id] = idx

    async def get_or_load(self, collection_id: int, model: str) -> CollectionIndex:
        if collection_id not in self._indexes:
            idx = CollectionIndex(collection_id, model)
            await idx.load()
            self._indexes[collection_id] = idx
        return self._indexes[collection_id]


registry = IndexRegistry()
