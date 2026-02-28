"""
tests/test_api.py
=================
End-to-end API tests.
Run with: pytest tests/test_api.py -v

Requires server running:
    uvicorn shsrs_rag.app:app --port 8000
"""

import pytest
import httpx
import time
import uuid

BASE_URL = "http://localhost:8000"

SAMPLE_TXT = b"""
SHSRS is a semantic search engine that uses hierarchical clustering
and HNSW graphs to achieve high recall at low latency.

The engine partitions vectors into clusters using KMeans, then refines
cluster boundaries using IGAR - Iterative Graph-Aligned Reassignment.

At 1M vectors, SHSRS achieves 99.1% Recall@10 at 1.19ms latency.
This makes it suitable for RAG applications where missing relevant
context leads to hallucination in LLM responses.

Vector search is the foundation of modern RAG pipelines. The key
metrics are recall, latency, and memory usage. SHSRS optimises
all three simultaneously. Approximate nearest neighbour search
is the core operation that powers modern AI retrieval systems.
High recall means fewer missed documents and better LLM answers.
"""

SAMPLE_TXT_2 = b"""
Transformer models revolutionised natural language processing.
BERT introduced bidirectional attention for better context understanding.
GPT models use autoregressive generation for text completion tasks.

Fine-tuning pretrained models on domain-specific data improves performance.
Retrieval augmented generation combines search with language model generation.
The retriever fetches relevant context and the generator produces answers.

Embedding models convert text into dense vector representations.
Cosine similarity measures the angle between two embedding vectors.
Nearest neighbour search finds the most similar vectors efficiently.
"""

SAMPLE_MD = b"""
# SHSRS Architecture

## Overview

SHSRS combines macro-cluster routing with per-cluster HNSW search.

## Key Components

IGAR stands for Iterative Graph-Aligned Reassignment for cluster refinement.
Gap-adaptive probe uses centroid score gap to set probe count dynamically.
Batched search groups queries by cluster for 2.5x throughput gain.

## Performance

Recall at ten is 99.1 percent with 1.19ms latency and 2506 QPS batch throughput.
Index RAM usage is 610MB for one million vectors.
The engine scales with probe count not dataset size.
"""


def unique_user():
    uid = uuid.uuid4().hex[:8]
    return {"username": f"user_{uid}", "email": f"{uid}@test.com", "password": "testpass123"}


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=120)


@pytest.fixture
def auth_headers(client):
    u = unique_user()
    client.post("/auth/register", json=u)
    resp  = client.post("/auth/token", data={"username": u["username"], "password": u["password"]})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def wait_for_index(client, col_id, headers, timeout=60):
    """Poll until collection is_indexed=True or timeout."""
    for _ in range(timeout):
        stats = client.get(f"/collections/{col_id}/stats", headers=headers)
        if stats.json().get("is_indexed"):
            return True
        time.sleep(1)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# BASIC
# ══════════════════════════════════════════════════════════════════════════════

def test_health(client):
    for _ in range(30):
        try:
            resp = client.get("/health")
            if resp.status_code == 200:
                assert resp.json()["status"] == "ok"
                return
        except Exception:
            pass
        time.sleep(1)
    pytest.fail("Server did not become healthy within 30 seconds")


def test_register_and_login(client):
    u = unique_user()
    resp = client.post("/auth/register", json=u)
    assert resp.status_code == 201

    resp = client.post("/auth/token", data={"username": u["username"], "password": u["password"]})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_duplicate_username_rejected(client):
    u = unique_user()
    client.post("/auth/register", json=u)
    resp = client.post("/auth/register", json=u)
    assert resp.status_code == 400


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def test_create_collection(client, auth_headers):
    resp = client.post("/collections", json={"name": "Test", "model": "local"},
                       headers=auth_headers)
    assert resp.status_code == 201
    assert resp.json()["name"] == "Test"
    assert resp.json()["is_indexed"] == False


def test_list_collections(client, auth_headers):
    client.post("/collections", json={"name": "Col A", "model": "local"}, headers=auth_headers)
    client.post("/collections", json={"name": "Col B", "model": "local"}, headers=auth_headers)
    resp = client.get("/collections", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 2


def test_delete_collection_cleans_up(client, auth_headers):
    # Create + ingest so index files exist on disk
    col = client.post("/collections", json={"name": "To Delete", "model": "local"},
                      headers=auth_headers).json()
    col_id = col["id"]

    client.post(f"/collections/{col_id}/ingest",
                files={"file": ("doc.txt", SAMPLE_TXT, "text/plain")},
                headers=auth_headers)
    wait_for_index(client, col_id, auth_headers)

    # Delete collection
    resp = client.delete(f"/collections/{col_id}", headers=auth_headers)
    assert resp.status_code == 204

    # Should 404 now
    resp = client.get(f"/collections/{col_id}/stats", headers=auth_headers)
    assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# INGEST
# ══════════════════════════════════════════════════════════════════════════════

def test_ingest_txt(client, auth_headers):
    col_id = client.post("/collections", json={"name": "TXT", "model": "local"},
                         headers=auth_headers).json()["id"]
    resp = client.post(f"/collections/{col_id}/ingest",
                       files={"file": ("test.txt", SAMPLE_TXT, "text/plain")},
                       headers=auth_headers)
    assert resp.status_code == 201
    assert resp.json()["n_chunks"] > 0
    print(f"\n  Ingested {resp.json()['n_chunks']} chunks")


def test_ingest_markdown(client, auth_headers):
    col_id = client.post("/collections", json={"name": "MD", "model": "local"},
                         headers=auth_headers).json()["id"]
    resp = client.post(f"/collections/{col_id}/ingest",
                       files={"file": ("test.md", SAMPLE_MD, "text/markdown")},
                       headers=auth_headers)
    assert resp.status_code == 201
    assert resp.json()["n_chunks"] > 0


def test_duplicate_document_rejected(client, auth_headers):
    col_id = client.post("/collections", json={"name": "Dedup", "model": "local"},
                         headers=auth_headers).json()["id"]
    client.post(f"/collections/{col_id}/ingest",
                files={"file": ("doc.txt", SAMPLE_TXT, "text/plain")},
                headers=auth_headers)
    time.sleep(2)
    resp = client.post(f"/collections/{col_id}/ingest",
                       files={"file": ("doc.txt", SAMPLE_TXT, "text/plain")},
                       headers=auth_headers)
    assert resp.status_code == 409


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════

def test_list_documents(client, auth_headers):
    col_id = client.post("/collections", json={"name": "DocList", "model": "local"},
                         headers=auth_headers).json()["id"]
    client.post(f"/collections/{col_id}/ingest",
                files={"file": ("a.txt", SAMPLE_TXT, "text/plain")},
                headers=auth_headers)
    client.post(f"/collections/{col_id}/ingest",
                files={"file": ("b.txt", SAMPLE_TXT_2, "text/plain")},
                headers=auth_headers)

    resp = client.get(f"/collections/{col_id}/documents", headers=auth_headers)
    assert resp.status_code == 200
    docs = resp.json()
    assert len(docs) == 2
    filenames = [d["filename"] for d in docs]
    assert "a.txt" in filenames
    assert "b.txt" in filenames
    print(f"\n  Listed {len(docs)} documents")


def test_delete_document_rebuilds_index(client, auth_headers):
    col_id = client.post("/collections", json={"name": "DelDoc", "model": "local"},
                         headers=auth_headers).json()["id"]

    # Ingest two documents
    r1 = client.post(f"/collections/{col_id}/ingest",
                     files={"file": ("doc1.txt", SAMPLE_TXT, "text/plain")},
                     headers=auth_headers)
    r2 = client.post(f"/collections/{col_id}/ingest",
                     files={"file": ("doc2.txt", SAMPLE_TXT_2, "text/plain")},
                     headers=auth_headers)
    doc1_id = r1.json()["document_id"]
    doc2_id = r2.json()["document_id"]

    wait_for_index(client, col_id, auth_headers)

    # Delete doc1
    resp = client.delete(f"/collections/{col_id}/documents/{doc1_id}",
                         headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["chunks_removed"] > 0

    # Wait for rebuild
    assert wait_for_index(client, col_id, auth_headers), "Rebuild did not complete"

    # Only doc2 should remain
    docs = client.get(f"/collections/{col_id}/documents", headers=auth_headers).json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "doc2.txt"
    print(f"\n  After delete: {len(docs)} document remaining")


def test_delete_last_document_empties_index(client, auth_headers):
    col_id = client.post("/collections", json={"name": "LastDoc", "model": "local"},
                         headers=auth_headers).json()["id"]
    r = client.post(f"/collections/{col_id}/ingest",
                    files={"file": ("only.txt", SAMPLE_TXT, "text/plain")},
                    headers=auth_headers)
    doc_id = r.json()["document_id"]
    wait_for_index(client, col_id, auth_headers)

    resp = client.delete(f"/collections/{col_id}/documents/{doc_id}",
                         headers=auth_headers)
    assert resp.status_code == 200

    time.sleep(2)
    stats = client.get(f"/collections/{col_id}/stats", headers=auth_headers).json()
    assert stats["n_documents"] == 0
    assert stats["n_vectors"]   == 0


# ══════════════════════════════════════════════════════════════════════════════
# QUERY
# ══════════════════════════════════════════════════════════════════════════════

def test_query(client, auth_headers):
    col_id = client.post("/collections", json={"name": "Query", "model": "local"},
                         headers=auth_headers).json()["id"]
    client.post(f"/collections/{col_id}/ingest",
                files={"file": ("shsrs.txt", SAMPLE_TXT, "text/plain")},
                headers=auth_headers)

    assert wait_for_index(client, col_id, auth_headers), "Index timeout"

    resp = client.post(f"/collections/{col_id}/query",
                       json={"query": "What is the recall of SHSRS?", "k": 3},
                       headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) > 0
    print(f"\n  Query: {data['query']}")
    for r in data["results"]:
        print(f"  [{r['score']:.3f}] {r['text'][:80]}...")


def test_query_unindexed_returns_503(client, auth_headers):
    col_id = client.post("/collections", json={"name": "Unindexed", "model": "local"},
                         headers=auth_headers).json()["id"]
    resp = client.post(f"/collections/{col_id}/query",
                       json={"query": "test", "k": 3},
                       headers=auth_headers)
    assert resp.status_code == 503


def test_stats(client, auth_headers):
    col_id = client.post("/collections", json={"name": "Stats", "model": "local"},
                         headers=auth_headers).json()["id"]
    resp = client.get(f"/collections/{col_id}/stats", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["collection_id"] == col_id
    assert resp.json()["n_documents"]   == 0


if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
