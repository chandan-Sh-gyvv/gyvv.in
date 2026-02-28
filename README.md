# GYVV.in — Semantic Document Search API

> Search your documents like you search your memory.

High-recall semantic search API built on SHSRS — 99.1% Recall@10 at 1.19ms latency, 2,506 QPS batch throughput on 1M vectors. Upload documents, ask questions in plain English, get ranked answers back.

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## What It Does

GYVV lets you build semantic search into any application in minutes:

1. **Upload** `.txt` or `.md` documents into a collection
2. **Ask** a natural language question
3. **Get back** the most relevant passages, ranked by similarity

Under the hood: documents are chunked, embedded into 384D vectors, and indexed using SHSRS — a hierarchical vector search engine that maintains 99.1% recall regardless of corpus size.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set SECRET_KEY at minimum

# Run
uvicorn shsrs_rag.app:app --host 0.0.0.0 --port 8000
```

- REST API docs: `http://localhost:8000/docs`
- gRPC server starts automatically on port `50051`

---

## API Reference

### Auth
```
POST /auth/register           — create account
POST /auth/token              — get JWT token
```

### Collections
```
POST   /collections                       — create collection
GET    /collections                       — list your collections
GET    /collections/{id}/stats            — document + vector counts
DELETE /collections/{id}                  — delete collection + index files
```

### Documents
```
POST   /collections/{id}/ingest                    — upload .txt or .md
GET    /collections/{id}/documents                 — list documents
DELETE /collections/{id}/documents/{doc_id}        — remove document + rebuild index
```

### Search
```
POST /collections/{id}/query              — semantic search
```

---

## Example Usage

```python
import httpx

client = httpx.Client(base_url="https://api.gyvv.in")

# Register + login
client.post("/auth/register", json={
    "username": "alice",
    "email": "alice@example.com",
    "password": "securepassword"
})
resp  = client.post("/auth/token", data={"username": "alice", "password": "securepassword"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Create a collection
col    = client.post("/collections", json={"name": "Research Papers", "model": "local"}, headers=headers)
col_id = col.json()["id"]

# Ingest a document
with open("paper.txt", "rb") as f:
    client.post(f"/collections/{col_id}/ingest", files={"file": f}, headers=headers)

# Query
resp = client.post(f"/collections/{col_id}/query",
    json={"query": "What methodology did they use?", "k": 5},
    headers=headers)

for result in resp.json()["results"]:
    print(f"[{result['score']:.3f}] {result['text'][:120]}")
```

---

## Embedding Models

| Model | Dimension | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| `local` — all-MiniLM-L6-v2 | 384D | Fast | Good | Free |
| `openai` — text-embedding-3-small | 1536D | Medium | Better | ~$0.02/1M tokens |

Selected per collection at creation time. Set `OPENAI_API_KEY` in `.env` to enable OpenAI.

---

## gRPC Usage

For high-throughput applications, use the gRPC interface directly:

```python
import grpc
from shsrs_rag.proto_gen import shsrs_rag_pb2, shsrs_rag_pb2_grpc

channel = grpc.insecure_channel("api.gyvv.in:50051")
stub    = shsrs_rag_pb2_grpc.SHSRSRagStub(channel)

resp = stub.Query(shsrs_rag_pb2.QueryRequest(
    collection_id = 1,
    query         = "what is the main finding?",
    k             = 5,
    token         = "your-jwt-token",
))
for r in resp.results:
    print(f"[{r.score:.3f}] {r.text[:80]}")
```

Generate stubs once before using gRPC:
```bash
python generate_proto.py
```

---

## Architecture

```
Client (REST or gRPC)
        ↓
   FastAPI + JWT Auth
        ↓
┌─────────────────────────────┐
│       Ingest Pipeline       │
│  .txt/.md → chunks → embed  │
│  → SHSRS index (on disk)    │
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│       Query Pipeline        │
│  query → embed → search     │
│  → score-ranked chunks      │
└─────────────────────────────┘
        ↓
   SHSRS Engine
   99.1% Recall@10 · 1.19ms · 2,506 QPS
```

---

## Performance

Benchmarked on SIFT1M (1 million 128D vectors):

| Mode | Recall@10 | Latency | QPS |
|------|-----------|---------|-----|
| Single query | 99.1% | 1.19ms | 838 |
| Single query | 99.9% | 2.18ms | 460 |
| Batch | 99.1% | — | 2,506 |
| Batch (fast) | 93.5% | — | 4,476 |

Latency scales with probe count, not corpus size — adding 10× more documents does not slow down queries.

---

## Deployment

### Render (recommended)

```bash
# 1. Copy SHSRS engine into the repo
xcopy ..\shsrs shsrs /E /I        # Windows
cp -r ../shsrs ./shsrs            # Mac/Linux

# 2. Push to GitHub
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/gyvv-api.git
git push -u origin main

# 3. Connect repo on render.com → New Web Service
# render.yaml is already configured — Render picks it up automatically
```

### Docker

```bash
docker build -t gyvv-api .
docker run -p 8000:8000 \
  -e SECRET_KEY=your-secret \
  -e INDEX_BASE_DIR=/tmp/indexes \
  gyvv-api
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SECRET_KEY` | — | **Required.** JWT signing key |
| `DEFAULT_MODEL` | `local` | `local` or `openai` |
| `LOCAL_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `OPENAI_API_KEY` | — | Required for OpenAI embeddings |
| `INDEX_BASE_DIR` | `indexes` | Where SHSRS index files are stored |
| `DATABASE_URL` | SQLite | SQLAlchemy async DB URL |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `DEBUG` | `false` | Enable debug logging |

---

## Running Tests

```bash
# Fresh start
del shsrs_rag.db          # Windows
rm shsrs_rag.db           # Mac/Linux

# Start server
uvicorn shsrs_rag.app:app --port 8000

# Run all 15 tests
pytest tests/test_api.py -v
```

---

## License

MIT
