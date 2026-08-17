# astra — RAG Microservice for PASTI PINTAR

RAG (Retrieval-Augmented Generation) microservice that embeds video transcript segments and lesson PDFs into pgvector and answers questions via a LangGraph retrieval graph with SSE streaming.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and fill in the values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (must have pgvector extension) |
| `REDIS_HOST` | Redis host (for Arq worker) |
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `OPENROUTER_API_KEY` | OpenRouter API key for chat generation |
| `INTERNAL_API_KEY` | Shared secret for brain → astra HTTP calls |
| `OPENROUTER_MODEL` | Model for answer generation |
| `EMBEDDING_MODEL` | OpenAI embedding model |

### Database Migrations

```bash
alembic upgrade head
```

This creates the three astra-owned tables (`rag_ingestion_log`, `rag_pdf_chunks`, `rag_pdf_ingestion_log`) and enables the pgvector extension.

## Running

### API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Background Worker (Arq)

```bash
arq app.ingestion.worker.WorkerSettings
```

The worker runs two cron jobs:
- `embed_pending_segments` — every 60 seconds
- `embed_pending_pdfs` — every 5 minutes

## API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | No | Health check → `{"status":"ok"}` |
| `/rag/query` | POST | `INTERNAL_API_KEY` header | RAG query → SSE stream |

### Query Request

```json
{
  "lesson_video_id": "uuid",
  "module_id": "uuid or null",
  "question": "string (max 1000 chars)",
  "session_history": [],
  "user_id": "string"
}
```

### SSE Events

| Event | Description |
|---|---|
| `started` | Query started, contains `request_id` |
| `chunk` | Token from generated answer |
| `completed` | Full answer with citations |
| `error` | Error message (user-friendly) |
| `heartbeat` | Sent every 30s if no token |

## Architecture Notes

- This service will later be converted to a git submodule.
- astra is read-only on brain-owned tables (`transcripts`, `transcript_segments`, etc.) except for writing embeddings to `transcript_segments.embedding`.
- All HTTP communication with the Go backend ("brain") is guarded by `INTERNAL_API_KEY`.
