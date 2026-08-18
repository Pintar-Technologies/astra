from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone

import httpx
import fitz  # PyMuPDF
from openai import RateLimitError
from sqlalchemy import text

from app.config import settings
from app.deps import get_brain_engine, get_engine, get_openai_client

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BATCH_SIZE = 20
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB
_CHUNK_SIZE_CHARS = 1200  # ~300 tokens
_CHUNK_OVERLAP_CHARS = 240  # ~20% overlap
_MIN_TEXT_CHARS = 50

# Idempotent guard so the cron can't crash-loop if the alembic migration
# hasn't created the rag_* tables yet. Mirrors 0001_initial_rag_tables.py.
_ENSURE_RAG_TABLES_SQL = text(
    """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS rag_pdf_ingestion_log (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        lesson_id UUID NOT NULL UNIQUE,
        pdf_hash VARCHAR(64) NOT NULL DEFAULT '',
        embedded_at TIMESTAMPTZ,
        status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
        retry_count INTEGER NOT NULL DEFAULT 0,
        error_message TEXT
    );

    CREATE TABLE IF NOT EXISTS rag_pdf_chunks (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        lesson_id UUID NOT NULL,
        module_id UUID,
        chunk_index INTEGER NOT NULL,
        page_start INTEGER,
        page_end INTEGER,
        text TEXT NOT NULL,
        embedding vector(1536),
        embedding_model VARCHAR(50) NOT NULL DEFAULT '',
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        UNIQUE (lesson_id, chunk_index)
    );
    """
)


def _ensure_rag_tables(engine) -> None:
    with engine.begin() as conn:
        conn.execute(_ENSURE_RAG_TABLES_SQL)


async def embed_pending_pdfs(ctx: dict | None = None) -> int:
    """Arq cron job: download and embed PDFs from lessons without embeddings.

    Returns the number of successfully ingested PDFs.
    """
    engine = get_engine()
    brain_engine = get_brain_engine()
    client = get_openai_client()
    model = settings.EMBEDDING_MODEL
    now = datetime.now(timezone.utc)

    # Ensure the pgvector rag_* tables exist (idempotent guard against a
    # missing alembic migration causing a crash-loop).
    _ensure_rag_tables(engine)

    # Fetch ingestion-log state from the pgvector DB (engine)
    with engine.connect() as conn:
        log_rows = conn.execute(
            text("SELECT lesson_id, status, retry_count FROM rag_pdf_ingestion_log")
        ).mappings().fetchall()
    done_ids: set[str] = set()
    exhausted_ids: set[str] = set()
    for log_row in log_rows:
        lid = str(log_row["lesson_id"])
        if log_row["status"] == "DONE":
            done_ids.add(lid)
        elif log_row["status"] == "FAILED" and log_row["retry_count"] >= _MAX_RETRIES:
            exhausted_ids.add(lid)

    # Fetch candidates from the brain DB (brain_engine) — no cross-db subqueries
    candidates_sql = text(
        """
        SELECT l.id, l.pdf_url, l.title, sm.module_id
        FROM lessons l
        JOIN sub_modules sm ON l.sub_module_id = sm.id
        WHERE l.pdf_url IS NOT NULL
          AND sm.module_id IS NOT NULL
        """
    )
    with brain_engine.connect() as conn:
        all_candidates = conn.execute(candidates_sql).mappings().fetchall()

    # Filter out already-done / retry-exhausted lessons in Python
    rows = []
    for row in all_candidates:
        lid = str(row["id"])
        if lid in done_ids or lid in exhausted_ids:
            continue
        rows.append(row)
        if len(rows) >= _BATCH_SIZE:
            break

    if not rows:
        logger.info("No pending PDFs to embed")
        return 0

    success_count = 0
    for row in rows:
        lesson_id = str(row["id"])
        pdf_url = row["pdf_url"]
        title = row["title"]
        module_id = str(row["module_id"]) if row["module_id"] else None

        logger.info("Processing PDF for lesson %s: %s", lesson_id, title)

        try:
            ok = await _process_single_pdf(engine, client, lesson_id, pdf_url, title, module_id, model, now)
            if ok:
                success_count += 1
        except Exception as exc:
            logger.exception("Failed to process PDF for lesson %s: %s", lesson_id, exc)
            _mark_pdf_failed(engine, lesson_id, str(exc)[:500])

    logger.info("Successfully ingested %d/%d PDFs", success_count, len(rows))
    return success_count


# ── Single PDF processing ──────────────────────────────────────────────────


async def _process_single_pdf(
    engine,
    client,
    lesson_id: str,
    pdf_url: str,
    title: str,
    module_id: str | None,
    model: str,
    now: datetime,
) -> bool:
    """Download, extract, chunk, embed, and store a single PDF."""
    # Download PDF
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            response = await http.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            data = response.content
    except Exception as exc:
        raise Exception(f"download_failed: {exc}") from exc

    if len(data) > _MAX_PDF_BYTES:
        raise Exception("pdf_too_large")

    # SHA-256 hash for dedup
    pdf_hash = hashlib.sha256(data).hexdigest()

    # Check existing ingestion for dedup
    with engine.connect() as conn:
        existing = conn.execute(
            text(
                "SELECT pdf_hash, status FROM rag_pdf_ingestion_log WHERE lesson_id = :lid AND status = 'DONE'"
            ),
            {"lid": lesson_id},
        ).mappings().first()

    if existing is not None:
        if existing["pdf_hash"] == pdf_hash:
            logger.info("PDF for lesson %s unchanged, skipping", lesson_id)
            return False  # Already ingested with same hash
        # Hash changed — delete old chunks and re-ingest
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM rag_pdf_chunks WHERE lesson_id = :lid"), {"lid": lesson_id})
            conn.execute(
                text("DELETE FROM rag_pdf_ingestion_log WHERE lesson_id = :lid"),
                {"lid": lesson_id},
            )

    # Extract text via PyMuPDF
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise Exception(f"corrupt_pdf: {exc}") from exc

    if doc.needs_pass:
        doc.close()
        raise Exception("need_password")

    # Extract text with page numbers
    pages_text: list[tuple[int, str]] = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        # Sort blocks by vertical position then horizontal
        blocks.sort(key=lambda b: (b[1], b[0]))
        page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
        if page_text:
            pages_text.append((page_num + 1, page_text))

    doc.close()

    total_text = "\n".join(t for _, t in pages_text)
    if len(total_text) < _MIN_TEXT_CHARS:
        raise Exception("no_text_extracted: scanned PDF? OCR not supported")

    # Chunk: ~300 tokens / ~1200 chars with 20% overlap
    chunks: list[dict] = []
    for page_num, page_text in pages_text:
        # Split page text into chunks
        start = 0
        while start < len(page_text):
            end = min(start + _CHUNK_SIZE_CHARS, len(page_text))
            chunk_text = page_text[start:end]

            chunks.append({
                "text": chunk_text,
                "page_start": page_num,
                "page_end": page_num,
            })

            next_start = end - _CHUNK_OVERLAP_CHARS
            if next_start <= start:
                next_start = end
            start = next_start
            if start >= len(page_text):
                break

    if not chunks:
        logger.warning("No chunks extracted for lesson %s", lesson_id)
        return False

    # Assign chunk_index and merge adjacent same-page chunks
    merged = []
    for i, c in enumerate(chunks):
        if merged and merged[-1]["page_start"] == c["page_start"] and merged[-1]["page_end"] == c["page_end"]:
            if len(merged[-1]["text"]) + len(c["text"]) < _CHUNK_SIZE_CHARS * 1.5:
                merged[-1]["text"] += " " + c["text"]
                continue
        c["chunk_index"] = len(merged)
        merged.append(c)

    # Batch embed all chunks
    texts_to_embed = [c["text"] for c in merged]

    try:
        resp = client.embeddings.create(input=texts_to_embed, model=model)
        embeddings = [d.embedding for d in resp.data]
    except RateLimitError:
        raise Exception("rate_limited")
    except Exception as exc:
        raise Exception(f"embedding_failed: {exc}") from exc

    # Insert chunks into rag_pdf_chunks
    with engine.begin() as conn:
        for chunk, emb in zip(merged, embeddings):
            emb_literal = "[" + ",".join(str(x) for x in emb) + "]"
            conn.execute(
                text(
                    """
                    INSERT INTO rag_pdf_chunks (lesson_id, module_id, chunk_index, page_start, page_end, text, embedding, embedding_model)
                    VALUES (:lesson_id, :module_id, :chunk_index, :page_start, :page_end, :text, :emb::vector, :model)
                    ON CONFLICT (lesson_id, chunk_index)
                    DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
                    """
                ),
                {
                    "lesson_id": lesson_id,
                    "module_id": module_id,
                    "chunk_index": chunk["chunk_index"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "text": chunk["text"],
                    "emb": emb_literal,
                    "model": model,
                },
            )

    # Upsert ingestion log
    _mark_pdf_done(engine, lesson_id, pdf_hash, now)

    logger.info("Ingested PDF lesson %s: %d chunks", lesson_id, len(merged))
    return True


# ── Internal helpers ───────────────────────────────────────────────────────


def _mark_pdf_done(engine, lesson_id: str, pdf_hash: str, now: datetime):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO rag_pdf_ingestion_log (lesson_id, pdf_hash, embedded_at, status)
                VALUES (:lid, :hash, :now, 'DONE')
                ON CONFLICT (lesson_id) DO UPDATE
                SET pdf_hash = EXCLUDED.pdf_hash,
                    embedded_at = EXCLUDED.embedded_at,
                    status = 'DONE',
                    retry_count = 0,
                    error_message = NULL
                """
            ),
            {"lid": lesson_id, "hash": pdf_hash, "now": now},
        )


def _mark_pdf_failed(engine, lesson_id: str, error_msg: str):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO rag_pdf_ingestion_log (lesson_id, pdf_hash, status, retry_count, error_message)
                VALUES (:lid, '', 'FAILED', 1, :error_msg)
                ON CONFLICT (lesson_id) DO UPDATE
                SET status = 'FAILED',
                    retry_count = rag_pdf_ingestion_log.retry_count + 1,
                    error_message = :error_msg
                """
            ),
            {"lid": lesson_id, "error_msg": error_msg},
        )
