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


async def embed_pending_pdfs(ctx: dict | None = None) -> int:
    """Arq cron job: download and embed PDFs from lessons without embeddings.

    Returns the number of successfully ingested PDFs.
    """
    engine = get_engine()
    brain_engine = get_brain_engine()
    client = get_openai_client()
    model = settings.EMBEDDING_MODEL
    now = datetime.now(timezone.utc)

    # Fetch lessons with PDFs that haven't been ingested
    select_sql = text(
        """
        SELECT l.id, l.pdf_url, l.title, sm.module_id
        FROM lessons l
        JOIN sub_modules sm ON l.sub_module_id = sm.id
        WHERE l.pdf_url IS NOT NULL
          AND sm.module_id IS NOT NULL
          AND l.id NOT IN (
              SELECT lesson_id
              FROM rag_pdf_ingestion_log
              WHERE status = 'DONE'
          )
          AND l.id NOT IN (
              SELECT lesson_id
              FROM rag_pdf_ingestion_log
              WHERE status = 'FAILED'
                AND retry_count >= :max_retries
          )
        LIMIT :limit
        """
    )

    with brain_engine.connect() as conn:
        rows = conn.execute(select_sql, {"max_retries": _MAX_RETRIES, "limit": _BATCH_SIZE}).mappings().fetchall()

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
