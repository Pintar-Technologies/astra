from __future__ import annotations

import logging
from datetime import datetime, timezone

from openai import RateLimitError
from sqlalchemy import text

from app.config import settings
from app.deps import get_engine, get_openai_client

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BATCH_SIZE = 100


async def embed_pending_segments(ctx: dict) -> int:
    """Arq cron job: embed transcript segments that have no embedding yet.

    Returns the number of segments successfully embedded.
    """
    engine = get_engine()
    client = get_openai_client()
    model = settings.EMBEDDING_MODEL
    now = datetime.now(timezone.utc)

    # Fetch pending segments
    select_sql = text(
        """
        SELECT ts.id, ts.text
        FROM transcript_segments ts
        JOIN transcripts t ON ts.transcript_id = t.id
        WHERE t.status = 'DONE'
          AND ts.embedding IS NULL
          AND ts.id NOT IN (
              SELECT transcript_segment_id
              FROM rag_ingestion_log
              WHERE source_type = 'segment'
                AND status = 'FAILED'
                AND retry_count >= :max_retries
          )
        LIMIT :limit
        """
    )

    with engine.connect() as conn:
        rows = conn.execute(select_sql, {"max_retries": _MAX_RETRIES, "limit": _BATCH_SIZE}).mappings().fetchall()

    if not rows:
        logger.info("No pending segments to embed")
        return 0

    ids = [str(r["id"]) for r in rows]
    texts = [r["text"] for r in rows]
    logger.info("Embedding %d pending segments", len(ids))

    # Insert pending log entries
    _insert_pending_logs(engine, ids, model)

    try:
        # Batch embed via OpenAI
        resp = client.embeddings.create(input=texts, model=model)
        embeddings = [d.embedding for d in resp.data]

        # Update transcript_segments (brain-owned table — allowed write exception)
        with engine.begin() as conn:
            for seg_id, emb in zip(ids, embeddings):
                emb_literal = "[" + ",".join(str(x) for x in emb) + "]"
                conn.execute(
                    text(
                        "UPDATE transcript_segments SET embedding = CAST(:emb AS vector), embedding_model = :model WHERE id = :id"
                    ),
                    {"emb": emb_literal, "model": model, "id": seg_id},
                )

        # Update ingestion log to DONE
        _update_logs_done(engine, ids, now)
        logger.info("Successfully embedded %d segments", len(ids))
        return len(ids)

    except RateLimitError:
        logger.warning("Rate limited while embedding segments, will retry")
        _update_logs_failed(engine, ids, "rate_limited")
        return 0
    except Exception as exc:
        logger.exception("Failed to embed segments: %s", exc)
        _update_logs_failed(engine, ids, str(exc)[:500])
        return 0


# ── Internal helpers ───────────────────────────────────────────────────────


def _insert_pending_logs(engine, ids: list[str], model: str):
    """Insert rag_ingestion_log rows with status PROCESSING."""
    with engine.begin() as conn:
        for seg_id in ids:
            conn.execute(
                text(
                    """
                    INSERT INTO rag_ingestion_log (transcript_segment_id, source_type, embedding_model, status)
                    VALUES (:seg_id, 'segment', :model, 'PROCESSING')
                    ON CONFLICT DO NOTHING
                    """
                ),
                {"seg_id": seg_id, "model": model},
            )


def _update_logs_done(engine, ids: list[str], now: datetime):
    """Mark ingestion log entries as DONE."""
    with engine.begin() as conn:
        for seg_id in ids:
            conn.execute(
                text(
                    """
                    UPDATE rag_ingestion_log
                    SET status = 'DONE', embedded_at = :now
                    WHERE transcript_segment_id = :seg_id AND source_type = 'segment'
                    """
                ),
                {"seg_id": seg_id, "now": now},
            )


def _update_logs_failed(engine, ids: list[str], error_msg: str):
    """Mark ingestion log entries as FAILED and increment retry_count."""
    with engine.begin() as conn:
        for seg_id in ids:
            conn.execute(
                text(
                    """
                    UPDATE rag_ingestion_log
                    SET status = 'FAILED',
                        retry_count = retry_count + 1,
                        error_message = :error_msg
                    WHERE transcript_segment_id = :seg_id AND source_type = 'segment'
                    """
                ),
                {"seg_id": seg_id, "error_msg": error_msg},
            )
