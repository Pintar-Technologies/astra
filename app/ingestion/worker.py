from __future__ import annotations

from arq import cron
from arq.connections import RedisSettings

from app.config import settings
from app.ingestion.embed_pdfs import embed_pending_pdfs

# Transcript segment embedding (v2.5): embed_pending_segments writes embeddings
# back to brain's transcript_segments.embedding column via get_brain_engine()
# (the two-database write path). The column is provisioned idempotently by both
# brain's migration and this cron's guard.
from app.ingestion.embed_segments import embed_pending_segments

redis_settings = RedisSettings(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD or None,
    database=settings.REDIS_DB,
)


class WorkerSettings:
    """Arq worker configuration."""

    cron_jobs = [  # noqa: RUF012
        cron(embed_pending_segments, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}, second={45}, keep_result=0),
        cron(embed_pending_pdfs, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}, second={15}, keep_result=0),
    ]
    redis_settings = redis_settings
    max_tries = 1  # Let the cron job retry on next cycle
    job_timeout = 300  # 5 minutes max per job
    keep_result = 3600  # Keep results (incl. failed) for 1 hour for debugging
