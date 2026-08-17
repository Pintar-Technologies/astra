from __future__ import annotations

from arq import cron
from arq.connections import RedisSettings

from app.config import settings
from app.ingestion.embed_pdfs import embed_pending_pdfs

# Transcript embedding is intentionally deferred to v2.5.  The transcript
# tables belong to brain's database, while this deployment's DATABASE_URL
# points at the astra-owned pgvector database.  Do not re-enable this cron
# until the two-database write path (or an internal astra embedding endpoint)
# is implemented and verified.
# from app.ingestion.embed_segments import embed_pending_segments

redis_settings = RedisSettings(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD or None,
    database=settings.REDIS_DB,
)


class WorkerSettings:
    """Arq worker configuration."""

    functions = [
        # Deferred to v2.5; see the module comment above.
        # cron(embed_pending_segments, second={0}, keep_result=0),
        cron(embed_pending_pdfs, minute={0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55}, second={15}, keep_result=0),
    ]
    redis_settings = redis_settings
    max_tries = 1  # Let the cron job retry on next cycle
    max_duration = 300  # 5 minutes max per job
    keep_result = 0  # Don't keep results
    keep_result_failed = 3600  # Keep failed results for 1 hour
