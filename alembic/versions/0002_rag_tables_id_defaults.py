"""Add server_default gen_random_uuid() to rag tables id columns.

Raw-SQL inserts into rag_ingestion_log omit id; 0001 created id as
Uuid() without server_default causing NotNullViolation on every cron run.
Set DB-side default gen_random_uuid() (built-in since PG13) for
rag_ingestion_log, rag_pdf_chunks and rag_pdf_ingestion_log so all
raw/ORM mixed writes are safe even without an explicit id.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-24
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    for table in ("rag_ingestion_log", "rag_pdf_chunks", "rag_pdf_ingestion_log"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN id SET DEFAULT gen_random_uuid()")


def downgrade() -> None:
    for table in ("rag_ingestion_log", "rag_pdf_chunks", "rag_pdf_ingestion_log"):
        op.execute(f"ALTER TABLE {table} ALTER COLUMN id DROP DEFAULT")
