"""Create initial RAG tables (rag_ingestion_log, rag_pdf_chunks, rag_pdf_ingestion_log).

Revision ID: 0001
Revises:
Create Date: 2026-07-18
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension (best-effort; may already exist)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── rag_ingestion_log ──────────────────────────────────────────────
    op.create_table(
        "rag_ingestion_log",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("transcript_segment_id", sa.Uuid(), nullable=True),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("embedding_model", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # ── rag_pdf_chunks ─────────────────────────────────────────────────
    op.create_table(
        "rag_pdf_chunks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("lesson_id", sa.Uuid(), nullable=False),
        sa.Column("module_id", sa.Uuid(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("embedding_model", sa.String(50), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lesson_id", "chunk_index"),
    )
    op.create_index(op.f("ix_rag_pdf_chunks_lesson_id"), "rag_pdf_chunks", ["lesson_id"])
    op.create_index(op.f("ix_rag_pdf_chunks_module_id"), "rag_pdf_chunks", ["module_id"])

    # ── rag_pdf_ingestion_log ──────────────────────────────────────────
    op.create_table(
        "rag_pdf_ingestion_log",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("lesson_id", sa.Uuid(), nullable=False),
        sa.Column("pdf_hash", sa.String(64), nullable=False),
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="PENDING"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("lesson_id"),
    )


def downgrade() -> None:
    op.drop_table("rag_pdf_ingestion_log")
    op.drop_table("rag_pdf_chunks")
    op.drop_table("rag_ingestion_log")
