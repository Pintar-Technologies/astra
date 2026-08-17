from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── SQLAlchemy models for astra-owned tables ──────────────────────────────


class Base(DeclarativeBase):
    pass


class RagIngestionLog(Base):
    __tablename__ = "rag_ingestion_log"

    id: uuid.UUID = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transcript_segment_id: uuid.UUID | None = Column(Uuid(as_uuid=True), nullable=True)
    source_type: str = Column(String(20), nullable=False)  # 'segment' | 'pdf_chunk'
    embedded_at: datetime | None = Column(DateTime(timezone=True), nullable=True)
    embedding_model: str = Column(String(50), nullable=False)
    status: str = Column(String(20), nullable=False, default="PENDING")  # PENDING|PROCESSING|DONE|FAILED
    retry_count: int = Column(Integer, nullable=False, default=0)
    error_message: str | None = Column(Text, nullable=True)


class RagPdfChunk(Base):
    __tablename__ = "rag_pdf_chunks"
    __table_args__ = (UniqueConstraint("lesson_id", "chunk_index"),)

    id: uuid.UUID = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lesson_id: uuid.UUID = Column(Uuid(as_uuid=True), index=True, nullable=False)
    module_id: uuid.UUID | None = Column(Uuid(as_uuid=True), index=True, nullable=True)
    chunk_index: int = Column(Integer, nullable=False)
    page_start: int | None = Column(Integer, nullable=True)
    page_end: int | None = Column(Integer, nullable=True)
    text: str = Column(Text, nullable=False)
    embedding: list[float] = Column(Vector(1536), nullable=True)
    embedding_model: str = Column(String(50), nullable=False)
    created_at: datetime = Column(DateTime(timezone=True), nullable=False, default=utcnow)


class RagPdfIngestionLog(Base):
    __tablename__ = "rag_pdf_ingestion_log"

    id: uuid.UUID = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    lesson_id: uuid.UUID = Column(Uuid(as_uuid=True), nullable=False)
    pdf_hash: str = Column(String(64), nullable=False)
    embedded_at: datetime | None = Column(DateTime(timezone=True), nullable=True)
    status: str = Column(String(20), nullable=False, default="PENDING")  # PENDING|PROCESSING|DONE|FAILED
    retry_count: int = Column(Integer, nullable=False, default=0)
    error_message: str | None = Column(Text, nullable=True)


# ── Pydantic request / response schemas ───────────────────────────────────


class QueryRequest(BaseModel):
    lesson_video_id: str
    module_id: str | None = None
    question: str = Field(..., max_length=1000)
    session_history: list[dict] = Field(default_factory=list)
    user_id: str = ""


class QueryResponse(BaseModel):
    request_id: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    model: str = ""
    tokens_used: dict = Field(default_factory=dict)
