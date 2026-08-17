from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    module_id: str | None
    lesson_video_id: str
    session_history: list[dict]
    retrieved_docs: list[dict]  # {id, text, start_sec, end_sec, source_type, score, video_id, ...}
    graded_docs: list[dict]
    answer: str
    citations: list[dict]
    needs_broaden: bool
    tokens_used: dict[str, Any]
