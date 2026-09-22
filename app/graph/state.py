from __future__ import annotations

from typing import Any, TypedDict


class RAGState(TypedDict, total=False):
    question: str
    module_id: str | None
    lesson_video_id: str
    session_history: list[dict]
    generation_id: str
    user_id: str
    routing: dict[str, Any]
    retrieved_docs: list[dict]
    graded_docs: list[dict]
    answer: str
    citations: list[dict]
    needs_broaden: bool
    tokens_used: dict[str, Any]
    usage_event: dict[str, Any]
