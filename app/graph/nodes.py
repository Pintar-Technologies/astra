from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessageChunk
from openai import RateLimitError
from sqlalchemy import text

from app.config import settings
from app.deps import get_engine, get_brain_engine, get_openai_client, get_openrouter_chat, get_openrouter_client
from app.graph.state import RAGState

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────────────


def _embed(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts via OpenAI embedding API."""
    client = get_openai_client()
    resp = client.embeddings.create(input=texts, model=settings.EMBEDDING_MODEL)
    return [d.embedding for d in resp.data]


def _format_embedding_for_query(emb: list[float]) -> str:
    """Format embedding list as pgvector-compatible literal string."""
    return "[" + ",".join(str(x) for x in emb) + "]"


# ── Retrieval ──────────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> dict[str, Any]:
    """Embed the question and run pgvector similarity search."""
    question = state.get("question", "")
    module_id = state.get("module_id")
    lesson_video_id = state.get("lesson_video_id")
    k = 20 if state.get("needs_broaden") else 10

    emb = _embed([question])[0]
    emb_literal = _format_embedding_for_query(emb)

    engine = get_engine()
    brain_engine = get_brain_engine()

    docs = []

    if module_id:
        # module-scoped: video segments (brain DB) + pdf chunks (pgvector DB)
        video_sql = text(
            """
            SELECT ts.id::text AS id, ts.text, ts.start_sec, ts.end_sec,
                   'video' AS source_type,
                   lv.id::text AS video_id, lv.title AS video_title,
                   NULL::text AS lesson_id, NULL::text AS lesson_title,
                   NULL::int AS page_start, NULL::int AS page_end,
                   ts.embedding <=> CAST(:emb AS vector) AS distance
            FROM transcript_segments ts
            JOIN transcripts t ON ts.transcript_id = t.id
            JOIN lesson_videos lv ON t.lesson_video_id = lv.id
            WHERE lv.module_id = :module_id
              AND t.status = 'DONE'
              AND ts.embedding IS NOT NULL
            ORDER BY distance
            LIMIT :k
            """
        )
        with brain_engine.connect() as conn:
            video_rows = conn.execute(
                video_sql, {"emb": emb_literal, "module_id": module_id, "k": k}
            ).mappings().fetchall()

        pdf_sql = text(
            """
            SELECT c.id::text AS id, c.text, NULL::int AS start_sec, NULL::int AS end_sec,
                   'pdf' AS source_type,
                   NULL::text AS video_id, NULL::text AS video_title,
                   c.lesson_id::text AS lesson_id, NULL::text AS lesson_title,
                   c.page_start, c.page_end,
                   c.embedding <=> CAST(:emb AS vector) AS distance
            FROM rag_pdf_chunks c
            WHERE c.module_id = :module_id
            ORDER BY distance
            LIMIT :k
            """
        )
        with engine.connect() as conn:
            pdf_rows = conn.execute(
                pdf_sql, {"emb": emb_literal, "module_id": module_id, "k": k}
            ).mappings().fetchall()

        # Fetch lesson titles from the brain DB (no cross-db join)
        lesson_ids = sorted({str(r["lesson_id"]) for r in pdf_rows if r["lesson_id"]})
        title_map: dict[str, str] = {}
        if lesson_ids:
            with brain_engine.connect() as conn:
                title_rows = conn.execute(
                    text("SELECT id::text AS id, title FROM lessons WHERE id::text = ANY(:ids)"),
                    {"ids": lesson_ids},
                ).mappings().fetchall()
            title_map = {r["id"]: r["title"] for r in title_rows}

        for row in video_rows:
            d = dict(row)
            d["score"] = float(1.0 - d.pop("distance", 0.0))
            docs.append(d)
        for row in pdf_rows:
            d = dict(row)
            d["score"] = float(1.0 - d.pop("distance", 0.0))
            d["lesson_title"] = title_map.get(row["lesson_id"])
            docs.append(d)
    else:
        # video-only scope (brain DB)
        video_sql = text(
            """
            SELECT ts.id::text AS id, ts.text, ts.start_sec, ts.end_sec,
                   'video' AS source_type,
                   lv.id::text AS video_id, lv.title AS video_title,
                   NULL::text AS lesson_id, NULL::text AS lesson_title,
                   NULL::int AS page_start, NULL::int AS page_end,
                   ts.embedding <=> CAST(:emb AS vector) AS distance
            FROM transcript_segments ts
            JOIN transcripts t ON ts.transcript_id = t.id
            JOIN lesson_videos lv ON t.lesson_video_id = lv.id
            WHERE t.lesson_video_id = :lesson_video_id
              AND t.status = 'DONE'
              AND ts.embedding IS NOT NULL
            ORDER BY distance
            LIMIT :k
            """
        )
        with brain_engine.connect() as conn:
            video_rows = conn.execute(
                video_sql, {"emb": emb_literal, "lesson_video_id": lesson_video_id, "k": k}
            ).mappings().fetchall()

        for row in video_rows:
            d = dict(row)
            d["score"] = float(1.0 - d.pop("distance", 0.0))
            docs.append(d)

    docs.sort(key=lambda x: x["score"], reverse=True)
    docs = docs[:k]

    logger.info("Retrieved %d docs (module_id=%s, video=%s)", len(docs), module_id, lesson_video_id)
    return {"retrieved_docs": docs}


# ── Relevance grading ──────────────────────────────────────────────────────


def grade_relevance(state: RAGState) -> dict[str, Any]:
    """Grade each retrieved doc for relevance to the question."""
    docs = state.get("retrieved_docs", [])
    question = state.get("question", "")
    needs_broaden = state.get("needs_broaden", False)

    if not docs:
        if not needs_broaden:
            return {"needs_broaden": True, "graded_docs": []}
        return {"graded_docs": [], "needs_broaden": True}

    # Build a prompt asking the LLM to grade each doc
    doc_texts = []
    for i, d in enumerate(docs):
        snippet = d.get("text", "")[:300]
        doc_texts.append(f"[{i}] {snippet}")

    prompt = (
        "Anda adalah sistem penilai relevansi. Tentukan apakah setiap potongan dokumen "
        "berikut RELEVAN (1) atau TIDAK RELEVAN (0) terhadap pertanyaan pengguna.\n\n"
        f"Pertanyaan: {question}\n\n"
        "Dokumen:\n" + "\n".join(doc_texts) + "\n\n"
        "Balas hanya dengan array JSON angka 0 atau 1, misal [1, 0, 1, ...]."
    )

    client = get_openrouter_client()
    try:
        resp = client.chat.completions.create(
            model=settings.OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content or "[]"
        # Try to parse JSON array
        raw_clean = raw.strip().strip("```json").strip("```").strip()
        grades = json.loads(raw_clean)
    except Exception:
        logger.warning("Grade parsing failed, marking all as relevant")
        grades = [1] * len(docs)

    graded = []
    for i, d in enumerate(docs):
        grade = grades[i] if i < len(grades) else 0
        if grade == 1:
            graded.append(d)

    has_relevant = len(graded) > 0

    if not has_relevant:
        if needs_broaden:
            logger.info("No relevant docs even after broaden, will use fallback")
        else:
            logger.info("No relevant docs found, triggering broaden search")
        return {"graded_docs": []}

    logger.info("Graded %d/%d docs as relevant", len(graded), len(docs))
    return {"graded_docs": graded, "needs_broaden": False}


# ── Broaden search ─────────────────────────────────────────────────────────


def broaden_search(state: RAGState) -> dict[str, Any]:
    """Mark state for broader retrieval."""
    logger.info("Broadening search")
    return {"needs_broaden": True}


# ── Generate ───────────────────────────────────────────────────────────────


async def generate(state: RAGState) -> dict[str, Any]:
    """Generate answer with streaming from OpenRouter via LangChain ChatOpenAI.

    Returns the full answer; the SSE handler captures tokens via astream_events.
    """
    graded_docs = state.get("graded_docs", [])
    question = state.get("question", "")
    session_history = state.get("session_history", [])

    # Build context from graded docs
    context_parts = []
    for d in graded_docs:
        src = d.get("source_type", "")
        if src == "video":
            start = d.get("start_sec")
            end = d.get("end_sec")
            ts = f"[{_fmt_sec(start)} - {_fmt_sec(end)}]" if start is not None else ""
            title = d.get("video_title", "")
            context_parts.append(f"{ts} {d.get('text', '')} (Video: {title})")
        else:
            title = d.get("lesson_title", "")
            pages = ""
            if d.get("page_start") is not None:
                pages = f", hlm {d['page_start']}"
                if d.get("page_end") and d["page_end"] != d["page_start"]:
                    pages += f"-{d['page_end']}"
            context_parts.append(f"[PDF: {title}{pages}] {d.get('text', '')}")

    context_str = "\n\n".join(context_parts) if context_parts else ""

    # Build messages for Asti persona
    system_msg = (
        "Kamu adalah Asti, asisten belajar yang ramah dan sabar. "
        "Kamu membantu siswa memahami materi pelajaran dengan bahasa Indonesia yang jelas dan mudah dipahami. "
        "Jawab pertanyaan berdasarkan konteks yang diberikan. "
        "Jika tidak yakin, akui saja dengan jujur. Jangan membuat informasi palsu. "
        "Gunakan nada yang hangat dan suportif."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add session history
    for h in session_history[-10:]:
        role = h.get("role", "user")
        content = h.get("content", "")
        messages.append({"role": role, "content": content})

    # User message with context
    user_content = f"Konteks:\n{context_str}\n\nPertanyaan: {question}" if context_str else question
    messages.append({"role": "user", "content": user_content})

    # Use LangChain ChatOpenAI for streaming support in astream_events
    llm = get_openrouter_chat()

    answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    # Estimate prompt tokens
    total_chars = sum(len(m.get("content", "")) for m in messages)
    prompt_tokens = total_chars // 4

    try:
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessageChunk):
                token = chunk.content or ""
                answer += token
                # Track usage from final chunk
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    prompt_tokens = usage.get("input_tokens", prompt_tokens)
                    completion_tokens = usage.get("output_tokens", completion_tokens)
    except Exception:
        logger.exception("OpenRouter streaming failed in generate")
        answer = "Maaf, Asti mengalami kendala teknis. Silakan coba lagi."

    if not completion_tokens:
        completion_tokens = len(answer) // 4

    logger.info("Generated answer (%d chars)", len(answer))
    return {
        "answer": answer,
        "tokens_used": {"prompt": prompt_tokens, "completion": completion_tokens},
    }


def _fmt_sec(sec: float | int | None) -> str:
    if sec is None:
        return "0:00"
    sec = int(sec)
    m, s = divmod(sec, 60)
    return f"{m}:{s:02d}"


# ── Cite ───────────────────────────────────────────────────────────────────


def cite(state: RAGState) -> dict[str, Any]:
    """Build citations from graded docs."""
    graded_docs = state.get("graded_docs", [])
    citations = []
    seen = set()

    for d in graded_docs:
        src_type = d.get("source_type", "")
        if src_type == "video":
            key = (d.get("video_id", ""), d.get("id", ""))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "segment_id": d.get("id"),
                "video_id": d.get("video_id"),
                "video_title": d.get("video_title"),
                "start_sec": d.get("start_sec"),
                "end_sec": d.get("end_sec"),
                "text": d.get("text", ""),
                "source_type": "video",
            })
        else:
            key = (d.get("lesson_id", ""), d.get("id", ""))
            if key in seen:
                continue
            seen.add(key)
            citations.append({
                "segment_id": None,
                "video_id": None,
                "video_title": None,
                "lesson_id": d.get("lesson_id"),
                "lesson_title": d.get("lesson_title"),
                "page_start": d.get("page_start"),
                "page_end": d.get("page_end"),
                "text": d.get("text", ""),
                "source_type": "pdf",
            })

    logger.info("Built %d citations", len(citations))
    return {"citations": citations}


# ── Generate fallback ──────────────────────────────────────────────────────


async def generate_fallback(state: RAGState) -> dict[str, Any]:
    """Generate a fallback answer without context."""
    question = state.get("question", "")
    session_history = state.get("session_history", [])

    system_msg = (
        "Kamu adalah Asti, asisten belajar yang ramah dan sabar. "
        "Kamu membantu siswa memahami materi pelajaran dengan bahasa Indonesia yang jelas dan mudah dipahami. "
        "Saat ini kamu tidak memiliki akses ke konteks materi atau transkrip video yang relevan. "
        "Jawab pertanyaan sebaik mungkin berdasarkan pengetahuan umummu, "
        "dan akui jika kamu tidak yakin. Tetap ramah dan suportif."
    )

    messages = [{"role": "system", "content": system_msg}]
    for h in session_history[-10:]:
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
    messages.append({"role": "user", "content": question})

    llm = get_openrouter_chat()

    answer = ""
    prompt_tokens = 0
    completion_tokens = 0

    total_chars = sum(len(m.get("content", "")) for m in messages)
    prompt_tokens = total_chars // 4

    try:
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessageChunk):
                token = chunk.content or ""
                answer += token
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = chunk.usage_metadata
                    prompt_tokens = usage.get("input_tokens", prompt_tokens)
                    completion_tokens = usage.get("output_tokens", completion_tokens)
    except Exception:
        logger.exception("OpenRouter streaming failed in generate_fallback")
        answer = "Maaf, Asti mengalami kendala teknis. Silakan coba lagi."

    if not completion_tokens:
        completion_tokens = len(answer) // 4

    return {
        "answer": answer,
        "tokens_used": {"prompt": prompt_tokens, "completion": completion_tokens},
        "citations": [],
    }
