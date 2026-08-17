from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.graph import CompiledGraph

from app.config import settings
from app.graph.build import build_graph
from app.models.schemas import QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Auth dependency ────────────────────────────────────────────────────────


async def verify_internal_key(internal_api_key: str = Header("", alias="INTERNAL_API_KEY")):
    if not settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured")
    if internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return internal_api_key


# ── SSE helpers ────────────────────────────────────────────────────────────


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Streaming endpoint ─────────────────────────────────────────────────────


@router.post("/rag/query")
async def rag_query(
    body: QueryRequest,
    request: Request,
    _auth: str = Depends(verify_internal_key),
):
    request_id = uuid.uuid4().hex
    logger.info("RAG query request_id=%s video=%s module=%s", request_id, body.lesson_video_id, body.module_id)

    graph: CompiledGraph = build_graph()

    inputs = {
        "question": body.question,
        "lesson_video_id": body.lesson_video_id,
        "module_id": body.module_id if body.module_id else None,
        "session_history": body.session_history or [],
        "needs_broaden": False,
        "retrieved_docs": [],
        "graded_docs": [],
        "answer": "",
        "citations": [],
        "tokens_used": {},
    }

    return StreamingResponse(
        _stream_events(request_id, graph, inputs, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── SSE event generator ────────────────────────────────────────────────────


async def _stream_events(
    request_id: str,
    graph: CompiledGraph,
    inputs: dict,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Generate SSE frames for the RAG query."""
    # Emit started
    yield _sse_frame("started", {"event": "started", "request_id": request_id})

    queue: asyncio.Queue = asyncio.Queue()
    answer_text = ""
    citations: list[dict] = []
    tokens_used: dict = {}
    model_name = settings.OPENROUTER_MODEL

    async def _run_graph():
        """Execute graph and push events to the queue."""
        nonlocal answer_text, citations, tokens_used
        try:
            async for event in graph.astream_events(inputs, version="v1"):
                if request and await request.is_disconnected():
                    break

                event_name = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                if event_name == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk is not None:
                        token = ""
                        if hasattr(chunk, "content"):
                            token = chunk.content or ""
                        elif isinstance(chunk, dict):
                            token = chunk.get("content", "")
                        if token:
                            await queue.put(("token", token))

                elif event_name == "on_chain_end" and name == "generate":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        answer_text = output.get("answer", "")
                        tokens_used = output.get("tokens_used", {})

                elif event_name == "on_chain_end" and name == "cite":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        citations = output.get("citations", [])

            await queue.put(("done", None))

        except Exception as exc:
            logger.exception("Graph execution failed for %s", request_id)
            await queue.put(("error", exc))

    task = asyncio.create_task(_run_graph())

    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Heartbeat — no event for 30s
                yield _sse_frame("heartbeat", {"event": "heartbeat"})
                continue

            event_type, payload = item

            if event_type == "token":
                yield _sse_frame("chunk", {
                    "event": "chunk",
                    "request_id": request_id,
                    "content": payload,
                })

            elif event_type == "done":
                final_answer = answer_text
                yield _sse_frame("completed", {
                    "event": "completed",
                    "request_id": request_id,
                    "answer": final_answer,
                    "citations": citations,
                    "model": model_name,
                    "tokens_used": tokens_used,
                })
                return

            elif event_type == "error":
                raise payload

    except asyncio.CancelledError:
        logger.info("Stream cancelled for %s", request_id)
        task.cancel()
        return
    except Exception as exc:
        logger.error("Stream error for %s: %s", request_id, str(exc))
        yield _sse_frame("error", {
            "event": "error",
            "request_id": request_id,
            "message": "ASTI sedang istirahat sebentar, coba lagi ya Kakak!",
            "error_code": "rag_error",
        })
        return
