from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from langgraph.graph.graph import CompiledGraph

from app.config import settings
from app.graph.build import build_graph
from app.models.schemas import QueryRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def verify_internal_key(internal_api_key: str = Header("", alias="INTERNAL_API_KEY")):
    if not settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="INTERNAL_API_KEY not configured")
    if internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return internal_api_key


def _sse_frame(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _empty_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _add_usage(total: dict[str, int], event: dict) -> None:
    for key in ("input_tokens", "output_tokens", "reasoning_tokens", "cache_read_tokens", "cache_write_tokens"):
        total[key] += int(event.get(key, 0) or 0)


@router.post("/rag/query")
async def rag_query(
    body: QueryRequest,
    request: Request,
    _auth: str = Depends(verify_internal_key),
):
    request_id = uuid.uuid4().hex
    logger.info("RAG query request_id=%s generation_id=%s video=%s", request_id, body.generation_id, body.lesson_video_id)
    graph: CompiledGraph = build_graph()
    inputs = {
        "question": body.question,
        "lesson_video_id": body.lesson_video_id,
        "module_id": body.module_id,
        "session_history": body.session_history or [],
        "generation_id": body.generation_id,
        "user_id": body.user_id,
        "routing": body.routing.model_dump(),
        "needs_broaden": False,
        "retrieved_docs": [],
        "graded_docs": [],
        "answer": "",
        "citations": [],
        "tokens_used": _empty_usage(),
    }
    return StreamingResponse(
        _stream_events(request_id, body.generation_id, graph, inputs, request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


async def _stream_events(
    request_id: str,
    generation_id: str,
    graph: CompiledGraph,
    inputs: dict,
    request: Request,
) -> AsyncGenerator[str, None]:
    yield _sse_frame("started", {"event": "started", "request_id": request_id, "generation_id": generation_id})
    queue: asyncio.Queue = asyncio.Queue()
    answer_text = ""
    citations: list[dict] = []
    aggregate_usage = _empty_usage()
    model_name = inputs.get("routing", {}).get("answer", {}).get("model_id", "")

    async def _run_graph():
        nonlocal answer_text, citations
        try:
            async for event in graph.astream_events(inputs, version="v1"):
                if request and await request.is_disconnected():
                    await queue.put(("cancelled", None))
                    return
                event_name = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})
                if event_name == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    token = ""
                    if chunk is not None and hasattr(chunk, "content"):
                        token = chunk.content or ""
                    elif isinstance(chunk, dict):
                        token = chunk.get("content", "")
                    if token:
                        await queue.put(("token", token))
                elif event_name == "on_chain_end":
                    output = data.get("output", {})
                    if isinstance(output, dict):
                        usage_event = output.get("usage_event")
                        if isinstance(usage_event, dict):
                            await queue.put(("usage", usage_event))
                        if name in {"generate", "generate_fallback"}:
                            answer_text = output.get("answer", "")
                        if name == "cite":
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
                yield _sse_frame("heartbeat", {"event": "heartbeat", "request_id": request_id})
                continue
            event_type, payload = item
            if event_type == "token":
                yield _sse_frame("chunk", {"event": "chunk", "request_id": request_id, "content": payload})
            elif event_type == "usage":
                _add_usage(aggregate_usage, payload)
                yield _sse_frame("usage", {"event": "usage", "request_id": request_id, **payload})
            elif event_type == "done":
                yield _sse_frame("completed", {
                    "event": "completed", "request_id": request_id, "generation_id": generation_id,
                    "answer": answer_text, "citations": citations, "model": model_name,
                    "usage": aggregate_usage, "tokens_used": aggregate_usage,
                })
                return
            elif event_type == "cancelled":
                yield _sse_frame("cancelled", {"event": "cancelled", "request_id": request_id, "generation_id": generation_id, "usage": aggregate_usage})
                return
            elif event_type == "error":
                raise payload
    except asyncio.CancelledError:
        logger.info("Stream cancelled for %s", request_id)
        task.cancel()
        return
    except Exception as exc:  # noqa: BLE001
        logger.error("Stream error for %s: %s", request_id, str(exc))
        yield _sse_frame("error", {
            "event": "error", "request_id": request_id, "generation_id": generation_id,
            "message": "ASTI sedang istirahat sebentar, coba lagi ya Kakak!", "error_code": "rag_error",
            "usage": aggregate_usage,
        })
