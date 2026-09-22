from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.deps import _openrouter_chat_cache, get_openrouter_chat
from app.graph import nodes
from app.models.schemas import QueryRequest


def route(provider="openrouter", model="model-a", version="v1"):
    return {"provider": provider, "model_id": model, "model_version_id": version, "policy_version": "p1", "max_input_tokens": 1000, "max_output_tokens": 100, "temperature": 0.2}


def test_query_request_rejects_missing_routing():
    with pytest.raises(ValidationError):
        QueryRequest(lesson_video_id="video", question="hello", generation_id="g", user_id="u")


def test_unknown_provider_and_model_are_rejected():
    with pytest.raises(ValueError):
        nodes._route({"routing": {"answer": route(provider="unknown")}}, "answer")
    with pytest.raises(ValueError):
        nodes._route({"routing": {"answer": {"provider": "openrouter"}}}, "answer")


def test_route_client_cache_isolated_by_model_version(monkeypatch):
    created = []
    class FakeChat:
        def __init__(self, **kwargs):
            created.append(kwargs)
    monkeypatch.setattr("app.deps.ChatOpenAI", FakeChat)
    _openrouter_chat_cache.clear()
    first = get_openrouter_chat(route("openrouter", "model-a", "v1"))
    second = get_openrouter_chat(route("openrouter", "model-b", "v2"))
    again = get_openrouter_chat(route("openrouter", "model-a", "v1"))
    assert first is again
    assert first is not second
    assert [item["model"] for item in created] == ["model-a", "model-b"]


def test_embedding_usage_is_estimated_when_provider_omits_usage(monkeypatch):
    class Item:
        def __init__(self):
            self.embedding = [0.1, 0.2]
    class Response:
        def __init__(self):
            self.data = [Item()]
            self.usage = None
            self.id = "embed-1"
    class Client:
        class embeddings:
            @staticmethod
            def create(**kwargs):
                return Response()
    monkeypatch.setattr(nodes, "get_openai_client", lambda: Client())
    _, usage, request_id, estimated = nodes._embed(["abcdefgh"], route("openai", "embedding-model", "v1"))
    assert estimated is True
    assert usage["input_tokens"] == 2
    assert request_id == "embed-1"
