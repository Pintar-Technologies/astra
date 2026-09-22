from __future__ import annotations

from collections import OrderedDict
from typing import Any

from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from openai import OpenAI
from redis import asyncio as aioredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_size=5)
    return _engine


_brain_engine = None


def get_brain_engine():
    global _brain_engine
    if _brain_engine is None:
        if not settings.BRAIN_DATABASE_URL:
            raise RuntimeError("BRAIN_DATABASE_URL is not configured")
        _brain_engine = create_engine(settings.BRAIN_DATABASE_URL, pool_size=5)
    return _brain_engine


def get_session_factory():
    return sessionmaker(bind=get_engine())


_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


_openrouter_client: OpenAI | None = None


def get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


# The route snapshot is the request authority. Cache only immutable client
# configuration; never cache a model-free singleton selected from settings.
_openrouter_chat_cache: OrderedDict[tuple[str, str, str, int, float], ChatOpenAI] = OrderedDict()
_MAX_ROUTE_CLIENTS = 32


def get_openrouter_chat(route: dict[str, Any] | None = None) -> ChatOpenAI:
    if route is None:
        route = {
            "provider": "openrouter",
            "model_id": settings.OPENROUTER_MODEL,
            "model_version_id": "legacy-settings",
            "max_output_tokens": 1024,
            "temperature": 0.4,
        }
    provider = str(route.get("provider", ""))
    if provider != "openrouter":
        raise ValueError(f"unsupported chat provider: {provider}")
    model = str(route.get("model_id", ""))
    version = str(route.get("model_version_id", ""))
    if not model or not version:
        raise ValueError("routing model and model version are required")
    max_tokens = int(route.get("max_output_tokens", 1024))
    temperature = float(route.get("temperature", 0.4) or 0.4)
    key = (provider, model, version, max_tokens, temperature)
    cached = _openrouter_chat_cache.get(key)
    if cached is not None:
        _openrouter_chat_cache.move_to_end(key)
        return cached
    client = ChatOpenAI(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={"stream_usage": True},
    )
    _openrouter_chat_cache[key] = client
    _openrouter_chat_cache.move_to_end(key)
    while len(_openrouter_chat_cache) > _MAX_ROUTE_CLIENTS:
        _openrouter_chat_cache.popitem(last=False)
    return client


_langfuse_handler: CallbackHandler | None = None


def get_langfuse_callback() -> CallbackHandler | None:
    global _langfuse_handler
    if _langfuse_handler is None and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
        _langfuse_handler = CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    return _langfuse_handler


_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
    return _redis
