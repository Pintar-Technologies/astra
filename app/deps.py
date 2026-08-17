from __future__ import annotations

from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from openai import OpenAI
from redis import asyncio as aioredis
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings

# ── SQLAlchemy engine ─────────────────────────────────────────────────────

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.DATABASE_URL, pool_size=5)
    return _engine


def get_session_factory():
    return sessionmaker(bind=get_engine())


# ── OpenAI client (embeddings) ────────────────────────────────────────────

_openai_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _openai_client


# ── OpenRouter client (chat generation) ───────────────────────────────────

_openrouter_client: OpenAI | None = None


def get_openrouter_client() -> OpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


# ── LangChain ChatOpenAI (for langgraph streaming support) ────────────────

_openrouter_chat: ChatOpenAI | None = None


def get_openrouter_chat() -> ChatOpenAI:
    global _openrouter_chat
    if _openrouter_chat is None:
        _openrouter_chat = ChatOpenAI(
            model=settings.OPENROUTER_MODEL,
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            temperature=0.4,
            max_tokens=1024,
            model_kwargs={"stream_usage": True},
        )
    return _openrouter_chat


# ── Langfuse callback ─────────────────────────────────────────────────────

_langfuse_handler: CallbackHandler | None = None


def get_langfuse_callback() -> CallbackHandler | None:
    global _langfuse_handler
    if _langfuse_handler is None:
        if settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY:
            _langfuse_handler = CallbackHandler(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
    return _langfuse_handler


# ── Redis ─────────────────────────────────────────────────────────────────

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
