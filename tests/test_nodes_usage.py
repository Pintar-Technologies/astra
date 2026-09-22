from __future__ import annotations

from app.graph import nodes


def test_fallback_has_no_answer_usage_event():
    result = __import__("asyncio").run(nodes.generate_fallback({"question": "x"}))
    assert "usage_event" not in result
    assert result["tokens_used"]["output_tokens"] == 0


def test_ingestion_is_not_user_billable():
    import app.ingestion.embed_pdfs as pdfs
    import app.ingestion.embed_segments as segments
    assert pdfs.BILLABLE_TO_USER is False
    assert segments.BILLABLE_TO_USER is False
