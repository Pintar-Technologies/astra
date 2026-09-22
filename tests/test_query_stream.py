from __future__ import annotations

import asyncio
import json

from app.routes.query import _stream_events


class FakeRequest:
    def __init__(self, disconnect_after: int | None = None):
        self.calls = 0
        self.disconnect_after = disconnect_after

    async def is_disconnected(self):
        self.calls += 1
        return self.disconnect_after is not None and self.calls > self.disconnect_after


class FakeGraph:
    def __init__(self, events):
        self.events = events

    async def astream_events(self, inputs, version):
        for event in self.events:
            yield event


def parse_frames(frames):
    parsed = []
    for frame in frames:
        data = frame.split("data: ", 1)[1].strip()
        parsed.append((frame.split("event: ", 1)[1].split("\n", 1)[0], json.loads(data)))
    return parsed


def run_stream(events, disconnect_after=None):
    async def collect():
        result = []
        async for frame in _stream_events("request-1", "generation-1", FakeGraph(events), {"routing": {"answer": {"model_id": "model"}}}, FakeRequest(disconnect_after)):
            result.append(frame)
        return result
    return asyncio.run(collect())


def usage(purpose, call_id):
    return {"call_id": call_id, "purpose": purpose, "provider": "openrouter", "model": "model", "model_version_id": "v1", "policy_version": "p1", "provider_request_id": call_id, "input_tokens": 10, "output_tokens": 2, "reasoning_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "estimated": False, "billable_to_user": True, "generation_id": "generation-1"}


def chain(name, payload):
    return {"event": "on_chain_end", "name": name, "data": {"output": payload}}


def test_one_pass_emits_three_ordered_usage_events():
    events = [chain("retrieve", {"usage_event": usage("embedding", "c1")}), chain("grade_relevance", {"usage_event": usage("grader", "c2")}), chain("generate", {"answer": "answer", "usage_event": usage("answer", "c3")}), chain("cite", {"citations": []})]
    parsed = parse_frames(run_stream(events))
    assert [data["purpose"] for event, data in parsed if event == "usage"] == ["embedding", "grader", "answer"]
    assert parsed[-1][0] == "completed"
    assert parsed[-1][1]["usage"]["input_tokens"] == 30


def test_broadened_pass_emits_five_child_usage_events():
    events = [
        chain("retrieve", {"usage_event": usage("embedding", "c1")}),
        chain("grade_relevance", {"usage_event": usage("grader", "c2")}),
        chain("retrieve", {"usage_event": usage("embedding", "c3")}),
        chain("grade_relevance", {"usage_event": usage("grader", "c4")}),
        chain("generate", {"answer": "answer", "usage_event": usage("answer", "c5")}),
    ]
    parsed = parse_frames(run_stream(events))
    assert [data["purpose"] for event, data in parsed if event == "usage"] == ["embedding", "grader", "embedding", "grader", "answer"]


def test_disconnect_after_grading_emits_cancelled_without_answer_call():
    events = [chain("retrieve", {"usage_event": usage("embedding", "c1")}), chain("grade_relevance", {"usage_event": usage("grader", "c2")}), chain("generate", {"answer": "should-not-run", "usage_event": usage("answer", "c3")})]
    parsed = parse_frames(run_stream(events, disconnect_after=2))
    assert [data["purpose"] for event, data in parsed if event == "usage"] == ["embedding", "grader"]
    assert parsed[-1][0] == "cancelled"
