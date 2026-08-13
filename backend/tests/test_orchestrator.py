"""Unit tests for orchestrator.handle_turn.

Every external dependency (the two agents, memory) is monkeypatched so these
tests are deterministic and make no network/LLM calls — they verify the
orchestration logic itself: streaming order, graceful quota handling, and the
"whisper failed -> return None" contract that main.py relies on for persistence.
"""

import asyncio
from types import SimpleNamespace

from app.agents import orchestrator
from prometheus_client import REGISTRY


class FakeWebSocket:
    """Captures every frame handle_turn streams, so tests can assert on them."""
    def __init__(self):
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


def _coaching_turns() -> float:
    value = REGISTRY.get_sample_value("muse_coaching_turns_total")
    return value if value is not None else 0.0


def _chunks(*texts):
    """Build a list of objects mimicking the conversation agent's stream chunks.

    The real stream yields objects with a `.text` attribute; SimpleNamespace is
    the cheapest stand-in.
    """
    return [SimpleNamespace(text=t) for t in texts]


def test_handle_turn_happy_path(monkeypatch):
    """Normal turn streams tokens, signals done, and emits a whisper, returning both values."""
    ws = FakeWebSocket()
    monkeypatch.setattr(orchestrator, "get_relevant_memories", lambda uid, q: [])
    monkeypatch.setattr(orchestrator, "add_memory", lambda uid, t: None)
    monkeypatch.setattr(orchestrator, "conversation_agent_stream",
                        lambda h, m: _chunks("Hello ", "there"))
    monkeypatch.setattr(orchestrator, "whisper_agent",
                        lambda h, m, r, p: ("Tone", "Nice opening."))
    # Stub the mentee-output moderation so the test stays model-free and
    # deterministic; a benign verdict skips the toxic-fallback/save_flagged path.
    monkeypatch.setattr(orchestrator, "moderate",
                        lambda text, role: {"emotions": None, "toxic_scores": None})

    before = _coaching_turns()
    full_reply, whisper_label, whisper, mentee_mod = asyncio.run(
        orchestrator.handle_turn(ws, [], "Hi Alex", "demo-user", "demo-conversation")
    )

    assert _coaching_turns() == before + 1
    assert full_reply == "Hello there"
    assert whisper_label == "Tone"
    assert whisper == "Nice opening."
    assert mentee_mod["flagged"] is False
    types_sent = [p["type"] for p in ws.sent]
    assert "token" in types_sent and "done" in types_sent and "whisper" in types_sent


def test_handle_turn_quota_exhausted(monkeypatch):
    """A 429 from the conversation agent degrades to the friendly QUOTA_MSG, not a crash."""
    ws = FakeWebSocket()
    monkeypatch.setattr(orchestrator, "get_relevant_memories", lambda uid, q: [])
    monkeypatch.setattr(orchestrator, "add_memory", lambda uid, t: None)

    def boom(h, m):
        raise Exception("429 RESOURCE_EXHAUSTED")  # noqa: TRY002 — matches production catch
    monkeypatch.setattr(orchestrator, "conversation_agent_stream", boom)
    monkeypatch.setattr(orchestrator, "whisper_agent", lambda h, m, r, p: ("Tone", "note"))
    monkeypatch.setattr(orchestrator, "moderate",
                        lambda text, role: {"emotions": None, "toxic_scores": None})

    before = _coaching_turns()
    full_reply, whisper_label, whisper, _mentee_mod = asyncio.run(
        orchestrator.handle_turn(ws, [], "Hi", "demo-user", "demo-conversation")
    )

    assert _coaching_turns() == before
    assert "limit" in full_reply.lower()   # the graceful QUOTA_MSG, not a crash
    assert whisper_label == "Tone"
    assert whisper == "note"


def test_handle_turn_conversation_fallback_does_not_count_turn(monkeypatch):
    """A failed conversation with a successful whisper is not a completed turn."""
    ws = FakeWebSocket()

    async def no_sleep(*a, **k):
        return
    monkeypatch.setattr(orchestrator.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(orchestrator, "get_relevant_memories", lambda uid, q: [])
    monkeypatch.setattr(orchestrator, "add_memory", lambda uid, t: None)

    def boom(h, m):
        raise Exception("503 overloaded")  # noqa: TRY002 — matches production catch
    monkeypatch.setattr(orchestrator, "conversation_agent_stream", boom)
    monkeypatch.setattr(orchestrator, "whisper_agent", lambda h, m, r, p: ("Tone", "note"))
    monkeypatch.setattr(orchestrator, "moderate",
                        lambda text, role: {"emotions": None, "toxic_scores": None})

    before = _coaching_turns()
    full_reply, _whisper_label, whisper, _mentee_mod = asyncio.run(
        orchestrator.handle_turn(ws, [], "Hi", "demo-user", "demo-conversation")
    )

    assert full_reply == orchestrator.CONVERSATION_FALLBACK
    assert whisper == "note"
    assert _coaching_turns() == before


def test_handle_turn_whisper_failure_returns_none(monkeypatch):
    """When the whisper agent fails every retry, the reply still succeeds and whisper is None.

    The None is the signal main.py uses to NOT persist a transient failure note.
    """
    ws = FakeWebSocket()

    async def no_sleep(*a, **k):
        return
    # Patch out the retry backoff so the 3-attempt loop runs instantly.
    monkeypatch.setattr(orchestrator.asyncio, "sleep", no_sleep)  # skip retry backoff
    monkeypatch.setattr(orchestrator, "get_relevant_memories", lambda uid, q: [])
    monkeypatch.setattr(orchestrator, "add_memory", lambda uid, t: None)
    monkeypatch.setattr(orchestrator, "conversation_agent_stream", lambda h, m: _chunks("ok"))
    monkeypatch.setattr(orchestrator, "moderate",
                        lambda text, role: {"emotions": None, "toxic_scores": None})

    def boom(h, m, r, p):
        raise Exception("503 overloaded")  # noqa: TRY002 — matches production catch
    monkeypatch.setattr(orchestrator, "whisper_agent", boom)

    before = _coaching_turns()
    full_reply, _whisper_label, whisper, _mentee_mod = asyncio.run(
        orchestrator.handle_turn(ws, [], "Hi", "demo-user", "demo-conversation")
    )

    assert _coaching_turns() == before
    assert full_reply == "ok"
    assert whisper is None   # so main.py won't persist a fallback note


def test_conversation_keeps_chunk_with_real_usage(monkeypatch):
    """Empty stream usage_metadata must not overwrite the chunk that has billed tokens."""
    ws = FakeWebSocket()
    recorded = []

    empty = SimpleNamespace(prompt_token_count=0, candidates_token_count=None)
    billed = SimpleNamespace(
        prompt_token_count=120, candidates_token_count=40, thoughts_token_count=10
    )
    chunks = [
        SimpleNamespace(text="Hello ", usage_metadata=empty, model_version=None),
        SimpleNamespace(text="there", usage_metadata=billed, model_version="gemini-3.5-flash"),
        SimpleNamespace(text="", usage_metadata=empty, model_version="gemini-3.5-flash"),
    ]

    monkeypatch.setattr(orchestrator, "get_relevant_memories", lambda uid, q: [])
    monkeypatch.setattr(orchestrator, "add_memory", lambda uid, t: None)
    monkeypatch.setattr(orchestrator, "conversation_agent_stream", lambda h, m: chunks)
    monkeypatch.setattr(orchestrator, "whisper_agent",
                        lambda h, m, r, p: ("Tone", "note"))
    monkeypatch.setattr(orchestrator, "moderate",
                        lambda text, role: {"emotions": None, "toxic_scores": None})
    monkeypatch.setattr(
        orchestrator, "record_usage",
        lambda agent, usage, model=None: recorded.append((agent, usage, model)),
    )

    asyncio.run(orchestrator.handle_turn(ws, [], "Hi", "demo-user", "demo-conversation"))

    assert recorded == [("conversation", billed, "gemini-3.5-flash")]
