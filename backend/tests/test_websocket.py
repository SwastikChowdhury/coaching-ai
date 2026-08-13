"""End-to-end tests of the /ws frame protocol and chat router lifecycle.

Covers transport contracts (history + streaming frames), auth gating, crisis
short-circuit (agents never run), and whisper persistence rules.
"""

import pytest
from app.api import chat
from app.main import app
from starlette.websockets import WebSocketDisconnect


def _stub_db(monkeypatch, *, saved_messages=None, saved_whispers=None):
    """Replace Mongo persistence with in-memory collectors."""
    saved_messages = saved_messages if saved_messages is not None else []
    saved_whispers = saved_whispers if saved_whispers is not None else []

    async def fake_get_history(uid, cid):
        return []

    async def fake_get_whispers(uid, cid):
        return []

    async def fake_save_message(msg):
        saved_messages.append(msg)

    async def fake_save_whisper(msg):
        saved_whispers.append(msg)

    monkeypatch.setattr(chat, "get_history", fake_get_history)
    monkeypatch.setattr(chat, "get_whispers", fake_get_whispers)
    monkeypatch.setattr(chat, "save_message", fake_save_message)
    monkeypatch.setattr(chat, "save_whisper", fake_save_whisper)

    async def fake_save_flagged(msg):
        return None

    monkeypatch.setattr(chat, "save_flagged", fake_save_flagged)
    return saved_messages, saved_whispers


def test_websocket_protocol(monkeypatch):
    """Connect -> history -> token/done/whisper; truthy whisper is persisted."""
    saved_messages, saved_whispers = _stub_db(monkeypatch)

    async def fake_handle_turn(ws, history, user_message, user_id, conversation_id):
        await ws.send_json({"type": "token", "content": "Hi"})
        await ws.send_json({"type": "done"})
        await ws.send_json({"type": "whisper", "content": "coaching note", "label": "Tone"})
        mentee_mod = {"flagged": False, "flag_type": None, "emotions": None}
        return "Hi", "Tone", "coaching note", mentee_mod

    monkeypatch.setattr(chat, "handle_turn", fake_handle_turn)
    monkeypatch.setattr(chat, "verify_access_token", lambda token: "test-user")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws?token=test") as ws:
        assert ws.receive_json()["type"] == "history"
        ws.send_text("hello")
        types_ = [ws.receive_json()["type"] for _ in range(3)]
        assert types_ == ["token", "done", "whisper"]

    assert len(saved_whispers) == 1
    assert saved_whispers[0].label == "Tone"
    assert len(saved_messages) == 2  # user + assistant


def test_websocket_rejects_invalid_token(monkeypatch):
    """An invalid access token closes the socket with code 4001 before accept."""
    _stub_db(monkeypatch)
    monkeypatch.setattr(chat, "verify_access_token", lambda token: None)

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with pytest.raises(WebSocketDisconnect) as exc, client.websocket_connect("/ws?token=bad-token") as ws:
        ws.receive_json()
    assert exc.value.code == 4001


def test_websocket_crisis_short_circuit(monkeypatch):
    """Crisis input streams the hotline response and never calls handle_turn."""
    saved_messages, saved_whispers = _stub_db(monkeypatch)
    handle_turn_calls = []

    async def boom(*args, **kwargs):
        handle_turn_calls.append(1)
        raise AssertionError("handle_turn must not run on crisis escalation")

    monkeypatch.setattr(chat, "handle_turn", boom)
    monkeypatch.setattr(chat, "verify_access_token", lambda token: "test-user")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws?token=test") as ws:
        ws.receive_json()  # history
        ws.send_text("I want to kill myself")
        token_frame = ws.receive_json()
        assert token_frame["type"] == "token"
        assert "988" in token_frame["content"]
        assert ws.receive_json()["type"] == "done"

    assert handle_turn_calls == []
    assert saved_whispers == []
    assert len(saved_messages) == 2  # flagged user msg + crisis assistant reply
    assert saved_messages[0].flagged is True
    assert saved_messages[0].flag_type == "crisis"


def test_websocket_skips_whisper_when_none(monkeypatch):
    """When orchestrator returns whisper=None, nothing is persisted to whispers."""
    saved_messages, saved_whispers = _stub_db(monkeypatch)

    async def fake_handle_turn(ws, history, user_message, user_id, conversation_id):
        await ws.send_json({"type": "token", "content": "ok"})
        await ws.send_json({"type": "done"})
        mentee_mod = {"flagged": False, "flag_type": None, "emotions": None}
        return "ok", None, None, mentee_mod

    monkeypatch.setattr(chat, "handle_turn", fake_handle_turn)
    monkeypatch.setattr(chat, "verify_access_token", lambda token: "test-user")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws?token=test") as ws:
        ws.receive_json()
        ws.send_text("hello")
        types_ = [ws.receive_json()["type"] for _ in range(2)]
        assert types_ == ["token", "done"]

    assert saved_whispers == []
    assert len(saved_messages) == 2


def test_websocket_redacts_pii_at_intake(monkeypatch):
    """PII is stripped before persistence — raw email must not reach save_message."""
    saved_messages, _ = _stub_db(monkeypatch)
    captured = []

    async def fake_handle_turn(ws, history, user_message, user_id, conversation_id):
        captured.append(user_message)
        await ws.send_json({"type": "token", "content": "ok"})
        await ws.send_json({"type": "done"})
        mentee_mod = {"flagged": False, "flag_type": None, "emotions": None}
        return "ok", None, None, mentee_mod

    monkeypatch.setattr(chat, "handle_turn", fake_handle_turn)
    monkeypatch.setattr(chat, "verify_access_token", lambda token: "test-user")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    with client.websocket_connect("/ws?token=test") as ws:
        ws.receive_json()
        ws.send_text("Email me at jo@x.com please")
        ws.receive_json()
        ws.receive_json()

    stored = saved_messages[0].content
    assert "jo@x.com" not in stored
    assert "jo@x.com" not in captured[0]
