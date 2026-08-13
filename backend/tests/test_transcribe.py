"""Test for the /transcribe endpoint with the Groq Whisper client stubbed out."""

import io
from types import SimpleNamespace

from app.api import chat
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_transcribe(monkeypatch):
    """Uploaded audio is passed to the (mocked) STT client and its text is returned verbatim.

    The nested SimpleNamespace mirrors the real client's
    `groq_client.audio.transcriptions.create(...)` call chain so main.py needs
    no changes; we assert only on the endpoint's response shape, not on Groq.
    """
    fake_groq = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(text="hello from voice")
            )
        )
    )
    monkeypatch.setattr(chat, "groq_client", fake_groq)

    resp = client.post(
        "/transcribe",
        files={"audio": ("audio.webm", io.BytesIO(b"fake-bytes"), "audio/webm")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "hello from voice"}