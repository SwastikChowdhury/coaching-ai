"""Schema tests for the Pydantic models: field defaults and Mongo serialization."""

from app.schemas.models import Conversation, Message, User


def test_message_fields_and_timestamp():
    """A Message keeps its fields and auto-populates created_at via the default factory."""
    m = Message(user_id="u1", conversation_id="c1", role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.created_at is not None


def test_message_serializes_for_mongo():
    """model_dump() (what db.py stores) preserves role/content and includes the timestamp.

    Also documents that "whisper" is a valid role on the same model, since
    whispers are persisted through this schema too.
    """
    m = Message(user_id="u1", conversation_id="c1", role="whisper", content="note", label="Tone")
    doc = m.model_dump()
    assert doc["role"] == "whisper"
    assert doc["content"] == "note"
    assert doc["label"] == "Tone"   # tone/category persisted with the whisper
    assert "created_at" in doc


def test_user_name_optional():
    """User.name is optional (None default) and Conversation stamps created_at."""
    assert User(user_id="u1").name is None
    assert Conversation(conversation_id="c1", user_id="u1").created_at is not None