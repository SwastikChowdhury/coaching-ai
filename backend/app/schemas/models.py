"""
Pydantic data models — the canonical shape of records stored in MongoDB.

These define the wire/storage contract used throughout the backend: db.py calls
`.model_dump()` on them to produce BSON documents, and main.py constructs them
per turn. Timestamps default to timezone-aware UTC so ordering is unambiguous
regardless of server locale.

The current single-user demo only actively persists `Message`; `User` and
`Conversation` describe the intended multi-tenant schema and are the seam where
real per-user/per-conversation modelling would plug in.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class User(BaseModel):
    """An end user. `name` is optional to allow anonymous/demo users."""
    user_id: str
    name: str | None = None


class Conversation(BaseModel):
    """A single chat session, owned by a user and stamped at creation.

    `created_at` defaults to now-UTC via a factory so each instance gets its own
    timestamp (a plain default would freeze one value at import time).
    """
    conversation_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Message(BaseModel):
    """One stored utterance in a conversation.

    `role` is a free-form string by convention rather than an enum: "user" and
    "assistant" for the transcript, plus "whisper" for coaching notes (which are
    persisted in a separate collection but reuse this same model). `created_at`
    is the sort key used by db.get_history/get_whispers to replay turns in order.

    `label` only applies to whispers: it's the coach's one-word tone/category for
    the note (e.g. "Tone", "Pattern", "Empathy"; see agents.WHISPER_LABELS). It is
    None for transcript messages so the UI can re-render a persisted whisper with
    its original tag instead of defaulting everything to "Insight" on reconnect.
    """
    user_id: str
    conversation_id: str
    role: str  # "user" or "assistant"
    content: str
    label: str | None = None
    # Moderation observability fields. Set from the moderation pipeline result and
    # persisted in MongoDB for offline review, but deliberately NEVER included in
    # the WebSocket history/whisper payloads sent to the frontend.
    flagged: bool = False
    flag_type: str | None = None  # "crisis", "toxic", "both"
    # Full emotion distribution recorded for EVERY message (mentor and mentee),
    # so emotional trends can be tracked over time. Observability only — never
    # gates anything and never sent to the frontend.
    emotions: dict | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FlaggedMessage(BaseModel):
    """A message the moderation pipeline flagged, stored in its own collection.

    This is an observation-only record: the full moderation signal (crisis proxy
    score and per-category toxicity scores) is captured here for later review.
    It is written alongside the normal transcript message but is never surfaced
    to the frontend.
    """
    user_id: str
    conversation_id: str
    role: str  # "mentor" or "mentee"
    content: str
    flag_type: str  # "crisis", "toxic", "both"
    suicide_score: float | None = None  # sentinet/suicidality probability (primary crisis signal)
    crisis_score: float | None = None  # emotion distress proxy: sadness + fear combined (secondary)
    toxic_scores: dict | None = None  # toxic-bert categories over threshold
    emotions: dict | None = None  # full emotion distribution at flag time
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))