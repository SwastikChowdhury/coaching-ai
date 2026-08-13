"""
MongoDB persistence layer for chat transcripts and coaching whispers.

Thin async data-access module: it owns the Mongo connection and the small set
of read/write helpers main.py uses to persist and replay a conversation. There
is intentionally no ORM and no business logic here — callers pass in validated
`models.Message` objects and get back plain dicts.

Two collections under the "muse" database:
  - messages  -> the mentor/mentee dialogue (role "user"/"assistant")
  - whispers  -> the private coaching notes (role "whisper")
They are kept separate so the conversation transcript and the coach's
side-channel can be queried and cleared independently.

Uses Motor (async MongoDB driver) so these calls don't block the event loop
that's also servicing the chat websocket.
"""

import os
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

from app.schemas.models import FlaggedMessage

load_dotenv()


def conversation_id_for(user_id: str) -> str:
    """Map a user to their single stable conversation id.

    The chat is one continuous thread per user, so deriving the conversation id
    deterministically from the user id keeps history/whispers replaying across
    reconnects (versus a fresh random id per socket, which would orphan history).
    Lives here because it's the key used for the MongoDB transcript queries.
    """
    return f"conv-{user_id}"

client = AsyncIOMotorClient(os.environ["MONGODB_URI"])
db = client["muse"]
messages_collection = db["messages"]
whispers_collection = db["whispers"]
# Separate observation-only collection for messages the moderation pipeline
# flagged. Kept apart from the transcript so flagged records can be reviewed and
# cleared independently of the conversation history.
flagged_collection = db["flagged_messages"]



async def save_message(message) -> None:
    """Insert one mentor/mentee message. `message` is a models.Message.

    model_dump() flattens the Pydantic model (including the auto `created_at`)
    into the BSON document Mongo stores. Side effect: one insert.
    """
    await messages_collection.insert_one(message.model_dump())


async def get_history(user_id: str, conversation_id: str) -> list[dict]:
    """Fetch a conversation's most recent message history, oldest first.

    Sorted descending to take the newest 1000, then reversed so agents and UI
    still see turns in chronological order. Capped at 1000 to avoid unbounded
    reads; revisit (windowing/pagination) before long-lived chats.
    """
    cursor = messages_collection.find(
        {"user_id": user_id, "conversation_id": conversation_id}
    ).sort("created_at", -1).limit(1000)
    docs = await cursor.to_list(length=1000)
    docs.reverse()
    return docs

async def save_whisper(message) -> None:
    """Insert one coaching whisper (stored separately from the transcript)."""
    await whispers_collection.insert_one(message.model_dump())


async def save_flagged(msg: FlaggedMessage) -> None:
    """Insert one moderation-flagged record into the flagged_messages collection.

    Observation-only: this never affects the transcript a user sees, it just
    captures the moderation signal (scores + flag type) for offline review.
    """
    await flagged_collection.insert_one(msg.model_dump())


async def get_whispers(user_id: str, conversation_id: str) -> list[dict]:
    """Fetch the most recent persisted whispers, oldest first.

    Mirrors get_history (newest 1000, then reversed to chronological) so the
    Muse side-panel rehydrates with recent coaching notes on websocket connect.
    """
    cursor = whispers_collection.find(
        {"user_id": user_id, "conversation_id": conversation_id}
    ).sort("created_at", -1).limit(1000)
    docs = await cursor.to_list(length=1000)
    docs.reverse()
    return docs


async def delete_user_chat_data(user_id: str) -> dict:
    """Delete every Mongo document tied to a user across all three collections.

    Backs GDPR-style erasure: removes the transcript (messages), the coaching
    side-channel (whispers), AND the observation-only moderation records
    (flagged_messages) — the last of which the earlier partial wipe missed,
    leaving message content + scores behind. Filtered by `user_id` alone (not the
    derived conversation_id) so nothing is left behind even if a record was ever
    written under a different conversation id.

    Returns per-collection deletion counts. Raises on a driver error so the
    caller (erasure.erase_user) can report Mongo failed rather than silently
    claiming success.
    """
    deleted_msgs = await messages_collection.delete_many({"user_id": user_id})
    deleted_whispers = await whispers_collection.delete_many({"user_id": user_id})
    deleted_flagged = await flagged_collection.delete_many({"user_id": user_id})
    return {
        "messages_deleted": deleted_msgs.deleted_count,
        "whispers_deleted": deleted_whispers.deleted_count,
        "flagged_deleted": deleted_flagged.deleted_count,
    }