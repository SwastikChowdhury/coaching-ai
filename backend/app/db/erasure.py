"""GDPR right-to-erasure orchestration.

Single source of truth for deleting everything we hold about a user, so the
user-facing (DELETE /auth/me) and admin (DELETE /admin/users/{id}) endpoints
behave identically. Spans all stores that hold personal data:

  - MongoDB:  messages, whispers, and flagged_messages (db.mongo)
  - ChromaDB: vector memories (memory.clear_memories)
  - Postgres: refresh tokens + the user account row (db.crud)

Best-effort by design: each store is wrapped independently so a failure in one
(e.g. a transient Mongo error) is reported in the result rather than aborting
the whole erasure and leaving the other stores untouched. The returned dict
carries per-store counts plus an `errors` list; callers decide how to surface a
partial failure.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.db import crud
from app.db.mongo import delete_user_chat_data
from app.memory.memory import clear_memories


async def erase_user(db: AsyncSession, user_id: str) -> dict:
    """Erase all data for `user_id` across Mongo, Chroma, and Postgres.

    Returns a summary dict:
        {
            "messages_deleted": int,
            "whispers_deleted": int,
            "flagged_deleted": int,
            "memories_deleted": int,
            "account_deleted": bool,
            "errors": list[str],   # per-store failures, empty on full success
        }

    Each store is attempted independently; a failure is appended to `errors` and
    does not prevent the remaining stores from being cleared.
    """
    summary: dict = {
        "messages_deleted": 0,
        "whispers_deleted": 0,
        "flagged_deleted": 0,
        "memories_deleted": 0,
        "account_deleted": False,
        "errors": [],
    }

    # MongoDB: transcript + whispers + flagged moderation records.
    try:
        mongo_counts = await delete_user_chat_data(user_id)
        summary.update(mongo_counts)
    except Exception as e:  # noqa: BLE001 - report, don't abort other stores
        summary["errors"].append(f"mongo: {e}")

    # ChromaDB: long-term vector memories (clear_memories already swallows its
    # own errors and returns 0, but guard anyway for symmetry).
    try:
        summary["memories_deleted"] = clear_memories(user_id)
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"chroma: {e}")

    # Postgres: refresh tokens + the account row.
    try:
        summary["account_deleted"] = await crud.delete_user(db, user_id)
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"postgres: {e}")

    return summary
