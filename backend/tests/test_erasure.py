"""Unit tests for db.erasure.erase_user.

Every store (Mongo, Chroma, Postgres) is monkeypatched so these tests stay
deterministic and touch no real database — they verify the orchestration:
all three stores are invoked, counts are aggregated, and a single store failure
is reported in `errors` without aborting the others.
"""

import asyncio

from app.db import erasure


def test_erase_user_invokes_all_stores(monkeypatch):
    """erase_user calls Mongo, Chroma, and Postgres and aggregates their counts."""
    calls = {}

    async def fake_mongo(user_id):
        calls["mongo"] = user_id
        return {"messages_deleted": 3, "whispers_deleted": 2, "flagged_deleted": 1}

    def fake_clear(user_id):
        calls["chroma"] = user_id
        return 4

    async def fake_delete_user(db, user_id):
        calls["postgres"] = user_id
        return True

    monkeypatch.setattr(erasure, "delete_user_chat_data", fake_mongo)
    monkeypatch.setattr(erasure, "clear_memories", fake_clear)
    monkeypatch.setattr(erasure.crud, "delete_user", fake_delete_user)

    result = asyncio.run(erasure.erase_user(db=object(), user_id="u1"))

    assert calls == {"mongo": "u1", "chroma": "u1", "postgres": "u1"}
    assert result["messages_deleted"] == 3
    assert result["whispers_deleted"] == 2
    assert result["flagged_deleted"] == 1
    assert result["memories_deleted"] == 4
    assert result["account_deleted"] is True
    assert result["errors"] == []


def test_erase_user_reports_store_failure_without_aborting(monkeypatch):
    """A Mongo failure is captured in errors but Chroma and Postgres still run."""
    ran = {"chroma": False, "postgres": False}

    async def boom_mongo(user_id):
        raise RuntimeError("mongo down")

    def fake_clear(user_id):
        ran["chroma"] = True
        return 0

    async def fake_delete_user(db, user_id):
        ran["postgres"] = True
        return True

    monkeypatch.setattr(erasure, "delete_user_chat_data", boom_mongo)
    monkeypatch.setattr(erasure, "clear_memories", fake_clear)
    monkeypatch.setattr(erasure.crud, "delete_user", fake_delete_user)

    result = asyncio.run(erasure.erase_user(db=object(), user_id="u1"))

    assert ran["chroma"] is True
    assert ran["postgres"] is True
    assert result["account_deleted"] is True
    assert any("mongo" in e for e in result["errors"])
