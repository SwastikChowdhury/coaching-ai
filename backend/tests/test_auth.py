"""Auth endpoint tests with the Postgres CRUD layer stubbed in memory.

Exercises the /auth router wiring (validation, token minting, rotation) without
a real database or production code changes.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from app.auth.jwt import verify_access_token
from app.db import crud, erasure
from app.db.postgres import get_db
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture
def register_payload():
    return {
        "email": "mentor@example.com",
        "password": "secure-pass-123",
        "first_name": "Alex",
        "last_name": "Mentor",
        "dob": "1990-05-15",
        "location": "NYC",
        "nationality": "US",
    }


@pytest.fixture
def auth_store(monkeypatch):
    """In-memory stand-in for users and refresh tokens."""
    users_by_email = {}
    users_by_id = {}
    refresh_tokens = {}

    async def fake_get_user_by_email(db, email):
        return users_by_email.get(email)

    async def fake_get_user_by_id(db, user_id):
        return users_by_id.get(user_id)

    async def fake_create_user(
        db,
        email,
        hashed_password,
        first_name,
        last_name,
        dob,
        location,
        nationality,
        google_oauth_id=None,
        is_verified=False,
    ):
        user = SimpleNamespace(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            dob=dob,
            location=location,
            nationality=nationality,
            google_oauth_id=google_oauth_id,
            is_verified=is_verified,
            created_at=datetime.now(timezone.utc),
        )
        users_by_email[email] = user
        users_by_id[user.id] = user
        return user

    async def fake_create_refresh_token(db, user_id, token, expires_at):
        row = SimpleNamespace(user_id=user_id, token=token, expires_at=expires_at)
        refresh_tokens[token] = row
        return row

    async def fake_get_refresh_token(db, token):
        return refresh_tokens.get(token)

    async def fake_delete_refresh_token(db, token):
        refresh_tokens.pop(token, None)

    async def fake_delete_user(db, user_id):
        user = users_by_id.pop(user_id, None)
        if user is None:
            return False
        users_by_email.pop(user.email, None)
        for tok in [t for t, r in refresh_tokens.items() if r.user_id == user_id]:
            refresh_tokens.pop(tok, None)
        return True

    # erase_user calls the Mongo helper directly; stub it so no real DB is hit.
    async def fake_delete_user_chat_data(user_id):
        return {"messages_deleted": 0, "whispers_deleted": 0, "flagged_deleted": 0}

    monkeypatch.setattr(crud, "get_user_by_email", fake_get_user_by_email)
    monkeypatch.setattr(crud, "get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr(crud, "create_user", fake_create_user)
    monkeypatch.setattr(crud, "create_refresh_token", fake_create_refresh_token)
    monkeypatch.setattr(crud, "get_refresh_token", fake_get_refresh_token)
    monkeypatch.setattr(crud, "delete_refresh_token", fake_delete_refresh_token)
    monkeypatch.setattr(crud, "delete_user", fake_delete_user)
    monkeypatch.setattr(erasure, "delete_user_chat_data", fake_delete_user_chat_data)
    monkeypatch.setattr(erasure, "clear_memories", lambda user_id: 0)

    return refresh_tokens


@pytest.fixture
def client(auth_store, monkeypatch):
    async def override_get_db():
        yield AsyncMock()

    monkeypatch.setattr(app.router, "on_startup", [])
    monkeypatch.setattr(app.router, "on_shutdown", [])

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_register_and_login(client, register_payload):
    """Register mints tokens; login with the same credentials returns a new pair."""
    reg = client.post("/auth/register", json=register_payload)
    assert reg.status_code == 200
    reg_body = reg.json()
    assert reg_body["token_type"] == "bearer"
    assert reg_body["user"]["email"] == register_payload["email"]
    assert "hashed_password" not in reg_body["user"]
    assert verify_access_token(reg_body["access_token"]) == reg_body["user"]["id"]

    login = client.post(
        "/auth/login",
        json={"email": register_payload["email"], "password": register_payload["password"]},
    )
    assert login.status_code == 200
    assert login.json()["user"]["id"] == reg_body["user"]["id"]


def test_register_duplicate_email(client, register_payload):
    """A second register with the same email is rejected."""
    assert client.post("/auth/register", json=register_payload).status_code == 200
    dup = client.post("/auth/register", json=register_payload)
    assert dup.status_code == 400
    assert dup.json()["detail"] == "Email already registered"


def test_login_invalid_credentials(client, register_payload):
    """Wrong password yields 401 without revealing whether the email exists."""
    client.post("/auth/register", json=register_payload)
    bad = client.post(
        "/auth/login",
        json={"email": register_payload["email"], "password": "wrong-password"},
    )
    assert bad.status_code == 401
    assert bad.json()["detail"] == "Invalid credentials"


def test_refresh_rotates_token(client, register_payload):
    """Presenting a valid refresh token returns a new pair and invalidates the old one."""
    tokens = client.post("/auth/register", json=register_payload).json()
    old_refresh = tokens["refresh_token"]

    rotated = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert rotated.status_code == 200
    new_body = rotated.json()
    assert new_body["refresh_token"] != old_refresh
    assert verify_access_token(new_body["access_token"]) == tokens["user"]["id"]

    stale = client.post("/auth/refresh", json={"refresh_token": old_refresh})
    assert stale.status_code == 401


def test_logout_revokes_refresh_token(client, register_payload):
    """Logout deletes the refresh token so it cannot be rotated again."""
    tokens = client.post("/auth/register", json=register_payload).json()
    refresh = tokens["refresh_token"]

    out = client.post("/auth/logout", json={"refresh_token": refresh})
    assert out.status_code == 200
    assert out.json()["message"] == "logged out"

    again = client.post("/auth/refresh", json={"refresh_token": refresh})
    assert again.status_code == 401


def test_me_returns_profile(client, register_payload):
    """Bearer access token on /auth/me returns the safe profile fields plus age."""
    tokens = client.post("/auth/register", json=register_payload).json()
    me = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {tokens['access_token']}"},
    )
    assert me.status_code == 200
    body = me.json()
    assert body["email"] == register_payload["email"]
    assert body["first_name"] == register_payload["first_name"]
    assert body["dob"] == register_payload["dob"]
    assert isinstance(body["age"], int)
    assert "hashed_password" not in body


def test_me_rejects_invalid_token(client):
    """Missing or bad bearer credentials are rejected before hitting the DB."""
    missing = client.get("/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert missing.status_code == 401

    no_bearer = client.get("/auth/me", headers={"Authorization": "Token abc"})
    assert no_bearer.status_code == 401


def test_delete_me_erases_account(client, register_payload):
    """DELETE /auth/me with the right password erases the account; /me then 401s."""
    tokens = client.post("/auth/register", json=register_payload).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    deleted = client.request(
        "DELETE", "/auth/me", headers=headers,
        json={"password": register_payload["password"]},
    )
    assert deleted.status_code == 200
    assert deleted.json()["account_deleted"] is True

    # The account is gone, so the (still-unexpired) token no longer resolves.
    after = client.get("/auth/me", headers=headers)
    assert after.status_code == 401


def test_delete_me_requires_correct_password(client, register_payload):
    """A password-based account cannot be deleted with a wrong/missing password."""
    tokens = client.post("/auth/register", json=register_payload).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}

    wrong = client.request(
        "DELETE", "/auth/me", headers=headers, json={"password": "not-it"},
    )
    assert wrong.status_code == 401

    # The account still exists after the failed attempt.
    still = client.get("/auth/me", headers=headers)
    assert still.status_code == 200


def test_delete_me_rejects_invalid_token(client):
    """Erasure requires a valid bearer token, like the rest of /auth/me."""
    bad = client.request(
        "DELETE", "/auth/me",
        headers={"Authorization": "Bearer not-a-jwt"},
        json={"password": "x"},
    )
    assert bad.status_code == 401
