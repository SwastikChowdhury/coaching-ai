"""HTTP surface for authentication: /auth/*.

Wires the pieces together — password hashing, JWT/refresh minting, Google OAuth,
and the Postgres CRUD layer — into the endpoints the frontend calls. Responses
NEVER include `hashed_password`; user payloads are built explicitly field by
field.

Session model:
  - On register/login/oauth we mint an access token (short-lived JWT) and a
    refresh token (opaque, stored in Postgres) and return both.
  - /auth/refresh rotates: it deletes the presented refresh token and issues a
    fresh pair, so a leaked refresh token has a bounded blast radius.
  - /auth/logout deletes the presented refresh token.
"""

from datetime import date, datetime
from urllib.parse import urlencode

from dateutil.relativedelta import relativedelta
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.jwt import (
    create_access_token,
    create_refresh_token,
    get_refresh_token_expiry,
    verify_access_token,
)
from app.auth.oauth import FRONTEND_URL, exchange_code_for_user, get_google_auth_url
from app.auth.password import hash_password, verify_password
from app.db import crud
from app.db.erasure import erase_user
from app.db.postgres import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


# ---- Request bodies -------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str
    first_name: str
    last_name: str
    dob: date  # parsed from a "YYYY-MM-DD" string
    location: str | None = None
    nationality: str | None = None


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class DeleteAccountRequest(BaseModel):
    # Required for email/password accounts (re-auth before an irreversible
    # erasure); ignored for Google-only accounts that have no password to verify.
    password: str | None = None


# ---- Helpers --------------------------------------------------------------

def _user_summary(user) -> dict:
    """The minimal, safe user object returned alongside tokens (no secrets)."""
    return {
        "id": str(user.id),
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
    }


async def _issue_tokens(db: AsyncSession, user_id: str) -> tuple[str, str]:
    """Mint an access token and a stored refresh token for `user_id`."""
    access_token = create_access_token(user_id)
    refresh_token = create_refresh_token()
    await crud.create_refresh_token(
        db, user_id=user_id, token=refresh_token, expires_at=get_refresh_token_expiry()
    )
    return access_token, refresh_token


# ---- Endpoints ------------------------------------------------------------

@router.post("/register")
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Create an email/password account and return a fresh token pair."""
    existing = await crud.get_user_by_email(db, body.email)
    if existing is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await crud.create_user(
        db,
        email=body.email,
        hashed_password=hash_password(body.password),
        first_name=body.first_name,
        last_name=body.last_name,
        dob=body.dob,
        location=body.location,
        nationality=body.nationality,
    )
    access_token, refresh_token = await _issue_tokens(db, str(user.id))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _user_summary(user),
    }


@router.post("/login")
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Verify email/password and return a fresh token pair."""
    user = await crud.get_user_by_email(db, body.email)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Google-only accounts have no password to verify against.
    if not user.hashed_password or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token, refresh_token = await _issue_tokens(db, str(user.id))
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": _user_summary(user),
    }


@router.post("/refresh")
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Rotate a refresh token: validate, delete the old, issue a new pair."""
    stored = await crud.get_refresh_token(db, body.refresh_token)
    if stored is None:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    # Compare against an aware "now". expires_at is stored timezone-aware.
    if stored.expires_at < datetime.now(stored.expires_at.tzinfo):
        await crud.delete_refresh_token(db, body.refresh_token)
        raise HTTPException(status_code=401, detail="Refresh token expired")

    user_id = str(stored.user_id)
    # Rotation: invalidate the presented token before minting a replacement.
    await crud.delete_refresh_token(db, body.refresh_token)
    access_token, new_refresh = await _issue_tokens(db, user_id)
    return {
        "access_token": access_token,
        "refresh_token": new_refresh,
        "token_type": "bearer",
    }


@router.post("/logout")
async def logout(body: LogoutRequest, db: AsyncSession = Depends(get_db)):
    """Revoke a single session by deleting its refresh token."""
    await crud.delete_refresh_token(db, body.refresh_token)
    return {"message": "logged out"}


@router.get("/google")
async def google_login():
    """Kick off the OAuth flow by redirecting the browser to Google."""
    return RedirectResponse(get_google_auth_url())


@router.get("/google/callback")
async def google_callback(code: str, db: AsyncSession = Depends(get_db)):
    """Handle Google's redirect: find-or-create the user and issue tokens."""
    info = await exchange_code_for_user(code)
    google_id = info.get("id")
    email = info.get("email")
    if not google_id or not email:
        raise HTTPException(status_code=400, detail="Incomplete Google profile")

    user = await crud.get_user_by_google_id(db, google_id)
    if user is None:
        # Same email may already exist from email/password signup. Link Google
        # instead of inserting (email is UNIQUE — a second row would 500).
        user = await crud.get_user_by_email(db, email)
        if user is not None:
            if user.google_oauth_id and user.google_oauth_id != google_id:
                raise HTTPException(
                    status_code=409,
                    detail="Email already linked to a different Google account",
                )
            user = await crud.link_google_oauth(db, user, google_id)
        else:
            # First Google sign-in: provision an account. dob is unknown, so we
            # use a placeholder the user can correct later; Google already
            # verified the email, so is_verified=True.
            user = await crud.create_user(
                db,
                email=email,
                hashed_password=None,
                first_name=info.get("given_name") or "",
                last_name=info.get("family_name") or "",
                dob=date(1900, 1, 1),
                location=None,
                nationality=None,
                google_oauth_id=google_id,
                is_verified=True,
            )

    access_token, refresh_token = await _issue_tokens(db, str(user.id))
    params = urlencode({"access_token": access_token, "refresh_token": refresh_token})
    return RedirectResponse(f"{FRONTEND_URL}/?{params}")


@router.get("/me")
async def me(
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    """Return the authenticated user's profile (no secrets) plus computed age.

    Expects `Authorization: Bearer <access_token>`. Age is derived from dob with
    relativedelta so it stays correct without storing a denormalized value.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]

    user_id = verify_access_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    age = relativedelta(date.today(), user.dob).years  # noqa: DTZ011 — local calendar age
    return {
        "id": str(user.id),
        "email": user.email,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "dob": user.dob.isoformat(),
        "location": user.location,
        "nationality": user.nationality,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "age": age,
    }


@router.delete("/me")
async def delete_me(
    body: DeleteAccountRequest | None = None,
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
):
    """GDPR right-to-erasure: delete the authenticated user's entire footprint.

    Identity comes from the bearer token (never a URL param) so a caller can only
    ever erase themselves. For email/password accounts the current password must
    be re-supplied and verified before the irreversible delete; Google-only
    accounts (no `hashed_password`) are erased on a valid token alone.

    Removes data across every store via erasure.erase_user: MongoDB messages,
    whispers, and flagged records; ChromaDB memories; and the Postgres account +
    refresh tokens. Returns per-store deletion counts.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]

    user_id = verify_access_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = await crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    # Re-authenticate password-based accounts before an irreversible action.
    if user.hashed_password:
        password = body.password if body else None
        if not password or not verify_password(password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")

    return await erase_user(db, user_id)
