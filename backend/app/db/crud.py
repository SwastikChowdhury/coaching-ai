"""Async data-access helpers for the auth schema.

Thin wrappers over SQLAlchemy that the auth router composes into endpoints. Each
function owns its own commit for writes, and every DB operation is wrapped so a
driver/integrity error surfaces as an HTTP 500 rather than an unhandled
exception that would crash the request. Uniqueness conflicts (e.g. duplicate
email) are validated by the router BEFORE calling create_user, so here an
IntegrityError is treated as an unexpected server error.
"""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models_sql import RefreshToken, User


async def create_user(
    db: AsyncSession,
    email: str,
    hashed_password: str | None,
    first_name: str,
    last_name: str,
    dob,
    location: str | None,
    nationality: str | None,
    google_oauth_id: str | None = None,
    is_verified: bool = False,
) -> User:
    """Insert a new user and return it (with server-generated id/created_at).

    `hashed_password` may be None for Google-only accounts. The caller is
    responsible for hashing and for the duplicate-email pre-check.
    """
    try:
        user = User(
            email=email,
            hashed_password=hashed_password,
            first_name=first_name,
            last_name=last_name,
            dob=dob,
            location=location,
            nationality=nationality,
            google_oauth_id=google_oauth_id,
            is_verified=is_verified,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create user")


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    """Return the user with this email, or None."""
    try:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Failed to look up user")


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    """Return the user with this id (UUID as str), or None."""
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Failed to look up user")


async def get_user_by_google_id(db: AsyncSession, google_id: str) -> User | None:
    """Return the user linked to this Google subject id, or None."""
    try:
        result = await db.execute(
            select(User).where(User.google_oauth_id == google_id)
        )
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Failed to look up user")


async def create_refresh_token(
    db: AsyncSession, user_id: str, token: str, expires_at: datetime
) -> RefreshToken:
    """Persist a refresh token for a user and return the stored row."""
    try:
        refresh = RefreshToken(user_id=user_id, token=token, expires_at=expires_at)
        db.add(refresh)
        await db.commit()
        await db.refresh(refresh)
        return refresh
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to store refresh token")


async def get_refresh_token(db: AsyncSession, token: str) -> RefreshToken | None:
    """Return the stored RefreshToken matching this opaque value, or None."""
    try:
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token == token)
        )
        return result.scalar_one_or_none()
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Failed to look up refresh token")


async def delete_refresh_token(db: AsyncSession, token: str) -> None:
    """Delete a single refresh token (logout / rotation). No-op if absent."""
    try:
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.token == token)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await db.delete(row)
            await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete refresh token")


async def delete_all_refresh_tokens(db: AsyncSession, user_id: str) -> None:
    """Delete every refresh token for a user (revoke all sessions)."""
    try:
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.user_id == user_id)
        )
        for row in result.scalars().all():
            await db.delete(row)
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete refresh tokens")


async def delete_user(db: AsyncSession, user_id: str) -> bool:
    """Delete a user's account row; return True if a row existed.

    Backs GDPR-style erasure of the Postgres account. The user's refresh tokens
    are removed first as a belt-and-suspenders revoke (the FK is also
    ON DELETE CASCADE, so deleting the user alone would clear them too — but the
    explicit pass keeps the intent obvious and works regardless of cascade
    config). Returns False when there's no such user so the caller can report
    "nothing to delete" rather than treating it as an error.
    """
    try:
        result = await db.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if user is None:
            return False
        await delete_all_refresh_tokens(db, user_id)
        await db.delete(user)
        await db.commit()
        return True
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete user")
