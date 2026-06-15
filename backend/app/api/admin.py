"""Operational/admin endpoints: live model registry, rollback, and data wipe.

The model registry/rollback endpoints remain unauthenticated for this local/demo
build. The destructive erasure endpoint, however, is guarded by a shared admin
key (X-Admin-Key header vs the ADMIN_API_KEY env var) so it can't be triggered
by anyone who simply knows a user id. Lock the remaining endpoints down too
before any real deployment.
"""

import os

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.erasure import erase_user
from app.db.postgres import get_db
from app.observability.metrics import model_rollbacks
from app.observability.model_registry import REGISTRY, rollback

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin_key(x_admin_key: str = Header(default="")) -> None:
    """Guard destructive admin actions behind a shared secret.

    Compares the X-Admin-Key request header against the ADMIN_API_KEY env var.
    Rejects with 401 when the key is unset (fail closed — a misconfigured server
    must not expose erasure) or when it doesn't match. Used as a FastAPI
    dependency so the check runs before the handler body.
    """
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected or x_admin_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


@router.get("/models")
def list_models():
    """Return the live model registry (current/previous model per agent).

    Response: the REGISTRY dict, used by ops to see what each agent is running.
    Unauthenticated — acceptable only because this is a local demo.
    """
    return REGISTRY


@router.post("/rollback/{agent}")
def rollback_model(agent: str):
    """Swap an agent back to its previous model at runtime (no redeploy).

    Path param `agent`: registry key, e.g. "conversation" or "whisper".
    Response: the new model state, or {"error": ...} if no previous model
    exists. The rollback metric is only incremented on a successful swap so
    failed attempts don't pollute the counter.
    """
    result = rollback(agent)
    if "error" not in result:
        model_rollbacks.labels(agent=agent).inc()
    return result


@router.delete("/users/{user_id}", dependencies=[Depends(require_admin_key)])
async def delete_user_account(user_id: str, db: AsyncSession = Depends(get_db)):
    """Admin GDPR erasure: wipe one user's entire footprint across all stores.

    For support/legal-driven deletion (vs the user-initiated DELETE /auth/me).
    Guarded by require_admin_key. Delegates to erasure.erase_user, which removes
    MongoDB messages/whispers/flagged records, ChromaDB memories, and the
    Postgres account + refresh tokens.

    Response: per-store deletion counts (plus any per-store errors). Side
    effects: irreversible deletes across MongoDB, the vector store, and Postgres.
    """
    return await erase_user(db, user_id)
