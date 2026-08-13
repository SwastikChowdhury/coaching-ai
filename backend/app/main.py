"""FastAPI entrypoint for muse-ai-lite.

Assembles the application and nothing else: middleware, Prometheus
instrumentation, the Postgres-table bootstrap, the liveness probe, the router
includes, and (in production) the built-frontend mount. All real behavior lives
in the routers and the packages they delegate to:

  - app.api.auth   -> /auth/*  (register/login/refresh/logout/google/me)
  - app.api.chat   -> /ws + /transcribe (the chat transport)
  - app.api.admin  -> /admin/* (model registry, rollback, data wipe)

Run with:  uvicorn app.main:app  (from the backend/ directory).
"""

import os

from dotenv import load_dotenv

# Must run before app.* imports: oauth/jwt modules read credentials at import time.
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router
from app.db.postgres import Base, engine

app = FastAPI(title="muse-ai-lite")

# CORS is fully open because the frontend dev server and the API run on
# different origins/ports. Lock this down before any production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-instrument every route and expose Prometheus metrics at GET /metrics.
# inprogress gauge is off by default in the library; the Grafana panel queries it.
Instrumentator(should_instrument_requests_inprogress=True).instrument(app).expose(app)

# Compose the transport surface. Auth + chat + admin each live in their own
# router module under app.api.
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)


@app.on_event("startup")
async def startup():
    """Create the Postgres auth tables if they don't yet exist.

    Lightweight bootstrap for dev/single-node so the schema is ready on first
    run without a separate migration step (Alembic is available for real
    migrations later). This is the only place the app connects to Postgres at
    boot, so a missing/unreachable database surfaces clearly here.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/health")
def health():
    """Liveness probe for Docker/CI/load balancers.

    Response: 200 with {"status": "ok"}. No auth, no side effects.
    """
    return {"status": "ok"}


# In production the Vite build is copied to ./app/static and served by this same
# app, so the API and SPA share one origin. In local dev the frontend runs on
# its own Vite server, the directory won't exist, and this mount is skipped.
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="frontend")
