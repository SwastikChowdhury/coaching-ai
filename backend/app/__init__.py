"""muse-ai-lite backend application package.

Organized by concern:
  - api/           HTTP/WebSocket routers (transport only)
  - agents/        the Gemini agents + per-turn orchestration + grounding
  - safety/        crisis filter, PII redaction, moderation
  - db/            MongoDB transcript store + Postgres auth store
  - memory/        Chroma vector memory
  - auth/          password hashing, JWT/refresh tokens, Google OAuth
  - observability/ Prometheus metrics, LLM token metrics, model registry
  - schemas/       Pydantic models

The FastAPI app is assembled in app.main.
"""
