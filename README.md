<div align="center">

# Coach AI

**A real-time communication practice partner with private, AI-driven coaching.**

Role-play a difficult conversation while an AI plays the other person — and a separate "Muse" agent privately coaches you on tone, subtext, and patterns as you go.

[![CI](https://github.com/SwastikChowdhury/muse-ai-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/SwastikChowdhury/muse-ai-lite/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)
![React](https://img.shields.io/badge/React-Vite-61dafb)
![Docker](https://img.shields.io/badge/Docker-ready-2496ed)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Overview

Muse-lite is a multi-agent conversational system built around a split-screen experience: the practice conversation on the left, and a private coaching channel ("Muse") on the right. You play a **mentor** giving feedback; the AI plays **Alex**, a realistic **mentee**; and the Muse agent reads the exchange in real time, surfacing one labeled coaching note per turn — on tone, pacing, subtext, or recurring patterns it remembers from past sessions.

The system streams token-by-token over WebSockets, remembers your communication habits across sessions via vector retrieval, and is fully instrumented for cost, latency, and safety.

> ![Muse-lite UI](docs/screenshot.png)

## Features

- **Real-time streaming chat** — token-by-token replies over a WebSocket, with live client-side state.
- **Multi-agent orchestration** — an orchestrator coordinates a Conversation Agent (the mentee), a Muse Whisper Agent (private coaching), and a Memory Agent — each independently swappable.
- **Long-term memory (RAG)** — communication patterns are embedded in ChromaDB and retrieved with recency-aware reranking; coaching notes that cite past patterns are verified for grounding to mitigate hallucination.
- **Self-classifying coaching** — the Muse agent tags each note (Tone, Pattern, Subtext, Opening, Suggestion, Pacing, Clarity, Empathy, Boundary).
- **Voice in and out** — speech-to-text via Groq Whisper, replies read aloud via the browser Speech Synthesis API.
- **Model registry + live rollback** — a single source of truth for each agent's model, with tiered model selection and a runtime rollback endpoint.
- **Safety & privacy** — a crisis-escalation filter that bypasses the model entirely, plus PII redaction applied at intake (before the LLM, the database, or the vector store).
- **Evaluation harness** — offline test cases (safety, privacy, grounding) in CI, plus a live eval runner that scores empathy and recall through the real pipeline.
- **Full observability** — Prometheus + Grafana dashboards tracking tokens, estimated cost, per-agent latency, quota pressure, grounding, and safety escalations.
- **Authentication** — email/password registration, JWT access tokens, refresh tokens in Postgres, and optional Google OAuth. Chat WebSocket requires a valid access token; each user's history, whispers, and vector memories are scoped by `user_id`.
- **Containerized & CI-tested** — one Docker image serves the whole app; GitHub Actions lints, tests, smoke-tests the container, and publishes to GHCR.

## Architecture
> ![Architecture](docs/architecture.png)


### Request lifecycle (one turn)

1. **Intake gate** — the message is PII-redacted and run through the safety filter. A crisis message returns an escalation response immediately and never reaches a model.
2. **Memory retrieve** — the Memory Agent pulls the mentor's relevant past patterns from ChromaDB (similarity × recency).
3. **Conversation Agent** — Gemini replies in character as the mentee, streamed token-by-token to the chat panel.
4. **Muse Whisper Agent** — a second model call produces one labeled coaching note; any citation of a past pattern is verified against what was actually retrieved.
5. **Persist & remember** — the turn and the whisper are written to MongoDB (separate collections), and the mentor's message is added to vector memory.

Conversation turns and coaching notes are stored in **separate MongoDB collections** so private coaching can never leak back into the model's conversation context.

## Tech stack

| Layer | Technology |
| --- | --- |
| Frontend | React + Vite (auth gate + chat; built and served by FastAPI in production) |
| Backend | FastAPI + Uvicorn, async WebSockets |
| Auth | Postgres (SQLAlchemy async) · JWT access tokens · refresh tokens · Google OAuth |
| LLM | Google Gemini, tiered per agent (registry-driven) |
| Persistence | MongoDB Atlas (transcripts, whispers) · Postgres (users, sessions) |
| Vector memory | ChromaDB (local, persistent; scoped per `user_id`) |
| Voice | Groq Whisper (ASR) · browser SpeechSynthesis (TTS) |
| Observability | Prometheus + Grafana |
| Packaging / CI | Docker · GitHub Actions · GHCR |

## Getting started

### Prerequisites

- Docker & Docker Compose (recommended — includes Postgres), or Python 3.10 + Node 20 for local dev
- MongoDB Atlas connection string, Google Gemini API key, and Groq API key
- For Google sign-in: OAuth client credentials and callback URI `http://localhost:8000/auth/google/callback`

### Configuration

Copy the example env file and fill in your values (never commit `backend/.env`):

```bash
cp backend/.env.example backend/.env
```

| Variable | Description |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key (billing-enabled recommended) |
| `MONGODB_URI` | MongoDB Atlas connection string (chat data) |
| `GROQ_API_KEY` | Groq API key (Whisper speech-to-text) |
| `POSTGRES_URI` | Postgres async URL for auth (`postgresql+asyncpg://user:pass@host:5432/db`) |
| `JWT_SECRET` | Long random string for signing access tokens (required in production) |
| `JWT_ALGORITHM` | JWT algorithm (default `HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime (default `15`) |
| `REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime (default `7`) |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID (optional) |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret (optional) |
| `FRONTEND_URL` | OAuth redirect target (`http://localhost:5173` for Vite dev, `http://localhost:8000` under Docker) |
| `PII_SALT` | Salt for PII tokenization (set a strong value in production) |
| `SUICIDE_THRESHOLD` | Suicide-model escalation threshold |
| `CRISIS_THRESHOLD` | Crisis emotion proxy threshold |
| `TOXIC_THRESHOLD` | Toxicity flag threshold |

Docker Compose starts Postgres automatically and overrides `POSTGRES_URI` and `FRONTEND_URL` for the app container. See `backend/.env.example` for the full list.

### Run the full stack (recommended)

Brings up the app, Postgres, Prometheus, and Grafana together:

```bash
docker compose up --build
# app        -> http://localhost:8000   (SPA + API; sign in / register first)
# Postgres   -> localhost:5432          (user muse / password muse / db muse)
# Prometheus -> http://localhost:9090
# Grafana    -> http://localhost:3000   (anonymous access; Muse dashboard provisioned)
```

### Run for local development

Start Postgres (e.g. `docker compose up postgres` or local install), then:

Backend (from `backend/`):

```bash
cp .env.example .env   # if you haven't already
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend (from `frontend/`):

```bash
npm install
npm run dev        # Vite dev server on :5173 (set FRONTEND_URL=http://localhost:5173 in .env)
```

The frontend calls `http://localhost:8000` for API/auth; chat WebSocket connects with `?token=<access_token>`.

## Project structure

```
muse-ai-lite/
├── backend/
│   ├── app/                      # application package (uvicorn app.main:app)
│   │   ├── main.py               # FastAPI app: middleware, lifespan, router includes, /health
│   │   ├── api/                  # transport/routers only
│   │   │   ├── auth.py           # /auth/* (register, login, refresh, logout, google, me)
│   │   │   ├── chat.py           # /ws + /transcribe
│   │   │   └── admin.py          # /admin/* (models, rollback, clear-data)
│   │   ├── agents/               # agents.py, orchestrator.py, grounding.py
│   │   ├── safety/               # safety.py, privacy.py, moderation.py
│   │   ├── db/                   # mongo.py (transcript) + postgres.py/models_sql.py/crud.py (auth)
│   │   ├── memory/               # ChromaDB vector memory (rerank, clear)
│   │   ├── auth/                 # password hashing, JWT/refresh tokens, Google OAuth
│   │   ├── observability/        # metrics.py, llm_metrics.py, model_registry.py
│   │   └── schemas/              # Pydantic models
│   ├── conftest.py               # pytest path / env setup
│   ├── .env.example              # env var template (copy to .env)
│   ├── requirements.txt
│   ├── tests/                    # pytest suite (runs in CI)
│   └── evals/                    # live evaluation harness
├── frontend/                 # React + Vite client
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/provisioning/
│       ├── dashboards/       # dashboard.yml + muse-dashboard.json
│       └── datasources/      # datasource.yml
├── .github/workflows/ci.yml  # lint · test · build · smoke-test · publish
├── Dockerfile                # multi-stage: build frontend, serve from FastAPI
├── docker-compose.yml        # app + postgres + Prometheus + Grafana
├── .dockerignore
├── LICENSE
└── README.md
```

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |
| `WS` | `/ws?token=<access_token>` | Streaming chat (history / token / done / whisper); requires a valid access token |
| `POST` | `/transcribe` | Audio → text via Groq Whisper |
| `GET` | `/admin/models` | Current model registry |
| `POST` | `/admin/rollback/{agent}` | Roll an agent back to its previous model |
| `DELETE` | `/admin/clear-data/{user_id}` | Wipe a user's conversation, whispers, and memories |
| `POST` | `/auth/register` | Create an email/password account, returns token pair |
| `POST` | `/auth/login` | Authenticate, returns token pair |
| `POST` | `/auth/refresh` | Rotate refresh token, returns new token pair |
| `POST` | `/auth/logout` | Revoke a refresh token |
| `GET` | `/auth/google` | Begin Google OAuth flow |
| `GET` | `/auth/google/callback` | Google OAuth callback, returns token pair |
| `GET` | `/auth/me` | Current user profile (bearer token) |

## Testing

The suite runs without external services (LLM, DB, and ASR calls are mocked) and executes in CI on every push:

```bash
cd backend
pytest tests -q
```

It covers auth-aware WebSocket wiring, data models, the orchestrator's resilience paths, grounding verification, and safety/PII filters. A separate live harness exercises the real agents:

```bash
python -m evals.run_eval     # makes real Gemini calls
```

### Manual smoke test (auth + chat)

1. `curl http://localhost:8000/health` → `{"status":"ok"}`
2. Open the app → auth screen (Sign In / Create Account)
3. Register or log in → chat shows your first name as Mentor
4. Send a message → mentee reply + Muse whisper
5. Refresh → still logged in; history reloads
6. Log out → auth screen; log in again → same conversation history

Optional: query Postgres users — `docker compose exec postgres psql -U muse -d muse -c "SELECT email, first_name FROM users;"` (run from repo root).

## Observability

The backend exposes Prometheus metrics at `/metrics`, including app-specific series:

- `muse_llm_tokens_total{agent, kind}` and `muse_llm_cost_usd_total{agent}` — token usage and estimated spend per agent
- `muse_agent_latency_seconds{agent}` — per-agent latency (motivates model tiering)
- `muse_gemini_calls_total{agent, outcome}` — throughput and quota/error pressure
- `muse_whisper_grounding_total{status}` — grounded vs ungrounded coaching notes
- `muse_safety_escalations_total` — messages caught by the safety filter
- `muse_active_websocket_connections` — live chat sockets
- `http_requests_total`, `http_request_duration_seconds`, `http_requests_inprogress` — FastAPI HTTP instrumentation

Grafana ships with a provisioned **Developer Observability** dashboard (chat activity, LLM spend, agent latency, grounding, mentor-focused safety), brought up automatically by `docker compose`. Each panel has a **Measures / Helps you** description — hover the **ⓘ** icon on any panel title.

### Persistence

Metrics history is stored in Docker named volumes:

| Volume | Service | Retention |
|--------|---------|-----------|
| `prometheus_data` | Prometheus TSDB | 30 days (configured in `docker-compose.yml`) |
| `grafana_data` | Grafana settings & UI edits | until removed |

- **`docker compose down` then `up`** — data is kept (volumes are not removed).
- **`docker compose down -v`** — **deletes all volumes**, including metrics history. Avoid `-v` unless you intentionally want a clean slate.
- **App container restarts** reset in-process counters, but Prometheus retains scraped history. The dashboard uses `rate()` / `increase()` so time-series stay meaningful across restarts.

After changing provisioned dashboard files, Grafana reloads them every 30 seconds, or restart the Grafana container to pick them up immediately.

### Alert rules

Prometheus evaluates a small set of dev-friendly rules in `monitoring/rules/muse-alerts.yml`:

| Alert | Fires when |
|-------|------------|
| `MuseAppDown` | App scrape target unreachable for 1m |
| `HighHTTP5xxRate` | 5xx rate &gt; 0.05 req/s for 2m |
| `HighHTTPLatencyP95` | HTTP p95 &gt; 5s for 5m |
| `GeminiCallErrors` | Any non-ok Gemini calls for 2m |
| `HighAgentLatencyP95` | Agent p95 &gt; 30s for 5m |
| `SafetyEscalationsDetected` | Any safety blocks in the last 10m |
| `HighLLMCostRate` | Estimated spend &gt; $1/hr for 10m |

View firing and pending alerts at **http://localhost:9090/alerts** after `docker compose up`. No Alertmanager is configured yet — alerts are visible in Prometheus (and can be wired to Slack/email later).

## Security notes

- **Never commit `backend/.env`** — it is gitignored; use `backend/.env.example` as the template.
- **JWT_SECRET** must be a strong random value in any shared or production deployment.
- **Multi-tenant data** — Mongo and Chroma use shared collections filtered by `user_id`; Postgres holds per-user auth rows.
- **`/admin/*`** is unauthenticated in this demo build; lock it down before production.
- **Google OAuth** redirects to the frontend with tokens in the query string briefly; tokens are then stored in `localStorage`.

## CI/CD

GitHub Actions runs on every push and PR to `main`: **ruff** lint → **pytest** → Docker **build** → container **smoke test** (`/health`) → **publish** a `latest` and commit-`SHA`-tagged image to the GitHub Container Registry.

**Continuous deployment (planned):** Google **Cloud Run** is the target — it runs the container and natively supports the persistent WebSocket that function-based serverless platforms cannot. Images push to **Artifact Registry**, deploys authenticate via **Workload Identity Federation**, and ship on a green CI run. SHA-tagged images make rollbacks traceable.

## Roadmap / how this scales

- **Privacy** — move generation to **Vertex AI** (no training on inputs) behind the same interface.
- **Orchestration** — adopt **Google ADK** or **LangGraph** for typed state, retries, and tracing.
- **Vector store** — graduate from ChromaDB to a managed store (**FAISS/Milvus**, or **MongoDB Atlas Vector Search** to unify the stack).
- **Context cache** — **Redis / Memorystore** for hot conversation context.
- **Multi-user sessions** — per-user conversations and optional multiple chats per user (auth + Postgres are in place).
- **Evaluation** — replace keyword scoring with **LLM-as-judge** rubrics and online A/B testing.
- **Deployment** — activate the Cloud Run CD pipeline for a live URL on every merge.

## License

Released under the [MIT License](LICENSE).
