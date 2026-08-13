"""Pytest bootstrap, loaded automatically before any test collection.

Two jobs, both of which must happen at import time (before test modules import
the app):
  1. Provide dummy values for required env vars so modules like agents.py and
     db.py — which read os.environ at import — can be imported without real
     credentials. setdefault means a real local .env still wins.
  2. Put the backend/ directory on sys.path so tests can do bare imports such as
     `from main import app` regardless of where pytest is invoked from.
  3. Stub chromadb so importing the app never touches a on-disk vector store.
"""

import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")

# Auth/Postgres stubs: the auth package (and, transitively, main) reads these at
# import. A dummy POSTGRES_URI is fine because create_async_engine doesn't open a
# connection until first use, and tests never trigger the startup hook.
os.environ.setdefault("POSTGRES_URI", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("JWT_ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "15")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "7")
os.environ.setdefault("GOOGLE_CLIENT_ID", "test")
os.environ.setdefault("GOOGLE_CLIENT_SECRET", "test")

sys.path.insert(0, os.path.dirname(__file__))

if "chromadb" not in sys.modules:
    _chroma = MagicMock()
    _collection = MagicMock()
    _client = MagicMock()
    _client.get_or_create_collection.return_value = _collection
    _chroma.PersistentClient.return_value = _client
    sys.modules["chromadb"] = _chroma
    # Nested imports in memory.py (chromadb.utils.embedding_functions) need
    # stub modules too — a bare MagicMock parent is not a real package.
    sys.modules["chromadb.utils"] = MagicMock()
    sys.modules["chromadb.utils.embedding_functions"] = MagicMock()
