"""Tests for LLM token accounting.

Prometheus counters are process-global, so each test uses a unique agent label
and asserts on the increment rather than an absolute process total.
"""

from types import SimpleNamespace
from uuid import uuid4

from app.observability.llm_metrics import has_usage, record_usage
from prometheus_client import REGISTRY


def _agent() -> str:
    return f"test-{uuid4().hex[:10]}"


def _sample(metric: str, labels: dict) -> float:
    value = REGISTRY.get_sample_value(metric, labels)
    return value if value is not None else 0.0


def test_has_usage_rejects_empty_metadata():
    """Empty/zero usage_metadata must not count as a real report (stream chunks)."""
    assert has_usage(None) is False
    assert has_usage(SimpleNamespace()) is False
    assert has_usage(SimpleNamespace(prompt_token_count=0, candidates_token_count=None)) is False
    assert has_usage(SimpleNamespace(prompt_token_count=12)) is True


def test_record_usage_noop_on_empty():
    """No series should be created for a stream chunk with no billed tokens."""
    agent = _agent()
    assert record_usage(agent, None) is None
    assert record_usage(agent, SimpleNamespace(prompt_token_count=0)) is None
    assert _sample("muse_llm_tokens_total", {"agent": agent, "kind": "prompt"}) == 0.0


def test_record_usage_counts_prompt_and_completion():
    """Prompt and completion tokens are recorded separately from usage_metadata."""
    agent = _agent()
    usage = SimpleNamespace(
        prompt_token_count=1_000,
        candidates_token_count=250,
        thoughts_token_count=0,
    )
    recorded = record_usage(agent, usage, model="gemini-3.5-flash")
    assert recorded["prompt_tokens"] == 1_000
    assert recorded["completion_tokens"] == 250
    assert _sample("muse_llm_tokens_total", {"agent": agent, "kind": "prompt"}) == 1_000
    assert _sample("muse_llm_tokens_total", {"agent": agent, "kind": "completion"}) == 250


def test_record_usage_folds_thoughts_into_completion():
    """Thinking tokens are billed as output, so they add to completion."""
    agent = _agent()
    usage = SimpleNamespace(
        prompt_token_count=0,
        candidates_token_count=100,
        thoughts_token_count=400,
    )
    recorded = record_usage(agent, usage, model="gemini-3.1-flash-lite")
    assert recorded["completion_tokens"] == 500
    assert _sample("muse_llm_tokens_total", {"agent": agent, "kind": "completion"}) == 500


def test_record_usage_response_token_count_fallback():
    """Some SDK shapes expose response_token_count instead of candidates_token_count."""
    agent = _agent()
    usage = SimpleNamespace(
        prompt_token_count=0,
        response_token_count=80,
        thoughts_token_count=0,
    )
    recorded = record_usage(agent, usage, model="gemini-3.1-flash-lite")
    assert recorded["completion_tokens"] == 80
