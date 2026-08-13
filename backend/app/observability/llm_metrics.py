"""
LLM token-usage and cost accounting.

Separate from metrics.py because this concerns spend rather than service health:
it turns each Gemini response's usage_metadata into Prometheus counters for
tokens consumed and estimated USD cost, broken out per agent. Lets us watch the
relative cost of the conversation vs. whisper agents and catch runaway spend.

Cost is an *estimate*: token counts are exact (from the API), but USD is computed
from Google's published per-1M-token rates below. The model id comes from
model_registry.get_model() at record time (including after live rollbacks),
or from an explicit `model=` override (grounding LLM judge, which is not in
the registry). Add a PRICES entry whenever REGISTRY adopts a new model id.
record_usage is called by agents.py (whisper), orchestrator.py (conversation,
after the stream), and grounding.py (LLM judge fallback).
"""

from prometheus_client import Counter

from app.observability.model_registry import get_model

llm_tokens = Counter(
    "muse_llm_tokens_total",
    "LLM tokens consumed, by agent and kind",
    ["agent", "kind"],          # kind: prompt | completion
)

llm_cost = Counter(
    "muse_llm_cost_usd_total",
    "Estimated LLM spend in USD, by agent",
    ["agent"],
)

# USD per token (input, output). Paid tier, standard — ai.google.dev/gemini-api/docs/pricing
PRICES = {
    # Active in REGISTRY (conversation)
    "gemini-3.5-flash":      {"in": 1.50 / 1_000_000, "out": 9.00 / 1_000_000},
    # Active in REGISTRY (whisper)
    "gemini-3.1-flash-lite": {"in": 0.25 / 1_000_000, "out": 1.50 / 1_000_000},
    # Rollback / alternate models referenced in model_registry.py
    "gemini-2.5-flash":      {"in": 0.30 / 1_000_000, "out": 2.50 / 1_000_000},
    "gemini-2.5-flash-lite": {"in": 0.10 / 1_000_000, "out": 0.40 / 1_000_000},
    "gemini-3-flash":        {"in": 0.50 / 1_000_000, "out": 3.00 / 1_000_000},
}
_DEFAULT = {"in": 1.50 / 1_000_000, "out": 9.00 / 1_000_000}


def record_usage(agent: str, usage, model: str | None = None) -> None:
    """Record token counts + estimated cost from a Gemini response's usage_metadata.

    `usage` is the SDK's usage_metadata object (or None — e.g. a stream that
    never reported usage, in which case we no-op). Attribute reads are defensive
    (getattr ... or 0) because not every response/chunk populates both counts.

    Thinking tokens (`thoughts_token_count`) are billed as output, so they are
    folded into `kind="completion"`.

    Cost uses `model` when given, otherwise get_model(agent); falls back to
    `_DEFAULT` if the model id isn't in PRICES yet. Side effect: increments
    llm_tokens and llm_cost.
    """
    if usage is None:
        return
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    candidates = getattr(usage, "candidates_token_count", 0) or 0
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    completion = candidates + thoughts

    llm_tokens.labels(agent=agent, kind="prompt").inc(prompt)
    llm_tokens.labels(agent=agent, kind="completion").inc(completion)

    model_id = model if model is not None else get_model(agent)
    price = PRICES.get(model_id, _DEFAULT)
    llm_cost.labels(agent=agent).inc(prompt * price["in"] + completion * price["out"])