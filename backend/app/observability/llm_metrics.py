"""
LLM token-usage accounting.

Separate from metrics.py because this concerns model usage rather than service
health: it turns each Gemini response's usage_metadata into Prometheus counters
for tokens consumed, broken out per agent. Lets us watch the relative volume of
the conversation vs. whisper agents.

Token counts are exact — they come from the API's usage_metadata, which is what
Google bills against. This module does not convert those counts into dollars;
the API does not return a per-call USD amount, and a local price table would
not match the invoice.

record_usage is called by agents.py (whisper), orchestrator.py (conversation,
after the stream), and grounding.py (LLM judge fallback).
"""

from prometheus_client import Counter

llm_tokens = Counter(
    "muse_llm_tokens_total",
    "LLM tokens consumed, by agent and kind",
    ["agent", "kind"],          # kind: prompt | completion
)


def has_usage(usage) -> bool:
    """True when a usage_metadata object reports any billed tokens.

    Stream chunks often carry an empty usage_metadata object (all counts None/0).
    Treating that as truthy would overwrite a later (or earlier) chunk that
    actually has counts. Callers should only keep a chunk's metadata when this
    returns True.
    """
    if usage is None:
        return False
    for attr in (
        "prompt_token_count",
        "candidates_token_count",
        "response_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "tool_use_prompt_token_count",
        "total_token_count",
    ):
        if (getattr(usage, attr, 0) or 0) > 0:
            return True
    return False


def record_usage(agent: str, usage, model: str | None = None) -> dict | None:
    """Record token counts from a Gemini response's usage_metadata.

    `usage` is the SDK's usage_metadata object (or None — e.g. a stream that
    never reported usage, in which case we no-op). Attribute reads are defensive
    (getattr ... or 0) because not every response/chunk populates every field.

    `model` is accepted so call sites can pass the served model_version without
    a special case; it is not used for pricing.

    Prompt tokens = prompt_token_count + tool-use prompt tokens.
    Completion tokens = candidates/response tokens + thoughts_token_count
    (thinking tokens are billed as output; we count them with completion).

    Returns a dict of the recorded amounts, or None on no-op. Side effect:
    increments llm_tokens.
    """
    if not has_usage(usage):
        return None

    prompt = getattr(usage, "prompt_token_count", 0) or 0
    tool_use = getattr(usage, "tool_use_prompt_token_count", 0) or 0
    candidates = (
        getattr(usage, "candidates_token_count", 0)
        or getattr(usage, "response_token_count", 0)
        or 0
    )
    thoughts = getattr(usage, "thoughts_token_count", 0) or 0
    completion = candidates + thoughts
    prompt_total = prompt + tool_use

    llm_tokens.labels(agent=agent, kind="prompt").inc(prompt_total)
    llm_tokens.labels(agent=agent, kind="completion").inc(completion)
    return {
        "agent": agent,
        "model": (model or "").split("/")[-1],
        "prompt_tokens": prompt_total,
        "completion_tokens": completion,
    }
