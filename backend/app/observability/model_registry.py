"""Lightweight model registry for API-hosted models.

Single source of truth for which model each agent runs on,
with version history and live rollback.

Why this exists: model choice is an operational lever, not a code constant.
agents.py asks `get_model(agent)` at call time, so flipping a model takes effect
on the next turn with no redeploy. The /admin endpoints in main.py read this
registry and trigger `rollback`, giving a fast "undo" if a newly promoted model
regresses in production. State is in-memory, so changes reset on restart — fine
for a demo, but a real deployment would back this with a datastore.

Token counts for each call are recorded in llm_metrics from the API's
usage_metadata; model id here is only used to select which model to call.
"""

REGISTRY = {
#   Better for the demo (needs billing):
#   "gemini-3.5-flash"            # $1.50 / $9.00  · near-Pro quality at Flash speed — best responses (Recommended)
#   "gemini-3-flash"              # $0.50 / $3.00  · cheaper middle ground
#   "gemini-2.5-flash"            # $0.30 / $2.50  · current
    "conversation": {
        "model": "gemini-3.5-flash",
        "previous": "gemini-2.5-flash",
        "version": 2,
        "reason": "Near-Pro quality for the mentee persona",
    },
#   "gemini-2.5-flash-lite"     # $0.10 / $0.40  · current
#   "gemini-3.1-flash-lite"       # $0.25 / $1.50  · newer, very low latency (Recommended)
    "whisper": {
        "model": "gemini-3.1-flash-lite",
        "previous": "gemini-2.5-flash-lite",
        "version": 2,
        "reason": "Low latency + low cost for short coaching notes",
    },
}


def get_model(agent: str) -> str:
    """Return the model id an agent is currently running.

    Called by agents.py on every LLM call, so it reflects live rollbacks. Raises
    KeyError if `agent` isn't registered — intentional, since an unknown agent
    key is a programming error, not a runtime condition to swallow.
    """
    return REGISTRY[agent]["model"]


def rollback(agent: str) -> dict:
    """Swap an agent back to its previous model and bump the version. Returns new state.

    The current and previous model ids are exchanged (so a second rollback acts
    as a redo), and `version` is incremented to keep a monotonic change count.
    Returns {"error": ...} when there's nothing to roll back to, which main.py
    uses to decide whether to count the rollback metric.

    Side effect: mutates REGISTRY in place (process-wide, in-memory state).
    """
    entry = REGISTRY.get(agent)
    if not entry or not entry.get("previous"):
        return {"error": f"No previous model for agent '{agent}'"}
    entry["model"], entry["previous"] = entry["previous"], entry["model"]
    entry["version"] += 1
    return {"agent": agent, "model": entry["model"], "version": entry["version"]}