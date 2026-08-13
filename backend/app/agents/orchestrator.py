"""
Per-turn multi-agent orchestration — the "brain" that sequences the two agents.

Sits between the transport layer (main.py) and the raw agent calls (agents.py).
Given one mentor message, `handle_turn` coordinates the whole turn:

  1. Retrieve relevant past-session memories for this user (vector search).
  2. Stream the mentee reply to the client (conversation agent), with retries.
  3. Signal the reply is done, then run the whisper/coach agent, with retries.
  4. Verify the whisper's memory citations are real (anti-hallucination), strip
     the [Mn] markers, and send the cleaned note to the client.
  5. Persist this mentor message to memory for future turns.

Why the agents run in this order (sequential, not parallel): the whisper agent
critiques the *latest exchange*, so it needs the mentee's reply as input.
Streaming the mentee reply first also gets visible output to the user fastest,
while the (slower, blocking) coaching note is computed afterward.

Resilience model: Gemini Flash models are rate-limited and occasionally
overloaded, so both agent calls are wrapped in bounded retry loops that
degrade gracefully (a friendly message) instead of crashing the socket. Every
attempt and outcome is recorded to Prometheus.

Side effects: streams JSON frames over the websocket, writes to vector memory,
and increments several metrics. It does NOT touch MongoDB — main.py persists
the returned values.
"""

import asyncio
import re
import time

from app.agents.agents import conversation_agent_stream, whisper_agent
from app.agents.grounding import verify_claim
from app.memory.memory import add_memory, get_relevant_memories
from app.observability.metrics import gemini_calls, agent_latency, whisper_grounding, moderation_flags, record_dominant_emotion
from app.observability.llm_metrics import record_usage
from app.safety.moderation import moderate
from app.db.mongo import save_flagged
from app.schemas.models import FlaggedMessage

# Shown in place of a mentee reply that toxic-bert flagged. Keeps the practice
# session constructive instead of surfacing an abusive AI turn.
TOXIC_FALLBACK = (
    "I want to make sure our conversation stays constructive. "
    "Let's refocus on the feedback."
)

# Shown to the user when the conversation model is rate-limited (HTTP 429 /
# RESOURCE_EXHAUSTED). Phrased for an end user, not a developer.
QUOTA_MSG = (
    "Sorry — I'm having trouble responding right now. "
    "The AI service hit its rate limit. Please try again shortly."
)

# Matches an entire memory-citation bracket the whisper agent may emit. Handles
# both the single form ("[M2]") and the combined form ("[M1, M2]") — the model
# sometimes groups several indices into one bracket, and an earlier single-index
# pattern silently left those combined markers in the displayed note. INDEX_RE
# then pulls the numeric indices back out of a matched bracket.
CITATION_RE = re.compile(r"\[M\d+(?:\s*,\s*M?\d+)*\]")
INDEX_RE = re.compile(r"\d+")


def _citation_indices(whisper: str) -> list[int]:
    """Every memory index cited anywhere in `whisper`, across single and combined
    brackets — e.g. "[M1] ... [M2, M3]" -> [1, 2, 3]."""
    indices: list[int] = []
    for bracket in CITATION_RE.findall(whisper):
        indices.extend(int(n) for n in INDEX_RE.findall(bracket))
    return indices


def _strip_citations(whisper: str) -> str:
    """Remove all [Mn] / [M1, M2] markers and tidy the leftover spacing.

    Removing a mid-sentence marker leaves artifacts like "seen in  ." (a double
    space) or "evidence ." (a space before punctuation); we pull punctuation back
    against the preceding word and collapse repeated whitespace so the cleaned
    note reads naturally.
    """
    text = CITATION_RE.sub("", whisper)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def verify_grounding(whisper: str, memories: list[str]) -> tuple[str, str]:
    """Validate the whisper's [Mn] memory citations and strip them for display.

    Two-stage anti-hallucination check for memory recall:

      Stage 1 — structural: every [Mn] index must point to a real entry in
        `memories` (1-indexed to match the [M1], [M2]... numbering built in
        agents.whisper_agent). A cheap, deterministic range check.
      Stage 2 — semantic: even a valid index can be mischaracterized, so for each
        in-range citation we run grounding.verify_claim() to confirm the note
        actually reflects the cited memory's content (DeBERTa NLI, with an LLM
        judge fallback for ambiguous cases).

    The [Mn] markers are always stripped from the returned text — they're an
    internal grounding signal, not something the mentor should see.

    Returns (cleaned_text, status) where status feeds the whisper_grounding
    metric and is one of:
      - "grounded":   citations present, all in range, and all semantically
                      consistent with the memory they cite
      - "ungrounded": at least one citation is out of range (Stage 1) or
                      mischaracterizes its memory (Stage 2), or the agent cited
                      a memory when none existed at all
      - "no_memory":  no citations, treated as the agent coaching only on the
                      current exchange (the expected, non-error case)
    """
    indices = _citation_indices(whisper)
    cleaned = _strip_citations(whisper)
    if not memories:
        # The agent cited something despite having no memories to draw on — a
        # clear hallucination. Scrub the bogus markers before returning.
        if indices:
            return cleaned, "ungrounded"
        return cleaned, "no_memory"
    if not indices:
        return cleaned, "no_memory"
    # Stage 1 — structural: every citation index must fall within the list.
    if not all(1 <= i <= len(memories) for i in indices):
        return cleaned, "ungrounded"
    # Stage 2 — semantic: confirm each cited memory actually backs the note.
    # The claim we verify is the note WITHOUT the [Mn] markers (the markers are
    # plumbing, not part of the assertion being grounded).
    for i in indices:
        cited_memory = memories[i - 1]
        if verify_claim(cleaned, cited_memory) == "ungrounded":
            return cleaned, "ungrounded"
    return cleaned, "grounded"


def _record_attempt(agent: str, outcome: str, start: float) -> None:
    """Record one Gemini attempt: latency of the API call plus the outcome counter.

    Convention (identical at conversation and whisper):
      - `start` is time.perf_counter() taken immediately before the API call
      - observe once per attempt, including failures and quota
      - do not include retry backoff in the observation
    """
    agent_latency.labels(agent=agent, outcome=outcome).observe(time.perf_counter() - start)
    gemini_calls.labels(agent=agent, outcome=outcome).inc()


async def _stream_mentee_reply(websocket, history, user_message) -> str:
    """Stream the mentee reply to the client, retrying on transient failures.

    Drives the conversation agent and forwards each chunk to the browser as a
    {"type": "token"} frame, accumulating the full text to return so the caller
    can persist it. Usage metadata only arrives on (usually the last) some
    chunks, so we capture it whenever present and record it once on success.

    Failure handling (up to 3 attempts):
      - Quota/rate-limit (429 / RESOURCE_EXHAUSTED): non-retryable in practice,
        so we send the user-facing QUOTA_MSG and bail immediately.
      - Other errors: count it and back off with increasing delay before
        retrying — 1.5s, 3s — to ride out brief model overload.
    If all attempts fail, a generic fallback string is streamed so the user
    always sees *something* and the turn can proceed to the whisper step.

    Returns the full reply text (or QUOTA_MSG/fallback). Side effects: websocket
    frames, latency/outcome metrics.
    """
    for attempt in range(3):
        full_reply = ""
        usage = None
        start = time.perf_counter()
        try:
            stream = conversation_agent_stream(history, user_message)
            for chunk in stream:
                if chunk.text:
                    full_reply += chunk.text
                    await websocket.send_json({"type": "token", "content": chunk.text})
                if getattr(chunk, "usage_metadata", None):
                    usage = chunk.usage_metadata
            # A stream that completes but yields no text is treated as a failed
            # attempt (fall through to retry) rather than returning "".
            if full_reply:
                _record_attempt("conversation", "ok", start)
                record_usage("conversation", usage)
                return full_reply
            _record_attempt("conversation", "error", start)
        except Exception as e:
            err = str(e)
            print(f"Conversation attempt {attempt + 1} failed: {e}")
            # Retrying a quota error won't help (the limit is time-based), so
            # short-circuit with the friendly message instead of burning retries.
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                _record_attempt("conversation", "quota", start)
                await websocket.send_json({"type": "token", "content": QUOTA_MSG})
                return QUOTA_MSG
            _record_attempt("conversation", "error", start)
            # Linear backoff between retries; cheap insurance against momentary overload.
            await asyncio.sleep(1.5 * (attempt + 1))

    fallback = "Sorry — I'm having trouble responding right now. Please try again in a moment."
    await websocket.send_json({"type": "token", "content": fallback})
    return fallback


async def handle_turn(websocket, history, user_message, user_id, conversation_id):
    """Run one full coaching turn: mentee reply + grounded coach whisper.

    This is the orchestration entrypoint called by main.py per inbound message.
    Sequence and rationale are documented at the module level; in short, the
    mentee reply is streamed first (fast, visible), then the whisper is computed
    from that reply (it critiques the just-completed exchange).

    Inputs:
      websocket    -- client connection; streamed to as a side effect
      history      -- prior turns, BEFORE this message (supplied by main.py)
      user_message -- the mentor's latest message (already PII-redacted)
      user_id      -- whose memories to retrieve/extend

    Returns (full_reply, whisper_label, whisper_text, mentee_mod): the assembled
    mentee reply, the coach's one-word tone/category for the note, the cleaned
    coaching note (or None if the whisper agent ultimately failed), and the
    mentee-output moderation verdict. main.py uses these to persist the turn; a
    None whisper is intentionally not persisted so transient "model busy" filler
    never lands in history. The label is returned even on failure (it falls back
    to "Insight") but is only persisted alongside a truthy whisper. mentee_mod
    lets main.py stamp the assistant message's flagged/flag_type fields.

    Side effects: websocket frames, a vector-memory write, and metrics.
    """
    # Pull memories first so they can be injected into the whisper prompt. Done
    # up front (before streaming) so retrieval latency overlaps with nothing
    # critical and is ready by the time the whisper agent runs.
    past_patterns = get_relevant_memories(user_id, user_message)

    full_reply = await _stream_mentee_reply(websocket, history, user_message)

    # Moderate the mentee (AI) output before signalling done. We record the
    # mentee's emotion distribution for tracking, but only ACT on toxicity: a
    # crisis/suicidality signal isn't meaningful for an AI playing a character,
    # so toxic-bert is the only gating layer for the mentee.
    raw_mod = moderate(full_reply, role="mentee")
    emotions = raw_mod.get("emotions")
    record_dominant_emotion("mentee", emotions)
    toxic_hits = raw_mod.get("toxic_scores")
    mentee_mod = {
        "flagged": bool(toxic_hits),
        "flag_type": "toxic" if toxic_hits else None,
        "suicide_score": None,
        "crisis_score": None,
        "toxic_scores": toxic_hits if toxic_hits else None,
        "emotions": emotions,
    }
    if toxic_hits:
        # Replace the abusive turn with a safe, on-task redirect and send that
        # instead so the practice session stays constructive.
        full_reply = TOXIC_FALLBACK
        await websocket.send_json({"type": "token", "content": full_reply})
        moderation_flags.labels(role="mentee", flag_type="toxic").inc()
        await save_flagged(FlaggedMessage(
            user_id=user_id, conversation_id=conversation_id,
            role="mentee", content=full_reply,
            flag_type="toxic", toxic_scores=toxic_hits, emotions=emotions,
        ))

    # Tell the client the mentee reply is complete; the UI flips to a
    # "Muse is reflecting…" state while the whisper is generated.
    await websocket.send_json({"type": "done"})

    # Default to a None whisper so that if every retry fails we fall through to
    # the "busy" branch below and signal main.py not to persist anything.
    whisper_label, whisper_text = "Insight", None
    for attempt in range(3):
        start = time.perf_counter()
        try:
            whisper_label, whisper_text = whisper_agent(history, user_message, full_reply, past_patterns)
            _record_attempt("whisper", "ok", start)
            break
        except Exception as e:
            err = str(e)
            print(f"Whisper attempt {attempt + 1} failed: {e}")
            # Unlike the conversation agent we still retry on quota here (the
            # whisper is best-effort and not in the user's critical path), but we
            # label the outcome so quota vs. generic errors stay distinguishable.
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                _record_attempt("whisper", "quota", start)
            else:
                _record_attempt("whisper", "error", start)
            # Reset so a partial/failed attempt never leaks a stale value out of
            # the loop if the next attempt also throws.
            whisper_text = None
            await asyncio.sleep(1.5)

    if whisper_text:
        # Strip + validate citations before the note is shown or persisted.
        whisper_text, grounding_status = verify_grounding(whisper_text, past_patterns)
        whisper_grounding.labels(status=grounding_status).inc()
        await websocket.send_json({"type": "whisper", "content": whisper_text, "label": whisper_label})
    else:
        # All whisper attempts failed: show transient filler. whisper_text stays
        # None so main.py skips persistence and this message is never replayed.
        await websocket.send_json({
            "type": "whisper",
            "content": "Muse is momentarily busy (the model is under load) — try another exchange.",
            "label": "Insight",
        })

    # Record this mentor message as a memory for future turns. Done at the end so
    # it can't influence retrieval for the turn that created it.
    add_memory(user_id, user_message)

    return full_reply, whisper_label, whisper_text, mentee_mod