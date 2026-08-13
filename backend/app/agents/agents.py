"""
Gemini agent definitions: the two LLM personas that power a coaching turn.

muse-ai-lite is a two-agent system, and this module is the boundary between
"our code" and the Gemini API. It owns the system prompts, the request shaping,
and the response parsing for both agents; it does NOT own retries, streaming to
the client, memory, or grounding checks — those live in orchestrator.py.

The two agents and why they're separate:

  - Conversation agent (`conversation_agent_stream`): role-plays the *mentee*
    ("Alex"). It talks directly to the user and is streamed token-by-token for
    a live chat feel. It must stay fully in character.

  - Whisper agent (`whisper_agent`): a private coach ("Muse") that speaks ONLY
    to the mentor/user. It observes the latest exchange (plus optional memories
    of past sessions) and returns a single labeled coaching note. It is a
    one-shot, non-streamed call because the note is short and is shown all at
    once in the side panel.

Keeping the personas in two prompts/models lets us tune cost and latency per
role (see model_registry.py) and prevents the coach's meta-commentary from
leaking into the in-character mentee dialogue.

Token usage for each call is reported to llm_metrics so volume can be tracked
per agent.
"""

import os
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.observability.llm_metrics import record_usage
from app.observability.model_registry import get_model

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Canonical set of one-word categories the whisper agent must tag each note
# with. Used to validate/normalize the model's LABEL line in `_parse_label`;
# anything off-list falls back to the generic "Insight".
WHISPER_LABELS = [
    "Tone", "Pattern", "Subtext", "Opening",
    "Suggestion", "Pacing", "Clarity", "Empathy", "Boundary",
]

# System prompt for the conversation agent. Pins it to the in-character mentee
# persona; the "never break character / never act like an AI" clauses are the
# main guardrail keeping the role-play immersive.
MENTEE_SYSTEM_PROMPT = """You are role-playing as a mentee in a practice mentoring conversation. \
The person talking to you is your mentor, giving you feedback on a proposal you worked hard on. \
Play the mentee realistically: you care about doing well, you're a little insecure about whether \
your work landed, and you can be slightly defensive at first but are ultimately receptive to good \
feedback. Keep replies short and natural, like real chat messages. Stay fully in character as the \
mentee — never break character, never give meta commentary, never act like an AI assistant."""

# System prompt for the whisper agent. The strict "LABEL: <word>" first line is
# a lightweight structured-output contract that _parse_label depends on, and the
# GROUNDING RULE defines the [Mn] citation scheme that orchestrator.verify_grounding
# enforces. Changing either format here requires updating those consumers.
WHISPER_SYSTEM_PROMPT = """You are Muse, a perceptive and warm communication coach. You privately \
observe a practice conversation between a mentor (the person you advise) and a mentee. You speak \
ONLY to the mentor, never to the mentee.

Begin your response with a category label on its own line, in exactly this format:
LABEL: <one word>
where <one word> is the single best fit from: Tone, Pattern, Subtext, Opening, Suggestion, \
Pacing, Clarity, Empathy, Boundary.

Then, on the following line, give the mentor exactly ONE short coaching note (1-2 sentences) that:
- reads the mentee's emotional tone and any subtext beneath their words,
- observes how the mentor's most recent message is landing,
- suggests a concrete next move, optionally with a brief example phrasing in quotes.

GROUNDING RULE: you may be given numbered notes from past sessions, labeled [M1], [M2], etc. \
If — and only if — you reference a recurring habit from a past session, cite its label inline, \
e.g. "you're softening again [M1]". Never invent a past pattern that isn't in the notes. If no \
notes are provided, coach only on the current exchange.

Address the mentor as "you". Be specific and insightful, not generic. After the LABEL line, \
output only the coaching note — no preamble, no other labels except [Mn] citations."""

# Matches the leading "LABEL: <word>" line the whisper prompt asks for. Anchored
# to the start and tolerant of surrounding whitespace; case-insensitive so the
# model's casing doesn't matter.
LABEL_RE = re.compile(r"^\s*LABEL:\s*([A-Za-z]+)\s*\n?", re.IGNORECASE)


def _parse_label(raw: str) -> tuple[str, str]:
    """Split the whisper model's 'LABEL: X\\n<note>' output into (label, note).

    Exists because the model returns label and note as one blob, but the UI
    renders them separately (a tag + a body). We defensively tolerate a model
    that ignores the format: if the LABEL line is missing, or names a category
    outside WHISPER_LABELS, we fall back to the generic "Insight" label and keep
    the full text as the note rather than dropping anything.

    Returns: (label, note) — both always non-empty strings.
    """
    m = LABEL_RE.match(raw)
    if not m:
        return "Insight", raw.strip()
    candidate = m.group(1)
    text = raw[m.end():].strip()
    for label in WHISPER_LABELS:
        if candidate.lower() == label.lower():
            # `text or raw.strip()` guards against a model that emitted only the
            # label line and no note — we'd rather show the raw output than blank.
            return label, text or raw.strip()
    return "Insight", text or raw.strip()


def build_contents(history, user_message):
    """Convert stored chat history into Gemini's `contents` turn format.

    Our DB stores roles as "user"/"assistant"; Gemini expects "user"/"model".
    From the conversation agent's perspective the *mentor* (our "user") is the
    one prompting it, and its own prior mentee lines are "model" turns, so every
    non-user role collapses to "model". The new mentor message is appended last
    as the turn the model must respond to.

    Returns: a list of {role, parts:[{text}]} dicts ready to pass to Gemini.
    """
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def conversation_agent_stream(history, user_message):
    """Start streaming the mentee ("Alex") reply to the mentor's latest message.

    Returns a Gemini streaming iterator (NOT the text) so the orchestrator can
    forward tokens to the client as they arrive — this is what makes the chat
    feel live. The caller is responsible for consuming the stream, error
    handling, and recording usage. Side effect: opens a streaming Gemini call.
    """
    contents = build_contents(history, user_message)
    return client.models.generate_content_stream(
        model=get_model("conversation"),
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=MENTEE_SYSTEM_PROMPT),
    )


def whisper_agent(history, user_message, mentee_reply, past_patterns=None):
    """Generate one private coaching note for the mentor about the latest exchange.

    Unlike the conversation agent, this is a single blocking call: the note is
    short and shown all at once, so streaming buys nothing.

    Inputs:
      history       -- prior turns (mentor/mentee) for context
      user_message  -- the mentor's newest message
      mentee_reply  -- the mentee reply that was just produced this turn
      past_patterns -- optional recurring-habit notes from the mentor's earlier
                       sessions (from vector memory). When present they're
                       injected as numbered [M1], [M2]... lines so the model can
                       cite them; the prompt forbids inventing un-listed patterns
                       and the orchestrator later verifies any citations.

    The full transcript is rebuilt here with explicit "Mentor:"/"Mentee:" labels
    (rather than reusing build_contents) because the coach reasons about the
    dialogue as an outside observer, not as a participant.

    Returns: (label, note) via _parse_label. Side effects: one Gemini call and a
    token-usage record for the "whisper" agent.
    """
    lines = []
    for m in history:
        speaker = "Mentor" if m["role"] == "user" else "Mentee"
        lines.append(f"{speaker}: {m['content']}")
    lines.append(f"Mentor: {user_message}")
    lines.append(f"Mentee: {mentee_reply}")
    transcript = "\n".join(lines)

    # Inject retrieved memories as numbered notes the model can cite by label.
    # The [Mn] numbering must align with `past_patterns` order so the
    # orchestrator's grounding check can validate each citation index.
    memory_context = ""
    if past_patterns:
        joined = "\n".join(f"[M{i+1}] {p}" for i, p in enumerate(past_patterns))
        memory_context = (
            f"\n\nNumbered notes from this mentor's PAST sessions "
            f"(cite [Mn] if you reference one):\n{joined}"
        )

    response = client.models.generate_content(
        model=get_model("whisper"),
        contents=f"Conversation so far:\n\n{transcript}{memory_context}\n\n"
                 f"Give the mentor one short coaching note about the latest exchange.",
        config=types.GenerateContentConfig(system_instruction=WHISPER_SYSTEM_PROMPT),
    )
    record_usage(
        "whisper",
        response.usage_metadata,
        model=getattr(response, "model_version", None),
    )
    return _parse_label(response.text.strip())