"""
Hybrid grounding verifier.
Step 1: MiniLM pair NLI — fast, local, free. Handles clear cases.
Step 2: LLM-as-judge (gemini-3.1-flash-lite) — only called when MiniLM is uncertain.

This is the content-level half of the anti-hallucination check: structural
validation (are the cited [Mn] indices real?) lives in orchestrator.py, while
this module answers the harder question — does the coaching note actually
reflect the memory it cites, or did it mischaracterize/invent details?

Cost model: the local MiniLM pass is free and resolves the clear-cut cases
(high-confidence entailment or contradiction). Only the genuinely ambiguous
middle falls through to the paid LLM judge, which we keep as cheap as possible
(one-word answer, max_output_tokens=10). The grounding_llm_judge_calls counter
tracks how often that fallback fires so the dashboard surfaces MiniLM's
uncertainty rate.

Both models are initialized once at import time and reused on every call.
"""

import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from google import genai
from google.genai import types

from app.observability.metrics import grounding_llm_judge_calls
from app.observability.llm_metrics import record_usage

load_dotenv()

# Local NLI model — loaded once. Native pair classification: premise = memory,
# hypothesis = claim, then the 3-way softmax (entailment / contradiction / neutral).
NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
_nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
_nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
_nli_model.eval()

# Reused for every LLM-judge fallback; never re-created per call.
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# MiniLM must clear this bar to short-circuit the (slower, paid) LLM judge.
# Below it, the case is considered ambiguous and escalated to the judge.
CONFIDENCE_THRESHOLD = 0.85

LLM_JUDGE_PROMPT = """You are a grounding verifier. You will be given:
1. A memory: a fact stored about a person's communication pattern from a past session.
2. A claim: a coaching note that references that memory.

Your job is to answer ONE word only: 'grounded' or 'ungrounded'.
- 'grounded' means the claim accurately reflects or reasonably follows from the memory.
- 'ungrounded' means the claim mischaracterizes, contradicts, or invents details not in the memory.

Memory: {memory}
Claim: {claim}

Answer (one word only):"""


def _nli_check(claim: str, memory: str) -> tuple[str, float]:
    """Run MiniLM pair NLI on (memory, claim).

    Treats the memory as the premise and the claim as the hypothesis, then reads
    the model's native 3-way softmax. Returns (label, confidence) where label is
    'entailment', 'neutral', or 'contradiction' (the top-scoring relation) and
    confidence is its probability.

    Best-effort: any failure degrades to ('neutral', 0.0) so the caller falls
    through to the LLM judge rather than crashing the turn.
    """
    try:
        inputs = _nli_tokenizer(
            memory,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = _nli_model(**inputs).logits[0]
        probs = F.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs).item())
        label = _nli_model.config.id2label[idx]
        return label, float(probs[idx])
    except Exception as e:  # noqa: BLE001 — grounding must never break a turn
        print(f"NLI check failed (degrading to neutral): {e}")
        return ("neutral", 0.0)


def _llm_judge(claim: str, memory: str) -> str:
    """Ask gemini-3.1-flash-lite whether the claim is grounded in the memory.

    The cheapest possible call: one-word answer, max_output_tokens=10. Returns
    'grounded' or 'ungrounded'. The response is stripped + lowercased and parsed
    leniently (the model may add punctuation). Any error or unexpected response
    falls back to 'ungrounded' — when we've already decided MiniLM was unsure,
    the conservative default is to treat the note as not-yet-verified.

    The grounding_llm_judge_calls counter is incremented BEFORE the API call so
    even failed attempts are counted (the dashboard tracks fallback frequency,
    not just successes).
    """
    grounding_llm_judge_calls.inc()
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=LLM_JUDGE_PROMPT.format(memory=memory, claim=claim),
            config=types.GenerateContentConfig(
                max_output_tokens=10,
                # gemini-3.1-flash-lite is a thinking model: left at its default
                # it spends the (tiny) token budget reasoning and returns empty
                # text, and even with thinking off its verdicts wobble on
                # borderline pairs. thinking_budget=0 frees the 10-token budget
                # for the answer; temperature=0 makes the one-word verdict stable.
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        record_usage(
            "grounding_judge",
            response.usage_metadata,
            model="gemini-3.1-flash-lite",
        )
        verdict = (response.text or "").strip().lower()
        # "ungrounded" contains "grounded" as a substring, so test it first.
        if verdict.startswith("ungrounded"):
            return "ungrounded"
        if verdict.startswith("grounded"):
            return "grounded"
        return "ungrounded"
    except Exception as e:  # noqa: BLE001 — grounding must never break a turn
        print(f"LLM judge failed (degrading to ungrounded): {e}")
        return "ungrounded"


def verify_claim(claim: str, memory: str) -> str:
    """Hybrid verifier: does `claim` accurately reflect `memory`?

    Step 1 — MiniLM pair NLI (local, free):
      - high-confidence contradiction → 'ungrounded'
      - high-confidence entailment   → 'grounded'
      - anything else (neutral, or low confidence) → fall through to Step 2

    Step 2 — LLM judge (paid, only on the ambiguous middle):
      - return _llm_judge's verdict

    Fails open: any unexpected error returns 'grounded'. A grounding glitch
    should never break a conversation turn — better to surface a borderline
    note than to drop it.
    """
    try:
        label, confidence = _nli_check(claim, memory)
        if label == "contradiction" and confidence > CONFIDENCE_THRESHOLD:
            return "ungrounded"
        if label == "entailment" and confidence > CONFIDENCE_THRESHOLD:
            return "grounded"
        return _llm_judge(claim, memory)
    except Exception as e:  # noqa: BLE001 — fail open, never break a turn
        print(f"verify_claim failed (failing open to grounded): {e}")
        return "grounded"
