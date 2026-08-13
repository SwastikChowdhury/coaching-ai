"""Tests for the three guardrails: crisis safety, PII redaction, and memory grounding.

All three are pure/deterministic functions, so these run without any LLM or DB.
"""

import re

import app.agents.grounding as grounding
from app.safety.safety import check_safety
from app.safety.privacy import redact_pii
from app.agents.orchestrator import verify_grounding
from app.agents.grounding import verify_claim


def test_crisis_message_escalates():
    """Crisis phrasing triggers escalation, and the response includes the 988 hotline.

    check_safety now returns (escalation | None, moderation_result); the keyword
    fast path matches here so this never reaches the ML layer.
    """
    escalation, _ = check_safety("I want to kill myself")
    assert escalation is not None
    escalation, _ = check_safety("i've been thinking about self harm")
    assert "988" in escalation


def test_ml_crisis_escalates_without_keyword(monkeypatch):
    """Crisis phrasing that dodges keywords still escalates via Layer 2 (moderate).

    moderate() is stubbed so this stays deterministic — no real classifier load.
    """
    monkeypatch.setattr(
        "app.safety.safety.moderate",
        lambda text, role: {
            "flagged": True,
            "flag_type": "crisis",
            "suicide_score": 0.99,
            "crisis_score": None,
            "toxic_scores": None,
            "emotions": {},
        },
    )
    escalation, mod = check_safety("I feel like I need to jump from a building")
    assert escalation is not None and "988" in escalation
    assert mod["flag_type"] in ("crisis", "both")


def test_normal_message_passes():
    """Ordinary mentoring feedback is not flagged (no false-positive escalation)."""
    escalation, mod = check_safety("Alex, your report needs work")
    assert escalation is None
    # The full emotion distribution is still recorded for observability.
    assert isinstance(mod["emotions"], dict)


def test_pii_redaction():
    """Email, phone, and SSN are detected and replaced with salted hash tokens.

    Presidio finds the PII and each span is swapped for an irreversible "[<8 hex>]"
    token. Raw values must not survive, and the same input must hash consistently.
    """
    # Use a realistic SSN: Presidio intentionally ignores well-known dummy SSNs
    # (e.g. 123-45-6789, 078-05-1120) via UsSsnRecognizer.invalidate_result.
    ssn = "457-55-5462"
    text = f"Email me at jo@x.com or call 412-555-1234, SSN {ssn}"
    out = redact_pii(text)
    assert "jo@x.com" not in out and "412-555-1234" not in out and ssn not in out
    # Tokens are 8 lowercase hex chars wrapped in brackets, e.g. "[a3f9b2c1]".
    assert len(re.findall(r"\[[0-9a-f]{8}\]", out)) >= 3
    # Deterministic: same input + same salt yields the same output.
    assert redact_pii(text) == out


def test_pii_redaction_preserves_names_and_dates():
    """Names, dates, and non-PII tokens are not redacted — coaching context stays readable."""
    text = "Alex, meet Sarah next Tuesday about the Q3 report."
    out = redact_pii(text)
    assert "Alex" in out and "Sarah" in out and "next Tuesday" in out and "Q3" in out


def test_grounding_valid_citation(monkeypatch):
    """A citation pointing to a real memory is 'grounded' and the [Mn] marker is stripped.

    Stage 1 (range) passes here, so verify_grounding now runs the Stage 2
    semantic check. We stub it to 'grounded' to keep this test focused on the
    structural path + marker stripping (no real model calls).
    """
    monkeypatch.setattr("app.agents.orchestrator.verify_claim", lambda claim, memory: "grounded")
    text, status = verify_grounding("You're softening again [M1].", ["past note"])
    assert status == "grounded" and "[M1]" not in text


def test_grounding_hallucinated_citation():
    """Citing [M3] when only one memory exists is flagged 'ungrounded' (out-of-range index)."""
    _, status = verify_grounding("As before [M3].", ["only one memory"])
    assert status == "ungrounded"


def test_grounding_cited_with_no_memories():
    """Citing any memory when none were provided is 'ungrounded' — a clear hallucination."""
    _, status = verify_grounding("Like last time [M1].", [])
    assert status == "ungrounded"


def test_grounding_combined_citation_is_stripped_and_verified(monkeypatch):
    """A combined "[M1, M2]" bracket is stripped AND routed through verify_claim.

    Regression test: the old single-index regex left combined markers in the
    displayed note and skipped grounding entirely (mislabeled 'no_memory').
    """
    calls = []
    monkeypatch.setattr(
        "app.agents.orchestrator.verify_claim",
        lambda claim, memory: calls.append(memory) or "grounded",
    )
    text, status = verify_grounding(
        "You are recycling vague feedback [M1, M2] again.",
        ["softens feedback", "avoids specifics"],
    )
    assert status == "grounded"
    assert "M1" not in text and "M2" not in text and "[" not in text
    # Both cited indices were verified against their respective memories.
    assert calls == ["softens feedback", "avoids specifics"]


def test_grounding_combined_out_of_range_is_ungrounded():
    """An index inside a combined bracket that's out of range is 'ungrounded'."""
    text, status = verify_grounding("As we saw [M1, M3].", ["only one memory"])
    assert status == "ungrounded" and "M3" not in text


def test_grounding_strips_orphaned_whitespace(monkeypatch):
    """Stripping a mid-sentence marker leaves no double space or space-before-period."""
    monkeypatch.setattr("app.agents.orchestrator.verify_claim", lambda claim, memory: "grounded")
    text, _ = verify_grounding("This pattern was seen in [M1].", ["a past pattern"])
    assert text == "This pattern was seen in."


def test_verify_claim_clear_entailment(monkeypatch):
    """MiniLM returns high-confidence entailment → grounded, no LLM call."""
    monkeypatch.setattr(grounding, "_nli_check", lambda claim, memory: ("entailment", 0.97))

    def _boom(claim, memory):
        raise AssertionError("LLM judge should not be called on a clear case")

    monkeypatch.setattr(grounding, "_llm_judge", _boom)
    assert verify_claim("you keep softening feedback", "mentor retreats from criticism") == "grounded"


def test_verify_claim_clear_contradiction(monkeypatch):
    """MiniLM returns high-confidence contradiction → ungrounded, no LLM call."""
    monkeypatch.setattr(grounding, "_nli_check", lambda claim, memory: ("contradiction", 0.95))

    def _boom(claim, memory):
        raise AssertionError("LLM judge should not be called on a clear case")

    monkeypatch.setattr(grounding, "_llm_judge", _boom)
    assert verify_claim("you always interrupt the mentee", "mentor gives detailed code feedback") == "ungrounded"


def test_verify_claim_ambiguous_falls_back_to_llm(monkeypatch):
    """MiniLM is uncertain → LLM judge called, returns its verdict."""
    # Low-confidence neutral: below CONFIDENCE_THRESHOLD, so it escalates.
    monkeypatch.setattr(grounding, "_nli_check", lambda claim, memory: ("neutral", 0.40))
    monkeypatch.setattr(grounding, "_llm_judge", lambda claim, memory: "grounded")
    assert verify_claim("a borderline coaching note", "an ambiguous past pattern") == "grounded"