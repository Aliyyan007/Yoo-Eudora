"""Deterministic outbound/persona boundary tests."""
from src.ai import output_guard


def ok(text, persona, surface="reply"):
    valid, reasons = output_guard.validate_outbound(text, persona_id=persona, surface=surface)
    return valid, reasons


def test_isla_never_uses_roadman_voice():
    valid, reasons = ok("nah bruv that's dead funny", "isla")
    assert not valid
    assert "persona-banned-vocabulary" in reasons
    assert ok("wait that's actually hilarious", "isla")[0]


def test_isla_never_sends_dry_k():
    assert not ok("k", "isla")[0]
    assert ok("okay wait that's fair", "isla")[0]


def test_rowan_emoji_policy_is_nearly_zero():
    assert not ok("that is so cute 🥰", "rowan")[0]
    assert not ok("aww mate", "rowan")[0]
    assert ok("actual disaster 💀", "rowan")[0]


def test_eudora_emoji_and_vocabulary_constraints():
    assert not ok("aww so cute", "eudora")[0]
    assert not ok("love you ❤️", "eudora")[0]
    assert not ok("that's jokes 😂💀", "eudora")[0]
    assert ok("that's jokes 💀", "eudora")[0]


def test_unified_guard_catches_leaks_on_statuses_and_proactive():
    assert not ok("probably groq api", "rowan", "status")[0]
    assert not ok("as an AI language model", "isla", "proactive")[0]


def test_status_length_cap_is_tighter_than_reply_cap():
    long_status = "x" * 120
    assert not ok(long_status, "rowan", "status")[0]
