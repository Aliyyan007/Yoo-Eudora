"""Persona life-continuity regressions."""
import datetime

from src.ai import life_state
from src.ai import prompts
from src.persona.profiles import EUDORA, ISLA, ROWAN


def _patch_memory(monkeypatch):
    store = {}
    monkeypatch.setattr(life_state.mem, "load_server_state", lambda key: store.get(key))
    monkeypatch.setattr(life_state.mem, "save_server_state", lambda key, value: store.__setitem__(key, value))
    monkeypatch.setattr(life_state, "_CACHE", None)
    monkeypatch.setattr(life_state, "_CACHE_AT", 0.0)
    return store


def test_activity_is_stable_within_slot_and_persists(monkeypatch):
    store = _patch_memory(monkeypatch)
    now = datetime.datetime(2026, 3, 4, 18, 30)  # evening

    first = life_state.current_activity("eudora", now=now)
    assert life_state.current_activity("eudora", now=now) == first

    # A cold cache simulates restart/redeploy: the same anchor comes back.
    monkeypatch.setattr(life_state, "_CACHE", None)
    monkeypatch.setattr(life_state, "_CACHE_AT", 0.0)
    assert life_state.current_activity("eudora", now=now) == first
    assert "persona_life_state" in store


def test_activity_anchors_are_persona_specific(monkeypatch):
    _patch_memory(monkeypatch)
    now = datetime.datetime(2026, 3, 4, 18, 30)
    anchors = {
        life_state.current_activity("eudora", now=now),
        life_state.current_activity("isla", now=now),
        life_state.current_activity("rowan", now=now),
    }
    # Determinism does not collapse the three different lives into one script.
    assert len(anchors) >= 2


def test_continuity_context_forbids_conflicting_invention(monkeypatch):
    _patch_memory(monkeypatch)
    ctx = life_state.continuity_context("rowan", now=datetime.datetime(2026, 3, 4, 22, 30))
    assert "[life continuity:" in ctx
    assert "do not invent a conflicting" in ctx
    assert "repair-shop" in ctx or "band practice" in ctx


def test_reply_prompt_contains_life_anchor(monkeypatch):
    _patch_memory(monkeypatch)
    monkeypatch.setattr("src.ai.d1_memory.get_user_memory_text", lambda uid, name: "no notes")
    monkeypatch.setattr("src.ai.d1_memory.get_user_profile", lambda uid: {})
    prompts.set_persona(EUDORA)
    try:
        prompt = prompts.build_reply_prompt(
            transcript="[10s ago] alice: wyd",
            username="alice",
            user_id="123",
            trigger_message="wyd",
            mood="chill",
            my_name="eudora",
        )
    finally:
        prompts.set_persona(EUDORA)
    assert "[life continuity:" in prompt
    assert "use THIS anchor" in prompt


def test_status_activities_follow_persona_routines(monkeypatch):
    _patch_memory(monkeypatch)
    now = datetime.datetime(2026, 3, 4, 15, 0)
    assert life_state.status_activity("isla", now=now)
    assert life_state.status_activity("rowan", now=now)
    assert life_state.status_activity("isla", now=now) != ""


def test_profile_prompts_keep_distinct_never_say_constraints():
    assert "bruv" in ISLA.never_says
    assert "hearts" in ROWAN.never_says or "sparkles" in ROWAN.never_says
