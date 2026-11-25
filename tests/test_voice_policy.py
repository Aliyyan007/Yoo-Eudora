"""Voice output policy and persona-specific proactive fallback tests."""
from types import SimpleNamespace

from src.voice import proactive as vp
from src.ai import output_guard


def test_voice_fallbacks_follow_active_persona():
    rowan_client = SimpleNamespace(persona=SimpleNamespace(id="rowan"))
    isla_client = SimpleNamespace(persona=SimpleNamespace(id="isla"))

    rowan_fallbacks, rowan_addressed = vp._fallback_pool(rowan_client)
    isla_fallbacks, isla_addressed = vp._fallback_pool(isla_client)

    assert rowan_fallbacks != isla_fallbacks
    assert rowan_addressed != isla_addressed
    assert any("playing" in q or "decent" in q for q in rowan_fallbacks)
    assert any("photos" in q or "song" in q for q in isla_fallbacks)


def test_voice_surface_uses_persona_output_boundaries():
    ok, why = output_guard.validate_outbound(
        "aww that's so cute ❤️", persona_id="rowan", surface="voice")
    assert not ok
    assert why

    ok, why = output_guard.validate_outbound(
        "as an AI language model I can help", persona_id="isla", surface="voice")
    assert not ok
    assert "internals-leak" in why
