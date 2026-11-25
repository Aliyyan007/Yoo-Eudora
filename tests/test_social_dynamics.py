"""Ambient social-dynamics policy tests."""
from src.ai.engagement import EngagementEngine
from src.ai.social_dynamics import AmbientSignals, ambient_reply_chance


def sig(**kw):
    base = dict(
        text="anyone know what happened with the update?",
        time_chance=0.30,
        engagement_probability=0.10,
        active_users=2,
    )
    base.update(kw)
    return AmbientSignals(**base)


def test_questions_and_new_participants_increase_ambient_chance():
    plain = ambient_reply_chance(sig(text="yeah that's cool"))
    question = ambient_reply_chance(sig())
    newcomer = ambient_reply_chance(sig(is_new_participant=True))
    assert question > plain
    assert newcomer > question


def test_busy_room_and_failed_engagement_reduce_chance():
    quiet = ambient_reply_chance(sig(active_users=1))
    busy = ambient_reply_chance(sig(active_users=6))
    ignored = ambient_reply_chance(sig(consecutive_engagement_fails=2))
    assert busy < quiet
    assert ignored < quiet


def test_recent_engagement_is_a_strong_hook():
    ordinary = ambient_reply_chance(sig(recent_engagement=False))
    answering_opener = ambient_reply_chance(sig(recent_engagement=True))
    assert answering_opener > ordinary + 0.2


def test_other_name_collapses_ambient_chance():
    ordinary = ambient_reply_chance(sig())
    other = ambient_reply_chance(sig(mentions_other_name=True))
    assert other < ordinary * 0.25


def test_zero_willingness_no_longer_has_high_probability_floor():
    engine = EngagementEngine()
    p0 = engine.get_engagement_probability("chan")
    engine.boost("chan", 2.0)
    p2 = engine.get_engagement_probability("chan")
    assert p0 < 0.10
    assert p2 > p0 * 5


def test_user_engagement_shapes_warmth():
    high = ambient_reply_chance(sig(user_engagement_level="high"))
    minimal = ambient_reply_chance(sig(user_engagement_level="minimal"))
    assert high > minimal
