"""Policy regressions for engagement sending and re-engagement escalation.

These are small deterministic tests for the concurrency and social-boundary
rules that keep proactive behavior from turning into ping spam.
"""
import asyncio
import time

from src.ai.engagement_log import EngagementLog
from src.ai.msg_cache import MessageCache
from src.ai.re_engagement import ReEngagementTracker, invites_response
import src.ai.re_engagement as re_mod
from src.ai import reply as ai_reply


async def test_engagement_reservation_is_atomic_per_channel():
    log = EngagementLog()
    log.min_send_gap_s = 3600

    first, second = await asyncio.gather(
        log.acquire_send_slot("chan"),
        log.acquire_send_slot("chan"),
    )

    assert (first is not None) != (second is not None)
    winner = first or second
    winner.abort()
    # Aborting a failed generation releases the slot instead of consuming the
    # shared cooldown for a message that never reached Discord.
    assert await log.acquire_send_slot("chan") is not None


async def test_engagement_reservation_commit_consumes_gap():
    log = EngagementLog()
    log.min_send_gap_s = 3600

    reservation = await log.acquire_send_slot("chan")
    assert reservation is not None
    reservation.commit()

    assert await log.acquire_send_slot("chan") is None


def test_only_explicit_invitations_arm_reengagement():
    assert invites_response("what's everyone playing tonight?")
    assert invites_response("drop one song you have on repeat")
    assert invites_response("yo <@123> what you think", pinged_user=object())
    assert not invites_response("yeah that's fair")
    assert not invites_response("lol same")
    assert not invites_response("i'm just sorting files rn")


def test_reengagement_ladder_is_bounded_and_user_first(monkeypatch):
    tracker = ReEngagementTracker()
    monkeypatch.setattr(re_mod.PingController, "can_ping_user", lambda self, ch: True)
    monkeypatch.setattr(re_mod.PingController, "can_ping_here", lambda self, ch: True)

    tracker.record_bot_message("chan")
    tracker._last_bot_msg["chan"] = time.time() - tracker._timeout_s - 1
    assert tracker.needs_re_engagement("chan")
    assert tracker.get_re_engagement_action("chan") == "user_ping"

    # After the first ignored ping, another full timeout may escalate once.
    tracker.mark_ping_sent("chan")
    tracker._last_bot_msg["chan"] = time.time() - tracker._timeout_s - 1
    assert tracker.needs_re_engagement("chan")
    assert tracker.get_re_engagement_action("chan") == "here"

    # A second ignored ping is the hard stop: no endless ladder.
    tracker.mark_ping_sent("chan")
    tracker._last_bot_msg["chan"] = time.time() - tracker._timeout_s - 1
    assert not tracker.needs_re_engagement("chan")


def test_reengagement_human_reply_resets_ladder(monkeypatch):
    tracker = ReEngagementTracker()
    monkeypatch.setattr(re_mod.PingController, "can_ping_user", lambda self, ch: True)
    monkeypatch.setattr(re_mod.PingController, "can_ping_here", lambda self, ch: True)
    monkeypatch.setattr(re_mod.PingController, "note_human_activity", lambda self, ch: None)

    tracker.record_bot_message("chan")
    tracker.mark_ping_sent("chan")
    tracker.record_human_reply("chan")
    tracker.record_bot_message("chan")
    tracker._last_bot_msg["chan"] = time.time() - tracker._timeout_s - 1

    assert tracker.get_re_engagement_action("chan") == "user_ping"


def test_msg_cache_interleaves_own_messages_and_excludes_bots_from_activity():
    cache = MessageCache()
    now = time.time()
    cache.add_message("chan", "what are we doing", "alice", "111", now - 20, is_bot=False, message_id=1)
    cache.add_message("chan", "checking the usual chaos", "eudora", "222", now - 10, is_bot=True, message_id=2)
    cache.add_message("chan", "nothing much", "alice", "111", now, is_bot=False, message_id=3)

    transcript = cache.get_context_text("chan", count=8, own_ids={"222"})
    assert "You: checking the usual chaos" in transcript
    assert cache.get_active_user_count("chan", window_minutes=10) == 1


def test_unicode_dashes_are_replaced_not_mojibake(monkeypatch):
    monkeypatch.setattr(ai_reply.random, "random", lambda: 1.0)
    out = ai_reply.humanize("one — two – three")
    assert "—" not in out
    assert "–" not in out
    assert "one - two - three" in out
