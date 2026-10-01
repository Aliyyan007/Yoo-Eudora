"""Regression tests for the engagement/pings fixes.

Reproduces the REAL production conditions that were broken:

1. SELF-BOT: our own messages have author.bot == False. Any "is human"
   check that only filters author.bot treats our next message as a human
   response to our previous one â€” so walls never looked stale and never
   got deleted.

2. select_online_user called `_re_engagement.recently_pinged(...)` â€” that
   method only exists on PingController â†’ AttributeError killed the whole
   send path and recently-pinged dedup never ran.

3. Ping pool was only `guild.members` (thin self-bot member cache) â†’ the
   same active member got pinged repeatedly. Now history speakers are
   weighted in.

Run:  python _test_engagement_fixes.py
"""
import asyncio
import os
import sys
import time
import random
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["ENGAGE_MAX_DELETES"] = "3"
os.environ["ENGAGE_STALE_MINUTES"] = "15"
os.environ["ENGAGE_MAX_UNANSWERED"] = "4"

import discord
from src.ai.engagement_log import EngagementLog
from src.ai import re_engagement as re_mod

NOW = time.time()


class FakeUser:
    def __init__(self, uid, bot=False, status=discord.Status.online):
        self.id = uid
        self.bot = bot
        self.status = status
        self.name = f"user{uid}"
        self.display_name = self.name


class FakePerms:
    def __init__(self, ok=True):
        self.read_messages = ok
        self.view_channel = ok


class FakeMsg:
    def __init__(self, mid, author, ts):
        self.id = mid
        self.author = author
        self.created_at = SimpleNamespace(timestamp=lambda: ts)
        self.reference = None
        self.reactions = []
        self.deleted = False

    async def delete(self):
        self.deleted = True


class FakeChannel:
    def __init__(self, msgs, cid=777):
        self.id = cid
        self.name = "general"
        self._msgs = msgs

    def permissions_for(self, member):
        # member.can_view False â†’ denied
        return FakePerms(ok=getattr(member, "can_view", True))

    def history(self, limit=60):
        msgs = sorted((m for m in self._msgs if not m.deleted),
                      key=lambda m: m.created_at.timestamp(), reverse=True)

        class It:
            def __init__(self, items):
                self._it = iter(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration

        return It(msgs[:limit])


class FakeGuild:
    def __init__(self, members):
        self.members = members

    def get_member(self, uid):
        return next((m for m in self.members if m.id == uid), None)


async def test_selfbot_wall_sweep():
    """THE production bug: wall of 10 msgs authored by a USER account
    (author.bot=False). Old code: every msg 'answered' by the next own-msg
    â†’ deleted=0, wall grows forever. New code: drains oldest-first."""
    print("=== self-bot wall sweep (author.bot=False â€” the real case) ===")
    me = FakeUser(1, bot=False)  # self-bot account
    wall = [FakeMsg(1000 + i, me, NOW - (110 - i * 10) * 60) for i in range(10)]
    ch = FakeChannel(wall)
    eng = EngagementLog()
    for m in wall:
        eng.record("777", m, kind="revive")

    n = eng.unanswered_count("777", history_msgs=[m for m in wall if not m.deleted], bot_user=me)
    assert n == 10, f"unanswered_count should be 10 (own msgs aren't responses), got {n}"

    deleted_total = 0
    for sweep_i in range(4):
        remaining = await eng.sweep_before_send(ch, me, force=True)
        deleted_total = sum(1 for m in wall if m.deleted)
        print(f"  sweep {sweep_i + 1}: deleted_total={deleted_total} remaining={remaining}")
    # Design is a SOFT cap: oldest-first, up to 3/sweep, while over
    # max_unanswered(4) â€” plus ANY msg older than stale*3 (45min) regardless.
    # Wall ages 110..20min â†’ the 7 msgs >45min must die; the 3 newest stay.
    very_old = [m for m in wall if NOW - m.created_at.timestamp() > eng.stale_s * 3]
    assert all(m.deleted for m in very_old), "every >45min msg must be deleted"
    assert remaining <= eng.max_unanswered, f"wall must drain to cap, got {remaining}"
    assert deleted_total == 7, f"expected exactly the 7 very-old msgs, got {deleted_total}"
    print("  wall drained to soft cap, oldest-first âœ”")

    # channel paused itself after repeated failed cycles (own msgs no longer
    # fake "human" activity) â€” clear the pause, then the gate must open
    assert eng.is_paused("777"), "channel should be paused after failed cycles"
    eng._paused_since.clear()
    eng._fails["777"] = 0
    assert eng.can_send_engagement(
        "777", history_msgs=[m for m in wall if not m.deleted], bot_user=me), \
        "after draining to cap the gate must open"
    print("  pause engaged during wall, gate opens after drain âœ”")


async def test_own_msgs_not_responses_but_humans_are():
    print("\n=== own message isn't a 'response'; a human's is ===")
    me = FakeUser(1, bot=False)
    human = FakeUser(99, bot=False)
    eng = EngagementLog()

    m1 = FakeMsg(2001, me, NOW - 60 * 60)      # stale engagement
    m2 = FakeMsg(2002, me, NOW - 50 * 60)      # our own next engagement â€” NOT a response
    ch = FakeChannel([m1, m2])
    eng.record("777", m1, kind="revive")
    eng.record("777", m2, kind="revive")
    await eng.sweep_before_send(ch, me, force=True)
    assert m1.deleted or m2.deleted, "own follow-up must not protect the previous msg"
    print(f"  own-follow-up msgs deletable: m1={m1.deleted} m2={m2.deleted} âœ”")

    eng2 = EngagementLog()
    b = FakeMsg(3001, me, NOW - 60 * 60)
    h = FakeMsg(3002, human, NOW - 55 * 60)    # human spoke after â€” protect
    ch2 = FakeChannel([b, h])
    eng2.record("888", b, kind="revive")
    await eng2.sweep_before_send(ch2, me, force=True)
    assert not b.deleted, "human reply must protect the engagement msg"
    print("  human reply protects the message âœ”")


async def test_pause_bookkeeping_with_selfbot():
    """Own msgs shouldn't count as 'human_recent' â€” failed cycles must
    actually accumulate so the channel pauses."""
    print("\n=== failure bookkeeping works with self-bot author ===")
    me = FakeUser(1, bot=False)
    eng = EngagementLog()
    eng.pause_after_fails = 2
    # 5 unanswered, all stale, in history â€” no humans at all
    wall = [FakeMsg(4000 + i, me, NOW - (90 - i * 8) * 60) for i in range(7)]
    ch = FakeChannel(wall)
    for m in wall:
        eng.record("777", m, kind="revive")
    await eng.sweep_before_send(ch, me, force=True)
    await eng.sweep_before_send(ch, me, force=True)
    assert eng._fails.get("777", 0) >= 2, f"fails should accumulate, got {eng._fails.get('777')}"
    assert "777" in eng._paused_since, "channel should be paused after repeated failed cycles"
    print(f"  paused after {eng._fails['777']} failed cycles âœ”")


async def test_ping_selection():
    """select_online_user must (a) not crash on the recently_pinged lookup,
    (b) not re-ping a recently-pinged user while alternatives exist,
    (c) favor recent speakers, (d) exclude self/bots/no-view members."""
    print("\n=== ping selection ===")
    me = FakeUser(1, bot=False)
    lina = FakeUser(50, status=discord.Status.online)          # the over-pinged user
    online_b = FakeUser(51, status=discord.Status.online)
    idle_c = FakeUser(52, status=discord.Status.idle)
    offline_d = FakeUser(53, status=discord.Status.offline)
    offline_e = FakeUser(54, status=discord.Status.offline)
    real_bot = FakeUser(55, bot=True)
    no_view = FakeUser(56, status=discord.Status.online)
    no_view.can_view = False
    guild = FakeGuild([me, lina, online_b, idle_c, offline_d, offline_e, real_bot, no_view])
    ch = FakeChannel([], cid=777)

    # history with a couple of speakers â€” they should get heavy weight
    spk = offline_d  # a lurker who actually talks sometimes
    hist = [FakeMsg(6000 + i, spk, NOW - i * 60) for i in range(3)]

    picked = re_mod.select_online_user(
        guild, exclude_ids={me.id}, channel=ch, history_msgs=hist)
    assert picked is not None, "selection returned None â€” regression (AttributeError path)"
    assert picked.id != me.id, "must never ping ourselves"
    assert not picked.bot, "must never ping a real bot"
    assert picked.id != no_view.id, "must never ping someone who can't see the channel"
    print(f"  picked {picked.name} âœ” (no crash, no self/bot/no-view)")

    # distribution check: ping lina first (marked recently-pinged) then ensure
    # subsequent picks avoid her while alternatives exist
    ctrl = re_mod._ping_controller
    ctrl._recently_pinged.clear()
    ctrl.record_user_ping("777", user_id=lina.id)
    counts = {}
    for _ in range(300):
        p = re_mod.select_online_user(
            guild, exclude_ids={me.id}, channel=ch, history_msgs=hist)
        counts[p.id] = counts.get(p.id, 0) + 1
    assert lina.id not in counts, \
        f"recently-pinged user must be excluded while alternatives exist, got {counts}"
    print(f"  recently-pinged excluded; distribution: {counts} âœ”")

    # without dedup pressure, everyone eligible should appear across many picks
    ctrl._recently_pinged.clear()
    counts = {}
    for _ in range(500):
        p = re_mod.select_online_user(
            guild, exclude_ids={me.id}, channel=ch, history_msgs=hist)
        counts[p.id] = counts.get(p.id, 0) + 1
    eligible = {lina.id, online_b.id, idle_c.id, offline_d.id, offline_e.id}
    assert set(counts) - eligible == set(), f"ineligible picked: {set(counts) - eligible}"
    assert len(counts) >= 3, f"pool too narrow â€” repeats same user: {counts}"
    # the recent speaker should be the most-picked (3.0 weight bonus)
    top = max(counts, key=counts.get)
    assert top == offline_d.id, f"recent speaker should dominate, got {counts}"
    print(f"  varied picks, recent speaker favored: {counts} âœ”")

    # empty-pool edge: ping everyone, then only recently-pinged remain â†’ fallback
    ctrl._recently_pinged.clear()
    for uid in eligible:
        ctrl.record_user_ping("777", user_id=uid)
    p = re_mod.select_online_user(
        guild, exclude_ids={me.id}, channel=ch, history_msgs=None)
    assert p is not None, "fallback pool should still return someone"
    print(f"  fallback pool works when everyone was recently pinged âœ”")
    ctrl._recently_pinged.clear()


async def test_sweep_then_gate_ordering():
    """With a full wall, gate-check-then-return (old order) deadlocks:
    blocked AND never drained. Sweep-first drains even while gating."""
    print("\n=== sweep drains even when gate would block ===")
    me = FakeUser(1, bot=False)
    eng = EngagementLog()
    wall = [FakeMsg(7000 + i, me, NOW - (80 - i * 5) * 60) for i in range(6)]
    ch = FakeChannel(wall)
    for m in wall:
        eng.record("777", m, kind="revive")

    # gate is blocked BEFORE sweep (old order would return here)
    blocked_before = not eng.can_send_engagement(
        "777", history_msgs=[m for m in wall if not m.deleted], bot_user=me)
    assert blocked_before, "6 unanswered > cap 4 â€” gate must block"
    # sweep still runs and deletes
    await eng.sweep_before_send(ch, me, force=True)
    deleted = sum(1 for m in wall if m.deleted)
    assert deleted == 3, f"sweep should still delete while gated, got {deleted}"
    print(f"  gate blocked yet sweep deleted {deleted} âœ”")


async def main():
    random.seed(42)
    await test_selfbot_wall_sweep()
    await test_own_msgs_not_responses_but_humans_are()
    await test_pause_bookkeeping_with_selfbot()
    await test_ping_selection()
    await test_sweep_then_gate_ordering()
    print("\nALL PASS")


asyncio.run(main())




