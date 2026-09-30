"""Wall-scenario tests: post-restart untracked walls, engagement-only
deletion (normal convo replies survive), ping reseeding from history,
shared send gap, pinged-user dedup."""
import asyncio
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["ENGAGE_MAX_DELETES"] = "3"
os.environ["ENGAGE_STALE_MINUTES"] = "15"
os.environ["ENGAGE_MAX_UNANSWERED"] = "4"

from src.ai.engagement_log import get_engagement_log, _looks_like_engagement
from src.ai.re_engagement import PingController, get_ping_controller

NOW = time.time()
BOT = None  # set in main


class FakeUser:
    def __init__(self, uid, bot=False):
        self.id = uid
        self.bot = bot
        self.status = "online"


class FakeMsg:
    def __init__(self, mid, author, ts, content=""):
        self.id = mid
        self.author = author
        self.created_at = SimpleNamespace(timestamp=lambda: ts)
        self.reference = None
        self.reactions = []
        self.content = content
        self.deleted = False

    async def delete(self):
        self.deleted = True


class FakeChannel:
    def __init__(self, msgs, cid=777):
        self.id = cid
        self.name = "general"
        self._msgs = msgs

    def history(self, limit=60):
        msgs = sorted((m for m in self._msgs if not m.deleted),
                      key=lambda m: m.created_at.timestamp(), reverse=True)

        class It:
            def __init__(self, items): self._it = iter(items)
            def __aiter__(self): return self
            async def __anext__(self):
                try:
                    return next(self._it)
                except StopIteration:
                    raise StopAsyncIteration
        return It(msgs[:limit])


async def main():
    bot = FakeUser(1, bot=True)
    human = FakeUser(99)
    eng = get_engagement_log()
    eng.max_deletes_per_sweep = 3

    # ── Scenario 1: the screenshot wall — 10 unanswered engagement msgs over
    # ~110min, ZERO human msgs, tracked empty (post-restart) ──
    print("=== post-restart wall of 10 engagement msgs ===")
    wall = [FakeMsg(1000 + i, bot, NOW - (110 - i * 10) * 60,
                    content=f"anyone up for a game? {i}") for i in range(10)]
    ch = FakeChannel(wall)
    remaining = await eng.sweep_before_send(ch, bot)
    deleted = sum(1 for m in wall if m.deleted)
    print(f"deleted={deleted} unanswered_remaining={remaining}")
    assert deleted == 3, f"expected 3 deletes (max/sweep), got {deleted}"
    assert wall[0].deleted and wall[1].deleted and wall[2].deleted and not wall[3].deleted
    assert not eng.can_send_engagement(
        "777", history_msgs=[m for m in wall if not m.deleted], bot_user=bot), \
        "wall still over cap - should block"
    print("oldest-first deletion + gate blocks while over cap ok")

    remaining = await eng.sweep_before_send(ch, bot, force=True)
    deleted = sum(1 for m in wall if m.deleted)
    print(f"after sweep2: deleted_total={deleted} remaining={remaining}")
    assert deleted == 6, f"expected 6 total after 2 sweeps, got {deleted}"

    # ── Scenario 2: normal convo replies are NEVER deleted via history path ──
    print("\n=== normal convo bot msgs survive (not engagement-looking) ===")
    convo = [
        FakeMsg(3001, bot, NOW - 40 * 60, content="gotcha, big deadline"),
        FakeMsg(3002, bot, NOW - 39 * 60, content="nice, looks solid"),
        FakeMsg(3003, bot, NOW - 38 * 60, content="yeah, been a few times"),
        # plus actual spam mixed in — one very old (>45min always deleted),
        # five merely stale
        FakeMsg(3004, bot, NOW - 50 * 60, content="anyone up rn"),
        FakeMsg(3005, bot, NOW - 37 * 60, content="<@99> anyone got a good playlist?"),
        FakeMsg(3006, bot, NOW - 36 * 60, content="who's active rn"),
        FakeMsg(3007, bot, NOW - 35 * 60, content="anyone up for a game"),
        FakeMsg(3008, bot, NOW - 34 * 60, content="anyone listening to music"),
        FakeMsg(3009, bot, NOW - 33 * 60, content="yo who's active rn"),
    ]
    ch2 = FakeChannel(convo)
    eng2 = get_engagement_log()
    await eng2.sweep_before_send(ch2, bot, force=True)
    convo_deleted = [m.content for m in convo if m.deleted]
    print(f"deleted: {convo_deleted}")
    # normal convo replies must survive unconditionally
    assert not convo[0].deleted and not convo[1].deleted and not convo[2].deleted, \
        "normal convo replies must NEVER be deleted"
    # spam deletes oldest-first down to the cap: 6 spam -> keep cap(4) -> 2+very_old
    # 50min one is very_old (always deleted), plus enough stale to reach cap
    spam = convo[3:]
    assert spam[0].deleted, "very-old spam must be deleted regardless of cap"
    assert convo[3].deleted and convo[4].deleted
    remaining_spam = [m for m in spam if not m.deleted]
    assert len(remaining_spam) <= 4, f"spam should drain to cap, {len(remaining_spam)} left"
    print("convo replies kept, spam drained oldest-first to cap ok")

    # ── Scenario 3: ping reseed from history (restart-proof dedup) ──
    print("\n=== ping memory reseeded from history after restart ===")
    pc = PingController()
    pc._recently_pinged.clear()  # simulate post-restart
    hist = [
        FakeMsg(4001, bot, NOW - 30 * 60, content="<@42> anyone up rn"),
        FakeMsg(4002, bot, NOW - 60 * 60, content="<@77> anyone up for a game?"),
        FakeMsg(4003, human, NOW - 5 * 60, content="hello"),
        FakeMsg(4004, bot, NOW - 200 * 60, content="<@55> old ping"),  # too old
    ]
    n = pc.seed_pings_from_history("777", hist, bot)
    print(f"seeded={n} recent={pc.recently_pinged('777')}")
    assert 42 in pc.recently_pinged("777")
    assert 77 in pc.recently_pinged("777")
    assert 55 not in pc.recently_pinged("777"), "ping older than window shouldn't count"
    assert n == 2
    print("reseed ok")

    # ── Scenario 4: seed happens automatically inside sweep ──
    print("\n=== sweep auto-seeds ping memory ===")
    pc2 = get_ping_controller()
    pc2._recently_pinged.clear()
    wall3 = [FakeMsg(5000 + i, bot, NOW - (100 - i * 10) * 60,
                     content=f"<@{500 + i}> anyone up for a game?") for i in range(6)]
    ch3 = FakeChannel(wall3, cid=555)
    eng._tracked.pop("555", None)
    await eng.sweep_before_send(ch3, bot, force=True)
    seeded = pc2.recently_pinged("555")
    print(f"auto-seeded ids={seeded}")
    assert 500 in seeded and 505 in seeded
    print("auto-seed ok")

    # ── Scenario 5: answered msgs + tracked deletion still work ──
    print("\n=== answered msg protected ===")
    m_bot = FakeMsg(6001, bot, NOW - 40 * 60, content="anyone up rn")
    m_h = FakeMsg(6002, human, NOW - 38 * 60, content="hi")
    ch4 = FakeChannel([m_bot, m_h])
    eng.record("999", m_bot, kind="revive")
    await eng.sweep_before_send(ch4, bot, force=True)
    assert not m_bot.deleted, "responded message must be protected"
    print("responded message kept ok")

    # ── Scenario 6: shared send gap ──
    print("\n=== shared engagement send gap ===")
    eng._last_send["555"] = time.time()
    assert not eng.can_send_engagement("555")
    eng._last_send["555"] = time.time() - 99999
    eng._tracked.pop("555", None)
    eng._paused_since.pop("555", None)
    assert eng.can_send_engagement("555")
    print("shared gap ok")

    print("\nALL PASS")


asyncio.run(main())
