"""Wall-scenario test: 10 unanswered bot msgs over ~2h (the screenshot),
post-restart (nothing in _tracked), then a new engagement send attempt.
Also exercises: pinged-user dedup + shared send gap."""
import asyncio
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["ENGAGE_MAX_DELETES"] = "3"
os.environ["ENGAGE_STALE_MINUTES"] = "15"
os.environ["ENGAGE_MAX_UNANSWERED"] = "4"

from src.ai.engagement_log import get_engagement_log
from src.ai.re_engagement import PingController

NOW = time.time()

class FakeUser:
    def __init__(self, uid, bot=False):
        self.id = uid
        self.bot = bot
        self.status = "online"

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
    def __init__(self, msgs):
        self.id = 777
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
    # bot=False on purpose — this is a SELF-BOT: our own messages have
    # author.bot == False, exactly like production. Modeling it as a real
    # bot user is what masked the sweep bug (own msgs counted as "responses").
    bot = FakeUser(1, bot=False)
    human = FakeUser(99)
    # The wall from the screenshot: 10 bot msgs across ~110min, ZERO human msgs
    wall = [FakeMsg(1000 + i, bot, NOW - (110 - i * 10) * 60) for i in range(10)]
    ch = FakeChannel(wall)
    eng = get_engagement_log()
    eng.max_deletes_per_sweep = 3

    # ── Scenario 1: post-restart sweep (tracked is empty) ──
    print("=== wall of 10 unanswered bot msgs, tracked empty (post-restart) ===")
    remaining = await eng.sweep_before_send(ch, bot)
    deleted = sum(1 for m in wall if m.deleted)
    print(f"deleted={deleted} unanswered_remaining={remaining}")
    assert deleted == 3, f"expected 3 deletes (max/sweep), got {deleted}"
    # oldest deleted first
    assert wall[0].deleted and wall[1].deleted and wall[2].deleted and not wall[3].deleted
    # wall now over cap → gate must block new sends
    assert not eng.can_send_engagement("777", history_msgs=[m for m in wall if not m.deleted],
                                       bot_user=bot), "wall still over cap — should block"
    print("gate blocks while wall over cap ok")

    # ── second sweep drains more ──
    remaining = await eng.sweep_before_send(ch, bot, force=True)
    deleted = sum(1 for m in wall if m.deleted)
    print(f"after sweep2: deleted_total={deleted} remaining={remaining}")
    assert deleted == 6, f"expected 6 total after 2 sweeps, got {deleted}"

    # ── tracked-path: same wall but tracked (normal runtime) ──
    print("\n=== tracked wall (normal runtime, no restart) ===")
    eng2 = get_engagement_log()
    wall2 = [FakeMsg(2000 + i, bot, NOW - (110 - i * 10) * 60) for i in range(10)]
    ch2 = FakeChannel(wall2)
    for m in wall2:
        eng2.record("888", m, kind="revive")
    remaining = await eng2.sweep_before_send(ch2, bot, force=True)
    deleted = sum(1 for m in wall2 if m.deleted)
    print(f"deleted={deleted} remaining_unanswered={remaining}")
    assert deleted == 3 and wall2[0].deleted and wall2[1].deleted and wall2[2].deleted
    print("oldest-first deletion on tracked wall ok")

    # ── human answered → protected ──
    print("\n=== answered msg protected ===")
    m_bot = FakeMsg(3001, bot, NOW - 40 * 60)
    m_h = FakeMsg(3002, human, NOW - 38 * 60)
    ch3 = FakeChannel([m_bot, m_h])
    eng.record("999", m_bot, kind="revive")
    await eng.sweep_before_send(ch3, bot, force=True)
    assert not m_bot.deleted, "responded message must be protected"
    print("responded message kept ok")

    # ── ping dedup ──
    print("\n=== same user can't be re-pinged within window ===")
    pc = PingController()
    pc.record_user_ping("777", user_id=42)
    assert 42 in pc.recently_pinged("777")
    assert 43 not in pc.recently_pinged("777")
    print("recently_pinged dedup ok")

    # ── shared send gap ──
    print("\n=== shared engagement send gap ===")
    eng._last_send["555"] = time.time()
    assert not eng.can_send_engagement("555"), "inside shared gap — must block"
    eng._last_send["555"] = time.time() - 99999
    assert eng.can_send_engagement("555"), "outside gap — should allow"
    print("shared gap ok")

    print("\nALL PASS")

asyncio.run(main())
