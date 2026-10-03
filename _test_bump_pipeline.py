"""Practical bump-pipeline test — exercises the REAL scheduler code path
with mocked Discord objects. Simulates exactly what happens on Render:

    python _test_bump_pipeline.py
"""
import os, sys, json, time, asyncio, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name} {extra}")

from src import bump_scheduler as bs
from src.bump_scheduler import BumpScheduler, DEFAULT_BUMP_BOTS

# Sandboxed state file
_tmp = tempfile.mkdtemp(prefix="bump_")
BumpScheduler._STATE_FILE = os.path.join(_tmp, "bump_state.json")

# Kill the human jitter so tests run fast
bs.random.randint = lambda a, b: 0
bs.random.uniform = lambda a, b: a


class FakeCmd:
    """Stands in for discord.py-self SlashCommand — name + app id, callable."""
    def __init__(self, name, app_id, should_fail=None):
        self.name = name
        self.application_id = app_id
        self.calls = 0
        self.should_fail = should_fail
    async def __call__(self, channel):
        self.calls += 1
        if self.should_fail:
            raise self.should_fail
        return object()


class FakeChannel:
    def __init__(self, commands):
        self._commands = commands
        self.name = "bump-room"
        self.fetch_count = 0
    async def application_commands(self):
        self.fetch_count += 1
        return self._commands


class FakeClient:
    def __init__(self, channel):
        self._channel = channel
        self.is_closed_flag = False
    def get_channel(self, cid):
        return self._channel
    async def fetch_channel(self, cid):
        return self._channel
    def is_closed(self):
        return self.is_closed_flag


def make_scheduler(commands):
    """Scheduler + tracking dict of FakeCmds keyed by bot name."""
    ch = FakeChannel(commands)
    client = FakeClient(ch)
    sched = BumpScheduler(client=client, channel_id=1)
    return sched, client


def cmds_for_all_bots(**kw):
    """One FakeCmd per configured bot, sharing a kwargs fail-map."""
    return {name: FakeCmd("bump", app_id, **(kw.get(name) or {}))
            for name, app_id, _cd in DEFAULT_BUMP_BOTS}


async def main():
    # ── 1. All bots due → all bump, state saved ──────────────────────────
    print("\n== all due -> all bump ==")
    cmds = cmds_for_all_bots()
    sched, client = make_scheduler(list(cmds.values()))
    await sched._check_and_bump()
    check("all 5 commands invoked", all(c.calls == 1 for c in cmds.values()),
          f"{[c.calls for c in cmds.values()]}")
    check("last_bump_time updated for all",
          all(sched._last_bump_time.get(n, 0) > 0 for n, *_ in DEFAULT_BUMP_BOTS))
    check("state file written",
          os.path.exists(BumpScheduler._STATE_FILE)
          and len(json.load(open(BumpScheduler._STATE_FILE))["last_bump"]) == 5)
    check("none ready right after bump", len(sched._get_ready_bots()) == 0)

    # ── 2. Cooldown blocks re-bump ───────────────────────────────────────
    print("\n== cooldown respected ==")
    cmds2 = cmds_for_all_bots()
    client._channel._commands = list(cmds2.values())
    sched._cached_commands = list(cmds2.values()); sched._last_cache_refresh = time.time()
    await sched._check_and_bump()
    check("no re-bump during cooldown",
          all(c.calls == 0 for c in cmds2.values()))

    # ── 3. Mixed readiness — only due bots fire ──────────────────────────
    print("\n== partial readiness ==")
    cmds3 = cmds_for_all_bots()
    sched2, _c = make_scheduler(list(cmds3.values()))
    now = time.time()
    # Bumper (1h) bumped 2h ago → due; Disboard (2h) bumped 1h ago → not due
    sched2._last_bump_time = {"Bumper": now - 7200, "Disboard": now - 3600}
    await sched2._check_and_bump()
    check("only due bots bumped",
          cmds3["Bumper"].calls == 1
          and cmds3["Disboard"].calls == 0
          and cmds3["Carl Bot"].calls == 1
          and cmds3["OneBump"].calls == 1
          and cmds3["Bump4You"].calls == 1,
          str({n: c.calls for n, c in cmds3.items()}))

    # ── 4. Missing /bump command → failure + 5-min retry, cache dropped ──
    print("\n== missing command ==")
    cmds4 = cmds_for_all_bots()
    bumper_cmd = cmds4.pop("Bumper")                       # Bumper's cmd gone
    sched3, _c = make_scheduler(list(cmds4.values()))
    sched3._last_bump_time = {}                            # fresh: all due
    await sched3._check_and_bump()
    check("missing bot didn't bump", bumper_cmd.calls == 0)
    check("others still bumped",
          all(c.calls == 1 for n, c in cmds4.items()))
    retry_at = sched3._last_bump_time["Bumper"]
    ready_in = sched3._bot_ready_time("Bumper", 1.0) - time.time()
    check("failure sets ~5min retry", 240 < ready_in < 360,
          f"ready_in={ready_in:.0f}s")
    # Missing command → cache invalidated → next bot's trigger refetches
    check("stale cache dropped + refetched",
          _c._channel.fetch_count >= 2,
          f"fetches={_c._channel.fetch_count}")

    # ── 5. Command raises → failure path, other bots unaffected ──────────
    print("\n== invoke failure ==")
    cmds5 = cmds_for_all_bots(Bumper={"should_fail": RuntimeError("interaction died")})
    sched4, _c = make_scheduler(list(cmds5.values()))
    sched4._last_bump_time = {}                            # fresh: all due
    await sched4._check_and_bump()
    check("failing bot retried later",
          sched4._bot_ready_time("Bumper", 1.0) - time.time() < 360)
    check("healthy bots still bumped",
          all(c.calls == 1 for n, c in cmds5.items() if n != "Bumper"))

    # ── 6. Channel cache-miss → fetch_channel fallback ───────────────────
    print("\n== channel cache miss recovery ==")
    cmds6 = cmds_for_all_bots()
    sched5, client5 = make_scheduler(list(cmds6.values()))
    sched5._last_bump_time = {}                            # fresh: all due
    client5.get_channel = lambda cid: None          # cache dropped it
    await sched5._check_and_bump()
    check("fetch_channel fallback recovered + bumped",
          all(c.calls == 1 for c in cmds6.values()))

    # Channel truly gone → graceful no-op
    client5.get_channel = lambda cid: None
    client5.fetch_channel = None
    async def _no_fetch(cid): raise Exception("404")
    client5.fetch_channel = _no_fetch
    sched5._last_bump_time = {}                      # everything due again
    try:
        await sched5._check_and_bump()
        check("dead channel handled gracefully", True)
    except Exception as e:
        check("dead channel handled gracefully", False, str(e))

    # ── 7. Multi-app matching — right /bump goes to right bot ────────────
    print("\n== app-id matching ==")
    dis = FakeCmd("bump", 302050872383242240)
    imposter = FakeCmd("bump", 999)                  # another app's /bump
    chan7 = FakeChannel([imposter, dis] +
        [FakeCmd("bump", a) for n, a, _ in DEFAULT_BUMP_BOTS if n != "Disboard"])
    sched6, _c = make_scheduler([])                    # channel not used here
    ok = await sched6._trigger_bump(chan7, "Disboard", 302050872383242240)
    check("matched correct application", ok and dis.calls == 1 and imposter.calls == 0)

    # ── 8. Manual batch ignores cooldowns ────────────────────────────────
    print("\n== manual batch ==")
    cmds7 = cmds_for_all_bots()
    sched7, _c = make_scheduler(list(cmds7.values()))
    sched7._last_bump_time = {n: time.time() for n, *_ in DEFAULT_BUMP_BOTS}
    await sched7._perform_bump_batch()
    check("batch bumps despite cooldowns",
          all(c.calls == 1 for c in cmds7.values()))

    # ── 9. Persisted state survives rebuild (persona rotation) ───────────
    print("\n== state persistence ==")
    sched8, _c = make_scheduler(list(cmds_for_all_bots().values()))
    sched8._last_bump_time["Bumper"] = time.time()
    sched8._save_state()
    sched9, _c = make_scheduler([])                  # fresh instance, same file
    check("rebuilt scheduler sees persisted cooldown",
          abs(sched9._last_bump_time.get("Bumper", 0)
              - sched8._last_bump_time["Bumper"]) < 1)

    print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
    sys.exit(1 if FAIL else 0)


asyncio.run(main())
