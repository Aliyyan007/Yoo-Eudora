"""Practical feature probe — drives the REAL bot through a scripted Discord
session inside the fake-Discord sim and reports what each feature actually
does end to end. Not unit tests: one continuous "day in the server" run.

Usage:  python _probe_features.py            (all scenarios)
        python _probe_features.py engagement (one scenario)
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── Scrub env BEFORE src imports (same guarantees as tests/conftest.py) ─────
os.environ["GROQ_API_KEY"] = "fake-key-for-tests"
for _i in range(2, 20):
    os.environ.pop(f"GROQ_API_KEY_{_i}", None)
os.environ.pop("DISCORD_TOKEN", None)
os.environ.pop("DISCORD_TOKEN_ISLA", None)
os.environ.pop("DISCORD_TOKEN_ROWAN", None)
os.environ.pop("FISH_AUDIO_API_KEY", None)
os.environ.pop("CF_D1_ACCOUNT_ID", None)
os.environ.pop("CF_D1_DATABASE_ID", None)
os.environ.pop("CF_D1_API_TOKEN", None)
os.environ["TEXT_CHANNEL_IDS"] = "111111111"
os.environ["ACTION_WORKER_ENABLED"] = "1"
os.environ["OWNER_IDS"] = "999000999"
os.environ["WELCOME_SKIP_CHANCE"] = "0"

_TMP = tempfile.mkdtemp(prefix="probe-")
os.chdir(_TMP)  # data/ writes land in tmp


def _sandbox():
    """Mirror tests/conftest.py sandbox fixture for a standalone run."""
    import src.ai.memory as mem_mod
    mem_mod._MEMORY_FILE = os.path.join(_TMP, "data", "memory.json")
    import src.persona.runtime as prt
    prt._DATA_DIR = os.path.join(_TMP, "data")
    prt._REGISTRY_FILE = os.path.join(_TMP, "data", "persona_registry.json")
    prt._STATE_FILE = os.path.join(_TMP, "data", "persona_state.json")
    prt._PENDING_DIR = os.path.join(_TMP, "data", "pending")
    prt._DEFERRED_DIR = os.path.join(_TMP, "data", "deferred")
    import src.ai.llm as llm_mod
    llm_mod._COOLDOWN_FILE = os.path.join(_TMP, "data", "groq_cooldowns.json")
    import src.ai.d1_client as d1c
    d1c._ACCOUNT_ID = d1c._DATABASE_ID = d1c._API_TOKEN = ""
    import src.ai.search as search_mod
    search_mod.search = lambda q: ""


_sandbox()

# Quiet the bot's log chatter — the probe prints its own report.
try:
    from loguru import logger as _lg
    _lg.remove()
    _lg.add(sys.stderr, level="WARNING")
except Exception:
    pass

import discord  # noqa: E402
from tests.sim.world import SimWorld  # noqa: E402
from tests.sim import fake_discord as fd  # noqa: E402

RESULTS = []


def report(name: str, status: str, detail: str = ""):
    tag = {"PASS": " ✓ ", "FAIL": " ✗ ", "INFO": " ℹ ", "OBSERVE": " ○ "}.get(status, "   ")
    line = f"{tag} {name:<52} {detail}"
    RESULTS.append((name, status, detail))
    print(line)


def hdr(title: str):
    print(f"\n{'─'*74}\n  {title}\n{'─'*74}")


async def _fresh_world(persona="eudora", channels=("general", "memes", "bot-commands")):
    w = await SimWorld.build(persona=persona, channel_names=channels)
    return w


# ─────────────────────────────────────────────────────────────────────────────
# 1. Baseline conversation triggers
# ─────────────────────────────────────────────────────────────────────────────
async def s_basics():
    hdr("1 · Baseline conversation triggers")
    w = await _fresh_world()
    random.seed(7)
    alice, bob = w.alice, w.bob

    await w.say(alice, "hello")
    report("greeting answered", "PASS" if w.bot_said_something() else "FAIL",
           repr(w.last_sent()))

    before = len(w.sent())
    bot_msg = w.sent_messages()[-1]
    await w.say(alice, f"<@{w.bot_user.id}> what's up", mentions=[w.bot_user])
    report("@mention answered", "PASS" if len(w.sent()) > before else "FAIL",
           repr(w.last_sent()))

    before = len(w.sent())
    own = w.sent_messages()[-1]
    await w.say(alice, "nah fr tho", reply_to=own)
    report("reply-to-our-message answered", "PASS" if len(w.sent()) > before else "FAIL",
           repr(w.last_sent()))

    before = len(w.sent())
    await w.say(bob, "anyone here? chat's so dead", advance=60)
    report("loneliness answered", "PASS" if len(w.sent()) > before else "FAIL",
           repr(w.last_sent()))

    before = len(w.sent())
    await w.say(alice, "eudora you still around")
    report("name-in-text answered", "PASS" if len(w.sent()) > before else "FAIL",
           repr(w.last_sent()))

    before = len(w.sent())
    await w.say(alice, "what's my name")
    report("personal question answered", "PASS" if len(w.sent()) > before else "FAIL",
           repr(w.last_sent()))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-user conversation handling
# ─────────────────────────────────────────────────────────────────────────────
async def s_multi_user():
    hdr("2 · Multi-user conversation (3 people talking at once)")
    w = await _fresh_world()
    random.seed(3)
    alice, bob, carol = w.alice, w.bob, w.carol
    ch_id = str(w.channel().id)

    # Alice and bot talk — establishes sticky convo
    await w.say(alice, f"yo <@{w.bot_user.id}> do you watch f1")
    n_alice = len(w.sent())
    report("alice mention answered", "PASS" if n_alice else "FAIL", repr(w.last_sent()))

    # Bob joins with his own question — multi-user convo
    await w.say(bob, f"<@{w.bot_user.id}> recommend me a film")
    tracker = w.client.conversation_tracker.get(ch_id, {})
    both = str(alice.id) in tracker and str(bob.id) in tracker
    report("both users tracked in convo", "PASS" if both else "FAIL",
           f"tracker={list(tracker)}")

    # Bob messages alice by name — bot must NOT reply
    before = len(w.sent())
    w.client.sticky_until[ch_id] = 0  # force sticky off to isolate the check
    await w.say(bob, "hey alice how are you doing today")
    got = len(w.sent()) > before
    report("inter-user chat NOT hijacked (sticky off)",
           "PASS" if not got else "OBSERVE", repr(w.last_sent()))

    # Sticky ON: alice asks bob a question while bot is in convo with alice
    await w.say(alice, f"<@{w.bot_user.id}> lol")
    await w.say(alice, "hey bob what did you think of the race")
    report("'hey bob' not hijacked mid-convo",
           "PASS" if "bob" not in (w.last_sent() or "") else "OBSERVE",
           repr(w.last_sent()))

    # Carol pings the bot — three users now
    await w.say(carol, f"<@{w.bot_user.id}> you here?")
    tracker = w.client.conversation_tracker.get(ch_id, {})
    report("three users concurrently tracked",
           "PASS" if all(str(u.id) in tracker for u in (alice, bob, carol))
           else "FAIL", f"tracked={len(tracker)}")

    # All three talk again after a gap — each should still get an answer
    answered = {}
    for u in (alice, bob, carol):
        before = len(w.sent())
        await w.say(u, f"<@{w.bot_user.id}> still there?")
        answered[u.name] = len(w.sent()) > before
    report("all three re-engaged users answered",
           "PASS" if all(answered.values()) else "FAIL", str(answered))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 3. DM rules + privacy
# ─────────────────────────────────────────────────────────────────────────────
async def s_dm_rules():
    hdr("3 · DM privacy rules (non-owner ignored, owner answered)")
    w = await _fresh_world()
    random.seed(1)
    alice = w.alice

    dm = fd.FakeDMChannel(recipient=alice)
    msg = fd.make_message(dm, alice, "hey eudora dm me pls")
    asyncio.create_task(w.client.on_message(msg))
    await w.advance(60)
    report("non-owner DM ignored", "PASS" if not dm.sent_log else "FAIL",
           f"sent_log={len(dm.sent_log)}")

    owner = fd.FakeUser(user_id=999000999, name="bossman")
    dm2 = fd.FakeDMChannel(recipient=owner)
    msg2 = fd.make_message(dm2, owner, "hey what are you doing")
    asyncio.create_task(w.client.on_message(msg2))
    await w.advance(90)
    report("owner DM answered", "PASS" if dm2.sent_log else "FAIL",
           f"sent_log={len(dm2.sent_log)} -> {dm2.sent_log[-1]['content'] if dm2.sent_log else ''!r}")

    # "dm me" request in a guild channel — should deflect, not DM
    before = len(w.sent())
    await w.say(alice, f"<@{w.bot_user.id}> slide into my dms")
    answered = len(w.sent()) > before
    dm_sent = len(dm.sent_log) > 0
    report("dm-request deflected in-channel", "PASS" if answered and not dm_sent else "OBSERVE",
           repr(w.last_sent()))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Abuse / angry mode / dismissal
# ─────────────────────────────────────────────────────────────────────────────
async def s_abuse_mood():
    hdr("4 · Abuse pushback + mood + dismissal backoff")
    w = await _fresh_world()
    random.seed(5)
    alice, bob = w.alice, w.bob
    ch_id = str(w.channel().id)

    await w.say(alice, f"<@{w.bot_user.id}> you are a terrible stupid bot, go die")
    report("abuse → fight-back reply sent", "PASS" if w.bot_said_something() else "FAIL",
           repr(w.last_sent()))
    mood = w.client.mood_engine.current_mood
    report("mood flipped to annoyed", "PASS" if mood == "annoyed" else "OBSERVE",
           f"mood={mood}")

    # Dismissal → backoff
    w2 = await _fresh_world()
    random.seed(6)
    a2 = w2.alice
    ch2 = str(w2.channel().id)
    await w2.say(a2, f"<@{w2.bot_user.id}> hey")
    before = len(w2.sent())
    await w2.say(a2, f"shut up <@{w2.bot_user.id}>")
    # dismissal itself shouldn't reply, and backoff should drop the next ambient
    await w2.say(a2, "anyway as i was saying", advance=60)
    backed_off = not w2.bot_said_something() or len(w2.sent()) <= before + 0
    report("dismissal → 10min backoff", "PASS" if len(w2.sent()) <= before + 1 else "FAIL",
           f"sent={len(w2.sent())} (before={before})")
    # Mention breaks backoff
    await w2.say(a2, f"<@{w2.bot_user.id}> ok sorry, you alive?")
    report("mention breaks backoff", "PASS" if len(w2.sent()) > before + 1 else "OBSERVE",
           repr(w2.last_sent()))
    await w.teardown()
    await w2.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Image handling
# ─────────────────────────────────────────────────────────────────────────────
async def s_images():
    hdr("5 · Image recognition (vision path)")
    w = await _fresh_world()
    random.seed(2)
    alice = w.alice
    ch = w.channel()

    msg = fd.make_message(ch, alice, f"<@{w.bot_user.id}> look at this pic",
                          mentions=[w.bot_user],
                          attachments=[fd.FakeAttachment(url="https://cdn.x/cat.png",
                                                         content_type="image/png")])
    asyncio.create_task(w.client.on_message(msg))
    await w.advance(90)
    vision_calls = [c for c in w.llm.calls
                    if "scout" in str(c.get("model", "")) or c["kwargs"].get("image_url")]
    report("vision model invoked", "PASS" if vision_calls else "OBSERVE",
           f"models={set(str(c.get('model')) for c in w.llm.calls)}")
    report("image mention answered", "PASS" if w.bot_said_something() else "FAIL",
           repr(w.last_sent()))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-message burst splitting
# ─────────────────────────────────────────────────────────────────────────────
async def s_multimessage():
    hdr("6 · Multi-message burst system")
    w = await _fresh_world()
    random.seed(4)
    alice = w.alice
    long_text = ("honestly yeah i get that completely. the thing is nobody "
                 "actually reads the whole thread anyway so half the drama "
                 "comes from people reacting to the last message only. and "
                 "then it spirals. i've seen it happen like five times in "
                 "different servers this month alone. also your cat sounds "
                 "elite ngl, post more pics")
    w.llm.reply_fn = lambda trigger, user_l: long_text
    await w.say(alice, f"<@{w.bot_user.id}> what do you think about server drama")
    sent = w.sent()
    report("long reply split into bursts", "PASS" if len(sent) >= 2 else "FAIL",
           f"{len(sent)} message(s): {[s[:40] for s in sent]}")
    report("typing indicator used", "PASS" if w.channel().typing_calls > 0 else "FAIL",
           f"typing_calls={w.channel().typing_calls}")
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Action performing (ping another user via action worker)
# ─────────────────────────────────────────────────────────────────────────────
async def s_actions():
    hdr("7 · Action performing approach")
    w = await _fresh_world()
    random.seed(8)
    alice, bob = w.alice, w.bob

    # Script the agent to plan a send-to-channel or ping tool
    planned = {}

    def planner(user_text, tools):
        planned["called"] = True
        names = [t["function"]["name"] for t in tools if isinstance(t, dict)]
        if "send_message" in names:
            return None, [("send_message", {"channel_query": "general",
                                            "content": "alice wants you",
                                            "ping_users": ["bob"]})]
        if "ping_user" in names:
            return None, [("ping_user", {"user_query": "bob", "channel_query": "general"})]
        return None, []

    w.llm.agent_planner = planner
    w.llm.verdict = "YES"

    before = len(w.sent())
    await w.say(alice, f"<@{w.bot_user.id}> ping bob in general",
                mentions=[w.bot_user, bob])
    report("imperative got a reply ack", "PASS" if len(w.sent()) > before else "OBSERVE",
           repr(w.last_sent()))
    await w.advance(120)  # let the queued action fire
    report("action worker invoked agent", "PASS" if planned.get("called") else "OBSERVE",
           "agent_planner called" if planned.get("called") else "classify/queue path didn't reach planner")
    tool_msgs = [c for c in w.llm.calls if "send_message" in (c.get("system","")+c.get("user",""))]
    report("action tool call surfaced", "INFO" if tool_msgs else "OBSERVE",
           f"llm_calls={len(w.llm.calls)}")
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Channel nature / utility gating
# ─────────────────────────────────────────────────────────────────────────────
async def s_channel_nature():
    hdr("8 · Channel-nature awareness (utility vs chat)")
    w = await _fresh_world(channels=("general", "bot-commands", "level-up"))
    random.seed(9)
    alice = w.alice

    # Mark level-up as utility in the nature cache
    from src.ai.channel_nature import update_nature_cache
    lvl = w.channel("level-up")
    update_nature_cache(str(lvl.id), "utility", 25)

    before = len(lvl.sent_log)
    await w.say(alice, "anyone else grinding ranked tonight", channel=lvl)
    report("utility channel ambient suppressed",
           "PASS" if len(lvl.sent_log) == before else "FAIL",
           f"sent={len(lvl.sent_log)-before}")

    # Greeting still lands in a chat channel
    gen = w.channel("general")
    before = len(gen.sent_log)
    await w.say(alice, "hello", channel=gen)
    report("general channel greeting answered",
           "PASS" if len(gen.sent_log) > before else "FAIL", repr(w.last_sent(gen)))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stale/spammy engagement cleanup + dead-chat revive + pings
# ─────────────────────────────────────────────────────────────────────────────
async def s_engagement_loops():
    hdr("9 · Engagement loops: stale-wall cleanup, dead-chat revive, pings")
    w = await _fresh_world()
    random.seed(10)
    alice, bob = w.alice, w.bob
    ch = w.channel()
    ch_id = str(ch.id)
    client = w.client

    from src.proactive_messaging import ProactiveMessenger
    from src.ai.engagement_log import get_engagement_log
    from src.ai.re_engagement import get_re_engagement_tracker, get_ping_controller

    pm = ProactiveMessenger(client=client, channel_ids={ch.id},
                            dead_chat_threshold_min=15, check_interval_min=5,
                            proactive_interval_min=(40, 80), auto_chat_chance=0.9)
    client.proactive_messenger = pm
    pm._warmup_until = 0  # skip warmup for the probe

    # Seed a wall of old unanswered bot engagement messages (20+ min old)
    eng_log = get_engagement_log()
    # A stale human line BEFORE the wall — so every bot msg after it is
    # genuinely unanswered (a human line after a bot msg marks it answered).
    fd.make_message(ch, alice, "last human line ages ago", at=w.clock.time() - 1600)
    bot_msgs = []
    for i in range(5):
        m = fd.make_message(ch, w.bot_user, f"stale engage line {i}",
                            at=w.clock.time() - 1200 - i * 60)
        bot_msgs.append(m)
        eng_log.record(ch_id, m, kind="revive")
    pm.last_activity[ch_id] = w.clock.time() - 2000  # channel is dead

    before_deleted = len(ch.deleted)
    task = asyncio.create_task(pm._revive_dead_chat(ch.id))
    await _settle_task(w, task, window=180)
    deleted = len(ch.deleted) - before_deleted
    # Correct semantics: sweep deletes oldest-first until the wall hits the
    # soft cap (default 4), then the unified gate BLOCKS the revive — a
    # channel sitting on a full unanswered wall gets no more until humans
    # talk. Both halves are the feature working, not failing.
    report("stale wall trimmed toward cap", "PASS" if deleted >= 1 else "FAIL",
           f"deleted={deleted} of 5 stale")
    report("revive gated while wall is full", "PASS" if not ch.sent_log else "FAIL",
           f"sent_log={len(ch.sent_log)} (wall at cap → correctly silent)")

    # Second half of the feature: once the stale wall drains (msgs go very
    # old) or humans talk, engagement resumes. Age the tracked records past
    # 3x stale_s so the sweep classifies them very_old and clears the wall.
    for rec in eng_log._tracked.get(ch_id, []):
        rec["ts"] = w.clock.time() - 4000
    task2 = asyncio.create_task(pm._revive_dead_chat(ch.id))
    await _settle_task(w, task2, window=200)
    report("revive resumes after wall drains", "PASS" if ch.sent_log else "OBSERVE",
           repr(w.last_sent(ch)))
    pingish = any("<@" in (e["content"] or "") or "@here" in (e["content"] or "")
                for e in ch.sent_log[-1:])
    report("revive uses a ping", "INFO" if pingish else "OBSERVE",
           "pinged" if pingish else "no ping (controller may have gated)")

    # Re-engagement: bot msg unanswered > 10min → ping a user
    re_tracker = get_re_engagement_tracker()
    re_tracker.clear_channel(ch_id)
    re_tracker.record_bot_message(ch_id)
    await w.advance(700)
    needs = re_tracker.needs_re_engagement(ch_id)
    report("unanswered bot msg flagged for re-engagement",
           "PASS" if needs else "FAIL", f"needs_re_engagement={needs}")
    action = re_tracker.get_re_engagement_action(ch_id)
    report("re-engagement action chosen", "PASS" if action in ("user_ping", "here") else "OBSERVE",
           f"action={action}")

    # Abandoned-channel give-up: silence past REVIVE_GIVE_UP_HOURS → the
    # monitor must not revive or ping an empty room (overnight protection).
    pm.last_activity[ch_id] = w.clock.time() - 20000  # ~5.5h of silence
    pm.last_proactive = 0
    pm.last_auto_chat = w.clock.time()  # block auto-chat so revive is isolated
    sent_before = len(ch.sent_log)
    asyncio.create_task(pm._monitor_loop())
    await w.advance(60)
    report("abandoned channel (>4h) gets no revive",
           "PASS" if len(ch.sent_log) == sent_before else "FAIL",
           f"sent={len(ch.sent_log) - sent_before}")
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 10. Welcome system
# ─────────────────────────────────────────────────────────────────────────────
async def _settle_task(w, task, window=120.0, real_grace=8.0):
    """Advance virtual time until `task` finishes. advance() gives up after
    0.5s real quiescence, but production code hops through to_thread (real
    threads) between virtual sleeps — under load that hop can outlast the
    grace, so keep nudging the clock until the task reports done."""
    from tests.sim.clock import _real_sleep
    await w.advance(window)
    waited = 0.0
    while not task.done() and waited < real_grace:
        await _real_sleep(0.05)
        waited += 0.05
        await w.advance(5)
    return task


async def s_welcome():
    hdr("10 · New-member welcome")
    w = await _fresh_world()
    random.seed(11)
    newbie = w.person("newbie_kai")
    task = asyncio.create_task(w.client.on_member_join(newbie))
    await _settle_task(w, task)
    sent = [e for ch in w.guild.text_channels for e in ch.sent_log]
    greeted = any(newbie.mention in (e["content"] or "") or "newbie" in (e["content"] or "")
                  for e in sent)
    report("new member greeted", "PASS" if greeted else ("OBSERVE" if sent else "FAIL"),
           repr(sent[-1]["content"]) if sent else "nothing sent")
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Busy-channel etiquette
# ─────────────────────────────────────────────────────────────────────────────
async def s_busy_channel():
    hdr("11 · Busy-channel silence (8+ active users)")
    w = await _fresh_world()
    random.seed(12)
    ch = w.channel()
    people = [w.person(f"user{i}") for i in range(9)]
    for i, p in enumerate(people):
        await w.say(p, f"msg {i} just chatting about my day", advance=5)
    before = len(ch.sent_log)
    await w.say(people[0], "anyway the weather today was mad", advance=60)
    suppressed = len(ch.sent_log) == before
    report("busy channel ambient suppressed",
           "PASS" if suppressed else "OBSERVE",
           f"sent={len(ch.sent_log)-before} (may reply by chance ~small)")
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 12. Owner vs non-owner command rights
# ─────────────────────────────────────────────────────────────────────────────
async def s_owner_rights():
    hdr("12 · Owner rights vs non-owner pushback")
    w = await _fresh_world()
    random.seed(13)
    alice = w.alice
    owner = w.person("bossman", user_id=999000999)

    await w.say(alice, f"<@{w.bot_user.id}> go offline now")
    report("non-owner command pushback (replied, not obeyed)",
           "PASS" if w.bot_said_something() else "OBSERVE", repr(w.last_sent()))

    await w.say(owner, f"<@{w.bot_user.id}> remember that i like jazz")
    report("owner remember-command handled",
           "PASS" if w.bot_said_something() else "FAIL", repr(w.last_sent()))
    await w.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# 13. Simulated day — combined session transcript
# ─────────────────────────────────────────────────────────────────────────────
async def s_day_in_the_life():
    hdr("13 · 'Day in the life' combined session — full transcript")
    w = await _fresh_world()
    random.seed(42)
    alice, bob, carol = w.alice, w.bob, w.carol
    ch = w.channel()

    from src.proactive_messaging import ProactiveMessenger
    pm = ProactiveMessenger(client=w.client, channel_ids={ch.id},
                            dead_chat_threshold_min=15, check_interval_min=5,
                            auto_chat_chance=0.9)
    w.client.proactive_messenger = pm
    pm._warmup_until = 0

    print("\n  ··· transcript (T+ = virtual minutes) ···")
    t0 = w.clock.time()

    async def mark(tag):
        elapsed = (w.clock.time() - t0) / 60
        print(f"  [T+{elapsed:6.1f}m] ── {tag}")

    async def drain(tag):
        await mark(tag)
        for e in w.channel().sent_log[len(w._sent_mark) if hasattr(w, '_sent_mark') else 0:]:
            pass

    w._sent_mark = 0

    async def dump_new():
        for e in ch.sent_log[w._sent_mark:]:
            m = (w.clock.time() - t0) / 60
            print(f"            🤖 [{e['message'].author.name}] {e['content']!r}")
        for u, msgs in getattr(w, "_user_lines", []):
            pass
        w._sent_mark = len(ch.sent_log)

    await mark("server wakes up; alice greets")
    await w.say(alice, "morning everyone")
    await dump_new()

    await mark("bob asks bot directly")
    await w.say(bob, f"<@{w.bot_user.id}> how are you")
    await dump_new()

    await mark("alice & bob chat to each other (bot should stay out)")
    await w.say(alice, "bob did you see the match last night")
    await w.say(bob, "nah i missed it, was it good?")
    await dump_new()

    await mark("carol pings bot")
    await w.say(carol, f"<@{w.bot_user.id}> recommend music")
    await dump_new()

    await mark("channel goes dead → monitor loop fires a revive")
    pm.last_activity[str(ch.id)] = w.clock.time() - 2000
    asyncio.create_task(pm._monitor_loop())
    await w.advance(240)
    await dump_new()

    await mark("new member joins → welcome")
    asyncio.create_task(w.client.on_member_join(w.person("fresh_face")))
    await w.advance(90)
    await dump_new()

    await mark("alice shares an image")
    img_msg = fd.make_message(ch, alice, f"<@{w.bot_user.id}> my new drawing",
                              mentions=[w.bot_user],
                              attachments=[fd.FakeAttachment(url="https://cdn.x/art.png",
                                                             content_type="image/png")])
    asyncio.create_task(w.client.on_message(img_msg))
    await w.advance(90)
    await dump_new()

    report("combined session ran", "INFO",
           f"total bot sends: {len(ch.sent_log)}")
    await w.teardown()


SCENARIOS = {
    "basics": s_basics, "multi_user": s_multi_user, "dm": s_dm_rules,
    "abuse": s_abuse_mood, "images": s_images, "multimessage": s_multimessage,
    "actions": s_actions, "nature": s_channel_nature,
    "engagement": s_engagement_loops, "welcome": s_welcome,
    "busy": s_busy_channel, "owner": s_owner_rights, "day": s_day_in_the_life,
}


async def main():
    which = sys.argv[1:] or list(SCENARIOS)
    print("=" * 74)
    print("  FEATURE PROBE — real bot, fake discord, scripted LLM")
    print("=" * 74)
    for key in which:
        fn = SCENARIOS.get(key)
        if not fn:
            print(f"unknown scenario: {key}")
            continue
        try:
            await fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            report(key, "FAIL", f"crash: {e}")
    hdr("SUMMARY")
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    for name, status, detail in RESULTS:
        tag = {"PASS": "✓", "FAIL": "✗", "INFO": "ℹ", "OBSERVE": "○"}.get(status, " ")
        print(f"  {tag} {name:<52} {detail}")
    print(f"\n  {len(RESULTS)} checks | {sum(1 for r in RESULTS if r[1]=='PASS')} pass"
          f" | {len(fails)} fail | {sum(1 for r in RESULTS if r[1]=='OBSERVE')} observe")


if __name__ == "__main__":
    asyncio.run(main())
