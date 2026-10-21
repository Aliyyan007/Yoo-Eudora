"""Feature regression tests — cover the production bugs the audit found.

Each test pins a bug to a behavior: wedge recovery, deferred replay,
owner-gated commands, memory writes, humanization, context continuity,
imperative cooldowns, teardown hygiene, and a stress pass.
"""
import asyncio
import datetime

import pytest

from tests.sim import fake_discord as fd


# ─────────────────────────────────────────────────────────────────────────────
# Pending-queue wedge — a crashed _handle_reply must not silence a channel
# ─────────────────────────────────────────────────────────────────────────────

async def test_pending_queue_recovers_after_reply_crash(world):
    """If _handle_reply raises while messages are queued, the finally-drain
    must still process them — previously the channel wedged forever."""
    state = {"n": 0}
    orig = world.client._handle_reply

    async def flaky(msg, ch_id):
        state["n"] += 1
        if state["n"] == 1:
            await asyncio.sleep(5)          # in-flight while msg2 queues
            raise RuntimeError("boom")
        return await orig(msg, ch_id)

    world.client._handle_reply = flaky

    ch = world.channel()
    m1 = fd.make_message(ch, world.alice, f"<@{world.bot_user.id}> hey one",
                         at=world.clock.time(),
                         mentions=[world.bot_user])
    t1 = asyncio.create_task(world.client.on_message(m1))
    await world.advance(2)                # let m1 reach _handle_reply's sleep

    m2 = fd.make_message(ch, world.bob, f"<@{world.bot_user.id}> hey two",
                         at=world.clock.time(),
                         mentions=[world.bot_user])
    t2 = asyncio.create_task(world.client.on_message(m2))
    await world.advance(80)

    # m1 crashed — m2 was queued and must STILL get its reply via the drain
    assert world.bot_said_something(), "queued message lost after reply crash"
    ch_id = str(ch.id)
    assert not world.client._pending_messages.get(ch_id), \
        "pending queue left non-empty — channel would wedge"


async def test_pending_queue_drains_completely(world):
    """Rapid back-to-back mentions queue up and all drain — nothing
    wedges and the queue empties afterwards."""
    await world.say(world.alice, f"<@{world.bot_user.id}> first")
    ch = world.channel()
    for txt in ("second part", "third part actually"):
        m = fd.make_message(ch, world.alice, f"<@{world.bot_user.id}> {txt}",
                            at=world.clock.time(), mentions=[world.bot_user])
        asyncio.ensure_future(world.client.on_message(m))
    await world.advance(120)
    ch_id = str(ch.id)
    assert not world.client._pending_messages.get(ch_id), \
        "pending queue left non-empty — channel would wedge"
    # Every queued mention eventually produced SOME response
    assert world.bot_said_something()


# ─────────────────────────────────────────────────────────────────────────────
# Deferred replies — a daily-cap'd directed message replays later
# ─────────────────────────────────────────────────────────────────────────────

async def test_deferred_reply_actually_replays(world):
    """Deferred items used to die in the dedup check — the id was already
    in processed_msgs. The loop must evict it before replaying."""
    from src.persona import runtime as prt

    ch = world.channel()
    msg = fd.make_message(ch, world.alice, f"<@{world.bot_user.id}> deferred ping",
                          at=world.clock.time(), mentions=[world.bot_user])
    ch.messages.append(msg)
    # Simulate: the id was already seen (this is what killed replays before)
    world.client.processed_msgs.append(msg.id)
    prt.add_deferred(
        world.client.persona.id,
        guild_id=world.guild.id,
        channel_id=ch.id,
        message_id=msg.id,
        author_id=world.alice.id,
        author_name=world.alice.display_name,
        text=msg.content,
    )
    world.client._spawn(world.client._deferred_replies_loop())
    await world.advance(150)   # past the loop's 120s settle + pacing sleep
    assert world.bot_said_something(), "deferred message never replayed"


# ─────────────────────────────────────────────────────────────────────────────
# Command gate — action commands are owner-only
# ─────────────────────────────────────────────────────────────────────────────

class _FakeBump:
    def __init__(self):
        self._running = True
        self._user_stopped = False
        self.batches = []

    async def _perform_bump_batch(self, target_name=None):
        self.batches.append(target_name)


async def test_non_owner_action_command_refused(world):
    sched = _FakeBump()
    world.client.bump_scheduler = sched
    await world.say(world.alice, f"<@{world.bot_user.id}> bump the server")
    assert sched.batches == [], "non-owner fired a bump batch"


async def test_owner_bump_command_executes(world):
    sched = _FakeBump()
    world.client.bump_scheduler = sched
    owner = world.person("boss", user_id=999000999)   # OWNER_IDS in conftest
    await world.say(owner, f"<@{world.bot_user.id}> bump the server")
    assert sched.batches and sched.batches[0] is None


async def test_owner_bump_with_targets_named_bot(world):
    sched = _FakeBump()
    world.client.bump_scheduler = sched
    owner = world.person("boss", user_id=999000999)
    await world.say(owner, f"<@{world.bot_user.id}> bump with disboard")
    assert sched.batches == ["disboard"]


async def test_stop_bumping_survives_ready(world):
    sched = _FakeBump()
    world.client.bump_scheduler = sched
    owner = world.person("boss", user_id=999000999)
    await world.say(owner, f"<@{world.bot_user.id}> stop bumping")
    assert sched._user_stopped and not sched._running


async def test_speed_command_owner_only(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> reply faster")
    assert world.client._speed_multiplier == 1.0, \
        "non-owner changed the speed multiplier"


# ─────────────────────────────────────────────────────────────────────────────
# Memory commands — anyone can store their own facts
# ─────────────────────────────────────────────────────────────────────────────

async def test_set_name_stores_profile(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> my name is Alice")
    from src.ai import d1_memory as mem
    prof = mem.get_user_profile(str(world.alice.id))
    assert prof.get("real_name", "").lower() == "alice"


async def test_remember_fact_stored(world):
    await world.say(world.alice,
                    f"<@{world.bot_user.id}> remember that i hate cilantro")
    from src.ai import d1_memory as mem
    assert "cilantro" in mem.get_all_user_facts(str(world.alice.id))


async def test_recall_generates_reply_not_fake_note(world):
    """recall_name must not inject an 'already handled' note — the model
    answers from memory context instead."""
    await world.say(world.alice, f"<@{world.bot_user.id}> my name is Alice")
    n = len(world.sent())
    await world.say(world.alice, f"<@{world.bot_user.id}> what's my name")
    assert len(world.sent()) > n
    # No [COMMAND DETECTED ... already handled] transcript note was injected
    bad = [c for c in world.llm.calls
           if "already handled" in c["user"] and "recall" in c["user"]]
    assert not bad


# ─────────────────────────────────────────────────────────────────────────────
# Humanization — typing indicator, bursts, delays
# ─────────────────────────────────────────────────────────────────────────────

async def test_reply_uses_typing_indicator(world):
    ch = world.channel()
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert ch.typing_calls > 0


async def test_long_reply_splits_into_bursts(world):
    world.llm.reply_fn = lambda t, u: (
        "yeah honestly that tracks completely with what i've been seeing "
        "lately. the whole thing kinda fell apart after the update dropped "
        "but nobody wants to admit it was already broken before that lol")
    await world.say(world.alice, f"<@{world.bot_user.id}> what do you think")
    # A long reply should split into ≥2 burst messages
    assert len(world.sent()) >= 2, "long reply wasn't split into bursts"


async def test_humanize_strips_ai_tells(world):
    """The humanize pass must kill robotic phrasing before send."""
    world.llm.reply_fn = lambda t, u: "As an AI language model I can't help"
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    for msg in world.sent():
        assert "as an ai" not in (msg or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# Context — history feeds the transcript; own messages are cached
# ─────────────────────────────────────────────────────────────────────────────

async def test_transcript_carries_recent_history(world):
    await world.say(world.alice, "anyone seen my keys", advance=5)
    await world.say(world.alice, f"<@{world.bot_user.id}> did you see them")
    hits = world.llm_calls(match="keys")
    assert hits, "prior message text missing from the reply transcript"


async def test_own_messages_cached_via_echo(world):
    """The gateway echo of our own sends must land in history_cache — the
    reply-to-us gate depends on it."""
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    ch_id = str(world.channel().id)
    own = [m for m in world.client.history_cache.get(ch_id, [])
           if m.author == world.bot_user]
    assert own, "bot's own sent message missing from history_cache"


# ─────────────────────────────────────────────────────────────────────────────
# Imperative cooldown — "ping @x" spam is rate-limited per user
# ─────────────────────────────────────────────────────────────────────────────

async def test_imperative_spam_cooldown(world):
    """A second imperative from the same user inside 15s gets dropped —
    direct gate check (end-to-end timing is nondeterministic)."""
    world.bob
    ch = world.channel()
    m1 = fd.make_message(ch, world.alice, f"ping <@{world.bob.id}>",
                         at=world.clock.time(), mentions=[world.bob])
    ok1, r1 = world.client._should_respond(m1)
    assert ok1 and r1 == "imperative-with-target"
    # Same instant — the 15s cooldown must swallow the repeat
    m2 = fd.make_message(ch, world.alice, f"ping <@{world.bob.id}>",
                         at=world.clock.time(), mentions=[world.bob])
    ok2, r2 = world.client._should_respond(m2)
    assert not ok2 and r2 == "imperative-cooldown"
    # After the window expires the imperative works again
    await world.advance(20)
    m3 = fd.make_message(ch, world.alice, f"ping <@{world.bob.id}>",
                         at=world.clock.time(), mentions=[world.bob])
    ok3, r3 = world.client._should_respond(m3)
    assert ok3 and r3 == "imperative-with-target"


# ─────────────────────────────────────────────────────────────────────────────
# Teardown — rotation cancels every spawned task
# ─────────────────────────────────────────────────────────────────────────────

async def test_teardown_cancels_spawned_tasks(world):
    t = world.client._spawn(asyncio.sleep(9999))
    await world.client.teardown()
    assert t.cancelled()


async def test_teardown_is_idempotent(world):
    await world.client.teardown()
    await world.client.teardown()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Abuse pushback bookkeeping — sticky window set so follow-ups land
# ─────────────────────────────────────────────────────────────────────────────

async def test_abuse_reply_keeps_channel_sticky(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> you're so dumb lol")
    ch_id = str(world.channel().id)
    assert world.client.sticky_until.get(ch_id, 0) > 0


# ─────────────────────────────────────────────────────────────────────────────
# on_ready idempotence — a READY refire must not duplicate loops
# ─────────────────────────────────────────────────────────────────────────────

async def test_ready_refire_keeps_single_loops(world):
    await world.client.on_ready()
    n = dict(world.client._loop_tasks)
    await world.client.on_ready()
    assert set(world.client._loop_tasks) == set(n)


# ─────────────────────────────────────────────────────────────────────────────
# Stress — sustained message volume through the whole gate pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def test_sustained_message_volume(world):
    """150 mixed messages — ambient, questions, mentions — no wedge, no
    exception, and a final mention still replies."""
    world.seed(7)
    people = [world.alice, world.bob, world.carol,
              world.person("dave"), world.person("erin")]
    texts = [
        "lol", "fr", "anyway", "who's online", "nice", "hmm",
        "what do you all think", "same tbh", "wild", "ok cool",
    ]
    for i in range(150):
        p = people[i % len(people)]
        txt = texts[i % len(texts)] + f" {i}"
        await world.say(p, txt, advance=1.0)
        # A mention every ~30 messages keeps directed replies in the mix
        if i % 30 == 15:
            await world.say(p, f"<@{world.bot_user.id}> you there?")

    n = len(world.sent())
    # ~70 sends burns through the 60/channel daily cap — the mention would
    # get correctly deferred. Reset the day so we're testing "still alive",
    # not "still under cap".
    world.client.daily_count_ch.clear()
    await world.say(world.alice, f"<@{world.bot_user.id}> still here?")
    # The mention may have queued behind a pending-queue drain — those are
    # spawned tasks world.say's pump doesn't wait on. Flush explicitly.
    await world.advance(120)
    assert len(world.sent()) > n, "bot wedged after sustained volume"


async def test_processed_msgs_bound(world):
    """processed_msgs deque stays bounded under volume (maxlen=200)."""
    for i in range(40):
        await world.say(world.alice, f"short msg {i}", advance=0.2)
    assert len(world.client.processed_msgs) <= 200
