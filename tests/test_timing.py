"""Timing & message-shape tests — the bot's human-ness lives in pacing.

Covered behaviors (src/discord_client.py + src/ai/reply.py + multi_message.py):

* read delay scales with message length (_read_delay_seconds)
* hard questions/advice/math/emotional messages buy a think beat
  (_think_delay_seconds)
* stale messages / quiet channels get a "late" catch-up delay
* typing indicator duration scales with burst length
* long replies split into burst messages (determine_message_count /
  split_into_messages)
* near-duplicate replies trigger one anti-repeat regeneration
* owner "reply faster"/"slow down" scales the read/notification delays
* a repeat generic follow-up burst ("wbu?"/"hbu?") is suppressed
* reply-quote behavior: mention/question always quote, ambient ~60%

How delay is measured: FakeMessage.created_at is stamped with the
clock-patched time.time(), so ``sent.created_at - t_dispatch`` is exactly
the wait a real user would perceive — read delay + think delay + typing
simulation included. (``world.say`` always advances the clock by
`advance`, so the say() window itself is NOT the measurement.)
"""
from __future__ import annotations

import asyncio
import json
import random

import pytest

from tests.sim import fake_discord as fd

from src.discord_client import _read_delay_seconds, _think_delay_seconds
from src.ai.multi_message import determine_message_count, split_into_messages


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mention(world) -> str:
    return f"<@{world.bot_user.id}>"


def _delay_since(world, start_ts: float, sent_before: int) -> float:
    """Virtual seconds from `start_ts` (when the bot saw the message) to
    the first bot send that followed it (burst 1 of the reply)."""
    fresh = world.sent_messages()[sent_before:]
    assert fresh, "bot never sent — nothing to measure"
    return fresh[0].created_at.timestamp() - start_ts


def _delay_of(world, incoming, sent_before: int) -> float:
    """Same, for a normally-dispatched message (created_at == dispatch)."""
    return _delay_since(world, incoming.created_at.timestamp(), sent_before)


async def _drive_task(world, task, advance: float, settle_rounds: int = 8):
    """Advance virtual time until the on_message task settles.

    clock.advance() breaks on ~0.5s of REAL quiescence; under executor
    lag a task can park its next virtual sleep just past the window and
    strand. Re-advancing in small windows catches those stragglers."""
    await world.clock.advance(advance)
    for _ in range(settle_rounds):
        if task.done():
            break
        await world.clock.advance(20)
        await world.clock.flush()


async def _dispatch(world, msg, advance: float = 90.0):
    """Feed a hand-built message into on_message and let virtual time run.
    Mirrors world.say() but lets the test control created_at/mentions."""
    task = asyncio.create_task(world.client.on_message(msg))
    await _drive_task(world, task, advance)
    return msg


async def _say(world, person, content, advance: float = 90.0, **kw):
    """world.say() equivalent with a settle loop — every measured send
    uses this so an executor-hop straggler can't strand a parked sleep."""
    channel = kw.pop("channel", None) or world.channel()
    reply_to = kw.pop("reply_to", None)
    ref = None
    if reply_to is not None:
        ref = fd.FakeMessageReference(message_id=reply_to.id, resolved=reply_to)
    mentions = world._resolve_mentions(content, kw.pop("mentions", None))
    msg = fd.make_message(channel, person, content,
                          at=world.clock.time(),
                          mentions=mentions, reference=ref, **kw)
    task = asyncio.create_task(world.client.on_message(msg))
    await _drive_task(world, task, advance)
    return msg


def _script_replies(world, *texts):
    """Force the reply generator to emit `texts` in order (one per reply
    call — including regeneration retries). Keeps consecutive replies
    distinct so the dedup guard doesn't swallow them."""
    box = {"i": 0}

    def fn(trigger, user_l):
        i = min(box["i"], len(texts) - 1)
        box["i"] += 1
        return texts[i]

    world.llm.reply_fn = fn


def _neutralize_extras(world, monkeypatch):
    """Pin sticker/GIF random sends to zero so send counts only ever
    reflect the reply bursts under test."""
    from src.ai.stickers_gifs import get_sticker_manager, get_gif_manager
    monkeypatch.setattr(get_sticker_manager(), "_base_chance", 0.0)
    monkeypatch.setattr(get_gif_manager(), "_base_chance", 0.0)


def _calm(world):
    """Neutral-ish mood so determine_message_count stays length-driven."""
    world.client.mood_engine.set_mood("flat")


# ── Trigger / reply text fixtures (scrubbed of command/sentiment tells) ──────

# ~410 chars, no '?', no commas, no advice/math/emotional keywords — the
# think-score here comes only from length; sentiment stays neutral.
LONG_PARAGRAPH = (
    "got around to sorting through all the old sketches i had piled up in "
    "that drawer near my desk over the weekend and honestly there were way "
    "more of them than i expected some of them go back years and years its "
    "kind of surreal seeing the way my style shifted over time plus a few "
    "i barely even recall making at all"
)

# Question + advice + math + multi-clause + emotional — max think score.
THINK_HEAVY = (
    "should i text her again? what do you think? how much is 5+5 percent "
    "of it anyway? help me decide, honestly i'm still kinda sad about the "
    "whole breakup, it keeps me up"
)

# ~135 chars — long enough to pin the typing cap but under the 150-char
# split threshold so it stays a single burst.
TYPING_LONG = (
    "honestly that whole thing took way longer than expected and i'm still "
    "not fully sure it was worth the hassle but at least its done now"
)
assert 120 < len(TYPING_LONG) <= 150

# >300 chars, 6 sentences, no '?' — and carefully no detect_conversation_nature
# question SUBSTRINGS ('who' hides inside 'whole', 'how' inside 'show', etc.)
# because is_question would cap the burst count at 2. Forces msg_count == 3.
BURST_REPLY = (
    "Honestly i spent the entire afternoon redoing that sketch. The lines "
    "kept coming out wrong no matter the angle i tried. Eventually i "
    "flipped the canvas upside down and everything finally clicked at "
    "once. Kinda proud of the end result not gonna lie. Might post it "
    "later once the shading is done. Plus the lighting took ages to get "
    "right too."
)
assert len(BURST_REPLY) > 300


# ─────────────────────────────────────────────────────────────────────────────
# Unit: _read_delay_seconds
# ─────────────────────────────────────────────────────────────────────────────

def test_read_delay_short_is_snappy():
    for _ in range(300):
        assert 0.4 <= _read_delay_seconds("hi") <= 1.5
        assert 0.4 <= _read_delay_seconds("") <= 1.5
        assert 0.4 <= _read_delay_seconds("lol ok") <= 1.5


def test_read_delay_medium_band():
    for _ in range(300):
        assert 1.0 <= _read_delay_seconds("x" * 50) <= 3.0


def test_read_delay_long_scales_at_reading_pace():
    # ~12-16 chars/sec skim pace: 200 chars -> 12.5-16.7s
    for _ in range(300):
        assert 12.5 <= _read_delay_seconds("x" * 200) <= 16.67


def test_read_delay_caps_at_22s():
    for _ in range(100):
        assert _read_delay_seconds("x" * 1000) == 22.0


def test_read_delay_bands_are_ordered():
    # short (0.4-1.5) and medium (1.0-3.0) overlap per-draw, so compare
    # means; the long band is disjoint (>= 12.5) so hard bounds hold.
    short = [_read_delay_seconds("hi") for _ in range(300)]
    medium = [_read_delay_seconds("x" * 50) for _ in range(300)]
    long = [_read_delay_seconds("x" * 400) for _ in range(200)]
    assert sum(short) / len(short) < sum(medium) / len(medium)
    assert max(medium) < min(long)


# ─────────────────────────────────────────────────────────────────────────────
# Unit: _think_delay_seconds
# ─────────────────────────────────────────────────────────────────────────────

def test_think_delay_banter_is_zero():
    assert _think_delay_seconds("lol ok") == 0.0
    assert _think_delay_seconds("hey") == 0.0
    assert _think_delay_seconds("") == 0.0


def test_think_delay_single_question_scores():
    # one '?' -> score 1.5 -> delay = 1.5 * uniform(0.8, 1.2) = 1.2-1.8
    for _ in range(200):
        assert 1.2 <= _think_delay_seconds("u coming tonight?") <= 1.8


def test_think_delay_more_questions_more_delay():
    one = [_think_delay_seconds("really?") for _ in range(200)]
    many = [_think_delay_seconds("a? b? c? d? e? f?") for _ in range(200)]
    # 6 '?' -> score caps at 4.0 -> min delay 3.2 > single-question max 1.8
    assert min(many) > max(one)


def test_think_delay_advice_ask():
    # 'should i' -> +2.5 -> >= 2.0
    for _ in range(100):
        assert _think_delay_seconds("should i tell her") >= 2.0


def test_think_delay_math():
    # 'how much' -> +2.0 -> >= 1.6
    for _ in range(100):
        assert _think_delay_seconds("how much is 5+5") >= 1.6


def test_think_delay_emotional_topic():
    # 'sad'/'breakup' -> +1.5 -> >= 1.2
    for _ in range(100):
        assert _think_delay_seconds("i'm still sad about the breakup") >= 1.2


def test_think_delay_hard_cap_at_9s():
    for _ in range(200):
        assert _think_delay_seconds(THINK_HEAVY) <= 9.0
    # ...and THINK_HEAVY saturates it (score ~10 -> floor ~6-9)
    assert min(_think_delay_seconds(THINK_HEAVY) for _ in range(50)) >= 6.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit: burst splitting (multi_message.py)
# ─────────────────────────────────────────────────────────────────────────────

def test_message_count_length_thresholds():
    assert determine_message_count("x" * 100, mood="flat") == 1
    assert determine_message_count("x" * 200, mood="flat") == 2
    assert determine_message_count("x" * 400, mood="flat") == 3


def test_message_count_annoyed_stays_single():
    for _ in range(50):
        assert determine_message_count("x" * 400, mood="annoyed") == 1
        assert determine_message_count("x" * 400, mood="angry") == 1


def test_message_count_question_caps_at_two():
    for _ in range(50):
        assert determine_message_count("x" * 500, mood="flat",
                                       is_question=True) == 2


def test_message_count_capped_at_four():
    for _ in range(50):
        n = determine_message_count("x" * 2000, mood="hyped",
                                    is_story=True, is_excited=True)
        assert 3 <= n <= 4


def test_split_on_sentence_boundaries():
    text = ("First part here. Second chunk follows. Third piece arrives. "
            "Fourth bit too. Fifth one lands.")
    parts = split_into_messages(text, 3)
    assert len(parts) == 3
    assert all(p.strip() for p in parts)


def test_split_on_line_breaks():
    parts = split_into_messages("line one\nline two\nline three", 2)
    assert parts == ["line one", "line two line three"]


def test_split_short_text_stays_whole():
    assert split_into_messages("hey", 3) == ["hey"]
    assert split_into_messages("single message", 1) == ["single message"]


# ─────────────────────────────────────────────────────────────────────────────
# E2E: read delay — a paragraph takes measurably longer than a short line
# ─────────────────────────────────────────────────────────────────────────────

async def test_long_message_waits_longer_before_reply(world):
    _calm(world)
    _script_replies(world, "yo", "heyy", "wait lemme finish reading that")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)  # warmup

    n0 = len(world.sent_messages())
    m_short = await _say(world, world.alice, _mention(world) + " hey",
                              advance=90)
    d_short = _delay_of(world, m_short, n0)

    n1 = len(world.sent_messages())
    m_long = await _say(world, world.alice,
                             _mention(world) + " " + LONG_PARAGRAPH,
                             advance=150)
    d_long = _delay_of(world, m_long, n1)

    # read ~22s + think ~2-3s vs read <=3s — clearly separated
    assert d_long > d_short + 8.0, \
        f"long paragraph ({d_long:.1f}s) barely slower than 'hey' ({d_short:.1f}s)"


async def test_read_delay_medium_band_e2e(world):
    """A ~40-char message out-waits a 'hey' on average — the medium band
    overlaps the short band per-draw, so compare means over samples."""
    _calm(world)
    world.seed(23)
    _script_replies(world, "yo", "heyy", "aite", "bet", "lol", "hmm",
                          "fr", "say less", "ok cool")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)

    # 'hey' is a greeting — always answered; ~3 chars -> short band.
    # The medium line rides the sticky convo — ~40 chars -> 1-3s band.
    d_shorts, d_mids = [], []
    for i in range(3):
        n0 = len(world.sent_messages())
        m = await _say(world, world.alice, "hey", advance=60)
        d_shorts.append(_delay_of(world, m, n0))
        n1 = len(world.sent_messages())
        m = await _say(world, 
            world.alice,
            f"yeah the party starts around nine i think {i}",
            advance=60)
        d_mids.append(_delay_of(world, m, n1))

    assert sum(d_mids) / len(d_mids) > sum(d_shorts) / len(d_shorts) + 0.3, \
        f"medium {d_mids} vs short {d_shorts} — no band separation"


# ─────────────────────────────────────────────────────────────────────────────
# E2E: think delay — hard questions take longer than banter
# ─────────────────────────────────────────────────────────────────────────────

async def test_hard_question_thinks_longer_than_banter(world):
    _calm(world)
    _script_replies(world, "yo", "haha real", "idk maybe give it a day")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)  # warmup

    n0 = len(world.sent_messages())
    m_banter = await _say(world, world.alice, "haha yeah that checks out",
                               advance=90)
    d_banter = _delay_of(world, m_banter, n0)

    n1 = len(world.sent_messages())
    m_hard = await _say(world, world.alice, THINK_HEAVY, advance=150)
    d_hard = _delay_of(world, m_hard, n1)

    # think ~7-9s + a longer read vs zero think — clear separation even
    # after sentiment modifiers and random typing draws
    assert d_hard > d_banter + 3.0, \
        f"heavy question ({d_hard:.1f}s) barely slower than banter ({d_banter:.1f}s)"


async def test_math_question_thinks_more_than_plain(world):
    _calm(world)
    # math reply is longer so typing_dur contributes too — same-order gap
    _script_replies(world, "yo", "lol sure", "uhh like two fifty i think")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)

    n0 = len(world.sent_messages())
    m_plain = await _say(world, world.alice, "that movie was pretty decent",
                              advance=90)
    d_plain = _delay_of(world, m_plain, n0)

    n1 = len(world.sent_messages())
    m_math = await _say(world, 
        world.alice, "how much would 25 + 17 be for the tickets?",
        advance=90)
    d_math = _delay_of(world, m_math, n1)

    assert d_math > d_plain, \
        f"math question {d_math:.1f}s vs plain {d_plain:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# E2E: late-message delay
# ─────────────────────────────────────────────────────────────────────────────

async def test_first_reply_of_session_gets_late_penalty(world):
    """A fresh client has last_replied={} — since_last_reply is huge, so
    the first reply always carries the 'opened discord late' delay."""
    _calm(world)
    _script_replies(world, "yo", "heyy")  # both <11 chars -> typing floor 2s
    m1 = await _say(world, world.alice, "hi", advance=90)
    d1 = _delay_of(world, m1, 0)

    n0 = len(world.sent_messages())
    m2 = await _say(world, world.alice, "hey", advance=90)
    d2 = _delay_of(world, m2, n0)

    # identical message shape — the only difference is the late delay
    assert d1 - d2 >= 2.0, \
        f"first reply {d1:.1f}s vs follow-up {d2:.1f}s — no late penalty seen"


async def test_stale_message_gets_late_delay(world):
    """created_at 700s in the past -> 'opening discord late' +3-8s."""
    _calm(world)
    _script_replies(world, "yo", "aite", "bet", "say less")
    TEXT = "still figuring out the weekend plans for the trip"

    await _say(world, world.alice, _mention(world) + " yo", advance=90)
    n0 = len(world.sent_messages())
    m_fresh = await _say(world, world.alice, TEXT, advance=90)
    d_fresh = _delay_of(world, m_fresh, n0)

    # Same text delivered 700s stale — measured from dispatch time, so the
    # delta vs the fresh arm is exactly the late penalty.
    n1 = len(world.sent_messages())
    stale = fd.make_message(world.channel(), world.alice, TEXT,
                            at=world.clock.time() - 700)
    t_dispatch = world.clock.time()
    await _dispatch(world, stale, advance=90)
    d_stale = _delay_since(world, t_dispatch, n1)

    assert d_stale > d_fresh + 1.5, \
        f"stale message {d_stale:.1f}s vs fresh {d_fresh:.1f}s"


async def test_quiet_channel_gets_late_delay(world):
    """>600s since the bot's last reply -> same late penalty even for a
    brand-new message."""
    _calm(world)
    _script_replies(world, "yo", "aite", "bet")
    TEXT = "still figuring out the weekend plans for the trip"

    await _say(world, world.alice, _mention(world) + " yo", advance=90)
    n0 = len(world.sent_messages())
    m_fresh = await _say(world, world.alice, TEXT, advance=90)
    d_fresh = _delay_of(world, m_fresh, n0)

    await world.advance(700)   # chat goes quiet for ~12 min
    n1 = len(world.sent_messages())
    m_quiet = await _say(world, world.alice, TEXT, advance=90)
    d_quiet = _delay_of(world, m_quiet, n1)

    assert d_quiet > d_fresh + 1.5, \
        f"post-quiet reply {d_quiet:.1f}s vs fresh {d_fresh:.1f}s"


# ─────────────────────────────────────────────────────────────────────────────
# E2E: typing indicator
# ─────────────────────────────────────────────────────────────────────────────

async def test_typing_indicator_fires_per_burst(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, BURST_REPLY)
    ch = world.channel()
    n0 = len(world.sent_messages())
    await _say(world, world.alice,
                    _mention(world) + " run me through the sketch drama",
                    advance=150)
    n_sent = len(world.sent_messages()) - n0
    assert n_sent >= 3, f"expected >=3 burst sends, got {n_sent}"
    assert ch.typing_calls >= n_sent, \
        f"typing() called {ch.typing_calls}x for {n_sent} sends"
    assert ch._last_typing is not None and ch._last_typing.enters >= 1


async def test_typing_duration_scales_with_reply_length(world, monkeypatch):
    """typing_dur = len/5.5-8 clamped to [2,12]: a ~135-char burst types
    ~10s longer than a 4-char one, same trigger shape both times."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, "yo", "nice", TYPING_LONG)
    await _say(world, world.alice, _mention(world) + " yo", advance=90)

    # 'gimme the recap' — deliberately free of mood-nudge keywords
    # (CONTENT_NUDGES substring-match even bare 'w') so mood stays flat
    # and the 135-char reply can't be force-split into two bursts.
    n0 = len(world.sent_messages())
    m1 = await _say(world, world.alice, _mention(world) + " gimme the recap",
                         advance=90)
    d_short = _delay_of(world, m1, n0)          # "nice" -> typing floor 2s

    n1 = len(world.sent_messages())
    m2 = await _say(world, world.alice, _mention(world) + " gimme the recap",
                         advance=120)
    d_long = _delay_of(world, m2, n1)           # 135ch -> typing pinned at 12s

    assert d_long > d_short + 5.0, \
        f"long burst typing {d_long:.1f}s vs short {d_short:.1f}s"


async def test_every_send_is_typed_first(world, monkeypatch):
    """No message leaves without a typing beat first — every send has a
    typing() call paired with it."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, "yo", "heyy", "hmm ok")
    ch = world.channel()
    for text in (" yo", " sup", " hey"):
        await _say(world, world.alice, _mention(world) + text, advance=90)
    assert ch.typing_calls >= len(world.sent_messages())


# ─────────────────────────────────────────────────────────────────────────────
# E2E: burst splitting
# ─────────────────────────────────────────────────────────────────────────────

async def test_long_reply_sends_as_multiple_bursts(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, BURST_REPLY)
    n0 = len(world.sent())
    await _say(world, world.alice,
                    _mention(world) + " tell me about the sketch saga",
                    advance=150)
    new_sends = world.sent()[n0:]
    assert len(new_sends) >= 3, \
        f">300-char reply should split into >=3 bursts, got {new_sends}"
    # bursts are separate sends with distinct content
    assert len({s for s in new_sends}) == len(new_sends)


async def test_short_reply_stays_single_burst(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, "yo", "aite bet")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)
    n0 = len(world.sent())
    await _say(world, world.alice, _mention(world) + " still up?", advance=90)
    assert len(world.sent()) - n0 == 1


# ─────────────────────────────────────────────────────────────────────────────
# E2E: near-repeat regeneration
# ─────────────────────────────────────────────────────────────────────────────

async def test_near_repeat_reply_regenerates_once(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    r1 = "nah the whole thing was pretty mid honestly"
    # shares the 4-word run 'whole thing pretty mid' with r1 but a low
    # Jaccard + different prefix -> passes is_duplicate, trips near-repeat
    r2 = "yeah the whole thing pretty mid was kinda rough ngl"
    r3 = "idk it grew on me a bit"
    _script_replies(world, r1, r2, r3)

    await _say(world, world.alice, _mention(world) + " did u see it",
                    advance=90)
    assert not world.llm_calls("already said this"), \
        "anti-repeat retry fired on the first reply (history was empty)"
    assert any("pretty mid" in s for s in world.sent())

    n0 = len(world.sent_messages())
    await _say(world, world.alice, _mention(world) + " be honest again",
                    advance=90)

    nudges = world.llm_calls("already said this")
    assert len(nudges) == 1, \
        f"expected exactly one anti-repeat regen call, got {len(nudges)}"
    # The regenerated candidate (r3) was sent; the near-duplicate (r2)
    # never reached the channel.
    assert any("grew on me" in s for s in world.sent()[n0:]), world.sent()
    assert all("kinda rough" not in s for s in world.sent())


async def test_verbatim_repeat_ambient_drops_silently(world, monkeypatch):
    """Identical to the last sent reply and NOT directly addressed ->
    is_duplicate drops it silently before the near-repeat regen path.
    ('directly addressed' includes anyone inside the 240s convo window,
    so the trigger comes from a fresh user the bot has no thread with.)"""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    bob = world.person("bob")
    _script_replies(world, "same same same", "same same same")
    await _say(world, world.alice, _mention(world) + " first take", advance=90)

    n0 = len(world.sent())
    # bob greets — earns a greeting reply, but the verbatim repeat of
    # alice's answer drops silently (no regen, no canned ack).
    await _say(world, bob, "hi", advance=60)
    assert len(world.sent()) == n0, "ambient verbatim repeat was sent anyway"
    assert not world.llm_calls("already said this"), \
        "silent drop shouldn't reach the regen path"


async def test_verbatim_repeat_directed_gets_canned_ack(world, monkeypatch):
    """A directed verbatim repeat can't stay silent AND can't echo — the
    regen fires (anti-repeat nudge) and, when the regen is also a repeat,
    the reply falls back to a 'u already asked me that' style ack."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, "same same same", "same same same")
    await _say(world, world.alice, _mention(world) + " first take", advance=90)

    n0 = len(world.sent_messages())
    await _say(world, world.alice, _mention(world) + " second take", advance=90)
    fresh = world.sent_messages()[n0:]
    assert fresh, "directed message can't stay silent"
    sent_text = fresh[0].content
    assert sent_text != "same same same", "verbatim echo went out anyway"
    assert world.llm_calls("already said this"), \
        "directed verbatim repeat should trigger the anti-repeat regen"
    assert any(k in sent_text for k in (
        "same as before", "already asked", "same thing",
        "just said", "nothing new", "still the same")), \
        f"expected a 'said twice' canned ack, got {sent_text!r}"


# ─────────────────────────────────────────────────────────────────────────────
# E2E: speed multiplier commands
# ─────────────────────────────────────────────────────────────────────────────

async def test_speed_commands_owner_gated_and_scale_delay(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    owner = world.person("boss", user_id=999000999)   # OWNER_IDS in conftest
    LONG = _mention(world) + " " + LONG_PARAGRAPH

    # Non-owner: detected but refused — multiplier untouched.
    await _say(world, world.alice, _mention(world) + " reply faster",
                    advance=90)
    assert world.client._speed_multiplier == 1.0

    # Baseline delay at 1.0x.
    n0 = len(world.sent_messages())
    m_normal = await _say(world, world.alice, LONG, advance=150)
    d_normal = _delay_of(world, m_normal, n0)

    # Owner: 'reply faster' -> 0.5x — read/think/notif delays halve.
    await _say(world, owner, _mention(world) + " reply faster", advance=90)
    assert world.client._speed_multiplier == 0.5

    n1 = len(world.sent_messages())
    m_fast = await _say(world, world.alice, LONG, advance=150)
    d_fast = _delay_of(world, m_fast, n1)
    assert d_fast < d_normal - 4.0, \
        f"0.5x speed didn't shrink the delay ({d_fast:.1f} vs {d_normal:.1f})"

    # Owner: 'slow down' -> 1.5x.
    await _say(world, owner, _mention(world) + " slow down", advance=90)
    assert world.client._speed_multiplier == 1.5

    n2 = len(world.sent_messages())
    m_slow = await _say(world, world.alice, LONG, advance=150)
    d_slow = _delay_of(world, m_slow, n2)
    assert d_slow > d_fast + 5.0, \
        f"1.5x speed didn't inflate the delay ({d_slow:.1f} vs {d_fast:.1f})"
    assert d_slow > d_normal


# ─────────────────────────────────────────────────────────────────────────────
# E2E: generic-burst suppression
# ─────────────────────────────────────────────────────────────────────────────

async def test_repeat_generic_burst_is_suppressed(world, monkeypatch):
    """A 'wbu?'-style follow-up burst is fine once — firing another generic
    ping right after reads robotic, so the second one is dropped."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    replies = iter([
        ("still thinking about it tbh", "wbu?"),
        ("no clue honestly", "hbu?"),
    ])

    def on_req(system, user, kwargs):
        s = (system or "").lower()
        if '"reply"' in s and "output json" in s:
            r, b = next(replies, ("no clue honestly", "hbu?"))
            return json.dumps({"reply": r, "burst_reply": b,
                               "reaction": None, "new_status": None,
                               "new_mood": None, "emotion_intensity": 2,
                               "search_query": None})
        return None

    world.llm.on_request = on_req

    # Pin the 40% burst roll open — 0.2 < 0.4 sends it — while staying above
    # the 0.15 mood-nudge threshold so 'w'-carrying trigger words ("what",
    # "now") can't drift the mood mid-test and fork the burst count.
    orig_random = random.random
    random.random = lambda: 0.2
    try:
        n0 = len(world.sent())
        await _say(world, world.alice, _mention(world) + " what do u reckon",
                        advance=90)
        assert len(world.sent()) - n0 == 2, \
            "first generic burst should have gone out"
        assert "wbu?" in world.sent()

        n1 = len(world.sent())
        await _say(world, world.alice, _mention(world) + " and now?",
                        advance=90)
        assert len(world.sent()) - n1 == 1, \
            "second generic burst should have been suppressed"
        assert "hbu?" not in world.sent()
        assert world.sent().count("wbu?") == 1
    finally:
        random.random = orig_random


async def test_nongeneric_burst_still_sends(world, monkeypatch):
    """Suppression only targets bare 'wbu?'-style pings — a specific
    follow-up question still goes out."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    replies = iter([
        ("still thinking about it tbh", "wbu?"),
        ("no clue honestly", "u gonna watch the match tonight?"),
    ])

    def on_req(system, user, kwargs):
        s = (system or "").lower()
        if '"reply"' in s and "output json" in s:
            r, b = next(replies, ("ok", None))
            return json.dumps({"reply": r, "burst_reply": b,
                               "reaction": None, "new_status": None,
                               "new_mood": None, "emotion_intensity": 2,
                               "search_query": None})
        return None

    world.llm.on_request = on_req
    orig_random = random.random
    random.random = lambda: 0.2
    try:
        await _say(world, world.alice, _mention(world) + " what do u reckon",
                        advance=90)
        n1 = len(world.sent())
        await _say(world, world.alice, _mention(world) + " and now?",
                        advance=90)
        new = world.sent()[n1:]
        assert len(new) == 2, f"specific follow-up suppressed too: {new}"
        assert any("match tonight" in s for s in new)
    finally:
        random.random = orig_random


# ─────────────────────────────────────────────────────────────────────────────
# E2E: reply-quote behavior
# ─────────────────────────────────────────────────────────────────────────────

async def test_mention_reply_quotes_source(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    m = await _say(world, world.alice, _mention(world) + " yo", advance=90)
    sent = world.sent_messages()
    assert sent and sent[0].reference is not None
    assert sent[0].reference.message_id == m.id


async def test_question_reply_quotes_source(world, monkeypatch):
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, "yo", "probably around six")
    await _say(world, world.alice, _mention(world) + " yo", advance=90)
    n0 = len(world.sent_messages())
    # non-mention question in the sticky convo — is_question -> always quote
    m = await _say(world, world.alice, "do you know when the shop closes",
                        advance=90)
    sent = world.sent_messages()[n0:]
    assert sent, "question in sticky convo got no reply"
    assert sent[0].reference is not None
    assert sent[0].reference.message_id == m.id


async def test_ambient_reply_quote_is_a_tossup(world, monkeypatch):
    """Plain conversational replies quote ~60% — never-always and
    never-never would both be tells."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    world.seed(11)
    await _say(world, world.alice, _mention(world) + " yo", advance=90)

    total = 12
    quoted = 0
    for i in range(total):
        n0 = len(world.sent_messages())
        await _say(world, 
            world.alice,
            f"been messing with the new brushes all evening honestly {i}",
            advance=60)
        fresh = world.sent_messages()[n0:]
        assert fresh, f"ambient message {i} got no reply"
        if fresh[0].reference is not None:
            quoted += 1

    assert 0 < quoted < total, \
        f"ambient quote rate should be a mix — {quoted}/{total} quoted"


async def test_reply_to_bot_message_quotes(world, monkeypatch):
    """Replying to the bot's own message is a direct trigger — it quotes
    ~80% (a small de-roll keeps it casual). At least one quote must land."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    world.seed(5)
    await _say(world, world.alice, _mention(world) + " yo", advance=90)
    bot_msg = world.sent_messages()[-1]
    hits = 0
    for i in range(5):
        n0 = len(world.sent_messages())
        m = await _say(world, world.alice, f"wait tell me more about that {i}",
                            reply_to=bot_msg, advance=90)
        fresh = world.sent_messages()[n0:]
        assert fresh
        if fresh[0].reference is not None and \
                fresh[0].reference.message_id == m.id:
            hits += 1
    assert hits >= 1, "reply-to-bot never quoted the source message"


# ─────────────────────────────────────────────────────────────────────────────
# E2E: send bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

async def test_send_bookkeeping_tracks_bursts(world, monkeypatch):
    """record_send fires once per reply (not per burst); reply_history
    remembers the full pre-split text."""
    _neutralize_extras(world, monkeypatch)
    _calm(world)
    _script_replies(world, BURST_REPLY)
    ch_id = str(world.channel().id)
    await _say(world, world.alice,
                    _mention(world) + " tell me about the sketch saga",
                    advance=150)
    assert world.client.daily_count_ch.get(ch_id, 0) == 1
    hist = world.client.reply_history.get(ch_id)
    assert hist and len(hist[-1]) > 150, \
        "full pre-split reply text should land in reply_history"
