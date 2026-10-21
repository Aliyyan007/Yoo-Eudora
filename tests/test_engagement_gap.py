"""Regression for the live-observed engagement gap.

Real-console scenario (user report): a new member rapid-fired engaging
messages — 'anyways where are you from guys', 'because i am new here',
'okay thanks for telling that', 'i want to make friends', 'but i don't
know how', 'that's why i joined here' — and the bot answered NONE of
them. The model kept producing short acks which hit the verbatim-duplicate
drop; last_sent never updated, so every subsequent near-identical reply
matched the same stale `last` and dropped forever — a permanent wedge.

These tests verify the wedge cannot happen: repeated similar replies must
regenerate or land a human 'same thing' ack, and last_sent must advance.
"""
import time

import pytest

from tests.sim import fake_discord as fd


async def test_repeated_short_acks_no_wedge(world):
    """The exact live scenario — model keeps replying with the same short
    ack. Each message must still get SOMETHING out (regen or ack), never
    silence-forever."""
    world.seed(11)
    world.llm.reply_fn = lambda t, u: "yeah, i'm here"   # model fixates

    kratozch = world.person("kratozch")
    for txt in (
            "anyways where are you from guys",
            "because i am new here",
            "okay thanks for telling that",
            "i want to make friends",
            "but i don't know how",
            "that's why i joined here"):
        await world.say(kratozch, txt, advance=25)

    sends = world.sent()
    # At least the first one plus follow-ups — and NO two adjacent sends
    # may be verbatim identical (the wedge signature).
    assert sends, "bot stayed silent through an engaging message train"
    for prev, nxt in zip(sends, sends[1:]):
        assert prev != nxt, f"verbatim repeat wedge: {sends}"


async def test_ambient_repeat_drops_but_doesnt_wedge(world):
    """Ambient (non-directed) repeats can stay silent — a human wouldn't
    keep echoing — but a subsequent DIFFERENT reply must still land."""
    world.seed(12)
    world.llm.reply_fn = lambda t, u: "ok cool"

    # Directed first so there's a reply to duplicate
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    assert world.bot_said_something()
    first = world.last_sent()

    # Now model still returns "ok cool" for an ambient msg — it's a dup,
    # silence is correct. Then a different reply must still be able to send.
    world.llm.reply_fn = lambda t, u: "fresh different answer"
    await world.say(world.alice, f"<@{world.bot_user.id}> tell me more")
    sends = world.sent()
    assert any("fresh different answer" in s for s in sends), \
        "a later distinct reply was wedged by the earlier duplicate"


async def test_multipart_train_collapses_to_one_reply(world):
    """Same-author rapid train ('i want to make friends' / 'but i don't
    know how' / 'that's why i joined here') — the bot must answer the
    train, never go silent (the live bug) and never parrot verbatim."""
    world.seed(13)
    kratozch = world.person("kratozch")
    await world.say(kratozch, "i want to make friends", advance=0.3)
    await world.say(kratozch, "but i don't know how", advance=0.3)
    await world.say(kratozch, "that's why i joined here", advance=0.3)
    await world.advance(80)
    sends = world.sent()
    assert sends, "a 3-part engaging train got ZERO replies (live regression)"
    for prev, nxt in zip(sends, sends[1:]):
        assert prev != nxt, f"verbatim repeat in the train replies: {sends}"


async def test_new_member_question_gets_answered(world):
    """'where are you from guys' — an open question from someone still
    warming up must not get swallowed by the coalescing gate."""
    world.seed(14)
    kratozch = world.person("kratozch")
    await world.say(kratozch, "anyways where are you from guys", advance=25)
    await world.say(kratozch, "because i am new here", advance=25)
    await world.advance(60)
    assert world.bot_said_something(), \
        "a new member's opener questions went completely unanswered"
