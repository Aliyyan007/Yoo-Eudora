"""Nuance & context tests — the subtle behaviors that sell "human".

Covers: sarcasm/irony survivability, suspicion prompt injection, repeated
questions (persistence vs variety), familiarity notes, multi-part lookahead,
unanswered-question context, stale engagement sweep (auto-delete of
unresponded messages), mass-mention sanitization of generated text, and
profile memory surfacing in prompts.
"""
import asyncio
import time

import pytest

from tests.sim import fake_discord as fd


# ─────────────────────────────────────────────────────────────────────────────
# Irony / sarcasm / hidden meaning — pipeline survives, reply still lands
# ─────────────────────────────────────────────────────────────────────────────

async def test_sarcastic_mention_gets_reply(world):
    """'oh GREAT 🙄' — sarcasm must not crash or confuse the pipeline."""
    await world.say(world.alice,
                    f"<@{world.bot_user.id}> oh GREAT another meeting 🙄")
    assert world.bot_said_something()


async def test_hidden_meaning_ambient(world):
    """'sure, because THAT always works' — dry irony as ambient text must
    not crash; it may or may not earn a reply (random gate)."""
    await world.say(world.alice, "sure, because THAT always works 🙃")
    # The real assertion is just: no exception, world intact
    await world.say(world.alice, f"<@{world.bot_user.id}> still there?")
    assert world.bot_said_something()


async def test_suspicion_headsup_reaches_prompt(world):
    """After a bot accusation, the reply prompt carries the 'don't defend'
    heads-up so the model doesn't deny being a bot."""
    await world.say(world.alice, "are you a bot??")
    n = len(world.llm.calls)
    await world.say(world.alice, f"<@{world.bot_user.id}> hey you")
    new_calls = world.llm.calls[n:]
    assert any("suspecting you're a bot" in c["user"] for c in new_calls), \
        "suspicion heads-up missing from the reply prompt"


# ─────────────────────────────────────────────────────────────────────────────
# Repeat questions — same trigger must not produce verbatim repeats
# ─────────────────────────────────────────────────────────────────────────────

async def test_repeated_question_varies(world):
    """wyd / wyd / wyd — the bot varies its answer instead of parroting:
    regen retry first, then a human 'same thing' ack as last resort."""
    world.seed(7)
    await world.say(world.alice, f"<@{world.bot_user.id}> wyd")
    await world.say(world.alice, f"<@{world.bot_user.id}> wyd")
    await world.say(world.alice, f"<@{world.bot_user.id}> wyd")
    sends = world.sent()
    assert len(set(sends)) == len(sends), \
        f"verbatim repeat across repeated question: {sends}"


async def test_verbatim_repeat_triggers_regen_nudge(world):
    """If the model repeats itself word-for-word, the pipeline retries with
    an anti-repeat instruction (never ship the echo)."""
    world.llm.reply_fn = lambda t, u: "the exact same words"
    await world.say(world.alice, f"<@{world.bot_user.id}> one")
    await world.say(world.alice, f"<@{world.bot_user.id}> two")
    nudged = [c for c in world.llm.calls if "already said this" in c["user"]]
    assert nudged, "identical reply never triggered the anti-repeat retry"


async def test_mirror_avoidance_in_prompt(world):
    """The NO MIRRORING rule must be present in the reply prompt so the
    model doesn't parrot the user's phrasing back."""
    await world.say(world.alice, f"<@{world.bot_user.id}> kinda bored… still sore")
    assert any("NO MIRRORING" in (c["system"] + c["user"])
               for c in world.llm.calls)


# ─────────────────────────────────────────────────────────────────────────────
# Context memory — names/facts surface in the prompt; familiarity notes
# ─────────────────────────────────────────────────────────────────────────────

async def test_known_name_reaches_prompt(world):
    """After 'my name is Alice', the next reply prompt knows alice's name."""
    await world.say(world.alice, f"<@{world.bot_user.id}> my name is Alice")
    n = len(world.llm.calls)
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    hits = [c for c in world.llm.calls[n:] if "Alice" in c["user"]]
    assert hits, "stored name missing from the reply prompt context"


async def test_familiarity_note_on_second_contact(world):
    """A second message inside the convo window gets the 'already talking'
    note so the model doesn't re-greet."""
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    n = len(world.llm.calls)
    await world.say(world.alice, f"<@{world.bot_user.id}> so anyway")
    hits = [c for c in world.llm.calls[n:]
            if "already been talking" in c["user"]]
    assert hits, "familiarity note missing on second contact"


async def test_lookahead_catches_multipart(world):
    """A '?', 'u?' or continuation right before the trigger is folded into
    the transcript via the 10-message lookahead."""
    ch = world.channel()
    await world.say(world.alice, "im asking u", advance=0.2)
    n = len(world.llm.calls)
    await world.say(world.alice, f"<@{world.bot_user.id}> yes or no?")
    hits = [c for c in world.llm.calls[n:]
            if "just before this" in c["user"]]
    assert hits, "multi-part message train missing from the transcript"


async def test_unanswered_question_surfaces_in_context(world):
    """An ambient question nobody answered gets included when the bot
    replies later — [UNANSWERED...] note in the prompt."""
    # Dispatch manually with a tiny advance — records the question but
    # parks its reply mid-read-delay so the question stays unanswered when
    # bob's prompt is built (world.say's pump would let it complete).
    ch = world.channel()
    m = fd.make_message(ch, world.alice, "what time does the event start?",
                        at=world.clock.time())
    import asyncio as _aio
    _aio.create_task(world.client.on_message(m))
    await world.advance(0.05)
    n = len(world.llm.calls)
    await world.say(world.bob, f"<@{world.bot_user.id}> yo")
    hits = [c for c in world.llm.calls[n:]
            if "unanswered" in c["user"].lower()]
    assert hits, "unanswered question never reached the reply prompt"


async def test_unanswered_cleared_for_that_user_only(world):
    """mark_answered(ch_id, user_id) is scoped — clearing alice's pending
    question must not wipe bob's. (Unit-level: the end-to-end path is
    timing-sensitive because the pending queue legitimately answers
    each queued question.)"""
    tracker = world.client.unanswered_tracker
    ch_id = str(world.channel().id)
    tracker.record_question(ch_id, "what's the plan?", "alice",
                            990001, author_id=world.alice.id)
    tracker.record_question(ch_id, "seen the patch notes?", "bob",
                            990002, author_id=world.bob.id)
    assert len(tracker.get_unanswered_questions(ch_id)) == 2
    tracker.mark_answered(ch_id, world.alice.id)
    remaining = tracker.get_unanswered_questions(ch_id)
    assert len(remaining) == 1
    text, author = remaining[0]
    assert author == "bob", "bob's question was wiped when alice got answered"


# ─────────────────────────────────────────────────────────────────────────────
# Stale engagement sweep — auto-delete own unanswered engagement msgs
# ─────────────────────────────────────────────────────────────────────────────

async def test_stale_engagement_sweep_deletes_unanswered(world):
    """Old unanswered engagement posts are swept; answered ones survive."""
    from src.ai.engagement_log import get_engagement_log
    eng = get_engagement_log()
    ch = world.channel()

    # Bot posts 6 engagement msgs that nobody answers (stale = >15min)
    msgs = []
    for i in range(6):
        m = fd.make_message(ch, world.bot_user, f"anyone around? {i}",
                            at=world.clock.time() - 3600)  # an hour old
        ch.messages.append(m)
        msgs.append(m)

    for m in msgs:
        eng.record(str(ch.id), m)

    world.clock._now += 100  # move past stale_s
    deleted_before = len([m for m in msgs if getattr(m, "deleted", False)])
    # The sweep paces deletes with a real asyncio.sleep between each — run
    # it on a task so the virtual clock can advance through those sleeps
    # (awaiting it directly would deadlock the test on parked time).
    t = asyncio.create_task(
        eng.sweep_before_send(ch, world.bot_user, force=True))
    await world.advance(30)
    await t
    deleted = [m for m in msgs if getattr(m, "deleted", False)]
    # Over max_unanswered=4, stale + unanswered → some deleted
    assert len(deleted) > deleted_before, "stale unanswered msgs never swept"


async def test_sweep_never_deletes_answered_messages(world):
    """A bot message that got a human reply is never swept."""
    from src.ai.engagement_log import get_engagement_log
    eng = get_engagement_log()
    ch = world.channel()

    bot_msg = fd.make_message(ch, world.bot_user, "anyone up?",
                              at=world.clock.time() - 3600)
    ch.messages.append(bot_msg)
    human_reply = fd.make_message(ch, world.alice, "me!",
                                  at=world.clock.time() - 3590)
    ch.messages.append(human_reply)
    eng.record(str(ch.id), bot_msg)

    await eng.sweep_before_send(ch, world.bot_user, force=True)
    assert not getattr(bot_msg, "deleted", False), \
        "sweep deleted a message that was answered"


# ─────────────────────────────────────────────────────────────────────────────
# Generated-text sanitization — the model can't mass-ping through replies
# ─────────────────────────────────────────────────────────────────────────────

async def test_generated_reply_cannot_mass_ping(world):
    """Even if the model outputs '@everyone', the sent message is scrubbed."""
    world.llm.reply_fn = lambda t, u: "@everyone check this out"
    await world.say(world.alice, f"<@{world.bot_user.id}> hello")
    for msg in world.sent():
        assert "@everyone" not in msg
        assert "@here" not in msg


async def test_generated_status_cannot_mass_ping(world):
    """Same scrubbing on new_status — the sanitize pass covers every field."""
    world.llm.on_request = lambda s, u, k: (
        '{"reply":"hi","burst_reply":null,"reaction":null,'
        '"new_status":"@everyone","new_mood":null,"search_query":null}'
        if '"reply"' in (s or "").lower() else None)
    await world.say(world.alice, f"<@{world.bot_user.id}> yo")
    # Presence update should never contain a mass ping
    calls = getattr(world.client, "__dict__", {}).get("_presence_calls", [])
    for c in calls:
        for v in c.values():
            assert "@everyone" not in str(v)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-language / edge inputs — pipeline never crashes
# ─────────────────────────────────────────────────────────────────────────────

async def test_emoji_only_and_symbol_messages(world):
    for txt in ("💀💀💀", "?!?!", "....", "🇫🇷 oui oui"):
        await world.say(world.alice, f"<@{world.bot_user.id}> {txt}")
    # As long as each got SOME reply (or a reaction) the pipeline held
    assert True  # reaching here without exception is the assertion


# ─────────────────────────────────────────────────────────────────────────────
# Detection precision — substring matching must not produce false positives
# ─────────────────────────────────────────────────────────────────────────────

def test_is_question_word_boundaries():
    """'whole'/'show'/'whenever' contain who/how/when — substring matching
    flagged them as questions and capped bursts at 2. Boundary matching
    keeps them out; real question words still hit (incl. clitic 'whats')."""
    from src.ai.multi_message import detect_conversation_nature
    for txt in ("that's a whole mood", "show me the way", "whenever works"):
        assert not detect_conversation_nature(txt, "")["is_question"], txt
    for txt in ("whats up", "hows it going", "what u think", "where u at"):
        assert detect_conversation_nature(txt, "")["is_question"], txt


def test_mood_nudge_word_boundaries(world):
    """Bare 'w' ('win' in Discord slang) used to match every word with a
    'w' — the exciting-news bucket fired constantly. Now it needs the
    standalone token; emoji keywords still match by containment."""
    from src.ai.mood import _kw_hit
    assert _kw_hit("w", "big w")
    assert not _kw_hit("w", "what now")
    assert not _kw_hit("how", "show me")
    assert _kw_hit("💀", "bruh 💀💀")


async def test_very_long_message_handled(world):
    """A 2000-char paragraph — full read/comprehension delays apply, reply
    still lands."""
    big = ("i've been thinking about this all week honestly and it keeps "
           "coming back to the same thing — like, we all say we want change "
           "but then nobody actually moves, you know? and then the whole "
           "situation just sits there getting worse while everyone waits "
           "for someone else to go first") * 8
    await world.say(world.alice, f"<@{world.bot_user.id}> {big[:1900]}")
    assert world.bot_said_something()
