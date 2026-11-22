"""Naturalness guard tests — deterministic turn-shape enforcement.

Pins the bot-tells observed in live logs: the same edge-slang twice
running ('hey bruv' → 'how's it going bruv'), echoed openers
('just vibes' → 'just a mix'), question-on-question chains
('what's up?' → 'what's on your playlist?'), and echo-of-own-question
misreads ('playlist?' answered with the bot's own playlist).

These are fixed in code (reply.enforce_turn_shape / question_chain /
is_question_echo) because prompt-level rules get ignored under drift.
"""
from src.ai import reply as ai_reply


# ── unit: helpers ────────────────────────────────────────────────────────────

def test_vocative_repeat_stripped():
    assert ai_reply.enforce_turn_shape(
        "how's it going bruv", ["hey bruv"]) == "how's it going"


def test_vocative_different_kept():
    # a vocative that wasn't used last turn is fine to keep
    assert ai_reply.enforce_turn_shape(
        "how's it going mate", ["hey bruv"]) == "how's it going mate"


def test_mid_sentence_vocative_is_content():
    # 'man' mid-sentence is content, not a tic — untouched
    t = "the man said it himself"
    assert ai_reply.enforce_turn_shape(t, ["man that hurt"]) == t


def test_opener_repeat_stripped():
    assert ai_reply.enforce_turn_shape(
        "just a mix of lo-fi beats", ["just vibes"]) == "a mix of lo-fi beats"


def test_opener_nonfiller_kept():
    # 'music' repeats but isn't a strippable filler opener — left alone
    t = "music is carrying me rn"
    assert ai_reply.enforce_turn_shape(t, ["music was mid today"]) == t


def test_question_chain_detection():
    assert ai_reply.question_chain(
        "what's on your playlist?", "cool, what's up?")
    # no '?' but still interrogative syntax
    assert ai_reply.question_chain("how's it going", "wyd?")
    # declarative openers aren't question-turns
    assert not ai_reply.question_chain("yeah same", "what's up?")
    assert not ai_reply.question_chain("innit though", "hru?")
    # new turn isn't a question → no chain
    assert not ai_reply.question_chain("nah same", "what's up?")
    # the live-log shape: a statement ending in an interrogative clause
    # still counts as a question-turn
    assert ai_reply.question_chain(
        "cool, what's up?", "tired but hanging in there, how's it going bruv")


def test_question_chain_keeps_statement():
    out = ai_reply.enforce_turn_shape(
        "just vibes, what's on your playlist?", ["cool, what's up?"])
    assert "?" not in out
    assert "vibes" in out


def test_pure_question_left_for_regen():
    # strip leaves nothing — text stays so the caller can regenerate,
    # and the chain is still detectable
    t = "what's on your playlist?"
    out = ai_reply.enforce_turn_shape(t, ["cool, what's up?"])
    assert ai_reply.question_chain(out, "cool, what's up?")


def test_echo_of_own_question():
    assert ai_reply.is_question_echo(
        "playlist?", ["just vibes, what's on your playlist?"])
    assert ai_reply.is_question_echo(
        "what playlist", ["what's on your playlist?"])
    # 'wdym' isn't an echo — the clarify rule already covers it
    assert not ai_reply.is_question_echo(
        "wdym", ["what's on your playlist?"])
    # a genuinely new question adds new words → not an echo
    assert not ai_reply.is_question_echo(
        "how are you feeling", ["what's on your playlist?"])
    # last own message wasn't a question → no echo possible
    assert not ai_reply.is_question_echo("playlist?", ["just vibes"])


# ── e2e: real client, scripted llm, live-log replay ───────────────────────────

async def test_no_repeated_edge_vocative(world):
    """'hey bruv' then 'how's it going bruv' — the second send must drop
    the vocative, not repeat the tic."""
    calls = []
    def fn(t, u):
        calls.append(1)
        return ["hey bruv", "how's it going bruv"][min(len(calls) - 1, 1)]
    world.llm.reply_fn = fn
    await world.say(world.alice, f"<@{world.bot_user.id}> hiii")
    await world.say(world.alice, f"<@{world.bot_user.id}> hru?")
    sends = world.sent()
    assert len(sends) >= 2
    assert "bruv" not in sends[-1].lower()


async def test_no_question_on_question_chain(world):
    """A question reply followed by another question — the second send
    must come out as a statement (strip, regen, or fallback)."""
    calls = []
    def fn(t, u):
        calls.append(1)
        # call 3+ is the regen path — hand it a statement
        return ["cool, what's up?",
                "what's on your playlist?",
                "just vibes tbh"][min(len(calls) - 1, 2)]
    world.llm.reply_fn = fn
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    await world.say(world.alice, f"<@{world.bot_user.id}> nothing much just chilling")
    sends = world.sent()
    assert len(sends) >= 2
    assert "?" not in sends[-1]


async def test_stubborn_question_chain_falls_back(world):
    """If even the regen comes back interrogative, the last-resort forces
    a statement — the chain can never reach the channel."""
    calls = []
    def fn(t, u):
        calls.append(1)
        return ["what's up?",
                "what's on your playlist?",
                "and what about films?"][min(len(calls) - 1, 2)]
    world.llm.reply_fn = fn
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    await world.say(world.alice, f"<@{world.bot_user.id}> hey again")
    sends = world.sent()
    assert len(sends) >= 2
    assert "?" not in sends[-1]


async def test_echo_gets_clarify_note(world):
    """'playlist?' right after our playlist question must carry the
    echo-clarify note into the reply prompt."""
    calls = []
    def fn(t, u):
        calls.append(1)
        return ["just vibes, what's on your playlist?",
                "lofi mix i guess"][min(len(calls) - 1, 1)]
    world.llm.reply_fn = fn
    await world.say(world.alice, f"<@{world.bot_user.id}> what u listening to")
    await world.say(world.alice, "playlist?")
    echo_calls = [c for c in world.llm.calls
                  if "echoing YOUR last question" in (c.get("user") or "")]
    assert echo_calls, "echo trigger didn't get the clarify note in the prompt"
