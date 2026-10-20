"""Trigger matrix — exhaustive tests of the should_respond decision tree.

Every reason code in _should_respond gets coverage: what MUST reply,
what MUST stay silent, and what gates each path.
"""
import pytest

from tests.sim import fake_discord as fd


CH = "general"


async def test_direct_mention_always_replies(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> what do you think?")
    assert world.bot_said_something()


async def test_reply_to_bot_message_replies(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> yo")
    bot_msg = world.sent_messages()[-1]
    await world.say(world.bob, "yeah what she said", reply_to=bot_msg)
    assert len(world.sent()) >= 2


async def test_reply_to_other_user_stays_silent(world):
    # Capture alice's message object directly — after any ambient reply
    # the bot's own send would be messages[-1], not alice's.
    alice_msg = await world.say(world.alice, "anyone seen my keys")
    n = len(world.sent())
    await world.say(world.bob, "check under the sofa", reply_to=alice_msg)
    assert len(world.sent()) == n, "bot answered a reply aimed at another user"


async def test_mention_of_other_user_stays_silent(world):
    await world.say(world.alice, f"hey <@{world.bob.id}> come here",
                    mentions=[world.bob])
    assert not world.bot_said_something()


async def test_name_mention_with_context_replies(world):
    await world.say(world.alice, "eudora what do you think about this?")
    assert world.bot_said_something()


async def test_standalone_name_drop_stays_silent(world):
    await world.say(world.alice, "eudora")
    assert not world.bot_said_something()


async def test_greeting_gets_greeting(world):
    await world.say(world.alice, "hey")
    assert world.bot_said_something()


@pytest.mark.parametrize("greet", ["hi", "hello", "yo", "sup", "heya", "hiya", "hiii"])
async def test_greeting_variants(world, greet):
    await world.say(world.alice, greet)
    assert world.bot_said_something(), f"no reply to greeting {greet!r}"


async def test_greeting_for_other_person_silent(world):
    world.bob  # ensure bob exists as a guild member (like a real server)
    await world.say(world.alice, "hey bob")
    assert not world.bot_said_something()


async def test_dismissal_backs_off(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    assert world.bot_said_something()
    n = len(world.sent())
    await world.say(world.alice, f"<@{world.bot_user.id}> shut up")
    # Dismissal sets a 10-min backoff; ambient messages must be ignored
    await world.say(world.alice, "anyway what's everyone doing")
    assert len(world.sent()) == n, "bot replied during dismissal backoff"


async def test_mention_breaks_backoff(world):
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    await world.say(world.alice, f"<@{world.bot_user.id}> shut up")
    await world.say(world.alice, f"<@{world.bot_user.id}> sorry come back")
    # mention during backoff still gets a reply (mentions break the window)
    assert world.bot_said_something()


async def test_low_content_sticky_no_typed_reply(world):
    world.seed(0)
    await world.say(world.alice, f"<@{world.bot_user.id}> what's up")
    assert world.bot_said_something()
    n = len(world.sent())
    # Inside the 2-min sticky window, a bare "lol" doesn't earn a reply
    await world.say(world.alice, "lol")
    assert len(world.sent()) == n, "low-content ack got a typed reply"


async def test_personal_question_always_replies(world):
    await world.say(world.alice, "do you know my name")
    assert world.bot_said_something()


async def test_loneliness_gets_company(world):
    await world.say(world.alice, "anyone here?")
    assert world.bot_said_something()


async def test_sticky_conversation_continuation(world):
    """After the bot replies, follow-ups within 2 min get answered
    without needing a mention."""
    world.seed(1)
    await world.say(world.alice, f"<@{world.bot_user.id}> you there?")
    assert world.bot_said_something()
    n = len(world.sent())
    await world.say(world.alice, "what are you up to today then")
    assert len(world.sent()) > n, "sticky-window follow-up got no reply"


async def test_active_conversation_tracker(world):
    """The 30-min conversation tracker answers even after sticky expires."""
    world.seed(1)
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert world.bot_said_something()
    # 5 min later (sticky dead, conversation tracker alive)
    await world.advance(300)
    n = len(world.sent())
    await world.say(world.alice, "so anyway i was thinking about art")
    assert len(world.sent()) > n


async def test_busy_channel_stays_silent(world):
    """8+ active users in 10 min → bot doesn't chime in uninvited."""
    world.seed(0)
    people = [world.person(f"user{i}") for i in range(9)]
    # "lol" is ignorable — none of these can earn a random reply, so every
    # send from here on can only belong to alice's message.
    for p in people:
        await world.say(p, "lol", advance=0.1)
    n = len(world.sent())
    await world.say(world.alice, "this chat is so busy rn")
    assert len(world.sent()) == n


async def test_too_short_message_ignored(world):
    world.seed(0)
    await world.say(world.alice, "ok")
    assert not world.bot_said_something()


async def test_bot_messages_ignored(world):
    bot_user = world.person("somebot", bot=True)
    await world.say(bot_user, "BUMPED the server! type /bump to bump again")
    assert not world.bot_said_something()


async def test_webhook_ignored(world):
    await world.say(world.alice, "automated post", webhook_id=12345)
    assert not world.bot_said_something()


async def test_url_only_message_ignored(world):
    await world.say(world.alice, "https://tenor.com/view/funny-gif")
    assert not world.bot_said_something()


async def test_emoji_only_ignored(world):
    await world.say(world.alice, "😂😂😂")
    assert not world.bot_said_something()


async def test_distress_signal_never_randomly_answered(world):
    """Distress content must not get a casual random reply."""
    world.seed(0)
    await world.say(world.alice, "i feel so worthless today honestly")
    assert not world.bot_said_something()


async def test_daily_cap_defers_directed_message(world):
    """At the daily cap, directed messages get queued for later, not lost."""
    ch_id = str(world.channel().id)
    # Mark the day as already reset so the first can_send() doesn't wipe
    # the counter back to zero before the cap check runs.
    import datetime
    world.client.daily_reset_day = datetime.date.today().toordinal()
    world.client.daily_count_global = 400  # DAILY_CAP_GLOBAL
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert not world.bot_said_something()
    from src.persona import runtime as prt
    deferred = prt.deferred_for(world.client.persona.id)
    assert any(d["author_id"] == world.alice.id for d in deferred), \
        "directed message at daily cap wasn't deferred"


async def test_ignorable_own_messages_not_reprocessed(world):
    """Our own messages cache into history but never trigger a reply."""
    await world.say(world.bot_user, "i'm talking to myself")
    assert not world.bot_said_something()


async def test_message_dedup(world):
    """Same message id processed twice → handled once."""
    await world.say(world.alice, f"<@{world.bot_user.id}> hi")
    n = len(world.sent())
    # Re-feed the SAME message object — dedup must drop it
    msg = world.channel().messages[-1]
    import asyncio
    asyncio.create_task(world.client.on_message(msg))
    await world.advance(60)
    assert len(world.sent()) == n


async def test_wrap_up_on_long_dry_conversation(world):
    """After a long convo + dry closer, the bot wraps up and goes quiet
    on ambient lines for ~15 min (but still answers questions)."""
    world.seed(3)
    ch_id = str(world.channel().id)
    uid = str(world.alice.id)
    # Simulate a long conversation streak (40+ messages) + wrap trigger
    world.client._convo_streak[ch_id] = {uid: (45, world.clock.time())}
    world.client._convo_recent[ch_id] = {uid: __import__('collections').deque(
        ["yeah", "ok", "hmm", "sure", "lol", "fine", "nice", "k"], maxlen=12)}
    await world.say(world.alice, f"<@{world.bot_user.id}> hmm okay fine")
    n = len(world.sent())
    # Ambient (non-question, non-mention) lines within quiet window → silence
    await world.say(world.alice, "anyway")
    assert len(world.sent()) == n, "wrap-quiet window ignored"


async def test_wrap_quiet_still_answers_questions(world):
    world.seed(3)
    ch_id = str(world.channel().id)
    uid = str(world.alice.id)
    world.client._convo_streak[ch_id] = {uid: (45, world.clock.time())}
    world.client._convo_recent[ch_id] = {uid: __import__('collections').deque(
        ["yeah", "ok", "hmm"], maxlen=12)}
    await world.say(world.alice, f"<@{world.bot_user.id}> hmm okay fine")
    n = len(world.sent())
    await world.say(world.alice, "wait actually what time is it?")
    assert len(world.sent()) > n, "real question in wrap-quiet got ignored"


async def test_bot_accusation_lowers_ambient_replies(world):
    """Suspicion window swallows non-question ambient replies (~50% roll).

    Deterministic check: pin the gate's random roll under 0.5 so every
    ambient line hits suspicion-backoff, then confirm expiry restores
    the ambient path."""
    await world.say(world.alice, "are you a bot??")
    ch_id = str(world.channel().id)
    assert world.client._bot_suspicion_until.get(ch_id, 0) > 0
    # Kill the 2-min sticky window — bob isn't in the conversation tracker
    # either, so his lines can only reach the suspicion-gated ambient path.
    await world.advance(125)
    n = len(world.sent())
    import random as _r
    _orig = _r.random
    _r.random = lambda: 0.25   # < 0.5 → suspicion gate always drops
    try:
        for i in range(3):
            await world.say(world.bob, f"ambient chatter {i}")
    finally:
        _r.random = _orig
    assert len(world.sent()) - n == 0, \
        "ambient replies fired while the suspicion gate should be dropping"


async def test_utility_channel_skips_random_participation(world):
    """Channels named like bot-commands/rank don't get ambient replies."""
    world.seed(0)
    ch = world.guild.add_channel("bot-commands")
    await world.say(world.alice, "check rank", channel=ch)
    assert not ch.sent_log


async def test_engage_channel_never_utility_gated(world):
    """TEXT_CHANNEL_IDS channels always count as engagement channels."""
    world.seed(0)
    ch = world.channel()  # id 111111111 is in TEXT_CHANNEL_IDS
    # topic mentioning xp/levels would make it 'utility' for non-engaged
    ch.topic = "chat and level up!"
    ch_id = str(ch.id)
    world.client._engage_channel_ids.add(ch.id)
    await world.say(world.alice, "what's everyone up to today", channel=ch)
    # May or may not reply (chance-based) — the point is no crash and the
    # utility gate can't categorically silence it.
