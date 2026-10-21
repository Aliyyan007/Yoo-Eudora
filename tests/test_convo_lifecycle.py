"""Conversation lifecycle tests — the state machines that decide when a
conversation opens, stays open, winds down, and re-opens.

Covered systems (src/discord_client.py `_should_respond` gate +
`_handle_reply` post-send bookkeeping):

* wrap-up wind-down   — long convo + dry closer -> "_is_wrap" reply
                        (questions stripped), `_wrap_quiet` armed, streak
                        reset, occasional AFK window.
* dismissal backoff   — "shut up" directed at the bot -> 600s channel
                        backoff; ambient dropped, a mention breaks it.
* sticky decay        — every reply opens a 120s sticky window; inside it
                        filler is dropped (2s cooldown / low-content) and
                        substance answered; after it, ambient gating.
* conversation tracker — replying tracks THAT user for 30 min
                        ("active-conversation"), independent per user.
* AFK window          — non-mentions dropped ("AFK — ignoring"), a
                        mention breaks it.
* cross-persona       — messages aimed at an OFFLINE persona are recorded
                        via prt.add_pending (mention / name hits).
* unanswered tracker  — questions are tracked per author; replying to
                        alice clears only ALICE's questions.

Determinism notes
-----------------
* `world.say(..., advance=60)` covers a full reply cycle; gate checks via
  `client._should_respond(msg)` are white-box where timing makes the
  end-to-end path nondeterministic.
* `random.random` is pinned (monkeypatched) where a probability roll is
  the behavior under test. 0.99 disables every roll; 0.20 fires the 25%
  AFK roll and the 35% silent-reaction roll WITHOUT reaching the sticker
  (~<=0.16) or GIF (~<=0.11) paths — those do real aiohttp calls and must
  never fire under test.
"""
from __future__ import annotations

import random

from tests.sim import fake_discord as fd


def _ch_id(world) -> str:
    return str(world.channel().id)


async def _say_reply(world, person, content, **kw) -> fd.FakeMessage:
    """world.say() for messages a reply IS expected for, hardened against
    the virtual-clock race: a reply task that parks a virtual sleep just
    after advance()'s settle window gets stranded — its send lands on the
    next pump, exactly like a beat-late human reply. For drops, plain
    world.say() is enough (there's nothing to wait for)."""
    n = len(world.sent())
    msg = await world.say(person, content, **kw)
    for _ in range(4):
        if len(world.sent()) > n:
            break
        await world.advance(15)
    return msg


def _msg(world, author, content, **kw) -> fd.FakeMessage:
    """A fabricated message for direct gate checks — `_should_respond` is
    deterministic given client state, so this exercises the real gate
    without paying for a full reply cycle."""
    return fd.make_message(world.channel(), author, content,
                           at=world.clock.time(), **kw)


def _seed_long_convo(world, user, streak: int = 45):
    """Stand in for a long-running conversation: `streak` of THEIR messages
    already sent (past the _WRAP_STREAK_MIN=40 wind-down floor) plus a live
    sticky window so the closer routes through the wrap-up gate."""
    ch_id = _ch_id(world)
    world.client._convo_streak[ch_id] = {
        str(user.id): (streak, world.clock.time())}
    world.client.sticky_until[ch_id] = world.clock.time() + 120


# ─────────────────────────────────────────────────────────────────────────────
# Wrap-up wind-down — the gate side
# ─────────────────────────────────────────────────────────────────────────────

async def test_closer_in_long_convo_hits_wrap_gate(world):
    """streak >= threshold + dry closer + live sticky -> (True, 'wrap-up')."""
    _seed_long_convo(world, world.alice)
    ok, reason = world.client._should_respond(
        _msg(world, world.alice, "hmm okay fine"))
    assert ok and reason == "wrap-up"


async def test_closer_without_streak_is_just_sticky(world):
    """Same closer early in a convo is NOT a wrap — control case."""
    ch_id = _ch_id(world)
    world.client.sticky_until[ch_id] = world.clock.time() + 120
    world.client._convo_streak[ch_id] = {
        str(world.alice.id): (5, world.clock.time())}
    ok, reason = world.client._should_respond(
        _msg(world, world.alice, "hmm okay fine"))
    assert ok and reason.startswith("sticky-convo"), reason


async def test_non_closer_in_long_convo_stays_sticky(world):
    """A long convo alone doesn't wind down — it takes a DRY closer."""
    _seed_long_convo(world, world.alice)
    ok, reason = world.client._should_respond(
        _msg(world, world.alice, "wait that actually makes a lot of sense now"))
    assert ok and reason.startswith("sticky-convo"), reason


# ─────────────────────────────────────────────────────────────────────────────
# Wrap-up — end to end
# ─────────────────────────────────────────────────────────────────────────────

async def test_wrap_reply_goes_out_and_arms_quiet_window(world):
    """Closer -> wrap ack sent, _wrap_quiet armed for ~15min, streak reset."""
    world.seed(0)
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    assert world.bot_said_something(), "wrap-up closer got no reply"
    ch_id, uid = _ch_id(world), str(world.alice.id)
    quiet_until = world.client._wrap_quiet.get(ch_id, {}).get(uid, 0)
    assert quiet_until > world.clock.time(), "wrap didn't arm the quiet window"
    assert quiet_until <= world.clock.time() + 900
    assert uid not in world.client._convo_streak.get(ch_id, {}), \
        "conversation streak wasn't reset after the wrap"


async def test_wrap_reply_never_asks_a_question(world):
    """_is_wrap strips ?-sentences — a closer that asks 'wbu?' re-opens the
    convo we just closed."""
    _seed_long_convo(world, world.alice)
    world.llm.reply_fn = lambda trigger, user_l: "fair enough. anything else?"
    await _say_reply(world, world.alice, "hmm okay fine")
    sent = world.sent()
    assert sent, "wrap-up closer got no reply"
    assert all("?" not in m for m in sent), \
        f"wrap reply kept a question: {sent}"
    assert any("fair enough" in m for m in sent)


async def test_wrap_quiet_drops_dry_lines(world, monkeypatch):
    """Inside the quiet window ambient acks/closers are dropped."""
    monkeypatch.setattr(random, "random", lambda: 0.99)  # pin the 25% AFK roll off
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    n = len(world.sent())
    await world.say(world.alice, "lol")            # low-content
    await world.say(world.alice, "alright cool")   # another dry closer
    assert len(world.sent()) == n, \
        "ambient lines got replies inside the wrap-quiet window"


async def test_wrap_quiet_reason_strings(world, monkeypatch):
    """White-box: dry lines -> 'wrap-quiet', first real line -> 're-engage
    1/2', second sustained line breaks the window entirely."""
    monkeypatch.setattr(random, "random", lambda: 0.99)
    ch_id, uid = _ch_id(world), str(world.alice.id)
    world.client._wrap_quiet[ch_id] = {uid: world.clock.time() + 900}

    ok, r = world.client._should_respond(_msg(world, world.alice, "lol"))
    assert not ok and r.startswith("wrap-quiet")
    ok, r = world.client._should_respond(_msg(world, world.alice, "alright cool"))
    assert not ok and r.startswith("wrap-quiet")

    # First substantive message — dropped, but counted as re-engagement 1/2
    ok, r = world.client._should_respond(
        _msg(world, world.alice,
             "wait no actually there's something else i wanted to say"))
    assert not ok and "wrap-quiet" in r and "re-engage 1/2" in r, r

    # Second sustained message — they weren't done after all: window breaks
    world.client._should_respond(
        _msg(world, world.alice,
             "the whole thing with the update still makes no sense to me"))
    assert uid not in world.client._wrap_quiet.get(ch_id, {}), \
        "sustained re-engagement didn't break the quiet window"


async def test_wrap_quiet_breaks_early_on_sustained_talk(world, monkeypatch):
    """End-to-end: two real messages after a wrap re-open the conversation."""
    monkeypatch.setattr(random, "random", lambda: 0.99)
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    n = len(world.sent())
    await world.say(
        world.alice,
        "ok wait im not actually done, one more thing about what you said")
    assert len(world.sent()) == n, \
        "first substantive line should still be quiet (re-engage 1/2)"
    await _say_reply(
        world, world.alice,
        "like actually though, the ending still makes no sense to me")
    assert len(world.sent()) > n, \
        "sustained re-engagement never re-opened the conversation"


async def test_wrap_quiet_still_answers_questions(world, monkeypatch):
    """Quiet window isn't ghosting — a real question still gets an answer."""
    monkeypatch.setattr(random, "random", lambda: 0.99)
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    n = len(world.sent())
    # margin past the 5s active-conversation cooldown
    await world.advance(15)
    await _say_reply(world, world.alice, "wait actually what did you mean by that?")
    assert len(world.sent()) > n, "real question was silenced by wrap-quiet"


async def test_wrap_quiet_still_answers_mentions(world, monkeypatch):
    """A direct ping cuts through the quiet window."""
    monkeypatch.setattr(random, "random", lambda: 0.99)
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    n = len(world.sent())
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> sorry one more thing")
    assert len(world.sent()) > n, "mention was silenced by wrap-quiet"


async def test_wrap_quiet_reply_to_us_bypasses(world, monkeypatch):
    """A reply-quote to OUR wrap message also bypasses the quiet window."""
    monkeypatch.setattr(random, "random", lambda: 0.99)
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    bot_msg = world.sent_messages()[-1]
    n = len(world.sent())
    # margin past the 15s reply-to-me cooldown before quoting her
    await world.advance(20)
    await _say_reply(world, world.alice,
                     "wait i actually changed my mind about this",
                     reply_to=bot_msg)
    assert len(world.sent()) > n, \
        "reply to the bot's own message was silenced by wrap-quiet"


async def test_wrap_can_trigger_afk_window(world, monkeypatch):
    """~25% of wraps also set a 30-90min AFK window — pin the roll to fire."""
    monkeypatch.setattr(random, "random", lambda: 0.20)  # <0.25 -> AFK fires
    _seed_long_convo(world, world.alice)
    await _say_reply(world, world.alice, "hmm okay fine")
    afk_end = world.client.afk_until.get(_ch_id(world), 0)
    assert afk_end >= world.clock.time() + 1700, \
        "wrap's AFK roll didn't arm afk_until"


# ─────────────────────────────────────────────────────────────────────────────
# AFK window — gate semantics
# ─────────────────────────────────────────────────────────────────────────────

async def test_afk_drops_ambient_and_mention_breaks(world):
    ch_id = _ch_id(world)
    world.client.afk_until[ch_id] = world.clock.time() + 1800

    ok, r = world.client._should_respond(
        _msg(world, world.alice, "anyone still here"))
    assert not ok and r == "AFK — ignoring", r

    await world.say(world.alice, "this is just ambient chatter")
    assert not world.bot_said_something(), "bot replied while AFK"

    # A mention breaks the window and gets a reply
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> you there?")
    assert world.bot_said_something(), "mention didn't break the AFK window"
    assert world.client.afk_until.get(ch_id, 0) <= world.clock.time()


# ─────────────────────────────────────────────────────────────────────────────
# Dismissal backoff
# ─────────────────────────────────────────────────────────────────────────────

async def test_dismissal_in_sticky_sets_backoff(world):
    """'shut up' mid-conversation (no mention needed — we were just talking)
    -> 600s backoff, sticky cleared, ambient dropped, mention breaks it."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    ch_id = _ch_id(world)
    assert world.client.sticky_until.get(ch_id, 0) > world.clock.time()
    # Spawned follow-ups (mood GIF, bursts) can still be in flight — flush
    # before capturing the baseline or they land inside the dismissal window
    await world.advance(30)

    n = len(world.sent())
    await world.say(world.alice, "shut up")
    assert len(world.sent()) == n, "dismissal itself got a reply"
    assert world.client.backoff_until.get(ch_id, 0) > world.clock.time()
    assert world.client.sticky_until.get(ch_id, 0) <= world.clock.time(), \
        "dismissal didn't clear the sticky window"

    await world.say(world.alice, "anyway what was everyone saying")
    assert len(world.sent()) == n, "ambient line got a reply during backoff"

    ok, r = world.client._should_respond(
        _msg(world, world.alice, "still there?"))
    assert not ok and r.startswith("backoff"), r

    # A direct mention breaks the backoff
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> ok come back")
    assert len(world.sent()) > n, "mention during backoff got no reply"
    assert world.client.backoff_until.get(ch_id, 0) <= world.clock.time()


async def test_dismissal_via_mention_sets_backoff(world):
    """Dismissal by @mention works even without a live sticky window."""
    await world.say(world.alice, f"<@{world.bot_user.id}> shut up")
    ch_id = _ch_id(world)
    assert world.client.backoff_until.get(ch_id, 0) > world.clock.time()
    ok, r = world.client._should_respond(_msg(world, world.alice, "hello?"))
    assert not ok and r.startswith("backoff"), r


async def test_undirected_dismissal_sets_no_backoff(world):
    """'shut up' aimed at someone else (no mention, no sticky) isn't ours
    to take personally — no backoff is armed."""
    ch_id = _ch_id(world)
    ok, r = world.client._should_respond(_msg(world, world.alice, "shut up"))
    assert r != "dismissed-by-user"
    assert world.client.backoff_until.get(ch_id, 0) <= world.clock.time()


async def test_backoff_expiry_lets_messages_through(world):
    """Once the 600s window elapses the channel gates normally again."""
    ch_id = _ch_id(world)
    world.client.backoff_until[ch_id] = world.clock.time() + 5
    ok, r = world.client._should_respond(
        _msg(world, world.alice, "anyone up for games"))
    assert not ok and r.startswith("backoff"), r

    await world.advance(10)
    ok, r = world.client._should_respond(
        _msg(world, world.alice, "anyone up for games"))
    assert not r.startswith("backoff"), f"backoff leaked past expiry: {r}"
    assert ok, f"post-backoff ambient gate rejected a seeking-chat line: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# Sticky conversation window — open, cooldown, low-content, expiry
# ─────────────────────────────────────────────────────────────────────────────

async def test_reply_opens_sticky_window_and_tracks_user(world):
    """A send sets sticky_until ~120s out and tracks THAT user for 30min."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> yo")
    ch_id = _ch_id(world)
    assert world.client.sticky_until.get(ch_id, 0) > world.clock.time()
    assert str(world.alice.id) in \
        world.client.conversation_tracker.get(ch_id, {})
    ok, r = world.client._should_respond(
        _msg(world, world.bob, "yeah that's actually insane"))
    assert ok and r.startswith("sticky-convo"), r


async def test_sticky_window_answers_then_expires(world):
    """Inside the window an ambient line gets an answer; after expiry the
    same line hits ambient gating (no 'sticky-convo' reason)."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    n = len(world.sent())
    await _say_reply(world, world.bob, "nah the update actually broke everything")
    assert len(world.sent()) > n, "sticky-window message got no reply"

    # Expire the window — next ambient line must not route through sticky
    ch_id = _ch_id(world)
    world.client.sticky_until[ch_id] = world.clock.time() - 1
    world.client.last_replied[ch_id] = world.clock.time() - 20
    ok, r = world.client._should_respond(
        _msg(world, world.carol, "whatever happened with that thing anyway"))
    assert not r.startswith("sticky"), f"sticky window leaked past expiry: {r}"


async def test_sticky_cooldown_drops_short_filler(world):
    """<2s after our reply (STICKY_COOLDOWN_S), <=3-word filler is dropped —
    but real content in the same window still gets through."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    ch_id = _ch_id(world)
    # bot literally just replied — inside the 2s cooldown
    world.client.last_replied[ch_id] = world.clock.time()

    ok, r = world.client._should_respond(_msg(world, world.bob, "lol"))
    assert not ok and r.startswith("sticky-cooldown"), r

    ok, r = world.client._should_respond(
        _msg(world, world.bob, "wait what did you mean by that earlier"))
    assert ok and r.startswith("sticky-convo"), r


async def test_sticky_low_content_drops_and_can_react(world, monkeypatch):
    """'lol' inside sticky (past the 2s cooldown) -> 'sticky-low-content':
    no typed reply, ~35% of the time a reaction instead — pin the roll."""
    monkeypatch.setattr(random, "random", lambda: 0.20)  # <0.35 -> reaction
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    ch_id = _ch_id(world)
    world.client.last_replied[ch_id] = world.clock.time() - 10
    world.client.sticky_until[ch_id] = world.clock.time() + 100

    n = len(world.sent())
    msg = await world.say(world.bob, "lol")
    assert len(world.sent()) == n, "low-content ack got a typed reply"
    assert world.reacted(msg), "sticky-low-content left no reaction either"

    ok, r = world.client._should_respond(_msg(world, world.bob, "fr"))
    assert not ok and r == "sticky-low-content", r


# ─────────────────────────────────────────────────────────────────────────────
# Conversation tracker — 30-min per-user window
# ─────────────────────────────────────────────────────────────────────────────

async def test_conversation_tracker_refreshes_on_followup(world):
    """Their follow-up refreshes the tracker timestamp even when the line
    itself doesn't earn a reply (low-content inside sticky)."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    ch_id, uid = _ch_id(world), str(world.alice.id)
    ts1 = world.client.conversation_tracker[ch_id][uid]
    await world.advance(30)
    await world.say(world.alice, "lol")   # sticky-low-content — no reply
    assert world.client.conversation_tracker[ch_id][uid] > ts1, \
        "follow-up didn't refresh the conversation tracker"


async def test_active_conversation_outlives_sticky(world):
    """After the 2-min window dies, the tracked user's lines still hit
    'active-conversation' — an untracked user's don't."""
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> hey")
    await world.advance(130)   # sticky (120s) dead, tracker (30min) alive

    ok, r = world.client._should_respond(
        _msg(world, world.alice, "so yeah anyway i was thinking more about it"))
    assert ok and r.startswith("active-conversation"), r

    ok, r = world.client._should_respond(
        _msg(world, world.carol, "so yeah anyway i was thinking more about it"))
    assert not r.startswith("active-conversation"), \
        f"untracked user got active-conversation treatment: {r}"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-persona pending capture
# ─────────────────────────────────────────────────────────────────────────────

async def test_message_mentioning_offline_persona_recorded(world, monkeypatch):
    """@isla while isla's offline -> pending item for her (kind='mention'),
    even though THIS persona must not reply to it."""
    from src.persona import runtime as prt
    isla = world.person("isla")
    # Registry normally maps known persona user_ids; simulate isla's id
    # being registered from an earlier activation.
    monkeypatch.setattr(prt, "other_persona_ids", lambda: {isla.id: "isla"})
    msg = await world.say(world.alice, f"hey <@{isla.id}> you around?")
    pending = prt.list_pending("isla")
    assert any(
        p["message_id"] == msg.id and p["kind"] == "mention"
        and p["author_id"] == world.alice.id for p in pending), \
        f"offline-persona mention not captured: {pending}"
    # ...and it stayed addressed to her — we didn't answer it ourselves
    assert not world.bot_said_something()


async def test_message_naming_offline_persona_recorded(world):
    """Another persona's casual name in text -> pending item (kind='name')."""
    from src.persona import runtime as prt
    await world.say(world.bob, "has anyone seen isla today")
    pending = prt.list_pending("isla")
    assert any(
        p["kind"] == "name" and p["author_id"] == world.bob.id
        for p in pending), f"name-drop not captured for offline persona: {pending}"


# ─────────────────────────────────────────────────────────────────────────────
# Unanswered questions — per-user scoping
# ─────────────────────────────────────────────────────────────────────────────

async def test_unanswered_questions_cleared_per_user(world):
    """Replying to alice clears ALICE's tracked questions only — bob's
    pending question must survive (worker-B regression: mark_answered
    used to wipe the whole channel deque)."""
    ch_id = _ch_id(world)
    world.client.unanswered_tracker.clear_channel(ch_id)  # singleton hygiene
    carol = world.carol
    # Questions aimed at someone else are recorded but never answered —
    # deterministic way to queue them without an early reply.
    await world.say(world.alice,
                    f"hey <@{carol.id}> what's the plan for tonight?",
                    advance=10)
    await world.say(world.bob,
                    f"hey <@{carol.id}> where did you get that hoodie?",
                    advance=10)
    qs = [t for t, _a in
          world.client.unanswered_tracker.get_unanswered_questions(ch_id)]
    assert any("plan for tonight" in t for t in qs), qs
    assert any("hoodie" in t for t in qs), qs

    # The bot answers alice — her questions clear, bob's survive.
    await _say_reply(world, world.alice, f"<@{world.bot_user.id}> what do you think?")
    qs = [t for t, _a in
          world.client.unanswered_tracker.get_unanswered_questions(ch_id)]
    assert not any("plan for tonight" in t for t in qs), \
        f"alice's question wasn't cleared after her answer: {qs}"
    assert any("hoodie" in t for t in qs), \
        f"bob's pending question was wiped by alice's reply: {qs}"
