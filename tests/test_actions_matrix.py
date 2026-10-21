"""Action-worker matrix — bridge gate (src/ai/action_bridge.py) + vendored
executor (src/action_engine/).

Covers the pipeline surface the conversational suite doesn't:
  classification   — exec vs info vs chat routing through ActionWorker
  owner gating     — _OWNER_ONLY denied at the dispatch gate, owner passes
  mass-mention     — @everyone/@here/<@&id> neutered for non-owners
  batch caps       — count args clamped (<=5 non-owner / <=10 owner)
  skip-channel     — non-owner sends into rules/bot/etc channels refused
  delete_last      — the bot's own messages resolved + deleted via ctx
  dedup            — _inflight blocks duplicate exec/info work
  stale client     — closed/torn-down clients can't run worker paths
  scheduler        — schedule_message lands + repeats on virtual time
  failure surface  — tool errors reach the agent, nothing crashes

Fidelity notes (where the fake forces gate-level asserts):
  * Klipy GIF search needs a live API key, so send_multiple_gifs can't
    execute end-to-end — its `count` clamp is asserted at _check_allowed.
  * Voice tools (join/move/say_in_vc) need a live voice connection — not
    exercised; the gate for them is covered via _check_allowed instead.
"""
import asyncio
import json

import pytest

from tests.sim import fake_discord as fd
from src.ai.action_bridge import get_action_worker


OWNER_ID = 999000999          # conftest seeds OWNER_IDS=999000999


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _worker(world):
    return get_action_worker(world.client)


async def _ready(world):
    """Live worker with the vendored engine initialised."""
    w = _worker(world)
    assert await w._ensure_ready(), "engine failed to initialise"
    return w


def _ctx(world, worker, person, channel=None):
    """A ToolContext exactly as the bridge builds per request."""
    return worker._ctx_for(channel or world.channel(), person.id)


def _msg(world, person, text, channel=None):
    ch = channel or world.channel()
    return fd.make_message(ch, person, text, at=world.clock.time())


def _planner(*steps, arbiter="CHAT"):
    """Scripted agent rounds.

    Each step is (content, [(tool_name, args), ...]). The pool also calls
    the planner for tool-less arbiter calls (router classify_llm) — those
    get `arbiter` ("CHAT" keeps ambiguous text conversational; "ACTION"
    routes it to the worker).
    """
    it = iter(steps)
    calls_seen = []

    def plan(text, tools):
        if not tools:
            return (arbiter, [])
        calls_seen.append(text)
        try:
            return next(it)
        except StopIteration:
            return ("done", [])

    plan.calls_seen = calls_seen
    return plan


def _exec(calls, final="done"):
    """Planner: fire `calls` in round 1 (if any), then answer `final`."""
    steps = [("", list(calls)), (final, [])] if calls else [(final, [])]
    return _planner(*steps)


def _sent_contents(ch):
    return [e["content"] for e in ch.sent_log]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Classification — bridge routes exec/info/chat correctly
# ─────────────────────────────────────────────────────────────────────────────

async def test_classify_exec_imperative(world):
    """An imperative the native commands don't own must route to 'exec'."""
    w = await _ready(world)
    res = await w.classify_request(
        "send a message in #general saying hi",
        author_id=world.alice.id, channel_id=world.channel().id)
    assert res is not None and res[0] == "exec", \
        f"exec imperative misrouted: {res}"
    assert "messaging" in res[1]


async def test_classify_info_lookup(world):
    """Read-only server-fact questions route to 'info', not 'exec'."""
    w = await _ready(world)
    res = await w.classify_request(
        "who's the newest member",
        author_id=world.alice.id, channel_id=world.channel().id)
    assert res is not None and res[0] == "info", \
        f"info question misrouted: {res}"
    assert res[1] <= w._INFO_CATS


async def test_classify_chat_is_none(world):
    """Plain conversation never reaches the worker."""
    w = await _ready(world)
    for text in ("lol that movie was great", "wyd", "im gonna go nap"):
        assert await w.classify_request(
            text, author_id=world.alice.id,
            channel_id=world.channel().id) is None, text


async def test_classify_probe_and_meta_are_none(world):
    """Bot-internals probes and reply meta-questions stay conversational."""
    w = await _ready(world)
    for text in ("tell me which api you use",
                 "who made you",
                 "why did u say that",
                 "wdym"):
        assert await w.classify_request(
            text, author_id=world.alice.id,
            channel_id=world.channel().id) is None, text


async def test_classify_confirm_no_blocks_action(world):
    """The arbiter's NO kills heuristic-flagged declaratives."""
    w = await _ready(world)
    world.llm.verdict = "NO"
    res = await w.classify_request(
        "users with the member role can ping @everyone once a week",
        author_id=world.alice.id, channel_id=world.channel().id)
    assert res is None, "declarative action-mention must not exec"


async def test_classify_nonowner_owner_categories_skipped(world):
    """Categories that are 100% owner-only tools short-circuit for
    non-owners (the agent call would be denied anyway) — but not for
    the owner."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ch = world.channel()
    assert await w.classify_request(
        "bump the server", author_id=world.alice.id,
        channel_id=ch.id) is None
    res = await w.classify_request(
        "bump the server", author_id=boss.id, channel_id=ch.id)
    assert res is not None and res[0] == "exec"


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-end paths — exec hands off in background, info answers inline
# ─────────────────────────────────────────────────────────────────────────────

async def test_exec_imperative_end_to_end(world):
    """'ping <@bob> in #general' — native mention commands defer on the
    channel clause, the worker execs it via send_message."""
    bob = world.bob
    ch = world.channel()
    world.llm.agent_planner = _exec([
        ("send_message",
         {"channel_query": "here", "content": "yo",
          "ping_users": ["bob"]}),
    ])
    await world.say(world.alice, f"ping <@{bob.id}> in #general")
    sent = _sent_contents(ch)
    assert any(f"<@{bob.id}>" in c for c in sent), \
        f"bob was never pinged by the worker: {sent}"
    assert "on it" not in sent, "agent chatter leaked into the channel"


async def test_info_question_end_to_end(world):
    """Info lookups run inline; the facts get injected into the reply."""
    world.bob
    world.llm.agent_planner = _exec([], final="alice is the newest member")
    await world.say(world.alice,
                    f"<@{world.bot_user.id}> who's our newest member")
    # the agent ran with the read-only info tool set
    info_tools = {"search_members", "get_member_count", "get_recent_joins"}
    assert any(info_tools & set(c["tools"]) for c in world.pool.calls), \
        "info lookup never reached the agent"
    # the answer was injected into the conversational reply's context
    assert any("looked up" in (c["system"] + c["user"]).lower()
               and "alice is the newest member" in (c["system"] + c["user"])
               for c in world.llm.calls), \
        "info_facts never reached the reply prompt"
    assert world.bot_said_something(), "bot went silent on an info request"


async def test_run_text_info_returns_answer(world):
    """run_text_info relays the agent's final text — and survives a real
    tool round-trip (get_member_count executes against the fake guild)."""
    w = await _ready(world)
    world.alice
    rounds = _planner(
        ("", [("get_member_count", {})]),
        ("2 members online", []),
    )
    world.llm.agent_planner = rounds
    msg = _msg(world, world.alice, "who's in this server")
    ans = await w.run_text_info(msg, "who's in this server")
    assert ans == "2 members online"
    assert any("get_member_count" in c["tools"] for c in world.pool.calls)


async def test_run_text_info_sentinel_returns_none(world):
    """Agent failure sentinels ('(sorry...', '(no reply') must not leak
    as 'facts' into the reply — run_text_info swallows them."""
    w = await _ready(world)
    world.llm.agent_planner = _exec([], final="(no reply)")
    msg = _msg(world, world.alice, "who joined last")
    assert await w.run_text_info(msg, "who joined last") is None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Owner gating
# ─────────────────────────────────────────────────────────────────────────────

async def test_gate_blocks_owner_only_for_nonowner(world):
    """_OWNER_ONLY tools are refused before dispatch for non-owners."""
    w = await _ready(world)
    ctx = _ctx(world, w, world.alice)
    for tool in ("schedule_message", "delete_last_message", "send_dm",
                 "change_status", "use_slash_command", "bump_with_bot"):
        ok, _ = w._check_allowed(ctx, tool, {})
        assert not ok, f"{tool} allowed for non-owner"


async def test_gate_allows_owner(world):
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ctx = _ctx(world, w, boss)
    for tool in ("schedule_message", "delete_last_message", "send_dm",
                 "change_status"):
        ok, _ = w._check_allowed(ctx, tool, {})
        assert ok, f"{tool} refused for owner"


async def test_owner_only_tool_denied_at_dispatch(world):
    """End-to-end through the guarded dispatch: non-owner schedule_message
    returns an error, records nothing as performed, schedules nothing."""
    w = await _ready(world)
    ctx = _ctx(world, w, world.alice)
    res = await w._agent_mod.dispatch(
        ctx, "schedule_message",
        {"channel_query": "here", "content": "spam", "delay_s": 1})
    data = json.loads(res)
    assert "error" in data
    assert ctx.performed == [], "denied tool must not record as performed"


async def test_owner_only_tool_executes_for_owner(world):
    """Same call as an owner reaches the real tool — and really schedules."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ctx = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx, "schedule_message",
        {"channel_query": "here", "content": "owner ping",
         "delay_s": 5, "label": "matrix-owner-once"})
    data = json.loads(res)
    assert data.get("ok") is True, res
    assert ("schedule_message", True) in ctx.performed
    from src.action_engine.core import scheduler as sched
    try:
        await world.advance(20)
        assert "owner ping" in _sent_contents(world.channel())
    finally:
        sched.scheduler.tasks.clear()


async def test_send_dm_owner_only(world):
    """send_dm is an abuse vector — owner-only. The DM lands on the user's
    DM channel (FakeUser.create_dm), never the guild channel."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    bob = world.bob
    ch = world.channel()

    ctx_owner = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx_owner, "send_dm", {"user_query": "bob", "content": "psst"})
    assert json.loads(res).get("ok") is True, res
    assert bob.dm_channel is not None
    assert bob.dm_channel.sent_log[0]["content"] == "psst"
    assert "psst" not in _sent_contents(ch)

    ctx_alice = _ctx(world, w, world.alice)
    res = await w._agent_mod.dispatch(
        ctx_alice, "send_dm", {"user_query": "bob", "content": "spam"})
    assert "error" in json.loads(res)
    assert len(bob.dm_channel.sent_log) == 1, "non-owner DM went through"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mass-mention sanitization
# ─────────────────────────────────────────────────────────────────────────────

async def test_mass_mention_stripped_for_nonowner(world):
    """'@everyone hi' in a send body must be neutered before the tool
    executes — the send STILL happens, just without the mass ping."""
    w = await _ready(world)
    ctx = _ctx(world, w, world.alice)
    res = await w._agent_mod.dispatch(
        ctx, "send_message",
        {"channel_query": "here",
         "content": "@everyone hi @here <@&123456>"})
    data = json.loads(res)
    assert data.get("ok") is True, res
    sent = data["sent"]
    for bad in ("@everyone", "@here", "<@&"):
        assert bad not in sent, f"mass-mention token survived: {sent!r}"
    assert "hi" in sent
    # and what actually landed in the channel is clean too
    ch_sent = _sent_contents(world.channel())[-1]
    for bad in ("@everyone", "@here", "<@&"):
        assert bad not in ch_sent, ch_sent


async def test_mass_mention_owner_passthrough(world):
    """Gate behaviour as built: owners keep raw args — @everyone is NOT
    stripped for them (explicit owner intent)."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ctx = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx, "send_message",
        {"channel_query": "here", "content": "@everyone hi"})
    data = json.loads(res)
    assert data.get("ok") is True, res
    assert "@everyone" in data["sent"], \
        "owner mass-mention should pass through untouched"


async def test_ping_roles_stripped_nonowner(world):
    """Non-owner role pings are emptied pre-dispatch; owners keep theirs."""
    w = await _ready(world)
    mods = fd.FakeRole(name="mods")
    world.guild.roles.append(mods)

    ctx = _ctx(world, w, world.alice)
    ok, args = w._check_allowed(
        ctx, "send_message",
        {"channel_query": "here", "content": "hi",
         "ping_roles": ["mods"],
         "ping_users": ["a", "b", "c", "d", "e", "f", "g"]})
    assert ok
    assert args["ping_roles"] == [], "non-owner kept role pings"
    assert len(args["ping_users"]) == 5, \
        "ping_users not capped at 5 for non-owner"

    boss = world.person("boss", user_id=OWNER_ID)
    ctx_o = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx_o, "send_message",
        {"channel_query": "here", "content": "hi",
         "ping_roles": ["mods"]})
    data = json.loads(res)
    assert data.get("ok") is True
    assert f"<@&{mods.id}>" in data["sent"], \
        "owner role ping was stripped"


async def test_every_string_arg_sanitized(world):
    """Mass-mention tokens hide in ANY string arg (send_gif's free text is
    `caption`, not `content`) and inside list args."""
    w = await _ready(world)
    ctx = _ctx(world, w, world.alice)
    ok, args = w._check_allowed(
        ctx, "send_gif",
        {"channel_query": "here", "query": "cat",
         "caption": "look @everyone",
         "ping_users": ["@here all", "bob"]})
    assert ok
    assert "@everyone" not in args["caption"]
    assert all("@here" not in p for p in args["ping_users"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Batch caps
# ─────────────────────────────────────────────────────────────────────────────

async def test_batch_count_clamped_gate(world):
    """count=99 → 5 (non-owner) / 10 (owner); garbage counts clamp too."""
    w = await _ready(world)
    ctx = _ctx(world, w, world.alice)
    boss = world.person("boss", user_id=OWNER_ID)
    ctx_o = _ctx(world, w, boss)

    for tool in ("react_to_recent", "send_multiple_gifs",
                 "send_multiple_stickers"):
        _, a = w._check_allowed(ctx, tool, {"count": 99})
        assert a["count"] == 5, (tool, a)
        _, a = w._check_allowed(ctx_o, tool, {"count": 99})
        assert a["count"] == 10, (tool, a)
        _, a = w._check_allowed(ctx, tool, {"count": "bogus"})
        assert a["count"] == 5, (tool, a)
        _, a = w._check_allowed(ctx, tool, {"count": 0})
        assert a["count"] == 1, (tool, a)

    # delete_last_message is ALSO a batch-count tool — but owner-only, so
    # the non-owner call is refused before its count is ever clamped.
    ok, _ = w._check_allowed(ctx, "delete_last_message", {"count": 99})
    assert not ok
    _, a = w._check_allowed(ctx_o, "delete_last_message", {"count": 99})
    assert a["count"] == 10


async def test_react_to_recent_clamped_end_to_end(world):
    """count=99 really clamps: only the 5 newest messages get the emoji."""
    w = await _ready(world)
    ch = world.channel()
    people = [world.alice, world.bob, world.carol]
    msgs = [fd.make_message(ch, people[i % 3], f"m{i}",
                            at=world.clock.time()) for i in range(9)]
    ctx = _ctx(world, w, world.alice)
    # react_to_recent resolves channels by fuzzy name (no "here" shortcut)
    res = await w._agent_mod.dispatch(
        ctx, "react_to_recent",
        {"channel_query": "general", "emoji": "🔥", "count": 99})
    data = json.loads(res)
    assert data.get("ok") is True and data["reacted_count"] == 5, res
    reacted = [m for m in msgs if any(r.me for r in m.reactions)]
    assert len(reacted) == 5


async def test_react_to_recent_owner_cap(world):
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ch = world.channel()
    msgs = [fd.make_message(ch, world.alice, f"m{i}",
                            at=world.clock.time()) for i in range(14)]
    ctx = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx, "react_to_recent",
        {"channel_query": "general", "emoji": "👍", "count": 99})
    data = json.loads(res)
    assert data["reacted_count"] == 10, res
    reacted = [m for m in msgs if any(r.me for r in m.reactions)]
    assert len(reacted) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 6. Skip-channel gate — non-owners can't post into rules/bot channels
# ─────────────────────────────────────────────────────────────────────────────

async def test_skip_channel_blocked_for_nonowner(world):
    """#rules is a skip-channel — a non-owner's send_message resolves it
    then gets refused inside the tool gate. Nothing lands."""
    w = await _ready(world)
    rules = world.guild.add_channel("rules")
    ctx = _ctx(world, w, world.alice)
    res = await w._agent_mod.dispatch(
        ctx, "send_message",
        {"channel_query": "rules", "content": "hello mods"})
    data = json.loads(res)
    assert "error" in data, res
    assert rules.sent_log == [], "non-owner posted into a skip-channel"


async def test_skip_channel_allowed_for_owner(world):
    """Owners can post anywhere — the skip gate returns early for them."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    rules = world.guild.add_channel("rules")
    ctx = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx, "send_message",
        {"channel_query": "rules", "content": "official notice"})
    assert json.loads(res).get("ok") is True, res
    assert "official notice" in _sent_contents(rules)


# ─────────────────────────────────────────────────────────────────────────────
# 7. delete_last_message — resolves and deletes the bot's own messages
# ─────────────────────────────────────────────────────────────────────────────

async def test_delete_last_message_owner(world):
    """Owner 'delete your last message' removes the newest bot message in
    the channel and nothing else."""
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ch = world.channel()
    bot_m1 = await ch.send("bot message one")
    bot_m2 = await ch.send("bot message two")
    human = fd.make_message(ch, world.alice, "keep me",
                            at=world.clock.time())
    ctx = _ctx(world, w, boss)
    res = await w._agent_mod.dispatch(
        ctx, "delete_last_message", {"channel_query": "here", "count": 1})
    data = json.loads(res)
    assert data["deleted_count"] == 1, res
    assert bot_m2 in ch.deleted, "newest bot message wasn't deleted"
    assert bot_m1 in ch.messages, "older bot message wrongly deleted"
    assert human in ch.messages, "human message wrongly deleted"

    # count=2 finishes the sweep — the tool paces bulk deletes with
    # asyncio.sleep(1.2) between messages, so it must run concurrently
    # with the clock advance (awaiting it directly would deadlock: the
    # test task would park on virtual time it alone can advance).
    res = await w._agent_mod.dispatch(
        ctx, "delete_last_message", {"channel_query": "here", "count": 1})
    t = asyncio.create_task(w._agent_mod.dispatch(
        ctx, "delete_last_message", {"channel_query": "here", "count": 2}))
    await world.advance(5)
    await t
    assert bot_m1 in ch.deleted


async def test_delete_last_message_denied_nonowner(world):
    """Deletion is owner-only — a non-owner's agent call gets refused and
    the bot's messages survive."""
    w = await _ready(world)
    ch = world.channel()
    bot_m = await ch.send("bot message")
    ctx = _ctx(world, w, world.alice)
    res = await w._agent_mod.dispatch(
        ctx, "delete_last_message", {"channel_query": "here", "count": 1})
    assert "error" in json.loads(res)
    assert bot_m in ch.messages
    assert bot_m not in ch.deleted


# ─────────────────────────────────────────────────────────────────────────────
# 8. Inflight dedup — same user+channel can't double-queue
# ─────────────────────────────────────────────────────────────────────────────

async def test_inflight_dedup(world):
    """Back-to-back identical requests: the second is refused by
    queue_text_action AND short-circuits classify_request while inflight."""
    w = await _ready(world)
    ch = world.channel()
    bob = world.bob
    world.llm.agent_planner = _exec([
        ("send_message",
         {"channel_query": "here", "content": "yo",
          "ping_users": ["bob"]}),
    ])
    msg = _msg(world, world.alice, "ping bob", channel=ch)
    assert w.queue_text_action(msg, "ping bob", {"messaging"}) is True
    # second identical request while the first is parked in its delay
    assert w.queue_text_action(msg, "ping bob", {"messaging"}) is False
    assert await w.classify_request(
        "ping bob", author_id=world.alice.id,
        channel_id=ch.id) is None, "inflight probe didn't short-circuit"

    await world.advance(30)
    assert not w._inflight, "inflight key leaked after job finished"
    # exactly ONE agent run happened — one send, not two
    sends = [c for c in _sent_contents(ch) if f"<@{bob.id}>" in c]
    assert len(sends) == 1, sends
    # and the gate is open again for the next request
    res = await w.classify_request(
        "ping bob", author_id=world.alice.id, channel_id=ch.id)
    assert res is not None and res[0] == "exec"


async def test_info_inflight_dedup(world):
    """run_text_info also honours _inflight — a dup while an exec is
    parked returns None instead of running a second agent."""
    w = await _ready(world)
    ch = world.channel()
    world.llm.agent_planner = _exec([
        ("send_message", {"channel_query": "here", "content": "hi"}),
    ])
    msg = _msg(world, world.alice, "send hi", channel=ch)
    assert w.queue_text_action(msg, "send hi", {"messaging"}) is True
    assert await w.run_text_info(msg, "who's newest") is None
    await world.advance(30)
    assert not w._inflight


# ─────────────────────────────────────────────────────────────────────────────
# 9. Stale client — closed/torn-down clients must not run worker paths
# ─────────────────────────────────────────────────────────────────────────────

async def test_closed_client_sends_nothing_e2e(world):
    """is_closed() gates _handle_reply upstream — nothing reaches the
    worker: no send, no inflight, no agent run."""
    w = await _ready(world)
    ch = world.channel()
    bob = world.bob
    world.llm.agent_planner = _exec([
        ("send_message", {"channel_query": "here", "content": "yo"}),
    ])
    n_before = len(ch.sent_log)
    agent_before = w._agent
    world.client._closing_task = True     # discord.py marks closing here
    await world.say(world.alice, f"ping <@{bob.id}> in #general")
    assert len(ch.sent_log) == n_before, "closed client sent something"
    assert not w._inflight
    assert w._agent is agent_before, "agent ran on a closed client"


async def test_worker_entry_points_refuse_closed_client(world):
    """The worker itself must not run on a dead client — a queued exec
    racing a rotation would fire tools on a closed session."""
    w = await _ready(world)
    ch = world.channel()
    world.client._closing_task = True
    msg = _msg(world, world.alice, "send hi", channel=ch)
    assert w.queue_text_action(msg, "send hi", {"messaging"}) is False, \
        "queue_text_action ran on a closed client"
    assert await w.run_text_info(msg, "who's newest") is None, \
        "run_text_info ran on a closed client"
    assert await w.classify_request(
        "send hi", author_id=world.alice.id, channel_id=ch.id) is None, \
        "classify_request ran on a closed client"
    await world.advance(30)
    assert ch.sent_log == []


async def test_teardown_cancels_inflight_exec(world):
    """A parked exec task must die with the persona — the action never
    fires after teardown."""
    w = await _ready(world)
    ch = world.channel()
    world.llm.agent_planner = _exec([
        ("send_message", {"channel_query": "here", "content": "too late"}),
    ])
    msg = _msg(world, world.alice, "send hi", channel=ch)
    assert w.queue_text_action(msg, "send hi", {"messaging"}) is True
    # job is parked in its 2–6s human delay — teardown now
    await world.client.teardown()
    await world.advance(60)
    assert "too late" not in _sent_contents(ch), \
        "queued action fired after the client was torn down"
    assert not w._inflight


async def test_worker_entry_points_refuse_torn_down_client(world):
    """teardown() marks the client dead without is_closed() flipping on
    the sim — entry points must honour _torn_down too."""
    w = await _ready(world)
    ch = world.channel()
    await world.client.teardown()
    msg = _msg(world, world.alice, "send hi", channel=ch)
    assert w.queue_text_action(msg, "send hi", {"messaging"}) is False
    assert await w.run_text_info(msg, "who's newest") is None
    await world.advance(30)
    assert ch.sent_log == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. schedule_message — one-shot lands, intervals repeat on virtual time
# ─────────────────────────────────────────────────────────────────────────────

async def test_schedule_message_fires_on_advance(world):
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ch = world.channel()
    world.llm.agent_planner = _exec([
        ("schedule_message",
         {"channel_query": "here", "content": "scheduled hello",
          "delay_s": 5, "label": "matrix-one-shot"}),
    ])
    msg = _msg(world, boss, "schedule a hello in 5 seconds", channel=ch)
    assert w.queue_text_action(
        msg, "schedule a hello in 5 seconds", {"scheduling"}) is True
    from src.action_engine.core import scheduler as sched
    try:
        await world.advance(20)
        sent = _sent_contents(ch)
        assert "scheduled hello" in sent, \
            f"one-shot never fired: {sent}"
        # one-shot: exactly once, then deleted from the store
        await world.advance(30)
        assert _sent_contents(ch).count("scheduled hello") == 1
    finally:
        sched.scheduler.tasks.clear()


async def test_schedule_message_interval_repeats(world):
    w = await _ready(world)
    boss = world.person("boss", user_id=OWNER_ID)
    ch = world.channel()
    world.llm.agent_planner = _exec([
        ("schedule_message",
         {"channel_query": "here", "content": "tick",
          "delay_s": 2, "interval_s": 10, "label": "matrix-repeat"}),
    ])
    msg = _msg(world, boss, "repeat tick every 10 seconds", channel=ch)
    assert w.queue_text_action(
        msg, "repeat tick every 10 seconds", {"scheduling"}) is True
    from src.action_engine.core import scheduler as sched
    try:
        await world.advance(40)
        n = _sent_contents(ch).count("tick")
        assert n >= 2, f"interval task fired {n}x, expected repeats: \
            {_sent_contents(ch)}"
    finally:
        sched.scheduler.tasks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Tool failure surface — errors feed back to the agent, nothing dies
# ─────────────────────────────────────────────────────────────────────────────

async def test_tool_error_result_reaches_agent(world):
    """A clean tool error (channel not found) is a normal tool result —
    the agent sees it and can still produce a final answer."""
    w = await _ready(world)
    world.llm.agent_planner = _planner(
        ("", [("send_message",
               {"channel_query": "channel-that-does-not-exist",
                "content": "hi"})]),
        ("sorry, couldn't find that channel", []),
    )
    msg = _msg(world, world.alice, "send hi to nowhere")
    out = await w.run_text_info(msg, "send hi to nowhere")
    assert out == "sorry, couldn't find that channel"
    # the error JSON was fed back as a tool message on round 2
    second = world.pool.calls[-1]
    assert '"error"' in second["user"], \
        "tool error never reached the agent's context"
    agent = w._agent
    assert ("send_message", False) in agent.ctx.performed


async def test_tool_exception_reaches_agent(world):
    """A tool that RAISES is caught by dispatch, serialised to an error,
    and fed back — the worker + conversation survive."""
    w = await _ready(world)
    from src.action_engine.tools import registry as reg
    orig = reg._DISPATCH["send_message"]

    async def _boom(ctx, **kw):
        raise RuntimeError("kaboom")

    reg._DISPATCH["send_message"] = _boom
    try:
        world.llm.agent_planner = _planner(
            ("", [("send_message",
                   {"channel_query": "here", "content": "hi"})]),
            ("it didn't work", []),
        )
        msg = _msg(world, world.alice, "send hi")
        out = await w.run_text_info(msg, "send hi")
        assert out == "it didn't work"
        assert "kaboom" in world.pool.calls[-1]["user"], \
            "exception text never reached the agent"
        assert ("send_message", False) in w._agent.ctx.performed
    finally:
        reg._DISPATCH["send_message"] = orig


async def test_agent_failure_does_not_kill_conversation(world):
    """After a failed exec the normal reply path still answers — the
    worker degrades silently instead of wedging the channel."""
    w = await _ready(world)
    ch = world.channel()
    bob = world.bob
    world.llm.agent_planner = _planner(
        ("", [("send_message",
               {"channel_query": "nowhere", "content": "hi"})]),
        ("(sorry, i had trouble processing that)", []),
    )
    await world.say(world.alice, f"ping <@{bob.id}> in #general")
    # conversational reply still landed even though the tool failed
    assert world.bot_said_something(), "reply path died with the tool"
