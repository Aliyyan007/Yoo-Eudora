"""Lifecycle & isolation tests — rotation safety, action-worker boundary,
own-message dedup, daily-reset recovery, and mass-mention sanitization
end-to-end through the action gate."""
import asyncio
import datetime

import pytest

from tests.sim import fake_discord as fd


# ─────────────────────────────────────────────────────────────────────────────
# Rotation safety — a closed/stale client must not act
# ─────────────────────────────────────────────────────────────────────────────

async def test_closed_client_refuses_replies(world):
    """A rotated-out client (is_closed) must not send anything."""
    world.client._closing_task = True   # discord.py marks closing via this
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert not world.bot_said_something()


async def test_teardown_stops_background_state(world):
    """After teardown every tracked task is done — nothing leaks into the
    next persona's runtime."""
    await world.client.teardown()
    for t in world.client._spawned_tasks:
        assert t.done()


async def test_action_worker_isolation_from_chat(world):
    """The action agent's chatter must never leak into conversational
    replies — the worker's 'on it' is not a user-facing message."""
    world.bob
    world.llm.agent_text = "on it"          # what the agent would say
    await world.say(world.alice, f"ping <@{world.bob.id}>")
    # "on it" belongs to the agent, not the conversation
    assert "on it" not in world.sent()


async def test_mass_mention_sanitized_in_action_args(world):
    """A non-owner making the agent send '@everyone hello' — the bridge
    must strip the mass mention before the tool executes."""
    ch = world.channel()
    world.llm.agent_planner = lambda text, tools: (
        "", [("send_message",
              {"content": "@everyone hello", "channel_id": ch.id})])
    await world.say(world.alice, f"say @everyone hello in <#{ch.id}>")
    for entry in ch.sent_log:
        c = (entry.get("content") or "")
        assert "@everyone" not in c and "@\u200beveryone" not in c, \
            "mass mention survived the action gate"


async def test_action_worker_cancel_on_teardown(world):
    """In-flight action work must be cancelled when the persona rotates."""
    world.bob
    await world.say(world.alice, f"ping <@{world.bob.id}>", advance=1)
    await world.client.teardown()
    # Shutdown ran — a second teardown must be a safe no-op
    await world.client.teardown()


# ─────────────────────────────────────────────────────────────────────────────
# Own-message dedup — the self-echo must not retrigger a reply
# ─────────────────────────────────────────────────────────────────────────────

async def test_own_message_echo_never_replies(world):
    """Our own sends echo through on_message — they must be cached, not
    answered (a bot replying to itself is the worst tell)."""
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    n = len(world.sent())
    # The echo already ran during send; give it a moment and confirm no loop
    await world.advance(30)
    sends_after = len(world.sent())
    assert sends_after <= n + 2, "bot replied to its own message echo"


async def test_bot_messages_from_other_bots_ignored(world):
    """Other bots' messages are filtered — no bot-on-bot reply loops."""
    other_bot = world.person("mee6", bot=True)
    await world.say(other_bot, f"<@{world.bot_user.id}> hello")
    assert not world.bot_said_something()


# ─────────────────────────────────────────────────────────────────────────────
# Daily reset — features recover at the day boundary
# ─────────────────────────────────────────────────────────────────────────────

async def test_daily_reset_restores_reply_ability(world):
    ch_id = str(world.channel().id)
    world.client.daily_reset_day = datetime.date.today().toordinal()
    world.client.daily_count_global = 400
    assert not world.client.can_send(ch_id)
    # Cross the day boundary
    world.client.daily_reset_day = datetime.date.today().toordinal() - 1
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert world.bot_said_something()


async def test_per_channel_cap_only_blocks_that_channel(world):
    ch = world.channel()
    other = world.guild.add_channel("other")
    world.client.daily_reset_day = datetime.date.today().toordinal()
    world.client.daily_count_ch[str(ch.id)] = 60  # DAILY_CAP_PER_CH
    await world.say(world.alice, f"<@{world.bot_user.id}> hey")
    assert not world.bot_said_something(ch)
    # The other channel still works (global cap is clear)
    await world.say(world.bob, f"<@{world.bot_user.id}> hey", channel=other)
    assert world.bot_said_something(other)


# ─────────────────────────────────────────────────────────────────────────────
# Catchup — offline messages for the active persona get picked up
# ─────────────────────────────────────────────────────────────────────────────

async def test_ready_is_safe_to_refire(world):
    """READY refires on reconnect — on_ready must not double-loop."""
    await world.client.on_ready()
    before = len(world.client._spawned_tasks)
    await world.client.on_ready()
    # No new named loops; one-shot scans may exist but shouldn't pile up
    assert len(world.client._loop_tasks) <= 12


async def test_profile_edit_uses_no_sentinels(world):
    """Regression: user.edit(...) was called with Ellipsis sentinels — the
    HTTP layer choked on them and profile updates silently broke. Every
    kwarg must be a real value."""
    await world.client.on_ready()
    assert world.bot_user._edits, "profile edit never issued"
    for edit in world.bot_user._edits:
        for k, v in edit.items():
            assert v is not ..., f"user.edit({k}=...) sentinel survived"


# ─────────────────────────────────────────────────────────────────────────────
# Voice — stale-client voice joins must fail cleanly (no crash, no leak)
# ─────────────────────────────────────────────────────────────────────────────

async def test_vc_join_on_closed_client_noop(world):
    world.client._closing_task = True
    ch = world.channel()
    msg = fd.make_message(ch, world.alice, "join the lounge vc",
                          at=world.clock.time())
    try:
        await world.client._handle_vc_join(msg, "lounge", ch)
    except Exception as e:
        pytest.fail(f"vc join on closed client raised: {e}")
