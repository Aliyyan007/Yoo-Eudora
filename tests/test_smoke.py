"""Smoke tests — validate the simulation harness end to end before the
real scenario suite. If these pass, the fake Discord + fake LLM + virtual
clock all work."""
import pytest


async def test_client_constructs(world):
    from src.discord_client import AIPersonaClient
    assert isinstance(world.client, AIPersonaClient)
    assert world.client.user.id == world.bot_user.id
    assert world.guild in world.client.guilds


async def test_mention_triggers_reply(world):
    world.seed(0)
    await world.say(world.alice, f"<@{world.bot_user.id}> yo")
    assert world.bot_said_something(), "bot must reply to a direct mention"


async def test_ambient_message_may_stay_silent(world):
    world.seed(0)
    # Ambient unaddressed message — reply is probabilistic, never an error
    await world.say(world.alice, "just had the worst day at work")
    # No assertion on send — just that the pipeline didn't crash.


async def test_no_reply_when_addressed_to_other(world):
    world.seed(0)
    await world.say(world.alice, f"hey <@{world.bob.id}> how are you",
                    mentions=[world.bob])
    assert not world.bot_said_something(), \
        "bot must not reply to messages addressed to another user"


async def test_virtual_clock_advances(world):
    t0 = world.clock.time()
    await world.advance(3600)
    assert world.clock.time() == t0 + 3600


async def test_dm_from_non_owner_ignored(world):
    from tests.sim import fake_discord as fd
    dm = fd.FakeDMChannel(recipient=world.bot_user)
    msg = fd.FakeMessage("hey there", world.alice, dm)
    import asyncio
    task = asyncio.create_task(world.client.on_message(msg))
    await world.advance(60)
    if not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except Exception:
            pass
    assert not dm.sent_log, "non-owner DM must be ignored"
