"""SimWorld — the Discord-like test environment.

Builds a REAL ``AIPersonaClient`` wired to fake guilds/channels/users and a
fake LLM brain, then lets tests drive a scripted Discord scenario:

    world = await SimWorld.build(persona="eudora")
    await world.say(world.alice, "hey eudora")
    assert world.channel().bot_sent()

Design notes
------------
* ``say()`` dispatches ``client.on_message`` as a background task and then
  advances the virtual clock (default 60s) so read/typing delays elapse
  instantly — one call = one full human-perceived message cycle.
* The fake LLM answers prompts deterministically but realistically;
  ``world.llm`` exposes all recorded LLM calls for assertions.
* Everything the bot "sees" is fake: no network, no Discord, no Groq.
"""
from __future__ import annotations

import asyncio
import os
import random
import time
from typing import List, Optional

import discord

from . import fake_discord as fd
from .clock import ClockPatch, VirtualClock
from .fake_groq import ScriptedLLM, install_llm_fakes, install_pool_fake


def _test_persona(persona_id: str = "eudora"):
    """A minimal PersonaProfile for tests (mirrors src/persona/profiles)."""
    from src.persona.profiles import PersonaProfile, get_profile
    try:
        return get_profile(persona_id)
    except Exception:
        return PersonaProfile(
            id=persona_id, token_env="DISCORD_TOKEN", name=persona_id,
            full_name=persona_id.capitalize(), age=22, gender="female",
            location="London, England, UK", heritage="british",
            bio="test bio", persona_block="you are a test persona",
            voice_identity="test", voice_style="casual",
            proactive_identity="test persona",
            short_identity="You are a test persona.",
        )


class SimWorld:
    def __init__(self):
        self.clock = VirtualClock()
        self._clock_patch = ClockPatch(self.clock).install()
        self.llm = ScriptedLLM()
        self.pool = None
        self.client = None
        self.guild: fd.FakeGuild = None
        self.bot_user: fd.FakeMember = None
        self.people: dict = {}
        self._msg_seq = 100

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    async def build(cls, persona: str = "eudora",
                    guild_name: str = "Sim Server",
                    channel_names=("general",), with_voice=True) -> "SimWorld":
        w = cls()
        install_llm_fakes(w.llm)
        w.pool, w._orig_get_pool = install_pool_fake(w.llm)

        from src.discord_client import AIPersonaClient
        profile = _test_persona(persona)
        client = AIPersonaClient(persona=profile)
        w.client = client

        # Guild + channels + bot identity
        w.guild = fd.FakeGuild(name=guild_name)
        w.bot_user = w.guild.add_member(
            fd.FakeUser(name=profile.name, display_name=profile.full_name))
        w.guild.me = w.bot_user
        for name in channel_names:
            w.guild.add_channel(name)
        if with_voice:
            w.guild.add_voice_channel("voice-general")

        # Inject fake state into the real client internals
        conn = client._connection
        conn.user = w.bot_user
        conn._guilds = {w.guild.id: w.guild}
        conn._users = {u.id: u for u in w.guild.members}
        conn._private_channels = {}
        conn._voice_clients = {}
        try:
            client.http.token = "fake-token"
        except Exception:
            pass
        client._ready = asyncio.Event()
        client._ready.set()

        # The HTTP endpoints on_ready touches — fake them on the instance
        async def _fake_profile(uid):
            return type("P", (), {"bio": "test bio", "pronouns": ""})()
        client.fetch_user_profile = _fake_profile

        async def _fake_presence(**kw):
            w.__dict__.setdefault("_presence_calls", []).append(kw)
        client.change_presence = _fake_presence

        # get_channel resolves through _connection._channels in discord.py —
        # route it through the fake guilds instead.
        client.get_channel = lambda cid: next(
            (g.get_channel(cid) for g in client.guilds if g.get_channel(cid)),
            None)

        # Self-message echo: Discord dispatches MESSAGE_CREATE for the
        # account's own sends — the client relies on that to cache its own
        # replies in history_cache (reply-to-us detection). Route every
        # channel send back through on_message like the real gateway.
        _orig_add_channel = w.guild.add_channel

        def _wire_dispatch(ch):
            ch.on_dispatch = lambda m: asyncio.create_task(client.on_message(m))
            return ch

        for _ch in w.guild.text_channels:
            _wire_dispatch(_ch)

        def _add_channel(name, topic=""):
            return _wire_dispatch(_orig_add_channel(name, topic))
        w.guild.add_channel = _add_channel

        w.people["bot"] = w.bot_user
        return w

    # ── people ───────────────────────────────────────────────────────
    def person(self, name: str, bot: bool = False,
               user_id: int = None) -> fd.FakeMember:
        """Get-or-create a server member the sim can puppet."""
        if name not in self.people:
            m = self.guild.add_member(fd.FakeUser(
                user_id=user_id, name=name, display_name=name, bot=bot))
            self.people[name] = m
            self.client._connection._users[m.id] = m
        return self.people[name]

    @property
    def alice(self):
        return self.person("alice")

    @property
    def bob(self):
        return self.person("bob")

    @property
    def carol(self):
        return self.person("carol")

    # ── channels ─────────────────────────────────────────────────────
    def channel(self, name: str = "general") -> fd.FakeTextChannel:
        for ch in self.guild.text_channels:
            if ch.name == name:
                return ch
        raise KeyError(name)

    # ── driving the world ────────────────────────────────────────────
    def _resolve_mentions(self, content: str,
                          mentions: list = None) -> list:
        """Parse <@id> / <@!id> tokens out of content like Discord does and
        merge with any explicitly passed mention objects."""
        found = list(mentions or [])
        import re
        for mid in re.findall(r"<@!?(\d+)>", content or ""):
            m = self.guild.get_member(int(mid))
            if m and m not in found:
                found.append(m)
        return found

    async def say(self, person: fd.FakeUser, content: str,
                  channel: fd.FakeTextChannel = None,
                  mentions: list = None, reply_to: fd.FakeMessage = None,
                  advance: float = 60.0, **msg_kw) -> fd.FakeMessage:
        """A person speaks; bot processes it; virtual time advances."""
        channel = channel or self.channel()
        ref = None
        if reply_to is not None:
            ref = fd.FakeMessageReference(message_id=reply_to.id, resolved=reply_to)
        msg = fd.make_message(channel, person, content,
                              at=self.clock.time(),
                              mentions=self._resolve_mentions(content, mentions),
                              reference=ref, **msg_kw)
        task = asyncio.create_task(self.client.on_message(msg))
        # Advance virtual time so read/typing/send delays elapse
        await self.clock.advance(advance)
        # Let any trailing task work finish
        if not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            await self.clock.flush()
        return msg

    async def say_many(self, person, contents: list, channel=None,
                       advance_each: float = 60.0):
        for c in contents:
            await self.say(person, c, channel=channel, advance=advance_each)

    async def advance(self, seconds: float):
        await self.clock.advance(seconds)

    async def quiet(self, seconds: float):
        """Let background loops run for `seconds` of virtual time."""
        await self.clock.advance(seconds)

    # ── assertions helpers ───────────────────────────────────────────
    def sent(self, channel: fd.FakeTextChannel = None) -> List[str]:
        """Contents of everything the bot sent to a channel."""
        channel = channel or self.channel()
        return [e["content"] for e in channel.sent_log]

    def sent_messages(self, channel: fd.FakeTextChannel = None) -> List[fd.FakeMessage]:
        channel = channel or self.channel()
        return [e["message"] for e in channel.sent_log]

    def bot_said_something(self, channel: fd.FakeTextChannel = None) -> bool:
        return bool(self.sent(channel))

    def last_sent(self, channel: fd.FakeTextChannel = None) -> Optional[str]:
        s = self.sent(channel)
        return s[-1] if s else None

    def reacted(self, msg: fd.FakeMessage, emoji: str = None) -> bool:
        for r in msg.reactions:
            if r.me and (emoji is None or r.emoji == emoji):
                return True
        return False

    def llm_calls(self, match: str = None) -> list:
        if match is None:
            return list(self.llm.calls)
        return [c for c in self.llm.calls if match.lower() in (c["system"] + c["user"]).lower()]

    async def teardown(self):
        try:
            await self.client.teardown()
        except Exception:
            pass
        self._clock_patch.uninstall()

    # ── random seed helper ───────────────────────────────────────────
    @staticmethod
    def seed(n: int = 42):
        random.seed(n)
