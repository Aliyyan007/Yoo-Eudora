"""Fake Discord object model for the simulation environment.

Duck-typed stand-ins for the discord.py-self objects the bot touches:
users/members, messages, text/voice channels, guilds, references,
reactions, attachments. Every attribute the production code reads is
present; every method it awaits is a real coroutine. Channels record
everything sent to them so tests can assert on bot output.
"""
from __future__ import annotations

import itertools
import time as _time
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Optional


_id_counter = itertools.count(10_000_000)


def next_id() -> int:
    return next(_id_counter)


def _snowflake_created_at(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Users / members
# ─────────────────────────────────────────────────────────────────────────────

class FakeUser:
    def __init__(self, user_id: int = None, name: str = "user",
                 display_name: str = None, bot: bool = False):
        self.id = user_id if user_id is not None else next_id()
        self.name = name
        self.display_name = display_name or name
        self.global_name = display_name or name
        self.bot = bot
        self.avatar = None
        self.discriminator = "0"
        self._edits: List[dict] = []      # recorded profile edits
        self.status = "online"

    @property
    def mention(self) -> str:
        return f"<@{self.id}>"

    def __eq__(self, other):
        return getattr(other, "id", None) == self.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<FakeUser {self.name}#{self.id}>"

    # ClientUser endpoints used by on_ready / status_manager
    async def edit(self, **kwargs):
        self._edits.append(kwargs)
        if "global_name" in kwargs and kwargs["global_name"] is not ...:
            self.global_name = kwargs["global_name"]
            self.display_name = kwargs["global_name"]
        return self


class FakeMember(FakeUser):
    def __init__(self, *args, guild=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.guild = guild
        self.nick = None
        self.roles: List[Any] = []
        self.joined_at = _snowflake_created_at(_time.time())
        self.premium_since = None

    @property
    def display_name(self):  # noqa: F811 — discord.Member computes it
        return self.nick or self.global_name or self.name

    @display_name.setter
    def display_name(self, value):
        self.global_name = value


class FakeRole:
    def __init__(self, role_id: int = None, name: str = "role"):
        self.id = role_id if role_id is not None else next_id()
        self.name = name

    @property
    def mention(self):
        return f"<@&{self.id}>"


# ─────────────────────────────────────────────────────────────────────────────
# Attachments / embeds / reactions / references
# ─────────────────────────────────────────────────────────────────────────────

class FakeAttachment:
    def __init__(self, url: str = "https://cdn.example/img.png",
                 content_type: str = "image/png"):
        self.url = url
        self.content_type = content_type
        self.filename = url.rsplit("/", 1)[-1]


class FakeEmbed:
    def __init__(self, url: str = None):
        self.url = url


class FakeReaction:
    def __init__(self, emoji: str, count: int = 1, me: bool = False):
        self.emoji = emoji
        self.count = count
        self.me = me


class FakeMessageReference:
    def __init__(self, message_id: int = None, resolved=None, cached_message=None):
        self.message_id = message_id
        self.resolved = resolved
        self.cached_message = cached_message
        self.channel_id = None


class FakeReadState:
    def __init__(self, last_acked_id: int = 0):
        self.last_acked_id = last_acked_id


class FakePermissions:
    def __init__(self, send=True, read=True, history=True):
        self.send_messages = send
        self.read_messages = read
        self.read_message_history = history


class _TypingCtx:
    def __init__(self):
        self.enters = 0

    async def __aenter__(self):
        self.enters += 1
        return self

    async def __aexit__(self, *exc):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, content: str, author: FakeUser, channel,
                 msg_id: int = None, created_at: float = None,
                 mentions: List[FakeUser] = None,
                 reference: FakeMessageReference = None,
                 attachments: List[FakeAttachment] = None,
                 embeds: List[FakeEmbed] = None,
                 webhook_id: int = None):
        self.id = msg_id if msg_id is not None else next_id()
        self.content = content
        self.author = author
        self.channel = channel
        self.created_at = _snowflake_created_at(created_at if created_at is not None else _time.time())
        self.mentions = list(mentions or [])
        self.reference = reference
        self.attachments = list(attachments or [])
        self.embeds = list(embeds or [])
        self.webhook_id = webhook_id
        self.reactions: List[FakeReaction] = []
        self.role_mentions: List[FakeRole] = []
        self.edited_at = None

    @property
    def clean_content(self) -> str:
        # Discord resolves <@id> to display names — approximation is enough
        text = self.content
        for u in self.mentions:
            text = text.replace(f"<@{u.id}>", f"@{u.display_name}")
        return text

    @property
    def guild(self):
        return getattr(self.channel, "guild", None)

    @property
    def jump_url(self):
        return f"https://discord.test/{self.id}"

    async def reply(self, content: str = None, **kwargs):
        return await self.channel.send(content, reply_to=self, **kwargs)

    async def add_reaction(self, emoji: str):
        for r in self.reactions:
            if r.emoji == emoji:
                r.count += 1
                return
        self.reactions.append(FakeReaction(emoji, count=1, me=True))

    async def delete(self):
        if self in self.channel.messages:
            self.channel.messages.remove(self)
        self.channel.deleted.append(self)


# ─────────────────────────────────────────────────────────────────────────────
# Channels
# ─────────────────────────────────────────────────────────────────────────────

class FakeTextChannel:
    def __init__(self, channel_id: int = None, name: str = "general",
                 guild=None, topic: str = ""):
        self.id = channel_id if channel_id is not None else next_id()
        self.name = name
        self.guild = guild
        self.topic = topic
        self.messages: List[FakeMessage] = []
        self.sent_log: List[dict] = []     # every send() call: {content, kwargs}
        self.deleted: List[FakeMessage] = []
        self.mention_count = 0
        self.read_state = FakeReadState()
        self._perms = FakePermissions()
        self.typing_calls = 0
        # Set by the world — echoes our own sends back through on_message,
        # exactly as the real gateway does for a user account (self-bots
        # see their own MESSAGE_CREATE; the client relies on that echo to
        # cache its own messages in history_cache).
        self.on_dispatch = None

    @property
    def mention(self):
        return f"<#{self.id}>"

    @property
    def last_message_id(self):
        return self.messages[-1].id if self.messages else None

    def permissions_for(self, member) -> FakePermissions:
        return self._perms

    def typing(self):
        self.typing_calls += 1
        ctx = _TypingCtx()
        self._last_typing = ctx
        return ctx

    async def send(self, content: str = None, reply_to: FakeMessage = None,
                   stickers=None, **kwargs) -> FakeMessage:
        bot_user = getattr(self.guild, "me", None) if self.guild else None
        author = bot_user if bot_user is not None else FakeUser(name="self")
        ref = FakeMessageReference(message_id=reply_to.id, resolved=reply_to) if reply_to else None
        msg = FakeMessage(content or "", author, self, reference=ref)
        self.messages.append(msg)
        self.sent_log.append({"content": content, "kwargs": kwargs, "message": msg})
        if self.on_dispatch is not None:
            self.on_dispatch(msg)
        return msg

    async def fetch_message(self, msg_id: int) -> Optional[FakeMessage]:
        for m in self.messages:
            if m.id == msg_id:
                return m
        return None

    def history(self, limit: int = 50, oldest_first: bool = False,
                after: datetime = None, **_) -> AsyncIterator[FakeMessage]:
        msgs = [m for m in self.messages
                if after is None or m.created_at > after]
        msgs = sorted(msgs, key=lambda m: m.created_at,
                      reverse=not oldest_first)[:limit]

        async def _gen():
            for m in msgs:
                yield m
        return _gen()

    def get_member(self, user_id: int):
        return self.guild.get_member(user_id) if self.guild else None


class FakeVoiceChannel:
    def __init__(self, channel_id: int = None, name: str = "voice",
                 guild=None, members: List[FakeMember] = None):
        self.id = channel_id if channel_id is not None else next_id()
        self.name = name
        self.guild = guild
        self.members = list(members or [])


class FakeDMChannel:
    """Stand-in patched over discord.DMChannel for isinstance checks."""
    def __init__(self, channel_id: int = None, recipient: FakeUser = None):
        self.id = channel_id if channel_id is not None else next_id()
        self.recipient = recipient
        self.guild = None
        self.name = None
        self.messages: List[FakeMessage] = []
        self.sent_log: List[dict] = []

    def typing(self):
        return _TypingCtx()

    async def send(self, content: str = None, **kwargs) -> FakeMessage:
        author = kwargs.pop("_author", None) or FakeUser(name="self")
        msg = FakeMessage(content or "", author, self)
        self.messages.append(msg)
        self.sent_log.append({"content": content, "kwargs": kwargs, "message": msg})
        return msg

    def history(self, limit: int = 50, **kw):
        msgs = sorted(self.messages, key=lambda m: m.created_at, reverse=True)[:limit]

        async def _gen():
            for m in msgs:
                yield m
        return _gen()


# ─────────────────────────────────────────────────────────────────────────────
# Guild
# ─────────────────────────────────────────────────────────────────────────────

class FakeGuild:
    def __init__(self, guild_id: int = None, name: str = "Test Guild"):
        self.id = guild_id if guild_id is not None else next_id()
        self.name = name
        self.members: List[FakeMember] = []
        self.text_channels: List[FakeTextChannel] = []
        self.voice_channels: List[FakeVoiceChannel] = []
        self.roles: List[FakeRole] = []
        self.me: Optional[FakeMember] = None
        self.owner_id = 0

    def add_member(self, user: FakeUser) -> FakeMember:
        m = user if isinstance(user, FakeMember) else FakeMember(
            user_id=user.id, name=user.name, display_name=user.display_name,
            bot=user.bot)
        m.guild = self
        if m not in self.members:
            self.members.append(m)
        return m

    def add_channel(self, name: str, topic: str = "") -> FakeTextChannel:
        ch = FakeTextChannel(name=name, guild=self, topic=topic)
        self.text_channels.append(ch)
        return ch

    def add_voice_channel(self, name: str, members=None) -> FakeVoiceChannel:
        vc = FakeVoiceChannel(name=name, guild=self, members=members)
        self.voice_channels.append(vc)
        return vc

    def get_member(self, user_id: int) -> Optional[FakeMember]:
        for m in self.members:
            if m.id == user_id:
                return m
        return None

    def get_channel(self, channel_id: int):
        for ch in self.text_channels + self.voice_channels:
            if ch.id == channel_id:
                return ch
        return None

    def _resolve_channel(self, channel_id: int):
        """discord.py Guild private used by several code paths."""
        return self.get_channel(channel_id)

    def get_role(self, role_id: int):
        for r in self.roles:
            if r.id == role_id:
                return r
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Convenience message factory
# ─────────────────────────────────────────────────────────────────────────────

def make_message(channel: FakeTextChannel, author: FakeUser, content: str,
                 at: float = None, **kw) -> FakeMessage:
    msg = FakeMessage(content, author, channel, created_at=at, **kw)
    channel.messages.append(msg)
    return msg
