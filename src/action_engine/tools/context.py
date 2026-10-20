"""Shared context handed to every tool invocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import discord


@dataclass
class ToolContext:
    """Runtime handle: the self-bot client + the primary guild we operate on.

    The `current_channel_id` is set per-message so tools can resolve
    "here" / "this" to the channel the user is talking in.
    """

    bot: discord.Client
    guild: Optional[discord.Guild] = None
    current_channel_id: Optional[int] = None
    # Per-turn state — set by the trigger path each message:
    author_id: Optional[int] = None   # who asked (owner-gating in tools)
    did_send: bool = False            # a messaging tool already delivered
    is_owner: bool = False            # requester is a configured owner
    # Callable returning the CURRENT live client. The bridge installs this
    # so deferred work (auto-bump timers, scheduled jobs) rebinds to the
    # new client after a persona rotation instead of dying on the old one.
    bot_getter: Optional[Callable[[], "discord.Client"]] = None

    def require_guild(self) -> discord.Guild:
        if self.guild is None:
            raise RuntimeError(
                "Can't tell which server to act in — the request came "
                "from DMs with no target channel. Ask which server.")
        return self.guild

    def get_current_channel(self):
        """Return the channel the user is currently talking in, or None."""
        if self.current_channel_id and self.guild:
            return self.guild.get_channel(self.current_channel_id)
        return None

    def get_live_bot(self):
        """The current live client. After a persona rotation `self.bot`
        may be a closed connection; prefer this for deferred work that
        fires long after the ctx was created."""
        if self.bot_getter is not None:
            try:
                bot = self.bot_getter()
                if bot is not None:
                    return bot
            except Exception:
                pass
        return self.bot
