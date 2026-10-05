"""Shared context handed to every tool invocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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

    def require_guild(self) -> discord.Guild:
        if self.guild is None:
            raise RuntimeError("No guild resolved yet — join a server first.")
        return self.guild

    def get_current_channel(self):
        """Return the channel the user is currently talking in, or None."""
        if self.current_channel_id and self.guild:
            return self.guild.get_channel(self.current_channel_id)
        return None
