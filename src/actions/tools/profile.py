"""Profile & presence tools: change nickname, status, bio, avatar, activity.

discord.py-self v2.1.0 exposes:
  - Client.change_presence(status=..., activity=...) for online status/custom status
  - Member.edit(nick=..., bio=..., avatar=..., banner=...) for profile changes
"""
from __future__ import annotations

from typing import Any

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.logger import logger


async def change_nickname(ctx: ToolContext, nick: str) -> dict:
    """Change the bot's nickname in the guild."""
    guild = ctx.require_guild()
    try:
        await guild.me.edit(nick=nick)
        return {"ok": True, "nickname": nick}
    except discord.HTTPException as e:
        return {"error": f"Nickname change failed: {e}"}


async def change_status(ctx: ToolContext, status: str) -> dict:
    """Change online status. Options: online, idle, dnd, invisible, offline."""
    status_map = {
        "online": discord.Status.online,
        "idle": discord.Status.idle,
        "dnd": discord.Status.do_not_disturb,
        "do_not_disturb": discord.Status.do_not_disturb,
        "invisible": discord.Status.invisible,
        "offline": discord.Status.offline,
    }
    status_lower = status.lower().strip()
    discord_status = status_map.get(status_lower)
    if discord_status is None:
        return {"error": f"Invalid status '{status}'. Use: {', '.join(status_map.keys())}"}
    try:
        await ctx.bot.change_presence(status=discord_status)
        return {"ok": True, "status": status_lower}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Status change failed: {e}"}


async def change_custom_status(ctx: ToolContext, text: str, emoji: str | None = None) -> dict:
    """Set a custom status text (the 'Playing...' or custom text under your name)."""
    try:
        # discord.py-self supports CustomActivity
        activity = discord.CustomActivity(name=text, emoji=emoji) if emoji else discord.CustomActivity(name=text)
        await ctx.bot.change_presence(activity=activity)
        return {"ok": True, "custom_status": text}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Custom status failed: {e}"}


async def change_bio(ctx: ToolContext, bio: str) -> dict:
    """Change the bot's bio / about me (global, not guild-specific)."""
    try:
        await ctx.bot.user.edit(bio=bio)
        return {"ok": True, "bio": bio}
    except discord.HTTPException as e:
        return {"error": f"Bio change failed: {e}"}


async def get_my_profile(ctx: ToolContext) -> dict:
    """Get the bot's own profile info (nick, status, bio, avatar)."""
    guild = ctx.require_guild()
    me = guild.me
    return {
        "username": str(me),
        "display_name": me.display_name,
        "nickname": me.nick,
        "id": me.id,
        "status": str(me.status) if me.status else "unknown",
        "bio": getattr(me, "bio", None) or getattr(me.public_flags, "bio", None) if hasattr(me, "public_flags") else None,
        "avatar_url": str(me.avatar.url) if me.avatar else None,
        "roles": [r.name for r in me.roles if r.name != "@everyone"],
    }
