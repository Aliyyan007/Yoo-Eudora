"""Slash command tools: use application commands like /bump via discord.py-self.

discord.py-self v2.1.0 supports slash commands via:
  cmds = await channel.application_commands()
  cmd = next(c for c in cmds if isinstance(c, discord.SlashCommand) and c.name == "bump")
  interaction = await cmd(channel)
"""
from __future__ import annotations

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search


async def list_slash_commands(ctx: ToolContext, channel_query: str = "here") -> dict:
    """List all available slash commands in a channel."""
    from src.actions.tools.messaging import _resolve_text_channel
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}
    try:
        cmds = await ch.application_commands()
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to fetch slash commands: {e}"}
    result = []
    for cmd in cmds:
        info = {
            "name": cmd.name,
            "type": type(cmd).__name__,
            "description": getattr(cmd, "description", ""),
            "application_id": str(getattr(cmd, "application_id", "")),
        }
        result.append(info)
    return {"channel": ch.name, "commands": result}


async def use_slash_command(
    ctx: ToolContext,
    command_name: str,
    channel_query: str = "here",
    application_id: str | None = None,
) -> dict:
    """Use a slash command (e.g. /bump) in a channel.

    If `application_id` is provided, targets a specific bot's command
    (useful when multiple bots register the same /bump command).
    """
    # Strip leading '/' if the agent included it (e.g. "/bump" -> "bump").
    command_name = command_name.lstrip("/").strip()
    from src.actions.tools.messaging import _resolve_text_channel
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}
    try:
        cmds = await ch.application_commands()
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to fetch slash commands: {e}"}

    # Find the command by name (and optionally by application_id).
    slash_cmds = [c for c in cmds if isinstance(c, discord.SlashCommand)]
    cmd = None
    if application_id:
        # Target a specific bot by application_id.
        cmd = next(
            (c for c in slash_cmds
             if c.name.lower() == command_name.lower()
             and str(getattr(c, "application_id", "")) == str(application_id)),
            None,
        )
    if cmd is None:
        # Fall back to first match by name.
        cmd = next((c for c in slash_cmds if c.name.lower() == command_name.lower()), None)
    if cmd is None:
        # Fuzzy match.
        results = fuzzy_search(command_name, slash_cmds, key=lambda c: c.name, limit=1)
        if results:
            cmd = results[0].item
    if cmd is None:
        available = [c.name for c in slash_cmds]
        return {"error": f"No slash command '{command_name}'. Available: {available}"}

    try:
        interaction = await cmd(ch)
        return {
            "ok": True,
            "command": cmd.name,
            "channel": ch.name,
            "application_id": str(getattr(cmd, "application_id", "")),
            "interaction_id": getattr(interaction, "id", None),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Slash command '{command_name}' failed: {e}"}
