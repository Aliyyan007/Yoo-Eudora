"""Messaging tools: send text, ping users/roles, mention channels, reply.

These are SIDE-EFFECTING tools — they actually post to Discord.

Key design: the `channel_query` accepts:
  - "here" / "this" → the channel the user is currently talking in
  - A channel ID or <#id> mention → direct resolution
  - A fuzzy name → fuzzy search
"""
from __future__ import annotations

import discord

from src.actions.tools.context import ToolContext
from src.actions.tools.channels import resolve_channel, _try_direct_channel
from src.actions.tools.members import resolve_member
from src.actions.tools.roles import resolve_role_sync
from src.actions.utils.fuzzy import fuzzy_search
from src.actions.utils.logger import logger


async def _resolve_text_channel(ctx: ToolContext, query: str):
    """Resolve a text-like channel from a query string.

    Handles: 'here'/'this' (current channel), direct IDs, <#id> mentions,
    and fuzzy name search.
    """
    query_lower = query.strip().lower() if query else ""

    # "here" / "this" → use the current channel the user is talking in.
    if query_lower in ("here", "this", "current"):
        ch = ctx.get_current_channel()
        if ch and isinstance(ch, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
            return ch
        # Fall through to fuzzy search if current channel not available.

    # Direct ID or <#id> mention.
    ch = _try_direct_channel(ctx, query)
    if ch and isinstance(ch, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
        return ch

    # Fuzzy search.
    guild = ctx.require_guild()
    text_like = [
        c for c in guild.channels
        if isinstance(c, (discord.TextChannel, discord.Thread, discord.VoiceChannel))
    ]
    results = fuzzy_search(query, text_like, key=lambda c: c.name, limit=1)
    if not results:
        return None
    return results[0].item


async def send_message(
    ctx: ToolContext,
    channel_query: str,
    content: str,
    *,
    ping_users: list[str] | None = None,
    ping_roles: list[str] | None = None,
    mention_channels: list[str] | None = None,
    reply_to_link: str | None = None,
) -> dict:
    """Send a message to a channel.

    - `channel_query`: 'here'/'this' for current channel, or fuzzy name/ID.
    - `ping_users`: list of fuzzy user names to ping (turned into <@id>).
    - `ping_roles`: list of fuzzy role names to ping (turned into <@&id>).
    - `mention_channels`: list of fuzzy channel names to mention (<#id>).
    - `reply_to_link`: a message jump link to reply to.
    """
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}

    parts: list[str] = []
    # Role pings first (so they appear at the start, natural).
    for r in (ping_roles or []):
        res = resolve_role_sync(ctx, r)
        if res:
            parts.append(res["mention"])
        else:
            parts.append(f"@{r}")  # fallback literal
    # User pings.
    for u in (ping_users or []):
        res = await resolve_member(ctx, u)
        if "mention" in res:
            parts.append(res["mention"])
        else:
            parts.append(f"@{u}")  # fallback literal
    # Channel mentions inline.
    for c in (mention_channels or []):
        res = await resolve_channel(ctx, c)
        if "mention" in res:
            parts.append(res["mention"])
    if content:
        parts.append(content)
    final = " ".join(parts) if parts else content

    if len(final) > 2000:
        final = final[:1997] + "..."

    reply_target = None
    if reply_to_link:
        try:
            p = [x for x in reply_to_link.split("/") if x.isdigit()]
            cid, mid = int(p[-2]), int(p[-1])
            tch = ctx.bot.get_channel(cid) or await ctx.bot.fetch_channel(cid)
            reply_target = await tch.fetch_message(mid)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"reply link resolve failed: {e}")

    try:
        if reply_target is not None and hasattr(ch, "send"):
            sent = await ch.send(final, reference=reply_target)
        else:
            sent = await ch.send(final)
    except discord.HTTPException as e:
        return {"error": f"Send failed: {e}"}
    ctx.did_send = True   # suppress the agent's own reply — avoid double send
    return {"ok": True, "channel": ch.name, "message_id": sent.id, "sent": final}


async def cleanup_my_messages(ctx: ToolContext, channel_query: str) -> dict:
    """Sweep the bot's own stale messages in a channel (older than 6h or
    beyond 15). Owner-facing hygiene — 'clean up your messages in #x'."""
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No text channel matching '{channel_query}'."}
    from src.actions.core import self_cleanup
    deleted = await self_cleanup.sweep_channel(ctx.bot, ch)
    return {"ok": True, "channel": ch.name, "deleted": deleted}


async def send_dm(ctx: ToolContext, user_query: str, content: str) -> dict:
    """Send a DM to a user (fuzzy name)."""
    res = await resolve_member(ctx, user_query)
    if "error" in res:
        return res
    user = ctx.bot.get_user(res["id"]) or await ctx.bot.fetch_user(res["id"])
    try:
        await user.send(content)
    except discord.HTTPException as e:
        return {"error": f"DM failed: {e}"}
    ctx.did_send = True
    return {"ok": True, "user": res["name"], "sent": content}


async def delete_message(
    ctx: ToolContext,
    message_link: str | None = None,
    *,
    channel_query: str = "here",
    message_id: str | None = None,
) -> dict:
    """Delete a message. Can delete by link or by channel+message_id.

    Only deletes the bot's own messages (self-bots can't delete others' messages).
    """
    # Resolve the target message.
    target = None
    if message_link:
        try:
            p = [x for x in message_link.split("/") if x.isdigit()]
            if len(p) < 3:
                return {"error": "Invalid message link."}
            cid, mid = int(p[-2]), int(p[-1])
            ch = ctx.bot.get_channel(cid) or await ctx.bot.fetch_channel(cid)
            target = await ch.fetch_message(mid)
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not fetch message: {e}"}
    elif message_id:
        ch = await _resolve_text_channel(ctx, channel_query)
        if ch is None:
            return {"error": f"No channel matching '{channel_query}'."}
        try:
            target = await ch.fetch_message(int(message_id))
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not fetch message: {e}"}
    else:
        return {"error": "Provide either message_link or message_id."}

    if target is None:
        return {"error": "Message not found."}

    # Check if it's the bot's own message.
    if target.author.id != ctx.bot.user.id:
        return {"error": "I can only delete my own messages."}

    try:
        await target.delete()
    except discord.HTTPException as e:
        return {"error": f"Delete failed: {e}"}
    return {"ok": True, "deleted": True, "message_id": target.id}


async def delete_last_message(
    ctx: ToolContext,
    channel_query: str = "here",
    count: int = 1,
) -> dict:
    """Delete the bot's last N messages in a channel."""
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}

    deleted = 0
    async for msg in ch.history(limit=50):
        if msg.author.id == ctx.bot.user.id:
            try:
                await msg.delete()
                deleted += 1
                if deleted >= count:
                    break
            except discord.HTTPException as e:
                logger.warning(f"Delete failed: {e}")
                break
    return {"ok": True, "deleted_count": deleted, "channel": ch.name}


async def edit_message(
    ctx: ToolContext,
    message_link: str | None = None,
    new_content: str = "",
    *,
    channel_query: str = "here",
    message_id: str | None = None,
) -> dict:
    """Edit the bot's own message. Provide new_content and either message_link or message_id."""
    target = None
    if message_link:
        try:
            p = [x for x in message_link.split("/") if x.isdigit()]
            if len(p) < 3:
                return {"error": "Invalid message link."}
            cid, mid = int(p[-2]), int(p[-1])
            ch = ctx.bot.get_channel(cid) or await ctx.bot.fetch_channel(cid)
            target = await ch.fetch_message(mid)
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not fetch message: {e}"}
    elif message_id:
        ch = await _resolve_text_channel(ctx, channel_query)
        if ch is None:
            return {"error": f"No channel matching '{channel_query}'."}
        try:
            target = await ch.fetch_message(int(message_id))
        except Exception as e:  # noqa: BLE001
            return {"error": f"Could not fetch message: {e}"}
    else:
        return {"error": "Provide either message_link or message_id."}

    if target is None:
        return {"error": "Message not found."}

    if target.author.id != ctx.bot.user.id:
        return {"error": "I can only edit my own messages."}

    if len(new_content) > 2000:
        new_content = new_content[:1997] + "..."

    try:
        await target.edit(content=new_content)
    except discord.HTTPException as e:
        return {"error": f"Edit failed: {e}"}
    return {"ok": True, "edited": True, "message_id": target.id, "new_content": new_content}
