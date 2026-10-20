"""Messaging tools: send text, ping users/roles, mention channels, reply.

These are SIDE-EFFECTING tools — they actually post to Discord.

Key design: the `channel_query` accepts:
  - "here" / "this" → the channel the user is currently talking in
  - A channel ID or <#id> mention → direct resolution
  - A fuzzy name → fuzzy search
"""
from __future__ import annotations

import asyncio
import re
import time

import discord

from src.action_engine.tools.context import ToolContext
from src.action_engine.tools.channels import resolve_channel, _try_direct_channel
from src.action_engine.tools.members import resolve_member
from src.action_engine.tools.roles import resolve_role_sync
from src.action_engine.utils.fuzzy import fuzzy_search
from src.action_engine.utils.logger import logger


_BAD_MENTION = re.compile(r"<@[!&?]*([^0-9>\s][^>]*)>|<#([^0-9>\s][^>]*)>")

# Channel types the bot can post text into — includes stage channels
# (their text chat is messageable on discord.py-self).
_TEXT_LIKE = (
    discord.TextChannel,
    discord.Thread,
    discord.VoiceChannel,
    discord.StageChannel,
)


# ------------------------------------------------------------------
#  Slash-command list cache — application_commands() is heavily
#  rate-limited, so bump/slash tools share a short per-channel TTL
#  (mirrors BumpScheduler._get_bump_commands' 5-minute cache).
# ------------------------------------------------------------------
_APP_CMDS_TTL_S = 300
_app_cmds_cache: dict[int, tuple[float, list]] = {}


async def _application_commands(channel) -> list:
    """Cached wrapper around channel.application_commands()."""
    now = time.time()
    hit = _app_cmds_cache.get(channel.id)
    if hit and (now - hit[0]) < _APP_CMDS_TTL_S:
        return hit[1]
    cmds = await channel.application_commands()
    _app_cmds_cache[channel.id] = (now, list(cmds or []))
    return _app_cmds_cache[channel.id][1]


def _invalidate_app_commands(channel) -> None:
    """Drop the cached command list (a lookup missed — refetch next time)."""
    _app_cmds_cache.pop(getattr(channel, "id", 0), None)


def _sanitize_content(text: str) -> str:
    """The agent sometimes writes malformed mention syntax into `content`
    ('<@&?Rishab>', '<@Global Bot>') — Discord renders those as literal
    text garbage. Render the inner name as plain text instead. Real
    <@id>/<@!id>/<@&id>/<#id> mentions pass through untouched."""
    if not text or ("<@" not in text and "<#" not in text):
        return text
    return _BAD_MENTION.sub(
        lambda m: (m.group(1) or m.group(2) or "").strip(), text)


def _check_send_channel(ctx: ToolContext, ch) -> None:
    """Gate a resolved send target through the host's channel policy.

    Non-owner exec requests may NOT post into skip-channels (#bump,
    #bot-commands, #rules, #mod-log, ...) or channels the account can't
    actually speak in — the persona never talks there, a random user's
    request shouldn't make it. Owner intent is explicit: owners can post
    anywhere. Raises PermissionError when blocked (dispatch surfaces it
    as a normal tool error)."""
    if getattr(ctx, "is_owner", False):
        return
    try:
        from src.channel_scanner import can_speak_in, is_skip_channel
    except Exception:  # noqa: BLE001
        return  # scanner unavailable — an HTTP error will surface anyway
    name = getattr(ch, "name", "") or ""
    if is_skip_channel(name):
        raise PermissionError(
            f"Can't post in #{name} — it's a restricted channel.")
    guild = getattr(ch, "guild", None)
    me = getattr(guild, "me", None) if guild else None
    if me is not None and not can_speak_in(ch, me):
        raise PermissionError(
            f"Can't post in #{name} — no speak permission there.")


async def _resolve_text_channel(ctx: ToolContext, query: str):
    """Resolve a text-like channel from a query string.

    Handles: 'here'/'this' (current channel), direct IDs, <#id> mentions,
    and fuzzy name search. Resolved channels pass through the non-owner
    send gate (_check_send_channel).
    """
    query_lower = query.strip().lower() if query else ""
    ch = None

    # "here" / "this" → use the current channel the user is talking in.
    if query_lower in ("here", "this", "current"):
        cand = ctx.get_current_channel()
        if cand and isinstance(cand, _TEXT_LIKE):
            ch = cand
        # Fall through to fuzzy search if current channel not available.

    # Direct ID or <#id> mention.
    if ch is None:
        cand = _try_direct_channel(ctx, query)
        if cand and isinstance(cand, _TEXT_LIKE):
            ch = cand

    # Fuzzy search.
    if ch is None:
        guild = ctx.require_guild()
        text_like = [c for c in guild.channels if isinstance(c, _TEXT_LIKE)]
        results = fuzzy_search(query, text_like, key=lambda c: c.name, limit=1)
        if not results:
            return None
        ch = results[0].item

    _check_send_channel(ctx, ch)
    return ch


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
    content = _sanitize_content(content)
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
    from src.action_engine.core import self_cleanup
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
                # Bulk deletes on a user account rate-limit fast — pace
                # them exactly like core/self_cleanup.py does.
                await asyncio.sleep(1.2)
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
