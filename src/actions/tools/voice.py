"""Voice / Stage tools (read-only subset vendored for the action worker).

Kept tools: list_voice_channels, get_voice_state, get_current_vc,
get_user_voice_state, get_vc_text_chat, send_vc_text.

The action/voice-session tools from the source (join_voice, leave_voice,
say_in_vc, mute_self, deafen_self, move_voice, speak_in_stage) are NOT
vendored — they depend on the source repo's ``voice`` package
(native_voice VoiceClient + VoiceSession pipeline).

In discord.py-self v2.1.0, VoiceChannel IS the Messageable — you can
call `voice_channel.send()` and `voice_channel.history()` directly.
There is NO `associated_text_channel` attribute.
"""
from __future__ import annotations

import discord

from src.actions.tools.context import ToolContext
from src.actions.utils.fuzzy import fuzzy_search


async def _resolve_voice_channel(ctx: ToolContext, query: str):
    """Resolve a voice/stage channel from a query (fuzzy or ID)."""
    guild = ctx.require_guild()
    # Direct ID resolution.
    import re
    m = re.match(r"<#(\d+)>", query.strip())
    cid = int(m.group(1)) if m else None
    if not cid and query.strip().isdigit():
        cid = int(query.strip())
    if cid:
        ch = guild.get_channel(cid)
        if ch and isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
            return ch
    voice_like = [
        c for c in guild.channels
        if isinstance(c, (discord.VoiceChannel, discord.StageChannel))
    ]
    results = fuzzy_search(query, voice_like, key=lambda c: c.name, limit=1)
    if not results:
        results = fuzzy_search(
            query, voice_like,
            key=lambda c: f"{c.category.name if c.category else ''} {c.name}",
            limit=1,
        )
    return results[0].item if results else None


async def get_voice_state(ctx: ToolContext) -> dict:
    """Report current voice connection + members in the channel."""
    conns = ctx.bot.voice_clients
    if not conns:
        return {"connected": False}
    out = []
    for vc in conns:
        ch = vc.channel
        me_voice = vc.guild.me.voice if vc.guild and vc.guild.me else None
        out.append({
            "channel": ch.name,
            "id": ch.id,
            "type": "stage" if isinstance(ch, discord.StageChannel) else "voice",
            "self_muted": me_voice.self_mute if me_voice else None,
            "self_deafened": me_voice.self_deaf if me_voice else None,
            "members": [
                {
                    "name": m.display_name, "id": m.id,
                    "mute": m.voice.mute if m.voice else None,
                    "deaf": m.voice.deaf if m.voice else None,
                }
                for m in ch.members
            ],
        })
    return {"connected": True, "connections": out}


async def get_current_vc(ctx: ToolContext) -> dict:
    """Get the voice channel the bot is currently connected to (for sending text there)."""
    conns = ctx.bot.voice_clients
    if not conns:
        return {"connected": False}
    vc = conns[0]
    ch = vc.channel
    return {
        "connected": True,
        "channel": ch.name,
        "id": ch.id,
        "type": "stage" if isinstance(ch, discord.StageChannel) else "voice",
        "mention": ch.mention,
        "members": [m.display_name for m in ch.members],
    }


async def get_user_voice_state(ctx: ToolContext, user_query: str) -> dict:
    """Check which voice channel a user is currently in.

    Useful for 'join my vc' — finds the user's current VC so the bot can join it.
    """
    guild = ctx.require_guild()
    # Resolve the member first.
    from src.actions.tools.members import _try_direct_member, resolve_member
    m = _try_direct_member(ctx, user_query)
    if not m:
        res = await resolve_member(ctx, user_query)
        if "error" in res:
            return res
        m = guild.get_member(res["id"])
    if not m:
        return {"error": f"Member '{user_query}' not found."}
    voice = m.voice
    if voice is None or voice.channel is None:
        return {"connected": False, "user": m.display_name}
    ch = voice.channel
    return {
        "connected": True,
        "user": m.display_name,
        "channel": ch.name,
        "channel_id": ch.id,
        "type": "stage" if isinstance(ch, discord.StageChannel) else "voice",
        "members": [mem.display_name for mem in ch.members],
    }


async def list_voice_channels(ctx: ToolContext) -> list[dict]:
    """List all voice + stage channels with current member counts."""
    guild = ctx.require_guild()
    out = []
    for ch in guild.channels:
        if isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
            out.append({
                "id": ch.id,
                "name": ch.name,
                "type": "stage" if isinstance(ch, discord.StageChannel) else "voice",
                "member_count": len(ch.members),
                "user_limit": ch.user_limit,
                "members": [m.display_name for m in ch.members],
            })
    return out


async def get_vc_text_chat(ctx: ToolContext, channel_query: str, limit: int = 20) -> dict:
    """Read the text chat of a voice channel.

    In discord.py-self v2.1.0, VoiceChannel IS the Messageable —
    we call voice_channel.history() directly.
    """
    ch = await _resolve_voice_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No voice channel matching '{channel_query}'."}

    from src.actions.tools.messages import _serialise_message
    try:
        history = [m async for m in ch.history(limit=limit)]
    except discord.Forbidden:
        return {"error": f"No permission to read text chat of '{ch.name}'."}
    history.reverse()
    msgs = [await _serialise_message(m, with_vision=False) for m in history]
    return {"voice_channel": ch.name, "count": len(msgs), "messages": msgs}


async def send_vc_text(
    ctx: ToolContext,
    channel_query: str,
    content: str,
    *,
    ping_users: list[str] | None = None,
) -> dict:
    """Send a message to a voice channel's text chat.

    VoiceChannel IS the Messageable in discord.py-self — we call
    voice_channel.send() directly. Supports pinging users.
    """
    ch = await _resolve_voice_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No voice channel matching '{channel_query}'."}

    parts: list[str] = []
    from src.actions.tools.members import resolve_member
    for u in (ping_users or []):
        res = await resolve_member(ctx, u)
        if "mention" in res:
            parts.append(res["mention"])
        else:
            parts.append(f"@{u}")
    if content:
        parts.append(content)
    final = " ".join(parts) if parts else content

    if len(final) > 2000:
        final = final[:1997] + "..."

    try:
        sent = await ch.send(final)
    except discord.HTTPException as e:
        return {"error": f"Send to VC text chat failed: {e}"}
    return {"ok": True, "channel": ch.name, "message_id": sent.id, "sent": final}
