"""Voice / Stage tools: join, leave, mute, deafen, move, list members, read/send text chat.

In discord.py-self v2.1.0, VoiceChannel IS the Messageable — you can
call `voice_channel.send()` and `voice_channel.history()` directly.
There is NO `associated_text_channel` attribute.
"""
from __future__ import annotations

from typing import Any
import asyncio
import re

import discord

from src.action_engine.config.settings import settings
from src.action_engine.tools.context import ToolContext
from src.action_engine.utils.fuzzy import fuzzy_search
from src.action_engine.utils.logger import logger


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


def _occupied_voice_channel(ctx: ToolContext):
    """The voice channel with humans in it — for 'join my vc' when the
    name doesn't resolve (e.g. churn-deleted 'type voice' channels)."""
    try:
        guild = ctx.require_guild()
    except Exception:  # noqa: BLE001
        return None
    occupied = [
        c for c in guild.channels
        if isinstance(c, (discord.VoiceChannel, discord.StageChannel))
        and any(not m.bot for m in c.members)
    ]
    if not occupied:
        return None
    # most-populated wins; ties fall back to first
    return max(occupied, key=lambda c: sum(1 for m in c.members if not m.bot))


async def join_voice(ctx: ToolContext, channel_query: str, self_mute: bool = False, self_deaf: bool = False) -> dict:
    """Join a voice or stage channel and start the live voice-chat session."""
    ch = await _resolve_voice_channel(ctx, channel_query)
    if ch is None:
        # "join my vc" means wherever the asker is — when the name lookup
        # fails, fall back to the channel that has people in it (only if
        # the query was vague or there's exactly one occupied channel)
        vague = bool(re.search(
            r"\b(my|their|the|us)\b", channel_query or "", re.I))
        guild = ctx.require_guild()
        occupied = [
            c for c in guild.channels
            if isinstance(c, (discord.VoiceChannel, discord.StageChannel))
            and any(not m.bot for m in c.members)
        ]
        if vague or len(occupied) == 1:
            ch = _occupied_voice_channel(ctx)
    if ch is None:
        return {"error": f"No voice/stage channel matching '{channel_query}'."}
    # a deafened bot can't hear anyone — never join deaf, no matter what
    # the model decided to pass
    self_deaf = False
    # INTEGRATION: delegate to the host bot's own voice manager instead of
    # starting the donor's VoiceSession (which we do not vendor).
    vm = getattr(ctx.bot, "voice_manager", None)
    try:
        if vm is not None:
            ok = await vm.join_vc(ctx.bot, ch, asyncio.get_running_loop())
        else:
            ok = bool(await ctx.bot.vc_manager.join_vc(
                ctx.bot, ch.guild, ch, ch.guild.me, ch))
        if not ok:
            return {"error": "Voice connect failed (host voice manager)."}
    except discord.ClientException as e:
        return {"error": f"Voice connect failed: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"Voice connect failed: {e}"}
    return {"ok": True, "channel": ch.name, "type": "stage" if isinstance(ch, discord.StageChannel) else "voice"}


async def leave_voice(ctx: ToolContext, **_extra) -> dict:
    """Disabled in the host integration — the host's own leave-vc path
    (commands.py leave_vc / vc_intent.leave_vc_score) handles this before
    requests ever reach the action worker."""
    return {"error": "leave_voice is handled by the host system."}


async def say_in_vc(ctx: ToolContext, text: str) -> dict:
    """Speak a line out loud in the current voice channel via TTS —
    delegated to the host bot's voice pipeline."""
    guild_id = ctx.guild.id if ctx.guild else 0
    if not guild_id and ctx.bot.voice_clients:
        guild_id = ctx.bot.voice_clients[0].guild.id
    vm = getattr(ctx.bot, "voice_manager", None)
    pipeline = getattr(vm, "_pipelines", {}).get(guild_id) if vm else None
    if pipeline is None:
        return {"error": "Not in a voice channel (no active voice session)."}
    try:
        await pipeline._speak(text)
        return {"ok": True, "said": text}
    except Exception as e:  # noqa: BLE001
        return {"error": f"TTS failed: {e}"}


async def mute_self(ctx: ToolContext, muted: bool = True) -> dict:
    """Self-mute or unmute in the current voice channel."""
    guild = ctx.require_guild()
    for vc in ctx.bot.voice_clients:
        try:
            me = guild.me.voice
            await guild.change_voice_state(
                channel=vc.channel,
                self_mute=muted,
                self_deaf=me.self_deaf if me else False,
            )
            return {"ok": True, "muted": muted, "channel": vc.channel.name}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Mute failed: {e}"}
    return {"error": "Not connected to any voice channel."}


async def deafen_self(ctx: ToolContext, deafened: bool = True) -> dict:
    """Self-deafen or undeafen in the current voice channel."""
    guild = ctx.require_guild()
    for vc in ctx.bot.voice_clients:
        try:
            me = guild.me.voice
            await guild.change_voice_state(
                channel=vc.channel,
                self_mute=me.self_mute if me else False,
                self_deaf=deafened,
            )
            return {"ok": True, "deafened": deafened, "channel": vc.channel.name}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Deafen failed: {e}"}
    return {"error": "Not connected to any voice channel."}


async def move_voice(ctx: ToolContext, channel_query: str) -> dict:
    """Move to a different voice/stage channel."""
    ch = await _resolve_voice_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No voice/stage channel matching '{channel_query}'."}
    for vc in ctx.bot.voice_clients:
        try:
            await vc.move_to(ch)
            return {"ok": True, "moved_to": ch.name}
        except Exception as e:  # noqa: BLE001
            return {"error": f"Move failed: {e}"}
    return {"error": "Not connected to any voice channel. Use join_voice first."}


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
    from src.action_engine.tools.members import _try_direct_member, resolve_member
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

    from src.action_engine.tools.messages import _serialise_message
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
    from src.action_engine.tools.members import resolve_member
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


async def speak_in_stage(ctx: ToolContext, suppress: bool = False) -> dict:
    """Request to speak in the current stage channel (if any)."""
    for vc in ctx.bot.voice_clients:
        if isinstance(vc.channel, discord.StageChannel):
            try:
                await vc.guild.me.edit(suppress=suppress)
                return {"ok": True, "speaking": not suppress}
            except Exception as e:  # noqa: BLE001
                return {"error": f"Stage speak request failed: {e}"}
    return {"error": "Not connected to a stage channel."}
