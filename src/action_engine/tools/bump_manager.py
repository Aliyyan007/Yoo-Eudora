"""Bump manager: auto-detect bump bots, track cooldowns, and auto-bump.

This module:
1. Detects bump bots by checking for /bump slash commands with server-promotion descriptions.
2. Monitors bump bot messages (embeds) to track successful bumps and cooldowns.
3. Schedules automatic re-bumps when cooldowns expire.
4. Provides tools for the agent to query bump status and trigger bumps.

Known bump bots and their cooldowns:
  - DISBOARD (302050872383242240): 2 hours
  - Bumper (1153715777594200074): 1 hour
  - OneBump (1028956609382199346): 1 hour
  - Bump Central (478290034773196810): 1 hour
  - Liam (389604896606781440): 1 hour
  - DSC (415773861486002186): 1 hour
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import discord

from src.action_engine.config.settings import settings
from src.action_engine.tools.context import ToolContext
from src.action_engine.utils.embeds import embed_to_json
from src.action_engine.utils.logger import logger


# ============================================================
#  Known bump bot signatures
# ============================================================
BUMP_BOTS: dict[int, dict] = {
    302050872383242240: {"name": "DISBOARD", "cooldown_sec": 7200},   # 2 hours
    1153715777594200074: {"name": "Bumper", "cooldown_sec": 3600},    # 1 hour
    1028956609382199346: {"name": "OneBump", "cooldown_sec": 3600},   # 1 hour
    235148962103951360: {"name": "Global Carl", "cooldown_sec": 3600}, # 1 hour
    478290034773196810: {"name": "Bump Central", "cooldown_sec": 3600},
    389604896606781440: {"name": "Liam", "cooldown_sec": 3600},
    415773861486002186: {"name": "DSC", "cooldown_sec": 3600},
    1006190394415005788: {"name": "BumpIt", "cooldown_sec": 3600},
    880766859534794764: {"name": "Bumpy.gg", "cooldown_sec": 3600},
    826100334534328340: {"name": "DiscordHome", "cooldown_sec": 3600},
}

# Keywords that distinguish a bump command from other /bump commands (e.g. Hydra music).
_BUMP_KEYWORDS = (
    "bump this server", "promote your server", "send your server",
    "advertise", "server", "discordhome", "discadia", "disboost",
    "bumpy", "onebump", "bump central", "bumper", "disboard",
    "showcase", "listing", "directory",
)

# In-memory state: bot_id -> {channel_id, next_bump_time, last_bumper}
_bump_state: dict[int, dict] = {}
# Scheduled re-bump tasks: bot_id -> asyncio.Task
_bump_tasks: dict[int, asyncio.Task] = {}


# ============================================================
#  Bump bot detection
# ============================================================
async def find_bump_commands(ctx: ToolContext, channel_query: str = "here") -> dict:
    """Find all /bump slash commands that look like server-promotion bumps.

    Distinguishes bump bots from music bots (e.g. Hydra) by checking the
    command description for server-promotion keywords.
    """
    from src.action_engine.tools.messaging import _resolve_text_channel
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}

    try:
        cmds = await ch.application_commands()
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to fetch slash commands: {e}"}

    guild = ctx.require_guild()
    candidates = []
    for c in cmds:
        if not isinstance(c, discord.SlashCommand):
            continue
        if c.name.lower() != "bump":
            continue
        desc = (getattr(c, "description", "") or "").lower()
        app_id = str(getattr(c, "application_id", ""))
        # Get bot display name from guild members.
        bot_name = None
        if app_id and app_id.isdigit():
            member = guild.get_member(int(app_id))
            if member:
                bot_name = member.display_name
        # Check if it's a known bump bot.
        known = BUMP_BOTS.get(int(app_id)) if app_id and app_id.isdigit() else None
        if known:
            bot_name = known["name"]
        # Determine if this is a bump bot (known or keyword match).
        is_bump = bool(known) or any(k in desc for k in _BUMP_KEYWORDS)
        if is_bump:
            candidates.append({
                "name": c.name,
                "description": getattr(c, "description", ""),
                "application_id": app_id,
                "bot_name": bot_name or f"Bot {app_id}",
                "cooldown_sec": known["cooldown_sec"] if known else 3600,
                "cooldown_hours": (known["cooldown_sec"] if known else 3600) / 3600,
            })

    return {"channel": ch.name, "bump_commands": candidates}


# ============================================================
#  Bump a specific bot by application_id
# ============================================================
async def bump_with_bot(
    ctx: ToolContext,
    application_id: str,
    channel_query: str = "here",
) -> dict:
    """Bump the server using a specific bump bot by its application ID."""
    from src.action_engine.tools.messaging import _resolve_text_channel
    ch = await _resolve_text_channel(ctx, channel_query)
    if ch is None:
        return {"error": f"No channel matching '{channel_query}'."}

    try:
        cmds = await ch.application_commands()
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to fetch slash commands: {e}"}

    # Find the /bump command for this specific bot.
    cmd = next(
        (c for c in cmds
         if isinstance(c, discord.SlashCommand)
         and c.name.lower() == "bump"
         and str(getattr(c, "application_id", "")) == str(application_id)),
        None,
    )
    if cmd is None:
        return {"error": f"No /bump command found for bot {application_id}."}

    bot_info = BUMP_BOTS.get(int(application_id))
    bot_name = bot_info["name"] if bot_info else f"Bot {application_id}"

    try:
        interaction = await cmd(ch)
        logger.info(f"Bumped with {bot_name} in #{ch.name}")

        # Wait for the bot to respond, then fetch the response message
        # to detect cooldown/success embeds. Bot embeds are often sent as
        # deferred updates, so they may not be in the MESSAGE_CREATE payload.
        await asyncio.sleep(3)
        bot_user_id = int(application_id)
        response_msg = None
        async for m in ch.history(limit=5):
            if m.author.id == bot_user_id and m.created_at > discord.utils.utcnow() - timedelta(seconds=30):
                response_msg = m
                break

        if response_msg:
            # Process the response message through the bump handler.
            await handle_bump_message(ctx, response_msg)

        return {
            "ok": True,
            "bot": bot_name,
            "application_id": application_id,
            "channel": ch.name,
            "interaction_id": getattr(interaction, "id", None),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"Bump with {bot_name} failed: {e}"}


async def bump_all(ctx: ToolContext, channel_query: str = "here") -> dict:
    """Bump the server with ALL detected bump bots in a channel."""
    find_result = await find_bump_commands(ctx, channel_query)
    if "error" in find_result:
        return find_result

    bump_cmds = find_result.get("bump_commands", [])
    if not bump_cmds:
        return {"error": "No bump bots found in this channel."}

    results = []
    for cmd_info in bump_cmds:
        app_id = cmd_info["application_id"]
        result = await bump_with_bot(ctx, app_id, channel_query)
        results.append({
            "bot": cmd_info["bot_name"],
            "application_id": app_id,
            "ok": result.get("ok", False),
            "error": result.get("error"),
        })
        # Small delay between bumps to avoid rate limiting.
        await asyncio.sleep(1)

    succeeded = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    return {
        "ok": len(succeeded) > 0,
        "total": len(results),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "results": results,
    }


# ============================================================
#  Bump status & cooldown tracking
# ============================================================
async def get_bump_status(ctx: ToolContext) -> dict:
    """Get the current bump status for all tracked bots."""
    now = datetime.now(timezone.utc)
    status = []
    for bot_id, state in _bump_state.items():
        bot_info = BUMP_BOTS.get(bot_id, {"name": f"Bot {bot_id}"})
        next_time = state.get("next_bump_time")
        if next_time:
            remaining = (next_time - now).total_seconds()
            status.append({
                "bot": bot_info["name"],
                "bot_id": bot_id,
                "channel_id": state.get("channel_id"),
                "next_bump_in_sec": max(0, int(remaining)),
                "next_bump_in_min": max(0, int(remaining / 60)),
                "ready": remaining <= 0,
                "last_bumper": state.get("last_bumper"),
            })
        else:
            status.append({
                "bot": bot_info["name"],
                "bot_id": bot_id,
                "ready": True,
                "note": "No bump recorded yet",
            })
    return {"bump_status": status}


# ============================================================
#  Embed detection — called from on_message BEFORE bot ignore
# ============================================================
def _extract_mention(text: str) -> int | None:
    m = re.search(r"<@!?(\d+)>", text)
    return int(m.group(1)) if m else None


_MINUTE_RE = re.compile(r"(\d+)\s*minute", re.IGNORECASE)
_HOUR_RE = re.compile(r"(\d+)\s*hour", re.IGNORECASE)
_SECOND_RE = re.compile(r"(\d+)\s*second", re.IGNORECASE)


def _parse_remaining_time(text: str) -> int | None:
    """Parse remaining cooldown time from text. Returns seconds."""
    total = 0
    found = False
    h = _HOUR_RE.search(text)
    if h:
        total += int(h.group(1)) * 3600
        found = True
    m = _MINUTE_RE.search(text)
    if m:
        total += int(m.group(1)) * 60
        found = True
    s = _SECOND_RE.search(text)
    if s:
        total += int(s.group(1))
        found = True
    return total if found else None


async def handle_bump_message(ctx: ToolContext, message: discord.Message) -> bool:
    """Check if a message is from a bump bot and track its cooldown.

    Called from on_message BEFORE the bot-ignore check.
    Returns True if the message was handled (was from a bump bot).
    """
    bot_info = BUMP_BOTS.get(message.author.id)
    if not bot_info:
        return False

    # Only track in bump channels if configured.
    if settings.bump_channel_ids and message.channel.id not in settings.bump_channel_ids:
        return False

    text = (message.content or "").lower()
    logger.debug(f"Bump handler: author={message.author.id} ({bot_info['name']}), channel={message.channel.id}, embeds={len(message.embeds)}, content={text[:100]}")
    embed_text = ""
    for e in message.embeds:
        ej = embed_to_json(e)
        embed_text += " ".join(filter(None, [
            ej.get("title") or "",
            ej.get("description") or "",
            " ".join(f"{f.get('name', '')} {f.get('value', '')}" for f in (ej.get("fields") or [])),
            ej.get("footer", {}).get("text", "") if isinstance(ej.get("footer"), dict) else "",
        ])).lower()

    combined = f"{text} {embed_text}"

    # SUCCESS: bump was done.
    if any(phrase in combined for phrase in (
        "bump done", ":thumbsup:", "bump successful", "server bumped",
        "successfully bumped", "successful bump", "bump complete",
        "server bumped", "your server has been bumped",
    )):
        bumper = _extract_mention(message.content or "")
        if not bumper:
            for e in message.embeds:
                ej = embed_to_json(e)
                bumper = _extract_mention(ej.get("description") or "")
                if bumper:
                    break
        next_time = message.created_at + timedelta(seconds=bot_info["cooldown_sec"])
        _bump_state[message.author.id] = {
            "channel_id": message.channel.id,
            "next_bump_time": next_time,
            "last_bumper": bumper,
            "last_bump_time": message.created_at,
        }
        logger.info(
            f"Bump success: {bot_info['name']} bumped in #{message.channel.name}, "
            f"next bump at {next_time.strftime('%H:%M:%S UTC')}"
        )
        # Schedule auto-bump if enabled.
        if settings.auto_bump:
            _schedule_auto_bump(ctx, message.author.id, message.channel.id, next_time)
        return True

    # COOLDOWN: bot says to wait.
    if any(phrase in combined for phrase in (
        "wait", "cooldown", "again", "not ready", "already bumped",
        "you can bump", "before bumping", "try again",
        "recently bumped", "able to use", "will be able",
    )):
        remaining = _parse_remaining_time(combined)
        if remaining is None:
            # Try to parse Discord timestamp tag: <t:UNIX:R> from content or embeds
            ts_match = re.search(r"<t:(\d+):R?>", combined)
            if ts_match:
                target_ts = int(ts_match.group(1))
                remaining = max(0, target_ts - int(datetime.now(timezone.utc).timestamp()))
            else:
                remaining = bot_info["cooldown_sec"]
        next_time = datetime.now(timezone.utc) + timedelta(seconds=remaining)
        _bump_state[message.author.id] = {
            "channel_id": message.channel.id,
            "next_bump_time": next_time,
            "last_bumper": None,
            "last_bump_time": None,
        }
        logger.info(
            f"Bump cooldown: {bot_info['name']} on cooldown for {remaining}s, "
            f"next bump at {next_time.strftime('%H:%M:%S UTC')}"
        )
        if settings.auto_bump:
            _schedule_auto_bump(ctx, message.author.id, message.channel.id, next_time)
        return True

    return False


# ============================================================
#  Auto-bump scheduler
# ============================================================
def _schedule_auto_bump(
    ctx: ToolContext,
    bot_id: int,
    channel_id: int,
    next_time: datetime,
) -> None:
    """Schedule an automatic re-bump when the cooldown expires."""
    # Cancel existing task for this bot.
    old = _bump_tasks.get(bot_id)
    if old and not old.done():
        old.cancel()

    bot_info = BUMP_BOTS.get(bot_id, {"name": f"Bot {bot_id}"})

    async def _auto_bump():
        now = datetime.now(timezone.utc)
        wait_sec = (next_time - now).total_seconds()
        if wait_sec > 0:
            await asyncio.sleep(wait_sec)
        # Re-bump.
        logger.info(f"Auto-bump: triggering {bot_info['name']} in channel {channel_id}")
        try:
            ch = ctx.bot.get_channel(channel_id) or await ctx.bot.fetch_channel(channel_id)
            cmds = await ch.application_commands()
            cmd = next(
                (c for c in cmds
                 if isinstance(c, discord.SlashCommand)
                 and c.name.lower() == "bump"
                 and str(getattr(c, "application_id", "")) == str(bot_id)),
                None,
            )
            if cmd:
                await cmd(ch)
                logger.info(f"Auto-bump: {bot_info['name']} bumped successfully")
            else:
                logger.warning(f"Auto-bump: {bot_info['name']} /bump command not found")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Auto-bump: {bot_info['name']} failed: {e}")

    task = asyncio.create_task(_auto_bump())
    task.add_done_callback(lambda t, bid=bot_id: _bump_tasks.pop(bid, None))
    _bump_tasks[bot_id] = task
    logger.info(f"Scheduled auto-bump for {bot_info['name']} at {next_time.strftime('%H:%M:%S UTC')}")


def cancel_all_auto_bumps() -> None:
    """Cancel all scheduled auto-bump tasks."""
    for task in _bump_tasks.values():
        if not task.done():
            task.cancel()
    _bump_tasks.clear()
