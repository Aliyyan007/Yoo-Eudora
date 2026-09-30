"""
Channel scanner — discovers all text channels, learns their styles,
finds the rules channel, and identifies the most active channels.
This is the bot's "understanding" of the server.
"""
import asyncio
import time
from typing import List, Optional, Tuple, Set
from collections import deque
from loguru import logger
import discord

from .ai import d1_memory as mem  # D1-backed memory (falls back to JSON if D1 unavailable)
from .ai import reply as ai_reply


# Channel name keywords that likely contain server rules
RULES_CHANNEL_NAMES = {
    "rules", "rule", "guidelines", "guideline", "info", "information",
    "welcome", "readme", "server-info", "server-rules", "read-me",
    "read-first", "start-here", "announcements", "announce", "updates",
    "verification", "verify",
}

# Channels we should NOT participate in
SKIP_CHANNEL_NAMES = RULES_CHANNEL_NAMES | {
    "bot-commands", "bot-logs", "logs", "audit", "mod-log", "mod-logs",
    "staff", "admin", "moderation",
    "rank", "ranks", "level", "levels", "counting", "boost", "bumps", "bump",
    "bot", "commands", "command", "verify", "verification", "ticket", "tickets",
    "support", "application", "applications", "introduction", "introductions",
    "self-role", "self-roles", "reaction-role", "reaction-roles", "roles",
    "giveaway", "giveaways", "poll", "polls", "suggestion", "suggestions",
    "starboard", "server-stats", "member-count", "server-count",
}


def is_rules_channel(channel_name: str) -> bool:
    """Check if a channel name suggests it's a rules/info channel."""
    name_lower = channel_name.lower().replace("_", "-")
    return any(kw in name_lower for kw in RULES_CHANNEL_NAMES)


def is_skip_channel(channel_name: str) -> bool:
    """Check if we should skip this channel entirely."""
    name_lower = channel_name.lower().replace("_", "-")
    return any(kw in name_lower for kw in SKIP_CHANNEL_NAMES)


def can_speak_in(channel: discord.TextChannel, me: discord.Member, *,
                 include_name_check: bool = True) -> bool:
    """Check if the bot has permission to read and send messages in a channel.

    include_name_check=False skips the skip-channel name filter — used when
    the message is directed at the bot (reply/sticky convo); the name list is
    only meant to gate unsolicited participation in utility channels."""
    try:
        perms = channel.permissions_for(me)
        if not (perms.send_messages and perms.read_messages
                and perms.read_message_history):
            return False
        return not include_name_check or not is_skip_channel(channel.name)
    except Exception:
        return False


async def fetch_guild_rules(guild: discord.Guild, cache: dict) -> str:
    """
    Find the rules channel in a guild and return its text content.
    Uses cache dict: guild_id -> rules_text.
    """
    guild_id = str(guild.id)
    if guild_id in cache:
        return cache[guild_id]

    rules_text = ""
    best_channel = None

    for channel in guild.text_channels:
        if is_rules_channel(channel.name):
            best_channel = channel
            break

    if best_channel:
        try:
            messages = [m async for m in best_channel.history(limit=30, oldest_first=True)]
            lines = []
            for m in messages:
                text = m.clean_content.strip()
                if text:
                    lines.append(text)
            rules_text = "\n".join(lines)
            logger.info(f"Fetched rules for '{guild.name}' from #{best_channel.name} ({len(rules_text)} chars)")
        except Exception as e:
            logger.warning(f"Could not read rules channel #{best_channel.name}: {e}")
    else:
        logger.debug(f"No rules channel found for '{guild.name}'")

    cache[guild_id] = rules_text
    return rules_text


async def discover_channels(guild: discord.Guild) -> List[discord.TextChannel]:
    """
    Discover all text channels in a guild where the bot can speak.
    Records them in memory for smart selection.
    """
    me = guild.me
    speakable = []
    for channel in guild.text_channels:
        if can_speak_in(channel, me):
            speakable.append(channel)
            mem.mark_channel_seen(str(channel.id), channel.name, str(guild.id))
    logger.info(f"Discovered {len(speakable)} speakable channels in '{guild.name}'")
    return speakable


async def find_most_active_channel(
    guild: discord.Guild,
    last_activity: dict,
    max_age_hours: int = 2,
) -> Optional[discord.TextChannel]:
    """
    Find the most recently active channel in a guild.
    Only considers channels active within the last max_age_hours.
    """
    now = time.time()
    best_ch = None
    best_ts = 0.0
    me = guild.me

    for ch in guild.text_channels:
        ch_id = str(ch.id)
        ts = last_activity.get(ch_id, 0)
        if ts > best_ts and (now - ts) < (max_age_hours * 3600):
            if can_speak_in(ch, me):
                best_ts = ts
                best_ch = ch

    return best_ch


async def learn_channel_style(channel: discord.TextChannel, max_messages: int = 50) -> str:
    """
    Read recent messages from a channel and extract its communication style.
    Stores the style in memory for future use.
    """
    try:
        messages = [m async for m in channel.history(limit=max_messages)]
        # Build a transcript of how people talk (exclude our own messages)
        transcript_lines = []
        for m in messages:
            if m.author.bot:
                continue
            text = m.clean_content.strip()
            if text and len(text) > 2:
                transcript_lines.append(f"{m.author.name}: {text}")

        if len(transcript_lines) < 5:
            logger.debug(f"Not enough messages in #{channel.name} to learn style")
            return ""

        transcript = "\n".join(transcript_lines[-30:])
        style = ai_reply.extract_channel_style(transcript)
        if style:
            mem.update_channel_style(str(channel.id), style)
            logger.info(f"Learned style for #{channel.name}: {style[:60]}...")
        return style
    except Exception as e:
        logger.warning(f"Failed to learn style for #{channel.name}: {e}")
        return ""


async def learn_channel_topic(channel: discord.TextChannel, max_messages: int = 50) -> str:
    """
    Read recent messages and summarize what the channel is about.
    """
    try:
        messages = [m async for m in channel.history(limit=max_messages)]
        transcript_lines = []
        for m in messages:
            if m.author.bot:
                continue
            text = m.clean_content.strip()
            if text:
                transcript_lines.append(text)

        if len(transcript_lines) < 3:
            return ""

        transcript = "\n".join(transcript_lines[-30:])
        topic = ai_reply.extract_channel_topic(transcript)
        if topic:
            mem.update_channel_topic(str(channel.id), topic)
            logger.info(f"Learned topic for #{channel.name}: {topic}")
        return topic
    except Exception as e:
        logger.warning(f"Failed to learn topic for #{channel.name}: {e}")
        return ""


async def scan_guild(guild: discord.Guild, rules_cache: dict):
    """
    Full guild scan: discover channels, fetch rules, learn styles and topics.
    Called on startup and when joining a new guild.
    """
    logger.info(f"Scanning guild '{guild.name}' ({guild.id})...")

    # Fetch rules
    await fetch_guild_rules(guild, rules_cache)

    # Discover channels
    channels = await discover_channels(guild)

    # Learn styles and topics for the top channels (limit to avoid rate limits)
    # Sort by member count or just take the first 5
    channels_to_learn = channels[:5]
    for channel in channels_to_learn:
        await asyncio.sleep(1)  # Rate limit safety
        await learn_channel_topic(channel)
        await asyncio.sleep(1)
        await learn_channel_style(channel)

    logger.info(f"Guild scan complete for '{guild.name}': {len(channels)} channels, rules={'yes' if rules_cache.get(str(guild.id)) else 'no'}")
