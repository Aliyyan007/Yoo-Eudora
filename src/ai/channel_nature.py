"""
Channel nature analysis module for the Eudora persona.

Features:
1. Fetch 20 messages to analyze channel nature (first time)
2. Cache the analysis with a 24h cooldown
3. After cache expires, fetch only 5 messages for a quick refresh
4. Determine channel "nature" (gaming, art, music, general, venting, etc.)
5. Track analysis timestamps per channel

Algorithmic approach:
- First analysis: fetch 20 messages, run full nature analysis
- Cache result for 24 hours
- After 24h: fetch only 5 messages for a quick refresh
- If channel is quiet (< 5 messages), skip analysis
- Nature classification based on keyword frequency and message patterns
"""
import time
import re
from typing import Optional, Dict, List, Tuple
from collections import Counter
from loguru import logger
import discord


# Channel nature categories with keyword indicators
NATURE_KEYWORDS = {
    "gaming": ["game", "play", "valorant", "minecraft", "fortnite", "lol", "csgo", "gta", "rpg", "level", "boss", "quest", "gaming", "server", "lag", "ping", "fps"],
    "art": ["draw", "art", "paint", "sketch", "design", "creative", "wip", "artwork", "doodle", "canvas", "illustration"],
    "music": ["song", "music", "album", "artist", "playlist", "spotify", "lofi", "beat", "track", "listen", "band"],
    "venting": ["sad", "depressed", "anxious", "stress", "hate", "tired", "alone", "lonely", "cry", "hurt", "pain", "struggle", "mental"],
    "study": ["study", "homework", "exam", "test", "school", "college", "university", "assignment", "grade", "professor", "class"],
    "tech": ["code", "programming", "python", "javascript", "bug", "error", "server", "api", "database", "linux", "git"],
    "general": ["hello", "hi", "hey", "how", "what", "anyone", "sup", "yo", "wbu", "hru"],
    "memes": ["meme", "lol", "lmao", "fr", "based", "cringe", "mid", "💀", "😭", "😂"],
    "anime": ["anime", "manga", "otaku", "weeb", "naruto", "one piece", "aot", "episode", "season"],
    "food": ["food", "eat", "cook", "recipe", "dinner", "lunch", "breakfast", "pizza", "coffee", "tea"],
    "utility": ["rank", "level", "bump", "count", "counting", "boost", "verify",
                "verification", "ticket", "command", "bot command", "leaderboard",
                "stats", "server stats", "member count", "level up", "xp",
                "points", "score", "leaderboard", "poll", "vote", "giveaway",
                "suggestion", "starboard"],
}

# Nature analysis cache: channel_id -> (nature, timestamp, message_count)
_nature_cache: Dict[str, Tuple[str, float, int]] = {}

# Cooldown: 24 hours
CACHE_COOLDOWN_S = 24 * 60 * 60  # 24 hours

# First-time fetch count
FIRST_FETCH_COUNT = 20
# Refresh fetch count (after cache expires)
REFRESH_FETCH_COUNT = 5


def classify_nature(messages: List[str]) -> str:
    """
    Algorithmically classify the nature of a channel based on message content.

    Uses keyword frequency analysis to determine the dominant topic.
    Returns one of the nature categories or "general" if no clear winner.
    """
    if not messages:
        return "general"

    # Combine all messages
    combined = " ".join(messages).lower()

    # Count keyword hits per category
    scores = Counter()
    for nature, keywords in NATURE_KEYWORDS.items():
        for kw in keywords:
            # Use word boundary for longer keywords, substring for short ones
            if len(kw) >= 4:
                count = len(re.findall(r'\b' + re.escape(kw) + r'\b', combined))
            else:
                count = combined.count(kw)
            scores[nature] += count

    # Find the dominant nature
    if not scores or scores.most_common(1)[0][1] == 0:
        return "general"

    top_nature, top_score = scores.most_common(1)[0]

    # Require at least 2 hits to classify as non-general
    if top_score < 2:
        return "general"

    return top_nature


def get_nature_summary(nature: str) -> str:
    """Get a human-readable summary of the channel nature."""
    summaries = {
        "gaming": "gaming discussion, mostly games and memes",
        "art": "art community, sharing drawings and creative work",
        "music": "music lovers, sharing songs and discussing artists",
        "venting": "support space, people sharing feelings and venting",
        "study": "study group, homework and academic discussion",
        "tech": "tech community, coding and technology discussion",
        "general": "general chat, random topics and daily life",
        "memes": "meme channel, jokes and funny content",
        "anime": "anime community, discussing shows and manga",
        "food": "food lovers, sharing recipes and meals",
        "utility": "This is a utility/commands channel (rank checks, bot commands, counting, etc.) — NOT a casual chat channel. Do NOT send casual messages or ping users here. Only respond if directly mentioned.",
    }
    return summaries.get(nature, "general chat, random topics")


def should_analyze(channel_id: str) -> bool:
    """
    Check if a channel needs nature analysis (cache expired or not cached).
    """
    if channel_id not in _nature_cache:
        return True

    _, timestamp, _ = _nature_cache[channel_id]
    age = time.time() - timestamp
    return age > CACHE_COOLDOWN_S


def get_fetch_count(channel_id: str) -> int:
    """
    Algorithmically determine how many messages to fetch.
    - First time: 20 messages (full analysis)
    - After cache expires: 5 messages (quick refresh)
    """
    if channel_id not in _nature_cache:
        return FIRST_FETCH_COUNT

    _, timestamp, _ = _nature_cache[channel_id]
    age = time.time() - timestamp

    if age > CACHE_COOLDOWN_S:
        return REFRESH_FETCH_COUNT

    # Cache is still valid — no fetch needed
    return 0


def get_cached_nature(channel_id: str) -> Optional[str]:
    """Get the cached nature for a channel, or None if not cached/expired."""
    if channel_id not in _nature_cache:
        return None

    nature, timestamp, _ = _nature_cache[channel_id]
    age = time.time() - timestamp

    if age > CACHE_COOLDOWN_S:
        return None

    return nature


def update_nature_cache(channel_id: str, nature: str, message_count: int):
    """Update the nature cache for a channel."""
    _nature_cache[channel_id] = (nature, time.time(), message_count)
    logger.debug(f"Channel nature cached: #{channel_id} → {nature} ({message_count} msgs)")


async def analyze_channel_nature(channel: discord.TextChannel) -> str:
    """
    Analyze the nature of a channel.

    Algorithm:
    1. Check if cache is valid → return cached nature
    2. If cache expired → fetch 5 messages for quick refresh
    3. If not cached → fetch 20 messages for full analysis
    4. Classify nature from message content
    5. Cache the result
    """
    channel_id = str(channel.id)

    # Check cache first
    cached = get_cached_nature(channel_id)
    if cached:
        return cached

    # Determine fetch count
    fetch_count = get_fetch_count(channel_id)
    if fetch_count == 0:
        # Cache is valid, shouldn't be here
        return get_cached_nature(channel_id) or "general"

    try:
        # Fetch messages
        messages = [m async for m in channel.history(limit=fetch_count)]
        # Extract text content (exclude bots)
        msg_texts = []
        for m in messages:
            if m.author.bot:
                continue
            text = m.clean_content.strip()
            if text and len(text) > 2:
                msg_texts.append(text)

        if len(msg_texts) < 3:
            # Not enough messages to analyze
            logger.debug(f"Channel #{channel.name}: not enough messages for nature analysis")
            return "general"

        # Classify nature
        nature = classify_nature(msg_texts)
        update_nature_cache(channel_id, nature, len(msg_texts))

        logger.info(f"Channel nature: #{channel.name} → {nature} (from {len(msg_texts)} msgs)")
        return nature

    except Exception as e:
        logger.warning(f"Failed to analyze channel nature for #{channel.name}: {e}")
        return "general"


def clear_cache():
    """Clear the entire nature cache (for testing)."""
    _nature_cache.clear()


def get_cache_status() -> dict:
    """Get cache status for debugging."""
    now = time.time()
    return {
        channel_id: {
            "nature": nature,
            "age_hours": (now - ts) / 3600,
            "message_count": count,
            "expired": (now - ts) > CACHE_COOLDOWN_S,
        }
        for channel_id, (nature, ts, count) in _nature_cache.items()
    }
