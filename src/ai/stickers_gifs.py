"""
Stickers and GIFs module for the Eudora persona.

Features:
1. Send stickers from the guild's sticker collection
2. Algorithmically decide when to send stickers (low frequency)
3. Select appropriate stickers based on message content
4. Track recent stickers to avoid repetition
5. Cooldown between sticker sends
6. Send GIFs using Discord's native internal GIF search API (falls back to Tenor API)
   - Uses Tenor page URLs that Discord unfurls into native gifv embeds
7. Send GIFs/stickers on user request ("send me a gif", "send a sticker")

Algorithmic approach:
- Sticker chance: 5-8% (very low frequency)
- Higher chance for: funny content, reactions, emotional messages
- Lower chance for: questions, serious discussions
- Track recent stickers per channel (avoid repetition)
- Cooldown: don't send stickers more than once every 5 minutes per channel
- Fallback: if no stickers available, skip silently
- GIF chance: 3-5% (very low frequency)
- GIFs sent as Tenor page URLs (Discord unfurls them into native gifv embeds)
- User-requested GIFs/stickers: always send (override cooldown)
"""
import os
import re
import time
import random
import asyncio
import aiohttp
from typing import Optional, List, Dict
from collections import deque
from loguru import logger
import discord


# ── Sticker selection patterns ───────────────────────────────────────────────

# When to use stickers (content patterns)
STICKER_TRIGGERS = {
    "funny": [r'\b(lol|lmao|lmfao|haha|hehe)\b', r'(?i)funny|joke|memes?'],
    "sad": [r'\b(sad|depressed|anxious|stressed|cry|hurt)\b'],
    "excited": [r'\b(omg|wow|amazing|incredible|yesss|finally)\b'],
    "love": [r'\b(love|cute|adorable|sweet|wholesome)\b'],
    "angry": [r'\b(angry|mad|annoying|stupid|dumb)\b'],
    "agree": [r'\b(true|facts|based|real|fr|agreed)\b'],
}

# Compiled patterns
_STICKER_RX = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in STICKER_TRIGGERS.items()
}

# ── User request detection (for GIF/sticker requests) ───────────────────────

GIF_REQUEST_PATTERNS = [
    r'\b(send|give|show).*(?:me\s+)?(?:a\s+)?(?:gif|gifs|giphy|tenor)\b',
    r'\b(gif|giphy|tenor)\s+(?:me|please|plz|pls)\b',
    r'\b(send|post).*(?:a\s+)?(?:gif|sticker)\b',
    r'\b(I\s+want|gimme|drop).*(?:a\s+)?(?:gif|sticker)\b',
    r'\b(send|post|drop).*(?:a\s+)?(?:sticker|stickers)\b',
    r'\b(sticker\s+me|send\s+sticker)\b',
]

_COMPILED_GIF_REQUESTS = [re.compile(p, re.IGNORECASE) for p in GIF_REQUEST_PATTERNS]


def is_gif_request(text: str) -> bool:
    """Check if a message is requesting a GIF or sticker."""
    if not text:
        return False
    return any(rx.search(text) for rx in _COMPILED_GIF_REQUESTS)


class StickerManager:
    """
    Manages sticker sending with algorithmic control.

    Algorithmic behavior:
    - Low frequency: 5-8% chance per eligible message
    - Cooldown: 5 minutes between stickers per channel
    - Track recent stickers (avoid repetition)
    - Daily limit: 10 stickers per channel
    - Select stickers based on message content
    - User requests override cooldown
    """

    def __init__(self):
        # Last sticker time per channel
        self._last_sticker: Dict[str, float] = {}
        # Recent stickers per channel (last 5)
        self._recent_stickers: Dict[str, deque] = {}
        # Daily count per channel
        self._daily_count: Dict[str, int] = {}
        # Cooldown: 5 minutes
        self._cooldown_s = 300
        # Daily limit
        self._daily_limit = 10
        # Base chance (env-tunable — slightly raised for more expressive convos)
        try:
            self._base_chance = float(os.getenv("STICKER_CHANCE", "0.08"))
        except ValueError:
            self._base_chance = 0.08

    def can_send_sticker(self, channel_id: str, force: bool = False) -> bool:
        """Check if we can send a sticker in this channel."""
        if force:
            return True

        now = time.time()

        # Check cooldown
        last = self._last_sticker.get(channel_id, 0)
        if now - last < self._cooldown_s:
            return False

        # Check daily limit
        if self._daily_count.get(channel_id, 0) >= self._daily_limit:
            return False

        return True

    def should_send_sticker(self, text: str, channel_id: str) -> bool:
        """
        Algorithmically decide whether to send a sticker.
        """
        if not self.can_send_sticker(channel_id):
            return False

        chance = self._base_chance

        # Increase chance for funny content
        if any(rx.search(text) for rx in _STICKER_RX["funny"]):
            chance += 0.03

        # Increase chance for emotional content
        if any(rx.search(text) for rx in _STICKER_RX["sad"]):
            chance += 0.02
        if any(rx.search(text) for rx in _STICKER_RX["excited"]):
            chance += 0.02
        if any(rx.search(text) for rx in _STICKER_RX["love"]):
            chance += 0.02

        # Cap at 16%
        chance = min(chance, 0.16)

        return random.random() < chance

    def select_sticker(
        self,
        guild: discord.Guild,
        text: str,
        channel_id: str,
    ) -> Optional[discord.GuildSticker]:
        """
        Algorithmically select a sticker from the guild's collection.

        Returns a GuildSticker or None if no suitable sticker found.
        """
        try:
            stickers = list(guild.stickers)
        except Exception:
            return None

        if not stickers:
            return None

        # Determine content category
        categories = []
        for cat, patterns in _STICKER_RX.items():
            if any(rx.search(text) for rx in patterns):
                categories.append(cat)

        # Filter out recently used stickers — and intimate-themed ones: a
        # "Kiss"/"Love" sticker to a stranger reads as flirting and is a
        # selfbot red flag. Only a direct romantic request could justify one,
        # and we never get that context here.
        _intimate = re.compile(r"kiss|love|heart|marry|flirt|babe|crush|smooch|valentine", re.I)
        pool = [s for s in stickers if not _intimate.search(s.name or "")]
        if not pool:
            return None
        recent = self._recent_stickers.get(channel_id, deque(maxlen=5))
        available = [s for s in pool if s.id not in recent]

        if not available:
            available = pool

        # Try to match sticker name to content category
        if categories:
            # Look for stickers with names matching the category
            for cat in categories:
                cat_keywords = {
                    "funny": ["lol", "laugh", "funny", "haha", "lmao", "dead", "cry_laugh"],
                    "sad": ["sad", "cry", "tears", "depressed", "alone"],
                    "excited": ["wow", "amazing", "excited", "hype", "party", "celebrate"],
                    "love": ["love", "heart", "cute", "adorable", "sweet", "kiss"],
                    "angry": ["angry", "mad", "rage", "annoyed"],
                    "agree": ["thumbs", "yes", "agree", "based", "ok", "cool"],
                }
                keywords = cat_keywords.get(cat, [])
                for sticker in available:
                    name_lower = sticker.name.lower()
                    if any(kw in name_lower for kw in keywords):
                        self._record_sticker(channel_id, sticker.id)
                        return sticker

        # No keyword match — pick random
        sticker = random.choice(available)
        self._record_sticker(channel_id, sticker.id)
        return sticker

    def _record_sticker(self, channel_id: str, sticker_id: int):
        """Record that a sticker was sent."""
        self._last_sticker[channel_id] = time.time()
        if channel_id not in self._recent_stickers:
            self._recent_stickers[channel_id] = deque(maxlen=5)
        self._recent_stickers[channel_id].append(sticker_id)
        self._daily_count[channel_id] = self._daily_count.get(channel_id, 0) + 1

    def reset_daily(self):
        """Reset daily counts."""
        self._daily_count.clear()


# ── GIF management (Tenor API) ───────────────────────────────────────────────

# GIF search queries by content category
GIF_QUERIES = {
    "funny": ["funny cat", "lol meme", "laughing", "dead"],
    "sad": ["sad", "crying", "depressed", "alone"],
    "excited": ["excited", "hype", "celebration", "yesss"],
    "love": ["love", "heart", "cute", "wholesome"],
    "agree": ["thumbs up", "based", "agree", "nodding"],
    "cool": ["cool", "chill", "vibing", "lofi"],
}

# Default GIF queries (when no category matches)
DEFAULT_GIF_QUERIES = ["cool", "vibing", "chill", "lofi", "mood"]


class GIFManager:
    """
    Manages GIF sending via Discord's native internal GIF search API,
    with fallback to the Tenor API.

    Primary: Discord's internal /api/v9/gifs/search endpoint (the same one
    the built-in GIF picker uses). Authenticates with the user's Discord
    token — no Tenor API key needed. Returns Tenor page URLs that Discord
    unfurls into native gifv embeds (with the "via Tenor" label), exactly
    like the built-in GIF picker.

    Fallback: Tenor API using the public GBoard/Vencord key (3Z0688EVWYKH).

    Algorithmic behavior:
    - Very low frequency: 3-5% chance
    - Cooldown: 10 minutes between GIFs per channel
    - Daily limit: 5 GIFs per channel
    - Select GIF query based on message content
    - User requests override cooldown (always send when asked)
    """

    def __init__(self):
        self._last_gif: Dict[str, float] = {}
        self._daily_count: Dict[str, int] = {}
        self._cooldown_s = 600  # 10 minutes
        self._daily_limit = 5
        # Base chance (env-tunable — slightly raised for more expressive convos)
        try:
            self._base_chance = float(os.getenv("GIF_CHANCE", "0.05"))
        except ValueError:
            self._base_chance = 0.05
        # Discord user token for native GIF search (set via set_discord_token)
        self._discord_token = None
        # Tenor API key — defaults to the public GBoard/Vencord key (3Z0688EVWYKH)
        # used as a fallback when Discord's internal API is unavailable.
        # Can be overridden via set_api_key().
        self._tenor_key = "3Z0688EVWYKH"

    def set_api_key(self, key: str):
        """Set the Tenor API key (overrides the default public key)."""
        self._tenor_key = key

    def set_discord_token(self, token: str):
        """Set the Discord user token for native GIF search."""
        self._discord_token = token

    def should_send_gif(self, text: str, channel_id: str, force: bool = False) -> bool:
        """Algorithmically decide whether to send a GIF."""
        if force:
            return True

        now = time.time()

        # Check cooldown
        last = self._last_gif.get(channel_id, 0)
        if now - last < self._cooldown_s:
            return False

        # Check daily limit
        if self._daily_count.get(channel_id, 0) >= self._daily_limit:
            return False

        chance = self._base_chance

        # Increase for funny content
        if any(rx.search(text) for rx in _STICKER_RX["funny"]):
            chance += 0.02

        # Cap at 11%
        chance = min(chance, 0.11)

        return random.random() < chance

    def get_gif_query(self, text: str) -> Optional[str]:
        """
        Algorithmically determine the best GIF search query based on content.
        Returns a search query string or None.
        """
        categories = []
        for cat, patterns in _STICKER_RX.items():
            if any(rx.search(text) for rx in patterns):
                categories.append(cat)

        if not categories:
            # Default to cool/vibing
            return random.choice(DEFAULT_GIF_QUERIES)

        # Pick a random category from matched ones
        cat = random.choice(categories)
        queries = GIF_QUERIES.get(cat, DEFAULT_GIF_QUERIES)

        return random.choice(queries)

    async def search_gif_discord(self, query: str, limit: int = 8) -> Optional[Dict[str, str]]:
        """
        Search for a GIF using Discord's internal /api/v9/gifs/search endpoint
        (the same one the built-in GIF picker uses). Authenticates with the
        user's Discord token — no Tenor API key needed.

        Returns a dict {"page_url": "...", "direct_url": "...", "id": "..."}
        or None if no token is set, the request fails, or no results are found.

        Note: Discord's API requires limit >= 20, so the limit is clamped
        to a minimum of 20. We still pick randomly from the top results.
        """
        if not self._discord_token:
            return None

        try:
            url = "https://discord.com/api/v9/gifs/search"
            params = {
                "q": query,
                "media_format": "gif",
                "provider": "tenor",
                "locale": "en-US",
                # Discord's API requires limit >= 20
                "limit": max(limit, 20),
            }
            headers = {"Authorization": self._discord_token}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.debug(f"Discord GIF search returned status {resp.status}")
                        return None

                    data = await resp.json()

                    # Discord returns a JSON array of GIF objects
                    if not data or not isinstance(data, list):
                        return None

                    # Pick a random result from the top results
                    result = random.choice(data[:limit])

                    # url = Tenor page URL (for native gifv embedding)
                    page_url = result.get("url")
                    # src = direct media URL (fallback)
                    direct_url = result.get("src")
                    # id = Tenor GIF ID (for registering the share)
                    gif_id = result.get("id")

                    if not page_url and not direct_url:
                        return None

                    return {
                        "page_url": page_url,
                        "direct_url": direct_url,
                        "id": gif_id,
                    }

        except asyncio.TimeoutError:
            logger.debug("Discord GIF search timeout")
            return None
        except Exception as e:
            logger.debug(f"Discord GIF search error: {e}")
            return None

    async def register_gif_select(self, gif_id: str, query: str):
        """
        Register a GIF share via Discord's /api/v9/gifs/select endpoint.
        This is the same call Discord makes when a user picks a GIF from the
        built-in picker (for analytics/tracking consistency).

        Fire-and-forget: all exceptions are caught silently.
        """
        if not self._discord_token or not gif_id:
            return

        try:
            url = "https://discord.com/api/v9/gifs/select"
            headers = {
                "Authorization": self._discord_token,
                "Content-Type": "application/json",
            }
            body = {"id": gif_id, "q": query}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    # We don't care about the response — just register the share
                    pass

        except Exception as e:
            logger.debug(f"Discord GIF select error: {e}")

    async def search_gif_native(self, query: str, limit: int = 8) -> Optional[Dict[str, str]]:
        """
        Search for a GIF and return both the Tenor page URL (for native gifv
        embedding) and the direct media URL (fallback).

        Priority order:
        1. Discord's internal /api/v9/gifs/search endpoint (if token is set)
        2. Tenor API (using the default public key 3Z0688EVWYKH)

        Returns a dict {"page_url": "...", "direct_url": "..."} or None if
        both methods fail. When using Discord's API, the dict also includes
        "id" (the Tenor GIF ID for registering the share).
        """
        # 1. Try Discord's internal GIF search API first
        if self._discord_token:
            result = await self.search_gif_discord(query, limit=limit)
            if result:
                return result
            logger.debug("Discord GIF search failed, falling back to Tenor API")

        # 2. Fall back to the Tenor API
        try:
            # Use Tenor v1 API (works with the public GBoard key 3Z0688EVWYKH)
            # The v2 API (tenor.googleapis.com) requires a different, private key.
            if self._tenor_key:
                url = "https://api.tenor.com/v1/search"
                params = {
                    "q": query,
                    "key": self._tenor_key,
                    "limit": limit,
                    "media_filter": "gif",
                    "contentfilter": "medium",
                }
            else:
                # Fall back to v1 API without key (may not work)
                url = "https://g.tenor.com/v1/search"
                params = {
                    "q": query,
                    "limit": limit,
                    "contentfilter": "medium",
                    "media_filter": "gif",
                }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.debug(f"Tenor API returned status {resp.status}")
                        return None

                    data = await resp.json()

                    results = data.get("results", [])
                    if not results:
                        return None

                    # Pick a random result from the top results
                    result = random.choice(results[:limit])

                    # Get the Tenor page URL (itemurl) for native gifv embedding
                    page_url = result.get("itemurl")

                    # Get the direct GIF URL as a fallback
                    # v2 API: result["media_formats"]["gif"]["url"]
                    # v1 API: result["media"][0]["gif"]["url"]
                    direct_url = None
                    if "media_formats" in result:
                        gif_data = result["media_formats"].get("gif", {})
                        direct_url = gif_data.get("url")
                        if not direct_url:
                            # Try mediumgif as fallback
                            gif_data = result["media_formats"].get("mediumgif", {})
                            direct_url = gif_data.get("url")
                    elif "media" in result and result["media"]:
                        gif_data = result["media"][0].get("gif", {})
                        direct_url = gif_data.get("url")

                    if not page_url and not direct_url:
                        return None

                    return {
                        "page_url": page_url,
                        "direct_url": direct_url,
                    }

        except asyncio.TimeoutError:
            logger.debug("Tenor API timeout")
            return None
        except Exception as e:
            logger.debug(f"Tenor API error: {e}")
            return None

    async def search_gif(self, query: str, limit: int = 8) -> Optional[str]:
        """
        Search for a GIF using Discord's native GIF search API (falls back to
        Tenor).

        Returns a URL that Discord will embed as a native gifv/image embed:
        - Tenor page URLs (tenor.com/view/...) → native gifv embed with "via Tenor"
        - Klipy/direct GIF URLs → image embed (from Discord's internal API)
        Falls back to the direct media URL if no page URL is available.
        Returns None if search fails.

        If the result includes a GIF "id" (from Discord's internal API), the
        share is registered via /api/v9/gifs/select for tracking consistency.
        """
        result = await self.search_gif_native(query, limit=limit)
        if not result:
            return None

        # Register the GIF share with Discord (analytics/tracking consistency)
        gif_id = result.get("id")
        if gif_id:
            await self.register_gif_select(gif_id, query)

        page_url = result.get("page_url")
        direct_url = result.get("direct_url")

        # Tenor page URLs unfurl into native gifv embeds in Discord.
        # Other providers (e.g. Klipy) may not unfurl, so use the direct
        # GIF URL for reliable image embedding.
        if page_url and "tenor.com" in page_url:
            return page_url
        # For non-Tenor page URLs, prefer the direct URL for reliable embedding
        return direct_url or page_url

    async def search_gif_fallback(self, query: str) -> Optional[str]:
        """
        Fallback: scrape the Tenor search page for direct GIF URLs.
        This works without an API key.
        """
        try:
            search_url = f"https://tenor.com/search/{query.replace(' ', '-')}-gifs"
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()
                    # Find direct GIF URLs (media.tenor.com)
                    urls = re.findall(r'https://media\.tenor\.com/[^"\']+\.gif', html)
                    if urls:
                        return random.choice(urls[:10])
                    return None
        except Exception:
            return None

    def record_gif(self, channel_id: str):
        """Record that a GIF was sent."""
        self._last_gif[channel_id] = time.time()
        self._daily_count[channel_id] = self._daily_count.get(channel_id, 0) + 1

    def reset_daily(self):
        """Reset daily counts."""
        self._daily_count.clear()


# Singleton instances
_sticker_manager = StickerManager()
_gif_manager = GIFManager()


def get_sticker_manager() -> StickerManager:
    """Get the global sticker manager."""
    return _sticker_manager


def get_gif_manager() -> GIFManager:
    """Get the global GIF manager."""
    return _gif_manager
