"""
Message cache with TTL (Time-To-Live) management.

Stores channel messages in memory with automatic expiration:
- General chats: 3 days TTL
- Important messages (contain facts, names, instructions): 2 months TTL

The cache provides:
1. Wider context window for the AI (last 20-30 messages)
2. Conversation flow understanding (sees the full back-and-forth)
3. Automatic cleanup of expired messages
4. Importance classification (general vs important)

This is an ALGORITHMIC system — it classifies messages based on content
patterns and circumstances, not just storing everything blindly.
"""
import time
import json
import os
import re
import threading
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple
from loguru import logger


# TTL constants (in seconds)
TTL_GENERAL = 3 * 24 * 3600       # 3 days for general chats
TTL_IMPORTANT = 60 * 24 * 3600    # 60 days for important messages

# How many messages to keep in memory per channel (rolling window)
MAX_CHANNEL_CACHE = 100

# Importance detection patterns — messages matching these are stored longer
IMPORTANCE_PATTERNS = [
    # Personal information
    (r'\b(my\s+name\s+is|call\s+me|i\'?m\s+\w+|i\s+am\s+\w+)\b', "personal_info"),
    (r'\b(i\s+live\s+in|i\'?m\s+from|i\s+reside)\b', "personal_info"),
    (r'\b(i\s+(?:like|love|enjoy|hate|prefer))\b', "personal_info"),
    (r'\b(i\s+work\s+(?:as|at)|my\s+job|i\s+study)\b', "personal_info"),
    (r'\b(i\'?m\s+(\d)+\s+years?)\b', "personal_info"),
    (r'\b(my\s+(?:birthday|bd)\s+is)\b', "personal_info"),

    # Instructions/commands
    (r'\b(remember|don\'?t\s+forget|note\s+this|keep\s+in\s+mind)\b', "instruction"),
    (r'\b(stop|start|don\'?t|always|never)\s+\w+', "instruction"),
    (r'\b(bump|ping|revive)\b', "instruction"),

    # Relationship/emotional
    (r'\b(i\s+love\s+you|i\s+hate\s+you|you\'?re\s+my|best\s+friend|bro|sis)\b', "emotional"),
    (r'\b(miss\s+you|thinking\s+about\s+you|worried\s+about)\b', "emotional"),

    # Memorable moments
    (r'\b(remember\s+when|that\s+time\s+when|yesterday\s+we|last\s+week)\b', "memorable"),
    (r'\b(lol|lmao|that\s+was\s+hilarious|good\s+times)\b', "memorable"),
]


def classify_importance(text: str) -> Tuple[bool, str]:
    """
    Algorithmically classify if a message is important (should be stored longer).
    Returns (is_important, category).
    """
    if not text:
        return False, "general"

    text_lower = text.lower()
    for pattern, category in IMPORTANCE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True, category

    return False, "general"


class MessageCache:
    """
    Per-channel message cache with TTL-based expiration.

    Each cached message stores:
    - content: The message text
    - author: Author name
    - author_id: Author Discord ID
    - timestamp: When the message was sent
    - ttl: Time-to-live (based on importance)
    - importance: "general" or category name
    - is_bot: Whether the message was from the bot
    """

    def __init__(self):
        # ch_id -> deque of message dicts
        self._cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=MAX_CHANNEL_CACHE))
        self._lock = threading.Lock()
        self._last_cleanup = 0

    def add_message(
        self,
        ch_id: str,
        content: str,
        author: str,
        author_id: str,
        timestamp: float = None,
        is_bot: bool = False,
        message_id: int = None,
    ):
        """Add a message to the cache with automatic importance classification."""
        if timestamp is None:
            timestamp = time.time()

        is_important, category = classify_importance(content) if not is_bot else (False, "bot")
        ttl = TTL_IMPORTANT if is_important else TTL_GENERAL

        msg_dict = {
            "content": content,
            "author": author,
            "author_id": author_id,
            "timestamp": timestamp,
            "expires_at": timestamp + ttl,
            "importance": category,
            "is_important": is_important,
            "is_bot": is_bot,
            "message_id": message_id,
        }

        with self._lock:
            self._cache[ch_id].append(msg_dict)

        if is_important:
            logger.debug(f"Message cache: stored IMPORTANT message ({category}) from {author}: {content[:50]}")

    def get_context(self, ch_id: str, count: int = 4) -> List[dict]:
        """
        Get context messages for a channel using a smart algorithm:

        Returns the last `count` messages, PLUS all messages from the same
        user sent within the last 10 minutes (so the bot sees the full
        flow of one user's burst, even if it's more than `count` messages).

        This prevents the bot from replying to just the latest message
        when a user sent a multi-message burst (e.g., "I'm gonna sleep" ->
        "OKay" — the bot sees both, not just "OKay").
        """
        now = time.time()
        with self._lock:
            messages = list(self._cache.get(ch_id, []))

        # Filter out expired messages
        active = [m for m in messages if m["expires_at"] > now]

        if not active:
            return []

        # Get the last `count` messages as the base context
        base = active[-count:]

        # Find the latest non-bot author (the user we're replying to)
        latest_user_id = None
        for m in reversed(active):
            if not m["is_bot"]:
                latest_user_id = m["author_id"]
                break

        if latest_user_id:
            # Include ALL messages from this user within the last 10 minutes
            ten_min_ago = now - 600
            user_burst = [
                m for m in active
                if m["author_id"] == latest_user_id
                and m["timestamp"] > ten_min_ago
            ]
            # Merge base + user_burst, dedup by message_id, keep chronological order
            seen_ids = set()
            merged = []
            for m in active:
                if m in base or m in user_burst:
                    mid = m.get("message_id")
                    if mid is None or mid not in seen_ids:
                        seen_ids.add(mid)
                        merged.append(m)
            return merged

        return base

    def get_context_text(self, ch_id: str, count: int = 4) -> str:
        """
        Get a formatted transcript of context messages for the AI prompt.
        Uses the smart context algorithm (4 messages + user burst within 10 min).
        """
        msgs = self.get_context(ch_id, count)
        if not msgs:
            return ""

        now = time.time()
        lines = []
        for m in msgs:
            age = int(now - m["timestamp"])
            if age < 60:
                time_label = f"{age}s ago"
            elif age < 3600:
                time_label = f"{age // 60}m ago"
            elif age < 86400:
                time_label = f"{age // 3600}h ago"
            else:
                time_label = f"{age // 86400}d ago"

            label = "You" if m["is_bot"] else m["author"]
            content = m["content"]
            if not content:
                content = "[shared an image/attachment]"

            lines.append(f"[{time_label}] {label}: {content}")

        return "\n".join(lines)

    def get_active_user_count(self, ch_id: str, window_minutes: int = 10) -> int:
        """
        Count how many UNIQUE users have sent messages in the last N minutes.
        Used to detect busy channels — if 5+ users are chatting, the bot
        should NOT auto-step in (only respond when called).
        """
        now = time.time()
        window_start = now - (window_minutes * 60)
        with self._lock:
            messages = list(self._cache.get(ch_id, []))

        active_users = set()
        for m in messages:
            if m["timestamp"] > window_start and not m["is_bot"]:
                active_users.add(m["author_id"])

        return len(active_users)

    def cleanup_expired(self) -> int:
        """
        Remove all expired messages from all channels.
        Returns the number of messages removed.
        """
        now = time.time()
        removed = 0
        with self._lock:
            for ch_id in list(self._cache.keys()):
                before = len(self._cache[ch_id])
                # Keep only non-expired messages
                self._cache[ch_id] = deque(
                    [m for m in self._cache[ch_id] if m["expires_at"] > now],
                    maxlen=MAX_CHANNEL_CACHE,
                )
                after = len(self._cache[ch_id])
                removed += (before - after)

        if removed > 0:
            logger.info(f"Message cache: cleaned up {removed} expired messages")

        self._last_cleanup = now
        return removed

    def should_cleanup(self) -> bool:
        """Check if cleanup should run (every 30 minutes)."""
        return (time.time() - self._last_cleanup) > 1800

    def get_channel_stats(self, ch_id: str) -> dict:
        """Get stats for a channel (for debugging)."""
        now = time.time()
        with self._lock:
            msgs = list(self._cache.get(ch_id, []))
        active = [m for m in msgs if m["expires_at"] > now]
        important = [m for m in active if m["is_important"]]
        return {
            "total_cached": len(msgs),
            "active": len(active),
            "important": len(important),
            "expired": len(msgs) - len(active),
        }


# Singleton instance
_cache = MessageCache()


def get_cache() -> MessageCache:
    """Get the global message cache instance."""
    return _cache
