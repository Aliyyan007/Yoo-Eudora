"""
Reaction algorithm module for the Eudora persona.

Features:
1. Algorithmically decide whether to react (not on every message)
2. Better emoji selection based on message content and mood
3. Track recent reactions to avoid repetition
4. Cooldown between reactions (don't over-react)
5. Context-aware emoji selection (funny, sad, angry, etc.)

Algorithmic approach:
- Reaction chance: 15-25% (not every message)
- Higher chance for: messages directed at us, funny content, questions
- Lower chance for: random chat, short messages, our own messages
- Emoji selection based on sentiment + content keywords
- Track recent reactions per channel (avoid repeating same emoji)
- Cooldown: don't react more than once every 30 seconds per channel
"""
import re
import time
import random
from typing import Optional, Dict, List, Set
from collections import deque
from loguru import logger


# ── Emoji categories ─────────────────────────────────────────────────────────

# Positive/funny reactions
FUNNY_EMOJIS = ["\U0001f480", "\U0001f602", "\U0001f923", "\U0001f605", "\U0001fa80", "\U0001f60f"]

# Agreement/acknowledgment
AGREE_EMOJIS = ["\U0001f44d", "\U0001f4af", "\U0001f91e", "\U0001f525", "\u2705"]

# Love/heart reactions
HEART_EMOJIS = ["\u2764\ufe0f", "\U0001f49c", "\U0001f499", "\U0001f497", "\U0001f970"]

# Sad/venting reactions
SAD_EMOJIS = ["\U0001f614", "\U0001f622", "\U0001f97a", "\U0001f494", "\U0001f62d"]

# Thinking/confused
THINK_EMOJIS = ["🤔", "💭", "🧐", "👀"]

# Cool/chill
COOL_EMOJIS = ["😎", "🤙", "✨", "🌙", "☕"]

# British/casual
BRITISH_EMOJIS = ["🫖", "🌧️", "🍻", "🇬🇧"]

# Excited
EXCITED_EMOJIS = ["🔥", "✨", "🎉", "🤩", "💥"]

# Annoyed/sarcastic
ANNOYED_EMOJIS = ["🙄", "😒", "🫠", "💀"]

# ── Content patterns for emoji selection ─────────────────────────────────────

FUNNY_PATTERNS = [
    r'\b(lol|lmao|lmfao|haha|hehe|fr|nah)\b',
    r'\b(bro|bruh|dead|killed)\b',
    r'(?i)funny|joke|memes?',
]

SAD_PATTERNS = [
    r'\b(sad|depressed|anxious|stressed|tired|alone|lonely|cry|hurt|pain)\b',
    r'\b(venting|rough|hard day|bad day|struggling)\b',
]

EXCITED_PATTERNS = [
    r'\b(omg|wow|no way|amazing|incredible|let\'?s go|yesss|finally)\b',
    r'\b(hyped|excited|can\'?t wait|pumped)\b',
]

AGREE_PATTERNS = [
    r'\b(true|facts|based|real|fr|agreed|exactly|right|correct)\b',
    r'\b(yeah|yes|yep|yup|indeed)\b',
]

QUESTION_PATTERNS = [
    r'\?\s*$',
    r'\b(what|why|how|when|where|who|which)\b',
]

COOL_PATTERNS = [
    r'\b(chill|vibing|vibe|relax|calm|lofi|coffee|tea)\b',
]

ANNOYED_PATTERNS = [
    r'\b(annoying|stupid|dumb|whatever|bruv|allow it)\b',
    r'\b(shut up|stfu|idiot)\b',
]

# Compiled patterns
_FUNNY_RX = [re.compile(p, re.IGNORECASE) for p in FUNNY_PATTERNS]
_SAD_RX = [re.compile(p, re.IGNORECASE) for p in SAD_PATTERNS]
_EXCITED_RX = [re.compile(p, re.IGNORECASE) for p in EXCITED_PATTERNS]
_AGREE_RX = [re.compile(p, re.IGNORECASE) for p in AGREE_PATTERNS]
_QUESTION_RX = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]
_COOL_RX = [re.compile(p, re.IGNORECASE) for p in COOL_PATTERNS]
_ANNOYED_RX = [re.compile(p, re.IGNORECASE) for p in ANNOYED_PATTERNS]


# ── Reaction tracker ─────────────────────────────────────────────────────────

class ReactionTracker:
    """
    Tracks reactions per channel to avoid over-reacting and repetition.

    Algorithmic behavior:
    - Track last reaction time per channel (cooldown)
    - Track recent emojis per channel (avoid repetition)
    - Track reaction count per channel (daily limit)
    """

    def __init__(self):
        # Last reaction time per channel
        self._last_reaction: Dict[str, float] = {}
        # Recent emojis per channel (last 5)
        self._recent_emojis: Dict[str, deque] = {}
        # Reaction count per channel (daily)
        self._daily_count: Dict[str, int] = {}
        # Daily limit
        self._daily_limit = 30
        # Cooldown between reactions (seconds)
        self._cooldown_s = 30

    def can_react(self, channel_id: str) -> bool:
        """Check if we can react in this channel (cooldown + daily limit)."""
        now = time.time()

        # Check cooldown
        last = self._last_reaction.get(channel_id, 0)
        if now - last < self._cooldown_s:
            return False

        # Check daily limit
        if self._daily_count.get(channel_id, 0) >= self._daily_limit:
            return False

        return True

    def is_recent_emoji(self, channel_id: str, emoji: str) -> bool:
        """Check if an emoji was recently used in this channel."""
        recent = self._recent_emojis.get(channel_id, deque(maxlen=5))
        return emoji in recent

    def record_reaction(self, channel_id: str, emoji: str):
        """Record that a reaction was sent."""
        self._last_reaction[channel_id] = time.time()
        if channel_id not in self._recent_emojis:
            self._recent_emojis[channel_id] = deque(maxlen=5)
        self._recent_emojis[channel_id].append(emoji)
        self._daily_count[channel_id] = self._daily_count.get(channel_id, 0) + 1

    def reset_daily(self):
        """Reset daily counts (call at midnight)."""
        self._daily_count.clear()


# Singleton tracker
_tracker = ReactionTracker()


def get_tracker() -> ReactionTracker:
    """Get the global reaction tracker."""
    return _tracker


# ── Reaction decision ────────────────────────────────────────────────────────

def should_react(
    text: str,
    is_directed_at_bot: bool = False,
    is_reply_to_bot: bool = False,
    is_mention: bool = False,
    mood: str = "chill",
    channel_id: str = "",
) -> bool:
    """
    Algorithmically decide whether to react to a message.

    Factors:
    - Base chance: 15%
    - +10% if directed at bot
    - +10% if reply to bot
    - +5% if mention
    - +5% if funny content
    - +5% if excited mood
    - -5% if very short message (< 5 chars)
    - -10% if in cooldown
    - 0% if daily limit reached
    """
    # Check tracker first
    if not _tracker.can_react(channel_id):
        return False

    # Base chance
    chance = 0.15

    # Directed at bot
    if is_directed_at_bot:
        chance += 0.10

    # Reply to bot
    if is_reply_to_bot:
        chance += 0.10

    # Mention
    if is_mention:
        chance += 0.05

    # Funny content
    if any(rx.search(text) for rx in _FUNNY_RX):
        chance += 0.05

    # Excited mood
    if mood in ("hyped", "excited", "giddy", "playful"):
        chance += 0.05

    # Very short message
    if len(text.strip()) < 5:
        chance -= 0.05

    # Annoyed mood — less likely to react
    if mood in ("annoyed", "angry"):
        chance -= 0.05

    # Cap at 40%
    chance = min(chance, 0.40)

    return random.random() < chance


def select_emoji(
    text: str,
    mood: str = "chill",
    channel_id: str = "",
) -> Optional[str]:
    """
    Algorithmically select an emoji based on message content and mood.

    Returns an emoji string or None.
    """
    # Determine content category
    categories = []

    if any(rx.search(text) for rx in _FUNNY_RX):
        categories.append("funny")
    if any(rx.search(text) for rx in _SAD_RX):
        categories.append("sad")
    if any(rx.search(text) for rx in _EXCITED_RX):
        categories.append("excited")
    if any(rx.search(text) for rx in _AGREE_RX):
        categories.append("agree")
    if any(rx.search(text) for rx in _QUESTION_RX):
        categories.append("question")
    if any(rx.search(text) for rx in _COOL_RX):
        categories.append("cool")
    if any(rx.search(text) for rx in _ANNOYED_RX):
        categories.append("annoyed")

    # Mood overrides
    if mood in ("hyped", "excited", "giddy") and "excited" not in categories:
        categories.append("excited")
    if mood in ("sad", "lonely") and "sad" not in categories:
        categories.append("sad")
    if mood in ("annoyed", "angry") and "annoyed" not in categories:
        categories.append("annoyed")
    if mood in ("chill", "playful") and not categories:
        categories.append("cool")

    # Default to cool if no category
    if not categories:
        categories = ["cool"]

    # Map categories to emoji pools
    emoji_pools = {
        "funny": FUNNY_EMOJIS,
        "sad": SAD_EMOJIS,
        "excited": EXCITED_EMOJIS,
        "agree": AGREE_EMOJIS,
        "question": THINK_EMOJIS,
        "cool": COOL_EMOJIS,
        "annoyed": ANNOYED_EMOJIS,
    }

    # Collect candidate emojis from all matched categories
    candidates = []
    for cat in categories:
        candidates.extend(emoji_pools.get(cat, []))

    if not candidates:
        return None

    # Filter out recently used emojis
    available = [e for e in candidates if not _tracker.is_recent_emoji(channel_id, e)]

    # If all were recently used, use candidates anyway
    if not available:
        available = candidates

    # Random selection
    emoji = random.choice(available)

    # Record the reaction
    _tracker.record_reaction(channel_id, emoji)

    return emoji


def react_to_message(
    text: str,
    is_directed_at_bot: bool = False,
    is_reply_to_bot: bool = False,
    is_mention: bool = False,
    mood: str = "chill",
    channel_id: str = "",
) -> Optional[str]:
    """
    Main entry point: decide whether to react and select an emoji.

    Returns an emoji string or None (if no reaction).
    """
    if not should_react(
        text, is_directed_at_bot, is_reply_to_bot, is_mention, mood, channel_id
    ):
        return None

    return select_emoji(text, mood, channel_id)
