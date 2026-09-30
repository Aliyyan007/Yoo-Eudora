"""
User behavior understanding module for the Eudora persona.

Features:
1. Algorithmically decide which messages to store in memory
2. Exclude low-value messages (greetings, filler, spam)
3. Include high-value messages (facts, questions, stories, emotions)
4. Track user engagement patterns
5. Classify message value (low, medium, high)

Algorithmic approach:
- Low value: greetings, single words, filler, bot commands
- Medium value: questions, opinions, reactions
- High value: personal facts, stories, emotional sharing, requests
- Exclude: spam, repeated messages, very short messages, pure emoji
- Track per-user engagement score (how much they interact with the bot)
"""
import re
import time
from typing import Tuple, List, Dict, Optional
from collections import defaultdict
from loguru import logger


# ── Message value classification ─────────────────────────────────────────────

# Low-value patterns (exclude from memory)
LOW_VALUE_PATTERNS = [
    r'^(hi|hey|hello|yo|sup|hiya|heya|howdy)\s*$',  # Greetings
    r'^(ok|okay|k|kk|sure|yeah|yes|no|nope|yep|yup)\s*$',  # Acknowledgments
    r'^(lol|lmao|lmfao|haha|hehe|fr|nah|bruh|bro)\s*$',  # Filler
    r'^(thanks|thx|ty|np|no problem)\s*$',  # Pleasantries
    r'^(bye|gtg|cya|later|goodnight|gn)\s*$',  # Departures
    r'^(wbu|hru|wyd|wys)\s*$',  # Short questions
    r'^[\U0001f300-\U0001f9ff\s]+$',  # Pure emoji
    r'^[\.!\?]+$',  # Pure punctuation
]

# High-value patterns (include in memory)
HIGH_VALUE_PATTERNS = [
    r'\b(my name is|i\'?m called|i go by)\b',  # Name facts
    r'\b(i\'?m\s+\d+\s+years?\s+old|i\'?m\s+\d+)\b',  # Age facts
    r'\b(i live in|i\'?m from|i reside in)\b',  # Location facts
    r'\b(i like|i love|i enjoy|i hate)\b',  # Preferences
    r'\b(i work as|i study|i\'?m a student|i\'?m a)\b',  # Occupation
    r'\b(my (?:favorite|favourite) is|i love (?:listening|watching|playing))\b',  # Favorites
    r'\b(i feel|i\'?m feeling|i\'?m sad|i\'?m happy|i\'?m stressed)\b',  # Emotions
    r'\b(yesterday i|today i|this weekend i|last night i)\b',  # Stories
    r'\b(can you|could you|would you|help me)\b',  # Requests
    r'\b(i think|i believe|in my opinion|imo|tbh)\b',  # Opinions
    r'\b(my (?:dog|cat|pet|brother|sister|mom|mum|dad|parents?))\b',  # Personal info
]

# Medium-value patterns (include with lower priority)
MEDIUM_VALUE_PATTERNS = [
    r'\?',  # Questions
    r'\b(what|why|how|when|where|who|which)\b',  # Question words
    r'\b(anyone|someone|everybody)\b',  # Addressing group
    r'\b(i (?:just|recently|also|too))\b',  # Personal statements
]

# Spam patterns (exclude entirely)
SPAM_PATTERNS = [
    r'(.)\1{10,}',  # Repeated characters (e.g., "aaaaaaaaaa")
    r'(\S+)(\s+\1){5,}',  # Repeated words
    r'^https?://\S+$',  # Pure links
    r'^<:\w+:\d+>$',  # Pure custom emoji
]

# Compiled patterns
_LOW_RX = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in LOW_VALUE_PATTERNS]
_HIGH_RX = [re.compile(p, re.IGNORECASE) for p in HIGH_VALUE_PATTERNS]
_MEDIUM_RX = [re.compile(p, re.IGNORECASE) for p in MEDIUM_VALUE_PATTERNS]
_SPAM_RX = [re.compile(p, re.IGNORECASE) for p in SPAM_PATTERNS]


def classify_message_value(text: str) -> Tuple[str, int]:
    """
    Algorithmically classify the value of a message for memory storage.

    Returns (value_level, score):
    - value_level: "exclude", "low", "medium", "high"
    - score: 0-10 (0 = exclude, 10 = highest value)
    """
    if not text or not text.strip():
        return ("exclude", 0)

    text = text.strip()

    # Check spam first (auto-exclude)
    for rx in _SPAM_RX:
        if rx.search(text):
            return ("exclude", 0)

    # Check low-value patterns
    for rx in _LOW_RX:
        if rx.search(text):
            return ("low", 1)

    # Check high-value patterns
    for rx in _HIGH_RX:
        if rx.search(text):
            return ("high", 8)

    # Check medium-value patterns
    for rx in _MEDIUM_RX:
        if rx.search(text):
            return ("medium", 5)

    # Length-based fallback
    if len(text) < 10:
        return ("low", 2)
    elif len(text) > 100:
        return ("medium", 4)

    return ("medium", 3)


def should_store_message(text: str) -> bool:
    """
    Algorithmically decide whether to store a message in memory.

    Returns True if the message should be stored, False otherwise.
    """
    level, score = classify_message_value(text)
    return level in ("medium", "high")


def extract_potential_facts(text: str) -> List[str]:
    """
    Algorithmically extract potential facts from a message.

    Returns a list of fact strings that could be stored in memory.
    """
    if not should_store_message(text):
        return []

    facts = []
    text_lower = text.lower()

    # Name facts
    m = re.search(r'\b(?:my name is|i\'?m called|i go by)\s+(\w+)', text, re.I)
    if m:
        facts.append(f"their name is {m.group(1)}")

    # Age facts
    m = re.search(r'\b(?:i\'?m\s+(\d+)\s+years?\s+old|i\'?m\s+(\d+))\b', text, re.I)
    if m:
        age = m.group(1) or m.group(2)
        facts.append(f"they are {age} years old")

    # Location facts
    m = re.search(r'\b(?:i live in|i\'?m from|i reside in)\s+([a-zA-Z\s]+)', text, re.I)
    if m:
        facts.append(f"they are from {m.group(1).strip()}")

    # Preference facts
    m = re.search(r'\b(?:i like|i love|i enjoy)\s+([a-zA-Z\s]+)', text, re.I)
    if m:
        facts.append(f"they like {m.group(1).strip()}")

    # Occupation facts
    m = re.search(r'\b(?:i work as|i study|i\'?m a student)\s+([a-zA-Z\s]+)', text, re.I)
    if m:
        facts.append(f"they work/study: {m.group(1).strip()}")

    return facts


# ── User engagement tracking ─────────────────────────────────────────────────

class UserEngagementTracker:
    """
    Tracks user engagement patterns.

    Algorithmic behavior:
    - Track message count per user
    - Track last activity time per user
    - Track average message length per user
    - Track question frequency per user
    - Compute engagement score (0-10)
    """

    def __init__(self):
        self._msg_count: Dict[str, int] = defaultdict(int)
        self._last_activity: Dict[str, float] = {}
        self._total_length: Dict[str, int] = defaultdict(int)
        self._question_count: Dict[str, int] = defaultdict(int)
        self._high_value_count: Dict[str, int] = defaultdict(int)

    def record_message(self, user_id: str, text: str):
        """Record a message from a user."""
        self._msg_count[user_id] += 1
        self._last_activity[user_id] = time.time()
        self._total_length[user_id] += len(text)

        if "?" in text:
            self._question_count[user_id] += 1

        level, _ = classify_message_value(text)
        if level == "high":
            self._high_value_count[user_id] += 1

    def get_engagement_score(self, user_id: str) -> float:
        """
        Algorithmically compute an engagement score (0-10) for a user.

        Factors:
        - Message count (more = higher)
        - Average message length (longer = higher)
        - Question frequency (more questions = higher)
        - High-value message ratio
        - Recency (recent activity = higher)
        """
        count = self._msg_count.get(user_id, 0)
        if count == 0:
            return 0.0

        # Average length
        avg_len = self._total_length.get(user_id, 0) / count
        len_score = min(avg_len / 50, 3.0)  # Max 3 points for length

        # Question frequency
        q_ratio = self._question_count.get(user_id, 0) / count
        q_score = min(q_ratio * 5, 2.0)  # Max 2 points for questions

        # High-value ratio
        hv_ratio = self._high_value_count.get(user_id, 0) / count
        hv_score = min(hv_ratio * 5, 3.0)  # Max 3 points for high-value

        # Activity score (based on count)
        count_score = min(count / 10, 2.0)  # Max 2 points for count

        total = len_score + q_score + hv_score + count_score
        return min(total, 10.0)

    def get_user_stats(self, user_id: str) -> dict:
        """Get engagement stats for a user."""
        count = self._msg_count.get(user_id, 0)
        avg_len = self._total_length.get(user_id, 0) / max(count, 1)
        return {
            "msg_count": count,
            "avg_length": avg_len,
            "question_count": self._question_count.get(user_id, 0),
            "high_value_count": self._high_value_count.get(user_id, 0),
            "engagement_score": self.get_engagement_score(user_id),
            "last_activity": self._last_activity.get(user_id, 0),
        }


# Singleton instance
_engagement_tracker = UserEngagementTracker()


def get_engagement_tracker() -> UserEngagementTracker:
    """Get the global engagement tracker."""
    return _engagement_tracker
