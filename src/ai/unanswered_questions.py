"""
Unanswered question detection module for the Eudora persona.

Features:
1. Scan recent messages for questions that the bot hasn't answered
2. When the bot is mentioned, include unanswered questions in context
3. Algorithmically detect question patterns
4. Track which messages the bot has already replied to

Algorithmic approach:
- Scan last 10 messages in channel for questions
- Check if the bot replied after each question
- If unanswered questions exist, prepend them to the transcript
- Mark questions as "answered" when the bot sends a reply
"""
import re
import time
from typing import List, Optional, Tuple, Dict
from collections import deque
from loguru import logger


# Question detection patterns
QUESTION_PATTERNS = [
    r'\?\s*$',  # Ends with question mark
    r'\b(what|why|how|when|where|who|which|whose)\b.*\?',  # Question word + ?
    r'\b(do you|are you|can you|could you|will you|would you)\b',  # Direct questions to bot
    r'\b(how old|how tall|where.*from|what.*name|what.*favorite|what.*favourite)\b',  # Personal questions
    r'\b(do you like|do you watch|do you play|do you listen)\b',  # Preference questions
    r'\b(are you|is it|is that)\b.*\?',  # Yes/no questions
]

_COMPILED_Q_PATTERNS = [re.compile(p, re.IGNORECASE) for p in QUESTION_PATTERNS]

# Messages that are NOT real questions (false positives)
NON_QUESTION_PATTERNS = [
    r'^[\?\s]+$',  # Just question marks
    r'(what|how|why)\s+(the|a|an)\s+(fuck|hell|heck|f)',  # Rhetorical
    r'\bwho cares\b',
    r'\bwho asked\b',
    r'\bidgaf\b',
]

_COMPILED_NON_Q = [re.compile(p, re.IGNORECASE) for p in NON_QUESTION_PATTERNS]


def is_question(text: str) -> bool:
    """Algorithmically detect if a message contains a real question."""
    if not text or not text.strip():
        return False

    # Check non-question patterns first (exclude rhetorical)
    for rx in _COMPILED_NON_Q:
        if rx.search(text):
            return False

    # Check question patterns
    for rx in _COMPILED_Q_PATTERNS:
        if rx.search(text):
            return True

    return False


def is_directed_at_bot(text: str, bot_name: str = "eudora") -> bool:
    """Check if a question seems directed at the bot (by name or context)."""
    text_lower = text.lower()
    return (
        bot_name in text_lower
        or "you" in text_lower
        or "u " in text_lower
        or "ur " in text_lower
        or "your " in text_lower
        or "?" in text_lower
    )


class UnansweredQuestionTracker:
    """
    Tracks unanswered questions per channel.

    Algorithmic behavior:
    - Track questions in recent messages
    - Mark questions as answered when the bot replies
    - When the bot is mentioned, return unanswered questions for context
    - Questions expire after 5 minutes (don't answer old questions)
    """

    def __init__(self):
        # channel_id -> list of (message_text, author_name, timestamp, message_id)
        self._unanswered: Dict[str, deque] = {}
        # Question expiry: 5 minutes
        self._expiry_s = 300

    def record_question(self, channel_id: str, text: str, author_name: str, message_id: int):
        """Record a potential unanswered question."""
        if not is_question(text):
            return

        if channel_id not in self._unanswered:
            self._unanswered[channel_id] = deque(maxlen=10)

        # Don't add duplicates
        for q_text, _, _, q_id in self._unanswered[channel_id]:
            if q_id == message_id or q_text == text:
                return

        self._unanswered[channel_id].append((text, author_name, time.time(), message_id))
        logger.debug(f"Unanswered question tracked in #{channel_id}: '{text[:50]}' by {author_name}")

    def mark_answered(self, channel_id: str):
        """Mark all unanswered questions in a channel as answered (bot replied)."""
        if channel_id in self._unanswered:
            count = len(self._unanswered[channel_id])
            self._unanswered[channel_id].clear()
            if count > 0:
                logger.debug(f"Marked {count} questions as answered in #{channel_id}")

    def get_unanswered_questions(self, channel_id: str, max_count: int = 3) -> List[Tuple[str, str]]:
        """
        Get unanswered questions for a channel.

        Returns list of (question_text, author_name) tuples.
        Filters out expired questions (older than 5 minutes).
        """
        if channel_id not in self._unanswered:
            return []

        now = time.time()
        valid = []
        expired = []

        for text, author, ts, msg_id in self._unanswered[channel_id]:
            age = now - ts
            if age > self._expiry_s:
                expired.append((text, author, ts, msg_id))
            else:
                valid.append((text, author))

        # Remove expired
        if expired:
            self._unanswered[channel_id] = deque(
                [(t, a, ts, mid) for t, a, ts, mid in self._unanswered[channel_id]
                 if (t, a, ts, mid) not in expired],
                maxlen=10
            )

        return valid[:max_count]

    def get_context_note(self, channel_id: str) -> str:
        """
        Get a context note for the AI transcript about unanswered questions.

        Returns a string like:
        "[PREVIOUS UNANSWERED QUESTIONS — please answer these:]
        - Mr. Alien asked: 'How old are you?'
        - User2 asked: 'What's your favorite color?'"

        Or empty string if no unanswered questions.
        """
        questions = self.get_unanswered_questions(channel_id)
        if not questions:
            return ""

        lines = ["[PREVIOUS UNANSWERED QUESTIONS — you missed these, please answer them now:]"]
        for text, author in questions:
            lines.append(f"- {author} asked: '{text}'")

        return "\n".join(lines) + "\n"

    def clear_channel(self, channel_id: str):
        """Clear all unanswered questions for a channel."""
        if channel_id in self._unanswered:
            self._unanswered[channel_id].clear()


# Singleton instance
_tracker = UnansweredQuestionTracker()


def get_unanswered_tracker() -> UnansweredQuestionTracker:
    """Get the global unanswered question tracker."""
    return _tracker
