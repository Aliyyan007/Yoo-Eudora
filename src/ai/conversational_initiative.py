"""
Conversational initiative module for the Eudora persona.

Features:
1. Bot asks questions to keep conversations going
2. Poses follow-up questions after answering
3. Asks about user preferences
4. Tracks user response time to gauge interest
5. Adjusts friendliness based on engagement patterns
6. Sends independent messages (not just reactive replies)

Algorithmic approach:
- After answering a question, 40% chance to ask a follow-up
- Track response time per user (fast = interested, slow = less interested)
- Track message length per user (longer = more engaged)
- Adjust tone: more friendly with engaged users, more distant with disengaged
- Generate conversation starters based on user interests
- Independent messages: occasionally start conversations without being prompted
"""
import re
import time
import random
from typing import Optional, List, Dict, Tuple
from collections import defaultdict
from loguru import logger


# ── Follow-up question templates ─────────────────────────────────────────────

# Generic follow-up questions (after answering)
FOLLOW_UP_QUESTIONS = [
    "what about you?",
    "hbu?",
    "you?",
    "what do you think?",
    "do you agree?",
    "any thoughts?",
    "what's urs like?",
    "wbu?",
]

# Preference questions (to learn about users)
PREFERENCE_QUESTIONS = [
    "what kind of music are u into?",
    "do u watch anime?",
    "what's ur fav food?",
    "u play any games?",
    "what do u do for fun?",
    "are u a student or working?",
    "what's ur fav colour?",
    "do u like art?",
    "what kind of movies do u watch?",
    "u more of an introvert or extrovert?",
    "do u like reading?",
    "what's ur fav season?",
    "are u a morning person or night owl?",
    "do u like coffee or tea?",
    "what's ur fav subject?",
]

# Conversation starters (independent messages)
CONVERSATION_STARTERS = [
    "anyone else just chilling rn",
    "what's everyone up to",
    "i'm so bored fr someone talk to me",
    "yo what's the move today",
    "anyone wanna share what they're listening to",
    "what's everyone's fav song rn",
    "yo recommend me smth to watch",
    "anyone else procrastinating rn",
    "what's the best thing that happened to u today",
    "yo what's everyone's take on [topic]",
]

# Interest-based conversation starters
INTEREST_STARTERS = {
    "gaming": ["anyone playing anything good rn", "what games is everyone into"],
    "music": ["what's everyone listening to", "drop ur fav song rn"],
    "art": ["anyone else into art", "show me what ur working on"],
    "anime": ["what anime is everyone watching", "best anime this season fr"],
    "food": ["what's everyone having for dinner", "fav food go"],
    "study": ["anyone else studying rn", "how's the study grind going"],
}


# ── User engagement analysis ─────────────────────────────────────────────────

class UserEngagementAnalyzer:
    """
    Analyzes user engagement patterns to adjust bot behavior.

    Algorithmic behavior:
    - Track response time per user (how fast they reply to the bot)
    - Track message length per user
    - Track conversation frequency per user
    - Compute engagement level: "high", "medium", "low", "minimal"
    - Adjust friendliness based on engagement level
    """

    def __init__(self):
        # user_id -> list of response times (seconds)
        self._response_times: Dict[str, list] = defaultdict(list)
        # user_id -> list of message lengths
        self._msg_lengths: Dict[str, list] = defaultdict(list)
        # user_id -> last bot message timestamp (to measure response time)
        self._last_bot_msg_to_user: Dict[str, float] = {}
        # user_id -> total message count
        self._msg_count: Dict[str, int] = defaultdict(int)
        # user_id -> last interaction timestamp
        self._last_interaction: Dict[str, float] = {}

    def record_bot_message_to_user(self, user_id: str):
        """Record that the bot sent a message to a user (for response timing)."""
        self._last_bot_msg_to_user[user_id] = time.time()

    def record_user_message(self, user_id: str, text: str):
        """Record a message from a user."""
        self._msg_count[user_id] += 1
        self._last_interaction[user_id] = time.time()
        self._msg_lengths[user_id].append(len(text))
        # Keep only last 20 messages
        if len(self._msg_lengths[user_id]) > 20:
            self._msg_lengths[user_id] = self._msg_lengths[user_id][-20:]

        # Calculate response time if we have a bot message timestamp
        last_bot = self._last_bot_msg_to_user.get(user_id)
        if last_bot:
            response_time = time.time() - last_bot
            self._response_times[user_id].append(response_time)
            if len(self._response_times[user_id]) > 20:
                self._response_times[user_id] = self._response_times[user_id][-20:]
            # Clear the bot message timestamp
            self._last_bot_msg_to_user[user_id] = None

    def get_avg_response_time(self, user_id: str) -> float:
        """Get average response time for a user (seconds)."""
        times = self._response_times.get(user_id, [])
        if not times:
            return 0.0
        return sum(times) / len(times)

    def get_avg_msg_length(self, user_id: str) -> float:
        """Get average message length for a user."""
        lengths = self._msg_lengths.get(user_id, [])
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def get_engagement_level(self, user_id: str) -> str:
        """
        Algorithmically determine engagement level.

        Returns: "high", "medium", "low", "minimal"

        Factors:
        - Average response time (faster = more engaged)
        - Average message length (longer = more engaged)
        - Message count (more = more engaged)
        - Recency of interaction
        """
        count = self._msg_count.get(user_id, 0)
        if count == 0:
            return "minimal"

        avg_response = self.get_avg_response_time(user_id)
        avg_length = self.get_avg_msg_length(user_id)

        # Score components
        # Response time score (0-3): <30s=3, <60s=2, <120s=1, >120s=0
        # Note: 0.0 means no data, not "very fast"
        if avg_response == 0.0:
            rt_score = 0  # No data
        elif avg_response < 30:
            rt_score = 3
        elif avg_response < 60:
            rt_score = 2
        elif avg_response < 120:
            rt_score = 1
        else:
            rt_score = 0

        # Message length score (0-3): >50=3, >20=2, >10=1, <10=0
        if avg_length > 50:
            ml_score = 3
        elif avg_length > 20:
            ml_score = 2
        elif avg_length > 10:
            ml_score = 1
        else:
            ml_score = 0

        # Count score (0-2): >10=2, >3=1, <3=0
        if count > 10:
            count_score = 2
        elif count > 3:
            count_score = 1
        else:
            count_score = 0

        total = rt_score + ml_score + count_score

        if total >= 6:
            return "high"
        elif total >= 3:
            return "medium"
        elif total >= 1:
            return "low"
        else:
            return "minimal"

    def get_friendliness_modifier(self, user_id: str) -> str:
        """
        Get a friendliness modifier for the AI prompt based on engagement.

        Returns a context note for the AI transcript.
        """
        level = self.get_engagement_level(user_id)

        modifiers = {
            "high": "[USER ENGAGEMENT: HIGH — this user is very engaged (fast responses, long messages). Be warm, friendly, and ask follow-up questions to keep the conversation going.]",
            "medium": "[USER ENGAGEMENT: MEDIUM — this user is moderately engaged. Be friendly but not overly familiar.]",
            "low": "[USER ENGAGEMENT: LOW — this user responds slowly or with short messages. Be casual, don't force conversation, keep replies short.]",
            "minimal": "[USER ENGAGEMENT: MINIMAL — this user rarely interacts. Be polite but distant, don't over-invest in the conversation.]",
        }

        return modifiers.get(level, modifiers["medium"])

    def get_user_stats(self, user_id: str) -> dict:
        """Get engagement stats for a user."""
        return {
            "msg_count": self._msg_count.get(user_id, 0),
            "avg_response_time": self.get_avg_response_time(user_id),
            "avg_msg_length": self.get_avg_msg_length(user_id),
            "engagement_level": self.get_engagement_level(user_id),
            "last_interaction": self._last_interaction.get(user_id, 0),
        }


# Singleton instance
_engagement_analyzer = UserEngagementAnalyzer()


def get_engagement_analyzer() -> UserEngagementAnalyzer:
    """Get the global engagement analyzer."""
    return _engagement_analyzer


# ── Follow-up question generation ────────────────────────────────────────────

def should_ask_followup(
    user_id: str,
    is_answering_question: bool = False,
) -> bool:
    """
    Algorithmically decide whether to ask a follow-up question.

    Factors:
    - Base chance: 40%
    - Higher if answering a question (50%)
    - Higher for engaged users (50%)
    - Lower for disengaged users (20%)
    """
    analyzer = get_engagement_analyzer()
    level = analyzer.get_engagement_level(user_id)

    chance = 0.40  # Base

    if is_answering_question:
        chance += 0.10

    if level == "high":
        chance += 0.10
    elif level == "low":
        chance -= 0.20
    elif level == "minimal":
        chance -= 0.30

    chance = max(0.10, min(chance, 0.60))

    return random.random() < chance


def get_followup_question(user_id: str, context: str = "") -> str:
    """
    Algorithmically generate a follow-up question.

    Returns a question string.
    """
    # If the context mentions preferences, ask about preferences
    if context:
        context_lower = context.lower()
        if any(w in context_lower for w in ["like", "enjoy", "love", "fav", "favourite", "favorite"]):
            return random.choice(PREFERENCE_QUESTIONS)

    # Default: generic follow-up
    return random.choice(FOLLOW_UP_QUESTIONS)


def get_conversation_starter(channel_nature: str = "general") -> str:
    """
    Algorithmically generate a conversation starter.

    Returns a message string to start a conversation.
    """
    if channel_nature in INTEREST_STARTERS and random.random() < 0.40:
        return random.choice(INTEREST_STARTERS[channel_nature])

    return random.choice(CONVERSATION_STARTERS)


def get_preference_question(user_id: str) -> Optional[str]:
    """
    Get a preference question for a user.

    Algorithmic: only ask if we don't already know their preferences
    (would need to check memory, but for now just return a random one).
    """
    return random.choice(PREFERENCE_QUESTIONS)


# ── Context for AI ───────────────────────────────────────────────────────────

def get_conversational_initiative_context(
    user_id: str,
    is_answering_question: bool = False,
    reply_text: str = "",
) -> str:
    """
    Get context for the AI to encourage conversational initiative.

    Returns a context note telling the AI to:
    - Ask follow-up questions
    - Show interest in the user
    - Adjust tone based on engagement
    """
    analyzer = get_engagement_analyzer()
    parts = []

    # Engagement-based friendliness
    friendliness = analyzer.get_friendliness_modifier(user_id)
    parts.append(friendliness)

    # Follow-up encouragement
    if should_ask_followup(user_id, is_answering_question):
        parts.append(
            "[CONVERSATIONAL INITIATIVE: Ask a follow-up question after your reply. "
            "Show genuine interest in the user. Keep the conversation going naturally.]"
        )

    return "\n".join(parts) + "\n"
