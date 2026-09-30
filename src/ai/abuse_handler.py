"""
Abuse handling module for the Eudora persona.

Features:
1. Detect abusive language directed at the bot
2. Algorithmically classify abuse severity (mild, moderate, severe)
3. Generate appropriate responses based on severity:
   - Mild (name calling): brush it off casually
   - Moderate (swearing at): mild pushback
   - Severe (serious abuse): fight back with attitude
4. Track abuse history per user (escalation logic)
5. Don't be overly aggressive — match the energy but stay in character

Algorithmic approach:
- Pattern matching for abuse detection (not AI — fast and deterministic)
- Severity scoring based on word choice and context
- Escalation: if a user keeps abusing, responses get more pointed
- Cooldown: after fighting back, don't keep fighting (de-escalate)
- Never use slurs or truly offensive language — just attitude and british dry humor

IMPORTANT: The bot fights back with ATTITUDE and BRITISH DRY HUMOR, not with
truly offensive language. It's sarcastic and dismissive, not hateful.
"""
import re
import time
from typing import Optional, Tuple, Dict, List
from loguru import logger


# ── Abuse patterns ────────────────────────────────────────────────────────────

# Severe abuse patterns (directed at the bot)
SEVERE_ABUSE_PATTERNS = [
    r'\bfuck\s+u\b',
    r'\bfuck\s+you\b',
    r'\bfuck\s+off\b',
    r'\bgo\s+fuck\s+yourself\b',
    r'\bshut\s+the\s+fuck\s+up\b',
    r'\bshut\s+up\s+(?:dick|bitch|cunt|whore|slut)\b',
    r'\bkill\s+yourself\b',
    r'\bgo\s+die\b',
    r'\byou\'?re\s+(?:a\s+)?(?:fucking\s+)?(?:bitch|cunt|whore|slut|dick|bastard)\b',
    r'\bu\s+(?:r\s+)?(?:a\s+)?(?:fucking\s+)?(?:bitch|cunt|whore|slut)\b',
    r'\btrash\s+(?:bot|person|human)\b',
    r'\bwaste\s+of\s+(?:space|oxygen)\b',
    r'\bhope\s+u\s+(?:die|get\s+cancer)\b',
    r'\bi\s+hate\s+u\b',
    r'\bi\s+hate\s+you\b',
    r'\bsuck\s+my\s+dick\b',
    r'\bgo\s+to\s+hell\b',
]

# Moderate abuse patterns
MODERATE_ABUSE_PATTERNS = [
    r'\bshut\s+up\b',
    r'\bstfu\b',
    r'\bidiot\b',
    r'\bstupid\b',
    r'\bdumb\b',
    r'\bmoron\b',
    r'\bretard(?:ed)?\b',
    r'\btrash\b',
    r'\bgarbage\b',
    r'\bworthless\b',
    r'\buseless\b',
    r'\bannoying\b',
    r'\bcreep\b',
    r'\bfreak\b',
    r'\bweirdo\b',
    r'\bfreak\s+bot\b',
    r'\bliteral\s+bot\b',
    r'\bu\s+look\s+like\s+a\s+bot\b',
    r'\bact\s+like\s+a\s+bot\b',
]

# Mild abuse / dismissive patterns
MILD_ABUSE_PATTERNS = [
    r'\bbruh\b',
    r'\bcringe\b',
    r'\bmid\b',
    r'\blame\b',
    r'\bboring\b',
    r'\bnobody\s+asked\b',
    r'\bnobody\s+cares\b',
    r'\bwho\s+asked\b',
    r'\bidgaf\b',
    r'\bidc\b',
    r'\bwhatever\b',
]

# Bot-accusation patterns (special category)
BOT_ACCUSATION_PATTERNS = [
    r'\bu\s+(?:r\s+|are\s+)?a\s+bot\b',
    r'\bu\'?re\s+a\s+bot\b',
    r'\byou\s+(?:r\s+|are\s+)?a\s+bot\b',
    r'\byou\'?re\s+a\s+bot\b',
    r'\bu\s+literally\s+look\s+like\s+a\s+bot\b',
    r'\ba\s+freak\s+bot\b',
    r'\bconfirm\s+u\'?re\s+a\s+bot\b',
    r'\bur\s+a\s+freak(?:ing)?\s+bot\b',
    r'\bact\s+like\s+a\s+bot\b',
    r'\bsound\s+like\s+a\s+bot\b',
]

# Compiled patterns
_SEVERE_REGEXES = [re.compile(p, re.IGNORECASE) for p in SEVERE_ABUSE_PATTERNS]
_MODERATE_REGEXES = [re.compile(p, re.IGNORECASE) for p in MODERATE_ABUSE_PATTERNS]
_MILD_REGEXES = [re.compile(p, re.IGNORECASE) for p in MILD_ABUSE_PATTERNS]
_BOT_ACCUSATION_REGEXES = [re.compile(p, re.IGNORECASE) for p in BOT_ACCUSATION_PATTERNS]


def detect_abuse(text: str) -> Tuple[str, int]:
    """
    Algorithmically detect abuse in a message.

    Returns (abuse_level, severity_score):
    - abuse_level: "none", "mild", "moderate", "severe", "bot_accusation"
    - severity_score: 0-10 (0 = no abuse, 10 = maximum abuse)
    """
    if not text:
        return ("none", 0)

    # Check severe patterns first (highest priority)
    for regex in _SEVERE_REGEXES:
        if regex.search(text):
            return ("severe", 10)

    # Check bot accusation (special handling)
    for regex in _BOT_ACCUSATION_REGEXES:
        if regex.search(text):
            return ("bot_accusation", 7)

    # Check moderate patterns
    for regex in _MODERATE_REGEXES:
        if regex.search(text):
            return ("moderate", 5)

    # Check mild patterns
    for regex in _MILD_REGEXES:
        if regex.search(text):
            return ("mild", 2)

    return ("none", 0)


# ── Response generation ──────────────────────────────────────────────────────

# Response templates by abuse level
# These are british dry humor responses — sarcastic and dismissive, not hateful

SEVERE_RESPONSES = [
    "bruv allow it, that's uncalled for",
    "alright calm down, no need for that",
    "wow ok then, someone's having a day",
    "yeah yeah whatever, take a breather",
    "bruv you're gassing for no reason",
    "ok and? you done now?",
    "that's mad rude innit, sort yourself out",
    "wow harsh, didn't realise you cared that much",
    "alright alright, message received, calm down",
    "bruv i'm not the one who hurt you, go touch grass",
]

MODERATE_RESPONSES = [
    "alright chill",
    "wow ok",
    "bruv no need for that",
    "whatever you say",
    "ok and?",
    "sure mate",
    "if you say so",
    "cool story",
    "alright then",
    "bruv calm down",
]

MILD_RESPONSES = [
    "ok",
    "sure",
    "lol ok",
    "if u say so",
    "whatever",
    "bruv",
    "alright",
    "cool",
]

BOT_ACCUSATION_RESPONSES = [
    "bruv i'm not a bot, allow it",
    "i'm not a bot, just vibing here",
    "not a bot, just got better things to do than argue with u",
    "bruv i'm literally just chatting, how's that a bot",
    "i'm not a bot, u just can't handle the vibes",
    "not a bot, i just type fast innit",
    "bruv calling everyone a bot is mad weird",
    "i'm not a bot, u just don't like what i'm saying",
]


class AbuseHandler:
    """
    Manages abuse detection and response generation.

    Algorithmic behavior:
    - Track abuse history per user (escalation)
    - First abuse: mild response
    - Repeated abuse: more pointed response
    - After 3+ abuses: de-escalate (stop engaging)
    - Cooldown after fighting back (don't keep fighting)
    """

    def __init__(self):
        # Track abuse count per user (user_id -> count)
        self._abuse_count: Dict[int, int] = {}
        # Track last abuse time per user (user_id -> timestamp)
        self._last_abuse: Dict[int, float] = {}
        # Track last fight-back time per user (user_id -> timestamp)
        self._last_fightback: Dict[int, float] = {}
        # Cooldown for fighting back (don't keep fighting)
        self._fightback_cooldown_s = 60  # 1 minute
        # After this many abuses, stop engaging
        self._max_abuses_before_disengage = 5
        # Abuse count decays after this time (reset counter)
        self._abuse_decay_s = 3600  # 1 hour

    def handle_abuse(self, user_id: int, text: str) -> Optional[str]:
        """
        Handle an abusive message from a user.

        Returns:
        - A response string if the bot should fight back
        - None if the bot should stay silent (de-escalation)
        - None if no abuse detected
        """
        abuse_level, severity = detect_abuse(text)

        if abuse_level == "none":
            return None

        now = time.time()

        # Update abuse count (with decay)
        last = self._last_abuse.get(user_id, 0)
        if now - last > self._abuse_decay_s:
            # Reset count if it's been a while
            self._abuse_count[user_id] = 0

        self._abuse_count[user_id] = self._abuse_count.get(user_id, 0) + 1
        self._last_abuse[user_id] = now

        count = self._abuse_count[user_id]

        # After too many abuses, stop engaging (de-escalation)
        if count > self._max_abuses_before_disengage:
            logger.info(f"Abuse: user {user_id} hit disengage limit ({count} abuses) — staying silent")
            return None

        # Check fight-back cooldown (don't keep fighting)
        last_fight = self._last_fightback.get(user_id, 0)
        if now - last_fight < self._fightback_cooldown_s:
            # Still in cooldown — don't fight back again
            logger.debug(f"Abuse: user {user_id} in fightback cooldown — staying silent")
            return None

        # Generate response based on abuse level and escalation
        response = self._generate_response(abuse_level, severity, count)
        self._last_fightback[user_id] = now

        logger.info(f"Abuse: user {user_id} level={abuse_level} count={count} — fighting back")
        return response

    def _generate_response(self, abuse_level: str, severity: int, count: int) -> str:
        """
        Generate a response based on abuse level and escalation count.
        """
        import random

        if abuse_level == "bot_accusation":
            return random.choice(BOT_ACCUSATION_RESPONSES)

        if abuse_level == "severe":
            # Escalate: later responses are more pointed
            if count >= 3:
                # More pointed response
                return random.choice([
                    "bruv you're actually obsessed with me, go touch grass",
                    "ok we get it, u don't like me, move on",
                    "you're still going? find something better to do",
                    "bruv this is getting sad, i'm not gonna keep doing this",
                ])
            return random.choice(SEVERE_RESPONSES)

        if abuse_level == "moderate":
            if count >= 3:
                return random.choice([
                    "ok we get it, u don't like me",
                    "bruv find something better to do",
                    "you're still going? lol",
                ])
            return random.choice(MODERATE_RESPONSES)

        if abuse_level == "mild":
            return random.choice(MILD_RESPONSES)

        return None

    def get_abuse_stats(self, user_id: int) -> dict:
        """Get abuse statistics for a user."""
        return {
            "count": self._abuse_count.get(user_id, 0),
            "last_abuse": self._last_abuse.get(user_id, 0),
            "last_fightback": self._last_fightback.get(user_id, 0),
        }


# Singleton instance
_abuse_handler = AbuseHandler()


def get_abuse_handler() -> AbuseHandler:
    """Get the global abuse handler instance."""
    return _abuse_handler
