"""
Owner respect system for the Eudora persona.

Features:
1. Identify server owners and bot creator
2. Store their instructions explicitly in the database
3. Follow their commands without question
4. When non-owners try to command the bot, respond algorithmically
   (question them, be dismissive, or comply based on context)
5. Track owner-specific preferences

Algorithmic approach:
- Owner IDs are hardcoded (can be extended via config)
- Owner commands are always followed
- Non-owner commands: 30% comply, 40% question, 30% dismiss
- Owner instructions stored in memory.json with high priority
- Bot is respectful and attentive to owners
- Bot can push back on non-owner commands
"""
import re
import time
import random
from typing import Optional, List, Dict, Tuple
from loguru import logger


# ── Owner IDs ────────────────────────────────────────────────────────────────

# Server owners and bot creator — these users have absolute authority.
# Hardcoded defaults + env extension (OWNER_IDS="id1,id2" or OWNER_ID="id").
OWNER_IDS = {
    "1391457611618062376",  # Server owner 1
    "1345822697078395032",  # Server owner 2 / bot creator
}


def _load_env_owners() -> set:
    import os
    ids = set()
    for var in ("OWNER_IDS", "OWNER_ID"):
        raw = os.getenv(var, "")
        ids.update(p.strip() for p in raw.split(",") if p.strip().isdigit())
    return ids


OWNER_IDS |= _load_env_owners()


def is_owner(user_id: str) -> bool:
    """Check if a user is an owner (server owner or bot creator)."""
    return str(user_id) in OWNER_IDS


# ── Command detection ────────────────────────────────────────────────────────

# Commands that owners can give (always followed)
OWNER_COMMAND_PATTERNS = [
    r'\b(bump\s+(?:the\s+)?(?:server|servers)|bump\s+all)\b',
    r'\b(remember\s+(?:that|this))\b',
    r'\b(forget\s+(?:that|this|about))\b',
    r'\b(go\s+(?:to\s+sleep|offline|away|afk))\b',
    r'\b(come\s+back|wake\s+up|stop\s+being\s+afk)\b',
    r'\b(change\s+(?:your|ur)\s+(?:name|bio|status|nickname))\b',
    r'\b(be\s+(?:quiet|silent|silent|loud))\b',
    r'\b(stop\s+(?:talking|replying|responding))\b',
    r'\b(start\s+(?:talking|replying|responding))\b',
    r'\b(join\s+(?:the\s+)?vc|leave\s+(?:the\s+)?vc)\b',
    r'\b(say\s+.+)\b',  # "say hello" — repeat after owner
    r'\b(go\s+to\s+(?:#|channel)\s*\w+)\b',
    r'\b(set\s+(?:status|bio|mood)\s+to\s+.+)\b',
    r'\b(be\s+more\s+(?:active|quiet|chatty|friendly))\b',
    r'\b(don\'?t\s+(?:reply|respond|talk)\s+(?:to|with)\s+.+)\b',
    r'\b(reply\s+(?:to|with)\s+.+)\b',
]

_COMPILED_OWNER_CMDS = [re.compile(p, re.IGNORECASE) for p in OWNER_COMMAND_PATTERNS]

# Commands that non-owners might try (bot can push back)
NON_OWNER_COMMAND_PATTERNS = [
    r'\b(bump\s+(?:the\s+)?(?:server|servers)|bump\s+all)\b',
    r'\b(go\s+(?:to\s+sleep|offline|away|afk))\b',
    r'\b(stop\s+(?:talking|replying|responding))\b',
    r'\b(change\s+(?:your|ur)\s+(?:name|bio|status))\b',
    r'\b(be\s+quiet|shut\s+up|stfu)\b',
    r'\b(say\s+.+)\b',
    r'\b(do\s+(?:this|that|it))\b',
    r'\b(tell\s+(?:me|him|her|them)\s+.+)\b',
]

_COMPILED_NON_OWNER_CMDS = [re.compile(p, re.IGNORECASE) for p in NON_OWNER_COMMAND_PATTERNS]


def detect_command_attempt(text: str) -> bool:
    """Detect if a message is attempting to give a command."""
    # Questions aren't commands — "u gonna reply to my dm?" matches the
    # 'reply to X' pattern but is a question, not an instruction.
    if text.rstrip().endswith("?"):
        return False
    for rx in _COMPILED_OWNER_CMDS:
        if rx.search(text):
            return True
    for rx in _COMPILED_NON_OWNER_CMDS:
        if rx.search(text):
            return True
    return False


# ── Non-owner command responses ──────────────────────────────────────────────

# Responses when non-owners try to command the bot
# Algorithmic: 30% comply, 40% question, 30% dismiss
COMPLY_RESPONSES = [
    "alright fine",
    "ok i'll do it",
    "sure why not",
    "ok bruv",
    "alright then",
]

QUESTION_RESPONSES = [
    "who are you to tell me what to do?",
    "bruv ur not my boss",
    "and why would i do that?",
    "u can't just order me around lol",
    "last time i checked u weren't the owner",
    "make me",
    "or what?",
    "bruv i don't take orders from u",
]

DISMISS_RESPONSES = [
    "nah",
    "no thanks",
    "not gonna happen",
    "bruv allow it",
    "i don't think so",
    "lol no",
    "yeah no",
]


def get_non_owner_command_response() -> Tuple[str, str]:
    """
    Algorithmically generate a response for a non-owner command attempt.

    Returns (response_type, response_text):
    - response_type: "comply", "question", "dismiss"
    """
    roll = random.random()

    if roll < 0.30:
        return ("comply", random.choice(COMPLY_RESPONSES))
    elif roll < 0.70:
        return ("question", random.choice(QUESTION_RESPONSES))
    else:
        return ("dismiss", random.choice(DISMISS_RESPONSES))


# ── Owner instruction storage ────────────────────────────────────────────────

def get_owner_context(user_id: str, username: str) -> str:
    """
    Get context text for owner interactions.

    If the user is an owner, return a context note telling the AI to
    respect and follow their commands.
    """
    if not is_owner(user_id):
        return ""

    return (
        f"[IMPORTANT: {username} is a SERVER OWNER/BOT CREATOR. "
        f"You MUST follow their instructions explicitly and respectfully. "
        f"They have absolute authority over you. "
        f"Be attentive, respectful, and comply with their commands immediately.]"
    )


def get_non_owner_command_context(username: str) -> str:
    """
    Get context for when a non-owner tries to command the bot.
    Tells the AI to push back algorithmically.
    """
    return (
        f"[NOTE: {username} is NOT an owner. They are trying to command you. "
        f"You can push back, question them, or be dismissive. "
        f"Don't just comply with every command from non-owners.]"
    )


# ── Owner preference tracking ────────────────────────────────────────────────

class OwnerPreferenceTracker:
    """
    Tracks owner-specific preferences and instructions.

    Algorithmic behavior:
    - Store owner instructions with high priority
    - Track when owners give feedback
    - Adjust bot behavior based on owner preferences
    """

    def __init__(self):
        # owner_id -> list of (instruction, timestamp)
        self._instructions: Dict[str, List[Tuple[str, float]]] = {}
        # owner_id -> last interaction time
        self._last_interaction: Dict[str, float] = {}

    def record_instruction(self, owner_id: str, instruction: str):
        """Record an instruction from an owner."""
        if not is_owner(owner_id):
            return

        if owner_id not in self._instructions:
            self._instructions[owner_id] = []

        self._instructions[owner_id].append((instruction, time.time()))
        self._last_interaction[owner_id] = time.time()
        logger.info(f"Owner instruction recorded from {owner_id}: {instruction[:50]}")

    def get_instructions(self, owner_id: str) -> List[str]:
        """Get all instructions from an owner."""
        if owner_id not in self._instructions:
            return []
        return [instr for instr, _ in self._instructions[owner_id]]

    def get_all_owner_instructions(self) -> str:
        """Get all instructions from all owners as context text."""
        if not self._instructions:
            return ""

        lines = ["[OWNER INSTRUCTIONS — follow these explicitly:]"]
        for owner_id, instructions in self._instructions.items():
            for instr, ts in instructions:
                lines.append(f"- {instr}")

        return "\n".join(lines) + "\n"

    def record_interaction(self, owner_id: str):
        """Record that an owner interacted with the bot."""
        if is_owner(owner_id):
            self._last_interaction[owner_id] = time.time()

    def get_last_interaction(self, owner_id: str) -> float:
        """Get last interaction time for an owner."""
        return self._last_interaction.get(owner_id, 0)


# Singleton instance
_owner_tracker = OwnerPreferenceTracker()


def get_owner_tracker() -> OwnerPreferenceTracker:
    """Get the global owner preference tracker."""
    return _owner_tracker
