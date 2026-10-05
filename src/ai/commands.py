"""
Command detection for natural language instructions.

Detects when a user is giving the bot a command in natural language
(e.g. "bump the server", "stop pinging", "remember that my name is X")
and extracts the intent + parameters.

This allows the bot to ACT on requests, not just talk about them.
"""
import re
from typing import Tuple, Optional, List, Dict
from loguru import logger


# Invalid targets — common words that are NOT usernames
INVALID_TARGETS = {
    "you", "me", "him", "her", "them", "us", "everyone", "here", "all",
    "someone", "anybody", "nobody", "this", "that", "there", "where",
    "what", "who", "why", "how", "when", "yes", "no", "maybe", "ok",
    "okay", "sure", "yeah", "nah", "nope", "please", "thanks", "thank",
    "the", "a", "an", "my", "your", "his", "her", "their", "our",
    "bot", "myself", "yourself", "himself", "herself", "itself",
    "today", "tomorrow", "yesterday", "now", "later", "soon",
    "guys", "people", "folks", "everyone", "everybody",
}


# Command patterns: (regex, command_type, description)
# NOTE: Order matters — stop/negative commands are checked BEFORE positive ones
COMMAND_PATTERNS = [
    # Stop bumping (must be BEFORE bump commands to avoid false trigger)
    (r'\b(stop\s+bump(?:ing)?|don\'?t\s+bump|no\s+more\s+bump(?:ing)?|quit\s+bump(?:ing)?)\b',
     "stop_bumping", "User wants to stop bumping"),

    # Bump commands (positive only — "stop bumping" is handled above)
    (r'\b(bump\s+(?:the\s+)?server|bump\s+servers?|re\s* bump|rebump|try\s+bump(?:ing)?)\b',
     "bump_server", "User wants to bump the server(s)"),
    (r'\b(bump\s+with\s+(\w+))\b', "bump_with", "User wants to bump with a specific bot"),

    # Ping control
    (r'\b(stop\s+pinging|don\'?t\s+ping|no\s+more\s+pinging|stop\s+the\s+ping)\b',
     "stop_pinging", "User wants to stop chat revive pings"),
    (r'\b(ping\s+after\s+(\d+)\s*h|ping\s+in\s+(\d+)\s*h|try\s+pinging\s+(?:again\s+)?after\s+(\d+)h?)\b',
     "ping_after_hours", "User wants pings delayed by N hours"),

    # VC (Voice Channel) commands
    # Matches: "join vc", "join the vc", "join my vc", "join this vc",
    # "join the current vc", "hop in vc", "get in voice", "come to call",
    # "join THE DEV vc" (channel name), "vc me", "join voice channel"
    (r'\b(join|hop\s+in|get\s+in|come\s+to|hop\s+on)\s+(?:(?:the|my|this|current|active|a)\s+)?(?:vc|voice|call|voice\s+channel)\b',
     "join_vc", "User wants the bot to join a voice channel"),
    (r'\b(join|hop\s+in|get\s+in)\s+\w+\s+(?:vc|voice|call)\b',
     "join_vc", "User wants the bot to join a named voice channel"),
    (r'\b(vc\s+me|vc\s+now|join\s+voice)\b',
     "join_vc", "User wants the bot to join voice (casual)"),
    (r'\b(leave|exit|disconnect|drop|dip)\s+(?:the\s+|this\s+|current\s+)?(?:vc|voice|call|voice\s+channel)\b',
     "leave_vc", "User wants the bot to leave the voice channel"),
    (r'\b(leave|exit|disconnect)\s+vc\b',
     "leave_vc", "User wants the bot to leave the voice channel (short form)"),

    # Mention/ping commands — mention any user by name or ID
    # Discord mention format: <@123> or <@!123>
    (r'\b(mention|ping)\s+<@!?(\d+)>',
     "mention_user_id", "User wants to mention someone by ID"),
    # Explicit @ prefix: "mention @name"
    (r'\b(mention|ping)\s+@(\w+)',
     "mention_user", "User wants to mention someone with @ prefix"),
    # "mention/ping the user called/named X"
    (r'\b(mention|ping)\s+(?:the\s+)?user\s+(?:called\s+|named\s+)(\w+)',
     "mention_user", "User wants to mention someone by name"),
    # "mention/ping user X"
    (r'\b(mention|ping)\s+user\s+(\w+)',
     "mention_user", "User wants to mention a user"),
    # "mention/ping X" — bare name (min 3 chars to avoid false positives)
    # Supports multi-word names like "Mr. Alien", "John Doe"
    (r'\b(mention|ping)\s+([A-Za-z][\w\s\.]{2,30}?)(?:\s*$|\s*[\n\r])',
     "mention_user", "User wants to mention someone by name"),
    # Start chatting with a user
    (r'\b(chat\s+with|talk\s+to|start\s+chatting\s+with)\s+<?@?(\w+)>?',
     "start_chat", "User wants the bot to start chatting with someone"),

    # Memory commands — "remember that..."
    (r'\b(remember(?:\s+that)?)\s+(.+)', "remember", "User wants the bot to remember something"),
    (r'\b(my\s+name\s+is\s+(\w+))\b', "set_name", "User is telling their name"),
    (r'\b(call\s+me\s+(\w+))\b', "set_name", "User wants to be called by a new name"),
    # Location BEFORE hobby — "im from india and i love it" is a location,
    # not the hobby "it" (search order = first match wins).
    (r"\b(i\s+live\s+in\s+((?:the\s+)?[a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,3}?))(?=\s+(?:and|but|so|i|im|i'm|tho|though|btw|lol|lmao|haha|rn|now|bro|bruh|too|which|where|who)\b|\s*[,.!?;:]|\s*$)",
     "set_location", "User is telling their location"),
    (r"\b((?:i'?m|i\s+am)\s+from\s+((?:the\s+)?[a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,3}?))(?=\s+(?:and|but|so|i|im|i'm|tho|though|btw|lol|lmao|haha|rn|now|bro|bruh|too|which|where|who)\b|\s*[,.!?;:]|\s*$)",
     "set_location", "User is telling their location"),
    (r'\b(i\s+(?:like|love|enjoy)\s+(.+?))(?:\.|$)', "set_hobby", "User is telling their hobby"),
    (r'\b(i\s+am\s+(?:a|an)\s+(\w+)(?:\s+person)?)(?:\.|$)', "set_personality", "User is telling their personality"),
    (r'\b(i\s+am\s+(\d+)\s+years?\s+old)\b', "set_age", "User is telling their age"),

    # Recall commands — "what's my name?"
    (r'\b(what\'?s\s+my\s+name|what\s+is\s+my\s+name|do\s+you\s+know\s+my\s+name|who\s+am\s+i)\b',
     "recall_name", "User is asking what their name is"),
    (r'\b(what\s+do\s+you\s+know\s+about\s+me|what\s+do\s+you\s+remember\s+about\s+me)\b',
     "recall_all", "User is asking what the bot knows about them"),

    # Speed control
    (r'\b(reply\s+faster|respond\s+faster|be\s+faster|type\s+faster|quicker)\b',
     "speed_up", "User wants faster responses"),
    (r'\b(reply\s+slower|slow\s+down|take\s+your\s+time|no\s+rush)\b',
     "slow_down", "User wants slower responses"),
]


def detect_command(text: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Detect if a message contains a command.
    Returns (command_type, parameters_dict) or (None, None).
    """
    if not text:
        return None, None

    text_lower = text.lower().strip()

    for pattern, cmd_type, description in COMMAND_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            params = {}
            groups = match.groups()

            if cmd_type == "bump_with":
                params["bot_name"] = groups[1] if len(groups) > 1 else ""
            elif cmd_type == "ping_after_hours":
                # Extract the hours number from whichever group matched
                hours = None
                for g in groups[1:]:
                    if g:
                        try:
                            hours = int(g)
                        except ValueError:
                            pass
                params["hours"] = hours or 16
            elif cmd_type == "remember":
                params["fact"] = groups[1] if len(groups) > 1 else ""
            elif cmd_type == "set_name":
                params["name"] = groups[1] if len(groups) > 1 else ""
            elif cmd_type == "set_hobby":
                params["hobby"] = groups[1] if len(groups) > 1 else ""
            elif cmd_type == "set_personality":
                params["personality"] = groups[1] if len(groups) > 1 else ""
            elif cmd_type == "set_location":
                location = (groups[1] if len(groups) > 1 else "") or ""
                location = re.sub(r"^the\s+", "", location.strip())
                if not location or location in {
                        "here", "there", "nowhere", "earth", "discord",
                        "the internet", "internet"}:
                    return None, None
                params["location"] = location
            elif cmd_type in ("mention_user", "mention_user_id", "start_chat"):
                # Extract the target name/ID from the regex groups
                # The regex captures: group(1) = verb (mention/ping/chat with),
                # group(2) = target name or ID
                target = ""
                if len(groups) > 2 and groups[2]:
                    target = groups[2]
                elif len(groups) > 1 and groups[1]:
                    target = groups[1]
                # Clean up the target (remove @ prefix, <@!...> format, trailing whitespace)
                if target:
                    target = target.strip().lstrip('@').strip()
                    # If it's a Discord mention format <@123>, extract the ID
                    mention_match = re.match(r'<@!?(\d+)>', target)
                    if mention_match:
                        target = mention_match.group(1)

                # Special case: "talk to you" / "chat with you" means the user
                # wants to talk to the BOT itself, not a member called "you".
                # Treat as a normal message so the bot replies conversationally.
                if cmd_type == "start_chat" and target.lower() == "you" and ("talk to you" in text_lower or "chat with you" in text_lower):
                    return None, {}

                # Filter out common words that aren't usernames
                if target.lower() in INVALID_TARGETS:
                    return None, {}  # Not a command — treat as normal message

                params["target"] = target
                params["name"] = target  # Also store as "name" for compatibility

            logger.info(f"Command detected: {cmd_type} — {description} (params: {params})")
            return cmd_type, params

    return None, None


def is_action_command(cmd_type: str) -> bool:
    """Check if a command requires immediate action (not just memory storage)."""
    return cmd_type in ("bump_server", "bump_with", "stop_pinging", "ping_after_hours")


def is_memory_command(cmd_type: str) -> bool:
    """Check if a command is about storing/recalling memory."""
    return cmd_type in (
        "remember", "set_name", "set_hobby", "set_personality",
        "set_age", "set_location", "recall_name", "recall_all"
    )


def is_preference_command(cmd_type: str) -> bool:
    """Check if a command changes bot behavior/preferences."""
    return cmd_type in ("speed_up", "slow_down", "stop_pinging", "ping_after_hours")
