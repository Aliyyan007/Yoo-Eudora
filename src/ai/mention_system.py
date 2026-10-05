"""
Mention system module for the Eudora persona.

Features:
1. Find and mention any user by display name (fuzzy match)
2. Mention users by ID directly
3. Start chatting with mentioned users (proactive engagement)
4. Parse "mention @username" or "ping @username" commands
5. Algorithmic user search across guild members

Algorithmic approach:
- When a user asks to mention someone, search all guild members
- Match by display name, username, or ID (fuzzy matching)
- Generate a natural mention with the user's Discord ID
- If starting a chat, generate a casual greeting for that user
"""
import re
from typing import Optional, List, Tuple
from loguru import logger
import discord


def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            insert = prev[j + 1] + 1
            delete = curr[j] + 1
            substitute = prev[j] + (ca != cb)
            curr.append(min(insert, delete, substitute))
        prev = curr
    return prev[-1]


def _token_fuzzy_match(query: str, target: str, max_distance: int = 2) -> bool:
    """
    Check if query matches target using token-based fuzzy matching.
    Splits both by whitespace, checks if each query token has a close match
    in the target tokens (within max_distance Levenshtein).

    Example: "Erick frank" matches "Ercik frank" (one token has 1 edit distance)
    """
    query_tokens = query.lower().split()
    target_tokens = target.lower().split()
    if not query_tokens or not target_tokens:
        return False
    matched = 0
    for qt in query_tokens:
        for tt in target_tokens:
            if _levenshtein(qt, tt) <= max_distance:
                matched += 1
                break
    return matched == len(query_tokens)


def find_member_by_name(guild: discord.Guild, name: str) -> Optional[discord.Member]:
    """
    Algorithmically find a guild member by display name, username, nickname, or ID.

    Search priority:
    1. Numeric ID match
    2. Exact display name match (case-insensitive)
    3. Exact username match (case-insensitive)
    4. Exact nickname match (case-insensitive)
    5. Partial display name match (starts with)
    6. Partial display name match (contains)
    7. Partial username match (contains)
    8. Partial nickname match (contains)
    9. Fuzzy match: Levenshtein distance <= 2 for display_name, username, nick
    10. Token-based fuzzy match for display_name, username, nick
    """
    if not name:
        return None

    name_lower = name.lower().strip()

    # Try numeric ID first
    if name_lower.isdigit():
        try:
            member = guild.get_member(int(name_lower))
            if member:
                return member
        except Exception:
            pass

    # 1. Exact display name match
    for member in guild.members:
        if member.bot:
            continue
        if member.display_name.lower() == name_lower:
            return member

    # 2. Exact username match
    for member in guild.members:
        if member.bot:
            continue
        if member.name.lower() == name_lower:
            return member

    # 3. Exact nickname match
    for member in guild.members:
        if member.bot:
            continue
        if member.nick and member.nick.lower() == name_lower:
            return member

    # 4. Partial display name match (starts with)
    for member in guild.members:
        if member.bot:
            continue
        if member.display_name.lower().startswith(name_lower):
            return member

    # 5. Partial display name match (contains)
    for member in guild.members:
        if member.bot:
            continue
        if name_lower in member.display_name.lower():
            return member

    # 6. Partial username match (contains)
    for member in guild.members:
        if member.bot:
            continue
        if name_lower in member.name.lower():
            return member

    # 7. Partial nickname match (contains)
    for member in guild.members:
        if member.bot:
            continue
        if member.nick and name_lower in member.nick.lower():
            return member

    # 8. Fuzzy match: Levenshtein distance <= 2
    for member in guild.members:
        if member.bot:
            continue
        if (_levenshtein(name_lower, member.display_name.lower()) <= 2 or
            _levenshtein(name_lower, member.name.lower()) <= 2 or
            (member.nick and _levenshtein(name_lower, member.nick.lower()) <= 2)):
            return member

    # 9. Token-based fuzzy match
    for member in guild.members:
        if member.bot:
            continue
        if (_token_fuzzy_match(name_lower, member.display_name.lower()) or
            _token_fuzzy_match(name_lower, member.name.lower()) or
            (member.nick and _token_fuzzy_match(name_lower, member.nick.lower()))):
            return member

    return None


def find_members_by_name(guild: discord.Guild, name: str, limit: int = 5) -> List[discord.Member]:
    """
    Find multiple guild members matching a name (for disambiguation).
    Returns up to `limit` matches.

    Matching layers (in priority order):
    1. Exact display name / username / nickname match
    2. Partial display name (starts with / contains)
    3. Partial username / nickname (contains)
    4. Fuzzy match: Levenshtein distance <= 2
    5. Token-based fuzzy match
    """
    if not name:
        return []

    name_lower = name.lower().strip()
    matches = []
    seen_ids = set()

    def _try_add(member: discord.Member) -> bool:
        if member.bot or member.id in seen_ids:
            return False
        matches.append(member)
        seen_ids.add(member.id)
        return len(matches) >= limit

    # Layer 1: Exact matches
    for member in guild.members:
        if (member.display_name.lower() == name_lower or
            member.name.lower() == name_lower or
            (member.nick and member.nick.lower() == name_lower)):
            if _try_add(member):
                return matches

    # Layer 2: Partial display name (starts with / contains)
    for member in guild.members:
        if (member.display_name.lower().startswith(name_lower) or
            name_lower in member.display_name.lower()):
            if _try_add(member):
                return matches

    # Layer 3: Partial username / nickname (contains)
    for member in guild.members:
        if (name_lower in member.name.lower() or
            (member.nick and name_lower in member.nick.lower())):
            if _try_add(member):
                return matches

    # Layer 4: Fuzzy match (Levenshtein distance <= 2)
    for member in guild.members:
        if (_levenshtein(name_lower, member.display_name.lower()) <= 2 or
            _levenshtein(name_lower, member.name.lower()) <= 2 or
            (member.nick and _levenshtein(name_lower, member.nick.lower()) <= 2)):
            if _try_add(member):
                return matches

    # Layer 5: Token-based fuzzy match
    for member in guild.members:
        if (_token_fuzzy_match(name_lower, member.display_name.lower()) or
            _token_fuzzy_match(name_lower, member.name.lower()) or
            (member.nick and _token_fuzzy_match(name_lower, member.nick.lower()))):
            if _try_add(member):
                return matches

    return matches


def format_mention(member: discord.Member) -> str:
    """Format a member as a Discord mention."""
    return member.mention


def parse_mention_command(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse a mention/ping command from text.

    Returns (command_type, target_name) or None.
    command_type is "mention" or "ping".
    target_name is the name/ID of the user to mention.

    Examples:
    - "mention @john" → ("mention", "john")
    - "ping sarah" → ("ping", "sarah")
    - "mention 123456789" → ("mention", "123456789")
    - "ping the user called alex" → ("ping", "alex")

    IMPORTANT: This must NOT match:
    - "I like ping pong" (ping pong, not a command)
    - "mention this to him" (not a mention command)
    - "ping me later" (not a mention command)
    """
    if not text:
        return None

    text_lower = text.lower().strip()

    # Words that should NOT be treated as targets (common words, not usernames)
    invalid_targets = {
        "this", "that", "it", "me", "him", "her", "them", "us", "you",
        "pong", "pong)", "everyone", "here", "all", "later", "now",
        "after", "before", "again", "too", "also", "and", "or", "but",
        "the", "a", "an", "my", "your", "his", "its", "our", "their",
    }

    # Patterns in priority order:
    patterns = [
        # "mention/ping <@123>" or "mention/ping <@!123>" — Discord mention format
        r'\b(mention|ping)\s+<@!?(\d+)>',
        # "mention/ping @name" — explicit @ prefix
        r'\b(mention|ping)\s+@(\w+)',
        # "mention/ping the user called/named X"
        r'\b(mention|ping)\s+(?:the\s+)?user\s+(?:called\s+|named\s+)(\w+)',
        # "mention/ping user X"
        r'\b(mention|ping)\s+user\s+(\w+)',
        # "mention/ping X" — bare name (only if X is not a common word and len >= 3)
        r'\b(mention|ping)\s+(\w{3,})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            cmd = match.group(1)
            target = match.group(2)
            # Clean up the target
            target = target.strip().lstrip('@')
            # Skip invalid targets (common words)
            if target and target.lower() not in invalid_targets:
                return (cmd, target)

    return None


def extract_mentioned_user_ids(text: str) -> List[int]:
    """
    Extract Discord user IDs from <@123> or <@!123> mention patterns in text.
    """
    if not text:
        return []
    pattern = r'<@!?(\d+)>'
    return [int(uid) for uid in re.findall(pattern, text)]


def generate_chat_starter(target_name: str, context: str = "") -> str:
    """
    Algorithmically generate a casual chat starter for a mentioned user.
    This is used when the bot is asked to "start chatting with @user".

    The actual message is generated by the AI, but this provides a fallback.
    """
    starters = [
        f"yo {target_name}",
        f"hey {target_name}, what's good",
        f"sup {target_name}",
        f"{target_name} you around?",
        f"hey {target_name}, how's it going",
    ]
    import random
    return random.choice(starters)


class MentionManager:
    """
    Manages user mentioning and chat initiation.

    Algorithmic behavior:
    - When asked to mention someone, find them in the guild
    - If found, generate the mention and optionally a chat starter
    - If multiple matches, ask for disambiguation
    - If not found, say "can't find that user"
    - Track recently mentioned users to avoid spam
    """

    def __init__(self):
        # Track recently mentioned users (user_id -> timestamp)
        self._recent_mentions = {}
        # Cooldown to avoid mentioning the same user too often (5 min)
        self._mention_cooldown_s = 300

    def was_recently_mentioned(self, user_id: int) -> bool:
        """Check if a user was recently mentioned (within cooldown)."""
        import time
        last = self._recent_mentions.get(user_id, 0)
        return (time.time() - last) < self._mention_cooldown_s

    def mark_mentioned(self, user_id: int):
        """Mark that a user was mentioned."""
        import time
        self._recent_mentions[user_id] = time.time()

    async def handle_mention_request(
        self,
        guild: discord.Guild,
        target_name: str,
        start_chat: bool = False,
    ) -> dict:
        """
        Handle a mention request algorithmically.

        Returns a dict with:
        - "found": bool
        - "mention": str (the formatted mention, or None)
        - "message": str (response message)
        - "member": discord.Member (or None)
        - "multiple_matches": list of members (if ambiguous)
        """
        # Find the member
        members = find_members_by_name(guild, target_name, limit=5)

        if len(members) == 0:
            return {
                "found": False,
                "mention": None,
                "message": f"can't find anyone called '{target_name}' in this server bruv",
                "member": None,
                "multiple_matches": [],
            }

        if len(members) == 1:
            member = members[0]
            mention = format_mention(member)
            self.mark_mentioned(member.id)

            if start_chat:
                # Generate a chat starter — the AI will handle the actual message
                return {
                    "found": True,
                    "mention": mention,
                    "message": f"yo {mention}",
                    "member": member,
                    "multiple_matches": [],
                }
            else:
                return {
                    "found": True,
                    "mention": mention,
                    "message": mention,
                    "member": member,
                    "multiple_matches": [],
                }

        # Multiple matches — ask for disambiguation
        names = [f"{m.display_name} (@{m.name})" for m in members]
        return {
            "found": True,
            "mention": None,
            "message": f"there's {len(members)} people matching '{target_name}': {', '.join(names)}. which one?",
            "member": None,
            "multiple_matches": members,
        }


# Singleton instance
_mention_manager = MentionManager()


def get_mention_manager() -> MentionManager:
    """Get the global mention manager instance."""
    return _mention_manager
