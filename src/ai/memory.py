"""
Persistent memory system for the AI persona.
Stores: user facts, channel topics, channel styles, self-reflection lessons,
instructions, memorable chats, and user relationships.
All data is saved to data/memory.json with atomic writes.
"""
import os
import json
import time
from typing import Optional, List, Dict, Any
from loguru import logger

_MEMORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "memory.json",
)

# In-memory cache of the loaded memory dict (invalidated on every write)
_memory_cache = None


def _invalidate_cache():
    """Clear the in-memory cache so the next read reloads from disk."""
    global _memory_cache
    _memory_cache = None


def _load_memory() -> dict:
    """Load the full memory dict from disk (cached in memory)."""
    global _memory_cache
    if _memory_cache is not None:
        return _memory_cache
    if os.path.exists(_MEMORY_FILE):
        try:
            with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
                _memory_cache = json.load(f)
                return _memory_cache
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")
    _memory_cache = {}
    return _memory_cache


def _save_memory(memory: dict):
    """Atomic write: write to temp, backup old, replace.

    On Windows, os.replace can fail if the target file is being accessed
    by another process (e.g. antivirus scanning). We retry a few times
    and fall back to a direct write if atomic replace keeps failing.
    """
    try:
        os.makedirs(os.path.dirname(_MEMORY_FILE), exist_ok=True)
        temp_file = _MEMORY_FILE + ".tmp"
        bak_file = _MEMORY_FILE + ".bak"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

        # Try atomic replace with retries for Windows file locking
        import time as _time
        for attempt in range(3):
            try:
                if os.path.exists(_MEMORY_FILE):
                    try:
                        os.replace(_MEMORY_FILE, bak_file)
                    except OSError:
                        # Can't create backup — just remove old file
                        if os.path.exists(_MEMORY_FILE):
                            os.remove(_MEMORY_FILE)
                os.replace(temp_file, _MEMORY_FILE)
                break
            except OSError:
                if attempt < 2:
                    _time.sleep(0.1)
                else:
                    raise
    except Exception as e:
        logger.warning(f"Failed to save memory: {e}")
    _invalidate_cache()


# ── User memory (facts about individual users) ────────────────────────────────

def get_user_memory_text(user_id: str, username: str) -> str:
    """Return a text description of what we know about a user."""
    memory = _load_memory()
    user = memory.get("users", {}).get(user_id, {})
    facts = user.get("facts", [])
    name = user.get("real_name", "")
    hobbies = user.get("hobbies", [])
    personality = user.get("personality", "")
    relationship = user.get("relationship", "")

    parts = []
    if name:
        parts.append(f"their real name is {name}")
    if facts:
        parts.append("things you know: " + "; ".join(facts))
    if hobbies:
        parts.append("their hobbies/interests: " + ", ".join(hobbies))
    if personality:
        parts.append(f"their personality: {personality}")
    if relationship:
        parts.append(f"your relationship with them: {relationship}")

    if not parts:
        return f"you don't know much about {username} yet."
    return f"about {username}: " + ". ".join(parts) + "."


def update_user_memory(user_id: str, username: str, new_facts: list):
    """Add new facts about a user (dedup, cap at 20).

    Uses ALGORITHMIC fact replacement: if a new fact contradicts an old one
    (e.g., "my name is Aliyyan" -> "call me Alien"), the old fact is REPLACED,
    not just added alongside.
    """
    if not new_facts:
        return
    memory = _load_memory()
    if "users" not in memory:
        memory["users"] = {}
    if user_id not in memory["users"]:
        memory["users"][user_id] = {"name": username, "facts": []}

    existing_facts = memory["users"][user_id]["facts"]
    existing_set = set(existing_facts)

    for fact in new_facts:
        fact = fact.strip()
        if not fact:
            continue

        # Check if this fact replaces an existing one
        replaced = _find_contradicting_fact(fact, existing_facts)
        if replaced:
            # Remove the old contradicting fact
            existing_facts = [f for f in existing_facts if f != replaced]
            logger.info(f"Fact replaced: '{replaced}' -> '{fact}'")
        elif fact not in existing_set:
            existing_facts.append(fact)
            existing_set.add(fact)

    memory["users"][user_id]["facts"] = existing_facts[-20:]
    _save_memory(memory)
    logger.debug(f"Memory updated for {username}: {new_facts}")


# ── Fact contradiction detection (algorithmic) ────────────────────────────────

# Categories of facts that can be UPDATED (replaced) rather than just added
# Each entry: (pattern_to_match, key_extractor) — if two facts produce the
# same key, the newer one replaces the older.
import re as _re

_FACT_CATEGORIES = [
    # Name facts: "their real name is X" / "name is X" / "called X"
    (r'(?:real\s+name\s+is|name\s+is|called|known\s+as)\s+(\w+)',
     lambda m: "name"),
    # Age facts: "is X years old" / "age is X"
    (r'(?:years?\s+old|age\s+is)\s*(\d+)',
     lambda m: "age"),
    # Location facts: "from X" / "lives in X"
    (r'(?:from|lives?\s+in|resides?\s+in)\s+(\w+)',
     lambda m: "location"),
    # Hobby facts: "likes X" / "enjoys X" / "loves X"
    (r'(?:likes?|enjoys?|loves?)\s+(\w+)',
     lambda m: "hobby"),
    # Relationship facts: "is my X" / "relationship is X"
    (r'(?:is\s+my|relationship\s+is)\s+(\w+)',
     lambda m: "relationship"),
]


def _find_contradicting_fact(new_fact: str, existing_facts: list) -> Optional[str]:
    """
    Algorithmically detect if a new fact contradicts/replaces an existing one.

    Example:
    - Existing: "their real name is Aliyyan"
    - New: "their real name is Alien"
    - Result: Returns "their real name is Aliyyan" (to be replaced)

    Returns the existing fact that should be replaced, or None.
    """
    new_fact_lower = new_fact.lower()

    for pattern, key_extractor in _FACT_CATEGORIES:
        new_match = _re.search(pattern, new_fact_lower, _re.IGNORECASE)
        if not new_match:
            continue

        new_key = key_extractor(new_match)

        # Search existing facts for the same category
        for existing in existing_facts:
            existing_lower = existing.lower()
            existing_match = _re.search(pattern, existing_lower, _re.IGNORECASE)
            if existing_match:
                existing_key = key_extractor(existing_match)
                # Same category = same key -> this is a replacement
                if existing_key == new_key:
                    # But make sure the actual value is different
                    if existing_lower != new_fact_lower:
                        return existing
    return None


def update_user_profile(user_id: str, username: str, field: str, value: str):
    """
    Update a specific profile field for a user.
    REPLACES the old value (doesn't keep both).

    Fields: real_name, hobbies (list), personality, relationship
    """
    if not value or not str(value).strip():
        return
    memory = _load_memory()
    if "users" not in memory:
        memory["users"] = {}
    if user_id not in memory["users"]:
        memory["users"][user_id] = {"name": username, "facts": []}

    value = str(value).strip()
    if field == "hobbies":
        if not isinstance(memory["users"][user_id].get("hobbies"), list):
            memory["users"][user_id]["hobbies"] = []
        # Split comma-separated hobbies
        new_hobbies = [h.strip() for h in value.split(",") if h.strip()]
        existing = set(memory["users"][user_id]["hobbies"])
        for h in new_hobbies:
            if h.lower() not in [e.lower() for e in existing]:
                memory["users"][user_id]["hobbies"].append(h)
        memory["users"][user_id]["hobbies"] = memory["users"][user_id]["hobbies"][-15:]
    elif field == "memorable_chats":
        if not isinstance(memory["users"][user_id].get("memorable_chats"), list):
            memory["users"][user_id]["memorable_chats"] = []
        memory["users"][user_id]["memorable_chats"].append({
            "text": value,
            "timestamp": time.time(),
        })
        # Cap at 10 memorable chats
        memory["users"][user_id]["memorable_chats"] = \
            memory["users"][user_id]["memorable_chats"][-10:]
    else:
        # REPLACES old value (for real_name, personality, relationship, etc.)
        old_value = memory["users"][user_id].get(field, "")
        if old_value and old_value.lower() != value.lower():
            logger.info(f"Profile field '{field}' updated: '{old_value}' -> '{value}'")
            # Also remove old fact that references the old value
            facts = memory["users"][user_id].get("facts", [])
            memory["users"][user_id]["facts"] = [
                f for f in facts if old_value.lower() not in f.lower()
            ]
        memory["users"][user_id][field] = value

    _save_memory(memory)
    logger.info(f"User profile updated: {username}.{field} = {value[:60]}")


def get_user_profile(user_id: str) -> dict:
    """Get the full profile dict for a user."""
    memory = _load_memory()
    return memory.get("users", {}).get(user_id, {})


def get_all_user_facts(user_id: str) -> str:
    """Get a comprehensive text summary of everything we know about a user."""
    memory = _load_memory()
    user = memory.get("users", {}).get(user_id, {})
    if not user:
        return ""

    parts = []
    if user.get("real_name"):
        parts.append(f"Name: {user['real_name']}")
    if user.get("facts"):
        parts.append("Facts: " + "; ".join(user["facts"]))
    if user.get("hobbies"):
        parts.append("Hobbies: " + ", ".join(user["hobbies"]))
    if user.get("personality"):
        parts.append(f"Personality: {user['personality']}")
    if user.get("relationship"):
        parts.append(f"Relationship: {user['relationship']}")
    if user.get("memorable_chats"):
        parts.append(f"Memorable chats: {len(user['memorable_chats'])} saved")

    return " | ".join(parts) if parts else ""


# ── Instructions (persistent commands/preferences from users) ─────────────────

def add_instruction(user_id: str, instruction: str, channel_id: str = ""):
    """Store a persistent instruction from a user.

    Examples:
    - "stop pinging, try after 16h"
    - "don't reply to messages from user X"
    - "always bump servers when I ask"
    """
    if not instruction or not str(instruction).strip():
        return
    memory = _load_memory()
    if "instructions" not in memory:
        memory["instructions"] = []
    instruction = str(instruction).strip()
    # Check for duplicates
    for existing in memory["instructions"]:
        if existing.get("text", "").lower() == instruction.lower():
            return  # Already stored
    memory["instructions"].append({
        "text": instruction,
        "user_id": user_id,
        "channel_id": channel_id,
        "timestamp": time.time(),
        "active": True,
    })
    # Cap at 50 instructions
    memory["instructions"] = memory["instructions"][-50:]
    _save_memory(memory)
    logger.info(f"Instruction stored from {user_id}: {instruction[:80]}")


def get_active_instructions() -> List[dict]:
    """Return all active instructions."""
    memory = _load_memory()
    instructions = memory.get("instructions", [])
    return [i for i in instructions if i.get("active", True)]


def get_instructions_text() -> str:
    """Return a text summary of all active instructions for the AI prompt."""
    instructions = get_active_instructions()
    if not instructions:
        return ""
    lines = []
    for i in instructions:
        age = int(time.time() - i.get("timestamp", 0))
        if age < 3600:
            age_str = f"{age // 60}m ago"
        elif age < 86400:
            age_str = f"{age // 3600}h ago"
        else:
            age_str = f"{age // 86400}d ago"
        lines.append(f"- [{age_str}] {i['text']}")
    return "ACTIVE INSTRUCTIONS FROM USERS (follow these):\n" + "\n".join(lines)


def deactivate_instruction(instruction_text: str):
    """Mark an instruction as inactive (e.g., user said 'ok you can ping now')."""
    memory = _load_memory()
    for i in memory.get("instructions", []):
        if instruction_text.lower() in i.get("text", "").lower():
            i["active"] = False
    _save_memory(memory)


# ── Memorable chats (special conversations to remember) ──────────────────────

def add_memorable_chat(user_id: str, username: str, chat_summary: str, channel_id: str = ""):
    """Store a memorable chat with a user."""
    update_user_profile(user_id, username, "memorable_chats", chat_summary)


def get_memorable_chats(user_id: str) -> List[dict]:
    """Return memorable chats with a user."""
    memory = _load_memory()
    user = memory.get("users", {}).get(user_id, {})
    return user.get("memorable_chats", [])


# ── Channel topics ────────────────────────────────────────────────────────────

def get_channel_topic(channel_id: str) -> str:
    """Return the stored topic summary for a channel."""
    return _load_memory().get("channel_topics", {}).get(channel_id, "")


def update_channel_topic(channel_id: str, topic: str):
    """Store a short topic summary for a channel."""
    if not topic:
        return
    memory = _load_memory()
    if "channel_topics" not in memory:
        memory["channel_topics"] = {}
    memory["channel_topics"][channel_id] = topic[:120]
    _save_memory(memory)
    logger.debug(f"Channel topic updated [{channel_id}]: {topic[:60]}")


# ── Channel styles (how people talk in a channel) ─────────────────────────────

def get_channel_style(channel_id: str) -> str:
    """Return the stored style profile for a channel."""
    return _load_memory().get("channel_styles", {}).get(channel_id, "")


def update_channel_style(channel_id: str, style: str):
    """Store a learned style profile for a channel."""
    if not style or len(style) < 5:
        return
    memory = _load_memory()
    if "channel_styles" not in memory:
        memory["channel_styles"] = {}
    memory["channel_styles"][channel_id] = style[:300]
    _save_memory(memory)
    logger.debug(f"Channel style updated [{channel_id}]: {style[:80]}")


# ── Channel lessons (self-reflection) ─────────────────────────────────────────

def get_channel_lessons(channel_id: str) -> List[str]:
    """Return the stored self-reflection lessons for a channel."""
    return _load_memory().get("channel_lessons", {}).get(channel_id, [])


def update_channel_lessons(channel_id: str, new_lessons: list):
    """Store learned conversational lessons (cap at 5)."""
    if not new_lessons:
        return
    memory = _load_memory()
    if "channel_lessons" not in memory:
        memory["channel_lessons"] = {}
    existing = set(memory["channel_lessons"].get(channel_id, []))
    for lesson in new_lessons:
        lesson = lesson.strip()
        if lesson and lesson not in existing:
            existing.add(lesson)
    memory["channel_lessons"][channel_id] = list(existing)[-5:]
    _save_memory(memory)
    logger.debug(f"Channel lessons updated [{channel_id}]: {new_lessons}")


# ── Channel discovery (which channels we've seen) ─────────────────────────────

def get_discovered_channels() -> dict:
    """Return dict of channel_id -> {name, guild_id, last_seen}."""
    return _load_memory().get("discovered_channels", {})


def mark_channel_seen(channel_id: str, name: str, guild_id: str):
    """Record that we've seen a channel (for smart selection)."""
    memory = _load_memory()
    if "discovered_channels" not in memory:
        memory["discovered_channels"] = {}
    memory["discovered_channels"][channel_id] = {
        "name": name,
        "guild_id": guild_id,
        "last_seen": time.time(),
    }
    _save_memory(memory)
