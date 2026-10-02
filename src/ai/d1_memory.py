"""
D1-backed memory system — drop-in replacement for memory.py.

Provides the EXACT same function signatures as memory.py, but stores data in
Cloudflare D1 instead of data/memory.json.

Design:
- Each function has an async implementation (``*_async``) that talks to D1.
- A sync wrapper (same name as in memory.py) runs the async version on a
  background event loop via ``asyncio.run_coroutine_threadsafe``.
- If D1 is unavailable or any error occurs, the sync wrapper falls back to the
  original JSON-backed ``memory.py`` functions.
- A simple in-memory cache with TTL avoids hitting D1 on every read.
"""
import asyncio
import json
import re as _re
import threading
import time
from typing import Optional, List, Dict, Any

from loguru import logger

# Import the original JSON-backed memory as a fallback.
from . import memory as _json_fallback

# ── Persona memory isolation ────────────────────────────────────────────────
# User-scoped keys (facts, profiles, memorable chats) are namespaced by the
# ACTIVE persona so one persona's private memory never bleeds into another's.
# Channel-scoped data (topics, styles, lessons, instructions) stays shared —
# that's public server context every persona can see.
_NS = ""


def set_namespace(persona_id: str) -> None:
    """Scope user-memory keys to the active persona (rotation calls this)."""
    global _NS
    _NS = f"{persona_id}:" if persona_id else ""


def _u(user_id) -> str:
    """Persona-namespace a user id. Cached lookups and stored rows all key
    off the result, so each persona's memory is fully independent."""
    return f"{_NS}{user_id}"


# Lazy import of the D1 client (created in parallel at src/ai/d1_client.py).
_d1_client = None


def _get_d1():
    """Return a cached D1Client instance, or None if it can't be imported."""
    global _d1_client
    if _d1_client is None:
        try:
            from .d1_client import get_d1_client
            _d1_client = get_d1_client()
        except Exception as e:
            logger.warning(f"D1 client unavailable: {e}")
            _d1_client = False  # sentinel: "tried and failed"
    if _d1_client is False:
        return None
    return _d1_client


# ── Background event loop for sync wrappers ───────────────────────────────────

_loop = None
_loop_thread = None
_loop_ready = threading.Event()


def _start_bg_loop():
    global _loop
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)
    _loop_ready.set()
    _loop.run_forever()


def _get_bg_loop():
    global _loop, _loop_thread
    if _loop is None:
        _loop_thread = threading.Thread(target=_start_bg_loop, daemon=True)
        _loop_thread.start()
        _loop_ready.wait(timeout=5)
    return _loop


def _run_async(coro):
    """Run an async coroutine on the background loop, return result synchronously."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=30)


# ── In-memory cache ───────────────────────────────────────────────────────────

_cache: Dict[str, tuple] = {}  # key -> (value, timestamp)
_CACHE_TTL = 300  # 5 minutes


def _cache_get(key: str):
    if key in _cache:
        value, ts = _cache[key]
        if time.time() - ts < _CACHE_TTL:
            return value
        _cache.pop(key, None)
    return None


def _cache_set(key: str, value):
    _cache[key] = (value, time.time())


def _cache_invalidate(key: Optional[str] = None):
    if key:
        _cache.pop(key, None)
    else:
        _cache.clear()


# ── Fact contradiction detection (copied verbatim from memory.py) ─────────────

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

    Returns the existing fact that should be replaced, or None.
    """
    new_fact_lower = new_fact.lower()

    for pattern, key_extractor in _FACT_CATEGORIES:
        new_match = _re.search(pattern, new_fact_lower, _re.IGNORECASE)
        if not new_match:
            continue

        new_key = key_extractor(new_match)

        for existing in existing_facts:
            existing_lower = existing.lower()
            existing_match = _re.search(pattern, existing_lower, _re.IGNORECASE)
            if existing_match:
                existing_key = key_extractor(existing_match)
                if existing_key == new_key:
                    if existing_lower != new_fact_lower:
                        return existing
    return None


# ── D1 availability helper ────────────────────────────────────────────────────

async def _d1_ok() -> bool:
    client = _get_d1()
    if client is None:
        return False
    try:
        return await client.is_available()
    except Exception:
        return False


# ── User memory (facts about individual users) ────────────────────────────────

async def get_user_memory_text_async(user_id: str, username: str) -> str:
    """Return a text description of what we know about a user."""
    if not await _d1_ok():
        return _json_fallback.get_user_memory_text(user_id, username)

    cache_key = f"user_memory_text:{user_id}:{username}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        profile_rows = await client.execute(
            "SELECT username, real_name, hobbies, personality, relationship "
            "FROM user_profiles WHERE user_id = ?",
            [str(user_id)],
        )
        fact_rows = await client.execute(
            "SELECT fact FROM user_facts WHERE user_id = ? ORDER BY created_at ASC",
            [str(user_id)],
        )
    except Exception as e:
        logger.warning(f"D1 get_user_memory_text failed: {e}")
        return _json_fallback.get_user_memory_text(user_id, username)

    profile = profile_rows[0] if profile_rows else {}
    facts = [r["fact"] for r in fact_rows]
    name = profile.get("real_name", "") or ""
    hobbies = []
    if profile.get("hobbies"):
        try:
            hobbies = json.loads(profile["hobbies"]) if isinstance(profile["hobbies"], str) else profile["hobbies"]
        except Exception:
            hobbies = []
    personality = profile.get("personality", "") or ""
    relationship = profile.get("relationship", "") or ""

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
        result = f"you don't know much about {username} yet."
    else:
        result = f"about {username}: " + ". ".join(parts) + "."

    _cache_set(cache_key, result)
    return result


def get_user_memory_text(user_id: str, username: str) -> str:
    """Sync wrapper — drop-in replacement for memory.get_user_memory_text."""
    user_id = _u(user_id)
    try:
        return _run_async(get_user_memory_text_async(user_id, username))
    except Exception:
        return _json_fallback.get_user_memory_text(user_id, username)


async def update_user_memory_async(user_id: str, username: str, new_facts: list):
    """Add new facts about a user (dedup, cap at 20, contradiction detection)."""
    if not new_facts:
        return
    if not await _d1_ok():
        return _json_fallback.update_user_memory(user_id, username, new_facts)

    client = _get_d1()
    user_id = str(user_id)
    try:
        # Ensure a profile row exists (for username tracking).
        await client.execute_write(
            "INSERT INTO user_profiles (user_id, username, hobbies, updated_at) "
            "VALUES (?, ?, '[]', ?) "
            "ON CONFLICT(user_id) DO UPDATE SET username = excluded.username",
            [user_id, username, time.time()],
        )

        # Load existing facts.
        rows = await client.execute(
            "SELECT id, fact FROM user_facts WHERE user_id = ? ORDER BY created_at ASC",
            [user_id],
        )
        existing_facts = [r["fact"] for r in rows]
        existing_ids = {r["fact"]: r["id"] for r in rows}
        existing_set = set(existing_facts)

        to_delete_ids = []
        to_add = []
        for fact in new_facts:
            fact = fact.strip()
            if not fact:
                continue
            replaced = _find_contradicting_fact(fact, existing_facts)
            if replaced:
                # Remove the old contradicting fact.
                existing_facts = [f for f in existing_facts if f != replaced]
                if replaced in existing_ids:
                    to_delete_ids.append(existing_ids[replaced])
                logger.info(f"Fact replaced: '{replaced}' -> '{fact}'")
            elif fact not in existing_set:
                to_add.append(fact)
                existing_facts.append(fact)
                existing_set.add(fact)

        # Enforce cap at 20: if we have more than 20, drop oldest.
        if len(existing_facts) > 20:
            overflow = existing_facts[: len(existing_facts) - 20]
            for of in overflow:
                if of in existing_ids:
                    to_delete_ids.append(existing_ids[of])
            existing_facts = existing_facts[-20:]

        # Apply deletions.
        if to_delete_ids:
            placeholders = ",".join("?" * len(to_delete_ids))
            await client.execute_write(
                f"DELETE FROM user_facts WHERE id IN ({placeholders})",
                to_delete_ids,
            )

        # Apply insertions.
        now = time.time()
        for fact in to_add:
            try:
                await client.execute_write(
                    "INSERT INTO user_facts (user_id, fact, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(user_id, fact) DO NOTHING",
                    [user_id, fact, now],
                )
            except Exception as e:
                logger.debug(f"Fact insert skipped ({fact}): {e}")

        _cache_invalidate(f"user_memory_text:{user_id}:")
        logger.debug(f"Memory updated for {username}: {new_facts}")
    except Exception as e:
        logger.warning(f"D1 update_user_memory failed: {e}")
        return _json_fallback.update_user_memory(user_id, username, new_facts)


def update_user_memory(user_id: str, username: str, new_facts: list):
    """Sync wrapper — drop-in replacement for memory.update_user_memory."""
    user_id = _u(user_id)
    try:
        return _run_async(update_user_memory_async(user_id, username, new_facts))
    except Exception:
        return _json_fallback.update_user_memory(user_id, username, new_facts)


# ── User profile ──────────────────────────────────────────────────────────────

async def update_user_profile_async(user_id: str, username: str, field: str, value: str):
    """Update a specific profile field for a user (REPLACES old value)."""
    if not value or not str(value).strip():
        return
    if not await _d1_ok():
        return _json_fallback.update_user_profile(user_id, username, field, value)

    client = _get_d1()
    user_id = str(user_id)
    value = str(value).strip()

    try:
        # Ensure profile row exists.
        await client.execute_write(
            "INSERT INTO user_profiles (user_id, username, hobbies, updated_at) "
            "VALUES (?, ?, '[]', ?) "
            "ON CONFLICT(user_id) DO UPDATE SET username = excluded.username",
            [user_id, username, time.time()],
        )

        if field == "hobbies":
            rows = await client.execute(
                "SELECT hobbies FROM user_profiles WHERE user_id = ?", [user_id])
            current = []
            if rows and rows[0].get("hobbies"):
                try:
                    current = json.loads(rows[0]["hobbies"]) if isinstance(rows[0]["hobbies"], str) else rows[0]["hobbies"]
                except Exception:
                    current = []
            new_hobbies = [h.strip() for h in value.split(",") if h.strip()]
            existing_lower = {h.lower() for h in current}
            for h in new_hobbies:
                if h.lower() not in existing_lower:
                    current.append(h)
                    existing_lower.add(h.lower())
            current = current[-15:]
            await client.execute_write(
                "UPDATE user_profiles SET hobbies = ?, updated_at = ? WHERE user_id = ?",
                [json.dumps(current, ensure_ascii=False), time.time(), user_id],
            )
        elif field == "memorable_chats":
            # Stored in the memorable_chats table.
            await client.execute_write(
                "INSERT INTO memorable_chats (user_id, chat_text, channel_id, timestamp) "
                "VALUES (?, ?, ?, ?)",
                [user_id, value, "", time.time()],
            )
            # Cap at 10: delete oldest beyond 10.
            await client.execute_write(
                "DELETE FROM memorable_chats WHERE user_id = ? AND id NOT IN "
                "(SELECT id FROM memorable_chats WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10)",
                [user_id, user_id],
            )
        else:
            # REPLACES old value (real_name, personality, relationship, etc.)
            rows = await client.execute(
                f"SELECT {field} as old_value FROM user_profiles WHERE user_id = ?",
                [user_id],
            )
            old_value = ""
            if rows:
                old_value = rows[0].get("old_value", "") or ""
            if old_value and old_value.lower() != value.lower():
                logger.info(f"Profile field '{field}' updated: '{old_value}' -> '{value}'")
                # Remove old facts that reference the old value.
                fact_rows = await client.execute(
                    "SELECT id, fact FROM user_facts WHERE user_id = ?", [user_id])
                del_ids = [r["id"] for r in fact_rows if old_value.lower() in (r["fact"] or "").lower()]
                if del_ids:
                    placeholders = ",".join("?" * len(del_ids))
                    await client.execute_write(
                        f"DELETE FROM user_facts WHERE id IN ({placeholders})", del_ids)
            await client.execute_write(
                f"UPDATE user_profiles SET {field} = ?, updated_at = ? WHERE user_id = ?",
                [value, time.time(), user_id],
            )

        _cache_invalidate(f"user_memory_text:{user_id}:")
        _cache_invalidate(f"user_profile:{user_id}")
        _cache_invalidate(f"user_facts:{user_id}")
        logger.info(f"User profile updated: {username}.{field} = {value[:60]}")
    except Exception as e:
        logger.warning(f"D1 update_user_profile failed: {e}")
        return _json_fallback.update_user_profile(user_id, username, field, value)


def update_user_profile(user_id: str, username: str, field: str, value: str):
    """Sync wrapper — drop-in replacement for memory.update_user_profile."""
    user_id = _u(user_id)
    try:
        return _run_async(update_user_profile_async(user_id, username, field, value))
    except Exception:
        return _json_fallback.update_user_profile(user_id, username, field, value)


async def get_user_profile_async(user_id: str) -> dict:
    """Get the full profile dict for a user (mirrors memory.py shape)."""
    if not await _d1_ok():
        return _json_fallback.get_user_profile(user_id)

    cache_key = f"user_profile:{user_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    user_id = str(user_id)
    try:
        prof_rows = await client.execute(
            "SELECT username, real_name, hobbies, personality, relationship "
            "FROM user_profiles WHERE user_id = ?",
            [user_id],
        )
        fact_rows = await client.execute(
            "SELECT fact FROM user_facts WHERE user_id = ? ORDER BY created_at ASC",
            [user_id],
        )
        chat_rows = await client.execute(
            "SELECT chat_text as text, timestamp FROM memorable_chats "
            "WHERE user_id = ? ORDER BY timestamp ASC",
            [user_id],
        )
    except Exception as e:
        logger.warning(f"D1 get_user_profile failed: {e}")
        return _json_fallback.get_user_profile(user_id)

    profile: Dict[str, Any] = {}
    if prof_rows:
        p = prof_rows[0]
        profile["name"] = p.get("username", "") or ""
        if p.get("real_name"):
            profile["real_name"] = p["real_name"]
        hobbies = []
        if p.get("hobbies"):
            try:
                hobbies = json.loads(p["hobbies"]) if isinstance(p["hobbies"], str) else p["hobbies"]
            except Exception:
                hobbies = []
        if hobbies:
            profile["hobbies"] = hobbies
        if p.get("personality"):
            profile["personality"] = p["personality"]
        if p.get("relationship"):
            profile["relationship"] = p["relationship"]
    profile["facts"] = [r["fact"] for r in fact_rows]
    profile["memorable_chats"] = [
        {"text": r["text"], "timestamp": r["timestamp"]} for r in chat_rows
    ]

    _cache_set(cache_key, profile)
    return profile


def get_user_profile(user_id: str) -> dict:
    """Sync wrapper — drop-in replacement for memory.get_user_profile."""
    user_id = _u(user_id)
    try:
        return _run_async(get_user_profile_async(user_id))
    except Exception:
        return _json_fallback.get_user_profile(user_id)


async def get_all_user_facts_async(user_id: str) -> str:
    """Get a comprehensive text summary of everything we know about a user."""
    if not await _d1_ok():
        return _json_fallback.get_all_user_facts(user_id)

    cache_key = f"user_facts:{user_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    profile = await get_user_profile_async(user_id)
    if not profile:
        return ""

    parts = []
    if profile.get("real_name"):
        parts.append(f"Name: {profile['real_name']}")
    if profile.get("facts"):
        parts.append("Facts: " + "; ".join(profile["facts"]))
    if profile.get("hobbies"):
        parts.append("Hobbies: " + ", ".join(profile["hobbies"]))
    if profile.get("personality"):
        parts.append(f"Personality: {profile['personality']}")
    if profile.get("relationship"):
        parts.append(f"Relationship: {profile['relationship']}")
    if profile.get("memorable_chats"):
        parts.append(f"Memorable chats: {len(profile['memorable_chats'])} saved")

    result = " | ".join(parts) if parts else ""
    _cache_set(cache_key, result)
    return result


def get_all_user_facts(user_id: str) -> str:
    """Sync wrapper — drop-in replacement for memory.get_all_user_facts."""
    user_id = _u(user_id)
    try:
        return _run_async(get_all_user_facts_async(user_id))
    except Exception:
        return _json_fallback.get_all_user_facts(user_id)


# ── Instructions ──────────────────────────────────────────────────────────────

async def add_instruction_async(user_id: str, instruction: str, channel_id: str = ""):
    """Store a persistent instruction from a user."""
    if not instruction or not str(instruction).strip():
        return
    if not await _d1_ok():
        return _json_fallback.add_instruction(user_id, instruction, channel_id)

    client = _get_d1()
    instruction = str(instruction).strip()
    user_id = str(user_id)
    try:
        # Check for duplicates (case-insensitive).
        rows = await client.execute(
            "SELECT id FROM instructions WHERE LOWER(text) = LOWER(?) AND active = 1",
            [instruction],
        )
        if rows:
            return  # Already stored

        await client.execute_write(
            "INSERT INTO instructions (user_id, text, channel_id, timestamp, active) "
            "VALUES (?, ?, ?, ?, 1) "
            "ON CONFLICT(user_id, text) DO UPDATE SET active = 1",
            [user_id, instruction, str(channel_id), time.time()],
        )

        # Cap at 50: deactivate oldest beyond 50.
        await client.execute_write(
            "UPDATE instructions SET active = 0 WHERE id NOT IN "
            "(SELECT id FROM instructions WHERE active = 1 ORDER BY timestamp DESC LIMIT 50) "
            "AND active = 1",
            [],
        )

        _cache_invalidate("active_instructions")
        logger.info(f"Instruction stored from {user_id}: {instruction[:80]}")
    except Exception as e:
        logger.warning(f"D1 add_instruction failed: {e}")
        return _json_fallback.add_instruction(user_id, instruction, channel_id)


def add_instruction(user_id: str, instruction: str, channel_id: str = ""):
    """Sync wrapper — drop-in replacement for memory.add_instruction."""
    try:
        return _run_async(add_instruction_async(user_id, instruction, channel_id))
    except Exception:
        return _json_fallback.add_instruction(user_id, instruction, channel_id)


async def get_active_instructions_async() -> List[dict]:
    """Return all active instructions."""
    if not await _d1_ok():
        return _json_fallback.get_active_instructions()

    cache_key = "active_instructions"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        rows = await client.execute(
            "SELECT user_id, text, channel_id, timestamp, active "
            "FROM instructions WHERE active = 1 ORDER BY timestamp ASC"
        )
    except Exception as e:
        logger.warning(f"D1 get_active_instructions failed: {e}")
        return _json_fallback.get_active_instructions()

    result = [
        {
            "text": r["text"],
            "user_id": r["user_id"],
            "channel_id": r.get("channel_id", "") or "",
            "timestamp": r.get("timestamp", 0) or 0,
            "active": bool(r.get("active", 1)),
        }
        for r in rows
    ]
    _cache_set(cache_key, result)
    return result


def get_active_instructions() -> List[dict]:
    """Sync wrapper — drop-in replacement for memory.get_active_instructions."""
    try:
        return _run_async(get_active_instructions_async())
    except Exception:
        return _json_fallback.get_active_instructions()


async def get_instructions_text_async() -> str:
    """Return a text summary of all active instructions for the AI prompt."""
    instructions = await get_active_instructions_async()
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


def get_instructions_text() -> str:
    """Sync wrapper — drop-in replacement for memory.get_instructions_text."""
    try:
        return _run_async(get_instructions_text_async())
    except Exception:
        return _json_fallback.get_instructions_text()


async def deactivate_instruction_async(instruction_text: str):
    """Mark an instruction as inactive."""
    if not await _d1_ok():
        return _json_fallback.deactivate_instruction(instruction_text)

    client = _get_d1()
    try:
        await client.execute_write(
            "UPDATE instructions SET active = 0 WHERE LOWER(text) LIKE LOWER(?)",
            [f"%{instruction_text}%"],
        )
        _cache_invalidate("active_instructions")
    except Exception as e:
        logger.warning(f"D1 deactivate_instruction failed: {e}")
        return _json_fallback.deactivate_instruction(instruction_text)


def deactivate_instruction(instruction_text: str):
    """Sync wrapper — drop-in replacement for memory.deactivate_instruction."""
    try:
        return _run_async(deactivate_instruction_async(instruction_text))
    except Exception:
        return _json_fallback.deactivate_instruction(instruction_text)


# ── Memorable chats ───────────────────────────────────────────────────────────

async def add_memorable_chat_async(user_id: str, username: str, chat_summary: str, channel_id: str = ""):
    """Store a memorable chat with a user."""
    if not await _d1_ok():
        return _json_fallback.add_memorable_chat(user_id, username, chat_summary, channel_id)

    client = _get_d1()
    user_id = str(user_id)
    try:
        # Ensure profile row exists.
        await client.execute_write(
            "INSERT INTO user_profiles (user_id, username, hobbies, updated_at) "
            "VALUES (?, ?, '[]', ?) "
            "ON CONFLICT(user_id) DO UPDATE SET username = excluded.username",
            [user_id, username, time.time()],
        )
        await client.execute_write(
            "INSERT INTO memorable_chats (user_id, chat_text, channel_id, timestamp) "
            "VALUES (?, ?, ?, ?)",
            [user_id, chat_summary, str(channel_id), time.time()],
        )
        # Cap at 10.
        await client.execute_write(
            "DELETE FROM memorable_chats WHERE user_id = ? AND id NOT IN "
            "(SELECT id FROM memorable_chats WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10)",
            [user_id, user_id],
        )
        _cache_invalidate(f"user_profile:{user_id}")
        _cache_invalidate(f"user_facts:{user_id}")
        _cache_invalidate(f"user_memory_text:{user_id}:")
    except Exception as e:
        logger.warning(f"D1 add_memorable_chat failed: {e}")
        return _json_fallback.add_memorable_chat(user_id, username, chat_summary, channel_id)


def add_memorable_chat(user_id: str, username: str, chat_summary: str, channel_id: str = ""):
    """Sync wrapper — drop-in replacement for memory.add_memorable_chat."""
    user_id = _u(user_id)
    try:
        return _run_async(add_memorable_chat_async(user_id, username, chat_summary, channel_id))
    except Exception:
        return _json_fallback.add_memorable_chat(user_id, username, chat_summary, channel_id)


async def get_memorable_chats_async(user_id: str) -> List[dict]:
    """Return memorable chats with a user."""
    if not await _d1_ok():
        return _json_fallback.get_memorable_chats(user_id)

    client = _get_d1()
    user_id = str(user_id)
    try:
        rows = await client.execute(
            "SELECT chat_text as text, timestamp FROM memorable_chats "
            "WHERE user_id = ? ORDER BY timestamp ASC",
            [user_id],
        )
    except Exception as e:
        logger.warning(f"D1 get_memorable_chats failed: {e}")
        return _json_fallback.get_memorable_chats(user_id)

    return [{"text": r["text"], "timestamp": r["timestamp"]} for r in rows]


def get_memorable_chats(user_id: str) -> List[dict]:
    """Sync wrapper — drop-in replacement for memory.get_memorable_chats."""
    user_id = _u(user_id)
    try:
        return _run_async(get_memorable_chats_async(user_id))
    except Exception:
        return _json_fallback.get_memorable_chats(user_id)


# ── Channel topics ────────────────────────────────────────────────────────────

async def get_channel_topic_async(channel_id: str) -> str:
    """Return the stored topic summary for a channel."""
    if not await _d1_ok():
        return _json_fallback.get_channel_topic(channel_id)

    cache_key = f"channel_topic:{channel_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        rows = await client.execute(
            "SELECT topic FROM channel_topics WHERE channel_id = ?",
            [str(channel_id)],
        )
    except Exception as e:
        logger.warning(f"D1 get_channel_topic failed: {e}")
        return _json_fallback.get_channel_topic(channel_id)

    result = rows[0]["topic"] if rows else ""
    _cache_set(cache_key, result)
    return result


def get_channel_topic(channel_id: str) -> str:
    """Sync wrapper — drop-in replacement for memory.get_channel_topic."""
    try:
        return _run_async(get_channel_topic_async(channel_id))
    except Exception:
        return _json_fallback.get_channel_topic(channel_id)


async def update_channel_topic_async(channel_id: str, topic: str):
    """Store a short topic summary for a channel."""
    if not topic:
        return
    if not await _d1_ok():
        return _json_fallback.update_channel_topic(channel_id, topic)

    client = _get_d1()
    channel_id = str(channel_id)
    topic = topic[:120]
    try:
        await client.execute_write(
            "INSERT INTO channel_topics (channel_id, topic, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET topic = excluded.topic, updated_at = excluded.updated_at",
            [channel_id, topic, time.time()],
        )
        _cache_invalidate(f"channel_topic:{channel_id}")
        logger.debug(f"Channel topic updated [{channel_id}]: {topic[:60]}")
    except Exception as e:
        logger.warning(f"D1 update_channel_topic failed: {e}")
        return _json_fallback.update_channel_topic(channel_id, topic)


def update_channel_topic(channel_id: str, topic: str):
    """Sync wrapper — drop-in replacement for memory.update_channel_topic."""
    try:
        return _run_async(update_channel_topic_async(channel_id, topic))
    except Exception:
        return _json_fallback.update_channel_topic(channel_id, topic)


# ── Channel styles ────────────────────────────────────────────────────────────

async def get_channel_style_async(channel_id: str) -> str:
    """Return the stored style profile for a channel."""
    if not await _d1_ok():
        return _json_fallback.get_channel_style(channel_id)

    cache_key = f"channel_style:{channel_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        rows = await client.execute(
            "SELECT style FROM channel_styles WHERE channel_id = ?",
            [str(channel_id)],
        )
    except Exception as e:
        logger.warning(f"D1 get_channel_style failed: {e}")
        return _json_fallback.get_channel_style(channel_id)

    result = rows[0]["style"] if rows else ""
    _cache_set(cache_key, result)
    return result


def get_channel_style(channel_id: str) -> str:
    """Sync wrapper — drop-in replacement for memory.get_channel_style."""
    try:
        return _run_async(get_channel_style_async(channel_id))
    except Exception:
        return _json_fallback.get_channel_style(channel_id)


async def update_channel_style_async(channel_id: str, style: str):
    """Store a learned style profile for a channel."""
    if not style or len(style) < 5:
        return
    if not await _d1_ok():
        return _json_fallback.update_channel_style(channel_id, style)

    client = _get_d1()
    channel_id = str(channel_id)
    style = style[:300]
    try:
        await client.execute_write(
            "INSERT INTO channel_styles (channel_id, style, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET style = excluded.style, updated_at = excluded.updated_at",
            [channel_id, style, time.time()],
        )
        _cache_invalidate(f"channel_style:{channel_id}")
        logger.debug(f"Channel style updated [{channel_id}]: {style[:80]}")
    except Exception as e:
        logger.warning(f"D1 update_channel_style failed: {e}")
        return _json_fallback.update_channel_style(channel_id, style)


def update_channel_style(channel_id: str, style: str):
    """Sync wrapper — drop-in replacement for memory.update_channel_style."""
    try:
        return _run_async(update_channel_style_async(channel_id, style))
    except Exception:
        return _json_fallback.update_channel_style(channel_id, style)


# ── Channel lessons ───────────────────────────────────────────────────────────

async def get_channel_lessons_async(channel_id: str) -> List[str]:
    """Return the stored self-reflection lessons for a channel."""
    if not await _d1_ok():
        return _json_fallback.get_channel_lessons(channel_id)

    cache_key = f"channel_lessons:{channel_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        rows = await client.execute(
            "SELECT lesson FROM channel_lessons WHERE channel_id = ?",
            [str(channel_id)],
        )
    except Exception as e:
        logger.warning(f"D1 get_channel_lessons failed: {e}")
        return _json_fallback.get_channel_lessons(channel_id)

    result = [r["lesson"] for r in rows]
    _cache_set(cache_key, result)
    return result


def get_channel_lessons(channel_id: str) -> List[str]:
    """Sync wrapper — drop-in replacement for memory.get_channel_lessons."""
    try:
        return _run_async(get_channel_lessons_async(channel_id))
    except Exception:
        return _json_fallback.get_channel_lessons(channel_id)


async def update_channel_lessons_async(channel_id: str, new_lessons: list):
    """Store learned conversational lessons (cap at 5)."""
    if not new_lessons:
        return
    if not await _d1_ok():
        return _json_fallback.update_channel_lessons(channel_id, new_lessons)

    client = _get_d1()
    channel_id = str(channel_id)
    try:
        rows = await client.execute(
            "SELECT lesson FROM channel_lessons WHERE channel_id = ?",
            [channel_id],
        )
        existing = {r["lesson"] for r in rows}
        for lesson in new_lessons:
            lesson = lesson.strip()
            if lesson and lesson not in existing:
                try:
                    await client.execute_write(
                        "INSERT INTO channel_lessons (channel_id, lesson) VALUES (?, ?) "
                        "ON CONFLICT(channel_id, lesson) DO NOTHING",
                        [channel_id, lesson],
                    )
                    existing.add(lesson)
                except Exception as e:
                    logger.debug(f"Lesson insert skipped ({lesson}): {e}")

        # Cap at 5: delete oldest beyond 5 (by id).
        await client.execute_write(
            "DELETE FROM channel_lessons WHERE channel_id = ? AND id NOT IN "
            "(SELECT id FROM channel_lessons WHERE channel_id = ? ORDER BY id DESC LIMIT 5)",
            [channel_id, channel_id],
        )
        _cache_invalidate(f"channel_lessons:{channel_id}")
        logger.debug(f"Channel lessons updated [{channel_id}]: {new_lessons}")
    except Exception as e:
        logger.warning(f"D1 update_channel_lessons failed: {e}")
        return _json_fallback.update_channel_lessons(channel_id, new_lessons)


def update_channel_lessons(channel_id: str, new_lessons: list):
    """Sync wrapper — drop-in replacement for memory.update_channel_lessons."""
    try:
        return _run_async(update_channel_lessons_async(channel_id, new_lessons))
    except Exception:
        return _json_fallback.update_channel_lessons(channel_id, new_lessons)


# ── Channel discovery ─────────────────────────────────────────────────────────

async def get_discovered_channels_async() -> dict:
    """Return dict of channel_id -> {name, guild_id, last_seen}."""
    if not await _d1_ok():
        return _json_fallback.get_discovered_channels()

    cache_key = "discovered_channels"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = _get_d1()
    try:
        rows = await client.execute(
            "SELECT channel_id, name, guild_id, last_seen FROM discovered_channels"
        )
    except Exception as e:
        logger.warning(f"D1 get_discovered_channels failed: {e}")
        return _json_fallback.get_discovered_channels()

    result = {
        r["channel_id"]: {
            "name": r.get("name", "") or "",
            "guild_id": r.get("guild_id", "") or "",
            "last_seen": r.get("last_seen", 0) or 0,
        }
        for r in rows
    }
    _cache_set(cache_key, result)
    return result


def get_discovered_channels() -> dict:
    """Sync wrapper — drop-in replacement for memory.get_discovered_channels."""
    try:
        return _run_async(get_discovered_channels_async())
    except Exception:
        return _json_fallback.get_discovered_channels()


async def mark_channel_seen_async(channel_id: str, name: str, guild_id: str):
    """Record that we've seen a channel (for smart selection)."""
    if not await _d1_ok():
        return _json_fallback.mark_channel_seen(channel_id, name, guild_id)

    client = _get_d1()
    channel_id = str(channel_id)
    try:
        await client.execute_write(
            "INSERT INTO discovered_channels (channel_id, name, guild_id, last_seen) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET name = excluded.name, "
            "guild_id = excluded.guild_id, last_seen = excluded.last_seen",
            [channel_id, name, str(guild_id), time.time()],
        )
        _cache_invalidate("discovered_channels")
    except Exception as e:
        logger.warning(f"D1 mark_channel_seen failed: {e}")
        return _json_fallback.mark_channel_seen(channel_id, name, guild_id)


def mark_channel_seen(channel_id: str, name: str, guild_id: str):
    """Sync wrapper — drop-in replacement for memory.mark_channel_seen."""
    try:
        return _run_async(mark_channel_seen_async(channel_id, name, guild_id))
    except Exception:
        return _json_fallback.mark_channel_seen(channel_id, name, guild_id)
