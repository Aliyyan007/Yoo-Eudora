"""
Cloudflare D1 REST API client for persistent memory storage.
Provides async query execution with retry logic and error handling.
"""
import os
import json
import time
import asyncio
import aiohttp
from typing import Any, List, Dict, Optional, Union
from loguru import logger


class D1Unavailable(Exception):
    """Raised when a D1 query fails (HTTP error, timeout, missing creds).
    Lets callers distinguish 'query failed' from 'query returned zero rows'
    instead of silently treating every outage as an empty result."""
    pass


def _load_env():
    """Load D1 credentials — real environment variables first (Render/Docker
    inject them there), falling back to config/.env for local runs."""
    keys = ("CF_D1_ACCOUNT_ID", "CF_D1_DATABASE_ID", "CF_D1_API_TOKEN")
    creds = {k: os.environ[k].strip() for k in keys if os.environ.get(k, "").strip()}
    if len(creds) == len(keys):
        return creds
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", ".env"
    )
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and key not in creds:
                        creds[key] = value
    return creds


_CREDS = _load_env()
_ACCOUNT_ID = _CREDS.get("CF_D1_ACCOUNT_ID", "")
_DATABASE_ID = _CREDS.get("CF_D1_DATABASE_ID", "")
_API_TOKEN = _CREDS.get("CF_D1_API_TOKEN", "")

_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{_ACCOUNT_ID}/d1/database/{_DATABASE_ID}/query"


class D1Client:
    """Async Cloudflare D1 REST API client."""

    # Tables init_schema creates — is_available verifies they exist so a
    # missing-schema DB reads as unavailable instead of silently empty.
    _REQUIRED_TABLES = {
        "user_facts", "user_profiles", "memorable_chats", "instructions",
        "channel_topics", "channel_styles", "channel_lessons",
        "discovered_channels", "server_state",
    }

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._schema_ok = False
        self._write_lock = asyncio.Lock()
        # Brief availability cache — every d1_memory op probes is_available,
        # without this each probe costs a full HTTP round-trip (and up to the
        # retry budget during an outage).
        self._avail_cached = False
        self._avail_cached_until = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _execute_raw(self, sql: str, params: Optional[List[Any]] = None) -> Dict:
        """Execute a SQL query and return D1's raw result[0] (results + meta).

        Raises D1Unavailable on ANY failure — never conflates an outage with
        an empty result set.
        """
        if not _ACCOUNT_ID or not _DATABASE_ID or not _API_TOKEN:
            raise D1Unavailable("D1 credentials not configured")

        body = {"sql": sql}
        if params:
            # Convert all params to strings/numbers (D1 REST API requirement)
            body["params"] = [str(p) if not isinstance(p, (int, float, bool, type(None))) else p for p in params]

        session = await self._get_session()
        last_err = "unknown error"

        for attempt in range(3):
            try:
                async with session.post(_BASE_URL, json=body) as resp:
                    data = await resp.json()

                    if not data.get("success"):
                        errors = data.get("errors", [])
                        err_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                        last_err = f"D1 query failed: {err_msg} (SQL: {sql[:80]})"
                        logger.warning(last_err)
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        raise D1Unavailable(last_err)

                    result = data.get("result", [])
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    return {}

            except D1Unavailable:
                raise
            except asyncio.TimeoutError:
                last_err = f"D1 query timeout (attempt {attempt+1}/3)"
                logger.warning(last_err)
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise D1Unavailable(last_err)
            except Exception as e:
                last_err = f"D1 query error: {e} (attempt {attempt+1}/3)"
                logger.warning(last_err)
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise D1Unavailable(last_err) from e

        raise D1Unavailable(last_err)

    async def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict]:
        """Execute a SQL query and return result rows.

        For SELECT: returns list of row dicts (empty list = genuinely no rows).
        For INSERT/UPDATE/DELETE: returns empty list (check meta for changes).
        Raises D1Unavailable on failure.
        """
        raw = await self._execute_raw(sql, params)
        results = raw.get("results", [])
        return results if isinstance(results, list) else []

    async def execute_write(self, sql: str, params: Optional[List[Any]] = None) -> int:
        """Execute a write query (INSERT/UPDATE/DELETE).

        Returns rows affected (D1 meta.changes). Raises D1Unavailable on
        failure so callers' fallback paths actually engage.
        """
        # Use lock to prevent concurrent writes from stepping on each other
        async with self._write_lock:
            raw = await self._execute_raw(sql, params)
            meta = raw.get("meta") or {}
            try:
                return int(meta.get("changes", meta.get("rows_written", 0)) or 0)
            except (TypeError, ValueError):
                return 0

    async def execute_batch(self, statements: List[tuple]) -> bool:
        """Execute multiple write statements in sequence.
        Each statement is (sql, params) tuple. Raises D1Unavailable on failure.
        """
        async with self._write_lock:
            for sql, params in statements:
                await self._execute_raw(sql, params)
            return True

    async def init_schema(self):
        """Create all required tables if they don't exist."""
        statements = [
            # User facts
            """CREATE TABLE IF NOT EXISTS user_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(user_id, fact)
            )""",

            # User profiles (one row per user)
            """CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                real_name TEXT DEFAULT '',
                hobbies TEXT DEFAULT '[]',
                personality TEXT DEFAULT '',
                relationship TEXT DEFAULT '',
                updated_at REAL NOT NULL
            )""",

            # Memorable chats
            """CREATE TABLE IF NOT EXISTS memorable_chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                chat_text TEXT NOT NULL,
                channel_id TEXT DEFAULT '',
                timestamp REAL NOT NULL
            )""",

            # Instructions from users
            """CREATE TABLE IF NOT EXISTS instructions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                text TEXT NOT NULL,
                channel_id TEXT DEFAULT '',
                timestamp REAL NOT NULL,
                active INTEGER DEFAULT 1,
                UNIQUE(user_id, text)
            )""",

            # Channel topics
            """CREATE TABLE IF NOT EXISTS channel_topics (
                channel_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                updated_at REAL NOT NULL
            )""",

            # Channel styles
            """CREATE TABLE IF NOT EXISTS channel_styles (
                channel_id TEXT PRIMARY KEY,
                style TEXT NOT NULL,
                updated_at REAL NOT NULL
            )""",

            # Channel lessons
            """CREATE TABLE IF NOT EXISTS channel_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                lesson TEXT NOT NULL,
                UNIQUE(channel_id, lesson)
            )""",

            # Discovered channels
            """CREATE TABLE IF NOT EXISTS discovered_channels (
                channel_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                guild_id TEXT NOT NULL,
                last_seen REAL NOT NULL
            )""",

            # Shared server-state KV — survives redeploys where the local
            # filesystem is ephemeral (ping cooldowns, rotation counters)
            """CREATE TABLE IF NOT EXISTS server_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )""",
        ]

        for sql in statements:
            await self.execute(sql)

        self._initialized = True
        logger.info("D1 schema initialized (9 tables)")

    async def _ensure_schema(self):
        """Verify required tables exist, creating them if missing.
        A missing-schema database would otherwise answer every query with a
        'successful' error or empty set — reads lie, writes evaporate."""
        if self._schema_ok:
            return
        rows = await self.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")
        present = {r.get("name") for r in rows}
        if not self._REQUIRED_TABLES.issubset(present):
            await self.init_schema()
        self._schema_ok = True

    async def is_available(self) -> bool:
        """Check if D1 is configured and accessible.

        True only when a real SELECT against the database succeeds AND the
        schema is present — credentials alone prove nothing. The result is
        cached briefly (30s healthy / 8s down) since every memory op probes
        this first; a stale 'up' answer is safe because the real query right
        after still raises D1Unavailable and falls back.
        """
        if not _ACCOUNT_ID or not _DATABASE_ID or not _API_TOKEN:
            return False
        now = time.time()
        if now < self._avail_cached_until:
            return self._avail_cached
        ok = False
        try:
            # Bounded probe — a hung D1 must not stall the memory loop for
            # the full session timeout while it retries.
            await asyncio.wait_for(self._ensure_schema(), timeout=12)
            if self._schema_ok:
                await asyncio.wait_for(self.execute("SELECT 1 as test"), timeout=12)
                ok = True
        except Exception:
            ok = False
        self._avail_cached = ok
        self._avail_cached_until = now + (30.0 if ok else 8.0)
        return ok


# Singleton
_client: Optional[D1Client] = None

def get_d1_client() -> D1Client:
    global _client
    if _client is None:
        _client = D1Client()
    return _client
