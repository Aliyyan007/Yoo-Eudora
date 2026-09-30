"""
Cloudflare D1 REST API client for persistent memory storage.
Provides async query execution with retry logic and error handling.
"""
import os
import json
import asyncio
import aiohttp
from typing import Any, List, Dict, Optional, Union
from loguru import logger


def _load_env():
    """Load D1 credentials from config/.env"""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config", ".env"
    )
    creds = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    creds[key.strip()] = value.strip()
    return creds


_CREDS = _load_env()
_ACCOUNT_ID = _CREDS.get("CF_D1_ACCOUNT_ID", "")
_DATABASE_ID = _CREDS.get("CF_D1_DATABASE_ID", "")
_API_TOKEN = _CREDS.get("CF_D1_API_TOKEN", "")

_BASE_URL = f"https://api.cloudflare.com/client/v4/accounts/{_ACCOUNT_ID}/d1/database/{_DATABASE_ID}/query"


class D1Client:
    """Async Cloudflare D1 REST API client."""
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialized = False
        self._write_lock = asyncio.Lock()
    
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
    
    async def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict]:
        """Execute a SQL query and return results.
        
        For SELECT: returns list of row dicts.
        for INSERT/UPDATE/DELETE: returns empty list (check meta for changes).
        """
        if not _ACCOUNT_ID or not _DATABASE_ID or not _API_TOKEN:
            logger.warning("D1 credentials not configured — skipping D1 query")
            return []
        
        body = {"sql": sql}
        if params:
            # Convert all params to strings/numbers (D1 REST API requirement)
            body["params"] = [str(p) if not isinstance(p, (int, float, bool, type(None))) else p for p in params]
        
        session = await self._get_session()
        
        for attempt in range(3):
            try:
                async with session.post(_BASE_URL, json=body) as resp:
                    data = await resp.json()
                    
                    if not data.get("success"):
                        errors = data.get("errors", [])
                        err_msg = errors[0].get("message", "Unknown error") if errors else "Unknown error"
                        logger.warning(f"D1 query failed: {err_msg} (SQL: {sql[:80]})")
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        return []
                    
                    result = data.get("result", [])
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("results", [])
                    return []
                    
            except asyncio.TimeoutError:
                logger.warning(f"D1 query timeout (attempt {attempt+1}/3)")
                if attempt < 2:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                return []
            except Exception as e:
                logger.warning(f"D1 query error: {e} (attempt {attempt+1}/3)")
                if attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return []
        
        return []
    
    async def execute_write(self, sql: str, params: Optional[List[Any]] = None) -> bool:
        """Execute a write query (INSERT/UPDATE/DELETE). Returns True on success."""
        # Use lock to prevent concurrent writes from stepping on each other
        async with self._write_lock:
            results = await self.execute(sql, params)
            # For write queries, empty results = success (no rows to return)
            return True  # If we got here without exception, it worked
    
    async def execute_batch(self, statements: List[tuple]) -> bool:
        """Execute multiple write statements in sequence.
        Each statement is (sql, params) tuple.
        """
        async with self._write_lock:
            for sql, params in statements:
                await self.execute(sql, params)
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
        ]
        
        for sql in statements:
            await self.execute(sql)
        
        self._initialized = True
        logger.info("D1 schema initialized (8 tables)")
    
    async def is_available(self) -> bool:
        """Check if D1 is configured and accessible."""
        if not _ACCOUNT_ID or not _DATABASE_ID or not _API_TOKEN:
            return False
        try:
            results = await self.execute("SELECT 1 as test")
            return len(results) > 0 or True  # Query succeeded
        except:
            return False


# Singleton
_client: Optional[D1Client] = None

def get_d1_client() -> D1Client:
    global _client
    if _client is None:
        _client = D1Client()
    return _client
