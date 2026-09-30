"""Environment shim for the vendored action-worker — mirrors the subset of
Engager settings its tools/pool actually read, sourced from the same env.

Plain class (no pydantic). Loads the project's ``config/.env`` first (where
this repo keeps its secrets), then the root ``.env`` as a fallback. Values
not present in the env fall back to the defaults below.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# src/actions/config/settings.py -> parents[0]=config, [1]=actions, [2]=src,
# [3]=project root (the "stable version" folder).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

load_dotenv(_PROJECT_ROOT / "config" / ".env")
load_dotenv(_PROJECT_ROOT / ".env")


def _collect_groq_keys() -> list[str]:
    """Discover every GROQ_API_KEY[_N] env var, preserving order.

    Bare ``GROQ_API_KEY`` sorts first, then ``GROQ_API_KEY_2..N`` in numeric
    order. Values are deduplicated so the same key listed twice only counts
    once.
    """
    keys: list[str] = []
    seen: set[str] = set()
    # Match GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_10, ...
    pattern = re.compile(r"^GROQ_API_KEY(?:_(\d+))?$")
    items: list[tuple[int, str]] = []
    for name, value in os.environ.items():
        m = pattern.match(name)
        if not m:
            continue
        v = (value or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        # bare GROQ_API_KEY sorts as index 1
        idx = int(m.group(1)) if m.group(1) else 1
        items.append((idx, v))
    items.sort(key=lambda t: t[0])
    keys = [v for _, v in items]
    return keys


def _parse_bump_channel_ids() -> list[int]:
    """Parse BUMP_CHANNEL_IDS (csv / semicolon / JSON-ish list) into ints.

    Falls back to the singular BUMP_CHANNEL_ID this repo's env actually uses.
    """
    raw = os.getenv("BUMP_CHANNEL_IDS") or os.getenv("BUMP_CHANNEL_ID") or ""
    out: list[int] = []
    for p in re.split(r"[,\s;]+", raw.strip().strip("[]")):
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
    return out


def _parse_int_env(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return int(default)


class _Settings:
    """Flat attribute bag matching the fields the vendored code reads."""

    # --- Discord ---
    discord_token = os.getenv("DISCORD_TOKEN", "")
    server_invite = os.getenv("SERVER_INVITE", "")
    # Comma/semicolon-separated owner IDs — "123" or "123,456" both work.
    owner_user_id = os.getenv("OWNER_USER_ID", os.getenv("OWNER_IDS", "0"))
    command_channel_id = 0

    # --- Groq ---
    groq_model_text = os.getenv(
        "ACTION_MODEL", os.getenv("GROQ_MODEL_TEXT", "openai/gpt-oss-120b"))
    groq_model_chat = os.getenv("GROQ_MODEL_CHAT", "openai/gpt-oss-20b")
    router_model = os.getenv("ACTION_ROUTER_MODEL", "llama-3.1-8b-instant")
    groq_model_vision = os.getenv(
        "GROQ_MODEL_VISION", "meta-llama/llama-4-scout-17b-16e-instruct")
    groq_keys = _collect_groq_keys()

    # --- Klipy ---
    klipy_api_key = os.getenv("KLIPY_API_KEY", "")

    # --- Behaviour ---
    message_history_limit = _parse_int_env("MESSAGE_HISTORY_LIMIT", "20")
    max_tool_rounds = _parse_int_env("ACTION_MAX_ROUNDS", "6")
    log_level = os.getenv("LOG_LEVEL", "INFO")

    # --- Auto-bump ---
    auto_bump = False
    bump_channel_ids = _parse_bump_channel_ids()

    # --- Voice / typing (vendored worker doesn't run voice sessions) ---
    voice_enabled = False
    typing_sim_enabled = False

    @property
    def owner_ids(self) -> set[int]:
        """All owner IDs — comma/semicolon separated, from owner_user_id
        merged with the OWNER_IDS env var."""
        out: set[int] = set()
        raw = f"{self.owner_user_id or ''};{os.getenv('OWNER_IDS', '')}"
        for p in re.split(r"[,\s;]+", raw):
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return out

    def is_owner(self, uid) -> bool:
        """True when uid (str|int|None) is one of the configured owners."""
        try:
            return int(uid) in self.owner_ids
        except (TypeError, ValueError):
            return False

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def logs_dir(self) -> Path:
        d = _PROJECT_ROOT / "logs"
        d.mkdir(exist_ok=True)
        return d

    @property
    def data_dir(self) -> Path:
        d = _PROJECT_ROOT / "data"
        d.mkdir(exist_ok=True)
        return d


settings = _Settings()
