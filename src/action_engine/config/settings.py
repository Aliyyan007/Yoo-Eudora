"""Central typed configuration loader.

Reads the `.env` file once at import time and exposes a validated
`settings` singleton. All Groq keys (`GROQ_API_KEY`, `GROQ_API_KEY_2`, ...)
are auto-discovered from the raw environment so you can add as many as you
like without touching this file.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the project root (the "Engager Bot" folder).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _collect_groq_keys() -> List[str]:
    """Discover every GROQ_API_KEY[_N] env var, preserving order."""
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


class Settings(BaseSettings):
    """Validated, typed settings for the whole bot."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- Discord ---
    discord_token: str = Field(..., alias="DISCORD_TOKEN")

    # Comma-separated list of owner IDs — "123" or "123,456" both work.
    owner_user_id: str = Field("0", alias="OWNER_USER_ID")
    command_channel_id: int = Field(0, alias="COMMAND_CHANNEL_ID")

    @property
    def owner_ids(self) -> frozenset[int]:
        """All owner IDs — comma/semicolon separated in .env."""
        out = set()
        for p in str(self.owner_user_id or "").replace(";", ",").split(","):
            p = p.strip()
            if p.isdigit():
                out.add(int(p))
        return frozenset(out)

    def is_owner(self, uid: int) -> bool:
        return uid in self.owner_ids

    # --- Groq ---
    groq_model_text: str = Field("openai/gpt-oss-120b", alias="GROQ_MODEL_TEXT")
    groq_model_chat: str = Field("openai/gpt-oss-20b", alias="GROQ_MODEL_CHAT")
    # ROUTER_MODEL: the cheap arbiter classifying ACTION vs CHAT. Scout
    # isn't available on every Groq tier; 3.3-70b is the universal fallback.
    router_model: str = Field(
        "llama-3.3-70b-versatile", alias="ROUTER_MODEL")
    groq_model_vision: str = Field(
        "meta-llama/llama-4-scout-17b-16e-instruct", alias="GROQ_MODEL_VISION"
    )

    # --- Klipy ---
    klipy_api_key: str = Field("", alias="KLIPY_API_KEY")

    # --- Behaviour ---
    message_history_limit: int = Field(20, alias="MESSAGE_HISTORY_LIMIT")
    max_tool_rounds: int = Field(8, alias="MAX_TOOL_ROUNDS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # --- Auto-bump ---
    bump_channel_ids: list[int] = Field(default_factory=list, alias="BUMP_CHANNEL_IDS")
    auto_bump: bool = Field(False, alias="AUTO_BUMP")

    # --- Voice chat ---
    voice_enabled: bool = Field(True, alias="VOICE_ENABLED")
    voice_stt_model: str = Field("whisper-large-v3-turbo", alias="VOICE_STT_MODEL")
    fish_api_key: str = Field("", alias="FISH_API_KEY")
    fish_voice_id: str = Field("", alias="FISH_VOICE_ID")
    fish_model: str = Field("s2.1-pro-free", alias="FISH_MODEL")
    # Behaviour timing
    voice_idle_ask_minutes: float = Field(12.0, alias="VOICE_IDLE_ASK_MINUTES")
    voice_alone_seconds: float = Field(75.0, alias="VOICE_ALONE_SECONDS")
    voice_greet_on_join: bool = Field(True, alias="VOICE_GREET_ON_JOIN")
    voice_auto_join: bool = Field(True, alias="VOICE_AUTO_JOIN")
    # smart-turn ONNX endpointing — on tiny hosts (Render free 512MB) the
    # transformers+onnxruntime footprint can OOM; set false to use the
    # regex fallback
    voice_smart_turn: bool = Field(True, alias="VOICE_SMART_TURN")

    # --- Derived (not from env directly) ---
    groq_keys: List[str] = Field(default_factory=list)

    @field_validator("command_channel_id", mode="before")
    @classmethod
    def _empty_to_zero(cls, v):
        if v in (None, "", "None"):
            return 0
        return int(v)

    @field_validator("bump_channel_ids", mode="before")
    @classmethod
    def _parse_bump_channels(cls, v):
        """Parse BUMP_CHANNEL_IDS from comma-separated string or list."""
        if v is None or v == "" or v == []:
            return []
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        if isinstance(v, str):
            # Handle comma-separated or JSON array format.
            parts = [p.strip() for p in v.strip("[]").split(",") if p.strip()]
            return [int(p) for p in parts]
        if isinstance(v, int):
            return [v]
        return []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Populate groq_keys from the raw env (not model fields).
        if not self.groq_keys:
            self.groq_keys = _collect_groq_keys()

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


settings = Settings()
