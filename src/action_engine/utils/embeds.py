"""Convert Discord embed objects into clean JSON for the AI.

discord.py-self's `Embed` exposes `.to_dict()`, but the result can contain
datetime objects and other non-JSON-serialisable bits. This module produces a
fully JSON-serialisable dict describing every part of an embed (author,
fields, footer, image, thumbnail, colour, timestamps) so the agent can reason
about the embed's "nature".
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if isinstance(dt, datetime) else None


def embed_to_json(embed: Any) -> dict:
    """Serialise a discord Embed into a plain JSON-able dict."""
    if embed is None:
        return {}
    try:
        raw = embed.to_dict()
    except Exception:
        raw = {}

    out: dict[str, Any] = {}
    out["title"] = raw.get("title")
    out["description"] = raw.get("description")
    out["url"] = raw.get("url")
    if "color" in raw:
        out["color"] = raw["color"]
    out["timestamp"] = _iso(embed.timestamp) if hasattr(embed, "timestamp") else raw.get("timestamp")

    if "author" in raw:
        out["author"] = {
            "name": raw["author"].get("name"),
            "url": raw["author"].get("url"),
            "icon_url": raw["author"].get("icon_url"),
        }
    if "footer" in raw:
        out["footer"] = {
            "text": raw["footer"].get("text"),
            "icon_url": raw["footer"].get("icon_url"),
        }
    if "image" in raw:
        out["image"] = raw["image"].get("url")
    if "thumbnail" in raw:
        out["thumbnail"] = raw["thumbnail"].get("url")
    if "fields" in raw:
        out["fields"] = [
            {
                "name": f.get("name"),
                "value": f.get("value"),
                "inline": f.get("inline", False),
            }
            for f in raw["fields"]
        ]
    # Provider / video (rich embeds)
    if "provider" in raw:
        out["provider"] = raw["provider"]
    if "video" in raw:
        out["video"] = raw["video"].get("url")
    return {k: v for k, v in out.items() if v is not None}
