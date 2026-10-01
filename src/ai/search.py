"""
DuckDuckGo Instant Answer search — free, no API key.
Used for factual questions so the bot can give accurate answers.
"""
import json
import urllib.request
import urllib.parse
from loguru import logger


def search(query: str) -> str:
    """
    Return a short plain-text answer from DuckDuckGo Instant Answer API.
    Returns empty string if no answer found.
    """
    try:
        q = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_redirect=1&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())

        # Try abstract summary first
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            return abstract[:400]

        # Fall back to answer field (calculations, conversions)
        answer = data.get("Answer", "").strip()
        if answer:
            return answer[:400]

        # Fall back to first related topic snippet
        topics = data.get("RelatedTopics", [])
        if topics and isinstance(topics[0], dict):
            snippet = topics[0].get("Text", "").strip()
            if snippet:
                return snippet[:400]

        return ""
    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {e}")
        return ""
