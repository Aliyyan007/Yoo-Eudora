"""
Sentiment-aware response modulation.

Detects the emotional tone of incoming messages and adjusts the bot's
response style accordingly. This creates emotional intelligence — the bot
responds differently to someone who's excited vs someone who's sad vs
someone who's angry.

Pattern from SignalSweep's "Sentiment/Hedging" gate and MaiBot's
emotional context system.

How it works:
1. Lightweight keyword-based sentiment detection (no LLM call needed)
2. Classifies messages as: excited, happy, neutral, sad, angry, urgent, questioning
3. The sentiment is passed to the AI prompt so the LLM can modulate its tone
4. Also affects timing: sadder messages get longer read delays (thoughtful),
   excited messages get faster responses (matching energy)
"""
import re
from typing import Tuple, Optional
from loguru import logger


# Sentiment categories with keyword patterns
# Each pattern is (regex, weight) — higher weight = stronger signal
SENTIMENT_PATTERNS = {
    "excited": [
        (r'\b(omg|omfg|yesss|let\'?s gooo?|woo+h?+|woot|hype+d?|pog|poggers|insane|crazy|amazing|incredible|wild)\b', 3),
        (r'!{2,}', 2),
        (r'[A-Z]{4,}', 2),  # ALL CAPS words
        (r'(🔥|🚀|⚡|💥|🎉|🎊|🤯|😱|😍|🥳)', 3),
    ],
    "happy": [
        (r'\b(happy|glad|love|great|awesome|cool|nice|sweet|perfect|good vibes|blessed|grateful)\b', 2),
        (r'(😊|😄|😃|🙂|💚|💛|❤️|🧡|💕|✨)', 2),
        (r'\b(lol|lmao|lmfao|haha|hehe|funny)\b', 1),
    ],
    "sad": [
        (r'\b(sad|depressed|lonely|hurt|pain|crying|tears|miss\s+you|broken|lost|empty|hopeless|tired\s+of)\b', 3),
        (r'(😢|😭|💔|😔|😞|😟|🥺)', 3),
        (r'\b(rip|gone|passed\s+away)\b', 2),
    ],
    "angry": [
        (r'\b(angry|mad|furious|pissed|hate|stupid|dumb|idiot|trash|garbage|bs|bullshit|wtf)\b', 3),
        (r'(😡|🤬|😤|👎)', 3),
        (r'\b(rage|tilted|salt+y?)\b', 2),
    ],
    "urgent": [
        (r'\b(help|emergency|asap|urgent|now|quick|fast|hurry)\b', 2),
        (r'\b(someone|anybody|anyone)\b.*\?', 1),
    ],
    "questioning": [
        (r'\?', 1),
        (r'\b(what|why|how|when|where|who|which|can\s+you|do\s+you|are\s+you|is\s+it)\b', 1),
    ],
    "bored": [
        (r'\b(bored|boring|dead\s+chat|quiet|nothing\s+to\s+do|anyone\s+on|who\'?s\s+on)\b', 2),
        (r'(🥱|💤)', 2),
    ],
    "grateful": [
        (r'\b(thanks|thank\s+you|ty|appreciate|grateful|blessed)\b', 3),
    ],
    "sarcastic": [
        (r'\b(sure|right|yeah\s+right|obviously|duh|wow\s+so\s+amazing)\b', 1),
        (r'\b(great|perfect|wonderful)\b.*\b(again|as\s+always|per\s+usual)\b', 2),
    ],
}


def detect_sentiment(text: str) -> Tuple[str, int]:
    """
    Detect the sentiment of a message using keyword matching.
    Returns (sentiment_category, confidence_score).
    Category is the highest-scoring sentiment, or "neutral" if none dominate.
    """
    if not text or not text.strip():
        return "neutral", 0

    text_lower = text.lower()
    scores = {}

    for category, patterns in SENTIMENT_PATTERNS.items():
        total = 0
        for pattern, weight in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                total += weight * min(len(matches), 3)  # cap repeats
        if total > 0:
            scores[category] = total

    if not scores:
        return "neutral", 0

    # Get the highest scoring category
    best = max(scores, key=scores.get)
    return best, scores[best]


def get_sentiment_timing_modifier(sentiment: str) -> float:
    """
    Return a timing multiplier based on sentiment.
    - Excited/urgent -> faster response (match the energy)
    - Sad/thoughtful -> slower response (being thoughtful)
    - Neutral -> normal timing
    """
    modifiers = {
        "excited": 0.6,      # 40% faster
        "urgent": 0.5,       # 50% faster
        "happy": 0.8,        # 20% faster
        "grateful": 0.9,     # 10% faster
        "questioning": 0.85, # 15% faster (people are waiting for an answer)
        "neutral": 1.0,      # normal
        "bored": 1.2,        # 20% slower (no rush)
        "sarcastic": 1.1,    # 10% slower (let it land)
        "sad": 1.4,          # 40% slower (be thoughtful)
        "angry": 1.3,        # 30% slower (don't react impulsively)
    }
    return modifiers.get(sentiment, 1.0)


def get_sentiment_context(sentiment: str, score: int) -> str:
    """
    Return a context string for the AI prompt describing the detected sentiment.
    This helps the LLM modulate its response tone.
    """
    if sentiment == "neutral" or score == 0:
        return ""

    descriptions = {
        "excited": "The person seems excited/hyped. Match their energy!",
        "happy": "The person seems happy/positive. Keep the good vibes going.",
        "sad": "The person seems sad/down. Be gentle and supportive, not overly cheerful.",
        "angry": "The person seems frustrated/angry. Don't be dismissive, but stay calm.",
        "urgent": "The person needs help urgently. Be responsive and direct.",
        "questioning": "The person is asking a question. Give a clear, helpful answer.",
        "bored": "The person is bored. Suggest something fun or start a topic.",
        "grateful": "The person is thankful. Acknowledge it casually, don't be weird about it.",
        "sarcastic": "The person is being sarcastic. Banter back, don't take it literally.",
    }

    desc = descriptions.get(sentiment, "")
    if desc:
        return f"[DETECTED SENTIMENT: {sentiment} (score: {score}) — {desc}]"
    return ""


# ── Loneliness / seeking-chat detection ───────────────────────────────────────

# Patterns that indicate someone is looking for chat partners
# These should ALWAYS trigger a response from the bot
LONELINESS_PATTERNS = [
    r'\b(anyone\s+(?:here|there|online|awake|up|around|active))\b',
    r'\b(someone\s+(?:talk|text|chat|reply|respond|help|here))\b',
    r'\b(anybody\s+(?:here|there|online|awake|want\s+to\s+chat))\b',
    r'\b(is\s+anyone\s+(?:here|there|online|awake|up))\b',
    r'\b(who\'?s\s+(?:here|there|online|awake|up|around|active))\b',
    r'\b(any\s+(?:one|body)\s+(?:wanna|want\s+to)\s+(?:chat|talk|text))\b',
    r'\b(dead\s+(?:chat|server)|chat\'?s\s+dead|so\s+quiet)\b',
    r'\b(i\'?m\s+(?:bored|lonely|alone|by\s+myself))\b',
    r'\b(someone\s+please|plz\s+someone|anyone\s+please)\b',
    r'\b(need\s+someone\s+to\s+talk|want\s+to\s+talk\s+to\s+someone)\b',
    r'\b(bored\s+af|so\s+bored|bored\s+out\s+of\s+my\s+mind)\b',
    r'\b(lets?\s+talk|lets?\s+chat|who\s+wants?\s+to\s+chat)\b',
    r'\b(wake\s+up|revive|liven\s+up)\b',
]


def detect_loneliness(text: str) -> bool:
    """
    Algorithmically detect if a message indicates the sender is looking for
    someone to chat with. This triggers an ALWAYS-RESPOND behavior.

    Circumstances checked:
    - Direct questions ("anyone here?", "who's online?")
    - Boredom expressions ("so bored", "dead chat")
    - Direct requests ("someone talk", "let's chat")
    - Loneliness signals ("I'm alone", "by myself")
    """
    if not text:
        return False

    text_lower = text.lower()
    for pattern in LONELINESS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False
