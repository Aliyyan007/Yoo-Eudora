"""
Mood engine — simulates emotional state that drifts over time.
11 mood states that affect reply style, with content-aware nudges.
"""
import time
import random
from loguru import logger

# All possible moods
MOODS = [
    "hyped", "giddy", "happy", "playful", "proud",
    "curious", "silly",
    "bored", "tired", "flat",
    "nostalgic", "annoyed",
]

# Mood transition neighbors (prefer nearby moods over extreme jumps)
MOOD_NEIGHBORS = {
    "hyped":     ["happy", "giddy", "playful", "silly"],
    "giddy":     ["hyped", "silly", "playful", "happy"],
    "happy":     ["hyped", "playful", "curious", "bored"],
    "playful":   ["silly", "happy", "curious", "hyped"],
    "curious":   ["playful", "happy", "bored", "silly"],
    "silly":     ["playful", "giddy", "happy", "bored"],
    "proud":     ["happy", "bored", "flat"],
    "bored":     ["flat", "tired", "curious", "annoyed"],
    "tired":     ["bored", "flat"],
    "flat":      ["bored", "tired", "nostalgic", "annoyed"],
    "nostalgic": ["happy", "bored", "flat"],
    "annoyed":   ["flat", "bored", "tired"],
}

# Content keywords that nudge mood
CONTENT_NUDGES = {
    # Funny / chaotic content
    ("lmao", "lol", "died", "💀", "😭", "bro what", "npc", "blud", "nah", "skull"):
        ["silly", "playful", "hyped"],
    # Interesting / intellectual content
    ("how", "why", "explain", "wait", "actually", "what if", "interesting"):
        ["curious", "curious", "playful"],
    # Negative / conflict
    ("stop", "blocked", "hate", "worst", "ugh", "annoying", "stupid"):
        ["flat"],
    # Wholesome / warm
    ("hug", "love", "miss", "remember", "used to", "nostalgia", "good times"):
        ["happy", "nostalgic"],
    # Exciting news / wins
    ("omg", "no way", "finally", "lets go", "yooo", "w", "clutch", "gg"):
        ["hyped", "giddy", "happy"],
}


class MoodEngine:
    """Manages the bot's current mood with time-based drift and content nudges."""

    def __init__(self):
        self.current_mood = random.choice(MOODS)
        self.mood_set_at = time.time()
        logger.info(f"Initial mood: {self.current_mood}")

    def maybe_drift(self, trigger_text: str = "") -> str:
        """
        Possibly shift the mood based on:
        1. Content of the trigger message (15% chance)
        2. Time elapsed since last mood change (20-45 min drift)
        Returns the current mood after potential drift.
        """
        # ── Content-aware immediate nudges ───────────────────────────────
        if trigger_text:
            txt = trigger_text.lower()
            nudge = None
            for keywords, target_moods in CONTENT_NUDGES.items():
                if any(kw in txt for kw in keywords):
                    nudge = random.choice(target_moods)
                    break

            if nudge and nudge != self.current_mood and random.random() < 0.15:
                logger.debug(f"Mood nudge: {self.current_mood} -> {nudge} (content)")
                self.current_mood = nudge
                self.mood_set_at = time.time()
                return self.current_mood

        # ── Time-based drift ─────────────────────────────────────────────
        DRIFT_MIN = 1200   # 20 min minimum
        DRIFT_MAX = 2700   # 45 min max
        elapsed = time.time() - self.mood_set_at

        if elapsed > DRIFT_MIN:
            drift_prob = min((elapsed - DRIFT_MIN) / (DRIFT_MAX - DRIFT_MIN), 1.0) * 0.3
            if random.random() < drift_prob:
                neighbors = MOOD_NEIGHBORS.get(self.current_mood, MOODS)
                new = random.choice(neighbors)
                if new != self.current_mood:
                    logger.debug(f"Mood drift: {self.current_mood} -> {new} (time-based)")
                    self.current_mood = new
                    self.mood_set_at = time.time()

        return self.current_mood

    def set_mood(self, mood: str):
        """Force-set the mood (e.g., from AI response)."""
        if mood in MOODS:
            self.current_mood = mood
            self.mood_set_at = time.time()
            logger.debug(f"Mood set to: {mood}")

    def force_mood(self, mood: str):
        """Alias for set_mood — force-set the mood immediately."""
        self.set_mood(mood)
