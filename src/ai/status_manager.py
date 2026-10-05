"""
Status and bio management module for the Eudora persona.

Features:
1. Algorithmic status generation based on mood, time of day, and activity
2. Monthly bio change (rotate through persona-appropriate bios)
3. Status mode management (online, idle, dnd)
4. Time-of-day aware status (morning, afternoon, evening, night)
5. Activity-based status (in VC, chatting, quiet)

Algorithmic approach:
- Status updates based on mood + time of day + activity
- Bio changes monthly (rotate through a list)
- Status mode: idle when quiet, dnd when annoyed, online when active
- Don't change status too often (cooldown: 30 min)
"""
import time
import random
from typing import Optional
from loguru import logger
import discord


# ── Status presets by mood ───────────────────────────────────────────────────

MOOD_STATUSES = {
    "flat": ["existing", "just here", "vibing"],
    "bored": ["so bored innit", "nothing to do", "bored af"],
    "giddy": ["feeling good today", "in a good mood", "happy rn"],
    "hyped": ["let's gooo", "hyped rn", "energy high"],
    "chill": ["vibing rn", "chilling", "just vibing"],
    "playful": ["in a silly mood", "feeling mischievous", "silly mode"],
    "proud": ["nailed it", "feeling accomplished", "did a thing"],
    "annoyed": ["bruv allow it", "not in the mood", "ugh"],
    "sad": ["not feeling great", "having a rough one", "down rn"],
    "angry": ["don't even", "mad rn", "not now"],
    "tired": ["so tired rn", "exhausted", "need sleep"],
    "curious": ["thinking about stuff", "wondering...", "curious rn"],
    "nostalgic": ["reminiscing", "thinking about old times", "memories"],
    "lonely": ["anyone up?", "feeling alone", "wish someone was here"],
    "excited": ["can't wait fr", "so excited", "looking forward to something"],
}

# ── Time-of-day statuses ─────────────────────────────────────────────────────

TIME_STATUSES = {
    "morning": ["morning coffee", "just woke up", "tea first", "rough morning"],
    "afternoon": ["afternoon vibes", "productive day", "getting things done"],
    "evening": ["evening chill", "winding down", "relaxing tonight"],
    "night": ["night owl", "can't sleep", "late night vibes", "still up"],
    "late_night": ["3am thoughts", "should be sleeping", "insomnia hits"],
}

# ── Bio rotation (monthly change) ────────────────────────────────────────────

BIO_ROTATION = [
    "22 | london | half french, fully chaotic | art student who procrastinates | lo-fi & coffee addict | je ne sais quoi but i try",
    "22 | london | art student | half french, half confused | coffee dependent | vintage shopping enthusiast",
    "london based | 22 | art & lo-fi | french-british mess | probably procrastinating rn | c'est la vie",
    "22 | uk | art student | half french, fully tired | lo-fi vibes & coffee | existentially chilling",
    "london | 22 | art school dropout energy | french-british | coffee > people | voilà",
    "22 | london | je ne sais quoi | art student | lo-fi & vintage | probably in a cafe rn",
]

# ── Status mode mapping ──────────────────────────────────────────────────────

MOOD_TO_STATUS_MODE = {
    "flat": discord.Status.online,
    "bored": discord.Status.idle,
    "giddy": discord.Status.online,
    "hyped": discord.Status.online,
    "chill": discord.Status.online,
    "playful": discord.Status.online,
    "proud": discord.Status.online,
    "annoyed": discord.Status.dnd,
    "sad": discord.Status.idle,
    "angry": discord.Status.dnd,
    "tired": discord.Status.idle,
    "curious": discord.Status.online,
    "nostalgic": discord.Status.idle,
    "lonely": discord.Status.online,
    "excited": discord.Status.online,
}


def get_time_of_day() -> str:
    """
    Algorithmically determine the time of day.
    Based on UTC hour (Discord uses UTC internally).
    """
    import datetime
    hour = datetime.datetime.utcnow().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    elif 22 <= hour < 24 or 0 <= hour < 2:
        return "night"
    else:
        return "late_night"


def generate_status(mood: str, time_of_day: Optional[str] = None) -> str:
    """
    Algorithmically generate a custom status based on mood and time of day.

    60% chance: mood-based status
    40% chance: time-of-day status
    """
    if time_of_day is None:
        time_of_day = get_time_of_day()

    # Persona-flavored statuses sometimes win so each account's presence
    # reads like THAT person, not a shared generic vibe
    try:
        from ..persona.runtime import active as _active_persona
        _ps = _active_persona().statuses
        if _ps and random.random() < 0.3:
            return random.choice(_ps)
    except Exception:
        pass

    if random.random() < 0.6:
        # Mood-based status
        statuses = MOOD_STATUSES.get(mood, MOOD_STATUSES["chill"])
        return random.choice(statuses)
    else:
        # Time-of-day status
        statuses = TIME_STATUSES.get(time_of_day, TIME_STATUSES["evening"])
        return random.choice(statuses)


def get_status_mode(mood: str) -> discord.Status:
    """
    Algorithmically determine the Discord status mode (online/idle/dnd) based on mood.
    """
    return MOOD_TO_STATUS_MODE.get(mood, discord.Status.online)


def get_monthly_bio(month_index: int) -> str:
    """
    Get the bio for a specific month (rotates monthly).
    month_index: 0-11 (January = 0, December = 11)
    """
    return BIO_ROTATION[month_index % len(BIO_ROTATION)]


def get_current_month_bio() -> str:
    """Get the bio for the current month."""
    import datetime
    month = datetime.datetime.utcnow().month - 1  # 0-indexed
    return get_monthly_bio(month)


class StatusManager:
    """
    Manages Discord status and bio updates.

    Algorithmic behavior:
    - Status updates: every 30 minutes (cooldown)
    - Bio updates: monthly (checked daily)
    - Status mode: based on mood
    - Don't update if status hasn't changed
    """

    def __init__(self):
        self._last_status_update: float = 0
        self._last_bio_update: float = 0
        self._current_status: str = ""
        self._current_bio: str = ""
        self._status_cooldown_s = 1800  # 30 minutes
        self._bio_check_interval_s = 86400  # 24 hours (check daily)

    def should_update_status(self) -> bool:
        """Check if status should be updated (cooldown expired)."""
        return (time.time() - self._last_status_update) > self._status_cooldown_s

    def should_update_bio(self) -> bool:
        """Check if bio should be checked for monthly update."""
        return (time.time() - self._last_bio_update) > self._bio_check_interval_s

    async def update_status(self, client: discord.Client, mood: str):
        """
        Update the Discord custom status and presence mode.
        """
        if not self.should_update_status():
            return

        status_text = generate_status(mood)
        status_mode = get_status_mode(mood)

        # Don't update if same as current
        if status_text == self._current_status:
            return

        try:
            activity = discord.CustomActivity(name=status_text)
            await client.change_presence(activity=activity, status=status_mode)
            self._current_status = status_text
            self._last_status_update = time.time()
            logger.info(f"Status updated: '{status_text}' ({status_mode})")
        except Exception as e:
            logger.warning(f"Status update failed: {e}")

    async def update_bio(self, client: discord.Client):
        """
        Update the Discord bio. Eudora keeps her monthly rotation; other
        personas hold their pinned profile bio (rotating generic bios onto
        a different person would break identity).
        """
        if not self.should_update_bio():
            return

        try:
            from ..persona.runtime import active as _active_persona
            _p = _active_persona()
        except Exception:
            _p = None
        if _p is not None and _p.id != "eudora":
            new_bio = _p.bio or ""
        else:
            new_bio = get_current_month_bio()

        # Don't update if same as current
        if new_bio == self._current_bio:
            self._last_bio_update = time.time()
            return

        try:
            await client.user.edit(bio=new_bio)
            self._current_bio = new_bio
            self._last_bio_update = time.time()
            logger.info(f"Bio updated (monthly): {new_bio[:50]}...")
        except Exception as e:
            logger.warning(f"Bio update failed: {e}")

    def get_status(self) -> dict:
        """Get current status info for debugging."""
        return {
            "current_status": self._current_status,
            "current_bio": self._current_bio[:50] + "..." if self._current_bio else "",
            "last_status_update": self._last_status_update,
            "last_bio_update": self._last_bio_update,
        }


# Singleton instance
_status_manager = StatusManager()


def get_status_manager() -> StatusManager:
    """Get the global status manager."""
    return _status_manager
