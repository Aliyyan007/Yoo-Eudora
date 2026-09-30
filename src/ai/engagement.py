"""
Dynamic engagement/willingness engine.

Based on MaiBot's WillingManager pattern: the bot has a "willingness" score
per channel that determines how likely it is to respond. The score:
- Decays over time (linear decay every few seconds)
- Increases when the bot is mentioned or replied to
- Increases slightly when relevant topics are discussed
- Decreases after the bot sends a message (satisfaction)
- Is modulated by channel activity level (more active = lower willingness,
  less active = higher willingness — "low activity, speak more; high activity,
  speak less")

This creates natural conversation dynamics: the bot doesn't reply to every
message, but becomes more likely to chime in when conversation is relevant
or when chat has been quiet.
"""
import time
import threading
from collections import defaultdict, deque
from typing import Dict, Optional
from loguru import logger


class EngagementEngine:
    """Manages per-channel willingness scores for natural conversation participation."""

    def __init__(self):
        self._willingness: Dict[str, float] = defaultdict(float)
        self._channel_activity: Dict[str, Dict] = {}  # ch_id -> {start_time, msg_count}
        self._last_reply_time: Dict[str, float] = {}
        self._lock = threading.Lock()

        # Tunable parameters
        self.MAX_WILLING = 5.0
        self.DECAY_RATE = 0.15          # per 3s tick
        self.DECAY_INTERVAL = 3.0       # seconds between decay ticks
        self.MENTION_BOOST = 3.0        # willingness gained from mention
        self.REPLY_BOOST = 2.5          # willingness gained from reply to us
        self.NAME_BOOST = 1.0           # willingness gained from name mention
        self.TOPIC_BOOST = 0.3          # willingness gained from relevant topic
        self.GREETING_BOOST = 2.0       # willingness gained from greeting
        self.SENT_PENALTY = 1.5         # willingness lost after sending
        self.NO_REPLY_PENALTY = 0.2     # willingness lost when we skip

        # Activity tracking
        self.ACTIVITY_WINDOW = 1800     # 30 min rolling window
        self.ACTIVITY_MAX = 50          # max messages in window

        self._started = False
        self._decay_task = None

    def start(self):
        """Start the background decay loop."""
        if not self._started:
            import asyncio
            self._started = True
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._decay_loop())
            except RuntimeError:
                # No event loop yet — will be started later
                self._started = False

    async def _decay_loop(self):
        """Background task that decays willingness over time."""
        import asyncio
        while True:
            await asyncio.sleep(self.DECAY_INTERVAL)
            with self._lock:
                now = time.time()
                for ch_id in list(self._willingness.keys()):
                    current = self._willingness[ch_id]
                    self._willingness[ch_id] = max(0.0, current - self.DECAY_RATE)

    def update_activity(self, ch_id: str):
        """Track message activity in a channel for activity-based modulation."""
        with self._lock:
            now = time.time()
            if ch_id not in self._channel_activity:
                self._channel_activity[ch_id] = {
                    'start_time': now,
                    'msg_count': 1
                }
            else:
                activity = self._channel_activity[ch_id]
                # Reset window if expired
                if now - activity['start_time'] > self.ACTIVITY_WINDOW:
                    activity['start_time'] = now
                    activity['msg_count'] = 1
                else:
                    activity['msg_count'] = min(activity['msg_count'] + 1, self.ACTIVITY_MAX)

    def get_activity_level(self, ch_id: str) -> float:
        """Return normalized activity level (0.0 = dead, 1.0 = very active)."""
        with self._lock:
            activity = self._channel_activity.get(ch_id)
            if not activity:
                return 0.0
            return min(activity['msg_count'] / self.ACTIVITY_MAX, 1.0)

    def boost(self, ch_id: str, amount: float, reason: str = ""):
        """Increase willingness for a channel."""
        with self._lock:
            current = self._willingness[ch_id]
            self._willingness[ch_id] = min(current + amount, self.MAX_WILLING)
            if reason:
                logger.debug(f"Engagement +{amount:.1f} for {ch_id} ({reason}) -> {self._willingness[ch_id]:.1f}")

    def penalize(self, ch_id: str, amount: float, reason: str = ""):
        """Decrease willingness for a channel."""
        with self._lock:
            current = self._willingness[ch_id]
            self._willingness[ch_id] = max(0.0, current - amount)
            if reason:
                logger.debug(f"Engagement -{amount:.1f} for {ch_id} ({reason}) -> {self._willingness[ch_id]:.1f}")

    def on_mention(self, ch_id: str):
        """Called when the bot is mentioned."""
        self.boost(ch_id, self.MENTION_BOOST, "mentioned")

    def on_reply_to_us(self, ch_id: str):
        """Called when someone replies to the bot's message."""
        self.boost(ch_id, self.REPLY_BOOST, "reply-to-us")

    def on_name_mentioned(self, ch_id: str):
        """Called when the bot's name is mentioned (without @)."""
        self.boost(ch_id, self.NAME_BOOST, "name-mentioned")

    def on_greeting(self, ch_id: str):
        """Called when a greeting is detected."""
        self.boost(ch_id, self.GREETING_BOOST, "greeting")

    def on_loneliness_detected(self, ch_id: str):
        """Called when someone is looking for chat partners ('anyone here?', 'someone talk')."""
        self.boost(ch_id, self.MAX_WILLING, "loneliness-detected")

    def on_topic_relevant(self, ch_id: str):
        """Called when a relevant topic is detected."""
        self.boost(ch_id, self.TOPIC_BOOST, "topic-relevant")

    def on_sent(self, ch_id: str):
        """Called after the bot sends a message."""
        self.penalize(ch_id, self.SENT_PENALTY, "sent-message")
        self._last_reply_time[ch_id] = time.time()

    def on_skipped(self, ch_id: str):
        """Called when the bot decides not to reply."""
        self.penalize(ch_id, self.NO_REPLY_PENALTY, "skipped")

    def get_willingness(self, ch_id: str) -> float:
        """Get the current willingness score for a channel."""
        with self._lock:
            return self._willingness.get(ch_id, 0.0)

    def should_engage(self, ch_id: str, threshold: float = 1.0) -> bool:
        """Check if willingness is high enough to engage."""
        return self.get_willingness(ch_id) >= threshold

    def get_engagement_probability(self, ch_id: str) -> float:
        """Get a 0-1 probability of engaging based on willingness and activity.

        High willingness + low activity = high probability
        Low willingness + high activity = low probability
        This implements the "speak more when quiet, speak less when busy" pattern.
        """
        willing = self.get_willingness(ch_id)
        activity = self.get_activity_level(ch_id)

        # Base probability from willingness (sigmoid)
        import math
        base_prob = 1.0 / (1.0 + math.exp(-willing + 1.0))

        # Modulate by activity: when chat is very active, reduce probability
        # (don't spam in busy conversations). When quiet, increase it.
        activity_modifier = 1.0 - (activity * 0.6)  # 0.4 to 1.0

        return min(base_prob * activity_modifier, 1.0)
