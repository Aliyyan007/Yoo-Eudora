"""
Engagement topic engine — kills the "same five openers every day" problem.

How engagement text used to work: one prompt ("say something random but
relatable") collapsed the model into its highest-probability small-talk
prior — rec-asks, weekend-plans asks, "anyone up" — and the only dedup
was a 12-item in-memory deque wiped on every restart.

New architecture — two-stage generation:
1. THIS module picks the topic *outside* the model: a (genre, angle) pair
   sampled from a large structured space (~18 genres x ~10 angles each),
   excluding anything recently used — persistent, survives redeploys.
2. reply.generate_proactive_message() is then told "write ONLY about this
   angle in your voice" — the model does wording, code does diversity.

Fallback pools (for when the LLM is down) are persona-voiced and served
through a per-channel shuffle-bag: every line is used once before any
repeats, instead of a 1-in-N resurface on every send.

Persistence: shared server_state KV ("engagement_topics") — D1 when
reachable, data/memory.json otherwise — the same pattern ping_state uses.
"""
import random
import threading
import time
from typing import Dict, List, Optional

from loguru import logger

from . import d1_memory as mem

_STATE_KEY = "engagement_topics"

# Freshness windows
_GENRE_REUSE_S = 48 * 3600      # a genre can't repeat in a channel for 48h
_ANGLE_REUSE_S = 72 * 3600      # an angle can't repeat globally for 72h
_TEXT_MEMORY_S = 12 * 3600      # recent texts injected as "don't repeat"
_CHANNEL_GENRE_CAP = 14
_CHANNEL_TEXT_CAP = 30
_ANGLE_CAP = 90
_PERSIST_DEBOUNCE_S = 45

# ── Genre space ──────────────────────────────────────────────────────────────
# Each genre: an instruction telling the model exactly the FORMAT to write,
# plus concrete angles. The model never picks the topic — it words ours.

GENRES: List[dict] = [
    {
        "key": "activity_share",
        "weight": 1.4,
        "instruction": "share what you're doing rn in one line — small and concrete, a statement not a question",
        "angles": [
            "something you're procrastinating on",
            "a song/playlist currently on repeat",
            "your nth coffee of the day",
            "being horizontal instead of productive",
            "a show you're half-watching",
            "scrolling for no reason",
            "your current snack situation",
            "tab-hoarding on your phone",
            "avoiding a task that takes 5 minutes",
            "your sleep schedule being cooked",
        ],
    },
    {
        "key": "petty_rant",
        "weight": 1.3,
        "instruction": "a tiny low-stakes gripe — one line, deadpan, not asking for engagement",
        "angles": [
            "the weather being indecisive",
            "wifi picking the worst time to lag",
            "someone ate the last of something",
            "the fridge having nothing good in it",
            "a phone battery that dies at 40%",
            "your bed being too comfortable to leave",
            "autocorrect betraying you",
            "the neighbour being loud at the wrong time",
            "losing a sock to the washing machine",
            "ads on everything now",
        ],
    },
    {
        "key": "would_you_rather",
        "weight": 1.0,
        "instruction": "a short either-or / would-you-rather the room can answer in a word or two — casual, not a formal poll",
        "angles": [
            "sweet or savoury for the rest of your life",
            "always early or always fashionably late",
            "no music for a month or no films for a month",
            "morning person or night owl",
            "summer or winter",
            "cats or dogs (keep it light)",
            "stay in or go out tonight",
            "text or voice call",
            "one big holiday or lots of small trips",
            "cooking or takeaway",
        ],
    },
    {
        "key": "hot_take",
        "weight": 0.9,
        "instruction": "a mild harmless hot take — food/sleep/habits, never politics, religion, or anything actually divisive",
        "angles": [
            "cereal being better dry than soggy",
            "naps being an elite activity",
            "staying in being better than most nights out",
            "cold leftover food being underrated",
            "early nights being secretly great",
            "the second film/episode often being mid",
            "breakfast food working at any hour",
            "group chats being mostly useless",
            "rain being good actually when you're inside",
            "putting your phone face down being a power move",
        ],
    },
    {
        "key": "nostalgia",
        "weight": 0.8,
        "instruction": "something you were just randomly reminded of — old game, song, show, school memory — one line",
        "angles": [
            "an old game you haven't thought about in years",
            "a song that takes you back",
            "a kids' show that was lowkey great",
            "an early-internet thing that vanished",
            "a school trip or school-canteen memory",
            "a toy/gadget everyone had",
            "an old phone you had",
            "a summer-holiday memory",
        ],
    },
    {
        "key": "channel_callback",
        "weight": 1.5,
        "instruction": "riff on something from the recent channel chat below — a follow-up thought, a joke, a 'wait what happened with that'. pick ONE thing, don't summarise",
        "needs_context": True,
        "angles": [
            "a funny line someone dropped earlier",
            "a topic that got dropped before it finished",
            "someone's plan — ask how it went",
            "the running bit/joke in this chat",
            "a thing someone said they were doing",
        ],
    },
    {
        "key": "observation",
        "weight": 1.1,
        "instruction": "a throwaway observation about life, people, or the internet — one line, slightly amused",
        "angles": [
            "how fast a week can disappear",
            "everyone being busy but doing what exactly",
            "how one good song fixes a whole mood",
            "the group chat being quiet at the same times always",
            "how everyone pretends to have a routine",
            "the weird comfort of the same few apps",
            "how time moves weird when you look at your phone",
            "people's typing energy when they're excited",
        ],
    },
    {
        "key": "rec_offer",
        "weight": 0.9,
        "instruction": "mention something YOU like rn — a song, food, a thing — as a statement, NOT asking anyone for recs back",
        "angles": [
            "a song you've had on loop",
            "a food you've been making too often",
            "a film/series you finally got round to",
            "a small thing that's been making your day",
            "a drink order you're loyal to",
            "a random hobby you're weirdly into rn",
            "a creator/channel you've been watching",
            "a place in your city you keep going back to",
        ],
    },
    {
        "key": "confession",
        "weight": 0.7,
        "instruction": "a small harmless confession — something you can't do, secretly like, or keep doing — one line",
        "angles": [
            "a basic skill you never learned",
            "a food everyone loves that you can't stand",
            "something you do that makes no sense",
            "a habit you can't quit",
            "something you pretended to know about for too long",
            "a thing you're weirdly bad at",
        ],
    },
    {
        "key": "time_gripe",
        "weight": 1.0,
        "instruction": "comment on the current time/day part ({tod}) — 'dragging', 'too early for this', 'nearly the weekend' energy. one line",
        "angles": [
            "the day dragging on",
            "it being too early/too late for anything useful",
            "the week being slow",
            "the evening disappearing too fast",
            "the weird dead hour of the day",
            "being nowhere near done with the day",
        ],
    },
    {
        "key": "poll_ish",
        "weight": 0.8,
        "instruction": "a casual one-word-answer question the room can pile onto — 'rate X out of 10', 'best X go', 'pick one: X'",
        "angles": [
            "best comfort food — go",
            "rate your day so far",
            "one song for the aux — pick one",
            "unpopular opinion: drop one",
            "best way to waste an evening — go",
            "describe your week in one word",
            "pick one superpower, keep it realistic",
            "best film to rewatch — one answer only",
        ],
    },
    {
        "key": "absurdist",
        "weight": 0.7,
        "instruction": "a silly hypothetical / absurdist what-if — keep it tiny and weird, one line",
        "angles": [
            "which animal would be worst at customer service",
            "if your sleep schedule had a boss fight",
            "which household object has the most attitude",
            "if your pet (or a pet you want) could text you",
            "the worst possible theme for a party",
            "if your search history could talk",
        ],
    },
    {
        "key": "plans_probe",
        "weight": 0.8,
        "instruction": "probe what people are up to — but a SPECIFIC angle, never 'any plans' verbatim. one line",
        "angles": [
            "tonight specifically — what's the move",
            "this weekend — anyone doing something actually fun",
            "the next meal — what everyone eating",
            "anyone got plans they're excited about",
            "what everyone's evening looks like",
            "anyone doing anything tomorrow or same old",
        ],
    },
    {
        "key": "game_check",
        "weight": 0.9,
        "instruction": "check what people are playing/watching/doing for entertainment rn — casual, one line",
        "angles": [
            "what everyone's playing rn",
            "anyone watching anything decent",
            "what's on everyone's screens",
            "anyone found a new thing to binge",
            "who's on what tonight",
        ],
    },
    {
        "key": "food_thought",
        "weight": 1.0,
        "instruction": "a food/coffee thought — craving, rating, deciding, or complaining about a meal decision. one line",
        "angles": [
            "what to eat rn — decision paralysis",
            "a craving that won't quit",
            "rating what you just ate",
            "coffee count getting silly",
            "cooking vs giving up and ordering",
            "breakfast/lunch/dinner being skipped or doubled",
            "a snack tier take",
            "being hungry but too lazy to fix it",
        ],
    },
    {
        "key": "bored_ping",
        "weight": 0.5,   # classic dead-chat call-out — keep rare, it's the most template-feeling
        "instruction": "call out that chat's gone quiet — but phrase it fresh, never the literal words 'dead chat' two days running. one line",
        "angles": [
            "the silence being loud rn",
            "everyone vanishing at once",
            "the chat flatlining",
            "talking to yourself in here apparently",
            "the tumbleweed moment",
            "everyone being mysteriously busy",
        ],
    },
    {
        "key": "weather_day",
        "weight": 0.6,
        "instruction": "a one-line weather/day comment — british weather energy, understated",
        "angles": [
            "the sky forgetting what it's doing",
            "it being proper grey out",
            "the wind having personal beef",
            "a surprise bit of sun",
            "it raining sideways",
            "it being dark stupidly early",
        ],
    },
    {
        "key": "sleepy",
        "weight": 0.7,
        "instruction": "being tired/sleepy in one line — universal, low effort, relatable",
        "angles": [
            "the 3pm crash hitting",
            "being awake but not functioning",
            "running on caffeine and spite",
            "needing a nap at a weird hour",
            "staying up too late last night catching up with you",
            "the eyes-closing-at-your-desk moment",
        ],
    },
]


# ── Persona-voiced fallback pools (LLM-down only) ────────────────────────────
# kind: "dead" = revive a silent channel, "chat" = spontaneous line in a
# slowing channel, "bump" = post-bump engagement. Served via shuffle-bag —
# every line used once before any repeats.

FALLBACK_LINES: Dict[str, Dict[str, List[str]]] = {
    "eudora": {
        "dead": [
            "bare quiet in here", "well this is dead innit", "oi anyone alive",
            "chat's asleep then", "ghost town vibes", "crickets innit",
            "this place flatlined", "all gone quiet", "someone say something",
            "dead o'clock apparently", "chat's doing that silent thing again",
            "anyone still here or", "it's giving empty room", "quiet quiet",
        ],
        "chat": [
            "what's everyone on about today", "yo what's the vibe",
            "thoughts on films rn", "coffee count of the day go",
            "slow day innit", "what we up to", "how's everyone's day going fr",
            "what's on rn", "talk to me people", "what are we all doing",
            "anything fun happening", "anyone been out today",
            "what's everyone doing tonight", "ngl i need entertainment",
        ],
        "bump": [
            "yo what's everyone up to", "what we talking about",
            "how's everyone doing today", "what's the vibe today",
            "anyone doing anything fun", "what's the move today",
            "chat what we on", "who's about rn", "talk to me innit",
            "what's happening people", "anyone up to much",
            "what's everyone been up to",
        ],
    },
    "isla": {
        "dead": [
            "hello?? anyone about", "it's gone proper quiet",
            "anyone awake or just me", "where did everyone go",
            "chat fell asleep i see", "hiii anyone there",
            "so quiet today honestly", "anyone about or am i talking to myself",
            "it's too quiet ngl", "wake up everyone",
            "everyone vanished then", "is anyone still here",
            "this is dead silent wow", "anyoneee",
        ],
        "chat": [
            "what's everyone up to today", "okay what did i miss",
            "tell me something good that happened today",
            "show me what you're working on", "anyone found a good song today",
            "what's the plan for today", "rate your day out of 10 go",
            "what are we all doing rn", "someone update me on life",
            "best thing that happened today go", "anyone got plans tonight",
            "what's everyone doing this evening", "gimme a life update",
            "what we all up to",
        ],
        "bump": [
            "what's everyone up to", "okay what did i miss",
            "hi hi what's going on", "what's everyone doing rn",
            "tell me your plans for today", "what's everyone been up to today",
            "anyone doing anything interesting", "what's the vibe today",
            "update me on your days", "what we all up to today",
            "anyone got anything fun planned", "what's happening today",
        ],
    },
    "rowan": {
        "dead": [
            "dead in here", "chat flatlined", "anyone alive or",
            "quiet server is quiet", "welp. dead", "echo echo",
            "it's just me and the void then", "crickets", "nobody home then",
            "channel's asleep", "peak activity right here", "well this is lively",
            "anyone still breathing in here", "chat's gone afk apparently",
        ],
        "chat": [
            "what's everyone playing rn", "anyone on for something",
            "what's the move", "anyone doing anything interesting",
            "what we on tonight", "anyone watching anything decent",
            "what's good tonight", "anyone up to anything",
            "new patch thoughts anyone", "what's everyone on",
            "who's about", "games tonight or what", "what are we doing",
            "anything happening tonight",
        ],
        "bump": [
            "what's everyone playing", "what's the move tonight",
            "anyone on for games", "what we doing tonight",
            "who's about rn", "what's everyone on today",
            "anyone up to anything tonight", "what's happening",
            "games or nah", "what's everyone up to tonight",
            "who's online then", "anyone doing owt interesting",
        ],
    },
}
_DEFAULT_POOL = FALLBACK_LINES["eudora"]


# ── Persistent state ─────────────────────────────────────────────────────────
# {"channels": {ch_id: {"genres": [[key,ts]...], "texts": [[text,ts]...]}},
#  "angles": [[angle_key,ts]...]}

_state: Optional[dict] = None
_persist_thread: Optional[threading.Thread] = None
_persist_pending = False
_lock = threading.Lock()


def _load() -> dict:
    """Lazy-load persistent state on first use (callers run off the event
    loop — generate paths are all executor-wrapped)."""
    global _state
    if _state is not None:
        return _state
    try:
        data = mem.load_server_state(_STATE_KEY)
    except Exception:
        data = None
    if isinstance(data, dict) and isinstance(data.get("channels"), dict):
        _state = data
    else:
        _state = {"channels": {}, "angles": []}
    return _state


def _export() -> dict:
    st = _load()
    return {"channels": st["channels"], "angles": st["angles"]}


def _persist_worker():
    global _persist_pending
    try:
        mem.save_server_state(_STATE_KEY, _export())
    except Exception as e:
        logger.debug(f"engagement_topics persist failed: {e}")
    with _lock:
        if _persist_pending:
            _persist_pending = False
            _schedule_persist()


def _schedule_persist():
    """Debounced background persist — safe to call from any thread: the
    D1/JSON write happens on a daemon thread so it never blocks a loop."""
    global _persist_thread, _persist_pending
    with _lock:
        t = _persist_thread
        if t is not None and t.is_alive():
            _persist_pending = True
            return
        _persist_thread = threading.Thread(target=_persist_worker, daemon=True)
        _persist_thread.start()


def _chan(ch_id: str) -> dict:
    st = _load()
    ch = st["channels"].setdefault(str(ch_id), {"genres": [], "texts": []})
    return ch


def _prune(now: float):
    st = _load()
    for ch_id in list(st["channels"]):
        ch = st["channels"][ch_id]
        ch["genres"] = [(g, t) for g, t in ch.get("genres", [])
                        if now - float(t) < _GENRE_REUSE_S][-_CHANNEL_GENRE_CAP:]
        ch["texts"] = [(x, t) for x, t in ch.get("texts", [])
                       if now - float(t) < 7 * 86400][-_CHANNEL_TEXT_CAP:]
        if not ch["genres"] and not ch["texts"]:
            del st["channels"][ch_id]
    st["angles"] = [(a, t) for a, t in st["angles"]
                    if now - float(t) < _ANGLE_REUSE_S][-_ANGLE_CAP:]


# ── Topic picking ────────────────────────────────────────────────────────────

def pick_angle(ch_id: str, persona_id: str = "",
               interests: Optional[List[str]] = None,
               has_context: bool = False,
               tod: str = "") -> dict:
    """Pick a fresh (genre, angle) pair for this channel.

    Exclusions (persistent — survive redeploys):
    - genre used in THIS channel within 48h
    - genre used in this channel's last 3 sends regardless of age
    - angle used ANYWHERE globally within 72h
    - context-required genres when there's no recent chat to riff on
    """
    now = time.time()
    _load()
    _prune(now)
    ch = _chan(ch_id)

    used_genres = {g for g, _ in ch.get("genres", [])}
    recent3 = {g for g, _ in ch.get("genres", [])[-3:]}
    used_angles = {a for a, _ in _state.get("angles", [])}

    def _eligible(g: dict, relax: bool = False) -> bool:
        if g.get("needs_context") and not has_context:
            return False
        if g["key"] in recent3:
            return False
        if not relax and g["key"] in used_genres:
            return False
        return True

    def _fresh_angles(g: dict) -> List[str]:
        out = [a for a in g["angles"] if a not in used_angles]
        return out or list(g["angles"])

    for relax in (False, True):
        pool = [g for g in GENRES if _eligible(g, relax)]
        if not pool:
            continue
        weights = [g.get("weight", 1.0) for g in pool]
        genre = random.choices(pool, weights=weights, k=1)[0]
        angle = random.choice(_fresh_angles(genre))
        prompt_line = genre["instruction"]
        if "{tod}" in prompt_line:
            prompt_line = prompt_line.replace("{tod}", tod or "this time of day")
        if "{angle}" in prompt_line:
            prompt_line = prompt_line.replace("{angle}", angle)
        else:
            prompt_line = f"{prompt_line} — angle: {angle}"
        return {
            "genre": genre["key"],
            "angle": angle,
            "prompt_line": prompt_line,
        }
    # Total exhaustion (~impossible): allow anything
    genre = random.choice(GENRES)
    return {"genre": genre["key"], "angle": random.choice(genre["angles"]),
            "prompt_line": genre["instruction"]}


def record_send(ch_id: str, genre: str, angle: str, text: str):
    """Record a sent engagement message — genre+angle for diversity,
    raw text for near-duplicate checking. Call AFTER the send lands."""
    try:
        now = time.time()
        ch = _chan(ch_id)
        if genre:
            ch.setdefault("genres", []).append((genre, now))
            ch["genres"] = ch["genres"][-_CHANNEL_GENRE_CAP:]
        if angle:
            _state.setdefault("angles", []).append((angle, now))
            _state["angles"] = _state["angles"][-_ANGLE_CAP:]
        if text:
            ch.setdefault("texts", []).append((text, now))
            ch["texts"] = ch["texts"][-_CHANNEL_TEXT_CAP:]
        _schedule_persist()
    except Exception as e:
        logger.debug(f"engagement_topics record failed: {e}")


def recent_texts(ch_id: str, limit: int = 12) -> List[str]:
    """Recent engagement texts for this channel (fresh < 7d) — for prompt
    injection and near-dup checks."""
    try:
        ch = _load()["channels"].get(str(ch_id)) or {}
        now = time.time()
        return [x for x, t in ch.get("texts", []) if now - float(t) < 7 * 86400][-limit:]
    except Exception:
        return []


def recent_genres(ch_id: str, limit: int = 5) -> List[str]:
    ch = _load()["channels"].get(str(ch_id)) or {}
    return [g for g, _ in ch.get("genres", [])][-limit:]


def global_texts(limit: int = 30) -> List[str]:
    """Recent engagement texts across ALL channels — the persona shouldn't
    repeat itself even in different rooms."""
    out: List[str] = []
    try:
        now = time.time()
        for ch in _load()["channels"].values():
            out.extend(x for x, t in ch.get("texts", []) if now - float(t) < 7 * 86400)
    except Exception:
        pass
    return out[-limit:]


# ── Shuffle-bag fallback pools ────────────────────────────────────────────────

_bags: Dict[str, List[str]] = {}   # "{persona}:{kind}:{ch_id}" -> remaining lines


def fallback_line(persona_id: str, kind: str, ch_id: str) -> str:
    """Pick a canned line via per-channel shuffle-bag: every line in the
    pool is used once before any can repeat (then the bag reshuffles).
    kind: 'dead' | 'chat' | 'bump'."""
    pools = FALLBACK_LINES.get(persona_id) or _DEFAULT_POOL
    pool = pools.get(kind) or pools.get("chat") or _DEFAULT_POOL["chat"]
    key = f"{persona_id}:{kind}:{ch_id}"
    bag = _bags.get(key)
    if not bag:
        bag = list(pool)
        random.shuffle(bag)
        _bags[key] = bag
    line = bag.pop()
    return line
