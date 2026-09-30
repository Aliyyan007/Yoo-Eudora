"""Action-worker integration tests — prefilter, heuristic classify, registry
pruning, real dispatch on fake Discord objects, gating, end-to-end worker run
(LLM layer mocked), ToolContext contract."""
import asyncio
import os
import sys
import time
from collections import deque
from types import SimpleNamespace

os.environ["OWNER_IDS"] = "999"          # test owner id — loaded at import
os.environ.setdefault("ACTIONS_ENABLED", "1")

sys.path.insert(0, os.path.dirname(__file__))

from src.actions.intent import looks_like_action
from src.actions.router import classify, Route, ROUTE_ACTION, ROUTE_CHAT
from src.actions.tools.registry import build_tools, dispatch, get_tool_names
from src.actions.tools.context import ToolContext
import src.actions.agent as agent_mod
import src.actions.router as router_mod
from src.actions.worker import get_action_worker

PASS = []


def ok(name, cond, detail=""):
    PASS.append((name, cond))
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail if not cond else ''}")


# ── fakes ──────────────────────────────────────────────────────────────
import discord


class FakeChannel(discord.TextChannel):
    """Passes the vendored tools' isinstance checks without real init."""

    def __init__(self, cid=777, name="general"):
        self.id = cid
        self.name = name
        self.sent = []
        self.guild = None

    async def send(self, content=None, **kw):
        self.sent.append(content)
        return SimpleNamespace(id=len(self.sent), content=content)

    def typing(self):
        class T:
            async def __aenter__(self): return self
            async def __aexit__(self, *a): return False
        return T()


class FakeMember:
    def __init__(self, uid, name):
        self.id = uid
        self.name = name
        self.display_name = name
        self.bot = False
        self.mention = f"<@{uid}>"
        self.status = SimpleNamespace(name="online")
        self.voice = None
        self.roles = []
        self.activities = []


class FakeGuild:
    def __init__(self, channels=(), members=()):
        self.id = 1
        self.name = "test"
        self._channels = {c.id: c for c in channels}
        self.channels = list(channels)
        self.members = list(members)
        self.roles = []
        self.stickers = []
        self.emojis = []
        self.me = FakeMember(1, "eudora")

    def get_channel(self, cid):
        return self._channels.get(cid)

    def get_member(self, uid):
        return next((m for m in self.members if m.id == uid), None)


class FakeBot:
    def __init__(self, channels=()):
        self.user = SimpleNamespace(id=1, bot=False)
        self._channels = {c.id: c for c in channels}
        self.engager_leaves = deque(maxlen=1000)

    def get_channel(self, cid):
        return self._channels.get(cid)

    async def fetch_channel(self, cid):
        return self._channels.get(cid)

    async def fetch_user(self, uid):
        return FakeMember(uid, f"u{uid}")


def msg(author_id, content, *, mention=None, channel=None, reference=None):
    ch = channel or FakeChannel()
    return SimpleNamespace(
        author=SimpleNamespace(id=author_id, name=f"u{author_id}"),
        channel=ch, content=content, clean_content=content,
        mentions=[mention] if mention else [], reference=reference,
        created_at=SimpleNamespace(timestamp=lambda: time.time()),
    )


async def main():
    # ── 1. prefilter ────────────────────────────────────────────────────
    print("=== looks_like_action prefilter ===")
    cases = {
        "eudora ping @john in general": True,
        "send him a dm saying hi": True,
        "react to his last message": True,
        "change your status to dnd": True,
        "schedule a message in 10 mins": True,
        "delete your last message": True,
        "bump the server": True,
        "im coding rn": False,
        "sounds epic fr": False,
        "what are you up to": False,
        "lol nice": False,
        "": False,
    }
    for t, want in cases.items():
        ok(f"prefilter {t!r}", looks_like_action(t) == want)

    # ── 2. zero-token heuristic classify ───────────────────────────────
    print("\n=== router.classify (heuristic, no LLM) ===")
    r = classify("ping @john in general")
    ok("classify ping -> ACTION", r is not None and r.kind == ROUTE_ACTION,
       f"got {r}")
    r = classify("how are you doing today")
    ok("classify chat -> CHAT", r is not None and r.kind == ROUTE_CHAT,
       f"got {r}")
    r = classify("send a message to #general saying hi")
    ok("classify send -> ACTION", r is not None and r.kind == ROUTE_ACTION,
       f"got {r}")

    # ── 3. registry pruning ────────────────────────────────────────────
    print("\n=== registry ===")
    names = set(get_tool_names())
    for dead in ("join_voice", "leave_voice", "say_in_vc", "mute_self",
                 "deafen_self", "move_voice", "speak_in_stage"):
        ok(f"{dead} absent from dispatch", dead not in names)
    for alive in ("send_message", "send_dm", "react_to_recent", "bump_all",
                  "schedule_message", "get_user_voice_state", "list_voice_channels"):
        ok(f"{alive} in dispatch", alive in names)
    tools = build_tools("command", {"messaging"})
    tool_names = {s["function"]["name"] for s in tools}
    ok("categories subset has send_message", "send_message" in tool_names)
    ok("categories subset keeps resolvers", "search_channels" in tool_names)
    ok("no voice actions in subset", not {"join_voice", "leave_voice"} & tool_names)

    # ── 4. real dispatch on fakes ──────────────────────────────────────
    print("\n=== dispatch ===")
    ch = FakeChannel()
    g = FakeGuild(channels=[ch])
    bot = FakeBot(channels=[ch])
    ctx = ToolContext(bot=bot, guild=g, current_channel_id=ch.id, author_id=999)
    res = await dispatch(ctx, "send_message", {"channel_query": "here", "content": "yo"})
    ok("send_message dispatched", '"ok": true' in res and ch.sent == ["yo"], res[:80])
    ok("did_send set", ctx.did_send is True)
    res = await dispatch(ctx, "not_a_tool", {})
    ok("unknown tool -> error", '"error"' in res, res[:80])
    res = await dispatch(ctx, "send_message<|channel|>", {"channel_query": "here", "content": "x"})
    ok("hallucinated suffix stripped", '"ok": true' in res, res[:80])

    ctx2 = ToolContext(bot=bot)
    try:
        ctx2.require_guild()
        ok("require_guild raises", False)
    except RuntimeError:
        ok("require_guild raises", True)

    # ── 5. worker gate (text path) ─────────────────────────────────────
    print("\n=== _wants_action_worker gating ===")
    from src.discord_client import AIPersonaClient
    cl = SimpleNamespace(
        _actions_enabled=True, _actions_owner_only=True,
        sticky_until={}, user=SimpleNamespace(id=1),
    )
    gate = AIPersonaClient._wants_action_worker

    owner_msg = msg(999, "ping @john", mention=cl.user)
    ok("owner+mention -> True", gate(cl, owner_msg, owner_msg.content, False))
    pleb_msg = msg(5, "ping @john", mention=cl.user)
    ok("non-owner+mention -> False", gate(cl, pleb_msg, pleb_msg.content, False) is False)
    plain_owner = msg(999, "ping @john")  # no mention/dm/reference
    ok("owner undirected -> False", gate(cl, plain_owner, plain_owner.content, False) is False)
    cl.sticky_until[str(plain_owner.channel.id)] = time.time() + 60
    ok("owner sticky -> True", gate(cl, plain_owner, plain_owner.content, False))
    chat_msg = msg(999, "how are you", mention=cl.user)
    ok("owner chat (no action) -> False", gate(cl, chat_msg, chat_msg.content, False) is False)
    cl._actions_owner_only = False
    ok("non-owner directed (open mode) -> True", gate(cl, pleb_msg, pleb_msg.content, False))
    cl._actions_enabled = False
    ok("master switch off -> False", gate(cl, owner_msg, owner_msg.content, False) is False)
    cl._actions_enabled = True
    cl._actions_owner_only = True

    # ── 6. end-to-end worker run (LLM mocked, real dispatch) ──────────
    print("\n=== ActionWorker.run_request ===")
    ch2 = FakeChannel()
    g2 = FakeGuild(channels=[ch2])
    bot2 = FakeBot(channels=[ch2])

    # mock router: always ACTION w/ messaging cats
    async def fake_classify(text):
        return Route(kind=ROUTE_ACTION, categories={"messaging"}, via="test")
    orig_classify = router_mod.classify_llm
    router_mod.classify_llm = fake_classify

    # mock Agent: executes send_message through the real dispatch spy
    class FakeAgent:
        def __init__(self, ctx):
            self.ctx = ctx
        async def run(self, text, **kw):
            return await agent_mod.dispatch(
                self.ctx, "send_message",
                {"channel_query": "here", "content": "pinged him"}) and "(done)"
    orig_agent = agent_mod.Agent
    agent_mod.Agent = FakeAgent
    try:
        worker = get_action_worker()
        note = await worker.run_request(
            bot2, guild=g2, channel=ch2, author_id=999,
            author_name="boss", text="ping john in general")
        ok("worker note has ACTION PERFORMED",
           note is not None and "ACTION PERFORMED" in note, str(note))
        ok("worker note has verb", "sent a message" in (note or ""), str(note))
        ok("tool actually sent", ch2.sent == ["pinged him"], str(ch2.sent))

        # CHAT route -> None
        async def chat_classify(text):
            return Route(kind=ROUTE_CHAT, via="test")
        router_mod.classify_llm = chat_classify
        note = await worker.run_request(
            bot2, guild=g2, channel=ch2, author_id=999,
            author_name="boss", text="how are you")
        ok("chat route -> None", note is None, str(note))

        # router failure -> None
        async def boom(text):
            raise RuntimeError("down")
        router_mod.classify_llm = boom
        note = await worker.run_request(
            bot2, guild=g2, channel=ch2, author_id=999,
            author_name="boss", text="ping john")
        ok("router crash -> None", note is None, str(note))
    finally:
        router_mod.classify_llm = orig_classify
        agent_mod.Agent = orig_agent

    fails = [n for n, c in PASS if not c]
    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + str(fails)}")
    sys.exit(1 if fails else 0)


asyncio.run(main())
