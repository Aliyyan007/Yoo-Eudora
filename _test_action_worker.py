"""Offline test harness for the action-worker integration.

No Discord, no Groq — the GroqPool is replaced with a scripted fake, and
the vendored tool layer runs against fake discord objects.

Verifies:
1. Heuristic classifier routes action vs chat correctly.
2. Text request -> agent tool call -> real tool execution -> silent
   (no "done"/agent reply is ever sent to a channel).
3. Human-like delay is applied before acting.
4. Non-owner permission gate denies dangerous tools + strips ping_roles.
5. Voice path: leave-VC short-circuits before the worker; action
   transcripts reach the worker with the VC as channel context.
6. Failure/CHAT requests return False -> normal reply path continues.
"""
import asyncio
import json
import os
import sys
import time

os.environ.setdefault("DISCORD_TOKEN", "test-dummy")
os.environ.setdefault("GROQ_API_KEY", "test-dummy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai.action_bridge import ActionWorker
from src.ai.commands import detect_command

import src.action_engine.core.groq_pool as gp
import src.action_engine.tools.messaging as ae_messaging
from src.action_engine.ai import router


# ── Fake discord objects ──────────────────────────────────────────────────
class FakeRole:
    def __init__(self, rid, name):
        self.id = rid; self.name = name
        self.mention = f"<@&{rid}>"


class FakeMember:
    def __init__(self, uid, name):
        self.id = uid; self.name = name; self.display_name = name
        self.nick = None; self.bot = False
        self.mention = f"<@{uid}>"


class FakeSentMsg:
    def __init__(self, mid): self.id = mid


class _Typing:
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class FakeChannel:
    def __init__(self, cid, name, guild=None):
        self.id = cid; self.name = name; self.guild = guild
        self.sent = []
        self.topic = ""
    async def send(self, content=None, **kw):
        self.sent.append((content, kw))
        return FakeSentMsg(9000 + len(self.sent))
    def typing(self): return _Typing()
    async def fetch_message(self, mid): return None


class FakeVoiceChannel(FakeChannel):
    pass


class FakeGuild:
    def __init__(self, gid, name="test"):
        self.id = gid; self.name = name
        self.channels = []
        self.members = []
        self.roles = []
        self.me = FakeMember(1, "eudora")
    def get_channel(self, cid):
        return next((c for c in self.channels if c.id == cid), None)
    def get_member(self, uid):
        return next((m for m in self.members if m.id == uid), None)


class FakeClient:
    def __init__(self):
        self.guilds = []
        self.voice_clients = []
        self.user = FakeMember(1, "eudora")
        self.voice_manager = None
        self.vc_manager = None
    def get_channel(self, cid):
        for g in self.guilds:
            c = g.get_channel(cid)
            if c: return c
        return None
    async def fetch_channel(self, cid): return None
    def get_user(self, uid): return None


class FakeMessage:
    def __init__(self, content, author, channel):
        self.clean_content = content; self.content = content
        self.author = author; self.channel = channel
        self.guild = channel.guild; self.mentions = []


# ── Fake Groq pool ────────────────────────────────────────────────────────
class _Fn:
    def __init__(self, name, args):
        self.name = name; self.arguments = json.dumps(args)


class _TC:
    def __init__(self, i, name, args):
        self.id = f"tc{i}"; self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content=None, tcs=None):
        self.content = content; self.tool_calls = tcs


class _Choice:
    def __init__(self, msg): self.message = msg


class _Resp:
    def __init__(self, msg):
        self.choices = [_Choice(msg)]; self.usage = None


class FakePool:
    """Scripted chat() — returns canned responses in order."""
    def __init__(self, script):
        self.script = list(script); self.calls = []
    async def chat(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0) if self.script else ("content", "(done)")
        kind = step[0]
        if kind == "tools":
            return _Resp(_Msg(tcs=[_TC(i, n, a) for i, (n, a) in enumerate(step[1])]))
        return _Resp(_Msg(content=step[1]))


def make_world():
    guild = FakeGuild(100)
    ch = FakeChannel(200, "general", guild)
    guild.channels.append(ch)
    author = FakeMember(42, "tester")
    guild.members.append(author)
    client = FakeClient()
    client.guilds.append(guild)
    return client, guild, ch, author


async def main():
    passed = []
    def check(name, cond, extra=""):
        passed.append((name, bool(cond)))
        print(f"  {'PASS' if cond else 'FAIL'}  {name} {extra}")

    print("== 1. classifier ==")
    r = router.classify("hey send a message to #general saying hello")
    check("send-msg -> action", r and r.kind == "action" and "messaging" in r.categories, f"{r}")
    r = router.classify("ping sarah for me")
    check("ping -> action", r and r.kind == "action")
    r = router.classify("how was your day")
    check("chat -> chat", r and r.kind == "chat")
    check("ambiguous -> None", router.classify("the channel") is None)

    print("== 2. native leave-vc never reaches worker ==")
    cmd, _ = detect_command("eudora leave the vc")
    check("detect_command catches leave_vc", cmd == "leave_vc")

    def set_script(worker, script):
        """Swap the pool the worker's agent will use next (agent caches the
        pool object at construction)."""
        gp._pool = FakePool(script)
        worker._agent = None

    print("== 3. text action end-to-end (silent) ==")
    gp._pool = FakePool([
        ("tools", [("send_message", {"channel_query": "here", "content": "hello from worker"})]),
        ("content", "done"),
    ])
    client, guild, ch, author = make_world()
    # the vendored resolver insists on real discord types — patch it to the
    # ctx's current channel so the fake one is used
    async def _fake_text_ch(ctx, q): return ctx.get_current_channel()
    ae_messaging._resolve_text_channel = _fake_text_ch

    worker = ActionWorker(client)
    msg = FakeMessage("eudora send a message saying hello", author, ch)
    t0 = time.monotonic()
    note = await worker.try_handle_text(msg, msg.clean_content)
    dt = time.monotonic() - t0
    check("worker handled (note)", isinstance(note, str) and "sent" in note, f"note={note!r}")
    check("action content sent once", len(ch.sent) == 1 and ch.sent[0][0] == "hello from worker", f"{ch.sent}")
    check("NO 'done'/reply sent", all(c[0] != "done" for c in ch.sent))
    check(f"human delay applied ({dt:.1f}s)", dt >= 1.5)

    print("== 4. chat request -> False (falls through) ==")
    gp._pool = FakePool([("content", "chat")])
    worker2 = ActionWorker(client)
    ch.sent.clear()
    msg2 = FakeMessage("how are you", author, ch)
    ok = await worker2.try_handle_text(msg2, msg2.clean_content)
    check("chat -> None", ok is None)
    check("nothing sent", not ch.sent)

    print("== 5. non-owner permission gate ==")
    ctx = worker._ctx_cls(bot=client, guild=guild, current_channel_id=ch.id, author_id=author.id)
    ctx.performed = []
    a, _ = worker._check_allowed(ctx, "delete_message", {})
    check("non-owner delete_message denied", not a)
    a, _ = worker._check_allowed(ctx, "send_dm", {})
    check("non-owner send_dm denied", not a)
    a, args = worker._check_allowed(ctx, "send_message", {"ping_roles": ["everyone"], "ping_users": ["x"] * 9})
    check("send_message allowed", a)
    check("ping_roles stripped", args["ping_roles"] == [])
    check("ping_users capped at 5", len(args["ping_users"]) == 5)
    a, _ = worker._check_allowed(ctx, "list_channels", {})
    check("read tool allowed", a)

    # scripted agent tries a denied tool on behalf of a non-owner
    set_script(worker, [
        ("tools", [("send_dm", {"user_query": "bob", "content": "spam"})]),
        ("content", "done"),
    ])
    ch.sent.clear()
    msg3 = FakeMessage("dm bob saying hi", author, ch)
    ok = await worker.try_handle_text(msg3, msg3.clean_content)
    # denied tool never ran -> performed empty -> None -> normal reply continues
    check("denied run -> None (falls back to reply)", ok is None)

    print("== 6. owner can run denied tools ==")
    import src.ai.owner_system as owners
    owners.OWNER_IDS.add(str(author.id))
    ctx2 = worker._ctx_cls(bot=client, guild=guild, current_channel_id=ch.id, author_id=author.id)
    ctx2.performed = []
    a, _ = worker._check_allowed(ctx2, "send_dm", {})
    check("owner send_dm allowed", a)
    owners.OWNER_IDS.discard(str(author.id))

    print("== 7. voice path ==")
    vc = FakeVoiceChannel(300, "lounge", guild)
    ok = await worker.try_handle_voice(author.id, "eudora leave the vc", vc)
    check("leave-vc short-circuits (None)", ok is None)

    set_script(worker, [
        ("tools", [("send_vc_text", {"channel_query": "here", "content": "on it"})]),
        ("content", "sure thing"),
    ])
    import src.action_engine.tools.voice as ae_voice
    # send_vc_text resolves its own channel — patch the resolver to our fake VC
    async def _fake_vc(ctx, q): return vc
    ae_voice._resolve_voice_channel = _fake_vc
    ok = await worker.try_handle_voice(author.id, "send that to the vc chat", vc)
    check("voice action handled", isinstance(ok, str) and "sent" in ok, f"note={ok!r}")
    check("vc text sent", vc.sent and vc.sent[-1][0] == "on it", f"{vc.sent}")
    check("no spoken/done reply forwarded", all(c[0] != "sure thing" for c in vc.sent))

    print("== 7b. malformed tool-call rescue (the Groq 400 bug) ==")
    class FakeBadReq(Exception):
        def __init__(self):
            super().__init__("400 tool_use_failed")
            self.body = {"error": {"failed_generation":
                json.dumps({"name": "send_message<|channel|>commentary",
                            "arguments": {"channel_query": "here",
                                          "content": "rescued call",
                                          "application_id": None}})}}

    class RescuePool(FakePool):
        async def chat(self, **kw):
            self.calls.append(kw)
            if len(self.calls) == 1:
                raise FakeBadReq()
            return _Resp(_Msg(content="ok"))

    gp._pool = RescuePool([])
    worker._agent = None
    ch.sent.clear()
    note = await worker.try_handle_text(
        FakeMessage("send a message", author, ch), "send a message")
    check("rescued hallucinated tool name", isinstance(note, str))
    check("rescued call actually ran", ch.sent and ch.sent[-1][0] == "rescued call", f"{ch.sent}")

    print("== 8. engine failure -> None, no crash ==")
    set_script(worker, [("content", "done")])
    async def boom(**kw): raise RuntimeError("groq down")
    gp._pool.chat = boom
    ok = await worker.try_handle_text(FakeMessage("send hello", author, ch), "send hello")
    check("engine crash -> None (reply path continues)", ok is None)

    print()
    n = sum(1 for _, ok in passed if ok)
    print(f"{n}/{len(passed)} checks passed")
    return 0 if n == len(passed) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
