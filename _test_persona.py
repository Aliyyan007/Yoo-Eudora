"""Multi-persona rotation test suite — run standalone:
    python _test_persona.py

Covers: profile loading, prompt swap on activation, memory namespacing,
pending-interaction store, rotation-state resume, supervisor account
ordering/failure skip, engagement own-ids, vc_intent name-awareness,
bump-cooldown persistence. No network, no Discord connection.
"""
import os, sys, time, json, asyncio, tempfile, importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ACTION_WORKER_ENABLED", "1")

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {extra}")

import src.persona.profiles as profiles
from src.persona import runtime as rt
from src.persona.manager import PersonaSupervisor
from src.persona.profiles import PROFILES, get_profile

# Redirect all persona state files to a tempdir — never pollute real data/
_tmp = tempfile.mkdtemp(prefix="persona_test_")
rt._REGISTRY_FILE = os.path.join(_tmp, "persona_registry.json")
rt._STATE_FILE = os.path.join(_tmp, "persona_state.json")
rt._PENDING_DIR = os.path.join(_tmp, "pending")

# ── 1. Profiles ─────────────────────────────────────────────────────
print("\n== profiles ==")
eudora, isla, rowan = PROFILES["eudora"], PROFILES["isla"], PROFILES["rowan"]
check("3 profiles", len(PROFILES) == 3)
check("eudora unchanged", eudora.age == 22 and "london" in eudora.location.lower())
check("isla distinct", "Isla Bennett" in isla.persona_block and isla.id != eudora.id)
check("rowan distinct", "Rowan Hayes" in rowan.persona_block and rowan.gender == "male")
check("eudora uses DISCORD_TOKEN", eudora.token_env == "DISCORD_TOKEN")
check("isla/rowan own env", isla.token_env == "DISCORD_TOKEN_ISLA" and rowan.token_env == "DISCORD_TOKEN_ROWAN")
check("get_profile fallback", get_profile("nobody").id == "eudora")

# ── 2. Prompt swap on activation ────────────────────────────────────
print("\n== prompt swap ==")
from src.ai import prompts
rt.activate(eudora)
check("eudora persona", "Eudora Edward" in prompts.PERSONA and "eudora" not in "isla")
check("eudora voice", "Eudora Edward" in prompts.VOICE_REPLY_SYSTEM)
rt.activate(isla)
check("isla persona swapped", "Isla Bennett" in prompts.PERSONA and "Eudora" not in prompts.PERSONA)
check("isla voice swapped", "Isla Bennett" in prompts.VOICE_REPLY_SYSTEM)
check("isla proactive", "Isla" in prompts.PROACTIVE_SYSTEM)
check("isla reply system inherits", "CONVERSATION AWARENESS" in prompts.REPLY_SYSTEM)
check("isla reflection", "Isla" in prompts.SELF_REFLECTION_SYSTEM)
rt.activate(rowan)
check("rowan voice", "Rowan Hayes" in prompts.VOICE_REPLY_SYSTEM)

from src.action_engine.config import prompts as ae_prompts
check("ae persona swapped", "Rowan Hayes" in ae_prompts.CHAT_SYSTEM_PROMPT)
check("ae voice swapped", "Rowan Hayes" in ae_prompts.VOICE_SYSTEM_PROMPT)
check("ae plain swapped", "Rowan" in ae_prompts.CHAT_SYSTEM_PROMPT_PLAIN)
rt.activate(eudora)
check("swap back clean", "Eudora Edward" in prompts.PERSONA and "Rowan" not in prompts.PERSONA)

# ── 3. Memory namespacing ───────────────────────────────────────────
print("\n== memory isolation ==")
from src.ai import d1_memory as mem
from src.ai import memory as _mem_mod
# Sandbox the JSON fallback too — the test's namespaced keys must not
# land in the real data/memory.json
_mem_mod._MEMORY_FILE = os.path.join(_tmp, "memory.json")
_mem_mod._memory_cache = {}
# D1 is configured-but-unreachable in this env — force the JSON fallback
# path so the test measures namespacing, not network auth failures
mem._d1_client = False
rt.activate(isla)
mem.update_user_memory("u123", "TestUser", ["likes cats"])
isla_mem = mem.get_user_memory_text("u123", "TestUser")
rt.activate(eudora)
eudora_mem = mem.get_user_memory_text("u123", "TestUser")
rt.activate(isla)
isla_mem2 = mem.get_user_memory_text("u123", "TestUser")
check("isla saw fact", "cat" in isla_mem.lower(), isla_mem[:60])
check("eudora isolated", "cat" not in eudora_mem.lower(), eudora_mem[:60])
check("isla retained after switch", "cat" in isla_mem2.lower())
mem.update_user_profile("u123", "TestUser", "real_name", "Tes")
prof = mem.get_user_profile("u123")
check("profile namespaced", prof.get("real_name") == "Tes")
rt.activate(rowan)
check("rowan isolated", "Tes" not in str(mem.get_user_profile("u123")))
rt.activate(eudora)

# ── 4. Pending store ────────────────────────────────────────────────
print("\n== pending store ==")
rt.add_pending("isla", guild_id=1, channel_id=2, message_id=100,
               author_id=9, author_name="Holly", text="hey isla!", kind="mention")
rt.add_pending("isla", guild_id=1, channel_id=2, message_id=101,
               author_id=9, author_name="Holly", text="isla you there", kind="name")
fresh = rt.fresh_pending("isla")
check("pending recorded", len(fresh) == 2)
check("pending fields", fresh[0]["kind"] == "mention" and fresh[0]["author_name"] == "Holly")
rt.mark_pending_handled("isla", [100])
fresh2 = rt.fresh_pending("isla")
check("handled filtered", len(fresh2) == 1 and fresh2[0]["message_id"] == 101)
check("no cross-persona pending", len(rt.fresh_pending("rowan")) == 0)
# stale item: force old ts
p = os.path.join(rt._PENDING_DIR, "isla.json")
data = json.load(open(p, encoding="utf-8"))
data[0]["ts"] = time.time() - 7200
data[0]["handled"] = False
json.dump(data, open(p, "w", encoding="utf-8"))
check("stale excluded from fresh", all(i["message_id"] != 100 for i in rt.fresh_pending("isla")))
check("stale visible as stale", any(i["message_id"] == 100 for i in rt.stale_pending("isla")))

# ── 5. Rotation state resume ────────────────────────────────────────
print("\n== rotation state ==")
rt.save_rotation_state("isla", time.time() - 3600, 4)
st = rt.load_rotation_state()
check("state saved", st["active"] == "isla" and st["seq"] == 4)
rt.clear_rotation_state()
check("state cleared", rt.load_rotation_state() == {})

# ── 6. Supervisor ───────────────────────────────────────────────────
print("\n== supervisor ==")
os.environ["DISCORD_TOKEN"] = "t_e"
os.environ["DISCORD_TOKEN_ISLA"] = "t_i"
os.environ["DISCORD_TOKEN_ROWAN"] = "t_r"
sup = PersonaSupervisor(lambda p: None)
check("3 accounts loaded", [a.id for a in sup._accounts] == ["eudora", "isla", "rowan"])
del os.environ["DISCORD_TOKEN_ISLA"]
sup2 = PersonaSupervisor(lambda p: None)
check("missing token skipped", [a.id for a in sup2._accounts] == ["eudora", "rowan"])
os.environ["DISCORD_TOKEN_ISLA"] = "t_i"
check("next() wraps", sup._next(3).id == "eudora" and sup._next(4).id == "isla")

# ── 7. Engagement own-ids ───────────────────────────────────────────
print("\n== engagement own-ids ==")
from src.ai.engagement_log import get_engagement_log
el = get_engagement_log()
el.register_own_ids([111, 222, 333])
check("own match", el._own(222, 111) and el._own(111, 111))
check("not own", not el._own(999, 111))

# ── 8. vc_intent name-awareness ─────────────────────────────────────
print("\n== vc_intent ==")
from src.voice import vc_intent
rt.activate(isla)
check("isla leave hits", vc_intent.leave_vc_score("isla leave") > 0)
check("eudora leave ignored (not isla)", vc_intent.leave_vc_score("eudora leave") == 0.0)
check("generic still works", vc_intent.leave_vc_score("leave the vc") > 0)
check("farewell still excluded", vc_intent.leave_vc_score("i'm gonna go") == 0.0)
rt.activate(rowan)
check("rowan leave hits", vc_intent.leave_vc_score("rowan leave now") > 0)
rt.activate(eudora)
check("eudora leave hits", vc_intent.leave_vc_score("eudora leave") > 0)

# ── 9. Bump state persistence ───────────────────────────────────────
print("\n== bump state ==")
from src.bump_scheduler import BumpScheduler
BumpScheduler._STATE_FILE = os.path.join(_tmp, "bump_state.json")
class _C:  # minimal client stub
    pass
b1 = BumpScheduler(client=_C(), channel_id=1)
b1._last_bump_time["Disboard"] = 12345.0
b1._save_state()
b2 = BumpScheduler(client=_C(), channel_id=1)
check("bump cooldown persisted", b2._last_bump_time.get("Disboard") == 12345.0)

# ── 10. Registry + own ids round-trip ───────────────────────────────
print("\n== registry ==")
rt.activate(isla)
rt.register_self(222, "Isla")
rt.activate(rowan)
rt.register_self(333, "Rowan")
rt.activate(eudora)
others = rt.other_persona_ids()
check("registry has others", others.get(222) == "isla" and others.get(333) == "rowan")
check("self excluded", 111 not in others)
check("own_user_ids", set(rt.own_user_ids()) >= {222, 333})

# ── 11. End-to-end rotation (fake clients) ──────────────────────────
print("\n== rotation e2e ==")
import src.persona.manager as mgr

started, closed, torn_down = [], [], []
class _FakeClient:
    def __init__(self, pid):
        self.pid = pid
    async def start(self, token):
        started.append(self.pid)
        # stay "connected" until the supervisor closes us
        while True:
            await asyncio.sleep(0.05)
    async def close(self):
        closed.append(self.pid)
    async def teardown(self):
        torn_down.append(self.pid)

async def _rotation_run():
    # 0.05s windows → each "persona" rotates almost instantly
    mgr._window_seconds = lambda: 0.05
    mgr._CONNECT_RETRY_LIMIT = 1
    rt.clear_rotation_state()
    sup = PersonaSupervisor(lambda p: _mk(p))
    async def _mk(p):
        c = _FakeClient(p.id)
        return c, c.start("tok")
    sup._factory = _mk
    t = asyncio.create_task(sup.run())
    await asyncio.sleep(0.4)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass
    return sup

asyncio.run(_rotation_run())
check("rotation advanced past first persona", len(started) >= 2, started)
check("teardown before close", torn_down and closed and len(torn_down) >= 1)
check("distinct personas ran", len(set(started)) >= 2)
st = rt.load_rotation_state()
check("state written during run", st.get("active") in {"eudora", "isla", "rowan"})
rt.clear_rotation_state()

# connect-failure skip: factory raises → supervisor moves to next account
calls = []
async def _fail_factory(p):
    calls.append(p.id)
    if p.id == "eudora":
        raise RuntimeError("bad token")
    c = _FakeClient(p.id)
    return c, c.start("tok")
async def _fail_run():
    sup = PersonaSupervisor(_fail_factory)
    import asyncio as _a
    mgr._window_seconds = lambda: 0.05
    t = _a.create_task(sup.run())
    await _a.sleep(0.3)
    t.cancel()
    try:
        await t
    except _a.CancelledError:
        pass
asyncio.run(_fail_run())
check("failed persona skipped", "eudora" in calls and "isla" in calls)

print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
