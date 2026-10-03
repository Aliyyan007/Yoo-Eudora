"""Ping-safety + deferred-reply tests:
    python _test_ping_safety.py

Verifies: unanswered-ping streak muting/scaling, streak reset on human
activity, D1-mirror merge (stricter wins), mass-mention sanitizing in
replies + worker args, deferred queue store.
"""
import os, sys, time, json, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name} {extra}")

from src.ai import re_engagement as re_mod

# Sandbox: fresh PingController writing to a temp file
_tmp = tempfile.mkdtemp(prefix="ping_test_")
pc = re_mod.PingController()
pc._state_path = type(pc._state_path)(os.path.join(_tmp, "ping_state.json"))

CH = "ch1"
now = time.time()

# ── 1. Unanswered streak blocks mass pings ──────────────────────────
print("\n== unanswered-ping streak ==")
check("role ping allowed initially", pc.can_ping_role(CH))
pc.record_role_ping(CH)
check("streak=1: cooldown blocks immediate re-ping", not pc.can_ping_role(CH))
# Simulate cooldown elapsed (4x scaled at streak 1 → need >2h)
pc._role_pings[CH][-1] = now - (pc._role_cooldown_s * 4 + 10)
check("streak=1: allowed after scaled cooldown", pc.can_ping_role(CH))
pc.record_role_ping(CH)
pc._role_pings[CH][-1] = now - (pc._role_cooldown_s * 7 + 10)  # even far past
check("streak=2: muted entirely", not pc.can_ping_role(CH))
check("streak=2: @here also muted", not pc.can_ping_here(CH))
check("streak=2: @everyone muted", not pc.can_ping_everyone(CH))
check("streak=2: user ping still ok", pc.can_ping_user(CH) or
      pc._unanswered(CH) >= 4, "user ping lighter threshold")

# human speaks → everything unblocked
pc.note_human_activity(CH)
check("human activity resets streak", pc._unanswered(CH) == 0)
pc._role_pings[CH][-1] = now - (pc._role_cooldown_s + 10)
check("role ping allowed again", pc.can_ping_role(CH))

# ── 1b. Cross-type mass-ping gap ────────────────────────────────────
print("\n== inter-ping mass gap ==")
from collections import deque
CH3 = "ch-gap"
# Inject a user ping without record_* (keeps streak at 0 — the streak
# cooldown-scaling would otherwise exceed the gap window on its own)
pc._user_pings[CH3] = deque([now - (pc._user_cooldown_s + 10)], maxlen=50)
check("user ping allowed after own cooldown", pc.can_ping_user(CH3))
check("role ping blocked by recent user ping", not pc.can_ping_role(CH3))
check("@here blocked by recent user ping", not pc.can_ping_here(CH3))
check("@everyone blocked by recent user ping", not pc.can_ping_everyone(CH3))
pc._user_pings[CH3][-1] = now - (pc._min_mass_gap_s + 10)  # gap elapsed
check("role ping allowed once gap elapsed", pc.can_ping_role(CH3))
pc._role_pings.pop(CH3, None); pc._unanswered_pings.pop(CH3, None)

# ── 2. Persistence + merge ──────────────────────────────────────────
print("\n== persistence ==")
pc.record_here_ping(CH)
pc2 = re_mod.PingController()
pc2._state_path = pc._state_path
pc2._load_state()
check("here pings persisted", len(pc2._here_pings.get(CH, [])) >= 1)
# stricter-merge: remote says streak 2, local 0 → take 2
pc2._merge_remote({"unanswered": {CH: 2}, "role": {CH: [123.0]}})
check("merge takes max streak", pc2._unanswered(CH) == 2)
check("merge unions role pings", 123.0 in list(pc2._role_pings[CH]))

# ── 3. Mass-mention sanitizer ───────────────────────────────────────
print("\n== sanitizer ==")
from src.ai.reply import sanitize_mass_mentions
dirty = "hey @everyone check <@&1395007934764945538> and @here lads"
clean = sanitize_mass_mentions(dirty)
check("all mass tokens stripped",
      "everyone" not in clean and "here" not in clean
      and "<@&" not in clean)
check("user ping kept", sanitize_mass_mentions("yo <@12345> sup")
      .strip() == "yo <@12345> sup")
check("plain text untouched",
      sanitize_mass_mentions("just text innit") == "just text innit")

# worker arg sanitization (non-owner)
from src.ai.action_bridge import ActionWorker
w = ActionWorker.__new__(ActionWorker)
class _Ctx: author_id = 999  # non-owner
_, args = w._check_allowed(_Ctx(), "send_message",
                           {"content": "oi @everyone meet here <@&123>"})
check("worker content sanitized", "@everyone" not in args["content"]
      and "<@&" not in args["content"])
_, args2 = w._check_allowed(_Ctx(), "send_message",
                            {"content": "normal text <@42>"})
check("user mention survives", "<@42>" in args2["content"])

# ── 4. Deferred queue ───────────────────────────────────────────────
print("\n== deferred store ==")
from src.persona import runtime as rt
rt._DEFERRED_DIR = os.path.join(_tmp, "deferred")
rt.add_deferred("isla", guild_id=1, channel_id=2, message_id=55,
                author_id=9, author_name="Holly", text="@isla hey")
rt.add_deferred("isla", guild_id=1, channel_id=2, message_id=55,
                author_id=9, author_name="Holly", text="dup")
items = rt.deferred_for("isla")
check("deferred recorded", len(items) == 1)
check("dedup by msg id", len(items) == 1)
rt.remove_deferred("isla", 55)
check("deferred removed", rt.deferred_for("isla") == [])
# per-channel cap
for i in range(5):
    rt.add_deferred("isla", guild_id=1, channel_id=7, message_id=100+i,
                    author_id=9, author_name="X", text="hi")
check("per-channel cap=3", len([i for i in rt.deferred_for("isla")
                                if i["channel_id"] == 7]) == 3)
check("persona isolation", rt.deferred_for("rowan") == [])

print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
