"""Re-engagement test — does the bot follow up when its message goes
unanswered, and does it stand down the moment a human speaks?

    python _test_reengage.py

Exercises ReEngagementTracker end-to-end at the logic level:
bot msg -> silence past timeout -> re-engage action fires -> bookkeeping ->
human reply -> no further re-engagement. Also covers streak muting.
"""
import os, sys, time, tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = FAIL = 0
def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name} {extra}")

from src.ai import re_engagement as re_mod

# Sandboxed controller (fresh state file — no real cooldowns leaking in)
_tmp = tempfile.mkdtemp(prefix="reengage_test_")
pc = re_mod.PingController()
pc._state_path = type(pc._state_path)(os.path.join(_tmp, "ping_state.json"))
re_mod._ping_controller = pc          # tracker reads this singleton

tracker = re_mod.ReEngagementTracker()
CH = "ch-engage"
now = time.time()

# ── 1. Baseline: nothing to re-engage ───────────────────────────────
print("\n== baseline ==")
check("no bot msg -> no re-engagement", not tracker.needs_re_engagement(CH))

# ── 2. Bot speaks, humans stay silent ───────────────────────────────
print("\n== unanswered bot message ==")
tracker.record_bot_message(CH)
check("fresh msg -> inside timeout, no re-engagement",
      not tracker.needs_re_engagement(CH))
# 9 minutes later — still inside the 10-min timeout
tracker._last_bot_msg[CH] = now - 540
check("9min silent -> no re-engagement yet",
      not tracker.needs_re_engagement(CH))
# 11 minutes — past timeout: the bot should now act
tracker._last_bot_msg[CH] = now - 660
check("11min silent -> needs re-engagement",
      tracker.needs_re_engagement(CH))

# ── 3. Re-engagement action is a real ping type ─────────────────────
print("\n== action selection ==")
actions = set()
for _ in range(200):  # randomised — collect the reachable set
    actions.add(tracker.get_re_engagement_action(CH))
check("actions are valid ping types",
      actions <= {"user_ping", "here", "everyone"} and actions != {"none"},
      f"got {actions}")
check("never 'none' while pings available", "none" not in actions)

# ── 4. Loop bookkeeping: after firing it must not instantly refire ──
print("\n== post-fire bookkeeping (mirrors _re_engagement_loop) ==")
action = tracker.get_re_engagement_action(CH)
if action == "everyone":
    pc.record_everyone_ping(CH)
elif action == "here":
    pc.record_here_ping(CH)
else:
    pc.record_user_ping(CH, user_id=42)
tracker.clear_channel(CH)
tracker.record_bot_message(CH)
check("after firing: state cleared, new timeout window",
      not tracker.needs_re_engagement(CH))
check("streak counted the unanswered ping", pc._unanswered(CH) == 1)

# ── 5. Human comes alive → everything stands down ───────────────────
print("\n== human wakes up ==")
tracker._last_bot_msg[CH] = now - 660   # silence past timeout again
check("silent again -> re-engagement pending",
      tracker.needs_re_engagement(CH))
tracker.record_human_reply(CH)         # ...then a human finally speaks
check("human reply -> no re-engagement",
      not tracker.needs_re_engagement(CH))
check("human reply reset ping streak", pc._unanswered(CH) == 0)

# ── 6. Dead channel stays muted (streak escalation) ─────────────────
print("\n== unanswered streak mutes mass pings ==")
pc.record_role_ping(CH)                 # streak 1
pc.record_here_ping(CH)                 # streak 2 -> mass pings muted
tracker.record_bot_message(CH)
tracker._last_bot_msg[CH] = now - 660
allowed = set()
for _ in range(200):
    allowed.add(tracker.get_re_engagement_action(CH))
check("streak=2: no mass pings chosen",
      allowed <= {"user_ping", "none"}, f"got {allowed}")
check("streak=2: user ping still reachable", "user_ping" in allowed)

pc.record_user_ping(CH); pc.record_user_ping(CH)  # streak 4
check("streak=4: fully muted",
      all(tracker.get_re_engagement_action(CH) == "none"
          for _ in range(50)))
tracker.record_human_reply(CH)
check("human reply clears streak", pc._unanswered(CH) == 0)
# Re-arming lifts the STREAK block — per-type cooldowns still apply, so
# backdate past them before checking pings are reachable again.
from collections import deque as _dq
for trk in (pc._here_pings, pc._everyone_pings, pc._role_pings, pc._user_pings):
    if CH in trk:
        trk[CH] = _dq([now - 25 * 3600], maxlen=50)
check("after cooldowns: pings reachable again",
      tracker.get_re_engagement_action(CH) != "none")
check("mass pings unblocked after reset", pc.can_ping_role(CH))

print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
