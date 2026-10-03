"""Stale-memory sweep test — the "auto reset" that isn't a wipe:
old users/channels are forgotten, fresh ones keep everything.

    python _test_memory_gc.py
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

from src.ai import memory as mem

# Sandbox: redirect the memory file + clear the module cache
_tmp = tempfile.mkdtemp(prefix="memgc_")
mem._MEMORY_FILE = os.path.join(_tmp, "memory.json")
mem._invalidate_cache()

DAY = 86400
now = time.time()

# Seed: 3 users — fresh, stale (90d), legacy (no last_seen)
mem.update_user_memory("fresh_u", "Fresh", ["likes tea"])
mem.update_user_memory("stale_u", "Stale", ["likes old stuff"])
mem.update_user_memory("legacy_u", "Legacy", ["was here before"])
mem.update_user_profile("fresh_u", "Fresh", "hobbies", "chess, tea")
m = mem._load_memory()
m["users"]["stale_u"]["last_seen"] = now - 90 * DAY
del m["users"]["legacy_u"]["last_seen"]   # simulate pre-sweep record
# discovered channels: 2 fresh + 2 stale
m["discovered_channels"] = {
    "c_fresh": {"name": "gen", "guild_id": "g", "last_seen": now},
    "c_fresh2": {"name": "vc", "guild_id": "g", "last_seen": now - 5 * DAY},
    "c_old": {"name": "dead", "guild_id": "g", "last_seen": now - 120 * DAY},
    "c_no_ts": {"name": "older", "guild_id": "g"},  # no timestamp at all
}
mem._save_memory(m)

print("\n== TTL sweep ==")
stats = mem.sweep_stale_memory(max_age_days=60, user_cap=2000, channel_cap=500)
m = mem._load_memory()
check("fresh user kept", "fresh_u" in m["users"])
check("fresh facts intact", "likes tea" in m["users"]["fresh_u"]["facts"])
check("fresh hobbies intact",
      "chess" in m["users"]["fresh_u"].get("hobbies", []))
check("stale user forgotten", "stale_u" not in m["users"])
check("legacy adopted not deleted", "legacy_u" in m["users"]
      and m["users"]["legacy_u"]["last_seen"] > now - 60)
check("stats: 1 evicted, 1 adopted",
      stats["users_evicted"] == 1 and stats["users_adopted"] == 1)
check("fresh channels kept",
      "c_fresh" in m["discovered_channels"]
      and "c_fresh2" in m["discovered_channels"])
check("old channel dropped", "c_old" not in m["discovered_channels"])
check("no-timestamp channel dropped",
      "c_no_ts" not in m["discovered_channels"])

# Legacy user must get the full TTL — next sweep right away keeps them
mem.sweep_stale_memory(max_age_days=60)
check("legacy survives immediate re-sweep",
      "legacy_u" in mem._load_memory()["users"])

print("\n== caps ==")
m = mem._load_memory()
for i in range(8):
    m["users"][f"cap_{i}"] = {"name": f"U{i}", "facts": [],
                              "last_seen": now - i * 100}  # cap_0 newest
mem._save_memory(m)
stats = mem.sweep_stale_memory(max_age_days=60, user_cap=6, channel_cap=500)
m = mem._load_memory()
check("user cap enforced", len(m["users"]) <= 6,
      f"users={len(m['users'])}")
check("newest users kept",
      "fresh_u" in m["users"] and "cap_0" in m["users"])
check("oldest evicted first", "cap_7" not in m["users"])

m["discovered_channels"] = {f"ch_{i}": {"name": "x", "guild_id": "g",
                                        "last_seen": now - i * 10}
                            for i in range(10)}
mem._save_memory(m)
mem.sweep_stale_memory(max_age_days=60, user_cap=2000, channel_cap=4)
check("channel cap enforced", len(mem._load_memory()
      ["discovered_channels"]) == 4)
check("newest channels kept",
      "ch_0" in mem._load_memory()["discovered_channels"])

print("\n== idempotent ==")
stats2 = mem.sweep_stale_memory(max_age_days=60, user_cap=2000, channel_cap=500)
check("second sweep evicts nothing",
      stats2["users_evicted"] == 0 and stats2["channels_evicted"] == 0)

print(f"\n{'='*50}\n{PASS} passed, {FAIL} failed\n{'='*50}")
sys.exit(1 if FAIL else 0)
