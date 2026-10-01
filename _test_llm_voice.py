"""Throwaway sanity test for the reworked llm.py rotation/voice logic."""
import sys, types, time, threading
from unittest.mock import MagicMock

sys.path.insert(0, r"D:\Aliyyan's Projects\Auto Bump System\stable version\src")
sys.path.insert(0, r"D:\Aliyyan's Projects\Auto Bump System\stable version")

# Stub 'src' package context: import src.ai.llm directly
import importlib
llm = importlib.import_module("src.ai.llm")


class FakeMsg:
    def __init__(self, content):
        self.content = content
        self.tool_calls = None

class FakeChoice:
    def __init__(self, content):
        self.message = FakeMsg(content)
        self.finish_reason = "stop"

class FakeResp:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]

class FakeCompletions:
    def __init__(self, behavior):
        self.behavior = behavior  # callable -> str or raises
        self.calls = 0
    def create(self, **kwargs):
        self.calls += 1
        return self.behavior(**kwargs)

class FakeClient:
    def __init__(self, key, behavior):
        self.api_key = key
        self.chat = types.SimpleNamespace(completions=FakeCompletions(behavior))

def ok(**kw):
    return FakeResp(f"reply-from-{kw['model']}")

def err429(**kw):
    raise Exception("Error 429: rate limit exceeded. Please try again in 1s")

def err404(**kw):
    raise Exception("404 model_not_found: does not exist")


def reset(keys):
    llm._groq_clients = keys
    llm._groq_cooldowns.clear()
    llm._groq_current_key_idx = 0


print("== test 1: basic success ==")
c1 = FakeClient("k1", ok)
reset([c1])
r = llm.call_voice("t", "sys", "user", want_json=False)
assert r == "reply-from-openai/gpt-oss-20b", r
print("ok:", r)

print("== test 2: 429 on key1 rotates to key2 ==")
c1 = FakeClient("k1", err429); c2 = FakeClient("k2", ok)
reset([c1, c2])
r = llm.call_voice("t", "sys", "user", want_json=False)
assert r == "reply-from-openai/gpt-oss-20b", r
assert c2.chat.completions.calls == 1
print("ok:", r, "| cooldown keys:", list(llm._groq_cooldowns))

print("== test 3: per-model cooldown — key cooled on 20b still serves 120b ==")
# key1 429s on 20b (cooling), key1 is also the only key; fallback should try 120b on key1
c1 = FakeClient("k1", lambda **kw: err429() if kw["model"] == "openai/gpt-oss-20b" else ok(**kw))
reset([c1])
r = llm.call_voice("t", "sys", "user", want_json=False)
assert r == "reply-from-openai/gpt-oss-120b", r
print("ok:", r)

print("== test 4: all keys cooling on both models -> bounded wait, returns '' ==")
c1 = FakeClient("k1", err429)
reset([c1])
start = time.time()
r = llm.call_voice("t", "sys", "user", want_json=False)
elapsed = time.time() - start
assert r == "", r
assert elapsed < 30, elapsed  # bounded, not 90s+90s unbounded
print(f"ok: empty after {elapsed:.1f}s")

print("== test 5: max_wait_s respected ==")
c1 = FakeClient("k1", err429)
reset([c1])
start = time.time()
r = llm._call_llm("t", "sys", "user", model="openai/gpt-oss-20b", want_json=False, max_wait_s=3)
elapsed = time.time() - start
assert r == "", r
assert elapsed < 8, elapsed
print(f"ok: gave up after {elapsed:.1f}s (bound=3 + retry churn)")

print("== test 6: 404 returns immediately, no rotation ==")
c1 = FakeClient("k1", err404); c2 = FakeClient("k2", err404)
reset([c1, c2])
start = time.time()
r = llm.call_voice("t", "sys", "user", want_json=False)
elapsed = time.time() - start
# first model 404s -> "" -> fallback model also 404s -> ""
assert r == "", r
assert elapsed < 5, elapsed
print(f"ok: empty after {elapsed:.1f}s")

print("== test 7: concurrent calls pick different keys ==")
barrier = threading.Barrier(2)
def slow_ok(**kw):
    barrier.wait(timeout=5)
    return FakeResp("done")
c1 = FakeClient("k1", slow_ok); c2 = FakeClient("k2", slow_ok)
reset([c1, c2])
results = []
threads = [threading.Thread(target=lambda: results.append(
    llm._call_llm("t", "s", "u", model="m", want_json=False))) for _ in range(2)]
start = time.time()
for t in threads: t.start()
for t in threads: t.join(10)
elapsed = time.time() - start
assert results == ["done", "done"], results
assert c1.chat.completions.calls == 1 and c2.chat.completions.calls == 1, "keys not spread"
print(f"ok: 2 concurrent calls, both done, keys used: {c1.chat.completions.calls}/{c2.chat.completions.calls}")

print("== test 8: call_fast/call_smart signatures still work ==")
c1 = FakeClient("k1", ok)
reset([c1])
r = llm.call_fast("t", "sys", "user")
assert "gpt-oss-20b" in r
r = llm.call_smart("t", "sys", "user")
assert "gpt-oss-20b" in r
print("ok")

print("\nALL TESTS PASSED")
