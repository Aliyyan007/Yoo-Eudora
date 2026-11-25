"""Reliability regressions for provider/network failure handling.

The production logs showed two expensive failure modes:
- transient Groq connection errors retried in a tight loop for the whole
  key-wait budget
- invalid D1 credentials retried forever, producing thousands of warnings

These tests pin the circuit/backoff behavior without hitting real services.
"""
import asyncio
import time

import pytest

from src.ai import llm
from src.ai.d1_client import D1Client, D1Unavailable
import src.ai.d1_client as d1_client


class _FakeCompletions:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self.exc


class _FakeGroqClient:
    def __init__(self, api_key, exc):
        self.api_key = api_key
        self.completions = _FakeCompletions(exc)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _patch_single_groq_key(monkeypatch, exc):
    client = _FakeGroqClient("test-key", exc)
    monkeypatch.setattr(llm, "_groq_clients", [client])
    monkeypatch.setattr(llm, "_groq_current_key_idx", 0)
    monkeypatch.setattr(llm, "_groq_cooldowns", {})
    monkeypatch.setattr(llm, "_groq_transient_cooldowns", {})
    monkeypatch.setattr(llm, "_groq_transient_failures", {})
    monkeypatch.setattr(llm, "_groq_auth_cooldowns", {})
    monkeypatch.setattr(llm.random, "uniform", lambda a, b: a)
    return client


def test_groq_transient_network_error_backs_off_instead_of_tight_loop(monkeypatch):
    client = _patch_single_groq_key(monkeypatch, ConnectionError("connection reset by peer"))

    started = time.time()
    out = llm._call_llm(
        "unit", "system", "user", want_json=False,
        model="openai/gpt-oss-20b", max_wait_s=0.05,
    )
    elapsed = time.time() - started

    assert out == ""
    assert elapsed < 1.0
    # One immediate failure is enough to park the key; without backoff the
    # old loop made dozens/hundreds of identical connection attempts here.
    assert client.completions.calls == 1
    ck = llm.cooldown_key("test-key", "openai/gpt-oss-20b")
    assert llm._groq_transient_cooldowns[ck] > time.time() + 1.0


def test_groq_invalid_request_fails_fast(monkeypatch):
    client = _patch_single_groq_key(
        monkeypatch, Exception("400 invalid_request_error: max_tokens is invalid"))

    out = llm._call_llm(
        "unit", "system", "user", want_json=False,
        model="openai/gpt-oss-20b", max_wait_s=5,
    )

    assert out == ""
    assert client.completions.calls == 1


def test_groq_auth_error_parks_key_across_models(monkeypatch):
    client = _patch_single_groq_key(monkeypatch, Exception("401 unauthorized invalid api key"))

    out = llm._call_llm(
        "unit", "system", "user", want_json=False,
        model="openai/gpt-oss-20b", max_wait_s=5,
    )

    assert out == ""
    assert client.completions.calls == 1
    assert llm._key_unavailable_until("test-key", "openai/gpt-oss-120b") > time.time() + 1000


class _FakeD1Response:
    def __init__(self, status=200, payload=None):
        self.status = status
        self.payload = payload or {"success": True, "result": [{"results": [], "meta": {}}]}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self.payload


class _FakeD1Session:
    def __init__(self, response):
        self.response = response
        self.posts = 0

    def post(self, *args, **kwargs):
        self.posts += 1
        return self.response


def _patch_d1_creds(monkeypatch):
    monkeypatch.setattr(d1_client, "_ACCOUNT_ID", "acct")
    monkeypatch.setattr(d1_client, "_DATABASE_ID", "db")
    monkeypatch.setattr(d1_client, "_API_TOKEN", "token")


def test_d1_auth_failure_is_not_retried_and_opens_circuit(monkeypatch):
    _patch_d1_creds(monkeypatch)
    client = D1Client()
    session = _FakeD1Session(_FakeD1Response(status=401, payload={"success": False}))

    async def fake_get_session():
        return session

    monkeypatch.setattr(client, "_get_session", fake_get_session)

    async def run():
        with pytest.raises(D1Unavailable):
            await client._execute_raw("SELECT 1")

    asyncio.run(run())

    assert session.posts == 1
    state = client.circuit_state()
    assert state["open"] is True
    assert state["reason"] == "authentication"


def test_d1_transient_failures_open_circuit_after_threshold(monkeypatch):
    _patch_d1_creds(monkeypatch)
    client = D1Client()

    assert client.circuit_state()["open"] is False
    client._note_failure("transient")
    client._note_failure("transient")
    assert client.circuit_state()["open"] is False
    client._note_failure("transient")
    assert client.circuit_state()["open"] is True

    async def run():
        with pytest.raises(D1Unavailable, match="circuit open"):
            await client._execute_raw("SELECT 1")

    asyncio.run(run())


def test_d1_success_closes_circuit_bookkeeping():
    client = D1Client()
    client._note_failure("transient")
    client._note_failure("transient")
    client._note_failure("transient")
    assert client.circuit_state()["open"] is True
    client._note_success()
    state = client.circuit_state()
    assert state["open"] is False
    assert state["consecutive_failures"] == 0
