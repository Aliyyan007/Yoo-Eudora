"""Scheduler persistence/execution semantics."""
import asyncio
import time

import pytest

from src.action_engine.core import scheduler as sched_mod
from src.action_engine.core.scheduler import Scheduler

_ORIGINAL_STORE = sched_mod.STORE


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch, tmp_path):
    monkeypatch.setattr(sched_mod, "STORE", tmp_path / "scheduled_tasks.json")


class FakeChannel:
    def __init__(self, fail=False):
        self.fail = fail
        self.sent = []

    async def send(self, content):
        if self.fail:
            raise RuntimeError("send failed")
        self.sent.append(content)


class FakeBot:
    def __init__(self, channel):
        self.channel = channel

    def get_channel(self, cid):
        return self.channel

    async def fetch_channel(self, cid):
        return self.channel


def make_task(**kw):
    base = {
        "id": "task1",
        "kind": "send_message",
        "guild_id": 1,
        "channel_id": 2,
        "channel_name": "general",
        "content": "hello",
        "interval_s": None,
        "run_at": time.time(),
        "created_by": 3,
    }
    base.update(kw)
    return base


async def test_successful_oneshot_is_removed_only_after_send():
    channel = FakeChannel()
    scheduler = Scheduler(FakeBot(channel))
    task = make_task()
    scheduler.tasks[task["id"]] = task

    await scheduler._run_and_finalize(task)

    assert channel.sent == ["hello"]
    assert task["id"] not in scheduler.tasks


async def test_failed_oneshot_is_retained_with_backoff():
    channel = FakeChannel(fail=True)
    scheduler = Scheduler(FakeBot(channel))
    task = make_task()
    scheduler.tasks[task["id"]] = task
    before = time.time()

    await scheduler._run_and_finalize(task)

    assert task["id"] in scheduler.tasks
    assert task["failure_count"] == 1
    assert task["run_at"] >= before + 25
    assert not task.get("dead")


async def test_oneshot_deadletters_after_bounded_retries():
    channel = FakeChannel(fail=True)
    scheduler = Scheduler(FakeBot(channel))
    task = make_task(failure_count=sched_mod.MAX_ONESHOT_FAILURES - 1)
    scheduler.tasks[task["id"]] = task

    await scheduler._run_and_finalize(task)

    assert task["id"] in scheduler.tasks
    assert task["dead"] is True
    assert task["dead_reason"] == "max retries exceeded"


async def test_interval_failure_keeps_future_cadence():
    channel = FakeChannel(fail=True)
    scheduler = Scheduler(FakeBot(channel))
    future = time.time() + 60
    task = make_task(id="repeat", interval_s=60, run_at=future)
    scheduler.tasks[task["id"]] = task

    await scheduler._run_and_finalize(task)

    assert task["id"] in scheduler.tasks
    assert task["run_at"] == future
    assert task["failure_count"] == 1
    assert not task.get("dead")


async def test_cancelled_run_does_not_consume_task():
    channel = FakeChannel()
    scheduler = Scheduler(FakeBot(channel))
    task = make_task()
    scheduler.tasks[task["id"]] = task

    async def slow_run(t):
        await asyncio.sleep(10)
        return True

    scheduler._run = slow_run
    runner = asyncio.create_task(scheduler._run_and_finalize(task))
    await asyncio.sleep(0)
    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert task["id"] in scheduler.tasks
    assert "_running" not in task
    assert not channel.sent


def test_store_is_repo_anchored_not_cwd_relative():
    assert _ORIGINAL_STORE.is_absolute()
    assert _ORIGINAL_STORE.name == "scheduled_tasks.json"
    assert _ORIGINAL_STORE.parent.name == "data"


def test_start_is_idempotent():
    async def run():
        scheduler = Scheduler(FakeBot(FakeChannel()))
        await scheduler.start()
        first = scheduler._job
        await scheduler.start()
        assert scheduler._job is first
        await scheduler.stop()
    asyncio.run(run())
