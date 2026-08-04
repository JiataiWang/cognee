import asyncio
import gc
import importlib
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve import state as serve_state

remember_module = importlib.import_module("cognee.api.v1.remember.remember")
improve_package = importlib.import_module("cognee.api.v1.improve")


class _Span:
    def set_attribute(self, *args, **kwargs):
        pass


class _PendingCall:
    """Suspend without keeping an external strong reference to the waiter."""

    def __init__(self):
        self.started = asyncio.Event()
        self.completed = False
        self._waiter_ref = None

    async def __call__(self, *args, **kwargs):
        waiter = asyncio.get_running_loop().create_future()
        self._waiter_ref = weakref.ref(waiter)
        self.started.set()
        await waiter
        self.completed = True

    def release(self):
        assert self._waiter_ref is not None
        waiter = self._waiter_ref()
        assert waiter is not None
        waiter.set_result(None)


@pytest.mark.asyncio
async def test_permanent_background_remember_task_is_anchored(monkeypatch):
    pending_add = _PendingCall()

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr("cognee.modules.migrations.startup.run_migrations_and_block", AsyncMock())
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", AsyncMock())
    monkeypatch.setattr("cognee.api.v1.add.add", pending_add)
    monkeypatch.setattr("cognee.api.v1.cognify.cognify", AsyncMock(return_value={}))
    monkeypatch.setattr(
        "cognee.tasks.ingestion.utils.materialize_stream_for_background",
        AsyncMock(side_effect=lambda data: data),
    )

    result = await remember_module._remember_inner(
        "memory",
        "dataset",
        dataset_id=uuid4(),
        session_id=None,
        chunk_size=None,
        chunker=object(),
        custom_prompt=None,
        run_in_background=True,
        self_improvement=False,
        session_ids=None,
        span=_Span(),
        user=SimpleNamespace(id=uuid4()),
    )
    await pending_add.started.wait()

    task_ref = weakref.ref(result._task)
    del result
    gc.collect()

    task = task_ref()
    assert task is not None
    pending_add.release()
    await task
    assert pending_add.completed


@pytest.mark.asyncio
async def test_session_improvement_task_is_anchored(monkeypatch):
    pending_improve = _PendingCall()

    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr("cognee.modules.migrations.startup.run_migrations_and_block", AsyncMock())
    monkeypatch.setattr("cognee.modules.engine.operations.setup.setup", AsyncMock())
    monkeypatch.setattr(remember_module, "_add_to_session", AsyncMock())
    monkeypatch.setattr(improve_package, "improve", pending_improve)

    result = await remember_module._remember_inner(
        "memory",
        "dataset",
        dataset_id=uuid4(),
        session_id="session-1",
        chunk_size=None,
        chunker=object(),
        custom_prompt=None,
        run_in_background=False,
        self_improvement=True,
        session_ids=None,
        span=_Span(),
        user=SimpleNamespace(id=uuid4()),
    )
    await pending_improve.started.wait()

    task_ref = weakref.ref(result._task)
    del result
    gc.collect()

    task = task_ref()
    assert task is not None
    pending_improve.release()
    await task
    assert pending_improve.completed


@pytest.mark.asyncio
async def test_completed_background_task_is_released_from_anchor():
    task = remember_module._schedule_background_task(asyncio.sleep(0, result="done"))

    assert task in remember_module._BACKGROUND_REMEMBER_TASKS
    assert await task == "done"
    await asyncio.sleep(0)
    assert task not in remember_module._BACKGROUND_REMEMBER_TASKS
