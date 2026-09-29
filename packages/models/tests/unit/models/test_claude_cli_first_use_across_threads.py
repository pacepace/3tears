"""the process-wide claude CLI pool and isolation roots are built once when threads race to them.

a sync ``invoke`` on the subscription chat model runs its own event loop on the CALLER's
thread, so a consumer building and calling models from several threads reaches these lazy
module-level singletons from several threads at once. both were checked and then filled
with no lock: every thread that looked before the first one stored its object built
another -- a second pool (twice the CLIs the limits allow, one orphaned), a second
isolation root for the same credential.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest

from threetears.models import claude_cli_pool as pool_module
from threetears.models.claude_cli_isolation import claude_cli_isolation

_THREADS = 8


def _race(call: Callable[[], Any]) -> tuple[list[Any], list[BaseException]]:
    """run ``call`` on several threads released together; collect results and errors.

    :param call: the first-use call every thread makes
    :ptype call: Callable[[], Any]
    :return: ``(results, errors)``
    :rtype: tuple[list[Any], list[BaseException]]
    """
    start = threading.Barrier(_THREADS)
    results: list[Any] = []
    errors: list[BaseException] = []
    record = threading.Lock()

    def run() -> None:
        start.wait()
        try:
            value = call()
        except BaseException as exc:  # noqa: BLE001 -- the caller's assertion reports every one
            with record:
                errors.append(exc)
            return
        with record:
            results.append(value)

    workers = [threading.Thread(target=run) for _ in range(_THREADS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
    return results, errors


class _SlowPool:
    """a pool whose construction takes long enough for every racing thread to arrive."""

    built = 0

    def __init__(self, **settings: Any) -> None:
        _ = settings
        type(self).built += 1
        time.sleep(0.05)

    def known_pids(self) -> list[int]:
        """no CLIs were ever started.

        :return: an empty list
        :rtype: list[int]
        """
        return []

    async def aclose(self) -> None:
        """nothing to stop.

        :return: None
        :rtype: None
        """


def test_threads_first_asking_for_the_pool_at_once_share_one(monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(pool_module.close_claude_cli_pool())
    monkeypatch.setattr(pool_module, "ClaudeCliPool", _SlowPool)
    _SlowPool.built = 0
    try:
        pools, errors = _race(pool_module.claude_cli_pool)

        assert errors == []
        assert len({id(pool) for pool in pools}) == 1
        assert _SlowPool.built == 1
    finally:
        asyncio.run(pool_module.close_claude_cli_pool())


def test_threads_first_isolating_one_credential_at_once_share_one_root(monkeypatch: pytest.MonkeyPatch) -> None:
    real_mkdtemp = tempfile.mkdtemp
    made: list[str] = []

    def slow_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        made.append(path)
        time.sleep(0.05)
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", slow_mkdtemp)
    token = f"test-token-{uuid4()}"

    isolations, errors = _race(lambda: claude_cli_isolation(token))

    assert errors == []
    assert len({isolation.cwd for isolation in isolations}) == 1
    assert len(made) == 1, f"{len(made)} isolation roots were created for one credential"
