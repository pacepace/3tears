"""SqlL3Backend re-raises a query's own error when asyncpg's pool release race masks it.

asyncpg 0.31.0's ``PoolConnectionHolder.release`` waits for an in-flight cancellation after
a query fails; if the server closes the connection during that wait, release calls
``reset`` on ``None`` and the ``AttributeError`` escapes ``pool.acquire()``'s exit in place
of the query's real error. Seen live on 2026-09-25 as ``'NoneType' object has no attribute
'reset'`` hiding a ``TimeoutError`` during a YugabyteDB restart.

The masking error is built here the way asyncpg raises it -- inside a function compiled
with asyncpg's ``pool.py`` as its filename, on ``None``, while the query's error is in
flight -- so the recognition runs against the real traceback shape.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import pytest

from threetears.core.backends.sql import SqlL3Backend

_POOL_FILE = os.path.join("site-packages", "asyncpg", "pool.py")
_OTHER_FILE = os.path.join("site-packages", "somelib", "module.py")


def _raiser(filename: str) -> Any:
    """a function whose frame reports ``filename``, raising AttributeError on ``None``.

    :param filename: the source file the raising frame claims
    :ptype filename: str
    :return: the compiled function
    :rtype: Any
    """
    namespace: dict[str, Any] = {}
    code = compile("def release(con):\n    con.reset(timeout=None)\n", filename, "exec")
    exec(code, namespace)  # noqa: S102 -- builds a frame whose filename is asyncpg's pool.py
    return namespace["release"]


def _masked(filename: str, *, with_context: bool = True) -> AttributeError:
    """the AttributeError asyncpg's release raises, optionally over a query's own error.

    :param filename: the file the raising frame claims
    :ptype filename: str
    :param with_context: whether a query error is in flight when release raises
    :ptype with_context: bool
    :return: the raised AttributeError, traceback attached
    :rtype: AttributeError
    """
    release = _raiser(filename)
    try:
        if with_context:
            try:
                raise TimeoutError("query timed out")
            except TimeoutError:
                release(None)
        else:
            release(None)
    except AttributeError as exc:
        return exc
    raise AssertionError("release did not raise")


class _RacingPool:
    """a pool whose every call raises a prebuilt exception, as the release race does."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def fetch(self, *_args: Any) -> list[Any]:
        raise self._exc

    async def fetchrow(self, *_args: Any) -> Any:
        raise self._exc

    async def fetchval(self, *_args: Any) -> Any:
        raise self._exc

    async def execute(self, *_args: Any) -> str:
        raise self._exc


_CALLS = ["fetch", "fetchrow", "fetchval", "execute"]


@pytest.mark.parametrize("method", _CALLS)
def test_the_query_error_is_raised_in_place_of_the_masking_one(method: str, caplog: pytest.LogCaptureFixture) -> None:
    backend = SqlL3Backend(_RacingPool(_masked(_POOL_FILE)))

    with caplog.at_level(logging.WARNING), pytest.raises(TimeoutError, match="query timed out"):
        asyncio.run(getattr(backend, method)("SELECT 1"))

    assert "asyncpg pool release raced" in caplog.text
    assert "TimeoutError" in caplog.text


@pytest.mark.parametrize("method", _CALLS)
def test_an_attribute_error_from_anywhere_else_propagates(method: str) -> None:
    backend = SqlL3Backend(_RacingPool(_masked(_OTHER_FILE)))

    with pytest.raises(AttributeError, match="reset"):
        asyncio.run(getattr(backend, method)("SELECT 1"))


def test_a_release_attribute_error_with_nothing_in_flight_propagates() -> None:
    """with no query error to restore there is nothing to unmask."""
    backend = SqlL3Backend(_RacingPool(_masked(_POOL_FILE, with_context=False)))

    with pytest.raises(AttributeError, match="reset"):
        asyncio.run(backend.fetchrow("SELECT 1"))


def test_execute_batch_without_a_transaction_is_unmasked_too() -> None:
    backend = SqlL3Backend(_RacingPool(_masked(_POOL_FILE)))

    with pytest.raises(TimeoutError, match="query timed out"):
        asyncio.run(backend.execute_batch([{"query": "SELECT 1"}], transaction=False))
