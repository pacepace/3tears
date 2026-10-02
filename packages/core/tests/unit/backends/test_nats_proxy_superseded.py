"""a pod's data version, refused by the broker, reaches the pod as a typed error.

The broker compares the data version in a pod's identity token with its space's target and refuses
two ways. ``DATA_VERSION_SUPERSEDED``: the pod is older than the target, and it must exit -- a new
pod at the new version replaces it. ``DATA_VERSION_NOT_READY``: the pod is AT the target but the
upgrade that brings the space there has not finished, and it must wait. The two answers demand
opposite responses, so each arrives as its own type, and the runtime is told about supersession
through the backend's ``on_superseded`` callback so the exit is the runtime's to own.

Every test here drives the real decode: a reply dict, JSON-encoded as the hub sends it, through the
proxy's own request path, on each of the reply paths a pod reads.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from threetears.nats import Subject

from threetears.core.backends.nats_proxy import (
    DATA_VERSION_NOT_READY_ERROR_CODE,
    DATA_VERSION_SUPERSEDED_ERROR_CODE,
    NatsProxyL3Backend,
)
from threetears.core.exceptions import (
    DataLayerUnavailableError,
    DataVersionNotReadyError,
    DataVersionSupersededError,
)

_TX_ID = "019d9a00-0000-7000-8000-000000000000"


def _refusal(code: str, message: str = "refused") -> dict[str, Any]:
    """
    a failed reply as the broker sends it for a data-version refusal.

    :param code: the broker's error code
    :ptype code: str
    :param message: the broker's human message
    :ptype message: str
    :return: the reply dict
    :rtype: dict[str, Any]
    """
    return {"success": False, "error_code": code, "error_message": message}


def _superseded() -> dict[str, Any]:
    """
    the supersession refusal.

    :return: the reply dict
    :rtype: dict[str, Any]
    """
    return _refusal(DATA_VERSION_SUPERSEDED_ERROR_CODE, "data version 3 is older than target 4")


def _not_ready() -> dict[str, Any]:
    """
    the not-ready refusal.

    :return: the reply dict
    :rtype: dict[str, Any]
    """
    return _refusal(DATA_VERSION_NOT_READY_ERROR_CODE, "target 4 is not yet applied")


def _proxy(
    *replies: dict[str, Any],
    on_superseded: Callable[[DataVersionSupersededError], None] | None = None,
) -> NatsProxyL3Backend:
    """
    a proxy whose broker answers each request with the next scripted reply.

    :param replies: reply dicts, in request order
    :ptype replies: dict[str, Any]
    :param on_superseded: the supersession callback to wire, if any
    :ptype on_superseded: Callable[[DataVersionSupersededError], None] | None
    :return: the proxy
    :rtype: NatsProxyL3Backend
    """
    queue = list(replies)

    async def request_raw(*, subject: Subject, payload: bytes, timeout: timedelta | None = None) -> bytes:
        del subject, payload, timeout
        return json.dumps(queue.pop(0)).encode("utf-8")

    nats = MagicMock()
    nats.request_raw = AsyncMock(side_effect=request_raw)
    return NatsProxyL3Backend(
        nats_client=nats,
        namespace_prefix="test",
        agent_id="019d9a00-0000-7000-8000-00000000000a",
        identity_token=lambda: "test-identity-token",
        on_superseded=on_superseded,
    )


class _Recorder:
    """a supersession callback that records every error it was handed."""

    def __init__(self) -> None:
        """
        starts with no calls recorded.
        """
        self.calls: list[DataVersionSupersededError] = []

    def __call__(self, error: DataVersionSupersededError) -> None:
        """
        records one call.

        :param error: the error about to be raised
        :ptype error: DataVersionSupersededError
        :return: nothing
        :rtype: None
        """
        self.calls.append(error)


class TestTheCodesAreTheBrokersContract:
    """the hub imports these names, so the two halves spell each code once."""

    def test_the_code_strings(self) -> None:
        """the wire values the broker answers."""
        assert DATA_VERSION_SUPERSEDED_ERROR_CODE == "DATA_VERSION_SUPERSEDED"
        assert DATA_VERSION_NOT_READY_ERROR_CODE == "DATA_VERSION_NOT_READY"


class TestSupersededIsFatal:
    """an older pod is told it is older, and the runtime is told once."""

    async def test_it_raises_the_typed_error_carrying_the_brokers_message(self) -> None:
        """the refusal becomes DataVersionSupersededError, not a generic unavailability."""
        proxy = _proxy(_superseded())

        with pytest.raises(DataVersionSupersededError) as raised:
            await proxy.fetch("SELECT 1")

        assert "data version 3 is older than target 4" in str(raised.value)

    async def test_the_callback_runs_before_the_raise_with_the_error_raised(self) -> None:
        """the runtime sees the very error the caller does, before the caller does."""
        recorder = _Recorder()
        proxy = _proxy(_superseded(), on_superseded=recorder)

        with pytest.raises(DataVersionSupersededError) as raised:
            await proxy.execute("UPDATE widgets SET n = 1")

        assert recorder.calls == [raised.value]

    async def test_the_callback_runs_once_across_many_refusals(self) -> None:
        """every in-flight request is refused once a pod is superseded; the exit is asked for once."""
        recorder = _Recorder()
        proxy = _proxy(_superseded(), _superseded(), _superseded(), on_superseded=recorder)

        for _ in range(3):
            with pytest.raises(DataVersionSupersededError):
                await proxy.fetch("SELECT 1")

        assert len(recorder.calls) == 1

    async def test_a_failing_callback_does_not_replace_the_error(self) -> None:
        """the runtime's own failure is logged; the caller still learns the pod is superseded."""
        calls: list[DataVersionSupersededError] = []

        def explode(error: DataVersionSupersededError) -> None:
            calls.append(error)
            raise RuntimeError("exit hook broke")

        proxy = _proxy(_superseded(), on_superseded=explode)

        with pytest.raises(DataVersionSupersededError):
            await proxy.fetch("SELECT 1")

        assert len(calls) == 1

    async def test_no_callback_still_raises(self) -> None:
        """a backend built without a callback still reports supersession by type."""
        proxy = _proxy(_superseded())

        with pytest.raises(DataVersionSupersededError):
            await proxy.fetch("SELECT 1")


class TestNotReadyIsAWait:
    """a pod at the target waits for the upgrade; nobody is told to exit."""

    async def test_it_raises_the_typed_error(self) -> None:
        """the refusal becomes DataVersionNotReadyError."""
        proxy = _proxy(_not_ready())

        with pytest.raises(DataVersionNotReadyError) as raised:
            await proxy.fetch("SELECT 1")

        assert "target 4 is not yet applied" in str(raised.value)

    async def test_the_supersession_callback_is_not_called(self) -> None:
        """waiting is not exiting."""
        recorder = _Recorder()
        proxy = _proxy(_not_ready(), on_superseded=recorder)

        with pytest.raises(DataVersionNotReadyError):
            await proxy.fetch("SELECT 1")

        assert recorder.calls == []

    async def test_not_ready_is_not_superseded(self) -> None:
        """the two answers demand opposite responses, so neither type is the other."""
        proxy = _proxy(_not_ready())

        with pytest.raises(DataVersionNotReadyError) as raised:
            await proxy.fetch("SELECT 1")

        assert not isinstance(raised.value, DataVersionSupersededError)


class TestBothAreUnavailability:
    """to code that only knows the data layer is unusable, both remain exactly that."""

    @pytest.mark.parametrize("reply", [_superseded(), _not_ready()], ids=["superseded", "not-ready"])
    async def test_each_is_a_data_layer_unavailable_error(self, reply: dict[str, Any]) -> None:
        """
        existing ``except DataLayerUnavailableError`` handlers still see the refusal.

        :param reply: the refusal reply
        :ptype reply: dict[str, Any]
        """
        proxy = _proxy(reply)

        with pytest.raises(DataLayerUnavailableError):
            await proxy.fetch("SELECT 1")


class TestEveryReplyPathDecodes:
    """a pod reads a failed reply on six paths; each must type a refusal the same way."""

    async def test_the_batch_path(self) -> None:
        """a batch refused for its data version."""
        recorder = _Recorder()
        proxy = _proxy(_superseded(), on_superseded=recorder)

        with pytest.raises(DataVersionSupersededError):
            await proxy.execute_batch([{"query": "UPDATE widgets SET n = 1", "params": []}], transaction=True)

        assert len(recorder.calls) == 1

    async def test_the_transaction_begin(self) -> None:
        """the broker refuses at ``tx.begin``, before any statement."""
        recorder = _Recorder()
        proxy = _proxy(_superseded(), on_superseded=recorder)

        with pytest.raises(DataVersionSupersededError):
            async with proxy.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("UPDATE widgets SET n = 1")

        assert len(recorder.calls) == 1

    @pytest.mark.parametrize("method", ["execute", "fetchrow", "fetch"])
    async def test_inside_a_transaction(self, method: str) -> None:
        """
        ``tx.execute`` / ``tx.fetchrow`` / ``tx.fetch``, then the rollback the error triggers.

        :param method: the connection method under test
        :ptype method: str
        """
        recorder = _Recorder()
        proxy = _proxy(
            {"success": True, "tx_id": _TX_ID},
            _superseded(),
            {"success": True},
            on_superseded=recorder,
        )

        with pytest.raises(DataVersionSupersededError):
            async with proxy.acquire() as conn:
                async with conn.transaction():
                    await getattr(conn, method)("UPDATE widgets SET n = 1")

        assert len(recorder.calls) == 1

    async def test_the_commit(self) -> None:
        """a refusal at ``tx.commit`` is typed too."""
        proxy = _proxy(
            {"success": True, "tx_id": _TX_ID},
            {"success": True, "row_count": 1},
            _not_ready(),
        )

        with pytest.raises(DataVersionNotReadyError):
            async with proxy.acquire() as conn:
                async with conn.transaction():
                    await conn.execute("UPDATE widgets SET n = 1")


class TestOtherFailuresAreUnchanged:
    """only the two data-version codes are typed; everything else keeps its type."""

    async def test_an_ordinary_refusal_stays_plain_unavailability(self) -> None:
        """a pool refusal is neither data-version type and calls nobody."""
        recorder = _Recorder()
        proxy = _proxy(
            {"success": False, "error_code": "POOL_EXHAUSTED", "error_message": "pool exhausted"},
            on_superseded=recorder,
        )

        with pytest.raises(DataLayerUnavailableError, match="POOL_EXHAUSTED") as raised:
            await proxy.fetch("SELECT 1")

        assert not isinstance(raised.value, (DataVersionSupersededError, DataVersionNotReadyError))
        assert recorder.calls == []
