"""Unit tests for :class:`SchemaPrimingIntegration`.

Exercises the agent-side integration's duck-typed read path: one-time name->id
resolution over the collection's ``l3_pool``, the memo discipline (a NON-EMPTY
success is cached; a zero-row success or a transient fault is NOT, so the next turn
retries), the soft-fails to empty (no collection / no pool / no names), and the
by-primary-key digest read.

The digest collection is stubbed by its duck-typed surface (``l3_pool`` +
``get``); the integration imports no concrete collection, so a stub with the same
shape suffices.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid7

from threetears.datasources.schema_priming import SchemaPrimingIntegration


class _StubPool:
    """asyncpg-pool stand-in whose ``fetch`` replays a scripted sequence.

    each entry in ``responses`` is either a list of rows to return or an exception
    instance to raise, popped in order; ``calls`` counts invocations so a test can
    assert the resolution ran once (memoized) or retried.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def fetch(self, query: str, *args: Any) -> Any:
        self.calls += 1
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _StubCollection:
    """digest-collection stand-in exposing ``l3_pool`` + an ``async get``."""

    def __init__(self, pool: Any = None, digests: dict[Any, Any] | None = None) -> None:
        self.l3_pool = pool
        self._digests = digests or {}

    async def get(self, datasource_id: Any) -> Any:
        return self._digests.get(datasource_id)


def _row(identifier: Any) -> dict[str, Any]:
    """build a fetch row exposing ``row["id"]``."""
    return {"id": identifier}


class TestDatasourceIds:
    def test_resolves_and_memoizes_nonempty(self) -> None:
        ds_id = uuid7()
        pool = _StubPool([[_row(ds_id)]])
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(pool),
            customer_id=uuid7(),
            datasource_names=["sales"],
        )
        first = asyncio.run(integration.datasource_ids())
        second = asyncio.run(integration.datasource_ids())
        assert first == [ds_id]
        assert second == [ds_id]
        # a non-empty success is memoized -> the scan runs exactly once.
        assert pool.calls == 1

    def test_zero_row_result_is_not_memoized(self) -> None:
        ds_id = uuid7()
        # first call resolves nothing (not yet applied), second call resolves it.
        pool = _StubPool([[], [_row(ds_id)]])
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(pool),
            customer_id=uuid7(),
            datasource_names=["sales"],
        )
        assert asyncio.run(integration.datasource_ids()) == []
        assert asyncio.run(integration.datasource_ids()) == [ds_id]
        assert pool.calls == 2

    def test_transient_fault_soft_fails_and_retries(self) -> None:
        ds_id = uuid7()
        pool = _StubPool([RuntimeError("proxy not ready"), [_row(ds_id)]])
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(pool),
            customer_id=uuid7(),
            datasource_names=["sales"],
        )
        # the fault soft-fails to [] WITHOUT memoizing; the next turn retries.
        assert asyncio.run(integration.datasource_ids()) == []
        assert asyncio.run(integration.datasource_ids()) == [ds_id]
        assert pool.calls == 2

    def test_no_collection_returns_empty(self) -> None:
        integration = SchemaPrimingIntegration(
            digest_collection=None,
            customer_id=uuid7(),
            datasource_names=["sales"],
        )
        assert asyncio.run(integration.datasource_ids()) == []

    def test_no_names_returns_empty(self) -> None:
        pool = _StubPool([[_row(uuid7())]])
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(pool),
            customer_id=uuid7(),
            datasource_names=[],
        )
        assert asyncio.run(integration.datasource_ids()) == []
        assert pool.calls == 0

    def test_no_pool_returns_empty(self) -> None:
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(pool=None),
            customer_id=uuid7(),
            datasource_names=["sales"],
        )
        assert asyncio.run(integration.datasource_ids()) == []


class TestGetDigest:
    def test_reads_digest_by_primary_key(self) -> None:
        ds_id = uuid7()
        entity = object()
        integration = SchemaPrimingIntegration(
            digest_collection=_StubCollection(digests={ds_id: entity}),
        )
        assert asyncio.run(integration.get_digest(ds_id)) is entity

    def test_missing_digest_returns_none(self) -> None:
        integration = SchemaPrimingIntegration(digest_collection=_StubCollection())
        assert asyncio.run(integration.get_digest(uuid7())) is None

    def test_no_collection_returns_none(self) -> None:
        integration = SchemaPrimingIntegration(digest_collection=None)
        assert asyncio.run(integration.get_digest(uuid7())) is None
