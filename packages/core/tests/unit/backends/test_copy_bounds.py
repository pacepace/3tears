"""Unit: a copy is cut by what its connection carries.

A copy cut at a thousand rows cost two and a half times one ``COPY`` of the rows on YugabyteDB: a direct
connection now streams one ``COPY``, the rail sizes each request to the cap its hub announces, and a hub
that announces none keeps the fixed batches.
"""

from __future__ import annotations

from typing import Any

from threetears.core.backends.protocol import copy_bounds


class _Streaming:
    """asyncpg's shape: a copy streams, nothing measures its request."""


class _Rail:
    def __init__(self, budget: int | None) -> None:
        self._budget = budget

    def copy_budget_bytes(self) -> int | None:
        return self._budget

    def copy_request_bytes(self, table: str, columns: Any, records: Any) -> int:
        return 0


def test_a_streaming_connection_copies_every_row_at_once() -> None:
    """Flips if a direct copy is cut into batches nothing on its path needs."""
    assert copy_bounds(_Streaming(), 50_000) == (50_000, None)


def test_the_rail_sizes_a_copy_to_its_hubs_cap() -> None:
    """Flips if a copy on the rail is cut by rows once the hub has announced its cap."""
    assert copy_bounds(_Rail(4 * 1024 * 1024), 50_000) == (50_000, 4 * 1024 * 1024)


def test_the_rail_of_a_hub_that_announces_no_cap_keeps_the_callers_batches() -> None:
    """Flips if a copy to an older hub is sent in requests it never sized for."""
    assert copy_bounds(_Rail(None), 50_000) == (None, None)
