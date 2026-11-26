"""A switched-on table without an epoch is refused at the run that declares it (append-only L3 chunk 7).

With the change-note ring retired, a table's epoch is what every reader of what it changed reads: a restore, the
writers' backstop, a keys-unknown page. A switched-on table that carries none would have its writes heard by none of
them, so the migration run refuses it, naming how to fix it. The refusal is held behind
``epoch_catalog.REFUSE_TABLES_WITHOUT_EPOCH`` until 3tears' own ten tables declare ``ROW_STAMPED``
(``work/0950-stamp-own-tables``); the first test here fails until it is on, so the hold cannot be forgotten.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.core.ddl import epoch_catalog
from threetears.core.ddl.epoch_catalog import TableWithoutEpochError, refuse_tables_without_epoch


class _Catalogue:
    """the catalogue: which tables of the schema carry the epoch column."""

    def __init__(self, carrying: set[str]) -> None:
        self.carrying = carrying
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.asked.append((query, args))
        _, names, _ = args
        return [{"table_name": name} for name in names if name in self.carrying]


def test_todo_stamp_own_tables_the_refusal_of_a_table_without_an_epoch_is_on() -> None:
    """TODO(work/0950-stamp-own-tables): fails until that branch merges and REFUSE_TABLES_WITHOUT_EPOCH is set True.
    Then every switched-on table carries an epoch, 3tears' own ten included, and none can be declared without one."""
    assert epoch_catalog.REFUSE_TABLES_WITHOUT_EPOCH, (
        "the refusal of a switched-on table without an epoch is held off: merge work/0950-stamp-own-tables and set "
        "threetears.core.ddl.epoch_catalog.REFUSE_TABLES_WITHOUT_EPOCH = True"
    )


async def test_a_switched_on_table_without_an_epoch_is_refused_naming_how_to_fix_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a switched-on table that carries no epoch passes the run once the refusal is on, or if a table that
    carries one is refused: the catalogue's one read decides, and the error names the table and the fix."""
    monkeypatch.setattr(epoch_catalog, "REFUSE_TABLES_WITHOUT_EPOCH", True)
    catalogue = _Catalogue({"stamped_rows"})
    assert await refuse_tables_without_epoch(catalogue, "agent_reports", ["stamped_rows"]) == frozenset()
    with pytest.raises(TableWithoutEpochError, match=r"agent_reports\.plain_rows.*ROW_STAMPED"):
        await refuse_tables_without_epoch(catalogue, "agent_reports", ["stamped_rows", "plain_rows"])
    assert len(catalogue.asked) == 2, "one catalogue read a run"


async def test_while_held_a_table_without_an_epoch_is_named_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """While the refusal is held, the run names the tables without an epoch (said at WARNING by the runner)."""
    monkeypatch.setattr(epoch_catalog, "REFUSE_TABLES_WITHOUT_EPOCH", False)
    catalogue = _Catalogue(set())
    assert await refuse_tables_without_epoch(catalogue, "agent_reports", ["plain_rows"]) == frozenset({"plain_rows"})


class _Schemas:
    """the catalogue over two schemas: which tables of each carry the epoch column."""

    def __init__(self, carrying: dict[str, set[str]]) -> None:
        self.carrying = carrying
        self.asked: list[str] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        schema, names, _ = args
        self.asked.append(schema)
        return [{"table_name": name} for name in names if name in self.carrying.get(schema, set())]


async def test_a_table_of_a_second_schema_is_judged_in_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flips if a table a run's package creates in a second schema (``platform_audit.audit_events``) is judged in the
    run's schema, where it is never found: it is judged in its own, one catalogue read a schema."""
    monkeypatch.setattr(epoch_catalog, "REFUSE_TABLES_WITHOUT_EPOCH", True)
    catalogue = _Schemas({"platform": {"customers"}, "platform_audit": {"audit_events"}})
    assert await refuse_tables_without_epoch(catalogue, "platform", ["customers", "platform_audit.audit_events"]) == (
        frozenset()
    )
    assert sorted(catalogue.asked) == ["platform", "platform_audit"]
    unstamped = _Schemas({"platform": {"customers"}})
    with pytest.raises(TableWithoutEpochError, match=r"'platform_audit\.audit_events'"):
        await refuse_tables_without_epoch(unstamped, "platform", ["customers", "platform_audit.audit_events"])
