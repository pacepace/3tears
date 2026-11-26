"""a run capped below a package's last migration leaves that package's switched-on tables unjudged for their epoch.

``target`` builds an older shape on purpose (an upgrade path's proof builds to the version before the one it proves,
writes the rows that version must carry forward, then runs on): a package whose epoch migration comes later than the
cap has, by design, no epoch yet. The end-of-run refusal (``switched_tables_without_epoch``) judges the tables of each
package the run brought to its last migration, so a capped run is not refused for a migration it was told not to
run, and the run that reaches the package's last migration still judges every table it declares.
"""

from __future__ import annotations

import pytest

from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.data.store import DataStore

from .fake_store import FakeDataStore


def _runner() -> MigrationRunner:
    """one package declaring a switched-on table, in two migrations: create it, then (in a real package) stamp it.

    :return: the runner
    :rtype: MigrationRunner
    """
    package = PackageMigrations(name="rows", scope=MigrationScope.PLATFORM)

    async def create_rows(store: DataStore) -> None:
        await store.execute("CREATE TABLE IF NOT EXISTS rows (id UUID PRIMARY KEY)")

    async def stamp_rows(store: DataStore) -> None:
        await store.execute("ALTER TABLE rows ADD COLUMN IF NOT EXISTS epoch BIGINT")

    package.version(1)(create_rows)
    package.version(2)(stamp_rows)
    package.mark_write_generation("rows")
    runner = MigrationRunner()
    runner.register(package)
    return runner


async def test_a_run_capped_below_the_package_s_last_migration_is_not_refused() -> None:
    """Flips if a capped run (``target`` below the package's last migration) is refused for a table its later
    migration stamps: the cap built the older shape it was asked for."""
    store = FakeDataStore(schema="platform")
    assert await _runner().apply_for_platform_schema(store, target=1) == 1


async def test_the_run_that_reaches_the_last_migration_still_judges_the_table() -> None:
    """Flips if the refusal is skipped for a package the run brought to its last migration: the catalogue (the fake
    store's, which finds no table carrying an epoch) is read, and the table is refused, whether the run was capped at
    the last version or not capped at all."""
    for target in (None, 2):
        store = FakeDataStore(schema="platform")
        with pytest.raises(RuntimeError, match=r"platform\.rows"):
            await _runner().apply_for_platform_schema(store, target=target)


async def test_a_package_already_at_its_last_migration_is_judged_by_a_capped_run() -> None:
    """Flips if a capped run skips a package whose every migration is already applied: it is at its last migration
    whatever the cap, so its tables are judged."""
    store = FakeDataStore(schema="platform")
    store.migrations_table_created = True
    store.migrations_rows = [
        {"version": 1, "package": "rows", "description": "create_rows", "date_applied": 1},
        {"version": 2, "package": "rows", "description": "stamp_rows", "date_applied": 2},
    ]
    with pytest.raises(RuntimeError, match=r"platform\.rows"):
        await _runner().apply_for_platform_schema(store, target=1)
