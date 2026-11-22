"""the migration runner announces a run's writes to switched tables: its window, handed to its announcer.

A migration writes rows with no collection and no advance; until 0.95 the write-mark trigger caught them. Now the run
hands its window (the database's clock at its start and end, and its schema) to its announcer, the process's
``WriteBackstop``, which advances every switched table the window's change notes name.
"""

from __future__ import annotations

from datetime import datetime

from threetears.core.data.migrations import MigrationRunner, MigrationScope, PackageMigrations
from threetears.core.data.store import DataStore

from .fake_store import FakeDataStore


class _Announcer:
    def __init__(self) -> None:
        self.windows: list[tuple[datetime, datetime, str | None]] = []

    async def announce_window(self, since: datetime, until: datetime, *, schema: str | None = None) -> None:
        self.windows.append((since, until, schema))


async def test_the_runner_announces_a_run_that_applied_a_migration_and_no_other() -> None:
    """Flips if a migration's writes to switched tables reach no follower (nothing advances for them), or a run that
    applied nothing announces a window."""
    announcer = _Announcer()
    runner = MigrationRunner(announcer=announcer)
    package = PackageMigrations(name="rows", scope=MigrationScope.PLATFORM)

    async def write_rows(store: DataStore) -> None:
        await store.execute("UPDATE rows SET n = n + 1")

    package.version(1)(write_rows)
    runner.register(package)
    store = FakeDataStore(schema="platform")
    assert await runner.apply_for_platform_schema(store) == 1
    ((since, until, schema),) = announcer.windows
    assert since <= until and schema == "platform"
    assert await runner.apply_for_platform_schema(store) == 0
    assert len(announcer.windows) == 1
