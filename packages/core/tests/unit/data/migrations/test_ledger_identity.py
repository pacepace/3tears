"""a recorded migration must be the migration the code has at that version.

THE INCIDENT THIS CLOSES. A feature branch renumbered its migrations to make
room for one that landed on the mainline: versions 55-65 became 56-66, and the
mainline's own ``users_approval_state`` took 55. Against a database that had
already applied the OLD numbering, the runner skipped all eleven -- it decides
what is pending from ``(version, package)`` alone, and every one of those
version numbers was already recorded. ``users_approval_state`` therefore never
ran, while its ten neighbours stayed recorded under names belonging to the
migration one number below them.

Nothing said so. The runner logged no warning, reported zero pending, and the
service came up healthy. The failure surfaced hours later and three layers
away, as ``asyncpg.exceptions.UndefinedColumnError: column "approval_state" of
relation "users" does not exist`` on an unrelated admin endpoint -- a symptom
that points at the endpoint, not at the ledger.

The information needed to catch it was already being written: ``_run_one``
records ``func.__name__`` in ``_schema_migrations.description`` on every apply.
It was simply never read back. So the runner now compares what the ledger says
version N was against what the code has at version N, and refuses to touch a
database where they disagree.

Only versions the code still defines are compared. A ledger row ahead of the
code -- an older deployment pointed at a database a newer one has migrated --
is a different condition with a different remedy, and failing this check for it
would turn a routine staged rollout into an outage.
"""

from __future__ import annotations

import pytest

from threetears.core.data.migrations import (
    MigrationScope,
    PackageMigrations,
)
from threetears.core.data.migrations.errors import LedgerMismatchError
from threetears.core.data.migrations.runner import MigrationRunner

from ._fake_store import FakeDataStore


async def users_approval_state(store: object) -> None:
    """stand in for the mainline migration that took version 55.

    :param store: the data store, unused
    :ptype store: object
    :return: nothing
    :rtype: None
    """
    return None


async def relation_layer_extension(store: object) -> None:
    """stand in for the branch migration that USED to be version 55.

    :param store: the data store, unused
    :ptype store: object
    :return: nothing
    :rtype: None
    """
    return None


async def dataset_definitions(store: object) -> None:
    """stand in for the migration after the renumber.

    :param store: the data store, unused
    :ptype store: object
    :return: nothing
    :rtype: None
    """
    return None


def _renumbered_package() -> PackageMigrations:
    """build the package as it looks AFTER the renumber.

    :return: package with 55=users_approval_state, 56=relation_layer_extension
    :rtype: PackageMigrations
    """
    package = PackageMigrations(name="hub_platform", scope=MigrationScope.PLATFORM)
    package.version(55)(users_approval_state)
    package.version(56)(relation_layer_extension)
    package.version(57)(dataset_definitions)
    return package


def _store_with_pre_renumber_ledger() -> FakeDataStore:
    """build a store whose ledger records the OLD numbering.

    :return: store with 55=relation_layer_extension, 56=dataset_definitions
    :rtype: FakeDataStore
    """
    store = FakeDataStore()
    store.migrations_rows.extend(
        [
            {"version": 55, "package": "hub_platform", "description": "relation_layer_extension", "date_applied": 1},
            {"version": 56, "package": "hub_platform", "description": "dataset_definitions", "date_applied": 2},
        ],
    )
    return store


class TestLedgerIdentity:
    """the ledger must name the same migration the code does, per version."""

    @pytest.mark.asyncio
    async def test_a_renumbered_ledger_is_refused(self) -> None:
        """the exact shape of the incident: eleven silently-skipped migrations."""
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        with pytest.raises(LedgerMismatchError):
            await runner.apply_for_platform_schema(_store_with_pre_renumber_ledger())

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_version_and_both_migrations(self) -> None:
        """an operator must not have to diff the ledger by hand to act on this."""
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        with pytest.raises(LedgerMismatchError) as excinfo:
            await runner.apply_for_platform_schema(_store_with_pre_renumber_ledger())

        message = str(excinfo.value)
        assert "55" in message
        assert "relation_layer_extension" in message, "the message must name what the ledger recorded"
        assert "users_approval_state" in message, "the message must name what the code expects"

    @pytest.mark.asyncio
    async def test_nothing_is_applied_when_the_ledger_disagrees(self) -> None:
        """the check runs BEFORE any migration body, not between them.

        Applying the pending tail of a ledger already known to be wrong would
        write new rows into a bookkeeping table nobody can yet trust.
        """
        store = _store_with_pre_renumber_ledger()
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        with pytest.raises(LedgerMismatchError):
            await runner.apply_for_platform_schema(store)

        inserts = [sql for sql, _ in store.executed if "INSERT INTO _schema_migrations" in sql]
        assert inserts == [], "no migration may be recorded once the ledger is known to disagree"

    @pytest.mark.asyncio
    async def test_a_matching_ledger_applies_the_pending_tail(self) -> None:
        """the check must not stand in the way of an ordinary upgrade."""
        store = FakeDataStore()
        store.migrations_rows.extend(
            [
                {
                    "version": 55,
                    "package": "hub_platform",
                    "description": "users_approval_state",
                    "date_applied": 1,
                },
                {
                    "version": 56,
                    "package": "hub_platform",
                    "description": "relation_layer_extension",
                    "date_applied": 2,
                },
            ],
        )
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        applied = await runner.apply_for_platform_schema(store)

        assert applied == 1, "only version 57 was pending"

    @pytest.mark.asyncio
    async def test_a_ledger_row_ahead_of_the_code_is_not_a_mismatch(self) -> None:
        """a database migrated by a NEWER deployment is a rollout, not a fault.

        Refusing here would turn an ordinary staged rollout -- old pods still
        serving while new ones migrate -- into an outage.
        """
        store = FakeDataStore()
        store.migrations_rows.extend(
            [
                {
                    "version": 55,
                    "package": "hub_platform",
                    "description": "users_approval_state",
                    "date_applied": 1,
                },
                {
                    "version": 56,
                    "package": "hub_platform",
                    "description": "relation_layer_extension",
                    "date_applied": 2,
                },
                {
                    "version": 57,
                    "package": "hub_platform",
                    "description": "dataset_definitions",
                    "date_applied": 3,
                },
                {
                    "version": 58,
                    "package": "hub_platform",
                    "description": "a_migration_this_build_has_never_heard_of",
                    "date_applied": 4,
                },
            ],
        )
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        applied = await runner.apply_for_platform_schema(store)

        assert applied == 0

    @pytest.mark.asyncio
    async def test_an_empty_ledger_is_not_a_mismatch(self) -> None:
        """a fresh database has nothing to disagree with."""
        runner = MigrationRunner()
        runner.register(_renumbered_package())

        applied = await runner.apply_for_platform_schema(FakeDataStore())

        assert applied == 3
