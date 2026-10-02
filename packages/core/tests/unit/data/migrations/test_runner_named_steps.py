"""a generated step is recorded, and verified, by the name it was registered with.

a pod's table upgrade is a sequence of steps the hub generates from the
difference between two table lists. each step's body is a closure, so every
one of them has the same ``__name__``; recording that would make the ledger
say nothing about which step a version number was. :meth:`PackageMigrations.step`
registers a step under an explicit, stable name such as
``v4_step2_rename_title_to_name``, and that name is what the ledger stores and
what a later run compares against.

the upgrade path applies one package at a time through
:meth:`MigrationRunner.apply_package`, and an interrupted upgrade is resumed by
the same call. so the ledger identity check has to run there too: a resumed run
whose step numbering shifted must refuse, not skip the shifted steps as done.
"""

from __future__ import annotations

import pytest

from threetears.core.data.migrations import (
    DuplicateVersionError,
    LedgerMismatchError,
    MigrationFailedError,
    MigrationFunc,
    MigrationRunner,
    MigrationScope,
    PackageMigrations,
)

from .fake_store import FakeDataStore

_PACKAGE = "tools_pentest_sqlmap"


def _step_body(calls: list[str], label: str, *, fail: bool = False) -> MigrationFunc:
    """
    build a generated step body, as the step generator does: a closure.

    every body this returns has the same ``__name__``, which is the point.

    :param calls: list each body appends its label to when it runs
    :ptype calls: list[str]
    :param label: what this body appends
    :ptype label: str
    :param fail: raise instead of completing
    :ptype fail: bool
    :return: async step body taking a data store
    :rtype: MigrationFunc
    """

    async def body(store: object) -> None:
        """
        record that this step ran, then fail when told to.

        :param store: the run's data store, unused
        :ptype store: object
        :raises RuntimeError: when built with ``fail=True``
        """
        calls.append(label)
        if fail:
            msg = f"step {label} failed"
            raise RuntimeError(msg)

    return body


def _v4_package(calls: list[str], *, fail_at: int | None = None) -> PackageMigrations:
    """
    build the package holding version 4's three generated steps.

    :param calls: list each step body appends its name to when it runs
    :ptype calls: list[str]
    :param fail_at: step number whose body raises, if any
    :ptype fail_at: int | None
    :return: package with steps 1-3 registered under stable names
    :rtype: PackageMigrations
    """
    names = {
        1: "v4_step1_add_table_reports",
        2: "v4_step2_rename_title_to_name",
        3: "v4_step3_drop_column_state",
    }
    package = PackageMigrations(name=_PACKAGE, scope=MigrationScope.AGENT)
    for number, name in names.items():
        package.step(number, name=name)(_step_body(calls, name, fail=number == fail_at))
    return package


def _runner(package: PackageMigrations) -> MigrationRunner:
    """
    build a runner holding one package.

    :param package: the package to register
    :ptype package: PackageMigrations
    :return: runner with the package registered
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    runner.register(package)
    return runner


def _ledger(store: FakeDataStore) -> list[tuple[int, str]]:
    """
    read back the ledger rows as (version, description) pairs.

    :param store: the fake session
    :ptype store: FakeDataStore
    :return: rows in apply order
    :rtype: list[tuple[int, str]]
    """
    return [(row["version"], row["description"]) for row in store.migrations_rows]


def _store_with_applied(*rows: tuple[int, str]) -> FakeDataStore:
    """
    build a store whose ledger already records the given steps.

    :param rows: (version, description) pairs already applied
    :ptype rows: tuple[int, str]
    :return: store with the ledger present and those rows recorded
    :rtype: FakeDataStore
    """
    store = FakeDataStore()
    store.migrations_table_created = True
    for order, (version, description) in enumerate(rows, start=1):
        store.migrations_rows.append(
            {"version": version, "package": _PACKAGE, "description": description, "date_applied": order},
        )
    return store


class TestStepRegistration:
    """what ``step`` records and what it refuses."""

    def test_the_registered_name_is_the_description(self) -> None:
        """a closure's shared ``__name__`` never stands in for the step's identity."""
        package = _v4_package([])

        assert package.descriptions == {
            1: "v4_step1_add_table_reports",
            2: "v4_step2_rename_title_to_name",
            3: "v4_step3_drop_column_state",
        }

    def test_a_version_decorated_function_is_described_by_its_name(self) -> None:
        """the existing registration keeps recording the function's own name."""

        async def users_approval_state(store: object) -> None:
            """
            stand in for a hand-written migration.

            :param store: the data store, unused
            :ptype store: object
            """

        package = PackageMigrations(name="hub_platform", scope=MigrationScope.PLATFORM)
        package.version(55)(users_approval_state)

        assert package.descriptions == {55: "users_approval_state"}

    def test_a_step_number_already_taken_is_refused(self) -> None:
        """steps and versions share one numbering within a package."""
        package = _v4_package([])

        with pytest.raises(DuplicateVersionError):
            package.step(2, name="v4_step2_something_else")(_step_body([], "x"))

    def test_a_step_name_already_taken_is_refused(self) -> None:
        """two steps with one name would make the ledger unable to tell them apart."""
        package = _v4_package([])

        with pytest.raises(DuplicateVersionError, match="v4_step2_rename_title_to_name"):
            package.step(4, name="v4_step2_rename_title_to_name")(_step_body([], "x"))

    @pytest.mark.parametrize("name", ["", "   "])
    def test_a_blank_step_name_is_refused(self, name: str) -> None:
        """a blank name records nothing the identity check could compare."""
        package = PackageMigrations(name=_PACKAGE, scope=MigrationScope.AGENT)

        with pytest.raises(ValueError, match="name"):
            package.step(1, name=name)


class TestApplyPackageNamedSteps:
    """``apply_package`` records, resumes and verifies by the registered name."""

    @pytest.mark.asyncio
    async def test_a_fresh_apply_records_every_step_by_name(self) -> None:
        """the ledger holds the stable names, never the closures' ``body``."""
        calls: list[str] = []
        store = FakeDataStore()

        applied = await _runner(_v4_package(calls)).apply_package(store, _PACKAGE)

        assert applied == 3
        assert calls == [
            "v4_step1_add_table_reports",
            "v4_step2_rename_title_to_name",
            "v4_step3_drop_column_state",
        ]
        assert _ledger(store) == [
            (1, "v4_step1_add_table_reports"),
            (2, "v4_step2_rename_title_to_name"),
            (3, "v4_step3_drop_column_state"),
        ]

    @pytest.mark.asyncio
    async def test_a_resumed_run_skips_the_steps_already_applied(self) -> None:
        """an interrupted upgrade picks up at the first step the ledger lacks."""
        calls: list[str] = []
        store = _store_with_applied((1, "v4_step1_add_table_reports"))

        applied = await _runner(_v4_package(calls)).apply_package(store, _PACKAGE)

        assert applied == 2
        assert calls == ["v4_step2_rename_title_to_name", "v4_step3_drop_column_state"]
        assert _ledger(store) == [
            (1, "v4_step1_add_table_reports"),
            (2, "v4_step2_rename_title_to_name"),
            (3, "v4_step3_drop_column_state"),
        ]

    @pytest.mark.asyncio
    async def test_resuming_a_completed_run_applies_nothing(self) -> None:
        """a second resume after success is a no-op, not a re-run."""
        calls: list[str] = []
        store = FakeDataStore()
        runner = _runner(_v4_package(calls))
        await runner.apply_package(store, _PACKAGE)
        calls.clear()

        applied = await runner.apply_package(store, _PACKAGE)

        assert applied == 0
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_renamed_step_at_an_applied_number_is_refused(self) -> None:
        """a shifted numbering must not read as done: nothing runs, nothing is recorded."""
        calls: list[str] = []
        store = _store_with_applied((1, "v4_step1_add_table_reports"), (2, "v4_step2_add_index_on_title"))

        with pytest.raises(LedgerMismatchError) as excinfo:
            await _runner(_v4_package(calls)).apply_package(store, _PACKAGE)

        message = str(excinfo.value)
        assert f"{_PACKAGE}:2" in message
        assert "v4_step2_add_index_on_title" in message, "the message must name what the ledger recorded"
        assert "v4_step2_rename_title_to_name" in message, "the message must name what the code has"
        assert calls == [], "no step body may run once the ledger is known to disagree"
        inserts = [sql for sql, _ in store.executed if "INSERT INTO _schema_migrations" in sql]
        assert inserts == [], "no step may be recorded once the ledger is known to disagree"

    @pytest.mark.asyncio
    async def test_a_ledger_row_ahead_of_the_package_is_not_a_mismatch(self) -> None:
        """a step this build does not define is not compared, as for a whole scope."""
        calls: list[str] = []
        store = _store_with_applied(
            (1, "v4_step1_add_table_reports"),
            (2, "v4_step2_rename_title_to_name"),
            (3, "v4_step3_drop_column_state"),
            (4, "v5_step1_add_table_audits"),
        )

        applied = await _runner(_v4_package(calls)).apply_package(store, _PACKAGE)

        assert applied == 0
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_failed_step_leaves_no_row_and_later_steps_unrun(self) -> None:
        """a failure stops the run at that step, and its number stays pending."""
        calls: list[str] = []
        store = FakeDataStore()

        with pytest.raises(MigrationFailedError, match="v4_step2_rename_title_to_name"):
            await _runner(_v4_package(calls, fail_at=2)).apply_package(store, _PACKAGE)

        assert calls == ["v4_step1_add_table_reports", "v4_step2_rename_title_to_name"]
        assert _ledger(store) == [(1, "v4_step1_add_table_reports")]

    @pytest.mark.asyncio
    async def test_a_run_resumed_after_a_failure_starts_at_the_failed_step(self) -> None:
        """fixing the failed step and resuming runs it and the steps after it, once each."""
        failing_calls: list[str] = []
        store = FakeDataStore()
        with pytest.raises(MigrationFailedError):
            await _runner(_v4_package(failing_calls, fail_at=2)).apply_package(store, _PACKAGE)

        calls: list[str] = []
        applied = await _runner(_v4_package(calls)).apply_package(store, _PACKAGE)

        assert applied == 2
        assert calls == ["v4_step2_rename_title_to_name", "v4_step3_drop_column_state"]
        assert _ledger(store) == [
            (1, "v4_step1_add_table_reports"),
            (2, "v4_step2_rename_title_to_name"),
            (3, "v4_step3_drop_column_state"),
        ]

    @pytest.mark.asyncio
    async def test_a_scope_apply_records_steps_by_name_too(self) -> None:
        """the name reaches the ledger through every apply entry point, not only ``apply_package``."""
        store = FakeDataStore()

        await _runner(_v4_package([])).apply_for_agent_schema(store)

        assert [description for _, description in _ledger(store)] == [
            "v4_step1_add_table_reports",
            "v4_step2_rename_title_to_name",
            "v4_step3_drop_column_state",
        ]
