"""The uuid7 cutoff bound and the new driver surface — pure pieces of the drill machinery."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid7

import pytest

from threetears.backup.compare import uuid7_upper_bound
from threetears.backup.drivers import PostgresDriver, YugabyteDriver, driver_by_name


class TestUuid7UpperBound:
    def test_ids_minted_before_the_moment_sort_under_the_bound(self) -> None:
        earlier = uuid7()
        bound = uuid7_upper_bound(datetime.now(UTC) + timedelta(seconds=1))
        assert earlier < bound

    def test_ids_minted_after_the_moment_sort_over_the_bound(self) -> None:
        bound = uuid7_upper_bound(datetime.now(UTC) - timedelta(seconds=2))
        later = uuid7()
        assert later > bound

    def test_the_bound_is_a_valid_version_7_uuid(self) -> None:
        bound = uuid7_upper_bound(datetime.now(UTC))
        assert bound.version == 7


class TestDriverByName:
    def test_resolves_both_known_drivers(self) -> None:
        assert isinstance(driver_by_name("postgres"), PostgresDriver)
        assert isinstance(driver_by_name("yugabyte"), YugabyteDriver)

    def test_an_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no dump driver named"):
            driver_by_name("oracle")


class TestGlobalsArgv:
    def test_postgres_globals_dump_excludes_role_passwords(self) -> None:
        argv = PostgresDriver().dump_globals_argv("postgresql://h/db")
        assert argv[0] == "pg_dumpall"
        assert "--globals-only" in argv
        assert "--no-role-passwords" in argv

    def test_yugabyte_globals_dump_uses_its_own_fork(self) -> None:
        argv = YugabyteDriver().dump_globals_argv("postgresql://h/db")
        assert argv[0] == "ysql_dumpall"
        assert "--globals-only" in argv

    def test_sql_replay_paths_stop_on_first_error(self) -> None:
        for driver in (PostgresDriver(), YugabyteDriver()):
            argv = driver.restore_sql_argv("postgresql://h/db")
            assert "ON_ERROR_STOP=1" in argv


class TestClusterEnumeration:
    def test_toolchain_transient_databases_are_never_backed_up(self) -> None:
        """scratch and verify databases are the toolchain's own; enumerating them
        makes a backup contain the previous restore's scratch, and one caught
        mid-drop hangs the enumeration connect."""
        from threetears.backup.cluster import _EXCLUDED_PREFIXES  # noqa: PLC0415

        assert "scratch_restore_x".startswith(_EXCLUDED_PREFIXES)
        assert "verify_restore_x".startswith(_EXCLUDED_PREFIXES)
        assert not "aibots".startswith(_EXCLUDED_PREFIXES)
