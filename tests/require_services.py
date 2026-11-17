"""pytest plugin: under ``THREETEARS_REQUIRE_SERVICES=1``, an integration test that skips at runtime fails.

The integration suite on the testament pool runs with that flag (``.testament.toml``'s
``[suites.integration]``), where every service a test needs is meant to be in the image. There a
skip would read like a pass: a fixture that could not start Postgres, a gate that found no binary,
an ``importorskip`` in a test of a module the lock dropped. :func:`threetears.core.testing.services.require_services`
already fails the gates it covers; this is the backstop for every other runtime skip of a test
marked ``integration``.

A module that skips while it is collected (a module-level ``pytest.importorskip``, or
``pytest.skip(allow_module_level=True)``) has no tests yet, so no marker to read: under the flag, one
under a ``tests/integration/`` directory fails its collection instead. An integration module
elsewhere that skips at collection is not caught.

A test skipped by an unconditional ``@pytest.mark.skip`` stays a skip: that is a decision written on
the test, visible in every report, not a service gone missing. A ``skipif`` gets no such pass,
since its condition is usually the environment the flag says is there. Without the flag nothing
changes.

**``process_only``.** A test marked ``process_only`` checks what only process mode has (the services
themselves, a cluster of child processes). Outside ``THREETEARS_TEST_SERVICES=process`` -- CI's
integration job, a laptop's testcontainers run -- it is deselected, whatever named it, rather than
run against the wrong services or skipped. ``scripts/test-integration.sh`` reads the same marker to
leave such modules out of a run it takes off the pool.

**``yugabyte_only``.** A test marked ``yugabyte_only`` proves what YugabyteDB alone does (a plan its planner picks, a
budget measured on it); on Postgres it has nothing to assert. A run that does not name YugabyteDB
(``THREETEARS_TEST_ENGINE``, PostgreSQL when unnamed, as ``tests/integration/conftest.py`` reads it) deselects it,
rather than skipping it: the pool's suite runs on Postgres, where a skip fails. A run naming YugabyteDB runs it, and
:func:`threetears.core.testing.fixtures.check_engine` stops that run when its server is not YugabyteDB.

**A run on YugabyteDB proves the database.** The pool runs the integration suite twice: on PostgreSQL
(``[suites.integration]``) and again on a YugabyteDB of each test process's own (``[suites.integration_yugabyte]``,
``THREETEARS_TEST_ENGINE=yugabyte``). Its engine changes only what a test asks the session's database
(``db_container``, named directly or through a fixture over it), so a run naming YugabyteDB deselects every
integration test that never reaches it (a NATS round trip, a backup of a Postgres of the test's own): the PostgreSQL
run proved it, and the second run would only repeat it. A test marked ``postgres_only`` proves what PostgreSQL alone
does (its major, its ``pg_dump``), the mirror of ``yugabyte_only``: a run naming YugabyteDB deselects it too. A
``yugabyte_only`` test stays, whatever it reaches.
"""

from __future__ import annotations

import os
from collections.abc import Generator

import pytest

#: the flag the pool's integration suite sets
REQUIRE_SERVICES_ENV = "THREETEARS_REQUIRE_SERVICES"

#: the variable that selects the services' process mode, and its value
SERVICES_ENV = "THREETEARS_TEST_SERVICES"
PROCESS_MODE = "process"

#: the variable naming the engine a run means its database to be, and the value that runs ``yugabyte_only``
ENGINE_ENV = "THREETEARS_TEST_ENGINE"
YUGABYTE = "yugabyte"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """deselect every ``process_only`` test outside process mode, every ``yugabyte_only`` test in a run that does
    not name YugabyteDB, and, in a run that does, every ``postgres_only`` test and every integration test that never
    reaches the session database.

    :param config: the session's config
    :ptype config: pytest.Config
    :param items: the collected tests, narrowed in place
    :ptype items: list[pytest.Item]
    :return: None
    :rtype: None
    """
    away: list[str] = []
    if os.environ.get(SERVICES_ENV) != PROCESS_MODE:
        away.append("process_only")
    if os.environ.get(ENGINE_ENV) != YUGABYTE:
        away.append("yugabyte_only")
    on_yugabyte = os.environ.get(ENGINE_ENV) == YUGABYTE
    if on_yugabyte:
        away.append("postgres_only")
    deselected = [
        item
        for item in items
        if any(item.get_closest_marker(name) is not None for name in away)
        or (on_yugabyte and _repeats_postgres_run(item))
    ]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        dropped = set(map(id, deselected))
        items[:] = [item for item in items if id(item) not in dropped]


#: the session database's fixture: a test whose fixtures reach it is one a run on YugabyteDB proves anew
DATABASE_FIXTURE = "db_container"


def _repeats_postgres_run(item: pytest.Item) -> bool:
    """whether a run on YugabyteDB would only repeat ``item``'s PostgreSQL run: an integration test, not marked
    ``yugabyte_only``, none of whose fixtures (their closure: a fixture over ``db_container`` counts) is the session
    database.

    :param item: a collected test
    :ptype item: pytest.Item
    :return: True for a test the YugabyteDB run leaves out
    :rtype: bool
    """
    if item.get_closest_marker("integration") is None or item.get_closest_marker("yugabyte_only") is not None:
        return False
    return DATABASE_FIXTURE not in getattr(item, "fixturenames", ())


#: the directory part every package's integration tests live under
_INTEGRATION_DIR = "/tests/integration/"


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    """turn a skipped collection of a module under ``tests/integration/`` into a failed one while the flag is set.

    :param collector: the collector
    :ptype collector: pytest.Collector
    :return: the collection's report, as pytest built it or turned into a failure
    :rtype: Generator[None, pytest.CollectReport, pytest.CollectReport]
    """
    report: pytest.CollectReport = yield
    if (
        os.environ.get(REQUIRE_SERVICES_ENV) == "1"
        and report.skipped
        and isinstance(collector, pytest.Module)
        and _INTEGRATION_DIR in f"/{collector.nodeid}"
    ):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        report.outcome = "failed"
        report.longrepr = (
            f"{collector.nodeid} skipped while it was collected ({reason.removeprefix('Skipped: ')}), and "
            f"{REQUIRE_SERVICES_ENV}=1 makes an integration module's skip a failure"
        )
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    """turn an integration test's runtime skip into a failure while the flag is set.

    :param item: the test
    :ptype item: pytest.Item
    :return: the phase's report, as pytest built it or turned into a failure
    :rtype: Generator[None, pytest.TestReport, pytest.TestReport]
    """
    report: pytest.TestReport = yield
    if (
        os.environ.get(REQUIRE_SERVICES_ENV) == "1"
        and report.skipped
        and not hasattr(report, "wasxfail")
        and item.get_closest_marker("integration") is not None
        and item.get_closest_marker("skip") is None
    ):
        reason = report.longrepr[2] if isinstance(report.longrepr, tuple) else str(report.longrepr)
        reason = reason.removeprefix("Skipped: ")
        report.outcome = "failed"
        report.longrepr = (
            f"{item.nodeid} skipped at runtime ({reason}), and {REQUIRE_SERVICES_ENV}=1 makes an "
            f"integration test's skip a failure: the services it needs are meant to be here"
        )
    return report
