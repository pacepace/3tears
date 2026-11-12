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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """deselect every ``process_only`` test outside process mode.

    :param config: the session's config
    :ptype config: pytest.Config
    :param items: the collected tests, narrowed in place
    :ptype items: list[pytest.Item]
    :return: None
    :rtype: None
    """
    if os.environ.get(SERVICES_ENV) == PROCESS_MODE:
        return
    deselected = [item for item in items if item.get_closest_marker("process_only") is not None]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = [item for item in items if item.get_closest_marker("process_only") is None]


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
