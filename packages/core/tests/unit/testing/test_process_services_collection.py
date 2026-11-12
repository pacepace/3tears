"""core's integration conftest collects the process-services checks only under process mode, where they fail, not skip.

Outside process mode (CI's integration job, a laptop's testcontainers run) those checks have nothing to check, and
collecting them added a skip per test to every such run. Under the pool's integration suite they must still be
collected, and a service they cannot get must fail them. The conftest runs here as written, beside a stand-in for
the module, in a session of its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_CONFTEST = Path(__file__).resolve().parents[2] / "integration" / "conftest.py"

_PROCESS_SERVICES = """
from threetears.core.testing.services import require_services


def test_a_service_it_cannot_get():
    require_services("no such service here")
"""


def _session(pytester: pytest.Pytester) -> pytest.RunResult:
    pytester.makeconftest(_CONFTEST.read_text(encoding="utf-8"))
    pytester.makepyfile(test_process_services=_PROCESS_SERVICES, test_other="def test_other():\n    pass\n")
    return pytester.runpytest_subprocess("-p", "no:cacheprovider")


def test_outside_process_mode_the_module_is_not_collected(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("THREETEARS_TEST_SERVICES", raising=False)
    monkeypatch.delenv("THREETEARS_REQUIRE_SERVICES", raising=False)
    result = _session(pytester)
    # the sibling is collected, so the directory was; the module is not, so nothing skips
    result.assert_outcomes(passed=1)


def test_under_the_pool_suite_s_env_it_is_collected_and_a_missing_service_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THREETEARS_TEST_SERVICES", "process")
    monkeypatch.setenv("THREETEARS_REQUIRE_SERVICES", "1")
    result = _session(pytester)
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*no such service here*THREETEARS_REQUIRE_SERVICES=1*"])
