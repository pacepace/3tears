"""under THREETEARS_REQUIRE_SERVICES=1 an integration test's runtime skip fails; nothing else changes."""

from __future__ import annotations

import pytest

_TESTS = """
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def service():
    pytest.skip("Docker not available")


def test_a_skip_in_the_body():
    pytest.skip("no binary")


def test_a_skip_in_a_fixture(service):
    pass


@pytest.mark.skipif(True, reason="the environment lacks it")
def test_a_skipif():
    pass


@pytest.mark.skip(reason="a microbenchmark, run by hand")
def test_a_static_skip():
    pass


@pytest.mark.xfail(reason="known")
def test_an_xfail():
    assert False


def test_a_pass():
    pass
"""

_UNIT = """
import pytest


def test_a_unit_skip():
    pytest.skip("not an integration test")
"""


#: a module that skips while it is collected, as a module-level importorskip of a missing module does
_GONE = """
import pytest

pytest.importorskip("threetears_no_such_module")
"""


def _session(pytester: pytest.Pytester) -> pytest.RunResult:
    pytester.makeconftest('pytest_plugins = ["tests.require_services"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n")
    pytester.makepyfile(test_integration=_TESTS, test_unit=_UNIT)
    pytester.makepyfile(**{"pkg/tests/integration/test_gone_live": _GONE, "pkg/tests/unit/test_gone_unit": _GONE})
    # a failed collection stops a session by default; carry on, to count every outcome
    return pytester.runpytest_inprocess(
        "-p", "no:cacheprovider", "-p", "no:asyncio", "-rA", "--continue-on-collection-errors"
    )


def test_with_the_flag_every_runtime_skip_of_an_integration_test_fails(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THREETEARS_REQUIRE_SERVICES", "1")
    result = _session(pytester)
    # the body fails; the fixture and the skipif skip in setup, so they error; so does the integration
    # module that skipped while collected; the static skip and the two unit skips stay skips
    result.assert_outcomes(passed=1, failed=1, errors=3, skipped=3, xfailed=1)
    result.stdout.fnmatch_lines(
        ["*test_a_skip_in_the_body skipped at runtime (no binary)*THREETEARS_REQUIRE_SERVICES=1*"],
        consecutive=False,
    )
    result.stdout.fnmatch_lines(["*test_gone_live.py skipped while it was collected*"], consecutive=False)


def test_without_the_flag_skips_are_skips(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THREETEARS_REQUIRE_SERVICES", raising=False)
    result = _session(pytester)
    result.assert_outcomes(passed=1, skipped=7, xfailed=1)


_PROCESS_ONLY = """
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.process_only]


def test_a_process_mode_check():
    pass
"""


def _named_session(pytester: pytest.Pytester) -> pytest.RunResult:
    """a session that names the process-only module itself, as an off-pool run of the suite's paths does."""
    pytester.makeconftest('pytest_plugins = ["tests.require_services"]')
    pytester.makeini("[pytest]\nmarkers =\n    integration: needs a service\n    process_only: process mode only\n")
    pytester.makepyfile(test_two_nodes=_PROCESS_ONLY, test_other="def test_other():\n    pass\n")
    return pytester.runpytest_inprocess(
        "-p", "no:cacheprovider", "-p", "no:asyncio", "test_two_nodes.py", "test_other.py"
    )


def test_outside_process_mode_a_process_only_test_is_deselected_even_when_named(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("THREETEARS_TEST_SERVICES", raising=False)
    result = _named_session(pytester)
    # deselected, not skipped: nothing reads like a pass
    result.assert_outcomes(passed=1, deselected=1)


def test_in_process_mode_a_process_only_test_runs(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THREETEARS_TEST_SERVICES", "process")
    result = _named_session(pytester)
    result.assert_outcomes(passed=2)
