"""A browser test with no Chromium skips, unless the run requires Chromium, and then it fails and says why.

``report_kit_support.chromium_unavailable`` is how every render test asks for headless Chromium. The testament pool's
manifest and CI's evals shards set ``THREETEARS_REQUIRE_CHROMIUM=1``, so a runner that lost its browser fails those
tests instead of reporting them skipped, which reads like a pass. Here a planted test asks for the browser in a fresh
pytest process whose playwright looks for its browsers in an empty directory, so the launch fails at once and no
browser runs: with the flag the test fails with the launch's reason; without it the test skips.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The repository root: the planted test imports the support module by its repo-root name, as the render tests do.
_ROOT = Path(__file__).resolve().parents[3]

_PLANTED = """
import pytest

from packages.evals.tests.report_kit_support import chromium_unavailable


def test_asks_for_the_browser():
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
"""


def _run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, *, required: bool) -> pytest.RunResult:
    """The planted test, run with no browser to launch, with or without the run requiring one."""
    pytester.makepyfile(test_planted=_PLANTED)
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(pytester.mkdir("no-browsers")))
    monkeypatch.setenv("PYTHONPATH", str(_ROOT))
    if required:
        monkeypatch.setenv("THREETEARS_REQUIRE_CHROMIUM", "1")
    else:
        monkeypatch.delenv("THREETEARS_REQUIRE_CHROMIUM", raising=False)
    # the workspace's entry-point plugins are not what is under test, and loading them is slow
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    return pytester.runpytest_subprocess("-p", "no:cacheprovider", "-rs")


def test_a_run_that_requires_chromium_fails_a_browser_test_that_cannot_launch_it(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(pytester, monkeypatch, required=True)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*headless Chromium cannot launch, and THREETEARS_REQUIRE_CHROMIUM=1 requires it here: *Executable doesn't exist*"
        ]
    )


def test_a_run_that_does_not_require_chromium_skips_it_and_says_why(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _run(pytester, monkeypatch, required=False)
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["SKIPPED*headless Chromium cannot launch: *Executable doesn't exist*"])
