"""the workspace's Hypothesis profile: no wall-clock bound on a property test.

Hypothesis bounds each example's run (``deadline``) and the time it takes to draw inputs (the ``too_slow``
health check) by wall clock. On the testament pool a test shares its runner with other batches, so either bound
fails a test whose property holds, on the runner's load alone. The root conftest loads a profile with both off;
these tests pin it, and that no test or module sets either back.
"""

from __future__ import annotations

import re
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_the_workspace_profile_is_loaded() -> None:
    """the profile every test runs under sets no deadline and skips the too_slow check."""
    assert settings.default.deadline is None
    assert HealthCheck.too_slow in settings.default.suppress_health_check


@given(st.lists(st.integers(), max_size=200))
def test_a_property_test_runs_under_it(values: list[int]) -> None:
    """a ``@given`` test with no settings of its own runs under the loaded profile, with no deadline to miss."""
    assert settings.default.deadline is None
    assert sorted(sorted(values)) == sorted(values)


def test_no_test_sets_a_wall_clock_bound_back() -> None:
    """a ``deadline=`` or a ``settings(`` that names one in a Hypothesis test file would undo the profile."""
    offenders = []
    for path in sorted([*(_REPO_ROOT / "packages").glob("**/tests/**/*.py"), *(_REPO_ROOT / "tests").glob("**/*.py")]):
        if ".venv" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "hypothesis" not in text or path == Path(__file__).resolve():
            continue
        if re.search(r"@settings\([^)]*deadline\s*=\s*(?!None)", text) or "HealthCheck.all()" in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == [], f"a Hypothesis test sets a wall-clock bound back: {offenders}"
