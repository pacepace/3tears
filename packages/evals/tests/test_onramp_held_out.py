"""The held-out apps (``onramp/held-out.jsonl``) stay sound: a twin that agrees with every label, a defect that reaches
part of the shipped data but not all of it, and an app whose own tests pass both as shipped and with its twin.

A held-out set is meant to be run rarely, so the first sign of a broken one must not be a paid run on it. These are the
same promises the development apps' slice test makes (``test_onramp_harness.py``), held over ``held-out.jsonl``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from packages.evals.onramp.held_out_shares import SHARES, copy_app, held_out_apps, measure

#: The least share of an app's shipped cases its defect must reach: a sample of twenty then sees it almost surely.
LEAST_SHARE = 0.2


def test_every_held_out_app_has_a_measure() -> None:
    assert held_out_apps(), "held-out.jsonl lists no app"
    assert sorted(held_out_apps()) == sorted(SHARES)


@pytest.mark.parametrize("app", held_out_apps())
def test_each_twin_agrees_with_every_label_and_the_defect_reaches_a_part(app: str, tmp_path: Path) -> None:
    reach = measure(app, tmp_path)
    assert reach.twin_off == 0, f"{app}: the twin is off on {reach.twin_off} cases ({reach.what})"
    assert LEAST_SHARE < reach.share < 1, f"{app}: the defect reaches {reach.reached} of {reach.total} ({reach.what})"


@pytest.mark.parametrize("twin", [False, True], ids=["app", "twin"])
@pytest.mark.parametrize("app", held_out_apps())
def test_each_apps_own_tests_pass_on_the_app_and_its_twin(app: str, twin: bool, tmp_path: Path) -> None:
    # As a trial runs them: from the app's directory, with the standard library's runner (a trial has no pytest).
    root = copy_app(app, tmp_path / app, twin=twin)
    done = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "tests"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "Ran 0 tests" not in done.stderr, f"{app} ran no tests"
