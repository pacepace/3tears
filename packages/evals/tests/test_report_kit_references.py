"""The reference memos in ``docs/eval-report-kit`` build from their sources and pass the kit's render checks.

Each reference is a page source (``*.src.html``) plus its data (``*.json``); ``scripts/build.py`` inlines the kit
and the data into the built page committed beside them. Two contracts hold for every reference:

- building the source reproduces the committed page byte for byte, so the built page never drifts from its
  source, its data or the kit;
- ``scripts/check.py`` passes on the built page at 1200 and 390 px, light and dark: no console errors, no
  sideways scroll, nothing past the right edge, no overlapping chart labels (GENERATOR.md §8, items 1-3).

The render check drives headless Chromium through playwright, which not every runner has, so it carries the
``integration`` marker and runs under ``./scripts/test-integration.sh evals``. Where Chromium cannot launch it
skips and says why; a skip there is a missing browser, never a pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, render_checked, run_script

REFERENCES = sorted(p.name.removesuffix(".src.html") for p in (KIT / "references").glob("*.src.html"))


def _build(name: str, out: Path) -> Path:
    page = out / f"{name}.html"
    built = run_script("build.py", KIT / "references" / f"{name}.src.html", KIT / "references" / f"{name}.json", page)
    assert built.returncode == 0, built.stderr
    return page


def test_the_kit_holds_all_five_references() -> None:
    assert REFERENCES == [
        "r1-model-selection",
        "r2-config-sweep",
        "r3-classifier",
        "r4-tradeoff",
        "r5-kairo-round2",
    ]


@pytest.mark.parametrize("name", REFERENCES)
def test_a_reference_builds_to_its_committed_page(name: str, tmp_path: Path) -> None:
    page = _build(name, tmp_path)
    assert page.read_text() == (KIT / "references" / f"{name}.html").read_text(), (
        f"{name}.html is stale: rebuild it with scripts/build.py from its .src.html and .json"
    )


@pytest.mark.integration
@pytest.mark.parametrize("name", REFERENCES)
def test_a_reference_passes_the_render_checks(name: str, tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = _build(name, tmp_path)
    checked, runs = render_checked(page, tmp_path / "screenshots")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
