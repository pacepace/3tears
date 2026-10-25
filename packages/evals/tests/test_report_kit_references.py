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

import subprocess
import sys
from pathlib import Path

import pytest

KIT = Path(__file__).resolve().parents[3] / "docs" / "eval-report-kit"
REFERENCES = sorted(p.name.removesuffix(".src.html") for p in (KIT / "references").glob("*.src.html"))


def _run_script(script: str, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(KIT / "scripts" / script), *(str(a) for a in args)],
        capture_output=True,
        text=True,
        check=False,
    )


def _build(name: str, out: Path) -> Path:
    page = out / f"{name}.html"
    built = _run_script("build.py", KIT / "references" / f"{name}.src.html", KIT / "references" / f"{name}.json", page)
    assert built.returncode == 0, built.stderr
    return page


def _chromium_unavailable() -> str | None:
    """Why headless Chromium cannot launch here, or ``None`` when it can."""
    probe = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "from playwright.sync_api import sync_playwright\n"
            "with sync_playwright() as p:\n"
            "    p.chromium.launch().close()\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return None
    # playwright boxes its install advice after the error itself; the error line names the cause
    errors = [line.strip() for line in probe.stderr.splitlines() if "Error" in line]
    return errors[-1] if errors else f"exit code {probe.returncode}: {probe.stderr.strip()[-300:]}"


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
    reason = _chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = _build(name, tmp_path)
    checked = _run_script("check.py", page, Path("--out"), tmp_path / "screenshots")
    runs = [line for line in checked.stdout.splitlines() if "px" in line and "scrollWidth=" in line]
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
