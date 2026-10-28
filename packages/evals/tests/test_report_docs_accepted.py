"""The report writer's acceptance runs: Kairo round 2 and the six misfits, each written live and stored.

Each run is a directory under ``fixtures/report_docs/prototype4/`` made by ``analyst_cli.py write`` (its
``replay.json`` names the fixture snapshot it was written on; ``test_report_writer_replay.py`` replays it). Here the
stored documents are held to the plan's bar, read from disk with no model: every run ended stored, the Kairo
document makes the reference call with the reference confidence, and every document passes each invariant and
renders strict, its page passing the kit's render checks in a browser.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.analyst_cli import REPLAY
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.doc_page.text import above_appendix_text
from threetears.evals.analysis.report_doc_checks import check_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

#: The acceptance runs.
_RUNS = Path(__file__).parent / "fixtures" / "report_docs" / "prototype4"

#: Kairo round 2 as ``analyst_cli.py`` names it on the command line.
_KAIRO_NAME = "kairo"

#: The endings a run may reach and still meet the bar.
_STORED = {"stored", "stored_with_findings"}

#: Every run, by its directory name.
_NAMES = tuple(sorted(path.parent.name for path in _RUNS.glob(f"*/{REPLAY}")))


def _replay(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((_RUNS / name / REPLAY).read_text())
    return loaded


@cache
def _input(name: str) -> ReportInput:
    """The report input of the fixture snapshot the run was written on."""
    snapshot_of = _replay(name)["snapshot_of"]
    snapshot = fixture_snapshots(CALLER.scope_id)[KAIRO if snapshot_of == _KAIRO_NAME else snapshot_of]
    return ReportInput.model_validate(snapshot.report_input)


def _doc(name: str) -> ReportDoc:
    return ReportDoc.model_validate_json((_RUNS / name / "doc.json").read_text())


def test_kairo_and_every_misfit_were_written_and_stored() -> None:
    """One run per acceptance snapshot, and each ended stored, its document beside its recording."""
    written = {_replay(name)["snapshot_of"]: name for name in _NAMES}

    assert set(written) == {_KAIRO_NAME, *MISFITS}
    short = [
        f"{snapshot_of}: ended {replay['ending']} ({replay['reason'] or 'no reason recorded'})"
        for snapshot_of, replay in ((snapshot_of, _replay(name)) for snapshot_of, name in sorted(written.items()))
        if replay["ending"] not in _STORED or not replay["stored"]
    ]
    assert short == [], "\n".join(short)
    for snapshot_of, name in written.items():
        assert (_RUNS / name / "doc.json").is_file(), f"{snapshot_of}: stored, but no doc.json"


def test_the_kairo_document_makes_the_reference_call_with_the_reference_confidence() -> None:
    """Keep September, high if the late and uncalibrated caveats are immaterial, the late margin checked."""
    name = next(name for name in _NAMES if _replay(name)["snapshot_of"] == _KAIRO_NAME)
    doc = _doc(name)
    caveats = {caveat.id: caveat for caveat in doc.confidence.caveats}
    open_ids = {caveat_id for caveat_id, caveat in caveats.items() if caveat.checked is None}

    assert doc.recommends == ["September"]
    assert doc.confidence.form == "high_if_immaterial"
    assert {"judge_uncalibrated", "late:aspect", "late:control"} <= open_ids
    assert caveats["late:margin"].checked == "immaterial"


@pytest.mark.parametrize("name", _NAMES)
def test_each_stored_document_passes_every_invariant_and_renders_strict(name: str) -> None:
    """The document as stored, checked with its page's text above the appendix, then drawn as HTML and Markdown."""
    doc, report_input = _doc(name), _input(name)
    page = build(doc, report_input, strict=True)
    verdicts = check_report_doc(doc, report_input, above_appendix_text=above_appendix_text(page))

    assert verdicts == [], "\n".join(f"{v.invariant} {v.path}: {v.fault}" for v in verdicts)
    assert doc_page_html(page).count("<h1>") == 1
    assert sum(line.startswith("# ") for line in doc_page_markdown(page).splitlines()) == 1


@pytest.mark.integration
@pytest.mark.parametrize("name", _NAMES)
def test_each_stored_document_s_page_passes_the_render_checks(name: str, tmp_path: Path) -> None:
    """``scripts/check.py`` at 1200 and 390 px, light and dark, on the page each stored document builds."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / f"{name}.html"
    page.write_text(doc_page_html(build(_doc(name), _input(name), strict=True)), encoding="utf-8")
    checked, runs = render_checked(page, tmp_path / "screenshots")

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
