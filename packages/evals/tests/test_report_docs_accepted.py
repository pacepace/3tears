"""The report writer's acceptance runs: Kairo round 2 and the six misfits, each written live and stored.

Each run is a directory under ``fixtures/report_docs/prototype4/`` made by ``analyst_cli.py write`` (its
``replay.json`` names the fixture snapshot it was written on; ``test_report_writer_replay.py`` replays it). Here the
stored records are held to the plan's bar with no model: one run per acceptance snapshot, each ended stored on the
snapshot the fixtures still build; the Kairo document makes the reference call with the reference confidence; and
every stored document passes each invariant on its page as the store built it (the editor's findings in Method, the
made line at its foot), that page drawing strict and passing the kit's render checks in a browser.

The record checked is the one this build stores: the run's recording replayed through ``write_report``. The
``record.json`` beside it is the record the live run stored, under the report document version of its day, which a
later version refuses on read and regenerates rather than migrates; it is read here as the JSON it is, to hold the
replay to the document, the ending and the snapshot the live run stored.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.fixtures.report_plans.analyst_cli import RECORDING, REPLAY, snapshot_ids
from packages.evals.tests.fixtures.report_plans.misfits import MISFITS
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_kit_support import chromium_unavailable, render_checked
from packages.evals.tests.report_writer_support import RecordedModel, written
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report_doc_store import doc_verdicts_and_page, made_line
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import EvalReportDoc, ReportDoc, ReportFinding
from threetears.evals.testing import RecordedTurns

#: The acceptance runs.
_RUNS = Path(__file__).parent / "fixtures" / "report_docs" / "prototype4"

#: The snapshots the runs are written on, as ``analyst_cli.py`` names them on the command line.
_ACCEPTANCE = {"kairo", *MISFITS}

#: The endings a run may reach and still meet the bar.
_STORED = {"stored", "stored_with_findings"}

#: Every run, by its directory name.
_NAMES = tuple(sorted(path.parent.name for path in _RUNS.glob(f"*/{REPLAY}")))


def _replay(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((_RUNS / name / REPLAY).read_text())
    return loaded


def _live_record(name: str) -> dict[str, Any]:
    """The record the live run stored, as the JSON it is: written under its day's report document version."""
    loaded: dict[str, Any] = json.loads((_RUNS / name / "record.json").read_text())
    return loaded


@cache
def _record(name: str) -> EvalReportDoc:
    """The record this build stores for the run: its recording replayed as ``replay.json`` says it was made."""
    expected = _replay(name)
    result = asyncio.run(
        written(
            RecordedModel(RecordedTurns.replay(_RUNS / name / RECORDING), expected["model"]),
            snapshot=snapshot_ids()[expected["snapshot_of"]],
            budget_usd=expected["budget_usd"],
            withhold=expected["withhold"],
        )
    )
    assert result.stored is not None, f"{name}: the replay ended {result.ending} and stored nothing"
    return result.stored


@cache
def _inputs() -> dict[str, ReportInput]:
    """Every fixture snapshot's report input, by the snapshot's id, built once."""
    return {
        snapshot.id: ReportInput.model_validate(snapshot.report_input)
        for snapshot in fixture_snapshots(CALLER.scope_id).values()
    }


def _input(name: str) -> ReportInput:
    """The report input of the fixture snapshot the run was written on, which the fixtures must still build."""
    snapshot_of, recorded = _replay(name)["snapshot_of"], _live_record(name)["snapshot_id"]
    built = snapshot_ids()[snapshot_of]

    assert built == recorded, (
        f"{name}: written on {recorded}, but the fixtures now build {snapshot_of!r} as {built}; the fixture input "
        "changed since the run, so its document would be checked against evidence it was not written on"
    )
    return _inputs()[built]


def _stored_page(name: str) -> DocPage:
    """The run's page as the store built it: strict, the editor's findings in Method and the made line at its foot.
    Fails on any verdict, validity's or an invariant's, against that page."""
    record = _record(name)
    verdicts, page = doc_verdicts_and_page(
        ReportDoc.model_validate(record.doc),
        _input(name),
        findings=[ReportFinding.model_validate(finding) for finding in record.findings],
        made=made_line(record),
    )

    assert verdicts == [], "\n".join(f"{v.invariant} {v.path}: {v.fault}" for v in verdicts)
    assert page is not None
    return page


def test_kairo_and_every_misfit_were_written_and_stored() -> None:
    """One run per acceptance snapshot, and each ended stored, its document beside its recording and in its record."""
    runs = Counter(_replay(name)["snapshot_of"] for name in _NAMES)

    assert set(runs) == _ACCEPTANCE
    assert [snapshot_of for snapshot_of, count in sorted(runs.items()) if count > 1] == [], (
        "more than one run of these snapshots: keep the one that counts and remove the rest"
    )
    short = [
        f"{replay['snapshot_of']}: ended {replay['ending']} ({replay['reason'] or 'no reason recorded'})"
        for replay in map(_replay, _NAMES)
        if replay["ending"] not in _STORED or not replay["stored"]
    ]
    assert short == [], "\n".join(short)
    for name in _NAMES:
        record, live = _record(name), _live_record(name)
        assert live["ending"] == _replay(name)["ending"], f"{name}: the record and the replay disagree on the ending"
        assert record.ending == live["ending"], (
            f"{name}: the replay ended {record.ending}, the live run {live['ending']}"
        )
        stored = ReportDoc.model_validate_json((_RUNS / name / "doc.json").read_text())
        assert stored == ReportDoc.model_validate(live["doc"]), f"{name}: doc.json is not the document in record.json"
        assert stored == ReportDoc.model_validate(record.doc), f"{name}: the replay stored another document"
        assert record.findings == live["findings"], f"{name}: the replay stored other findings than the live run"


def test_the_kairo_document_makes_the_reference_call_with_the_reference_confidence() -> None:
    """Keep September, high if the late and uncalibrated caveats are immaterial, the late margin checked."""
    (name,) = (name for name in _NAMES if _replay(name)["snapshot_of"] == "kairo")
    doc = ReportDoc.model_validate(_record(name).doc)
    caveats = {caveat.id: caveat for caveat in doc.confidence.caveats}
    open_ids = {caveat_id for caveat_id, caveat in caveats.items() if caveat.checked is None}

    assert doc.recommends == ["September"]
    assert doc.confidence.form == "high_if_immaterial"
    assert {"judge_uncalibrated", "late:aspect", "late:control"} <= open_ids
    assert caveats["late:margin"].checked == "immaterial"


@pytest.mark.parametrize("name", _NAMES)
def test_each_stored_document_passes_every_invariant_and_renders_strict(name: str) -> None:
    """The stored record checked on its page as the store built it, then that page drawn as HTML and Markdown."""
    page = _stored_page(name)

    assert doc_page_html(page).count("<h1>") == 1
    assert sum(line.startswith("# ") for line in doc_page_markdown(page).splitlines()) == 1


@pytest.mark.integration
@pytest.mark.parametrize("name", _NAMES)
def test_each_stored_document_s_page_passes_the_render_checks(name: str, tmp_path: Path) -> None:
    """``scripts/check.py`` at 1200 and 390 px, light and dark, on each stored record's page as the store built it."""
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / f"{name}.html"
    page.write_text(doc_page_html(_stored_page(name)), encoding="utf-8")
    checked, runs = render_checked(page, tmp_path / "screenshots")

    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert len(runs) == 4 and all(line.startswith("ok") for line in runs), checked.stdout
