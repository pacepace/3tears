"""The converted prototype report documents, and the snapshot each was written from.

Each prototype record under ``fixtures/report_docs/prototype*/<name>/`` keeps the run as it was (``report.json``,
the editor's reviews, the notes) and holds its document converted to ``report-doc/1`` as ``doc.json`` (each
prototype's README says how). Prototype 1 ran before code assigned caveat ids, so most of its ids are made up and kept
as planted negatives; prototypes 2 and 3 name only ids code assigns on their snapshots.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots, shape_snapshots
from threetears.evals.analysis.report_decision import implied_caveats, report_caveats
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_snapshot import EvalReportSnapshot

__all__ = [
    "LATER_DOCS",
    "PROTOTYPE1_DOCS",
    "REPORT_DOCS",
    "converted_doc",
    "converted_docs",
    "snapshot_arms",
    "snapshot_caveats",
    "snapshot_of",
]

#: Where the prototype records live.
REPORT_DOCS = Path(__file__).parent / "fixtures" / "report_docs"

#: The fixture snapshot each record's directory name stands for, where the two differ.
_SNAPSHOT_NAMES = {
    "kairo": KAIRO,
    "r1-bars-only": "r1 reading only bars",
    "r2-no-gates": "r2 without its gates",
    "r2-with-its-gates-read-in-each-task": "r2 with its gates read in each task",
    "r2-with-one-gate": "r2 with one gate",
    "r3-without-its-configs": "r3 without its configs",
}

#: The scope the snapshots are frozen in.
_SCOPE = "uni-1"


def _names(prototype: str) -> tuple[str, ...]:
    return tuple(sorted(f"{prototype}/{path.parent.name}" for path in (REPORT_DOCS / prototype).glob("*/doc.json")))


#: The first prototype's documents, by ``prototype/<name>``.
PROTOTYPE1_DOCS = _names("prototype")

#: The second and third prototypes' documents, which name only what their snapshots hold.
LATER_DOCS = _names("prototype2") + _names("prototype3")


def converted_doc(name: str) -> dict[str, Any]:
    """One converted document as stored, read fresh so a test may change it.

    :param name: ``prototype<n>/<name>``.
    :ptype name: str
    :return: Its ``doc.json``.
    :rtype: dict[str, Any]
    """
    loaded: dict[str, Any] = json.loads((REPORT_DOCS / name / "doc.json").read_text())
    return loaded


def converted_docs() -> dict[str, dict[str, Any]]:
    """Every converted document, by ``prototype<n>/<name>``.

    :return: The documents.
    :rtype: dict[str, dict[str, Any]]
    """
    return {name: converted_doc(name) for name in PROTOTYPE1_DOCS + LATER_DOCS}


@cache
def _snapshots() -> dict[str, EvalReportSnapshot]:
    return fixture_snapshots(_SCOPE) | shape_snapshots(_SCOPE)


def snapshot_of(name: str) -> EvalReportSnapshot:
    """The snapshot a converted document was written from.

    :param name: ``prototype<n>/<name>``.
    :ptype name: str
    :return: The snapshot, unsaved.
    :rtype: EvalReportSnapshot
    """
    directory = name.split("/", 1)[1]
    return _snapshots()[_SNAPSHOT_NAMES.get(directory, directory)]


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


def snapshot_arms(name: str) -> set[str]:
    """The arm ids of the snapshot a converted document was written from.

    :param name: ``prototype<n>/<name>``.
    :ptype name: str
    :return: The arm ids.
    :rtype: set[str]
    """
    return {arm.id for arm in _input(name).arms}


def snapshot_caveats(name: str) -> set[str]:
    """The caveat ids code assigns on the snapshot a converted document was written from, stated and implied.

    :param name: ``prototype<n>/<name>``.
    :ptype name: str
    :return: The caveat ids.
    :rtype: set[str]
    """
    held = _input(name)
    return {caveat.id for caveat in report_caveats(held)} | {caveat.id for caveat in implied_caveats(held)}
