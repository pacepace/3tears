"""An editor's finding names each claim by its place on a reader's page, whatever form the claim's id takes.

The report document's schema lets a claim's id be any non-empty text (``c2``, ``c-cost``, ``lead``), and the editor
quotes the id it was shown. A reader's page never prints one: each id a finding quotes reads as the claim's place
(``claim 2``), held whole and in its own case, the longer of two ids that share a start read as itself. An id of digits
alone reads as a number, so a finding's numbers are never taken for it.

Mutations that turn this file red: claim ids matched by one fixed shape (``c`` and digits), so ``c-cost`` or ``lead``
prints raw; ids tried shortest first, so ``lead-2`` reads as ``claim 1-2``; an id of digits alone rewriting a number.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from packages.evals.tests.report_doc_support import REPORT_DOCS, snapshot_of
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding

#: A written document with five claims, whose ids it chose itself.
_NAME = "prototype5/kairo"


def _doc(ids: list[str] | None) -> dict[str, Any]:
    """Kairo's stored document, its first claims' ids replaced by ``ids`` where given."""
    doc: dict[str, Any] = json.loads((REPORT_DOCS / _NAME / "record.json").read_text(encoding="utf-8"))["doc"]
    for claim, claim_id in zip(doc["claims"], ids or [], strict=False):
        claim["id"] = claim_id
    return doc


def _page(ids: list[str] | None, finding: str) -> tuple[DocPage, str]:
    """The page built with one finding about the first claim, and that finding as the page prints it."""
    doc = ReportDoc.model_validate(_doc(ids))
    planted = ReportFinding.model_validate({"claim": doc.claims[0].id, "severity": "should_fix", "finding": finding})
    page = build(
        doc, ReportInput.model_validate(snapshot_of(_NAME).report_input), strict=False, findings=[planted], draft=True
    )
    return page, page.findings[-1].said


@pytest.mark.parametrize(
    "ids",
    [
        pytest.param(None, id="as-written"),
        pytest.param(["lead", "lead-2"], id="a-word-and-a-longer-id-it-starts"),
        pytest.param(["c3", "Q_margin"], id="numbered-and-snake"),
    ],
)
def test_a_finding_names_each_claim_by_its_place_whatever_its_id(ids: list[str] | None) -> None:
    doc = _doc(ids)
    first, second = doc["claims"][0]["id"], doc["claims"][1]["id"]
    page, said = _page(ids, f"{first}'s block repeats {second}; {second} should cite {first} instead.")
    places = {claim.id: claim.place for claim in page.claims}
    assert said == (
        f"Claim {places[first]}'s block repeats claim {places[second]}; claim {places[second]} should cite "
        f"claim {places[first]} instead."
    ), said
    leaked = [claim.id for claim in page.claims if re.search(rf"(?<!\w){re.escape(claim.id)}(?!\w)", said)]
    assert not leaked, (leaked, said)


def test_kairo_s_own_ids_read_as_names_not_numbers() -> None:
    """The stored document's ids are words, not ``c`` and a number: the case a fixed shape missed."""
    assert not any(re.fullmatch(r"c\d+", claim["id"]) for claim in _doc(None)["claims"])


def test_a_claim_id_of_digits_alone_leaves_a_finding_s_numbers_as_written() -> None:
    _, said = _page(["7"], "The lead says 7 points where its table reads 9.")
    assert said == "The lead says 7 points where its table reads 9.", said
