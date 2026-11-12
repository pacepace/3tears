"""An editor's finding names each claim by its place on a reader's page, whatever form the claim's id takes.

The report document's schema lets a claim's id be any non-empty text (``c2``, ``c-cost``, ``lead``), and the editor
quotes the id it was shown. A reader's page never prints one that reads as a tag, as the document's checks read it
(``reads_as_a_tag``): each such id a finding quotes reads as the claim's place (``claim 2``), held whole and in its own
case, the longer of two ids that share a start read as itself. A plain word (``lead``) reads as the word and an id of
digits alone as a number, so neither is rewritten; nor is an id inside the words a figure key was written out as.

Mutations that turn this file red: claim ids matched by one fixed shape (``c`` and digits), so ``c-cost`` or
``Q_margin`` prints raw; ids tried shortest first, so ``c3-b`` reads as ``claim 1-b``; a plain word or an id of digits
alone rewritten; an id rewritten inside a figure's words (``character adherence (v2)``); an id read at the start of a
longer hyphenated word (``c-cost-share`` as ``claim 2-share``).
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
        pytest.param(["c3", "c3-b"], id="an-id-and-a-longer-id-it-starts"),
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


def test_a_claim_id_that_is_a_plain_word_reads_as_the_word() -> None:
    page, said = _page(["lead", "c-cost"], "The lead repeats c-cost; lead with the cost instead.")
    places = {claim.id: claim.place for claim in page.claims}
    assert said == f"The lead repeats claim {places['c-cost']}; lead with the cost instead.", said


def test_a_claim_id_inside_a_longer_hyphenated_word_stands() -> None:
    """``c-cost-share`` is a word of its own, not the claim ``c-cost`` and a tail: only the id standing whole moves."""
    page, said = _page(["c-cost"], "The c-cost-share row repeats c-cost.")
    place = next(claim.place for claim in page.claims if claim.id == "c-cost")
    assert said == f"The c-cost-share row repeats claim {place}.", said


def test_a_claim_id_inside_a_figure_s_words_stands() -> None:
    """``aspect.name`` reads ``character adherence (v2)`` on this input: a claim ``v2`` is named in the editor's own
    words, never inside the aspect's name the key was written out as."""
    page, said = _page(["v2"], "v2's block reads `aspect.name` as its aspect.")
    place = next(claim.place for claim in page.claims if claim.id == "v2")
    assert said == f"Claim {place}'s block reads character adherence (v2) as its aspect.", said


def test_a_claim_id_of_digits_alone_leaves_a_finding_s_numbers_as_written() -> None:
    _, said = _page(["7"], "The lead says 7 points where its table reads 9.")
    assert said == "The lead says 7 points where its table reads 9.", said
