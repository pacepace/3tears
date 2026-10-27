"""The ``report-doc/1`` model: the converted prototype documents read, and a document out of shape is refused.

Every converted prototype document (``fixtures/report_docs/prototype*/<name>/doc.json``) validates and is stored in the
model's canonical form. The model refuses what is out of shape on its own (an unknown slot, a claim with no evidence
or three blocks, a caveat both checked and open, an analyst caveat with no label, two claims of one id, and the rest
below), and validity refuses what a well-shaped document names that its snapshot does not hold: on the later
prototypes nothing, on the first prototype the caveat ids its analyst made up, which stay as planted negatives.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.report_doc_support import (
    LATER_DOCS,
    PROTOTYPE1_DOCS,
    converted_doc,
    snapshot_arms,
    snapshot_caveats,
)
from threetears.evals.kernel.report_doc import ReportDoc, report_doc_validity

#: The Kairo document of the third prototype, which every refusal below starts from.
_KAIRO = "prototype3/kairo"


def test_there_are_eighteen_converted_documents() -> None:
    """Seven from the first prototype, six from the second and five from the third."""
    assert (len(PROTOTYPE1_DOCS), len(LATER_DOCS)) == (7, 11)


@pytest.mark.parametrize("name", PROTOTYPE1_DOCS + LATER_DOCS)
def test_each_converted_document_validates_and_is_stored_canonical(name: str) -> None:
    """The document reads, and its file is the model's own JSON of it, so a stored read changes nothing."""
    stored = converted_doc(name)

    doc = ReportDoc.model_validate(stored)

    assert doc.model_dump(mode="json", by_alias=True) == stored
    assert json.loads(doc.model_dump_json(by_alias=True)) == stored


def _set(path: tuple[str | int, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        node: Any = doc
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value

    return change


def _evidence_times(count: int) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        block = doc["claims"][0]["evidence"][0]
        doc["claims"][0]["evidence"] = [dict(block) for _ in range(count)]

    return change


def _duplicate_claim(doc: dict[str, Any]) -> None:
    doc["claims"][1]["id"] = doc["claims"][0]["id"]


def _analyst_caveat(**fields: Any) -> Callable[[dict[str, Any]], None]:
    def change(doc: dict[str, Any]) -> None:
        doc["confidence"]["caveats"].append(
            {"id": "analyst:case-mix", "material_if": "Only if traffic differs.", **fields}
        )

    return change


def _named_on_a_claim(doc: dict[str, Any]) -> None:
    doc["claims"][0]["caveats"] = ["analyst:case-mix"]


def _recommended_twice(doc: dict[str, Any]) -> None:
    doc["recommends"] = ["September", "September"]


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (_set(("decision", 0, "slot"), "verdict"), "decision.0.slot"),
        (_evidence_times(0), "at least 1 item"),
        (_evidence_times(3), "at most 2 items"),
        (_set(("confidence", "caveats", 3, "material_if"), "Only if the margin moves."), "exactly one"),
        (_set(("confidence", "caveats", 0, "material_if"), None), "exactly one"),
        (_analyst_caveat(), "carries its label"),
        (_analyst_caveat(label="Case mix", id="analyst:Case Mix"), "lower-case letters"),
        (_duplicate_claim, "each claim has its own id"),
        (_recommended_twice, "names each arm once"),
        (_named_on_a_claim, "declared once in confidence.caveats"),
        (_set(("verdict",), "keep"), "verdict"),
        (_set(("claims", 0, "evidence", 0, "query", "colour"), "red"), "colour"),
        (_set(("claims", 0, "evidence", 0, "block"), "pie"), "block"),
        (_set(("claims", 0, "evidence", 0, "reference"), "target"), "reference"),
        (_set(("story",), "memo"), "story"),
        (_set(("schema",), "report-doc/2"), "schema"),
        (_set(("headline",), "x" * 241), "at most 240 characters"),
        (_set(("lede",), "x" * 421), "at most 420 characters"),
        (_set(("flag",), "x" * 301), "at most 300 characters"),
        (_set(("decision",), []), "at least 1 item"),
        (_set(("method",), []), "at least 1 item"),
    ],
    ids=[
        "unknown slot",
        "no evidence",
        "three evidence blocks",
        "checked and material_if",
        "neither checked nor material_if",
        "unlabelled analyst caveat",
        "analyst slug out of form",
        "duplicate claim ids",
        "an arm recommended twice",
        "an analyst caveat named but not declared",
        "an unknown field",
        "an unknown query field",
        "an unknown block kind",
        "an unknown reference line",
        "an unknown story",
        "another schema",
        "a long headline",
        "a long lede",
        "a long flag",
        "no decision entry",
        "no method item",
    ],
)
def test_a_document_out_of_shape_is_refused(change: Callable[[dict[str, Any]], None], refusal: str) -> None:
    """Each change alone makes a valid document one the model refuses, naming why."""
    doc = converted_doc(_KAIRO)
    ReportDoc.model_validate(doc)
    change(doc)

    with pytest.raises(ValidationError, match=refusal):
        ReportDoc.model_validate(doc)


def test_a_labelled_analyst_caveat_declared_in_confidence_may_be_named_on_a_claim() -> None:
    """The analyst's own caveat reads where it is declared with its label, and a claim names it by id."""
    doc = converted_doc(_KAIRO)
    _analyst_caveat(label="Case mix")(doc)
    _named_on_a_claim(doc)

    read = ReportDoc.model_validate(doc)

    assert read.claims[0].caveats == ["analyst:case-mix"]
    assert report_doc_validity(read, arms=snapshot_arms(_KAIRO), caveats=snapshot_caveats(_KAIRO)) == []


@pytest.mark.parametrize("name", LATER_DOCS)
def test_each_later_document_names_only_what_its_snapshot_holds(name: str) -> None:
    """Every arm the later prototypes recommend and every caveat they name is one their snapshot holds."""
    doc = ReportDoc.model_validate(converted_doc(name))

    assert report_doc_validity(doc, arms=snapshot_arms(name), caveats=snapshot_caveats(name)) == []


@pytest.mark.parametrize("name", [name for name in PROTOTYPE1_DOCS if name != "prototype/kairo"])
def test_the_first_prototype_s_made_up_caveat_ids_are_refused_as_planted_negatives(name: str) -> None:
    """The first prototype's analyst named caveats before code assigned ids; validity refuses each made-up one."""
    doc = ReportDoc.model_validate(converted_doc(name))
    held = snapshot_caveats(name)
    made_up = [i for i, caveat in enumerate(doc.confidence.caveats) if caveat.id not in held]

    faults = report_doc_validity(doc, arms=snapshot_arms(name), caveats=held)

    assert made_up, "a planted negative names at least one caveat its snapshot does not assign"
    assert [path for path, _ in faults] == [f"confidence.caveats[{i}].id" for i in made_up]
    assert all("caveats_list lists the caveat ids" in why for _, why in faults)


def test_validity_refuses_an_arm_and_a_caveat_the_snapshot_does_not_hold_wherever_they_are_named() -> None:
    """A recommended arm, and a code caveat id on a claim or a method item, are each refused at their path."""
    doc = converted_doc(_KAIRO)
    doc["recommends"] = ["September", "gpt-9"]
    doc["claims"][2]["caveats"] = ["late:aspect", "late:judge"]
    doc["method"][1]["caveats"] = ["borderline:pass_rate"]

    faults = report_doc_validity(
        ReportDoc.model_validate(doc), arms=snapshot_arms(_KAIRO), caveats=snapshot_caveats(_KAIRO)
    )

    assert [path for path, _ in faults] == ["recommends[1]", "claims[2].caveats[1]", "method[1].caveats[0]"]
    assert "campaign_overview lists the snapshot's arm ids" in faults[0][1]
