"""The ``report-doc/1`` model: the converted prototype documents read, and a document out of shape is refused.

Every converted prototype document (``fixtures/report_docs/prototype*/<name>/doc.json``) validates and is stored in the
model's canonical form. The model refuses what is out of shape on its own (an unknown slot, a claim with no evidence
or three blocks, a caveat both checked and open, an analyst caveat with no label, two claims of one id, and the rest
below), and validity refuses what a well-shaped document names that its snapshot does not hold: on the later
prototypes nothing, on the first prototype the caveat ids its analyst made up, which stay as planted negatives.
"""

from __future__ import annotations

import json
import types
from collections.abc import Callable, Iterator
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from packages.evals.tests.report_doc_support import (
    LATER_DOCS,
    PROTOTYPE1_DOCS,
    converted_doc,
    snapshot_arms,
    snapshot_caveats,
)
from threetears.evals.kernel.report_doc import REPORT_DOC_PROSE_FIELDS, ReportDoc, report_doc_validity
from threetears.evals.schema.prose import field_is_prose

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


#: The documents whose every id their snapshot holds: the later prototypes, and the first prototype's Kairo run, whose
#: analyst already named code's caveat ids.
_VALID = ("prototype/kairo", *LATER_DOCS)

#: The caveat ids the first prototype's analysts made up before code assigned ids, by document: the planted negatives.
_MADE_UP: dict[str, tuple[str, ...]] = {
    "prototype/r1": ("resolves_borderline",),
    "prototype/r1-bars-only": ("judge_not_stated", "unpaired_intervals", "no_traffic_mix", "overall_scale_not_stated"),
    "prototype/r2": (
        "pick_clears_gate_on_point_estimate",
        "selection_of_the_one_inside",
        "case_mix_vs_traffic",
        "latency_from_harness",
    ),
    "prototype/r2-no-gates": ("cost_basis", "no_pass_rate_interval", "best_of_many", "pass_undefined"),
    "prototype/r3": ("overall_interval_inferred", "gold_label_agreement"),
    "prototype/r4": ("f64_guardrail_clearance", "f16_facts_miss", "judge_not_described", "contrasts_uncorrected"),
}


def test_every_converted_document_is_either_valid_or_a_planted_negative() -> None:
    """The two lists below cover the eighteen documents, once each."""
    assert sorted(_VALID + tuple(_MADE_UP)) == sorted(PROTOTYPE1_DOCS + LATER_DOCS)


@pytest.mark.parametrize("name", _VALID)
def test_each_valid_document_names_only_what_its_snapshot_holds(name: str) -> None:
    """Every arm these documents recommend and every caveat they name is one their snapshot holds."""
    doc = ReportDoc.model_validate(converted_doc(name))

    assert report_doc_validity(doc, arms=snapshot_arms(name), caveats=snapshot_caveats(name)) == []


@pytest.mark.parametrize("name", sorted(_MADE_UP))
def test_the_first_prototype_s_made_up_caveat_ids_are_refused_as_planted_negatives(name: str) -> None:
    """The first prototype's analysts named caveats before code assigned ids; validity refuses exactly those."""
    doc = ReportDoc.model_validate(converted_doc(name))

    faults = report_doc_validity(doc, arms=snapshot_arms(name), caveats=snapshot_caveats(name))

    refused = [doc.confidence.caveats[int(path.split("[")[1].split("]")[0])].id for path, _ in faults]
    assert all(path.startswith("confidence.caveats[") and path.endswith("].id") for path, _ in faults)
    assert tuple(refused) == _MADE_UP[name]
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


#: The tradeoff documents: each states a lean, "not a recommendation", so each recommends nothing (ANALYST §5).
_TRADEOFFS = ("prototype/r1-bars-only", "prototype/r2-no-gates", "prototype/r4", "prototype3/r4")


@pytest.mark.parametrize("name", _TRADEOFFS)
def test_a_tradeoff_s_lean_is_not_a_recommendation(name: str) -> None:
    """A tradeoff whose lean stays in its prose recommends no arm."""
    doc = ReportDoc.model_validate(converted_doc(name))

    assert doc.story == "tradeoff"
    assert doc.recommends == []


#: The text fields of a report document that hold ids, not prose: what code matches, never what a reader reads.
_ID_FIELDS = frozenset(
    {
        "recommends[]",
        "confidence.caveats[].id",
        "claims[].id",
        "claims[].caveats[]",
        "claims[].evidence[].emphasis",
        "claims[].evidence[].query.aspect",
        "claims[].evidence[].query.measures[]",
        "claims[].evidence[].query.arms[]",
        "claims[].evidence[].query.strata[]",
        "method[].caveats[]",
    }
)


def _text_fields(model: type[BaseModel], prefix: str = "") -> Iterator[tuple[str, bool]]:
    """Every text field under ``model`` by path, with whether it is typed model prose; vocabularies are skipped."""
    for name, field in model.model_fields.items():
        path = f"{prefix}{field.alias or name}"
        annotation: Any = field.annotation
        suffix = ""
        while True:
            origin = get_origin(annotation)
            if origin in (Union, types.UnionType):
                annotation = next(arg for arg in get_args(annotation) if arg is not type(None))
            elif origin is list:
                annotation, suffix = get_args(annotation)[0], suffix + "[]"
            elif origin is Annotated:
                annotation = get_args(annotation)[0]
            else:
                break
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            yield from _text_fields(annotation, f"{path}{suffix}.")
        elif annotation is str:
            yield f"{path}{suffix}", field_is_prose(field)
        else:
            assert get_origin(annotation) is Literal or annotation in (bool, int), (path, annotation)


def test_the_prose_fields_are_every_text_field_a_reader_reads() -> None:
    """REPORT_DOC_PROSE_FIELDS and the id fields split every text field once, and every typed prose field is prose.

    A text field added to the document goes red here until it is named one or the other, so a check over a
    document's prose (invariants 1 and 5) walks every field a reader reads, typed model prose or not.
    """
    fields = dict(_text_fields(ReportDoc))

    assert len(REPORT_DOC_PROSE_FIELDS) == len(set(REPORT_DOC_PROSE_FIELDS))
    assert set(REPORT_DOC_PROSE_FIELDS).isdisjoint(_ID_FIELDS)
    assert set(fields) == set(REPORT_DOC_PROSE_FIELDS) | _ID_FIELDS
    assert {path for path, prose in fields.items() if prose} <= set(REPORT_DOC_PROSE_FIELDS)
    assert {path for path in REPORT_DOC_PROSE_FIELDS if not fields[path]} == {
        "explore[].outcome",
        "next[].settles",
        "confidence.caveats[].label",
    }
