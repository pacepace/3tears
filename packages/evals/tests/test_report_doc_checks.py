"""The report document's invariants (``analysis/report_doc_checks.py``): what code proves wrong about a document.

Every converted document of the second and third prototypes passes with no verdict. Each invariant then has planted
negatives, each a small change to a passing document, and each is held two ways: its invariant finds it, and with that
check removed it passes, since every other check gives the same verdicts on it as on the document it was planted in.
A check emits verdicts under its own number only, so dropping one invariant's verdicts is removing its check. The
verdicts are byte-identical whatever the document's ``story`` says. Invariant 5's page half reads the claims
renderer's text above the appendix, which is chunk R's and not on this branch: a fake stands in for it, built from the
document's own prose with its keys written out.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, get_args

import pytest

from packages.evals.tests.report_doc_support import LATER_DOCS, converted_doc, snapshot_of
from threetears.evals.analysis.report_doc_checks import DocVerdict, check_report_doc
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.report_doc import ReportDoc, Story

#: The Kairo document of the third prototype, which most negatives are planted in.
_KAIRO = "prototype3/kairo"
#: The third prototype's r2 document, whose arm ids read as ids and print as names.
_R2 = "prototype3/r2"

#: A change to a document as stored.
_Change = Callable[[dict[str, Any]], None]


@cache
def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


def _verdicts(
    name: str,
    *changes: _Change,
    after: Callable[[ReportDoc], ReportDoc] | None = None,
    page: Callable[[ReportDoc, ReportInput], str] | None = None,
) -> list[DocVerdict]:
    raw = converted_doc(name)
    for change in changes:
        change(raw)
    doc = ReportDoc.model_validate(raw)
    if after is not None:
        doc = after(doc)
    text = page(doc, _input(name)) if page is not None else ""
    return check_report_doc(doc, _input(name), above_appendix_text=text)


def _said(verdicts: list[DocVerdict]) -> str:
    return "\n".join(f"{v.invariant} {v.path}: {v.fault}" for v in verdicts)


# --- changes -------------------------------------------------------------------------------------------------------


def _append(where: tuple[str | int, ...], words: str) -> _Change:
    def change(doc: dict[str, Any]) -> None:
        node: Any = doc
        for step in where[:-1]:
            node = node[step]
        node[where[-1]] += words

    return change


def _set(where: tuple[str | int, ...], value: Any) -> _Change:
    def change(doc: dict[str, Any]) -> None:
        node: Any = doc
        for step in where[:-1]:
            node = node[step]
        node[where[-1]] = value

    return change


def _replace(where: tuple[str | int, ...], old: str, new: str) -> _Change:
    def change(doc: dict[str, Any]) -> None:
        node: Any = doc
        for step in where[:-1]:
            node = node[step]
        assert old in node[where[-1]]
        node[where[-1]] = node[where[-1]].replace(old, new)

    return change


def _caveat(caveat_id: str, **fields: Any) -> _Change:
    """Change one confidence caveat's fields."""

    def change(doc: dict[str, Any]) -> None:
        entry = next(entry for entry in doc["confidence"]["caveats"] if entry["id"] == caveat_id)
        entry.update(fields)

    return change


def _drop_caveat(caveat_id: str) -> _Change:
    def change(doc: dict[str, Any]) -> None:
        doc["confidence"]["caveats"] = [entry for entry in doc["confidence"]["caveats"] if entry["id"] != caveat_id]

    return change


def _repeat_caveat(caveat_id: str) -> _Change:
    def change(doc: dict[str, Any]) -> None:
        entry = next(entry for entry in doc["confidence"]["caveats"] if entry["id"] == caveat_id)
        doc["confidence"]["caveats"].append(dict(entry))

    return change


def _only_checked_caveats(doc: dict[str, Any]) -> None:
    """Keep only the caveats checked immaterial, with a plain ``high``: no form fault, every open caveat left out."""
    doc["confidence"]["caveats"] = [entry for entry in doc["confidence"]["caveats"] if entry["checked"]]
    doc["confidence"]["form"] = "high"


def _no_method(doc: ReportDoc) -> ReportDoc:
    """The document with no Method item, which the model refuses on its own: the invariant holds it for a document
    built without validation too."""
    return doc.model_copy(update={"method": []})


#: Where Kairo's recommendation reads.
_RECOMMENDATION = ("decision", 0, "text")
#: The key in it that prints a cost per 1,000 replies.
_COST_KEY = "{scoreboard.September.reply_cost_usd|usd_per_1k}"


# --- the converted documents pass ----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", LATER_DOCS)
def test_each_later_converted_document_passes_every_invariant(name: str) -> None:
    """The eleven documents of the second and third prototypes name only what their snapshots hold, and pass."""
    verdicts = _verdicts(name)

    assert verdicts == [], _said(verdicts)


@pytest.mark.parametrize(
    ("what", "changes"),
    [
        ("a percentile label", (_append(_RECOMMENDATION, " Its p95 reply is no slower."),)),
        ("a cost's unit", (_append(_RECOMMENDATION, " Costs are read per 1k replies, or $/1,000."),)),
        ("a claim id in Method", (_append(("method", 0, "text"), " The cost claim is c1."),)),
        ("a low confidence with open caveats", (_set(("confidence", "form"), "low"),)),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_what_an_invariant_allows_passes(what: str, changes: tuple[_Change, ...]) -> None:
    """A percentile label, a cost's unit and a claim's own id are typed with their digits; a low confidence holds any
    caveat."""
    verdicts = _verdicts(_KAIRO, *changes)

    assert verdicts == [], f"{what}:\n{_said(verdicts)}"


# --- the rendered page's text, by a fake of the claims renderer's ------------------------------------------------


def _fake_above_appendix_text(doc: ReportDoc, report_input: ReportInput, extra: str = "") -> str:
    """A stand-in for the claims renderer's ``above_appendix_text(page)`` (chunk R, not on this branch): the
    document's prose above the appendix with every key written out as the page prints it, example cards left out,
    and ``extra`` for what the renderer itself would print."""
    names = {arm.id: arm.name for arm in fact_arms(report_input)}
    texts = [doc.headline, doc.lede, doc.flag or ""]
    texts += [entry.text for entry in doc.decision]
    texts += [part for claim in doc.claims for part in (claim.text, *(block.caption for block in claim.evidence))]
    texts += [step.text for step in doc.next]
    return "\n".join([*(render(text, report_input, names) for text in texts), extra])


def _page_with(extra: str) -> Callable[[ReportDoc, ReportInput], str]:
    return lambda doc, report_input: _fake_above_appendix_text(doc, report_input, extra)


@pytest.mark.parametrize("name", LATER_DOCS)
def test_each_later_document_passes_with_its_page_text(name: str) -> None:
    """The page's own text above the appendix, the keys written out, types no internal id on any converted document."""
    verdicts = _verdicts(name, page=_fake_above_appendix_text)

    assert verdicts == [], _said(verdicts)


def test_a_plain_high_with_no_open_caveat_breaks_no_form() -> None:
    """``high`` with every caveat checked holds the form; the open caveats it left out are coverage's to find."""
    verdicts = _verdicts(_KAIRO, _only_checked_caveats)

    assert verdicts and {verdict.invariant for verdict in verdicts} == {3}, _said(verdicts)


# --- planted negatives ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Negative:
    """One planted fault: the document it is planted in (with any change first), the plant, and its invariant."""

    name: str
    invariant: int
    plant: tuple[_Change, ...] = ()
    doc: str = _KAIRO
    base: tuple[_Change, ...] = ()
    after: Callable[[ReportDoc], ReportDoc] | None = None
    page: Callable[[ReportDoc, ReportInput], str] | None = None


_NEGATIVES = [
    _Negative("a typed digit", 1, (_append(_RECOMMENDATION, " It is 12 percent cheaper."),)),
    _Negative("a doubled unit", 1, (_replace(_RECOMMENDATION, _COST_KEY, f"{_COST_KEY} per 1,000 replies"),)),
    _Negative("a key that does not render", 1, (_append(_RECOMMENDATION, " {scoreboard.September.no_such_figure}"),)),
    _Negative("a query naming no measure", 2, (_set(("claims", 0, "evidence", 0, "query", "measures"), ["nope"]),)),
    _Negative("a report-wide caveat left out", 3, (_drop_caveat("judge_uncalibrated"),)),
    _Negative("a report-wide caveat beside a claim", 3, (_set(("claims", 0, "caveats"), ["judge_uncalibrated"]),)),
    _Negative("a scoped caveat in Method", 3, (_set(("method", 0, "caveats"), ["confound:model"]),)),
    _Negative("a caveat listed twice", 3, (_repeat_caveat("judge_uncalibrated"),)),
    _Negative(
        "a caveat checked that code does not prove",
        3,
        (_caveat("judge_uncalibrated", material_if=None, checked="immaterial"),),
    ),
    _Negative("a plain high with open caveats", 4, (_set(("confidence", "form"), "high"),)),
    _Negative(
        "high_if_immaterial with no open caveat",
        4,
        (_set(("confidence", "form"), "high_if_immaterial"),),
        base=(_only_checked_caveats,),
    ),
    _Negative("a raw arm id", 5, (_append(_RECOMMENDATION, " So atlas-m.off.r0 stays out."),), doc=_R2),
    _Negative("a measure key", 5, (_append(_RECOMMENDATION, " production_replicating_cost reads it."),)),
    _Negative("a slot word", 5, (_append(_RECOMMENDATION, " See why_not."),)),
    _Negative("a caveat id", 5, (_append(_RECOMMENDATION, " The late:margin holds."),)),
    _Negative("a card's id in prose", 5, (_append(_RECOMMENDATION, " Trial T15 shows it."),)),
    _Negative("a measure key on the page", 5, page=_page_with("Cost | production_replicating_cost")),
    _Negative("a caveat id on the page", 5, page=_page_with("Open caveats: judge_uncalibrated")),
    _Negative("no Method item", 6, after=_no_method),
]

_PLANTED = pytest.mark.parametrize("negative", _NEGATIVES, ids=[negative.name for negative in _NEGATIVES])


def _base(negative: _Negative) -> list[DocVerdict]:
    return _verdicts(negative.doc, *negative.base, page=_fake_above_appendix_text if negative.page else None)


def _planted(negative: _Negative) -> list[DocVerdict]:
    return _verdicts(negative.doc, *negative.base, *negative.plant, after=negative.after, page=negative.page)


@_PLANTED
def test_each_planted_negative_is_found_by_its_invariant(negative: _Negative) -> None:
    """The plant adds a verdict of its invariant to what the document it was planted in already draws."""
    base = _base(negative)
    found = [verdict for verdict in _planted(negative) if verdict not in base]

    assert found, f"{negative.name}: no verdict"
    assert {verdict.invariant for verdict in found} == {negative.invariant}, _said(found)


@_PLANTED
def test_each_planted_negative_passes_once_its_check_is_removed(negative: _Negative) -> None:
    """With its invariant's check removed, the plant draws exactly the verdicts the document it was planted in does:
    no other check stands in for it, so each check is the one thing that refuses its fault."""
    others = [verdict for verdict in _planted(negative) if verdict.invariant != negative.invariant]
    base = [verdict for verdict in _base(negative) if verdict.invariant != negative.invariant]

    assert others == base, _said(others)


def test_validity_comes_first_and_the_invariants_still_run() -> None:
    """A recommended arm the snapshot does not hold is validity's verdict, first; a fault beside it is found too."""
    verdicts = _verdicts(
        _KAIRO, _set(("recommends",), ["nobody"]), _append(_RECOMMENDATION, " It is 12 percent cheaper.")
    )

    assert [(verdict.invariant, verdict.path) for verdict in verdicts] == [
        (0, "recommends[0]"),
        (1, "decision[0].text"),
    ]


def test_a_verdict_names_where_it_sits_and_how_to_mend_it() -> None:
    """Each verdict carries its field path, the fault and a remedy that sends the analyst somewhere."""
    verdicts = _verdicts(_KAIRO, _set(("claims", 0, "evidence", 0, "query", "measures"), ["nope"]))

    assert verdicts
    assert all(verdict.path.startswith("claims[0].evidence[0]") for verdict in verdicts), _said(verdicts)
    assert any("nope" in verdict.fault for verdict in verdicts), _said(verdicts)
    assert all("block_render" in verdict.remedy for verdict in verdicts)


# --- story is read by no check -------------------------------------------------------------------------------------

#: A document drawing a verdict of every invariant the prose and the confidence can show, so a story that moved one
#: would show.
_EVERY_FAULT = (
    _append(_RECOMMENDATION, " It is 12 percent cheaper; see why_not."),
    _set(("claims", 0, "evidence", 0, "query", "measures"), ["nope"]),
    _drop_caveat("judge_uncalibrated"),
    _set(("confidence", "form"), "high"),
)


@pytest.mark.parametrize("name", [*LATER_DOCS, "planted"])
def test_the_verdicts_are_byte_identical_whatever_the_story(name: str) -> None:
    """``story`` is the analyst's label: no check reads it."""
    doc, faults = (_KAIRO, _EVERY_FAULT) if name == "planted" else (name, ())
    said = {
        story: [verdict.model_dump_json() for verdict in _verdicts(doc, *faults, _set(("story",), story))]
        for story in get_args(Story)
    }

    assert len(set(map(tuple, said.values()))) == 1
    if name == "planted":
        assert {int(verdict.invariant) for verdict in _verdicts(doc, *faults)} == {1, 2, 3, 4, 5}
