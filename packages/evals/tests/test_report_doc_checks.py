"""The report document's invariants (``analysis/report_doc_checks.py``): what code proves wrong about a document.

Every converted document of the second and third prototypes passes with no verdict. Each invariant then has planted
negatives, each a small change to a passing document, and each is held two ways: its invariant finds it, and with that
check removed it passes, since every other check gives the same verdicts on it as on the document it was planted in.
A check emits verdicts under its own number only, so dropping one invariant's verdicts is removing its check. The
verdicts are byte-identical whatever the document's ``story`` says. Invariant 5's page half reads the claims
renderer's text above the appendix: every converted document passes on the page it builds, and a planted negative
adds a line to a fake of that text, built from the document's own prose with its keys written out, or plants an id
in a chart's title on the rendered page.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any, get_args

import pytest

from packages.evals.tests.report_doc_support import LATER_DOCS, converted_doc, snapshot_of
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.charts import ExamplesChart
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import above_appendix_text
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report_doc_checks import DocVerdict, check_report_doc
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.report_doc import ReportDoc, Story

#: The Kairo document of the third prototype, which most negatives are planted in.
_KAIRO = "prototype3/kairo"
#: The third prototype's r2 document, whose arm ids read as ids and print as names.
_R2 = "prototype3/r2"
#: The second prototype's s1 document, whose snapshot's example cases carry digits (``tc-06``).
_S1 = "prototype2/s1"

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


def _only_at(caveat_id: str, where: tuple[str | int, ...]) -> _Change:
    """Name a caveat at one place alone: out of the confidence and every claim, then in ``where``'s caveats."""

    def change(doc: dict[str, Any]) -> None:
        _drop_caveat(caveat_id)(doc)
        for part in (*doc["claims"], *doc["method"]):
            part["caveats"] = [named for named in part["caveats"] if named != caveat_id]
        node: Any = doc
        for step in where:
            node = node[step]
        node["caveats"] = [*node["caveats"], caveat_id]

    return change


def _number_claims(doc: dict[str, Any]) -> None:
    """Name the claims ``1``, ``2``, ``3`` and on, ids of digits alone."""
    for number, claim in enumerate(doc["claims"], start=1):
        claim["id"] = str(number)


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
        ("a report-wide caveat named again beside a claim", (_set(("claims", 0, "caveats"), ["judge_uncalibrated"]),)),
        ("a report-wide caveat named again in Method", (_set(("method", 0, "caveats"), ["judge_uncalibrated"]),)),
        ("a scoped caveat named again in Method", (_set(("method", 0, "caveats"), ["confound:model"]),)),
        ("claims numbered by digits alone", (_number_claims,)),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_what_an_invariant_allows_passes(what: str, changes: tuple[_Change, ...]) -> None:
    """A percentile label, a cost's unit and a claim's own id are typed with their digits; a low confidence holds any
    caveat."""
    verdicts = _verdicts(_KAIRO, *changes)

    assert verdicts == [], f"{what}:\n{_said(verdicts)}"


# --- the rendered page's text ------------------------------------------------------------------------------------


def _rendered_above_appendix_text(doc: ReportDoc, report_input: ReportInput) -> str:
    """The claims renderer's text above the appendix, on the page the document builds strict."""
    return above_appendix_text(build(doc, report_input, strict=True))


@pytest.mark.parametrize("name", LATER_DOCS)
def test_each_later_document_passes_with_its_rendered_page_text(name: str) -> None:
    """The page each converted document builds, as a reader reads it above the appendix, types no internal id."""
    verdicts = _verdicts(name, page=_rendered_above_appendix_text)

    assert verdicts == [], _said(verdicts)


def _chart_titled(words: str) -> Callable[[ReportDoc, ReportInput], str]:
    """The rendered page's text above the appendix, its first chart a reader reads ending its title in ``words``:
    the title its values table prints as its caption."""

    def text(doc: ReportDoc, report_input: ReportInput) -> str:
        page = build(doc, report_input, strict=True)
        at, place = next(
            (at, place)
            for at, claim in enumerate(page.claims)
            for place, evidence in enumerate(claim.evidence)
            if evidence.chart is not None
            and isinstance(evidence.chart.chart, ChartBlock)
            and not isinstance(evidence.chart.chart.data, ExamplesChart)
        )
        claim = page.claims[at]
        evidence = claim.evidence[place]
        assert evidence.chart is not None
        block = evidence.chart.chart
        titled = evidence.chart.model_copy(update={"chart": block.model_copy(update={"title": block.title + words})})
        evidences = [
            *claim.evidence[:place],
            evidence.model_copy(update={"chart": titled}),
            *claim.evidence[place + 1 :],
        ]
        claims = [*page.claims[:at], claim.model_copy(update={"evidence": evidences}), *page.claims[at + 1 :]]
        return above_appendix_text(page.model_copy(update={"claims": claims}))

    return text


def _fake_above_appendix_text(doc: ReportDoc, report_input: ReportInput, extra: str = "") -> str:
    """A stand-in for the claims renderer's ``above_appendix_text(page)``, for a plant of what the renderer itself
    would print: the document's prose above the appendix with every key written out as the page prints it, example
    cards left out, and ``extra``."""
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
    _Negative(
        "typed digits beside claims numbered by digits alone",
        1,
        (_append(_RECOMMENDATION, " It wins 2 of 3 aspects."),),
        base=(_number_claims,),
    ),
    _Negative("a doubled unit", 1, (_replace(_RECOMMENDATION, _COST_KEY, f"{_COST_KEY} per 1,000 replies"),)),
    _Negative("a key that does not render", 1, (_append(_RECOMMENDATION, " {scoreboard.September.no_such_figure}"),)),
    _Negative("a query naming no measure", 2, (_set(("claims", 0, "evidence", 0, "query", "measures"), ["nope"]),)),
    _Negative("a report-wide caveat left out", 3, (_drop_caveat("judge_uncalibrated"),)),
    _Negative("a report-wide caveat only beside a claim", 3, (_only_at("judge_uncalibrated", ("claims", 0)),)),
    _Negative("a scoped caveat only in Method", 3, (_only_at("confound:model", ("method", 0)),)),
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
    _Negative("a case id in prose, in its own case", 5, (_append(("decision", 0, "text"), " See tc-06."),), doc=_S1),
    _Negative("a claim id above the appendix", 5, (_append(_RECOMMENDATION, " (see c2)"),)),
    _Negative("a claim id in another case above the appendix", 1, (_append(_RECOMMENDATION, " (see C2)"),)),
    _Negative("a measure key on the page", 5, page=_page_with("Cost | production_replicating_cost")),
    _Negative("a caveat id on the page", 5, page=_page_with("Open caveats: judge_uncalibrated")),
    _Negative("a measure key in a chart's title on the page", 5, page=_chart_titled(" by production_replicating_cost")),
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


def test_an_example_id_in_another_case_is_a_word() -> None:
    """A case id is matched in its own case: "TC-06" is not the case ``tc-06``."""
    verdicts = _verdicts(_S1, _append(("decision", 0, "text"), " See TC-06."))

    assert verdicts == [], _said(verdicts)


def test_an_id_with_no_reader_name_is_sent_to_words() -> None:
    """A trial, a case or a claim has no name the page shows, so its remedy is to say it in words, not to name it."""
    remedies = [
        verdict.remedy
        for verdict in _verdicts(_KAIRO, _append(_RECOMMENDATION, " Trial T15 shows it (see c2)."))
        if verdict.invariant == 5
    ]

    assert len(remedies) == 2
    assert all("in words" in remedy and "name the page prints" not in remedy for remedy in remedies), remedies


def _r1_with_arm(old: str, new: str, words: str = "", **fields: str) -> list[DocVerdict]:
    """The third prototype's r1 document on its snapshot, the arm ``old`` renamed ``new`` throughout and named by its id
    as an input built from a run names it (its alias is its id), with ``fields`` set on it and ``words`` appended to
    the recommendation; checked with the page it builds."""
    held = json.loads(_input("prototype3/r1").model_dump_json(by_alias=True).replace(old, new))
    arm = next(arm for arm in held["arms"] if arm["id"] == new)
    arm.update({"name": new, **fields})
    report_input = ReportInput.model_validate(held)
    raw = json.loads(json.dumps(converted_doc("prototype3/r1")).replace(old, new))
    _append(_RECOMMENDATION, words)(raw)
    doc = ReportDoc.model_validate(raw)
    return check_report_doc(doc, report_input, above_appendix_text=_rendered_above_appendix_text(doc, report_input))


#: Lever arms' ids beside the names the page prints for them: Discodon's (a hyphen and a digit), one a hyphen alone
#: marks, and one a digit alone marks.
_LEVER_ARMS = [("fanout-1", "fanout 1"), ("fan-wide", "fan wide"), ("fanout1", "fanout 1")]


@pytest.mark.parametrize(("arm_id", "printed"), _LEVER_ARMS, ids=[arm_id for arm_id, _ in _LEVER_ARMS])
def test_a_lever_arms_id_the_page_prints_by_another_name_is_refused_though_a_hyphen_or_digit_alone_marks_it(
    arm_id: str, printed: str
) -> None:
    """A lever arm's id is the campaign's label (``fanout-1``): where the page prints the arm otherwise ("fanout 1"),
    prose typing the id is refused, even where its one id mark is a hyphen or a digit, in the document and on the
    page it builds; its alias, which an input built from a run sets to the id, does not hide it."""
    assert _r1_with_arm("brisk-s", arm_id, short_name=printed) == []

    verdicts = _r1_with_arm("brisk-s", arm_id, f" Most {arm_id} answers still read well.", short_name=printed)

    said = f"types the arm id {arm_id!r}, which the page prints as {printed!r}"
    assert [(verdict.invariant, verdict.path, verdict.fault) for verdict in verdicts] == [
        (5, "decision[0].text", f"{said} above the appendix"),
        (5, "page", f"{said} on the page above the appendix"),
    ]


def test_a_model_arms_id_that_is_its_served_model_is_the_readers_word() -> None:
    """A model arm's id is the model's name a reader knows: typed beside a short name the page prints, it stands."""
    model = "deepseek/deepseek-v4.1-flash"
    fields = {"short_name": "DeepSeek Flash", "served_model": model}
    assert _r1_with_arm("atlas-m", model, **fields) == []

    verdicts = _r1_with_arm("atlas-m", model, f" The {model} replies were short.", **fields)

    assert verdicts == [], _said(verdicts)


def test_an_ordinary_hyphenated_word_and_an_arm_id_its_printed_name_differs_from_only_in_case_stand() -> None:
    """A hyphen marks only a token that is an arm's id the page prints otherwise: "cost-per-reply", "off-lane" and
    "high-effort" are words, and the id ``open-70b``, printed "Open-70B", is its own name in any case."""
    words = " The cost-per-reply gap is small, off-lane asks hold, high-effort runs cost more, and open-70b is close."

    verdicts = _r1_with_arm("brisk-s", "fanout-1", words, short_name="fanout 1")

    assert verdicts == [], _said(verdicts)


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
