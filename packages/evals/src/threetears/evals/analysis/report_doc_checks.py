"""The report document's invariants: what code can prove wrong about a valid ``report-doc/1`` on its snapshot.

:func:`check_report_doc` returns every verdict at once, each a :class:`DocVerdict` naming the invariant, the field
path, what is wrong and where to go to mend it, so one revision can mend them all. They go back to the analyst as
text, and storing a document re-runs them and refuses on any. ``docs/eval-report-kit/ANALYST.md`` §7 lists the
invariants; the W3 plan §3 says how each is checked.

**Every check is mechanical.** A check reads the document's fields by path and the snapshot's report input, and
nothing else: no code path here reads ``story``, a slot's word as a choice, a claim's weight or a block's kind to
decide what to check, so the verdicts are byte-identical when only ``story`` changes. What is interpretive (a
recommended arm failing a gate, a call that disagrees with dominance, a margin missing from Watch) is a fact handed to
the editor and never a verdict here.

**Validity first** (:func:`~threetears.evals.kernel.report_doc.report_doc_validity`), as invariant ``0``: an arm or a
caveat id the snapshot does not hold. Its verdicts come first; the invariants still run, and pass over an id validity
refused, so a revision sees everything at once.

1. **Numbers come through figure keys.** Every key in a prose field (read by
   :func:`~threetears.evals.kernel.report_doc.report_doc_prose`) renders on the snapshot
   (:func:`~threetears.evals.analysis.report.figure_keys.render`); no prose types a number outside its keys, except the
   names the snapshot states (arm, aspect, measure, setting and the other names
   :func:`~threetears.evals.analysis.report.figure_keys.literal_numbers` sets aside), the percentile labels (``p95``),
   a cost's unit (``per 1,000``, ``per 1k``, ``$/1k``) and the document's own claim ids (``c1``, never an id of digits
   alone, which reads as a number; above the appendix invariant 5 refuses a claim id); and no prose writes
   a unit right after a key that prints it, nor keys sub-cent money per reply on a page that states its costs per
   1,000 (:mod:`~threetears.evals.analysis.report_prose_checks`).
2. **Every evidence block resolves and renders**, as ``block_render`` draws it: completed
   (:func:`~threetears.evals.analysis.report.block_spec.complete_chart`), held to the chart rules
   (:func:`~threetears.evals.analysis.report.block_spec.chart_faults`,
   :func:`~threetears.evals.analysis.report.block_spec.small_multiple_faults`) and drawn by the page's resolver
   (:func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart`).
3. **Caveat coverage.** Each caveat code assigns, stated or implied
   (:func:`~threetears.evals.analysis.report_decision.every_caveat`) and touching the call's confidence lands, and only
   where its reach allows: a report-wide one in the confidence; a scoped one in the confidence or beside a claim. A
   method-level one (latency not compared, an unreported observation, an apparatus confound that is not report-wide)
   does not touch the confidence (ANALYST §7, invariant 3), so it may land in either place or a Method item, or not
   at all. Once a caveat lands where its reach allows, naming it again beside a claim or in Method is the analyst's
   emphasis, not a fault. Each id appears once in the confidence, and ``checked: "immaterial"`` stands only where
   :func:`~threetears.evals.analysis.report_decision.caveat_materiality` proves it.
4. **The confidence form** matches its caveats: ``high`` holds no open caveat, ``high_if_immaterial`` at least one (each
   open caveat states its ``material_if``), and ``low`` holds whatever it holds.
5. **No internal id above the appendix.** The prose above it (everything but ``omitted`` and ``method``) types no id
   that reads as one (:func:`~threetears.evals.analysis.report_prose_checks.reads_as_id`) where a reader is shown a
   name: an aspect, measure, judged dimension, stratum, task or criterion id, an arm id the page prints by another
   name, a caveat id, a decision slot's word, an example's trial or case id, or one of the document's own claim ids
   (the page places claims and never shows their ids; an example or claim id is matched in its own case, so a case
   ``q3`` is not the word "Q3"). The rendered page's text above the
   appendix is checked the same way where the caller passes it (``above_appendix_text``, the renderer's
   ``above_appendix_text(page)``), which catches what the renderer itself prints: a caveat id, a measure key in a
   header. That text leaves out the example cards the renderer draws, which print their trial and case ids and are
   exempt (plan §2.2, input 11).
6. **Method is present**: at least one item. The full-results appendix is the renderer's, always built.

Invariant 7 (``check.py`` at 1200 and 390 px, light and dark) needs a browser, so it is not checked at storage; the
integration suite runs it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.block_spec import block_look, chart_faults, complete_chart, small_multiple_faults
from threetears.evals.analysis.report.figure_keys import keys_in, reader_names, render
from threetears.evals.analysis.report.plan_page.look import PageLook
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_decision import Caveat, ImpliedCaveat, caveat_materiality, every_caveat
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.analysis.report_prose_checks import (
    internal_ids,
    literal_faults,
    reads_as_id,
    typed_ids,
    unit_faults,
)
from threetears.evals.kernel.report_doc import (
    DecisionSlotWord,
    ReportDoc,
    is_analyst_caveat,
    report_doc_prose,
    report_doc_validity,
)
from threetears.evals.kernel.report_plan import PlannedChart

#: The names with a caller outside this module. :func:`check_report_doc` and :class:`DocVerdict` are this build's
#: invariants surface; each joins this list with its first production caller (the store, the ``report_check`` action,
#: the writing loop), never for a test alone.
__all__: list[str] = []

#: Which check a verdict comes from: ``0`` validity, then the invariants 1 to 6 (7 is not checked at storage).
InvariantNumber = Literal[0, 1, 2, 3, 4, 5, 6]


class DocVerdict(BaseModel):
    """One thing code proves wrong about a report document on its snapshot: which check, where, what, and the mend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invariant: InvariantNumber = Field(description="The check: 0 validity, 1 to 6 the invariants.")
    path: str = Field(description="The field path it sits at, as `claims[2].evidence[0].query.arms`.")
    fault: str = Field(description="What is wrong there.")
    remedy: str = Field(description="How to mend it, and which tool shows what to mend it with.")


# --- the remedies --------------------------------------------------------------------------------------------------

_VALIDITY_REMEDY = "name only what the snapshot holds: campaign_overview lists its arms, caveats_list its caveat ids"
_KEY_REMEDY = (
    "figure_keys_check reads each key as the page prints it, or says why it reads nothing; cite one that reads"
)
_DIGIT_REMEDY = (
    "write every figure as a figure key (figure_keys_check reads it); a number the snapshot does not hold is written in "
    "words or left out. Names the snapshot states, percentile labels and a cost's unit (per 1,000, per 1k) may be "
    "typed; a claim id may be typed only in omitted and Method, and never one made of digits alone"
)
_UNIT_REMEDY = "a key prints its own unit and cost basis: drop the repeated word, or key the figure as the fault says"
_BLOCK_REMEDY = (
    "block_render draws the block as the page will, naming every fault; block_catalog lists what each kind draws. Mend "
    "the evidence until block_render draws it"
)
_CAVEAT_REMEDY = {
    "report": "it is report-wide: list it in confidence.caveats, open with its material_if or checked as code proves; "
    "it may be named beside a claim or in Method too",
    "scoped": "it touches some comparisons: list it in confidence.caveats, or name it in the caveats of the claim it "
    "qualifies; it may be named in Method too",
}
_REPEAT_REMEDY = "the confidence carries each caveat once: drop the repeat"
_CHECKED_REMEDY = (
    "state it open, with the material_if that would make it matter; code checks only a late margin, floor or bar, and "
    "caveats_list says which it proved immaterial"
)
_FORM_REMEDY = {
    "high": "a plain `high` carries no open caveat: make the form high_if_immaterial, or low",
    "high_if_immaterial": "high_if_immaterial rests on open caveats, each with its material_if: name the one the call "
    "is conditional on, or make the form high",
}
_ID_REMEDY = "a reader is shown names, never ids: write the name the page prints, or a figure key"
_EXAMPLE_REMEDY = "a {kind} has no name a reader is shown: describe it in words; its example card shows the id"
_CLAIM_REMEDY = "the page places claims and shows no id above the appendix: say what the claim found, in words"
_METHOD_REMEDY = "Method holds at least one term: say how the evidence was gathered and read"

#: A cost's unit, which prose may type with its digits: the count (``1,000``, ``1k``, ``1,000,000``) after ``per`` or
#: a slash (``per 1,000``, ``per 1k``, ``$/1k``).
_COST_UNIT = re.compile(r"(?:(?<!\w)per\s+|/\s?)(?:1,000,000|1,000|1000|1k)(?!\w)", re.IGNORECASE)

#: The path a verdict on the rendered page's text sits at.
_PAGE = "page"
#: The fields below the page's appendix, where ids may appear.
_APPENDIX = ("omitted[", "method[")

#: Where a caveat of each reach that touches the call's confidence lands: the kinds of place a document names a
#: caveat in. A method-level one may land in any of them (a Method item too), or in none.
_PLACES: dict[str, frozenset[str]] = {
    "report": frozenset({"confidence"}),
    "scoped": frozenset({"confidence", "claim"}),
}
#: How a fault names each place a caveat lands.
_LANDING_WORDS = {"confidence": "the confidence", "claim": "a claim's caveats"}
#: Where a caveat left out is reported: the one place every reach may land.
_HOME = "confidence.caveats"


def check_report_doc(doc: ReportDoc, report_input: ReportInput, *, above_appendix_text: str = "") -> list[DocVerdict]:
    """Every verdict code proves against a report document on its snapshot (this module's account): validity's, then
    each invariant's, in order.

    :param doc: The document.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from.
    :ptype report_input: ReportInput
    :param above_appendix_text: The rendered page's text above the appendix, example cards left out (the claims
        renderer's ``above_appendix_text(page)``), for invariant 5; empty to check the document's prose alone.
    :ptype above_appendix_text: str
    :return: The verdicts; empty for a document that may be stored.
    :rtype: list[DocVerdict]
    """
    caveats = every_caveat(report_input)
    verdicts = [
        DocVerdict(invariant=0, path=path, fault=fault, remedy=_VALIDITY_REMEDY)
        for path, fault in report_doc_validity(
            doc, arms=[arm.id for arm in report_input.arms], caveats=[caveat.id for caveat in caveats]
        )
    ]
    prose = list(report_doc_prose(doc))
    verdicts += _number_verdicts(doc, prose, report_input)
    verdicts += _block_verdicts(doc, report_input)
    verdicts += _coverage_verdicts(doc, report_input, caveats)
    verdicts += _form_verdicts(doc)
    verdicts += _id_verdicts(doc, prose, report_input, caveats, above_appendix_text)
    verdicts += _method_verdicts(doc)
    return verdicts


# --- 1. numbers come through figure keys ---------------------------------------------------------------------------


def _claim_ids(doc: ReportDoc) -> list[str]:
    """The document's claim ids that read as names, longest first: never one made only of digits, which reads as a
    number (``literal_numbers``' own rule for an id of digits alone)."""
    return sorted((claim.id for claim in doc.claims if not claim.id.isdecimal()), key=len, reverse=True)


def _allowed_phrases(doc: ReportDoc) -> re.Pattern[str]:
    """What prose may type with its digits beside the snapshot's names: a cost's unit, and the document's own claim
    ids (``c1``), matched in their own case. The appendix names a claim by its id; above it, invariant 5 refuses one,
    so each draws that one verdict."""
    claims = [rf"(?<!\w)(?-i:{re.escape(claim)})(?!\w)" for claim in _claim_ids(doc)]
    return re.compile("|".join([_COST_UNIT.pattern, *claims]), re.IGNORECASE)


def _number_verdicts(doc: ReportDoc, prose: list[tuple[str, str]], report_input: ReportInput) -> Iterator[DocVerdict]:
    names = {arm.id: arm.name for arm in fact_arms(report_input)}
    for path, text in prose:
        try:
            keys = keys_in(text)
        except UnresolvableReference as refused:
            yield DocVerdict(invariant=1, path=path, fault=str(refused), remedy=_KEY_REMEDY)
            continue
        for key in keys:
            assert key.span is not None
            start, end = key.span
            try:
                render(text[start:end], report_input, names)
            except UnresolvableReference as refused:
                yield DocVerdict(invariant=1, path=path, fault=str(refused), remedy=_KEY_REMEDY)
    for path, fault in literal_faults(prose, report_input, allowed=_allowed_phrases(doc)):
        yield DocVerdict(invariant=1, path=path, fault=fault, remedy=_DIGIT_REMEDY)
    for path, fault in unit_faults(prose, report_input):
        yield DocVerdict(invariant=1, path=path, fault=fault, remedy=_UNIT_REMEDY)


# --- 2. every evidence block resolves and renders ------------------------------------------------------------------


def _block_verdicts(doc: ReportDoc, report_input: ReportInput) -> Iterator[DocVerdict]:
    look: PageLook | None = None
    for index, claim in enumerate(doc.claims):
        for number, evidence in enumerate(claim.evidence):
            path = f"claims[{index}].evidence[{number}]"
            planned = PlannedChart.model_validate(
                {
                    "kind": evidence.block,
                    "title": "",
                    "query": evidence.query,
                    "reference": [{"kind": evidence.reference}] if evidence.reference is not None else [],
                    "emphasis": evidence.emphasis,
                }
            )
            spec, faults = complete_chart(planned, report_input, path)
            faults += chart_faults(path, spec, report_input)
            faults += small_multiple_faults(path, spec)
            if not faults:
                look = look or block_look(report_input, fact_arms(report_input))
                try:
                    resolve_chart(spec, report_input, look, title="")
                except UnresolvableReference as refused:
                    faults.append((path, f"is refused by the page, which draws it from this snapshot: {refused}"))
            for at, fault in faults:
                yield DocVerdict(invariant=2, path=at, fault=fault, remedy=_BLOCK_REMEDY)


# --- 3. caveat coverage --------------------------------------------------------------------------------------------


def _landings(doc: ReportDoc) -> Iterator[tuple[str, str, str]]:
    """Every place the document names a caveat: the id, the field path, and the kind of place."""
    for index, caveat in enumerate(doc.confidence.caveats):
        yield caveat.id, f"confidence.caveats[{index}].id", "confidence"
    for index, claim in enumerate(doc.claims):
        for number, named in enumerate(claim.caveats):
            yield named, f"claims[{index}].caveats[{number}]", "claim"
    for index, item in enumerate(doc.method):
        for number, named in enumerate(item.caveats):
            yield named, f"method[{index}].caveats[{number}]", "method"


def _coverage_verdicts(
    doc: ReportDoc, report_input: ReportInput, caveats: Sequence[Caveat | ImpliedCaveat]
) -> Iterator[DocVerdict]:
    reach = {caveat.id: caveat.reach for caveat in caveats}
    landed: dict[str, list[tuple[str, str]]] = {}
    for named, path, place in _landings(doc):
        landed.setdefault(named, []).append((path, place))
    for caveat in caveats:
        if caveat.reach == "method":
            continue  # lands anywhere a caveat may, or nowhere: it does not touch the call's confidence
        places = landed.get(caveat.id, [])
        if any(place in _PLACES[caveat.reach] for _path, place in places):
            continue  # it lands where its reach allows; naming it again elsewhere is the analyst's emphasis
        if not places:
            yield DocVerdict(
                invariant=3,
                path=_HOME,
                fault=f"leaves out the caveat {caveat.id!r}, which code assigns on this snapshot ({caveat.reach} reach)",
                remedy=_CAVEAT_REMEDY[caveat.reach],
            )
            continue
        path, place = places[0]
        yield DocVerdict(
            invariant=3,
            path=path,
            fault=(
                f"names {caveat.id!r} only in a {place}, and a caveat of {caveat.reach} reach lands in "
                f"{' or '.join(sorted(_LANDING_WORDS[where] for where in _PLACES[caveat.reach]))}"
            ),
            remedy=_CAVEAT_REMEDY[caveat.reach],
        )
    first: dict[str, int] = {}
    for index, entry in enumerate(doc.confidence.caveats):
        if entry.id in first:
            yield DocVerdict(
                invariant=3,
                path=f"confidence.caveats[{index}].id",
                fault=f"lists {entry.id!r} again (also at confidence.caveats[{first[entry.id]}])",
                remedy=_REPEAT_REMEDY,
            )
            continue
        first[entry.id] = index
    materiality = caveat_materiality(report_input)
    for index, entry in enumerate(doc.confidence.caveats):
        if entry.checked is None or (not is_analyst_caveat(entry.id) and entry.id not in reach):
            continue  # open, or an id validity refuses
        proved = materiality.get(entry.id)
        if proved is None or not proved.immaterial:
            yield DocVerdict(
                invariant=3,
                path=f"confidence.caveats[{index}].checked",
                fault=f"checks {entry.id!r} immaterial, which code does not prove on this snapshot",
                remedy=_CHECKED_REMEDY,
            )


# --- 4. the confidence form ----------------------------------------------------------------------------------------


def _form_verdicts(doc: ReportDoc) -> Iterator[DocVerdict]:
    form = doc.confidence.form
    open_ids = [caveat.id for caveat in doc.confidence.caveats if caveat.material_if is not None]
    if form == "high" and open_ids:
        yield DocVerdict(
            invariant=4,
            path="confidence.form",
            fault=f"is high while the confidence carries open caveats: {open_ids}",
            remedy=_FORM_REMEDY[form],
        )
    if form == "high_if_immaterial" and not open_ids:
        yield DocVerdict(
            invariant=4,
            path="confidence.form",
            fault="is high_if_immaterial while the confidence carries no open caveat",
            remedy=_FORM_REMEDY[form],
        )


# --- 5. no internal id above the appendix --------------------------------------------------------------------------


def _id_verdicts(
    doc: ReportDoc,
    prose: list[tuple[str, str]],
    report_input: ReportInput,
    caveats: Sequence[Caveat | ImpliedCaveat],
    above_appendix_text: str,
) -> Iterator[DocVerdict]:
    said: dict[str, tuple[str, str]] = {
        identifier: (f"the {kind} id {identifier!r}", _ID_REMEDY)
        for identifier, (kind, _name) in internal_ids(report_input).items()
    }
    arms = {arm.id: arm.name for arm in fact_arms(report_input)}
    printed = {
        arm.id: {name.casefold() for name in (arms[arm.id], arm.name, arm.short_name, arm.served_model) if name}
        for arm in report_input.arms
    }
    for arm, names in printed.items():
        if reads_as_id(arm) and arm.casefold() not in names:
            said.setdefault(arm, (f"the arm id {arm!r}, which the page prints as {arms[arm]!r}", _ID_REMEDY))
    named = [caveat.id for caveat in caveats] + [caveat.id for caveat in doc.confidence.caveats]
    for caveat in named:
        if reads_as_id(caveat):
            said.setdefault(caveat, (f"the caveat id {caveat!r}", _ID_REMEDY))
    for slot in get_args(DecisionSlotWord):
        if reads_as_id(slot):
            said.setdefault(slot, (f"the slot word {slot!r}, which titles its entry", _ID_REMEDY))
    shown = {name for names in reader_names(report_input).values() for name in names.values()}
    shown |= {
        name for arm in report_input.arms for name in (arm.id, arm.name, arm.short_name, arm.served_model) if name
    }
    shown |= set(arms.values())
    exact: dict[str, tuple[str, str]] = {}
    for example in report_input.examples:
        for kind, identifier in (("trial", example.trial), ("case", example.case)):
            if identifier and identifier not in said and identifier not in shown and _reads_as_a_tag(identifier):
                exact.setdefault(identifier, (f"the {kind} id {identifier!r}", _EXAMPLE_REMEDY.format(kind=kind)))
    for claim in _claim_ids(doc):
        if claim not in said and claim not in shown and _reads_as_a_tag(claim):
            exact.setdefault(claim, (f"the claim id {claim!r}, which the page never shows", _CLAIM_REMEDY))
    ids = said | exact
    hidden = shown - set(ids)
    above = [(path, text) for path, text in prose if not path.startswith(_APPENDIX)]
    yield from _typed(above, ids, hidden, set(exact), "above the appendix")
    if above_appendix_text:
        yield from _typed([(_PAGE, above_appendix_text)], ids, hidden, set(exact), "on the page above the appendix")


def _reads_as_a_tag(identifier: str) -> bool:
    """An id no reader would take for a word: one that reads as an id, or carries a digit (``T15``, ``c2``)."""
    return reads_as_id(identifier) or any(character.isdigit() for character in identifier)


def _typed(
    texts: Sequence[tuple[str, str]], said: dict[str, tuple[str, str]], shown: set[str], exact: set[str], where: str
) -> Iterator[DocVerdict]:
    for path, text in texts:
        for identifier in typed_ids(text, said, shown, exact=exact):
            what, remedy = said[identifier]
            yield DocVerdict(invariant=5, path=path, fault=f"types {what} {where}", remedy=remedy)


# --- 6. Method is present ------------------------------------------------------------------------------------------


def _method_verdicts(doc: ReportDoc) -> Iterator[DocVerdict]:
    if not doc.method:
        yield DocVerdict(invariant=6, path="method", fault="holds no item", remedy=_METHOD_REMEDY)
