"""The planner's two replies assembled into one plan: the structure's choices, then the prose its keys address.

The planner writes a page in two calls (:mod:`threetears.evals.kernel.report_plan`): a
:class:`~threetears.evals.kernel.report_plan.PlanStructure` holding every choice and no prose, then a
:class:`~threetears.evals.kernel.report_plan.PlanProse` holding the prose as flat ``{key, text}`` pairs. This module is
the seam between them:

- :func:`check_structure` refuses a structure before any prose is asked for: a mode outside the modes code allows, and
  every fault the plan checks find in its choices
  (:func:`~threetears.evals.analysis.report_plan_checks.validate_plan`, run on the structure completed with
  a placeholder in every prose slot, which carries no digit and no key but, in a decide plan's Watch, the weakest
  margin's, so only the choices can be faulted).
- :func:`prose_slots` lists the prose slots a structure creates, each by its key, what it holds and whether it may be
  empty. The prose call is asked for exactly these keys.
- :func:`read_prose` refuses a prose reply holding a key twice, a key no slot has, or a slot left out or left empty
  where it must be written.
- :func:`planned_report` assembles the planner's part of the plan from the structure and the prose.

**The keys** (:func:`prose_slots`), in the order they are asked for: ``headline``; ``lede``; ``flag.title`` and
``flag.text`` when the flag discloses anything (the flag is present exactly then: ``no_rule`` where it applies, and each
open report-wide caveat the confidence does not list,
:func:`~threetears.evals.analysis.report_plan_facts.flag_disclosures`); ``confidence.caveats.<id>`` for each caveat
the structure lands at ``confidence``, its clause; ``decision.<slot>`` for each of the mode's four slots; ``summary.<n>.title`` for each summary chart; for each section,
``sections.<id>.eyebrow``, ``.headline``, ``.takeaway``, ``.title`` (its chart's) and ``.caption``; ``answers.<n>`` for
each warning code raised on the recommended arm, ``<n>`` its position in code's list; ``durable_claims.<n>`` for each
claim; ``method.<n>.term`` and ``method.<n>.text`` for each Method pair; and, with an explore part, ``explore.title``,
``explore.recomputes`` and ``explore.controls.<id>.title``. Chart titles, takeaways and captions may be empty (the
section headline may say it); every other slot is written. A key is matched whole, never split, so an id holding dots
is safe.

**Caveat landings.** The structure lands each open caveat once (``caveat_landings``), by an address: ``confidence``,
``decision.<slot key>`` (one of the mode's slots), ``sections.<section id>`` or ``method.<n>`` (a Method pair, from 0).
:func:`planned_report` turns them into the confidence's clauses and each slot's, section's and Method pair's
``caveats``, and refuses an address naming nothing the structure holds. Where each caveat may land, and that each lands
exactly once, is the plan checks' (:func:`~threetears.evals.analysis.report_plan_checks.validate_plan`).

**Answers are code's set.** The structure names no answers: one is owed per warning code raised on the arm it
recommends, so the slots are exactly those warnings, and an answer can neither be missed nor name a warning code never
raised.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.errors import PlanRefusal
from threetears.evals.analysis.report_plan_checks import MODE_SLOTS
from threetears.evals.analysis.report_plan_facts import PlanFacts, complete_plan, flag_disclosures
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import PlannedReport, PlanProse, PlanStructure, WarningRef

__all__ = ["ProseSlot", "check_structure", "planned_report", "prose_slots", "read_prose"]

#: What every prose slot holds while a structure is checked: no figure key and no digit, so no prose check can fault it.
_PROSE_PLACEHOLDER = "To be written."

#: What a decision slot holds where code checks more of it than its key, by mode and slot: a decide plan's Watch cites
#: the weakest margin by a ``weakest.*`` key (:func:`~threetears.evals.analysis.report_plan_checks.validate_plan`).
_SLOT_HOLDS = {
    ("decide", "watch"): (
        "the decision block's watch slot: open with the weakest margin, citing it by its `weakest.` keys "
        "(weakest_margin under WHAT CODE HAS ALREADY DECIDED), then any scoped confound or unmeasured risk"
    ),
}

#: A fault: the field path it sits at, and why.
_Fault = tuple[str, str]


class ProseSlot(BaseModel):
    """One prose slot a structure creates: the key the prose call writes it under, what it holds, and whether it may be
    left empty."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(description="The slot's key.")
    holds: str = Field(description="What the slot holds, as the prose call is told.")
    optional: bool = Field(description="Whether the slot may be left empty.")


def _recommended_warnings(structure: PlanStructure, facts: PlanFacts) -> tuple[WarningRef, ...]:
    """The warnings code raised on the arm the structure recommends; none when it recommends nothing."""
    row = next((row for row in structure.options.rows if row.role == "recommended"), None)
    return facts.warnings.get(row.option, ()) if row is not None else ()


def _confidence_ids(structure: PlanStructure) -> list[str]:
    """The caveats the structure lands in the decision's confidence, in its order."""
    return [landing.caveat for landing in structure.caveat_landings if landing.at == "confidence"]


def _flagged(structure: PlanStructure, facts: PlanFacts) -> tuple[str, ...]:
    return flag_disclosures(facts, structure.mode, _confidence_ids(structure))


def prose_slots(structure: PlanStructure, facts: PlanFacts) -> tuple[ProseSlot, ...]:
    """The prose slots a structure creates, in the order the prose call is asked for them (this module's keys).

    :param structure: The checked structure.
    :ptype structure: PlanStructure
    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :return: Each slot, once.
    :rtype: tuple[ProseSlot, ...]
    """
    return tuple(_slots(structure, facts))


def _slots(structure: PlanStructure, facts: PlanFacts) -> Iterator[ProseSlot]:
    def slot(key: str, holds: str, *, optional: bool = False) -> ProseSlot:
        return ProseSlot(key=key, holds=holds, optional=optional)

    yield slot("headline", "the decision as an instruction with its reason; in tradeoff mode the trade")
    yield slot("lede", "at most two sentences: the strongest supporting fact, then what would change the answer")
    if flagged := _flagged(structure, facts):
        yield slot("flag.title", f"the flag in one line, disclosing {', '.join(flagged)}")
        yield slot("flag.text", "which findings the disclosures weaken and which survive")
    for caveat in dict.fromkeys(_confidence_ids(structure)):
        yield slot(
            f"confidence.caveats.{caveat}",
            f"the clause for caveat {caveat!r}: what it is, and what would make it material",
        )
    for name in MODE_SLOTS[structure.mode]:
        yield slot(f"decision.{name}", _SLOT_HOLDS.get((structure.mode, name), f"the decision block's {name} slot"))
    for index in range(len(structure.summary)):
        yield slot(f"summary.{index}.title", f"summary chart {index}'s title", optional=True)
    for section in structure.sections:
        where = f"sections.{section.id}"
        yield slot(f"{where}.eyebrow", "the question the section answers, in a few words")
        yield slot(f"{where}.headline", "the finding, with subject and direction")
        yield slot(f"{where}.takeaway", "one line on how to read the chart", optional=True)
        yield slot(f"{where}.title", "the chart's title; empty when the headline says it", optional=True)
        yield slot(f"{where}.caption", "one line under the chart, only where one is owed", optional=True)
    for position, warning in enumerate(_recommended_warnings(structure, facts)):
        yield slot(
            f"answers.{position}",
            f"why the recommendation stands over {warning.kind} {warning.name!r} on aspect {warning.aspect!r}",
        )
    for index in range(len(structure.durable_claims)):
        yield slot(f"durable_claims.{index}", f"durable claim {index}, naming its figures by key")
    for index in range(structure.method_items):
        yield slot(f"method.{index}.term", f"Method pair {index}'s term")
        yield slot(f"method.{index}.text", f"Method pair {index}'s definition")
    for explore in structure.explore:
        yield slot("explore.title", "the explore part's title")
        yield slot("explore.recomputes", "what the controls recompute and where the answer flips")
        for control in explore.controls:
            yield slot(f"explore.controls.{control.id}.title", f"control {control.id!r}'s label")


def read_prose(reply: PlanProse, slots: Sequence[ProseSlot]) -> dict[str, str]:
    """The prose reply as text by key, or the refusal of every key it got wrong.

    :param reply: The validated prose reply.
    :ptype reply: PlanProse
    :param slots: The slots it was asked for (:func:`prose_slots`).
    :ptype slots: Sequence[ProseSlot]
    :return: Each slot's text, by key.
    :rtype: dict[str, str]
    :raises PlanRefusal: A key appears twice, names no slot, or a slot is missing or empty where it must be written.
    """
    wanted = {slot.key: slot for slot in slots}
    counts = Counter(item.key for item in reply.prose)
    faults: list[_Fault] = []
    for index, item in enumerate(reply.prose):
        if item.key not in wanted:
            faults.append((f"prose[{index}].key", f"is {item.key!r}, which names no prose slot this plan has"))
    faults += [
        ("prose", f"holds {key!r} {count} times; write each key once") for key, count in counts.items() if count > 1
    ]
    texts = {item.key: item.text for item in reply.prose if item.key in wanted}
    for key, slot in wanted.items():
        if key not in texts:
            faults.append(("prose", f"leaves out {key!r} ({slot.holds})"))
        elif not slot.optional and not texts[key].strip():
            faults.append(("prose", f"leaves {key!r} empty, and it must be written ({slot.holds})"))
    if faults:
        raise PlanRefusal(faults)
    return texts


def planned_report(structure: PlanStructure, prose: Mapping[str, str], facts: PlanFacts) -> PlannedReport:
    """The planner's part of the plan, from its structure and its prose by key.

    :param structure: The checked structure.
    :ptype structure: PlanStructure
    :param prose: Every slot's text, by key (:func:`read_prose`).
    :ptype prose: Mapping[str, str]
    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :return: The assembled reply.
    :rtype: PlannedReport
    :raises PlanRefusal: The assembled text breaks a field's bounds (a lede over its length, an empty required text).
    """

    def chart(planned: BaseModel, title: str) -> dict[str, object]:
        return {**planned.model_dump(), "title": title}

    warnings = _recommended_warnings(structure, facts)
    landed = _landed(structure)
    try:
        return PlannedReport.model_validate(
            {
                "mode": structure.mode,
                "headline_kind": structure.headline_kind,
                "headline": prose["headline"],
                "lede": prose["lede"],
                "flag": (
                    [{"title": prose["flag.title"], "text": prose["flag.text"]}] if _flagged(structure, facts) else []
                ),
                "decision": {
                    "slots": [
                        {"key": name, "text": prose[f"decision.{name}"], "caveats": landed.get(f"decision.{name}", [])}
                        for name in MODE_SLOTS[structure.mode]
                    ],
                    "confidence": {
                        "form": structure.confidence,
                        "caveats": [
                            {"id": caveat, "material_if": prose[f"confidence.caveats.{caveat}"]}
                            for caveat in _confidence_ids(structure)
                        ],
                    },
                },
                "options": structure.options,
                "summary": [
                    chart(summary, prose[f"summary.{index}.title"]) for index, summary in enumerate(structure.summary)
                ],
                "sections": [
                    {
                        "id": section.id,
                        "eyebrow": prose[f"sections.{section.id}.eyebrow"],
                        "headline": prose[f"sections.{section.id}.headline"],
                        "takeaway": prose[f"sections.{section.id}.takeaway"],
                        "chart": chart(section.chart, prose[f"sections.{section.id}.title"]),
                        "caption": prose[f"sections.{section.id}.caption"],
                        "caveats": landed.get(f"sections.{section.id}", []),
                    }
                    for section in structure.sections
                ],
                "explore": [
                    {
                        "title": prose["explore.title"],
                        "controls": [
                            {**control.model_dump(), "title": prose[f"explore.controls.{control.id}.title"]}
                            for control in explore.controls
                        ],
                        "recomputes": prose["explore.recomputes"],
                    }
                    for explore in structure.explore
                ],
                "method": [
                    {
                        "term": prose[f"method.{index}.term"],
                        "text": prose[f"method.{index}.text"],
                        "caveats": landed.get(f"method.{index}", []),
                    }
                    for index in range(structure.method_items)
                ],
                "durable_claims": [
                    {"text": prose[f"durable_claims.{index}"], "confidence": claim.confidence, "keys": claim.keys}
                    for index, claim in enumerate(structure.durable_claims)
                ],
                "answers": [
                    {"warning": position, "text": prose[f"answers.{position}"]} for position in range(len(warnings))
                ],
            }
        )
    except ValidationError as refused:
        raise PlanRefusal(
            [(".".join(str(step) for step in error["loc"]), error["msg"]) for error in refused.errors()]
        ) from refused


def _addresses(structure: PlanStructure) -> set[str]:
    """Every address a caveat may land at in this structure, but ``confidence``."""
    return (
        {f"decision.{name}" for name in MODE_SLOTS[structure.mode]}
        | {f"sections.{section.id}" for section in structure.sections}
        | {f"method.{index}" for index in range(structure.method_items)}
    )


def _landed(structure: PlanStructure) -> dict[str, list[str]]:
    """The caveats landed at each slot, section and Method pair, by address; refused where an address names nothing.

    :raises PlanRefusal: A landing's address is neither ``confidence`` nor a slot, section or Method pair the structure
        holds.
    """
    addresses = _addresses(structure)
    landed: dict[str, list[str]] = {}
    faults: list[_Fault] = []
    for index, landing in enumerate(structure.caveat_landings):
        if landing.at == "confidence":
            continue
        if landing.at not in addresses:
            faults.append(
                (
                    f"caveat_landings[{index}].at",
                    f"is {landing.at!r}, which names nothing this structure holds; it is `confidence` or one of "
                    f"{sorted(addresses)}",
                )
            )
            continue
        landed.setdefault(landing.at, []).append(landing.caveat)
    if faults:
        raise PlanRefusal(faults)
    return landed


def _placeholder(key: str, facts: PlanFacts) -> str:
    """A slot's text while its structure is checked: the placeholder, and in Watch the weakest margin's key the prose
    must cite, so the check faults only the choices."""
    if key == "decision.watch" and facts.weakest is not None:
        return f"{_PROSE_PLACEHOLDER} {{{facts.weakest.keys[0]}}}"
    return _PROSE_PLACEHOLDER


def check_structure(
    structure: PlanStructure, report_input: ReportInput, facts: PlanFacts, allowed_modes: Sequence[str]
) -> None:
    """Refuse a structure whose mode code does not allow, or whose choices the plan checks fault (this module's
    account).

    :param structure: The planner's validated structure reply.
    :ptype structure: PlanStructure
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param facts: Code's facts about the input.
    :ptype facts: PlanFacts
    :param allowed_modes: The modes code allows for this input.
    :ptype allowed_modes: Sequence[str]
    :raises PlanRefusal: Every fault, together.
    """
    faults: list[_Fault] = []
    if structure.mode not in allowed_modes:
        faults.append(("mode", f"is {structure.mode!r}, and code allows only {list(allowed_modes)} for this input"))
    placeholder = {slot.key: _placeholder(slot.key, facts) for slot in prose_slots(structure, facts)}
    try:
        complete_plan(planned_report(structure, placeholder, facts), report_input, facts)
    except PlanRefusal as refused:
        faults.extend(refused.faults)
    if faults:
        raise PlanRefusal(faults)
