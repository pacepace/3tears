"""A claim criterion's readings, computed on read from its stored tally, and the tally, derived from its grade.

A cell's grade (:class:`~threetears.evals.schema.ClaimGrade`) lists every claim a criterion's judge found and how each
was graded; its tally (:class:`~threetears.evals.schema.ClaimTally`) keeps the counts; every reading is a pure function
of the tally, so none is stored and a diagnostic costs nothing. Three readings rank, each on its own and never in a
composite:

- **R1** ``<c>_false_per_unit``: asserted world claims contradicted by evidence, per unit. Lower is better.
- **R1b** ``<c>_false_unit_share``: the share of units holding at least one. Lower is better.
- **R2** ``<c>_grounded_share``: of asserted specific world claims, the share a document the candidate itself was given
  supports, for a criterion declaring ``grounded_in``. Higher is better.

Every other reading is a diagnostic: it explains a movement and has no better end. What a claim's tags mean for each
count (a session claim counts in none of the world readings; a hedged or attributed claim is graded for truth and kept
out of the asserted counts; a statement the text backs with a source is asserted) is settled once, in
:func:`claim_tally`.

Engine-internal: the judge phase tallies, the ranking path reads, and nothing here is a host's to import.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final

from threetears.evals.kernel.metrics import CLAIMS_FAMILY, MetricDescriptor, list_metrics
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.schema import (
    ClaimEvidence,
    ClaimGrade,
    ClaimJudgeCounts,
    ClaimLoss,
    ClaimTally,
    EvidenceProvenance,
    EvalTemplate,
)
from threetears.evals.schema.models import ClaimCall, GradedClaim, KnownClaimLoss, request_settings_order


@dataclass(frozen=True)
class _Reading:
    """One reading a criterion's tally gives, as its descriptor and its reader state it.

    :ivar words: What a reader calls it, before the criterion it reads (``False claims per {unit}``).
    :ivar lower_is_better: ``True`` or ``False`` for a ranked reading; ``None`` for a diagnostic.
    :ivar share: Whether it is a share on 0 to 1, rather than a count or an unbounded rate.
    :ivar formula: How it is computed from the tally.
    :ivar population: Which cells carry it.
    :ivar evidence: What decided the claims it counts.
    """

    words: str
    lower_is_better: bool | None
    share: bool
    formula: str
    population: str
    evidence: str


_TRUTH = (
    "a claim is false when the case's own evidence (its key facts and case documents, the same for every arm) "
    "contradicts it, or, for a claim no case evidence covers that the judge doubts, when a quote from a web page does; "
    "the candidate's own documents never decide truth"
)
_TALLIED = "every cell the criterion graded, not lost"
_WITH_UNITS = "cells with at least one unit, not lost; a claim in a segment outside every unit counts in no unit"
_WITH_ASSERTED = "cells with at least one asserted world claim, not lost"
_ASSERTED = (
    "an asserted claim is a world claim the text neither hedges nor attributes to another party (a statement it backs "
    "with a source is asserted); a restatement counts as heard, with its original's verdict"
)

#: Every reading, by the suffix its name takes after the criterion's (``<criterion>_<suffix>``), ranked first.
_READINGS: Final[Mapping[str, _Reading]] = MappingProxyType(
    {
        "false_per_unit": _Reading(
            "False claims per {unit}",
            lower_is_better=True,
            share=False,
            formula="asserted world claims contradicted / units",
            population=_WITH_UNITS,
            evidence=f"{_TRUTH}; {_ASSERTED}. Unbounded and often zero, so it declares no range and is compared on "
            "the t-path over per-case means",
        ),
        "false_unit_share": _Reading(
            "Share of {unit}s with a false claim",
            lower_is_better=True,
            share=True,
            formula="units holding an asserted world claim contradicted / units",
            population=_WITH_UNITS,
            evidence=f"{_TRUTH}; {_ASSERTED}. It counts a unit once however many false claims it holds, so it and the "
            "false claims per unit can disagree",
        ),
        "grounded_share": _Reading(
            "Grounded share of specific claims",
            lower_is_better=False,
            share=True,
            formula="asserted specific world claims a grounded_in document supports / asserted specific world claims",
            population="cells with at least one asserted specific world claim whose documents were declared, not lost; "
            "a cell whose candidate was given no document reads 0",
            evidence="a claim is grounded when a document the candidate itself was given, of a provenance the "
            "criterion's grounded_in names, supports it by a quote; case evidence counts only where grounded_in names "
            "case, and a claim only the web or case evidence the candidate was not given confirms is ungrounded. "
            f"A specific claim states a name, number, date or rule a source could settle; {_ASSERTED}",
        ),
        "false": _Reading(
            "False claims",
            None,
            share=False,
            formula="asserted world claims contradicted",
            population=_TALLIED,
            evidence=f"{_TRUTH}; {_ASSERTED}",
        ),
        "false_hedged": _Reading(
            "False hedged claims",
            None,
            share=False,
            formula="hedged world claims contradicted",
            population=_TALLIED,
            evidence=f"{_TRUTH}; a hedged claim is one the text marks as its own uncertain recollection",
        ),
        "false_attributed": _Reading(
            "False attributed claims",
            None,
            share=False,
            formula="attributed world claims contradicted",
            population=_TALLIED,
            evidence=f"{_TRUTH}; an attributed claim is one the text reports without vouching for it (a rumour, "
            "another party's statement)",
        ),
        "false_share": _Reading(
            "False share of asserted claims",
            None,
            share=True,
            formula="asserted world claims contradicted / asserted world claims",
            population=_WITH_ASSERTED,
            evidence=f"{_TRUTH}; {_ASSERTED}",
        ),
        "claims": _Reading(
            "World claims",
            None,
            share=False,
            formula="world claims",
            population=_TALLIED,
            evidence="the claims about the world the judge listed in the output, of every tag; a claim about the "
            "output or session itself is no world claim",
        ),
        "units": _Reading(
            "{Unit}s",
            None,
            share=False,
            formula="units",
            population=_TALLIED,
            evidence="the distinct units the host cut the output into, each counted once however many segments it spans",
        ),
        "hedged_share": _Reading(
            "Hedged share of claims",
            None,
            share=True,
            formula="hedged world claims / world claims",
            population="cells with at least one world claim, not lost",
            evidence="a hedged claim is one the text marks as its own uncertain recollection, as the judge read it",
        ),
        "unverifiable": _Reading(
            "Unverifiable claims",
            None,
            share=False,
            formula="world claims neither supported nor contradicted",
            population=_TALLIED,
            evidence="world claims of every tag no case evidence covered and no web check settled",
        ),
        "true_ungrounded": _Reading(
            "Claims only the web confirmed",
            None,
            share=False,
            formula="world claims a web quote confirmed and no case evidence covered",
            population=_TALLIED,
            evidence="world claims of every tag the judge doubted, that no case evidence covered, and that a quote "
            "from a web page showed true",
        ),
        "web_settled_share": _Reading(
            "Web-settled share of asserted claims",
            None,
            share=True,
            formula="asserted world claims the web decided / asserted world claims",
            population=_WITH_ASSERTED,
            evidence="a web check runs only on a claim no case evidence covers that the judge doubts, so this moves "
            f"with the judge's suspicion and the case's coverage as much as with the arm; {_ASSERTED}",
        ),
        "evidence_conflicts": _Reading(
            "Claims case evidence both supports and contradicts",
            None,
            share=False,
            formula="world claims with case evidence on both sides",
            population=_TALLIED,
            evidence="world claims of every tag; such a claim reads supported",
        ),
        "web_checked": _Reading(
            "Claims checked on the web",
            None,
            share=False,
            formula="world claims checked on the web",
            population=_TALLIED,
            evidence="world claims of every tag the judge doubted, no case evidence covered, and the web budget reached",
        ),
        "web_skipped": _Reading(
            "Claims past the web budget",
            None,
            share=False,
            formula="world claims nominated for the web check past the cell's budget",
            population=_TALLIED,
            evidence="world claims of every tag the judge doubted, no case evidence covered, and the cell's web budget "
            "did not reach; each reads unverifiable",
        ),
        "session_claims": _Reading(
            "Claims about the session itself",
            None,
            share=False,
            formula="session claims",
            population=_TALLIED,
            evidence="claims about the output or session itself (what was just presented, what comes next, a count "
            "of the session's own state), which no truth step grades and no world reading counts",
        ),
        "ground_lost": _Reading(
            "Cells whose groundedness was lost",
            None,
            share=True,
            formula="1 when the cell lost its ground step alone, else 0",
            population="every cell a criterion declaring grounded_in graded, not lost",
            evidence="a lost ground step (a reply cut short or unreadable, the repair budget spent, on the candidate's "
            "own documents) costs the cell its grounded share and no other reading: its truth readings stand",
        ),
        "lost": _Reading(
            "Cells the criterion lost",
            None,
            share=True,
            formula="1 when the criterion lost the cell, else 0",
            population="every cell the criterion was asked to grade",
            evidence="a lost cell (a reply cut short or unreadable, a budget spent, a failed port or trace write, the "
            "cost cap, the deadline) carries no other reading of the criterion, and keeps every other reading",
        ),
    }
)

#: The readings that rank, each on its own.
_RANKED: Final[frozenset[str]] = frozenset(
    suffix for suffix, reading in _READINGS.items() if reading.lower_is_better is not None
)
#: The readings only a criterion declaring ``grounded_in`` has.
_GROUNDED: Final[frozenset[str]] = frozenset({"grounded_share", "ground_lost"})


def claim_reading_names(criterion: str, *, grounded: bool) -> tuple[str, ...]:
    """Every reading name a criterion's tally can give, ranked first, in one order.

    :param criterion: The criterion's name.
    :param grounded: Whether it declares ``grounded_in``, which alone gives the grounded share.
    :return: ``<criterion>_<suffix>`` for each of its readings.
    """
    return tuple(f"{criterion}_{suffix}" for suffix in _READINGS if grounded or suffix not in _GROUNDED)


def claim_readings(criterion: str, outcome: ClaimTally | ClaimLoss, *, grounded: bool) -> dict[str, float]:
    """One cell's readings under one criterion, computed from its tally, or its one reading when the criterion lost it.

    A reading whose population leaves the cell out is absent, never zero: a cell with no unit has no per-unit reading,
    one with no asserted specific claim, or whose ground step was lost (:attr:`ClaimTally.ground_loss`), no grounded
    share. A cell whose documents were declared and are none (the
    candidate was given or fetched nothing) reads a grounded share of 0 when it asserts a specific claim: it stated
    specifics with no source, which is what the reading ranks.

    :param criterion: The criterion's name, which every reading name starts with.
    :param outcome: The cell's tally under it (:attr:`EvalResult.claim_tallies`), or why the cell lost it
        (:attr:`EvalResult.claim_failures`).
    :param grounded: Whether the criterion declares ``grounded_in``.
    :return: Each reading the cell carries, by name.
    """
    if isinstance(outcome, str):
        return {f"{criterion}_lost": 1.0}
    tally = outcome
    count = _summed(tally.by_judge)
    values: dict[str, float | None] = {
        "false_per_unit": _ratio(count.asserted_contradicted, tally.units),
        "false_unit_share": _ratio(tally.units_with_false, tally.units),
        "grounded_share": (
            _ratio(count.grounded_specific, count.asserted_specific)
            if grounded and tally.documents_declared and tally.ground_loss is None
            else None
        ),
        "false": count.asserted_contradicted,
        "false_hedged": count.hedged_contradicted,
        "false_attributed": count.attributed_contradicted,
        "false_share": _ratio(count.asserted_contradicted, count.asserted),
        "claims": count.claims,
        "units": tally.units,
        "hedged_share": _ratio(count.hedged, count.claims),
        "unverifiable": count.unverifiable,
        "true_ungrounded": count.true_ungrounded,
        "web_settled_share": _ratio(count.web_decided, count.asserted),
        "evidence_conflicts": count.evidence_conflicts,
        "web_checked": count.web_checked,
        "web_skipped": count.web_skipped,
        "session_claims": tally.session_claims,
        "ground_lost": (0.0 if tally.ground_loss is None else 1.0) if grounded else None,
        "lost": 0.0,
    }
    return {f"{criterion}_{suffix}": float(value) for suffix, value in values.items() if value is not None}


def describe_claim_reading(name: str, *, criterion: str, unit: str = "unit") -> MetricDescriptor:
    """Describe one of a criterion's readings by construction, as a rubric dimension is described.

    Judge-mediated: a model listed and graded every claim it counts, so a reading is comparable only under the same
    judge. The description states the reading's population and the evidence its claims were graded on.

    :param name: The reading's name, ``<criterion>_<suffix>``.
    :param criterion: The criterion it reads, whose name the reading's begins with. Needed because a criterion's
        name may hold ``_``, so a reading's name alone does not say where the criterion's ends.
    :param unit: What a page calls one unit of the criterion (``EvalTemplate.claim_unit_names``), for its reader
        name only: presentation, never part of what the reading is.
    :return: The descriptor.
    :raises ValueError: ``name`` is not ``criterion`` and one of its reading suffixes.
    """
    prefix = f"{criterion}_"
    reading = _READINGS.get(name[len(prefix) :]) if name.startswith(prefix) else None
    if reading is None:
        raise ValueError(
            f"{name!r} is no reading of claim criterion {criterion!r}; its readings are "
            + ", ".join(f"{criterion}_{suffix}" for suffix in _READINGS)
        )
    words = reading.words.format(unit=unit, Unit=unit[:1].upper() + unit[1:])
    ranked = reading.lower_is_better is not None
    return MetricDescriptor(
        name=name,
        reader_name=f"{words} ({criterion})",
        data_type="numeric",
        family=CLAIMS_FAMILY,
        transferability_class="judge_mediated",
        attribution_scope="end_to_end",
        description=(
            f"Claim criterion {criterion!r}: {reading.formula}. "
            + (
                f"Ranked, {'lower' if reading.lower_is_better else 'higher'} is better"
                if ranked
                else "A diagnostic, with no better end"
            )
            + f". Read over {reading.population}. Evidence: {reading.evidence}."
        ),
        higher_is_better=None if reading.lower_is_better is None else not reading.lower_is_better,
        value_range=(0.0, 1.0) if reading.share else None,
        nonnegative=not reading.share,
        scale="ratio",
        formula=reading.formula,
        diagnostic=not ranked,
    )


def claim_reading_ranks(name: str, *, criterion: str) -> bool:
    """Whether a criterion's reading is one of the three that rank, rather than a diagnostic.

    :param name: The reading's name.
    :param criterion: The criterion it reads.
    :return: True for the false claims per unit, the share of units with a false claim and the grounded share.
    """
    return name.startswith(f"{criterion}_") and name[len(criterion) + 1 :] in _RANKED


def claim_names_a_measure_holds(template: EvalTemplate, measures: MeasureRegistry) -> str | None:
    """Say which of a template's claim criteria names a measure the host or the engine describes, or None.

    A criterion owns its name and every name ``<criterion>_...`` for its readings, which are judged readings keyed by
    name: a described measure of such a name would be read two ways wherever a name is resolved (a bar, a report's
    figure), so a template whose criterion takes one is refused where it is admitted and where it launches
    (:func:`~threetears.evals.run.authoring.refuse_claim_names_a_measure_holds`). Within the
    template the same rule holds against its rubric and its other criteria (``EvalTemplate``'s own validation).

    :param template: The template.
    :param measures: The host's measure registry, whose catalogue joins the engine's core.
    :return: The refusal, naming each criterion and the measures it takes; None when no criterion takes one.
    """
    described = [descriptor.name for descriptor in list_metrics(measures)]
    taken = {
        criterion.name: sorted(
            name for name in described if name == criterion.name or name.startswith(f"{criterion.name}_")
        )
        for criterion in template.claim_criteria
    }
    if not (clashes := {criterion: names for criterion, names in taken.items() if names}):
        return None
    return "; ".join(
        f"claim criterion {criterion!r} owns its name and every name {criterion + '_...'!r} for its readings, and the "
        f"host describes the measure{'s' if len(names) > 1 else ''} {', '.join(map(repr, names))}"
        for criterion, names in sorted(clashes.items())
    )


def claim_tally(
    grade: ClaimGrade,
    evidence: ClaimEvidence,
    *,
    grounded_in: Collection[EvidenceProvenance] | None,
    ground_loss: KnownClaimLoss | None = None,
) -> ClaimTally:
    """A cell's tally under one criterion, from its grade and the evidence it graded: the one place a claim's tags
    and grades become counts.

    - A **session** claim counts in ``session_claims`` alone, in no world count, whatever it says.
    - A **world** claim's truth rests on case evidence alone: a truth citation of the candidate's own page is refused.
    - A **world** claim counts under the judge key of the call that decided it (:attr:`GradedClaim.decided_by`).
    - An **asserted** claim is neither hedged nor attributed; a statement the text backs with a source is asserted.
    - A **grounded** claim counts only when asserted and specific, and every ground citation must cite a document of a
      provenance ``grounded_in`` names: case evidence the candidate was not given never grounds, and is refused.
      A key fact is never a document the candidate was given, so a ground citation of one is refused too.
    - A **lost ground step** (``ground_loss``) costs only groundedness: every world claim's ``grounded`` reads None,
      no claim is grounded, the truth counts stand, and the tally records the loss so no grounded share is read. A
      world claim whose ``grounded`` is None under ``grounded_in`` with no such loss is refused, never read as 0.
    - **Units** are the evidence's distinct units, however many segments each spans; a unit holds a false claim when
      any of its segments holds an asserted contradicted world claim, read off the segment, which a claim's ``unit``
      must agree with.

    :param grade: The criterion's grade of the cell.
    :param evidence: The claim evidence it graded.
    :param grounded_in: The provenances the criterion declares its candidate was given, or None for a criterion with
        no groundedness reading.
    :param ground_loss: Why the verifier lost the ground step alone (its ``Verification.ground_loss``), or None.
    :return: The tally.
    :raises ValueError: A claim cites the candidate's own (``run``) evidence for its truth, which only case evidence
        decides; a claim cites evidence of a provenance ``grounded_in`` does not name as its ground (any, where the
        criterion asks no groundedness), or a key fact as its ground; a world claim's groundedness disagrees with
        whether the criterion asks it and whether its ground step was lost; a criterion with ``grounded_in`` grades a
        cell whose documents are undeclared; a world claim names no deciding call; or a claim sits in a segment the
        evidence does not hold, or names a unit other than its segment's.
    """
    if grounded_in is None and ground_loss is not None:
        raise ValueError(f"a ground step was lost ({ground_loss!r}), and the criterion asks no groundedness")
    if grounded_in is not None and evidence.documents is None:
        raise ValueError(
            "a criterion with grounded_in reads the documents the candidate was given, and the cell declares none"
        )
    provenance = {entry.ref: entry.provenance for entry in grade.evidence}
    key_facts = {entry.ref for entry in grade.evidence if entry.fact is not None}
    allowed = frozenset(grounded_in or ())
    groups: dict[tuple[object, ...], list[GradedClaim]] = {}
    calls: dict[tuple[object, ...], ClaimCall] = {}
    units_with_false: set[int] = set()
    session = 0
    for place, claim in enumerate(grade.claims):
        if claim.segment >= len(evidence.segments):
            raise ValueError(
                f"claim {place} sits in segment {claim.segment}, and the evidence holds {len(evidence.segments)}"
            )
        if claim.unit != (unit := evidence.segments[claim.segment].unit):
            raise ValueError(f"claim {place} names unit {claim.unit}, and its segment {claim.segment} is unit {unit}")
        # prose-canary: allow — `scope` here is the closed Literal world/session, not an insight's free-text scope
        if claim.scope == "session":
            session += 1
            continue
        if any(provenance[c.ref] != "case" for c in claim.truth_citations):
            raise ValueError(
                f"claim {place} cites run evidence for its truth, and truth is graded against case evidence alone"
            )
        if stray := sorted({provenance[c.ref] for c in claim.ground_citations} - allowed):
            raise ValueError(
                f"claim {place} cites {', '.join(stray)} evidence as its ground, and the criterion is grounded in "
                + (", ".join(sorted(allowed)) if allowed else "nothing")
            )
        if facts := sorted(c.ref for c in claim.ground_citations if c.ref in key_facts):
            raise ValueError(
                f"claim {place} cites key fact {', '.join(facts)} as its ground, and a key fact is no document the "
                "candidate was given"
            )
        asked = grounded_in is not None and ground_loss is None
        if (claim.grounded is None) == asked:
            raise ValueError(
                f"claim {place} reads grounded={claim.grounded}, and the criterion "
                + (
                    "asks no groundedness"
                    if grounded_in is None
                    else f"lost its ground step ({ground_loss!r})"
                    if ground_loss is not None
                    else "asked its ground step, which was not lost; a lost step is the tally's ground_loss"
                )
            )
        if claim.decided_by is None:
            raise ValueError(f"world claim {place} names no call that decided it")
        if claim.verdict == "contradicted" and _asserted(claim) and unit is not None:
            units_with_false.add(unit)
        call = grade.calls[claim.decided_by]
        key = (
            call.served_model,
            call.judge_temperature,
            None if call.judge_sent_settings is None else request_settings_order(call.judge_sent_settings),
        )
        groups.setdefault(key, []).append(claim)
        calls.setdefault(key, call)
    return ClaimTally(
        criterion_fingerprint=grade.criterion_fingerprint,
        documents_declared=evidence.documents is not None,
        units=len({segment.unit for segment in evidence.segments if segment.unit is not None}),
        units_with_false=len(units_with_false),
        session_claims=session,
        by_judge=tuple(_counted(calls[key], claims) for key, claims in groups.items()),
        ground_loss=ground_loss,
    )


def _asserted(claim: GradedClaim) -> bool:
    return not claim.hedged and not claim.attributed


def _counted(call: ClaimCall, claims: list[GradedClaim]) -> ClaimJudgeCounts:
    """The counts of the world claims one judge key decided, under the identity of the first call of that key."""
    asserted = [claim for claim in claims if _asserted(claim)]
    false = [claim for claim in claims if claim.verdict == "contradicted"]
    return ClaimJudgeCounts(
        served_model=call.served_model,
        judge_temperature=call.judge_temperature,
        judge_sent_settings=call.judge_sent_settings,
        claims=len(claims),
        asserted=len(asserted),
        hedged=sum(claim.hedged for claim in claims),
        attributed=sum(claim.attributed for claim in claims),
        asserted_specific=sum(claim.specific for claim in asserted),
        contradicted=len(false),
        asserted_contradicted=sum(_asserted(claim) for claim in false),
        hedged_contradicted=sum(claim.hedged for claim in false),
        attributed_contradicted=sum(claim.attributed for claim in false),
        grounded_specific=sum(bool(claim.specific and claim.grounded) for claim in asserted),
        unverifiable=sum(claim.verdict == "unverifiable" for claim in claims),
        true_ungrounded=sum(claim.verdict == "true_ungrounded" for claim in claims),
        web_decided=sum(claim.graded_against == "web" for claim in asserted),
        evidence_conflicts=sum(claim.evidence_conflict for claim in claims),
        web_checked=sum(claim.web == "checked" for claim in claims),
        web_skipped=sum(claim.web == "skipped_cap" for claim in claims),
    )


@dataclass(frozen=True)
class _Counts:
    """A cell's world-claim counts, summed over its judge keys."""

    claims: int = 0
    asserted: int = 0
    hedged: int = 0
    asserted_specific: int = 0
    asserted_contradicted: int = 0
    hedged_contradicted: int = 0
    attributed_contradicted: int = 0
    grounded_specific: int = 0
    unverifiable: int = 0
    true_ungrounded: int = 0
    web_decided: int = 0
    evidence_conflicts: int = 0
    web_checked: int = 0
    web_skipped: int = 0


def _summed(entries: tuple[ClaimJudgeCounts, ...]) -> _Counts:
    return _Counts(**{field.name: sum(getattr(entry, field.name) for entry in entries) for field in fields(_Counts)})


def _ratio(part: int, whole: int) -> float | None:
    return part / whole if whole else None


__all__: list[str] = []
