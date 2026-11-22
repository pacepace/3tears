"""A claim criterion's readings, computed on read from its stored tally, and the tally, derived from its grade.

A cell's grade (:class:`~threetears.evals.schema.ClaimGrade`) lists every claim a criterion's judge found and how each
was graded; its tally (:class:`~threetears.evals.schema.ClaimTally`) keeps the counts; every reading is a pure function
of the tally, so none is stored and a diagnostic costs nothing. Three readings rank, each on its own and never in a
composite:

- **R1** ``<c>_false_per_unit``: asserted world claims contradicted by evidence, per unit. Lower is better.
- **R1b** ``<c>_false_unit_share``: the share of units holding at least one. Lower is better.
- **R2** ``<c>_grounded_share``: of asserted specific world claims, the share a document the candidate itself was given
  supports, for a criterion declaring ``grounded_in``. Higher is better.

Every other reading is a diagnostic: it explains a movement and has no better end. A host may add readings of its own,
computed from each cell's grade and claim evidence by its claim reader and stored on the tally
(:mod:`~threetears.evals.kernel.claims.derived`): each is described by the host's descriptor, ranks where that declares
a better end, and is lost, pending and withheld as the criterion's own are. What a claim's tags mean for each
count (a session claim counts in none of the world readings; a hedged or attributed claim is graded for truth and kept
out of the asserted counts; a statement the text backs with a source is asserted) is settled once, in
:func:`claim_tally`.

The judge phase tallies and the ranking path reads. One name is a host's: :func:`claim_readings`, exported from
:mod:`threetears.evals.kernel` (provisional), so a host reads a cell's readings off its stored tally exactly as the
ranking path does and never restates a formula. Nothing else here is a host's to import.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Final

from threetears.evals.kernel.claims.derived import derived_descriptors
from threetears.evals.kernel.metrics import CLAIMS_FAMILY, MetricDescriptor, describe_rubric_dim, list_metrics
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
from threetears.evals.schema.models import (
    CLAIM_PENDING,
    ClaimCall,
    ClaimCriterion,
    GradedClaim,
    RubricScale,
    request_settings_order,
    rubric_dim_scale,
)


@dataclass(frozen=True)
class _Reading:
    """One reading a criterion's tally gives, as its descriptor and its reader state it.

    :ivar words: What a reader calls it, before the criterion it reads (``False claims per {unit}``; ``{units}`` and
        ``{Units}`` its plural).
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
            "Share of {units} with a false claim",
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
            "{Units}",
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
            "Claims the web did not check",
            None,
            share=False,
            formula="world claims nominated for the web check and not checked: past the cell's budget, refused by the "
            "cost cap, or cut by the cell's deadline",
            population=_TALLIED,
            evidence="world claims of every tag the judge doubted and no case evidence covered, that the web did not "
            "check; each reads unverifiable, never false",
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


def claim_readings(
    criterion: str, outcome: ClaimTally | ClaimLoss, *, grounded: bool | None = None
) -> dict[str, float]:
    """One cell's readings under one criterion, computed from its tally, or its one reading when the criterion lost it.

    A reading whose population leaves the cell out is absent, never zero: a cell with no unit has no per-unit reading,
    one with no asserted specific claim, or whose ground step was lost (:attr:`ClaimTally.ground_loss`), no grounded
    share. A cell whose documents were declared and are none (the
    candidate was given or fetched nothing) reads a grounded share of 0 when it asserts a specific claim: it stated
    specifics with no source, which is what the reading ranks.

    :param criterion: The criterion's name, which every reading name starts with.
    :param outcome: The cell's tally under it (:attr:`EvalResult.claim_tallies`), or why the cell lost it
        (:attr:`EvalResult.claim_failures`).
    :param grounded: Whether the criterion declares ``grounded_in``; None reads it off the tally
        (:attr:`ClaimTally.grounded`), which a lost cell's single reading does not need.
    :return: Each reading the cell carries, by name, the host's derived readings (:attr:`ClaimTally.derived`) after the
        engine's; none while its tally waits on its trace
        (:data:`~threetears.evals.schema.models.CLAIM_PENDING`), which is not yet known and never a loss, and none for
        a cell whose candidate said nothing (:attr:`ClaimTally.silent`), which has no denominator and is no loss.
    :raises ValueError: ``grounded`` disagrees with the tally's own record of whether its criterion asks groundedness.
    """
    if outcome == CLAIM_PENDING:
        return {}
    if isinstance(outcome, str):
        return {f"{criterion}_lost": 1.0}
    tally = outcome
    if tally.silent:
        # The candidate said nothing: no denominator, so no reading, and never a count of 0 false claims in a mean.
        return {}
    if grounded is None:
        grounded = tally.grounded
    elif grounded != tally.grounded:
        raise ValueError(
            f"claim criterion {criterion!r} is read as grounded={grounded}, and its tally records grounded={tally.grounded}"
        )
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
    readings = {f"{criterion}_{suffix}": float(value) for suffix, value in values.items() if value is not None}
    return {**readings, **{name: value for name, value in tally.derived.items() if name not in readings}}


def describe_claim_reading(
    name: str, *, criterion: str, unit: str = "unit", measures: MeasureRegistry | None = None
) -> MetricDescriptor:
    """Describe one of a criterion's readings by construction, as a rubric dimension is described, or a reading the
    host derives from it by the host's own descriptor.

    Judge-mediated: a model listed and graded every claim it counts, so a reading is comparable only under the same
    judge. The description states the reading's population and the evidence its claims were graded on.

    :param name: The reading's name, ``<criterion>_<suffix>``.
    :param criterion: The criterion it reads, whose name the reading's begins with. Needed because a criterion's
        name may hold ``_``, so a reading's name alone does not say where the criterion's ends.
    :param unit: What a page calls one unit of the criterion (``EvalTemplate.claim_unit_names``), for its reader
        name only: presentation, never part of what the reading is.
    :param measures: The host's measure registry, which describes the readings it derives from the criterion
        (``MetricDescriptor.derived_from``); None where a surface holds none, which describes the engine's alone.
    :return: The descriptor.
    :raises ValueError: ``name`` is neither ``criterion`` and one of its reading suffixes nor a measure ``measures``
        derives from it.
    """
    prefix = f"{criterion}_"
    reading = _READINGS.get(name[len(prefix) :]) if name.startswith(prefix) else None
    if reading is None:
        if (derived := derived_descriptors(measures, criterion).get(name)) is not None:
            return derived
        raise ValueError(
            f"{name!r} is no reading of claim criterion {criterion!r}"
            + (", and no host registry was given to describe one it derives" if measures is None else "")
            + "; its readings are "
            + ", ".join(f"{criterion}_{suffix}" for suffix in _READINGS)
            + "".join(f", {derived}" for derived in derived_descriptors(measures, criterion))
        )
    units = _plural(unit)
    words = reading.words.format(unit=unit, units=units, Units=units[:1].upper() + units[1:])
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


def _plural(noun: str) -> str:
    """A unit noun's plural by the regular English rules (``answer`` → ``answers``, ``entry`` → ``entries``, ``match``
    → ``matches``): the host names one unit, and a page names many."""
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    return noun + "s"


def claim_reading_ranks(name: str, *, criterion: str, measures: MeasureRegistry | None = None) -> bool:
    """Whether a criterion's reading is one that ranks, rather than a diagnostic.

    :param name: The reading's name.
    :param criterion: The criterion it reads.
    :param measures: The host's measure registry, whose readings derived from the criterion rank where they declare a
        better end; None for the engine's readings alone.
    :return: True for the false claims per unit, the share of units with a false claim and the grounded share, and a
        derived reading with a better end.
    """
    if name.startswith(f"{criterion}_") and name[len(criterion) + 1 :] in _READINGS:
        return name[len(criterion) + 1 :] in _RANKED
    derived = derived_descriptors(measures, criterion).get(name)
    return derived is not None and derived.higher_is_better is not None


#: The ``outcome`` a projected claim row carries where its criterion lost the cell, or, on the grounded share, where
#: the cell lost its ground step alone: the row holds no value, stays out of the reading's mean, and counts as a loss.
CLAIM_LOST_OUTCOME: Final = "claim_lost"
#: The ``outcome`` a projected claim row carries while its criterion's tally waits on its trace
#: (:data:`~threetears.evals.schema.models.CLAIM_PENDING`): no value and no loss, counted apart so the population
#: never shrinks without a word.
CLAIM_PENDING_OUTCOME: Final = "claim_pending"

#: The ``outcome`` a projected claim row carries where the candidate said nothing in the cell
#: (:attr:`~threetears.evals.schema.models.ClaimTally.silent`): no value and no loss, since the cell has no denominator,
#: counted apart so an arm silent in most of its cells is seen to be.
CLAIM_SILENT_OUTCOME: Final = "claim_silent"

#: The ``outcome`` a projected row of a reading the host derives from a criterion carries where its claim reader could
#: not read the cell (:attr:`~threetears.evals.schema.models.ClaimTally.derived_unread`): no value and no loss, since the
#: cell's grade stands and a re-measure with a fixed reader reads it, counted apart so the population never shrinks
#: without a word.
CLAIM_UNREAD_OUTCOME: Final = "claim_unread"

#: What each reading's claims rest on, as a calibration question names it (``<criterion>.<question>``): the
#: groundedness readings on whether the documents shown support a claim, the counts of claims on whether the judge's
#: claim is one, every other on whether a claim is true.
_QUESTION_OF: Final[Mapping[str, str]] = MappingProxyType(
    {
        "grounded_share": "grounded",
        "ground_lost": "grounded",
        "claims": "is_claim",
        "units": "is_claim",
        "hedged_share": "is_claim",
        "session_claims": "is_claim",
    }
)


def ranked_claim_reading_names(criterion: str, *, grounded: bool) -> tuple[str, ...]:
    """The readings of a criterion that rank, in one order: the two truth readings, then the grounded share.

    :param criterion: The criterion's name.
    :param grounded: Whether it declares ``grounded_in``.
    :return: Their names.
    """
    return tuple(
        name
        for name in claim_reading_names(criterion, grounded=grounded)
        if claim_reading_ranks(name, criterion=criterion)
    )


def projected_claim_readings(
    criterion: str, outcome: ClaimTally | ClaimLoss, *, grounded: bool, derived_ranked: Collection[str] = ()
) -> list[tuple[str, float | None, str | None]]:
    """One cell's rows under one criterion, as the score projection emits them: ``(reading, value, outcome)``, the
    outcome None where the row takes the cell's own.

    Every reading the cell carries (:func:`claim_readings`) is a valued row. A ranked reading the cell would carry but
    lost is a null row whose outcome says so, so each arm counts what it lost: every ranked reading where the criterion
    lost the cell (:data:`CLAIM_LOST_OUTCOME`), the grounded share alone where only its ground step was, and every ranked
    reading, under :data:`CLAIM_PENDING_OUTCOME`, while the tally waits on its trace, which is no loss; and every ranked
    reading, under :data:`CLAIM_SILENT_OUTCOME`, where the candidate said nothing, which is no loss either.

    :param criterion: The criterion's name.
    :param outcome: The cell's tally under it, or why it lost it.
    :param grounded: Whether the criterion asks groundedness, for a cell with no tally to say so (a lost or pending
        one); a tally's own record wins.
    :param derived_ranked: The readings the host derives from the criterion that rank, which a lost or pending cell
        counts as the criterion's own ranked readings are counted, and a cell its reader could not read counts under
        :data:`CLAIM_UNREAD_OUTCOME`; a ground or web step's loss never touches them.
    :return: The rows, valued ones first.
    """
    rows: list[tuple[str, float | None, str | None]] = [
        (name, value, None) for name, value in claim_readings(criterion, outcome).items()
    ]
    if isinstance(outcome, str):
        missing = CLAIM_PENDING_OUTCOME if outcome == CLAIM_PENDING else CLAIM_LOST_OUTCOME
        rows += [
            (name, None, missing)
            for name in (*ranked_claim_reading_names(criterion, grounded=grounded), *sorted(derived_ranked))
        ]
    elif outcome.silent:
        rows += [
            (name, None, CLAIM_SILENT_OUTCOME)
            for name in (*ranked_claim_reading_names(criterion, grounded=outcome.grounded), *sorted(derived_ranked))
        ]
    else:
        if outcome.grounded and outcome.ground_loss is not None:
            rows.append((f"{criterion}_grounded_share", None, CLAIM_LOST_OUTCOME))
        if outcome.derived_unread is not None:
            rows += [(name, None, CLAIM_UNREAD_OUTCOME) for name in sorted(derived_ranked)]
    return rows


def claim_reading_is_share(name: str, criterion: str, measures: MeasureRegistry | None = None) -> bool:
    """Whether a criterion's reading is a share on 0 to 1, rather than a count or a rate per unit.

    :param name: The reading's name.
    :param criterion: The criterion it reads.
    :param measures: The host's measure registry, which describes the readings it derives from the criterion. Where
        a surface holds none, a derived reading reads as no share, so a page prints its number as it stands.
    :return: Whether it is a share.
    """
    if measures is None and not (name.startswith(f"{criterion}_") and name[len(criterion) + 1 :] in _READINGS):
        return False
    return describe_claim_reading(name, criterion=criterion, measures=measures).value_range == (0.0, 1.0)


def claim_reading_question(name: str, *, criterion: str) -> str:
    """The question a reading's claims rest on, as a judge key and a calibration read name it: ``<criterion>.true``
    for a truth reading, ``<criterion>.grounded`` for a groundedness one, ``<criterion>.is_claim`` for a count of
    claims.

    A reading the host derives from the criterion rests on ``is_claim``: code read the claims the judge listed (the
    session ones above all, whose truth no step grades) against the cell's own record.

    :param name: The reading's name.
    :param criterion: The criterion it reads.
    :return: The question.
    """
    suffix = name.removeprefix(f"{criterion}_")
    if suffix not in _READINGS:
        return f"{criterion}.is_claim"
    return f"{criterion}.{_QUESTION_OF.get(suffix, 'true')}"


def claim_judge_config_id(criterion: str, fingerprint: str) -> str:
    """The judge config a claim reading's judge key names: ``claims:<criterion>@<the fingerprint's first 12>``.

    :param criterion: The criterion's name.
    :param fingerprint: The criterion fingerprint its tally counts under.
    :return: The id.
    """
    return f"claims:{criterion}@{fingerprint[:12]}"


def describe_judged_reading(
    name: str,
    *,
    scales: Mapping[str, RubricScale],
    instruments: Mapping[str, str],
    units: Mapping[str, str] | None = None,
    measures: MeasureRegistry | None = None,
) -> MetricDescriptor:
    """Describe a reading of the judged namespace: a claim criterion's reading, or a rubric dimension.

    The one dispatch every reader of a judged name goes through, so a claim reading is never described on a rubric
    scale it was not answered on, and a rubric dimension never as a rate.

    :param name: The reading's name.
    :param scales: The scale each rubric dimension was answered on.
    :param instruments: The criterion each claim reading belongs to, by reading name (:func:`claim_instruments`).
    :param units: What a page calls one unit of each criterion (``EvalTemplate.claim_unit_names``); ``unit`` where
        none is named.
    :param measures: The host's measure registry, which describes the readings it derives from a criterion.
    :return: The descriptor.
    :raises ValueError: ``name`` is no claim reading and no scale was recorded for it, or a claim reading neither the
        engine nor ``measures`` describes.
    """
    criterion = instruments.get(name)
    if criterion is not None:
        return describe_claim_reading(
            name, criterion=criterion, unit=(units or {}).get(criterion, "unit"), measures=measures
        )
    return describe_rubric_dim(name, scale=rubric_dim_scale(name, scales))


def claim_instruments(rows: Iterable[tuple[str | None, str | None]]) -> dict[str, str]:
    """The criterion each claim reading belongs to, from ``(reading, instrument)`` pairs the projection's rows carry.

    :param rows: Each row's reading name and instrument; a row with no instrument is no claim row and is passed over.
    :return: ``{reading: criterion}``.
    :raises ValueError: One reading name belongs to two criteria (``a`` and ``a_session`` both give
        ``a_session_claims``), which no reader could tell apart.
    """
    found: dict[str, str] = {}
    for name, instrument in rows:
        if name is None or instrument is None:
            continue
        if found.setdefault(name, instrument) != instrument:
            raise ValueError(
                f"claim reading {name!r} belongs to criteria {found[name]!r} and {instrument!r}; name one of them apart"
            )
    return found


def template_claim_readings(
    template: EvalTemplate, measures: MeasureRegistry | None = None
) -> dict[str, MetricDescriptor]:
    """Every reading a template's claim criteria give, described, by name: what a bar may name before any result.

    :param template: The template.
    :param measures: The host's measure registry, whose readings derived from one of the template's criteria join the
        criterion's own; None for the engine's alone.
    :return: ``{reading: descriptor}``.
    """
    return {
        **{
            name: describe_claim_reading(
                name, criterion=criterion.name, unit=template.claim_unit_names.get(criterion.name, "unit")
            )
            for criterion in template.claim_criteria
            for name in claim_reading_names(criterion.name, grounded=criterion.grounded_in is not None)
        },
        **{
            name: descriptor
            for criterion in template.claim_criteria
            for name, descriptor in derived_descriptors(measures, criterion.name).items()
        },
    }


def claim_names_a_measure_holds(template: EvalTemplate, measures: MeasureRegistry) -> str | None:
    """Say which of a template's claim criteria names a measure the host or the engine describes, or None.

    A criterion owns its name and every name ``<criterion>_...`` for its readings, which are judged readings keyed by
    name: a described measure of such a name would be read two ways wherever a name is resolved (a bar, a report's
    figure), so a template whose criterion takes one is refused where it is admitted and where it launches
    (:func:`~threetears.evals.run.authoring.refuse_claim_names_a_measure_holds`). Within the
    template the same rule holds against its rubric and its other criteria (``EvalTemplate``'s own validation).
    A measure the host derives from the criterion (``MetricDescriptor.derived_from``) is one of its readings and takes
    its name rightly, unless the name is one the criterion's own readings take.

    :param template: The template.
    :param measures: The host's measure registry, whose catalogue joins the engine's core.
    :return: The refusal, naming each criterion and the measures it takes; None when no criterion takes one.
    """
    described = list_metrics(measures)
    taken = {
        criterion.name: sorted(
            descriptor.name
            for descriptor in described
            if (descriptor.name == criterion.name or descriptor.name.startswith(f"{criterion.name}_"))
            and not _derived_from(descriptor, criterion)
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


def _derived_from(descriptor: MetricDescriptor, criterion: ClaimCriterion) -> bool:
    """Whether ``descriptor`` is a reading the host derives from ``criterion`` under a name of its own: one the
    criterion's engine readings do not take, so it sits under the criterion's key as one of its readings."""
    return descriptor.derived_from == criterion.name and descriptor.name not in claim_reading_names(
        criterion.name, grounded=True
    )


def claim_tally(
    grade: ClaimGrade,
    evidence: ClaimEvidence,
    *,
    grounded_in: Collection[EvidenceProvenance] | None,
    ground_loss: ClaimLoss | None = None,
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
    - **Undeclared documents** (None, not empty) under ``grounded_in`` lose the ground step alone
      (``documents_undeclared``), which the grade records as any ground loss.
    - **A silent cell** (evidence with no segment: the candidate said nothing) is tallied ``silent``, with no unit and
      no claim, and carries no reading at all (:func:`claim_readings`).
    - **Units** are the evidence's distinct units, however many segments each spans; a unit holds a false claim when
      any of its segments holds an asserted contradicted world claim, read off the segment, which a claim's ``unit``
      must agree with.
    - A **failed web step** costs only the web's decisions: the claims it left undecided read ``lost`` and count
      unverifiable, and the tally carries the grade's ``web_loss``, read off the stored grade so a tally recomputed
      from it keeps the loss.

    :param grade: The criterion's grade of the cell.
    :param evidence: The claim evidence it graded.
    :param grounded_in: The provenances the criterion declares its candidate was given, or None for a criterion with
        no groundedness reading.
    :param ground_loss: Why the ground step alone was lost (the verifier's ``Verification.ground_loss``, as the grade
        records it in ``ClaimGrade.ground_loss``), or None.
    :return: The tally.
    :raises ValueError: A claim cites the candidate's own (``run``) evidence for its truth, which only case evidence
        decides; a claim cites evidence of a provenance ``grounded_in`` does not name as its ground (any, where the
        criterion asks no groundedness), or a key fact as its ground; a world claim's groundedness disagrees with
        whether the criterion asks it and whether its ground step was lost; a criterion with ``grounded_in`` grades a
        cell whose documents are undeclared with its ground step not lost as ``documents_undeclared``; a silent cell
        lists a claim or loses a step; a world claim names no deciding call; or a claim sits in a segment the
        evidence does not hold, or names a unit other than its segment's.
    """
    if grounded_in is None and ground_loss is not None:
        raise ValueError(f"a ground step was lost ({ground_loss!r}), and the criterion asks no groundedness")
    silent = not evidence.segments
    if grounded_in is not None and evidence.documents is None and not silent and ground_loss != "documents_undeclared":
        raise ValueError(
            "a criterion with grounded_in reads the documents the candidate was given, and the cell declares none, "
            "which loses its ground step as documents_undeclared"
        )
    if silent and (grade.claims or ground_loss is not None):
        raise ValueError("a cell whose candidate said nothing lists no claim and loses no step")
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
        silent=silent,
        grounded=grounded_in is not None,
        ground_loss=ground_loss,
        web_loss=grade.web_loss,
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
