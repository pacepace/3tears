"""A host's own readings of a claim criterion's grade: run through its claim reader, held to their descriptors.

A host declares a measure with ``derived_from: <criterion>`` (:attr:`~threetears.evals.kernel.metrics.MetricDescriptor.derived_from`)
and registers one :data:`~threetears.evals.kernel.host.measures.ClaimReader` for that criterion on its profile
(:attr:`~threetears.evals.kernel.host.profile.HostProfile.claim_readers`). Wherever a criterion's tally is made from
its grade (the claims phase, a re-judge, the settling of a row pending its trace), the reader is handed a frozen view of
the grade and the cell's claim evidence (:func:`claim_view`), never the stored shapes, and its answer is stored on the
tally (:attr:`~threetears.evals.schema.models.ClaimTally.derived`), so the readings are lost, pending and withheld
exactly as the criterion's own are. A re-measure reads them again from the stored grade when the host moves a
definition, or where a reader once failed.

**A reader that raises, or answers what its readings cannot hold, is the host's defect** (:class:`ClaimReaderDefect`),
never read as a value and never as the criterion's loss, which would charge the arm for the host's code. Launch runs
each reader on two synthetic views first (:func:`claim_reader_refusal`) and refuses one that fails there, before any
spend. On a graded cell the tally keeps its counts and records the defect instead of the readings
(``ClaimTally.derived_unread``); a re-measure, which pays for nothing, raises it.

Engine-internal: the grading and re-measure paths call it, and nothing here is a host's to import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from threetears.evals.kernel.host.measures import ClaimReaderDefect, ClaimView, ViewedClaim, ViewedSegment
from threetears.evals.kernel.metrics import MetricDescriptor

if TYPE_CHECKING:
    from threetears.evals.kernel.host.measures import ClaimReader, MeasureRegistry
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.models import ClaimEvidence, ClaimGrade


@dataclass(frozen=True)
class ClaimDerivation:
    """One criterion's host readings: the reader that answers them and the measures it answers.

    :ivar criterion: The criterion's name.
    :ivar reader: The host's reader of its grade.
    :ivar readings: Each measure derived from it, by name.
    """

    criterion: str
    reader: ClaimReader
    readings: Mapping[str, MetricDescriptor]


def claim_derivations(profile: HostProfile) -> Mapping[str, ClaimDerivation]:
    """Each criterion a host derives readings from, with its reader and measures, by criterion name.

    The profile refuses a reader with no measure and a measure with no reader at registration, so every entry holds
    both.

    :param profile: The host.
    :return: criterion -> its derivation; empty for a host that derives nothing.
    """
    return MappingProxyType(
        {
            criterion: ClaimDerivation(
                criterion=criterion,
                reader=reader,
                readings=MappingProxyType({d.name: d for d in profile.measures.derived if d.derived_from == criterion}),
            )
            for criterion, reader in profile.claim_readers.items()
        }
    )


def derived_descriptors(measures: MeasureRegistry | None, criterion: str) -> dict[str, MetricDescriptor]:
    """The measures a host derives from ``criterion``, by name; none without a registry.

    :param measures: The host's measure registry, or None where a surface holds none.
    :param criterion: The criterion.
    :return: name -> descriptor.
    """
    if measures is None:
        return {}
    return {d.name: d for d in measures.derived if d.derived_from == criterion}


def _frozen(value: object) -> object:
    """``value``, a JSON value, with every mapping read-only and every list a tuple, all the way down: a copy, so
    nothing done to it reaches the stored record."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(held) for key, held in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_frozen(held) for held in value)
    return value


def claim_view(grade: ClaimGrade, evidence: ClaimEvidence) -> ClaimView:
    """What a claim reader is handed of one cell's grade and claim evidence: a frozen copy of the fields it reads.

    :param grade: The criterion's grade of the cell.
    :param evidence: The cell's claim evidence.
    :return: The view.
    """
    session = evidence.session
    return ClaimView(
        claims=tuple(
            ViewedClaim(
                text=claim.claim,
                excerpt=claim.excerpt,
                scope=claim.scope,
                verdict=claim.verdict,
                specific=claim.specific,
                hedged=claim.hedged,
                attributed=claim.attributed,
                unit=claim.unit,
                segment=claim.segment,
            )
            for claim in grade.claims
        ),
        segments=tuple(
            ViewedSegment(text=segment.text, unit=segment.unit, tags=MappingProxyType(dict(segment.tags)))
            for segment in evidence.segments
        ),
        session=None if session is None else MappingProxyType({key: _frozen(held) for key, held in session.items()}),
    )


#: The views launch reads each reader on, before any spend: a cell with no claim and no session record, and one with a
#: world claim in a unit and a session claim outside every unit under an empty session record. A reader that handles a
#: cell only through what its own runs happen to hold fails here, by name, rather than on a paid cell.
_SYNTHETIC_VIEWS: tuple[ClaimView, ...] = (
    ClaimView(
        claims=(),
        segments=(ViewedSegment(text="A synthetic segment.", unit=None, tags=MappingProxyType({})),),
        session=None,
    ),
    ClaimView(
        claims=(
            ViewedClaim(
                text="A synthetic world claim.",
                excerpt="A synthetic world claim",
                scope="world",
                verdict="unverifiable",
                specific=True,
                hedged=False,
                attributed=False,
                unit=0,
                segment=0,
            ),
            ViewedClaim(
                text="A synthetic session claim.",
                excerpt="A synthetic session claim",
                scope="session",
                verdict=None,
                specific=True,
                hedged=False,
                attributed=False,
                unit=None,
                segment=1,
            ),
        ),
        segments=(
            ViewedSegment(text="A synthetic world claim.", unit=0, tags=MappingProxyType({})),
            ViewedSegment(text="A synthetic session claim.", unit=None, tags=MappingProxyType({})),
        ),
        session=MappingProxyType({}),
    ),
)


def claim_reader_refusal(derivation: ClaimDerivation) -> str | None:
    """Why a host's claim reader cannot be trusted with a paid cell, read on :data:`_SYNTHETIC_VIEWS`; None when it
    reads both.

    :param derivation: The criterion's derivation.
    :return: The refusal, naming the criterion, the reader and what it did; None when it read both views.
    """
    for view in _SYNTHETIC_VIEWS:
        try:
            _held_to_its_measures(derivation, view)
        except ClaimReaderDefect as defect:
            reader = getattr(derivation.reader, "__qualname__", type(derivation.reader).__name__)
            return f"{defect} (the reader {reader}, on a synthetic cell with {len(view.claims)} claim(s))"
    return None


def derive_claim_readings(derivation: ClaimDerivation, grade: ClaimGrade, evidence: ClaimEvidence) -> dict[str, float]:
    """The host's readings of one cell's grade under one criterion, held to their descriptors, as the tally stores them.

    The reader is handed :func:`claim_view` of the grade and the evidence, never the stored shapes.

    :param derivation: The criterion's derivation (:func:`claim_derivations`).
    :param grade: The criterion's grade of the cell.
    :param evidence: The cell's claim evidence.
    :return: Each reading the reader answered, by name: a boolean one as 1.0 or 0.0. Sorted by name.
    :raises ClaimReaderDefect: The reader raised; answered something other than a mapping (an awaitable: a reader is
        synchronous); answered a name no measure derived from the criterion holds; or answered a value its measure
        cannot hold (a boolean for a numeric measure or a number for a boolean one, a value that is not finite, outside
        the measure's value range, or below zero for a nonnegative one).
    """
    return _held_to_its_measures(derivation, claim_view(grade, evidence))


def _held_to_its_measures(derivation: ClaimDerivation, view: ClaimView) -> dict[str, float]:
    """The reader's answer on ``view``, held to its measures (:func:`derive_claim_readings`)."""
    criterion = derivation.criterion
    try:
        answered = derivation.reader(view)
    except Exception as raised:  # prawduct:ok-broad-except — the host's reader is its code; raised as its defect
        raise ClaimReaderDefect(
            f"the claim reader for criterion {criterion!r} raised {type(raised).__name__}: {raised}"
        ) from raised
    if not isinstance(answered, Mapping):
        raise ClaimReaderDefect(
            f"the claim reader for criterion {criterion!r} answered a {type(answered).__name__}, not a mapping of "
            "reading names to values; a claim reader is synchronous and answers a mapping"
        )
    readings: dict[str, float] = {}
    defects: list[str] = []
    for name, value in sorted(answered.items(), key=lambda item: str(item[0])):
        descriptor = derivation.readings.get(name) if isinstance(name, str) else None
        if descriptor is None:
            defects.append(f"{name!r} is no measure derived from it (MetricDescriptor.derived_from={criterion!r})")
            continue
        if (why := _misfit(descriptor, value)) is not None:
            defects.append(f"{name!r} reads {value!r}, which {why}")
            continue
        readings[name] = float(value)
    if defects:
        raise ClaimReaderDefect(f"the claim reader for criterion {criterion!r} answered: " + "; ".join(defects))
    return readings


def _misfit(descriptor: MetricDescriptor, value: object) -> str | None:
    """Why ``value`` is no value of ``descriptor``, or None when it is one."""
    if descriptor.data_type == "boolean":
        return None if isinstance(value, bool) else "is not a bool, and the measure is boolean"
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "is not a number, and the measure is numeric"
    if not math.isfinite(value):
        return "is not finite"
    if descriptor.value_range is not None and not descriptor.value_range[0] <= value <= descriptor.value_range[1]:
        return f"is outside the measure's value range {descriptor.value_range}"
    if descriptor.nonnegative and value < 0:
        return "is below zero, and the measure is nonnegative"
    return None


__all__: list[str] = []
