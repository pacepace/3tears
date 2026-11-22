"""A host's own readings of a claim criterion's grade: run through its claim reader, held to their descriptors.

A host declares a measure with ``derived_from: <criterion>`` (:attr:`~threetears.evals.kernel.metrics.MetricDescriptor.derived_from`)
and registers one :data:`~threetears.evals.kernel.host.measures.ClaimReader` for that criterion on its profile
(:attr:`~threetears.evals.kernel.host.profile.HostProfile.claim_readers`). Wherever a criterion's tally is made from
its grade (the claims phase, a re-judge, the settling of a row pending its trace), the reader is handed the grade and
the cell's claim evidence and its answer is stored on the tally (:attr:`~threetears.evals.schema.models.ClaimTally.derived`),
so the readings are lost, pending and withheld exactly as the criterion's own are. A re-measure reads them again from
the stored grade when the host moves a definition.

**A reader that raises, or answers what its readings cannot hold, is the host's defect** (:class:`ClaimReaderDefect`):
it fails what asked, naming the criterion and the measure. It is never read as a value, and never as the criterion's
loss, which would charge the arm for the host's code.

Engine-internal: the grading and re-measure paths call it, and nothing here is a host's to import.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from threetears.evals.kernel.metrics import MetricDescriptor

if TYPE_CHECKING:
    from threetears.evals.kernel.host.measures import ClaimReader, MeasureRegistry
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.models import ClaimEvidence, ClaimGrade


class ClaimReaderDefect(RuntimeError):
    """A host's claim reader raised, or answered a reading its measures do not hold.

    The host's code, not an outcome of the cell: the same reader reads every cell of the run. So it fails what asked
    (the run, a re-judge, a re-measure), passed through every per-criterion catch, and is never recorded as the
    criterion's loss or as a value.
    """


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


def derive_claim_readings(derivation: ClaimDerivation, grade: ClaimGrade, evidence: ClaimEvidence) -> dict[str, float]:
    """The host's readings of one cell's grade under one criterion, held to their descriptors, as the tally stores them.

    :param derivation: The criterion's derivation (:func:`claim_derivations`).
    :param grade: The criterion's grade of the cell.
    :param evidence: The cell's claim evidence.
    :return: Each reading the reader answered, by name: a boolean one as 1.0 or 0.0. Sorted by name.
    :raises ClaimReaderDefect: The reader raised; answered something other than a mapping (an awaitable: a reader is
        synchronous); answered a name no measure derived from the criterion holds; or answered a value its measure
        cannot hold (a boolean for a numeric measure or a number for a boolean one, a value that is not finite, outside
        the measure's value range, or below zero for a nonnegative one).
    """
    criterion = derivation.criterion
    try:
        answered = derivation.reader(grade, evidence)
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
