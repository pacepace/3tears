"""What a host can see — the measures it declares, and what each one is a number *about*.

The engine computes over measures and never interprets one. A host declares its own catalogue;
:class:`MeasureRegistry` holds one host's, validates it, and answers the questions the analysis
layer asks of a measure it was handed.

**Two properties this registry adds to a descriptor, both from evidence rather than design.**

*A measure names its own population.* ``mean_score`` is computed differently by different
surfaces — one drops a cell an apparatus fault produced, another keeps its raw rows — so over any
corpus with one infra-excluded cell the two figures differ **by construction** and must never be
quoted side by side. Inside one product that is a caveat a reader carries; across a package
boundary it is a cross-repo ambiguity. Two surfaces reporting "mean_score" must either agree or
be forced to disagree in their names.

*A judge-mediated number and a mechanical one are different kinds of number.* Latency, cost,
convergence rate, call counts and ordering predicates carry no judge. A rubric mean carries one,
and it bears only the evidence tier its judge's measured reliability earns
(:mod:`threetears.evals.kernel.evidence_tiers`). A consumer inheriting a score field with no
marking will quote it as though it were a latency. ``transferability_class`` already draws this
line on the descriptor; the registry is where a surface can ask it without knowing the
descriptor's shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from threetears.evals.kernel.host.attribution import HostAttributed
from threetears.evals.kernel.metrics import (
    CLAIMS_FAMILY,
    ENGINE_FAMILIES,
    METRIC_DESCRIPTORS,
    containment_defects,
    measure_definition_digests,
    reader_name_defects,
)

if TYPE_CHECKING:
    from threetears.evals.kernel.metrics import MeasureFamily, MeritAxis, MetricDescriptor
    from threetears.evals.schema.models import EvalResult, EvalRun, EvalTrace


@dataclass(frozen=True, slots=True)
class MeasureRereadRequest:
    """One stored result a re-measure asks the host to read again, and the measures to read.

    :ivar run: The result's run, as stored.
    :ivar result: The stored result (one trial), holding the values its cell wrote.
    :ivar trace: What the cell stored beside it (its transcript, call ledger, end state, judge evidence), or ``None``
        for a result that stored no trace.
    :ivar measures: The keys of the host measures to read again, sorted: each one the host declares now, and whose
        definition the run did not record or recorded otherwise.
    """

    run: EvalRun
    result: EvalResult
    trace: EvalTrace | None
    measures: tuple[str, ...]


#: Read a stored result's host measures again, under the host's CURRENT definition of each, from what the cell stored
#: and nothing else. Host code the engine calls from a re-measure (``threetears.evals.run.remeasure_run``) and never
#: inspects. It answers ``{measure key: value}`` for each requested measure it read; a requested measure it leaves out
#: is one this result has no value of under the current definition. A measure the result HOLDS a value of must be
#: answered: the re-measure refuses the run otherwise, since the stored value was read under another definition and
#: nothing may guess the new one. Raise to refuse a result outright.
MeasureRereader = Callable[[MeasureRereadRequest], "Mapping[str, bool | float | str]"]


@dataclass(frozen=True, slots=True)
class ViewedClaim:
    """PROVISIONAL. One claim a criterion's judge listed in a cell, as a claim reader sees it.

    :ivar text: The proposition, stated on its own.
    :ivar excerpt: The span of its segment that states it, verbatim.
    :ivar scope: ``world`` (about the world: graded for truth) or ``session`` (about the output or session itself:
        graded by no truth step, and what a reader most often reads against the session record).
    :ivar verdict: Its truth verdict (``supported``, ``contradicted``, ``unverifiable``, ``true_ungrounded``); None for
        a session claim, which no truth step grades.
    :ivar specific: Whether it states a name, number, date or rule a source could settle.
    :ivar hedged: Whether the text marks it as its own uncertain recollection.
    :ivar attributed: Whether the text reports it without vouching for it.
    :ivar unit: Its segment's unit, or None for a segment outside every unit.
    :ivar segment: The position of its segment among the view's ``segments``.
    """

    text: str
    excerpt: str
    scope: Literal["world", "session"]
    verdict: str | None
    specific: bool
    hedged: bool
    attributed: bool
    unit: int | None
    segment: int


@dataclass(frozen=True, slots=True)
class ViewedSegment:
    """PROVISIONAL. One segment of a cell's output, as a claim reader sees it.

    :ivar text: Its text.
    :ivar unit: Its unit, or None for a segment outside every unit.
    :ivar tags: The host's own tags on it, read-only.
    """

    text: str
    unit: int | None
    tags: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ClaimView:
    """PROVISIONAL. What a claim reader is handed of one cell under one criterion: a frozen copy, never the stored grade
    or evidence, so nothing a reader does reaches what the engine stores, and the stored shapes may move under it.

    It grows by adding fields, never by exposing the grade. A host whose reader comes to read a field the view gains
    raises its measures' ``definition_revision``.

    :ivar claims: Every claim the judge listed, in the grade's order.
    :ivar segments: The output's segments, in order.
    :ivar session: The run's own record the cell's kind stored (``ClaimEvidence.session``), deep-frozen: every mapping
        read-only and every list a tuple; None where the kind stored none.
    """

    claims: tuple[ViewedClaim, ...]
    segments: tuple[ViewedSegment, ...]
    session: Mapping[str, object] | None


#: PROVISIONAL. Read one cell's grade under a claim criterion into the host's own readings of it: the measures it
#: declares with ``derived_from`` naming that criterion (:attr:`~threetears.evals.kernel.metrics.MetricDescriptor.derived_from`),
#: by name. Pure and synchronous, and handed a frozen :class:`ClaimView` of the cell: its claims (each one's ``scope``,
#: excerpt, tags and verdict), its segments with their ``tags``, and the run's own ``session`` record (a play order, a
#: game state). Host code the engine calls from the claims phase, a re-judge, the settling of a pending row and a
#: re-measure, and never inspects; launch runs it first on two synthetic views and refuses a reader that fails there.
#: A reading it leaves out is one the cell does not carry (none of its claims is about what the reading counts), never
#: a zero. Where it raises, or answers what its measures cannot hold, on a graded cell, the cell keeps its grade and
#: tally, its derived readings are marked unread with the defect, and a re-measure with a fixed reader reads them.
ClaimReader = Callable[[ClaimView], "Mapping[str, float | bool]"]


class ClaimReaderDefect(RuntimeError):
    """PROVISIONAL. A host's claim reader (:data:`ClaimReader`) raised, or answered a reading its measures do not hold.

    The host's code, not an outcome of the cell, and never recorded as the criterion's loss or as a value. Where it
    meets a graded cell, the cell's derived readings are marked unread (``ClaimTally.derived_unread``) and the defect is
    recorded where an operator sees it; at launch, and in a re-measure, which pay for nothing, it refuses what asked.
    """


class MeasureRegistrationError(ValueError):
    """A measure declaration contradicts what this registry promises."""


class MeasureRegistry(HostAttributed):
    """One host's declared measures, validated at construction.

    Validated here rather than at module import for the reason
    :class:`~threetears.evals.kernel.host.sweepables.SweepableRegistry` is: two hosts coexist, and a
    module-level table validated at import can only ever express one.

    **This registry IS R10's observability map, and it answers by handing over the record.**
    There is no membership predicate here and adding one back is the thing to resist: every
    caller that wants to know whether a measure is declared also wants its descriptor, which
    :meth:`get` returns in the same lookup, so a boolean asked first cannot change any verdict.
    Why the axis is derived rather than gated at all is on
    :class:`~threetears.evals.kernel.host.profile.HostProfile`.
    """

    def __init__(self, descriptors: Iterable[MetricDescriptor], *, families: Iterable[MeasureFamily] = ()) -> None:
        """Validate and store one host's measure catalogue.

        :param descriptors: The measures this host can see.
        :param families: The host's own measure families, beyond the engine's
            (:data:`~threetears.evals.kernel.metrics.ENGINE_FAMILIES`). A descriptor may name one of
            these or an engine family, and nothing else.
        :raises MeasureRegistrationError: The catalogue is unsound. Every defect is reported at once.
        """
        self._init_attribution()
        self._descriptors: tuple[MetricDescriptor, ...] = tuple(descriptors)
        self._families: tuple[MeasureFamily, ...] = tuple(families)
        if defects := self._defects():
            raise MeasureRegistrationError("measure declaration is unsound: " + "; ".join(defects))
        self._by_name: dict[str, MetricDescriptor] = {d.name: d for d in self._descriptors}
        self._family_by_name: dict[str, MeasureFamily] = {f.name: f for f in self._families}

    def _family_defects(self) -> list[str]:
        """Name every way the host's own families contradict the engine's or each other."""
        defects: list[str] = []
        names = [family.name for family in self._families]
        defects.extend(
            f"family {name} is declared twice — a descriptor naming it could not say which it meant"
            for name in sorted({name for name in names if names.count(name) > 1})
        )
        defects.extend(
            f"family {name} is one of the engine's own — a host family needs a name of its own, or the engine's "
            "would be redefined under every host that shares the process"
            for name in sorted(set(names) & set(ENGINE_FAMILIES))
        )
        return defects

    def _defects(self) -> list[str]:
        """Name every way the catalogue contradicts what this registry promises."""
        defects: list[str] = self._family_defects()
        # Every rendered surface heads a measure by what a reader calls it, so a host measure with no reader name
        # would print its key there — refused here, at startup, rather than discovered on a chart. Checked with the
        # engine's core alongside, because a host measure called what a core measure is called heads a column a
        # reader cannot tell from the core's. A core-named measure is refused below on its own account, so it is
        # left out here rather than reported twice.
        host = [d for d in self._descriptors if d.name not in METRIC_DESCRIPTORS]
        defects.extend(reader_name_defects([*host, *METRIC_DESCRIPTORS.values()]))
        # A host measure named like a core one cannot be described as the host's: every resolver
        # consults the core first, so it would read with the core's meaning, direction and range,
        # and its values would pool into the engine's own distribution under that name with `n`
        # inflated. Refused here, by name, rather than resolved quietly in either direction.
        defects.extend(
            f"{name} is one of the engine's core measures — a host's {name} would be read under the core's meaning, "
            "direction and range and pooled into the engine's own observations of it; give the host's measure a "
            "name of its own"
            for name in sorted({d.name for d in self._descriptors} & set(METRIC_DESCRIPTORS))
        )
        seen: set[str] = set()
        known_families = set(ENGINE_FAMILIES) | {family.name for family in self._families}
        graded_by = {
            **{name: family.graded_by for name, family in ENGINE_FAMILIES.items()},
            **{family.name: family.graded_by for family in self._families},
        }

        by_name = {d.name: d for d in self._descriptors}
        for descriptor in self._descriptors:
            # A check fails its trial, and so moves the ranked pass rate, on its value alone. A judge-graded number
            # carries no judge identity and no check of the judge behind it here, and nothing ranks on one
            # (`is_code_graded`), so a check in such a family would rank by the back door. Refused at startup, by
            # name, rather than filtered out of the pass rule where no host would see it go.
            # Read off the shape, not `is_check`: that answers False in an engine judged family, the very case named here.
            check_shaped = descriptor.higher_is_better is not None and (
                descriptor.data_type == "boolean" or descriptor.pass_fail
            )
            if check_shaped and descriptor.family is not None and graded_by.get(descriptor.family) == "judge":
                shape = "a pass/fail scorer" if descriptor.pass_fail else "a boolean"
                defects.append(
                    f"{descriptor.name} is a check ({shape} with a better end) in family {descriptor.family}, which "
                    "is graded by a judge — a check fails its trial on its value alone, and a judge-graded measure "
                    "may not decide a pass: declare the family graded_by='code' if code made the number, or drop "
                    "the measure's better end (higher_is_better=None)"
                )
            if descriptor.derived_from is not None:
                defects.extend(self._derived_defects(descriptor, graded_by))
            if descriptor.name in seen:
                defects.append(
                    f"{descriptor.name} is declared twice — lookups key by name, so one would shadow the other"
                )
            seen.add(descriptor.name)
            if descriptor.family is not None and descriptor.family not in known_families:
                defects.append(
                    f"{descriptor.name} names family {descriptor.family!r}, which neither the engine nor this host "
                    "declares — declare it as a MeasureFamily on the registry, or name one that exists"
                )
            # The bundle's measure surfaces keep only a typed measure, so a direction or a diagnostic declared without
            # a data type would put a measure on no bundle at all, numbers included. Refused here, at startup, rather
            # than discovered as a missing column. An undirected measure that is no diagnostic stays legal untyped:
            # it is a raw count either way, and no surface would carry it whatever its type.
            if descriptor.data_type is None and (descriptor.higher_is_better is not None or descriptor.diagnostic):
                declared = "a better end" if descriptor.higher_is_better is not None else "itself a diagnostic"
                defects.append(
                    f"{descriptor.name} declares {declared} and no data_type — no bundle would carry it; declare "
                    "its data_type (numeric, for a number)"
                )
            if descriptor.value_range is not None:
                low, high = descriptor.value_range
                if low > high:
                    defects.append(f"{descriptor.name} declares a value range whose floor is above its ceiling")
            # A containment declaration licenses a SUBTRACTION downstream, and the engine has
            # checked its own seed at import since the rule was written. A host catalogue is the
            # first one outside that seed, so the check follows it here rather than applying to
            # only half the measure space — the defect it catches produces a plausible number
            # describing nothing, whichever catalogue declared the part.
            defects.extend(f"{descriptor.name} {defect}" for defect in containment_defects(descriptor, by_name))
        return defects

    @staticmethod
    def _derived_defects(descriptor: MetricDescriptor, graded_by: Mapping[str, str]) -> list[str]:
        """Name every way a reading derived from a claim criterion contradicts what such a reading is.

        It is a judged reading of the criterion's grade, ranked, withheld and lost with the criterion's own readings, so
        it sits in a family a judge grades (code made none of its claims), under the criterion's key, and is a number;
        and it is none of the things a measure a kind lands is (a guardrail held per arm, a merit axis an optimizing
        surface reads, a definition that shaped the run). Which criterion it names is checked where a template is in
        hand, since one registry serves every template.

        :param descriptor: A descriptor declaring ``derived_from``.
        :param graded_by: Who grades each family the engine and this host declare, by name.
        :return: The defects, each naming the measure.
        """
        criterion = descriptor.derived_from
        name = descriptor.name
        defects: list[str] = []
        family = descriptor.family
        if family is None or graded_by.get(family) != "judge":
            defects.append(
                f"{name} is derived from claim criterion {criterion!r} in "
                + ("no family" if family is None else f"family {family}, which is graded by code")
                + " — a reading of a judge's claims is judge-graded: declare family 'claims', or a host family "
                "graded_by='judge'"
            )
        elif family in ENGINE_FAMILIES and family != CLAIMS_FAMILY:
            defects.append(
                f"{name} is derived from claim criterion {criterion!r} in the engine's {family} family, which holds "
                "the engine's own judged numbers — declare family 'claims', or a host family graded_by='judge'"
            )
        if not name.startswith(f"{criterion}_") or name == f"{criterion}_":
            defects.append(
                f"{name} is derived from claim criterion {criterion!r} and is not named under its key — name it "
                f"{criterion}_<what it reads>, as the criterion's own readings are"
            )
        if descriptor.data_type not in ("numeric", "boolean"):
            defects.append(
                f"{name} is derived from claim criterion {criterion!r} with data_type {descriptor.data_type!r} — a "
                "reading of claims is a number, numeric or boolean"
            )
        defects.extend(
            f"{name} is derived from claim criterion {criterion!r} and declares {what} — {why}"
            for what, held, why in (
                (
                    "itself a guardrail",
                    descriptor.guardrail,
                    "a claim reading ranks arms and is never held to a margin",
                ),
                (
                    f"merit axis {descriptor.merit_axis}",
                    descriptor.merit_axis is not None,
                    "a claim reading ranks on its own, never on an axis an optimizing surface trades",
                ),
                ("shapes_run", descriptor.shapes_run, "it is read after the cell is graded, so it shaped nothing"),
            )
            if held
        )
        return defects

    @property
    def derived(self) -> tuple[MetricDescriptor, ...]:
        """The measures this host derives from claim criteria (``derived_from``), in declaration order."""
        return tuple(d for d in self._descriptors if d.derived_from is not None)

    @classmethod
    def from_catalog(
        cls, catalog: Mapping[str, MetricDescriptor], *, families: Iterable[MeasureFamily] = ()
    ) -> MeasureRegistry:
        """Build a registry from an existing ``{name: descriptor}`` catalogue.

        :param catalog: The descriptors, keyed by name.
        :param families: The host's own measure families.
        :return: A validated registry over the catalogue's values.
        """
        return cls(catalog.values(), families=families)

    @property
    def families(self) -> tuple[MeasureFamily, ...]:
        """The host's own measure families, in declaration order — the engine's are not repeated here."""
        return self._families

    def family(self, name: str) -> MeasureFamily | None:
        """The host family named ``name``, or None when this host declares no family by that name."""
        return self._family_by_name.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared measure name, in declaration order."""
        return tuple(d.name for d in self._descriptors)

    def get(self, name: str) -> MetricDescriptor | None:
        """The descriptor for ``name``, or None when this host cannot see that measure."""
        return self._by_name.get(name)

    def definition_digests(self) -> dict[str, str]:
        """Each declared measure's definition digest, by its key: what a run records as its ``measure_definitions``.

        :func:`~threetears.evals.kernel.metrics.measure_definition_digests` over this catalogue. The launch stamps it
        on every run; a host that assembles a run itself (a witnessed one) stamps it the same way, since a run that
        records none pools with nothing.

        :return: Measure key to digest, sorted; ``{}`` for a host that declares no measure.
        """
        return measure_definition_digests(self._descriptors)

    def merit_axis(self, name: str) -> MeritAxis | None:
        """Which merit axis ``name`` serves, when its host declared one.

        The axis vocabulary is the engine's — quality, cost, latency, reliability — and for a
        measure the HOST declares, so is the assignment: that is the same split the sweepables
        registry draws one level up, generic slot, host-declared content. The engine's own core
        measures are the exception, and it assigns their axes itself in
        :mod:`threetears.evals.kernel.metrics` (the turn-latency family to latency, the production-replicating
        cost to cost), because every host shares them and a decision surface with no cost or
        latency column is what leaving them unassigned produced.

        :param name: A declared measure name.
        :return: The declared axis, or None when the measure serves none.
        :raises KeyError: This host declared no such measure.
        """
        return self._require(name).merit_axis

    def _require(self, name: str) -> MetricDescriptor:
        """The descriptor for ``name``, or a KeyError naming the host that lacks it."""
        descriptor = self._by_name.get(name)
        if descriptor is None:
            raise KeyError(f"{self._host}{name} is not a measure this host declared")
        return descriptor


__all__ = [
    "ClaimReader",
    "ClaimReaderDefect",
    "ClaimView",
    "ViewedClaim",
    "ViewedSegment",
    "MeasureRegistrationError",
    "MeasureRegistry",
    "MeasureRereadRequest",
    "MeasureRereader",
]
