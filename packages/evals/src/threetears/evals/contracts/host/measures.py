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
(:mod:`threetears.evals.contracts.evidence_tiers`). A consumer inheriting a score field with no
marking will quote it as though it were a latency. ``transferability_class`` already draws this
line on the descriptor; the registry is where a surface can ask it without knowing the
descriptor's shape.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from threetears.evals.contracts.host.attribution import HostAttributed

# The metric vocabulary is read for annotations here and called once, inside a method. At module
# level it would close a cycle as soon as this package's root is imported eagerly: the root loads
# this module, ``contracts.metrics`` loads ``contracts.models``, and ``contracts.models`` itself
# imports leaves of this package, so whichever of the two a process reaches first would find the
# other half-initialised.
if TYPE_CHECKING:
    from threetears.evals.contracts.metrics import MeasureFamily, MeritAxis, MetricDescriptor


class MeasureRegistrationError(ValueError):
    """A measure declaration contradicts what this registry promises."""


class MeasureRegistry(HostAttributed):
    """One host's declared measures, validated at construction.

    Validated here rather than at module import for the reason
    :class:`~threetears.evals.contracts.host.sweepables.SweepableRegistry` is: two hosts coexist, and a
    module-level table validated at import can only ever express one.

    **This registry IS R10's observability map, and it answers by handing over the record.**
    There is no membership predicate here and adding one back is the thing to resist: every
    caller that wants to know whether a measure is declared also wants its descriptor, which
    :meth:`get` returns in the same lookup, so a boolean asked first cannot change any verdict.
    Why the axis is derived rather than gated at all is on
    :class:`~threetears.evals.contracts.host.profile.HostProfile`.
    """

    def __init__(self, descriptors: Iterable[MetricDescriptor], *, families: Iterable[MeasureFamily] = ()) -> None:
        """Validate and store one host's measure catalogue.

        Args:
            descriptors: The measures this host can see.
            families: The host's own measure families, beyond the engine's
                (:data:`~threetears.evals.contracts.metrics.ENGINE_FAMILIES`). A descriptor may name one of
                these or an engine family, and nothing else.

        Raises:
            MeasureRegistrationError: The catalogue is unsound. Every defect is reported at once.
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
        from threetears.evals.contracts.metrics import ENGINE_FAMILIES

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
        from threetears.evals.contracts.metrics import ENGINE_FAMILIES, containment_defects

        defects: list[str] = self._family_defects()
        seen: set[str] = set()
        known_families = set(ENGINE_FAMILIES) | {family.name for family in self._families}

        by_name = {d.name: d for d in self._descriptors}
        for descriptor in self._descriptors:
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

    @classmethod
    def from_catalog(
        cls, catalog: Mapping[str, MetricDescriptor], *, families: Iterable[MeasureFamily] = ()
    ) -> MeasureRegistry:
        """Build a registry from an existing ``{name: descriptor}`` catalogue.

        Args:
            catalog: The descriptors, keyed by name.
            families: The host's own measure families.

        Returns:
            A validated registry over the catalogue's values.
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

    def merit_axis(self, name: str) -> MeritAxis | None:
        """Which merit axis ``name`` serves, when its host declared one.

        The axis vocabulary is the engine's — quality, cost, latency, reliability — and for a
        measure the HOST declares, so is the assignment: that is the same split the sweepables
        registry draws one level up, generic slot, host-declared content. The engine's own core
        measures are the exception, and it assigns their axes itself in
        :mod:`threetears.evals.contracts.metrics` (the turn-latency family to latency, the production-replicating
        cost to cost), because every host shares them and a decision surface with no cost or
        latency column is what leaving them unassigned produced.

        Args:
            name: A declared measure name.

        Returns:
            The declared axis, or None when the measure serves none.

        Raises:
            KeyError: This host declared no such measure.
        """
        return self._require(name).merit_axis

    def _require(self, name: str) -> MetricDescriptor:
        """The descriptor for ``name``, or a KeyError naming the host that lacks it."""
        descriptor = self._by_name.get(name)
        if descriptor is None:
            raise KeyError(f"{self._host}{name} is not a measure this host declared")
        return descriptor


__all__ = ["MeasureRegistrationError", "MeasureRegistry"]
