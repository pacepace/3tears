"""The bar a behavior has to clear — registered by the host, ratcheted by the engine.

A bar is the incumbent standard for one behavior on one measure: *field accuracy is at least
0.92*. Registering it makes "did this clear the bar" a question with an answer, rather than a
judgement each reader makes from the number.

**A bar is read by the interval, never the mean, and three ways.** Against the threshold less the
measure's declared margin (its materiality threshold), a cell clears when its whole interval is on the
good side, misses when its whole interval is on the bad side, and is undecided when the interval
straddles the line — neither a pass nor a failure. A proposal seeds the threshold at the incumbent's
mean moved by the share of its interval its own error accounts for — see
:func:`~threetears.evals.analysis.stats.interval_clears` and
:func:`~threetears.evals.analysis.stats.bar_seed`. A threshold at the incumbent's mean, read against a
cell's mean, failed an unchanged incumbent about half the time.

**The ratchet is the whole point, and it only tightens.** A campaign may declare a bar stricter
than the registry's; one looser is refused, quoting the registered value. A standard that can be
lowered by the run being measured against it is not a standard.

**One operation calls the ratchet.** :func:`~threetears.evals.analysis.bar_proposals.propose_bars`
measures a baseline campaign's incumbent the way an analysis does and asks :meth:`BarRegistry.propose`
for a bar on each measure with a better end. It returns the proposals; it registers none, because
this registry has no mutation API and a bar reaches it only through a host's own registrations.

**A vacuous seed is flagged, never adopted.** :meth:`BarRegistry.propose` computes the incumbent
configuration's measured baseline and offers it as the bar; what it will not do is register it.
Where that baseline is one nothing could fail — bottomed out at the permissive end of the
measure's own range, or not tightening a bar already registered — the proposal says so, because a
standard every value clears records the current state instead of setting one. The engine cannot
tell a deliberate low bar from an accidental one, so it does not try: it marks the seed and leaves
adoption to a person, which is structural rather than promised since this registry has no
mutation API at all.

**The pass threshold is declared here too, per behavior.** pass^k passes an attempt only when every
goal-state check passed and every capability criterion cleared a threshold on its 1–5 scale (a pass/fail
criterion clears on its pass). That threshold is a standard of the same kind as a bar — what counts as
good enough for this behavior — so a host declares it beside its bars (:class:`PassThreshold`), and
:meth:`BarRegistry.pass_threshold` answers it, :data:`DEFAULT_PASS_THRESHOLD` where none is declared.
Every stored pass^k figure records the threshold it was computed at.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, assert_never

from threetears.evals.contracts.host.attribution import HostAttributed

if TYPE_CHECKING:
    from threetears.evals.contracts.host.measures import MeasureRegistry
    from threetears.evals.contracts.metrics import MetricDescriptor


#: The 1–5 level a criterion must reach for pass^k where the behavior declares no threshold of its own.
DEFAULT_PASS_THRESHOLD = 3

#: The ordinal scale a pass threshold is a level on.
_SCALE_TOP = 5


def pass_threshold_label(k: int | None, threshold: int) -> str:
    """How every surface names pass^k: its depth and the bar a criterion had to clear, e.g. ``pass^k (k=3, criterion >= 4 of 5)``.

    One spelling, so no surface can print a pass^k without the threshold it was computed at.

    Args:
        k: The depth the figure is read at, or None where no depth was reached.
        threshold: The 1–5 level a criterion had to reach.

    Returns:
        The label.
    """
    depth = "k=?" if k is None else f"k={k}"
    return f"pass^k ({depth}, criterion >= {threshold} of {_SCALE_TOP})"


@dataclass(frozen=True)
class PassThreshold:
    """The 1–5 level a capability criterion must reach for an attempt to pass, for one behavior.

    pass^k conjoins every goal-state check with every capability criterion at this level (a pass/fail
    criterion's bar is its pass, whatever this says). A behavior whose quality only counts at "good" sets 4;
    one where "acceptable" is enough keeps the default 3.
    """

    behavior: str
    """The behavior this threshold governs — host vocabulary the engine never interprets."""

    threshold: int
    """The level a criterion must reach, from 2 to 5. 1 is refused: every score clears it, so it would drop
    every criterion from pass^k while the figure still read as a conjunction over them."""

    rationale: str
    """Why this is the level. A threshold with no reason can only be obeyed."""


class BarRegistrationError(ValueError):
    """A bar declaration contradicts what this registry promises."""


@dataclass(frozen=True)
class Bar:
    """The incumbent standard for one behavior on one measure."""

    behavior: str
    """The behavior this bar governs — host vocabulary the engine never interprets."""

    measure: str
    """The measure the bar is read on. Must be a measure the host's registry declares."""

    threshold: float
    """The value the measure must reach — read against a cell's interval and the measure's margin, never its mean."""

    higher_is_better: bool
    """Whether clearing means at-or-above the threshold.

    **Never trusted on a proposal.** This is a second copy of a fact the measure descriptor owns,
    and a ratchet that read it from the bar being checked could be defeated by flipping it: a
    proposal of 0.80 against a registered 0.92 computes ``0.80 < 0.92`` and registers as *tighter*.
    :meth:`BarRegistry.check_override` therefore takes the direction from the **incumbent**, and
    :meth:`BarRegistry.validate_against` refuses a bar that contradicts its measure's descriptor —
    so the two copies cannot disagree for long enough to matter.

    **A third enforcement site covers the case neither of these can.** Both of
    the above need a REGISTERED bar: ``check_override`` compares against an incumbent, and
    ``validate_against`` runs at registration. A host that registers none leaves a
    campaign's own ``BarOverride.direction`` as the only statement of which way clearing runs, and
    nothing above sees it. ``refuse_an_undeclarable_design`` in the analysis layer closes that at
    authoring time by reading the descriptor directly, which is why the rule fires here with an
    empty registry.
    """

    rationale: str
    """Why this is the standard. A threshold with no reason cannot be argued with, only obeyed."""

    vacuous_seed: bool = False
    """True when the incumbent was failing at the moment this was seeded — flagged, never silently adopted."""

    def clears(self, value: float) -> bool:
        """Whether one value is at or beyond this bar — a point, for a value that is exactly known.

        Asked of a measure's declared range end, which is certain. A cell's verdict is never this: a
        measured mean is uncertain, and :func:`~threetears.evals.analysis.stats.interval_clears` decides
        it by the interval.

        Args:
            value: The observed measure value.

        Returns:
            True when the value is at or beyond the threshold in the better direction.
        """
        return value >= self.threshold if self.higher_is_better else value <= self.threshold

    def is_tighter_than(self, incumbent: Bar) -> bool:
        """Whether this bar demands strictly more than the standard already registered.

        **The incumbent's direction governs, not this bar's.** Reading ``self.higher_is_better``
        here is what let a looser proposal register as tighter by flipping one boolean — the
        proposal would be asserting both the threshold and the rule the threshold is judged by.

        Args:
            incumbent: The registered bar this one is proposed against.

        Returns:
            True when this threshold is strictly stricter in the incumbent's better direction.
        """
        if incumbent.higher_is_better:
            return self.threshold > incumbent.threshold
        return self.threshold < incumbent.threshold


@dataclass(frozen=True)
class BarProposal:
    """A bar the ratchet computed from a measurement, for a person to confirm or tighten.

    The ratchet is a **proposal** mechanism. The engine can compute what the incumbent
    configuration currently measures; it cannot know whether that number is a standard worth
    holding, and a registry that adopted it would encode "never ship worse than whatever we
    happen to be shipping" as policy on the strength of one run.

    Adoption is structural rather than promised: :class:`BarRegistry` has no mutation API, so a
    proposal reaches a registry only by a person putting it in a host's registrations.
    """

    bar: Bar
    """The bar being proposed. Carries ``vacuous_seed`` so the flag survives adoption."""

    reason: str
    """Why this seed is vacuous, or ``""`` when it discriminates. Rendered beside the proposal."""

    @property
    def vacuous(self) -> bool:
        """Whether adopting this seed would register a standard nothing can fail."""
        return self.bar.vacuous_seed


class BarRegistry(HostAttributed):
    """One host's registered bars, keyed by ``(behavior, measure)`` and validated at construction."""

    def __init__(self, bars: Iterable[Bar] = (), *, pass_thresholds: Iterable[PassThreshold] = ()) -> None:
        """Validate and store one host's bars and per-behavior pass thresholds.

        Args:
            bars: The registered incumbents. A host with no standards yet registers none, which
                is a well-formed empty registry rather than an error.
            pass_thresholds: The behaviors whose pass^k threshold is not :data:`DEFAULT_PASS_THRESHOLD`.

        Raises:
            BarRegistrationError: The declaration set is unsound.
        """
        self._init_attribution()
        self._bars: tuple[Bar, ...] = tuple(bars)
        self._pass_thresholds: tuple[PassThreshold, ...] = tuple(pass_thresholds)
        if defects := self._defects():
            raise BarRegistrationError("bar declaration is unsound: " + "; ".join(defects))
        self._by_key: dict[tuple[str, str], Bar] = {(b.behavior, b.measure): b for b in self._bars}
        self._threshold_by_behavior = {t.behavior: t.threshold for t in self._pass_thresholds}

    def _defects(self) -> list[str]:
        """Name every way the declaration set contradicts what this registry promises."""
        defects: list[str] = []
        seen: set[tuple[str, str]] = set()
        for bar in self._bars:
            key = (bar.behavior, bar.measure)
            if key in seen:
                defects.append(
                    f"{bar.behavior}/{bar.measure} is declared twice — lookups key by the pair, so one would shadow the other"
                )
            seen.add(key)
            if not bar.rationale.strip():
                defects.append(
                    f"{bar.behavior}/{bar.measure} states no rationale — a threshold with no reason can only be obeyed"
                )
        behaviors: set[str] = set()
        for declared in self._pass_thresholds:
            if declared.behavior in behaviors:
                defects.append(f"{declared.behavior}'s pass threshold is declared twice — one would shadow the other")
            behaviors.add(declared.behavior)
            if isinstance(declared.threshold, bool) or not isinstance(declared.threshold, int):
                defects.append(f"{declared.behavior}'s pass threshold must be a whole level of the 1-5 scale")
            elif not 2 <= declared.threshold <= _SCALE_TOP:
                defects.append(
                    f"{declared.behavior}'s pass threshold is {declared.threshold}, outside 2-{_SCALE_TOP} — at 1 every "
                    "score clears it, so pass^k would conjoin no criterion while reading as though it did"
                )
            if not declared.rationale.strip():
                defects.append(
                    f"{declared.behavior}'s pass threshold states no rationale — a threshold with no reason can only "
                    "be obeyed"
                )
        return defects

    @property
    def pass_thresholds(self) -> tuple[PassThreshold, ...]:
        """Every declared pass threshold, in declaration order."""
        return self._pass_thresholds

    def pass_threshold(self, behavior: str | None) -> int:
        """The 1–5 level a criterion must reach for pass^k on ``behavior``.

        Args:
            behavior: The behavior a figure is computed for, or None where no behavior applies (a lens over a
                scope's runs, which belong to no one behavior).

        Returns:
            The declared threshold, or :data:`DEFAULT_PASS_THRESHOLD` when none is declared.
        """
        if behavior is None:
            return DEFAULT_PASS_THRESHOLD
        return self._threshold_by_behavior.get(behavior, DEFAULT_PASS_THRESHOLD)

    @property
    def bars(self) -> tuple[Bar, ...]:
        """Every registered bar, in declaration order."""
        return self._bars

    def get(self, behavior: str, measure: str) -> Bar | None:
        """The registered bar for one behavior on one measure, or None when none is registered."""
        return self._by_key.get((behavior, measure))

    def check_override(self, proposed: Bar) -> None:
        """Refuse a campaign bar looser than the registered incumbent.

        Args:
            proposed: The bar a campaign wants to be measured against.

        Raises:
            BarRegistrationError: The proposal is looser than the registered bar, quoting the
                registered value so the refusal names what it is protecting.
        """
        incumbent = self.get(proposed.behavior, proposed.measure)
        if incumbent is None:
            return
        if proposed.higher_is_better != incumbent.higher_is_better:
            raise BarRegistrationError(
                f"{self._host}{proposed.behavior}/{proposed.measure} is registered as "
                f"{'higher' if incumbent.higher_is_better else 'lower'}-is-better and this campaign proposes the "
                "opposite — a proposal does not get to restate the rule its own threshold is judged by"
            )
        if not proposed.is_tighter_than(incumbent) and proposed.threshold != incumbent.threshold:
            raise BarRegistrationError(
                f"{self._host}{proposed.behavior}/{proposed.measure} is registered at {incumbent.threshold} and this campaign "
                f"proposes {proposed.threshold}, which is looser — a standard the run being measured can lower is not a standard"
            )

    def propose(
        self, *, behavior: str, measure: str, observed: float, measures: MeasureRegistry, rationale: str
    ) -> BarProposal:
        """Propose the incumbent's measured baseline as this behavior's bar — never adopt it.

        The standard a behavior is held to should start where the behavior already
        performs, so that shipping something worse is a visible regression rather than a matter
        of opinion. What the engine must not do is *set* it: the number is one measurement, and
        a registry that took it would turn an accident of the corpus into policy.

        **A vacuous seed is flagged.** Two shapes of vacuity, and both mean the same thing — the
        bar would be cleared by everything, so registering it buys nothing while looking like a
        standard:

        * the observation sits at (or past) the permissive end of the measure's declared range,
          so no in-range value can fail it. This is the "incumbent currently failing outright"
          case stated in terms the engine can actually check: a quality measure bottomed out at
          0.0 proposes ``>= 0.0``, which is *"never ship worse than something already broken"*;
        * a bar is already registered and the proposal would not tighten it. A ratchet that can
          turn the other way is not a ratchet, and the seed arriving from a measurement rather
          than from an author does not exempt it.

        Args:
            behavior: The behavior the bar would govern.
            measure: The measure it is read on. Must be one this host declares — a proposal on a
                measure nobody can see is a standard nobody can check.
            observed: The incumbent configuration's measured baseline — its mean moved toward the
                permissive end of its interval (:func:`~threetears.evals.analysis.stats.bar_seed`), as
                :func:`~threetears.evals.analysis.propose_bars` passes it.
            measures: The host's measure registry, which owns the better-direction and the range.
            rationale: Why this is the standard. Required for the same reason
                :meth:`__init__` refuses a bar without one.

        Returns:
            The proposal, with ``bar.vacuous_seed`` set and ``reason`` naming which vacuity it is.

        Raises:
            BarRegistrationError: The measure is not declared by this host, it declares no better
                direction (so "clearing" has no meaning), or ``rationale`` is blank.
        """
        descriptor = measures.get(measure)
        if descriptor is None:
            raise BarRegistrationError(
                f"{self._host}cannot propose a bar on '{measure}' — this host does not declare it, so nothing could ever check it"
            )
        if (what := no_better_end(descriptor)) is not None:
            raise BarRegistrationError(
                f"{self._host}cannot propose a bar on '{measure}' — its descriptor declares no better direction "
                f"— {what} — so there is no such thing as clearing it"
            )
        if not rationale.strip():
            raise BarRegistrationError(
                f"{self._host}a proposed bar on {behavior}/{measure} states no rationale — a threshold with no "
                "reason can only be obeyed"
            )

        higher_is_better = descriptor.higher_is_better
        assert higher_is_better is not None, "no_better_end refused a directionless measure above"
        proposed = Bar(
            behavior=behavior,
            measure=measure,
            threshold=observed,
            higher_is_better=higher_is_better,
            rationale=rationale,
        )
        reason = self._vacuity(proposed, descriptor.value_range)
        return BarProposal(bar=replace(proposed, vacuous_seed=bool(reason)), reason=reason)

    def _vacuity(self, proposed: Bar, value_range: tuple[float, float] | None) -> str:
        """Why adopting ``proposed`` would register a standard nothing can fail, or ``""``.

        Args:
            proposed: The bar the ratchet computed.
            value_range: The measure's declared inclusive bounds, or None when it has none.

        Returns:
            The reason, ready to render, or the empty string when the seed discriminates.
        """
        if value_range is not None:
            floor, ceiling = value_range
            permissive = floor if proposed.higher_is_better else ceiling
            if proposed.clears(permissive):
                return (
                    f"the incumbent's measured baseline is {proposed.threshold} on {proposed.measure}, whose declared range is "
                    f"[{floor}, {ceiling}] — a bar there is cleared by every value the measure can take, so it "
                    "records the current state as the standard rather than setting one"
                )
        incumbent = self.get(proposed.behavior, proposed.measure)
        if incumbent is not None and not proposed.is_tighter_than(incumbent):
            return (
                f"{proposed.behavior}/{proposed.measure} is already registered at {incumbent.threshold} and the "
                f"measured baseline is {proposed.threshold}, which does not tighten it — a ratchet that can turn "
                "the other way is not a ratchet, whether the number came from an author or from a run"
            )
        return ""

    def validate_against(self, measures: MeasureRegistry) -> None:
        """Refuse a bar naming a measure this host cannot see, cannot clear, or contradicts.

        ``Bar.measure`` and ``Bar.higher_is_better`` both restate facts the measure descriptor
        owns. Two sources of truth for one discrete fact stay agreed only while somebody checks,
        so this is the check — called from :class:`~threetears.evals.contracts.host.profile.HostProfile` at
        construction, where both registries are in hand.

        A bar on a measure with no better end — a diagnostic, a raw count, a text measure — is refused here as
        :meth:`propose` refuses to seed one, through the same predicate (:func:`no_better_end`):
        nothing ever reads it, so registering it would only make a sentence look like a standard.

        Args:
            measures: The host's measure registry.

        Raises:
            BarRegistrationError: A bar names an undeclared measure or one with no better end, or
                declares the opposite better-direction from the descriptor. Every defect is reported
                at once.
        """
        defects: list[str] = []
        for bar in self._bars:
            descriptor = measures.get(bar.measure)
            if descriptor is None:
                defects.append(f"{bar.behavior}/{bar.measure} names a measure this host does not declare")
                continue
            if (what := no_better_end(descriptor)) is not None:
                defects.append(
                    f"{bar.behavior}/{bar.measure} names a measure that declares no better direction — {what} — "
                    "so there is no such thing as clearing it, and the bar would never be read"
                )
                continue
            if contradicts_descriptor(descriptor, bar.higher_is_better):
                defects.append(
                    f"{bar.behavior}/{bar.measure} declares higher_is_better={bar.higher_is_better} but its measure "
                    f"declares {descriptor.higher_is_better} — one of the two is wrong and the descriptor owns the fact"
                )
        if defects:
            raise BarRegistrationError(f"{self._host}bar declaration is unsound: " + "; ".join(defects))


def contradicts_descriptor(descriptor: MetricDescriptor | None, higher_is_better: bool) -> bool:
    """Whether a proposed better-direction contradicts the one its measure declares.

    The rule the descriptor owns, in one place because it is enforced in TWO: at registration
    (:meth:`BarRegistry.validate_against`) and at campaign authoring
    (``refuse_an_undeclarable_design``), the latter being the only one that fires on a host
    registering no bars. Two hand-written copies of one boolean is how they come to disagree.

    :meth:`BarRegistry.check_override` is deliberately NOT one of them and cannot call this: it
    compares a proposal's direction against the INCUMBENT BAR's, never against a descriptor, which
    is what lets it judge tighter-versus-looser at all. A maintainer tightening this predicate
    should not expect that method to follow.

    A missing descriptor and a directionless one both answer False, and for the same reason rather
    than by coincidence: there is nothing to contradict. Neither enforcement site hands it a missing
    one — registration refuses a bar on an undeclared measure first, and campaign authoring refuses
    a bar naming nothing a result carries first, resolving a template's rubric dimensions and
    goal-state checks to descriptors of their own — so that arm keeps the predicate total rather
    than carrying a case. A declared measure with no better direction has no notion of clearing at
    all, so neither site hands it one either: both refuse it first through :func:`no_better_end`, as
    :meth:`BarRegistry.propose` does before seeding.

    Args:
        descriptor: The measure's descriptor, or None when this host declares no such measure.
        higher_is_better: The direction being proposed.

    Returns:
        True only when the host declared a direction and it is the opposite one.
    """
    return (
        descriptor is not None
        and descriptor.higher_is_better is not None
        and descriptor.higher_is_better != higher_is_better
    )


def no_better_end(descriptor: MetricDescriptor) -> str | None:
    """Say what kind of directionless measure this is — ``"a raw count"``, ``"a text measure"``, … — or None.

    **The one predicate for "a bar on this can never be cleared".** A threshold means something only
    against a better end, so every place that admits a bar asks this: registration
    (:meth:`BarRegistry.validate_against`), the ratchet (:meth:`BarRegistry.propose`), the baseline
    proposer (:func:`~threetears.evals.analysis.propose_bars`) and the campaign gate
    (:func:`~threetears.evals.contracts.declaration.resolve_bar_name`). Each frames its own refusal
    around the answer; none restates the rule, so they cannot come to disagree about which measures it
    covers or what to call them.

    The name comes off the descriptor's own declaration, never off the values: its ``data_type`` says
    what the measure IS, and only a numeric one splits further, on
    :attr:`~threetears.evals.contracts.metrics.MetricDescriptor.diagnostic` — a declared diagnostic, or
    else a raw count (the same split the bundle makes when it decides which directionless measures it
    carries). A text, categorical or boolean measure is not a count of anything, and calling one that
    sends an operator looking for a number that was never recorded.

    Args:
        descriptor: The measure's descriptor.

    Returns:
        A noun phrase naming the measure's kind when it declares no better end, else None.
    """
    if descriptor.higher_is_better is not None:
        return None
    match descriptor.data_type:
        case "numeric":
            return "a diagnostic" if descriptor.diagnostic else "a raw count"
        case "text":
            return "a text measure"
        case "categorical":
            return "a categorical measure"
        case "boolean":
            return "a boolean condition"
        case None:
            return "an undescribed measure, whose type was never declared"
        case unreachable:
            assert_never(unreachable)


__all__ = [
    "DEFAULT_PASS_THRESHOLD",
    "Bar",
    "BarProposal",
    "BarRegistrationError",
    "BarRegistry",
    "PassThreshold",
    "contradicts_descriptor",
    "pass_threshold_label",
]
