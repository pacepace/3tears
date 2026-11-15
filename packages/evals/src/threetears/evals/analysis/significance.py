"""Significance: the three-way read of a test, how often it was looked at, and the disclosure that rides with it.

:func:`significance_read` decides whether a significance flag may be stated at all, :func:`format_significance`
spells the read with its effect size, and :func:`significance_disclosure` / :func:`cross_subject_disclosure`
say what the test did and did not cover.

**Looks.** A look is an occasion on which a contrast's test could have been read again on more or different
evidence: one more chance for a test with no true difference behind it to separate, the same thing a family's size
counts. Reading a test at α after every batch of new cases, and stopping at the first separation, errs far more often
than α (``tests/test_simulated_looks.py`` measures how much), so a design may declare how many looks it plans
(``AspectDesign.planned_looks``), before the first: each of its ``L`` looks is then read at α/L (:func:`looks_alpha`),
and a claim that a drop stays within a margin at ``1 − (1 − level)/L`` (:func:`looks_confidence`).

**What that guarantees, and on what condition.** α/L at each look is a union bound: the chance any of ``L`` readings
errs is at most the sum of their chances, whatever the dependence among them, PROVIDED each reading errs at most α/L.
For looks at sizes fixed before the data, each reading is a fixed-size test and does. Where the next batch is sized
from the readings so far (a step that adds the cases an unclear reading would need), the reading at the data-chosen
size is no fixed-size test, and nothing in the bound says it still errs at most α/L: what holds there is what the
simulation of that procedure measures (:data:`FALSE_SEPARATION`, ``planner_extends``), and an
extend is one of the ``L`` looks only because it is counted as one. Past ``L`` looks the bound says nothing. The union
bound is conservative (for three looks, about a third more cases at the last than Pocock's boundaries, and more than
O'Brien–Fleming's); it is chosen because looks here are irregular and their sizes not fixed in advance, which group
sequential boundaries need. A stage-wise betting product, its stakes fixed before each stage, would hold α under any
sizing of the next stage (Ville's inequality), and is the principled exit for data-sized extends.

:func:`looks_taken` counts the looks one contrast has had, from stored facts at read time; what it counts and what it
does not is said there.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, NamedTuple

from threetears.evals.analysis.stats import PAIRED_TEST_NAME, SIGNIFICANCE_ALPHA, UNPAIRED_TEST_NAME
from threetears.evals.kernel.numbers import format_number


# The reader-facing words for each read. Spelled out, never abbreviated: "n.s."
# was once read as nanoseconds by an operator — a fair guess in a report whose
# metrics are mostly named ``*_ms``, and the tell that an abbreviation a reader
# has to decode is not communication.
SIGNIFICANT_LABEL = "significant"
NOT_SIGNIFICANT_LABEL = "not significant"
NOT_TESTED_LABEL = "not tested"


# How a paired and an unpaired effect size are each named to a reader. They are
# not the same statistic: the paired one standardises the mean of the per-case
# DIFFERENCES by their own SD (Cohen's d_z), the unpaired one standardises the
# difference of means by the pooled SD (Cohen's d). Printing the paired name over
# the unpaired number claims a within-case comparison that never happened, which
# is the more persuasive of the two possible errors.
PAIRED_EFFECT_LABEL = "d_z"
UNPAIRED_EFFECT_LABEL = "d"
# The same two, bias-corrected (Hedges' g): what the engine's own tests report since the change from
# Cohen's d — ``compare_two_runs``' ``hedges_g`` and a regression flag's. A different number from d at the
# sample sizes an eval runs (0.5 against d's 0.88 at three pairs), so it is never printed under d's name.
PAIRED_HEDGES_LABEL = "g_z"
UNPAIRED_HEDGES_LABEL = "g"


def significance_read(*, significant: bool | None, p: float | None = None, effect: float | None = None) -> str:
    """Which of three things a significance flag may honestly be rendered as.

    **"Not significant" and "not tested" are different facts.** A flag arriving
    with no statistic behind it says only that nobody computed one; rendering it
    as a negative publishes a measured null the campaign never measured, and
    rendering it as a positive is the identical defect in the more persuasive
    direction. So a verdict is reported only when the statistic it came from
    travels with it.

    The predicate for "a test ran" is ``p is not None or effect is not None``,
    and a browser renderer must use the identical predicate so the two surfaces cannot answer
    differently about the same row. **A sample size is not a test:** ``n`` says
    how much data there was, never whether a difference cleared a threshold, so
    it is not part of the predicate and is not a parameter here.

    :param significant: The verdict, or ``None`` when no test was run.
    :ptype significant: bool | None
    :param p: The p-value the verdict was thresholded against, when one exists.
    :ptype p: float | None
    :param effect: The effect size, when one exists — paired or not, since which
        test produced it changes what it is CALLED but not whether one ran.
    :ptype effect: float | None
    :return: :data:`SIGNIFICANT_LABEL`, :data:`NOT_SIGNIFICANT_LABEL`, or
        :data:`NOT_TESTED_LABEL`.
    :rtype: str
    """
    if p is None and effect is None:
        return NOT_TESTED_LABEL
    if significant is True:
        return SIGNIFICANT_LABEL
    if significant is False:
        return NOT_SIGNIFICANT_LABEL
    return NOT_TESTED_LABEL


def format_significance(
    *,
    significant: bool | None,
    paired: bool,
    p: float | None = None,
    effect: float | None = None,
    n: int | None = None,
    hedges: bool = False,
) -> str:
    """The read plus the statistics behind it, as one cell a surface prints verbatim.

    **The single renderer of this rule.** Every server-side surface that shows a
    significance verdict calls this one: a host's compare table and its history
    table's regression flags. The same branch written by hand
    diverged on both its not-tested predicate and its number formatting before it
    was collapsed here, and the copy that outlived the collapse — the history
    table's — printed a bare "significant" for the one verdict that arrives with
    no p and no effect size at all.

    A browser kit's copy is the one that is deliberate, because nothing in TypeScript
    can call this; ``tests/test_significance_rule.py`` asserts this side as the facts
    that copy is pinned against, and the pin itself lives with the kit.

    The statistics are appended so a reader can check the verdict rather than
    take it — which is the whole difference between a descriptive report and an
    assertion. ``n`` is shown when known even on an untested row: a reader who
    sees ``not tested (n=1)`` learns *why* nothing was tested, where a bare
    "not tested" looks like a fault in the harness.

    :param significant: The verdict, or ``None`` when no test was run.
    :ptype significant: bool | None
    :param paired: Whether the test that produced ``effect`` paired its samples.
        Decides only the effect size's NAME (:data:`PAIRED_EFFECT_LABEL` vs
        :data:`UNPAIRED_EFFECT_LABEL`) — a caller must pass what actually
        ran, not what the surface is called.
    :ptype paired: bool
    :param p: The p-value the verdict was thresholded against.
    :ptype p: float | None
    :param effect: The effect size.
    :ptype effect: float | None
    :param n: The sample size the test would have run over.
    :ptype n: int | None
    :param hedges: Whether ``effect`` is Hedges' g (the engine's tests, ``hedges_g``) rather than Cohen's d (a
        stored delta-table row's historical ``d_z``). Decides the name with ``paired``.
    :ptype hedges: bool
    :return: e.g. ``"significant (p=0.0123, d_z=1.42, n=8)"`` or ``"not tested (n=1)"``.
    :rtype: str
    """
    if hedges:
        effect_label = PAIRED_HEDGES_LABEL if paired else UNPAIRED_HEDGES_LABEL
    else:
        effect_label = PAIRED_EFFECT_LABEL if paired else UNPAIRED_EFFECT_LABEL
    # The one number rule, not a fixed spelling of their own. A p-value is not
    # read against a column of its peers the way pass^k is — it is checked
    # against one threshold (α=0.05 vs p=0.04998), which the rule's four
    # significant figures already allow, and its small end is the one that
    # matters: a p of 3e-7 must not round to a zero. The effect size is a
    # measured magnitude with no bound, so its large end needs the rule too.
    parts = [
        part
        for part in (
            f"p={format_number(p)}" if p is not None else "",
            f"{effect_label}={format_number(effect)}" if effect is not None else "",
            f"n={n}" if n is not None else "",
        )
        if part
    ]
    read = significance_read(significant=significant, p=p, effect=effect)
    return f"{read} ({', '.join(parts)})" if parts else read


def significance_disclosure(*, paired: bool) -> str:
    """The sentence naming the test and threshold a comparison's verdicts rest on.

    Rendered verbatim by every surface that shows a significance verdict, for
    the reason :func:`completeness_disclosure` is: a reader must be able to see
    what was measured without reconstructing it from the code that measured it.
    Which test ran is not a detail — a paired test over shared cases and an
    unpaired test over different ones answer different questions, and most of
    what a small eval arm's verdict rests on is that difference.

    It closes by saying the output is descriptive, because until the judge is
    calibrated against human labels a flag here is not a trustworthy quality
    signal, and nothing routes it anywhere.

    :param paired: Whether the comparison paired its samples by test case.
    :ptype paired: bool
    :return: The disclosure sentence.
    :rtype: str
    """
    test_name = PAIRED_TEST_NAME if paired else UNPAIRED_TEST_NAME
    return f"Significance: {test_name}. Descriptive only — no alerting, and no verdict without the statistic behind it."


def cross_subject_disclosure(subject_a: str | None, subject_b: str | None) -> str | None:
    """Say when a two-run comparison's composites belong to different subjects.

    Composite quality is comparable *within* a subject and never across one:
    rubric dimensions are derived from each subject's own self-description and
    tools, so two subjects' 0.8s are different measurements wearing the same
    number, and their difference is not a quantity.

    :func:`~threetears.evals.analysis.reads.compare_two_runs` calls this and,
    when it returns a sentence, withholds the composite delta AND the
    significance test on it rather than emitting numbers nothing downstream could
    falsify — in the service, so REST and MCP inherit one answer instead of each
    deciding. A surface that re-derived the rule from subject ids of its own
    would be the second implementation this exists to prevent.

    There is no undecidable case. Two runs whose subject nobody recorded would not be evidence of
    sameness, and comparing two blanks for equality would manufacture exactly the claim this refuses,
    but a subject key cannot be blank, so the pair is not constructible.

    :param subject_a: Run A's subject key.
    :ptype subject_a: str | None
    :param subject_b: Run B's subject key.
    :ptype subject_b: str | None
    :return: The disclosure, or ``None`` when both sides name the same subject and
        the composites are therefore comparable.
    :rtype: str | None
    """
    left = (subject_a or "").strip()
    right = (subject_b or "").strip()
    if left == right:
        return None
    return (
        f"Cross-subject comparison: run A scored subject `{left}` and run B scored subject `{right}`. "
        "Rubric dimensions are derived from each subject's own self-description, so these composites "
        "are different measurements wearing one name — the composite delta is withheld, and the "
        "per-run composites must be read separately."
    )


def looks_alpha(planned: int | None, alpha: float = SIGNIFICANCE_ALPHA) -> float:
    """The level each look of a test is read at: α/L under a plan of ``L`` looks, α with none.

    A union bound: if each of ``L`` readings errs at most α/L, the chance any of them does is at most α, whatever the
    dependence among them. Each does for looks at sizes fixed before the data; for a look at a size chosen from the
    readings before it, what holds is what the simulation of that procedure measures (see the module). With no plan
    the first look is read at α, as every test always was; how many more were taken is counted
    (:func:`looks_taken`) and qualifies the reading rather than tightening it after the fact.

    :param planned: The looks the aspect's design declares (``AspectDesign.planned_looks``); None where none is.
    :ptype planned: int | None
    :param alpha: The error rate the readings state.
    :ptype alpha: float
    :return: The level each look reads at.
    :rtype: float
    """
    return alpha / planned if planned else alpha


def looks_confidence(confidence: float, planned: int | None) -> float:
    """The two-sided level a claim of no worse than a margin is read at under a plan of ``planned`` looks.

    ``held`` (a floor) and ``within_margin`` (a campaign's row) claim the drop stays within the margin, read off one
    end of a two-sided interval; at ``1 − (1 − confidence)/L`` that end errs at most its nominal tail over all ``L``
    looks. A broken or unclear reading still warns at every look at the nominal level: only the claim of safety is
    tightened.

    :param confidence: The nominal two-sided level.
    :ptype confidence: float
    :param planned: The looks the aspect's design declares; None where none is.
    :ptype planned: int | None
    :return: The level the claim is read at: ``confidence`` with no plan or a plan of one.
    :rtype: float
    """
    return 1.0 - (1.0 - confidence) / planned if planned and planned > 1 else confidence


#: How a contrast's looks were added: in batches whose sizes were fixed before the data (``fixed_batches``), or each
#: after the first sized by a planner from the reading before it (``planner_extends``: a step that adds the cases an
#: unclear reading would need, which is optional stopping on the data).
LooksProcedure = Literal["fixed_batches", "planner_extends"]

#: How often a test with no true difference separates at some look, stopped at the first separation, as the seeded
#: simulation in ``tests/test_simulated_looks.py`` measures it (which holds this table to its rows): by procedure, then
#: by the looks planned (None: every look read at α) and the looks taken. ``fixed_batches``: the mean of 20 to 60 cases
#: on the t test in equal batches with no plan, and under a plan of 2 or 3 the mean of 20 to 120 cases in batches of 20
#: run past it. ``planner_extends``: a per-case pass rate of three trials, launched at 20 or 40 cases, each extend the
#: cases the caveat step's planner sizes (at most 100), the largest rate over the settings measured. Past a row's last
#: count the rate only grows.
FALSE_SEPARATION: Final[Mapping[LooksProcedure, Mapping[tuple[int | None, int], float]]] = MappingProxyType(
    {
        "fixed_batches": MappingProxyType({}),
        "planner_extends": MappingProxyType({}),
    }
)


class MeasuredRate(NamedTuple):
    """The false-separation rate a caveat on looks quotes (:func:`looks_false_separation`).

    :ivar rate: The rate, as a share.
    :ivar bound: ``measured`` where a row states it; ``at_least`` past the last count measured (the rate only grows
        with looks); ``at_most`` for a plan no row measured, read as no plan (a plan only lowers it).
    """

    rate: float
    bound: Literal["measured", "at_least", "at_most"]


def looks_false_separation(planned: int | None, taken: int) -> MeasuredRate:
    """The false-separation rate measured for a contrast looked at ``taken`` times under a plan of ``planned``: the
    larger of the two procedures' (:data:`FALSE_SEPARATION`), since which one added the looks is not stored.

    :param planned: The looks the plan in force allows; None where none is.
    :ptype planned: int | None
    :param taken: The looks taken, at least one.
    :ptype taken: int
    :return: The rate, and whether it is measured, a floor or a ceiling.
    :rtype: MeasuredRate
    """
    bound: Literal["measured", "at_least", "at_most"] = "measured"
    rows = [table for table in FALSE_SEPARATION.values() if any(key[0] == planned for key in table)]
    if not rows:
        # A plan no row measured: reading at α/L only lowers each look's chance, so the unplanned rate is a ceiling.
        planned, bound = None, "at_most"
        rows = [table for table in FALSE_SEPARATION.values() if any(key[0] is None for key in table)]
    found: list[float] = []
    for table in rows:
        counts = [key[1] for key in table if key[0] == planned]
        last = max(counts)
        if taken > last and bound == "measured":
            bound = "at_least"
        found.append(table[(planned, min(taken, last))])
    return MeasuredRate(max(found), bound)


def looks_taken(
    deliveries: Iterable[tuple[datetime, str, str, Hashable]],
    *,
    arms: tuple[str, str],
    changes: Iterable[tuple[datetime, Hashable]] = (),
    departed: Iterable[tuple[datetime, Hashable]] = (),
    rejudges: Iterable[Iterable[datetime]] = (),
) -> tuple[int, bool] | None:
    """How many looks one contrast has had, counted from what is stored, and whether re-judges made it a lower bound.

    **A look is an occasion on which the contrast's test could have been read again on more or different evidence**:
    one more chance for a test with no true difference behind it to separate. Each occasion is the act that changed the
    evidence, counted once however many results it brought: a launch (its launch group, or a run launched alone), a
    membership write (runs joining after their results were scored, runs leaving), a re-measure, a re-judge pass.

    - **The first look** is the first moment both arms held a result. Where a run that has since left held results
      earlier, the first look is dated by it, since which arm it read is not stored.
    - **A result changes what the test reads** where it is on a case the other arm also holds, once the arms share
      two cases (the test then pairs over the shared cases, :func:`~threetears.evals.analysis.stats.contrast_samples`),
      and wherever it lands while they share fewer (the test then reads every case unpaired). So more cases run on one
      arm and then the other are one look, at the launch that paired them: the first changed nothing the test reads.
    - **Each occasion after the first look is one more**, dated by the first change it made; ``changes`` (a run
      leaving, a re-measure) and ``departed`` (a departed run's own arrival) are occasions that change every contrast.
    - **Re-judges** are stored per result, with no pass they belong to: the most any one result had after the first
      look is the fewest passes that could have made them, so the count is then a lower bound.

    What is not counted, and why: a judge repeat and a second judge's pass measure the judge and change no value the
    test reads, so they give it no new chance to separate; a measure attestation states the definitions a run was
    launched under and touches no result. A reading by a person or an agent that changed nothing is no new chance
    either: the same evidence read twice separates or not alike.

    :param deliveries: Each result of the two arms, as when its evidence reached the contrast (scored, or joined where
        it joined later), its arm, its case, and the occasion that brought it.
    :ptype deliveries: Iterable[tuple[datetime, str, str, Hashable]]
    :param arms: The contrast's two arms.
    :ptype arms: tuple[str, str]
    :param changes: Occasions that changed the evidence of every contrast, each with when it was made.
    :ptype changes: Iterable[tuple[datetime, Hashable]]
    :param departed: The arrivals of runs that have since left, each with the occasion that brought it.
    :ptype departed: Iterable[tuple[datetime, Hashable]]
    :param rejudges: Each result's re-judges, as when each was made.
    :ptype rejudges: Iterable[Iterable[datetime]]
    :return: ``(looks, lower_bound)``; None where the two arms never both held a result, so there is no contrast.
    :rtype: tuple[int, bool] | None
    """
    held: dict[str, set[str]] = {arm: set() for arm in arms}
    first: datetime | None = None
    occasions: dict[Hashable, datetime] = {}
    for at, arm, case, occasion in sorted(deliveries, key=lambda delivery: delivery[0]):
        held[arm].add(case)
        if not all(held.values()):
            continue
        shared = held[arms[0]] & held[arms[1]]
        if len(shared) >= 2 and case not in shared:
            continue
        if first is None:
            first = at
        occasions.setdefault(occasion, at)
    if first is None:
        return None
    arrivals = list(departed)
    first = min([first, *(at for at, _ in arrivals)])
    for at, occasion in [*arrivals, *changes]:
        occasions[occasion] = min(at, occasions.get(occasion, at))
    later = sum(1 for at in occasions.values() if at > first)
    passes = max((sum(1 for at in each if at > first) for each in rejudges), default=0)
    return 1 + later + passes, passes > 0


__all__ = [
    "cross_subject_disclosure",
    "FALSE_SEPARATION",
    "LooksProcedure",
    "MeasuredRate",
    "looks_false_separation",
    "format_significance",
    "looks_alpha",
    "looks_confidence",
    "looks_taken",
    "NOT_SIGNIFICANT_LABEL",
    "NOT_TESTED_LABEL",
    "PAIRED_EFFECT_LABEL",
    "PAIRED_HEDGES_LABEL",
    "significance_disclosure",
    "significance_read",
    "SIGNIFICANT_LABEL",
    "UNPAIRED_EFFECT_LABEL",
    "UNPAIRED_HEDGES_LABEL",
]
