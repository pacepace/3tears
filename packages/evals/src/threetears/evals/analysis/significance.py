"""Significance: the three-way read of a test, how often it was looked at, and the disclosure that rides with it.

:func:`significance_read` decides whether a significance flag may be stated at all, :func:`format_significance`
spells the read with its effect size, and :func:`cross_subject_disclosure`
says what the test did and did not cover.

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
extend is one of the ``L`` looks only because it is counted as one. It measures the bound holding on the bounded test
and failing on the t test (a plan of 2, one planner-sized extend: 5.6%), so keeping a plan is shown only where the
measured rate's upper bound is at most α (:func:`plan_kept`). Past ``L`` looks the bound says nothing. The union
bound is conservative (for three looks, about a third more cases at the last than Pocock's boundaries, and more than
O'Brien–Fleming's); it is chosen because looks here are irregular and their sizes not fixed in advance, which group
sequential boundaries need. A stage-wise betting product, its stakes fixed before each stage, would hold α under any
sizing of the next stage (Ville's inequality), and is the principled exit for data-sized extends.

:func:`looks_taken` counts the looks one contrast has had, from stored facts at read time; what it counts and what it
does not is said there.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Iterable, Mapping
from datetime import datetime
from math import sqrt
from statistics import NormalDist
from types import MappingProxyType
from typing import Final, Literal, NamedTuple

from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
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
    looks where each look's size was fixed before its data (the union bound). Over extends sized by the planner from
    the readings it is not shown: the cost floor's held rows measure just under the tail with upper bounds above it,
    and no held row is measured on the bounded test. A broken or unclear reading still warns at every look at the nominal level: only the claim of safety is
    tightened.

    :param confidence: The nominal two-sided level.
    :ptype confidence: float
    :param planned: The looks the aspect's design declares; None where none is.
    :ptype planned: int | None
    :return: The level the claim is read at: ``confidence`` with no plan or a plan of one.
    :rtype: float
    """
    return 1.0 - (1.0 - confidence) / planned if planned and planned > 1 else confidence


#: The test a reading's claim is read by: ``bounded`` (the bounded test by betting on a declared range: a pass rate, a
#: judged score on its scale, a guardrail) or ``t`` (the t test: a mean, a measure's score, a cost per reply). The same
#: two names a comparison's ``interval_basis`` uses.
SeparationTest = Literal["bounded", "t"]

#: How a contrast's looks were added: in batches whose sizes were fixed before the data (``fixed_batches``), or each
#: after the first sized by a planner from the reading before it (``planner_extends``: a step that adds the cases an
#: unclear reading would need, which is optional stopping on the data).
LooksProcedure = Literal["fixed_batches", "planner_extends"]

#: Every procedure a contrast's looks could have been added by. Which one added them is not stored, so a rate is
#: quoted only where every one of them was measured for the situation.
LOOKS_PROCEDURES: Final[tuple[LooksProcedure, ...]] = ("fixed_batches", "planner_extends")

#: The one-sided level a measured rate's upper confidence bound is read at (:func:`measured_upper_bound`).
UPPER_BOUND_LEVEL: Final = 0.975


class MeasuredRow(NamedTuple):
    """One row of :data:`FALSE_SEPARATION`: how often a test with no true difference separated at some look, read by
    one test, its looks added by one procedure, under one plan, stopped at the first separation, in one reference
    setting.

    :ivar test: The test each look is read by.
    :ivar procedure: How the looks after the first were added.
    :ivar planned: The looks planned, each read at α/L; None where every look is read at α.
    :ivar rates: The rate by the looks taken: ``rates[0]`` after one look, ``rates[k]`` after ``k + 1``. No rate is
        stated past the last: the simulation measured none. Within a row a rate only grows with looks (a replicate
        that separated by one look has separated by every later one).
    :ivar replicates: The replicates each rate was measured on (for a row that is the largest over several settings,
        each setting's).
    :ivar launched: The cases the first look read, in each setting simulated.
    :ivar largest_step: The most cases one later look added in the simulation.
    :ivar setting: What was simulated, in one line.
    """

    test: SeparationTest
    procedure: LooksProcedure
    planned: int | None
    rates: tuple[float, ...]
    replicates: int
    launched: tuple[int, ...]
    largest_step: int
    setting: str


_T_FIXED = "a single paired contrast, a mean on the t test, normal per-case values"
_T_PLANNER = "a single paired contrast, a mean on the t test, normal per-case values, launched at 20 or 40 cases"
_BOUNDED_FIXED = (
    "a single paired contrast, a pass rate of three trials at 0.8 or 0.5 on the bounded test, cases 20 to 80 in "
    "batches of 20"
)
_BOUNDED_PLANNER = (
    "a single paired contrast, a pass rate of three trials at 0.8 or 0.5 on the bounded test, launched at 20 or 40 "
    "cases"
)
_PLANNER = "; each extend the cases the caveat step's planner sizes, at most 100 a step; the larger over the launches"

#: How often a test with no true difference separates at some look, stopped at the first separation, as the seeded
#: simulation in ``tests/test_simulated_looks.py`` measures it (which holds every row here to its measured values,
#: replicates and setting), by name. Each row is one test, one procedure and one plan in one reference setting
#: (:class:`MeasuredRow`); a caveat on looks quotes one of them by name with its setting
#: (:func:`looks_false_separation`) and never a rate no row states. A row is a reference figure, never a campaign's
#: own rate: a campaign launched smaller, extended by more cases a step, or reading several claims at once can err more.
#: The planner-sized rows measure the capped procedure: with no true difference the planner's size is often thousands of
#: cases, so the cap of 100 a step makes most extends +100; a planner sized otherwise, or a cap set by a corpus, is
#: measured anew.
FALSE_SEPARATION: Final[Mapping[str, MeasuredRow]] = MappingProxyType(
    {
        "t, fixed batches, no plan": MeasuredRow(
            "t",
            "fixed_batches",
            None,
            (0.05, 0.097, 0.112, 0.125, 0.129),
            4000,
            (20,),
            40,
            _T_FIXED + ", cases 20 to 60 in equal batches",
        ),
        "t, fixed batches, plan of 2": MeasuredRow(
            "t",
            "fixed_batches",
            2,
            (0.02492, 0.04408, 0.05858, 0.06925, 0.0785, 0.08612),
            40_000,
            (20,),
            20,
            _T_FIXED + ", cases 20 to 120 in batches of 20",
        ),
        "t, fixed batches, plan of 3": MeasuredRow(
            "t",
            "fixed_batches",
            3,
            (0.01525, 0.02675, 0.035, 0.042, 0.04925, 0.05375),
            4000,
            (20,),
            20,
            _T_FIXED + ", cases 20 to 120 in batches of 20",
        ),
        "t, planner-sized, no plan": MeasuredRow(
            "t", "planner_extends", None, (0.056, 0.1008, 0.1304, 0.1513), 10_000, (20, 40), 100, _T_PLANNER + _PLANNER
        ),
        "t, planner-sized, plan of 2": MeasuredRow(
            "t", "planner_extends", 2, (0.0257, 0.0559, 0.0749, 0.087), 10_000, (20, 40), 100, _T_PLANNER + _PLANNER
        ),
        "t, planner-sized, plan of 3": MeasuredRow(
            "t", "planner_extends", 3, (0.0171, 0.0333, 0.0481, 0.0569), 10_000, (20, 40), 100, _T_PLANNER + _PLANNER
        ),
        "bounded, fixed batches, no plan": MeasuredRow(
            "bounded", "fixed_batches", None, (0.0036, 0.0207, 0.0416, 0.0588), 10_000, (20,), 20, _BOUNDED_FIXED
        ),
        "bounded, fixed batches, plan of 2": MeasuredRow(
            "bounded", "fixed_batches", 2, (0.0002, 0.007, 0.0165, 0.0263), 10_000, (20,), 20, _BOUNDED_FIXED
        ),
        "bounded, fixed batches, plan of 3": MeasuredRow(
            "bounded", "fixed_batches", 3, (0.0001, 0.0026, 0.0075, 0.0156), 10_000, (20,), 20, _BOUNDED_FIXED
        ),
        "bounded, planner-sized, no plan": MeasuredRow(
            "bounded",
            "planner_extends",
            None,
            (0.0166, 0.0479, 0.0731, 0.0923),
            10_000,
            (20, 40),
            100,
            _BOUNDED_PLANNER + _PLANNER,
        ),
        "bounded, planner-sized, plan of 2": MeasuredRow(
            "bounded",
            "planner_extends",
            2,
            (0.0059, 0.0203, 0.0348, 0.0448),
            10_000,
            (20, 40),
            100,
            _BOUNDED_PLANNER + _PLANNER,
        ),
        "bounded, planner-sized, plan of 3": MeasuredRow(
            "bounded",
            "planner_extends",
            3,
            (0.0026, 0.0125, 0.0212, 0.0288),
            10_000,
            (20, 40),
            100,
            _BOUNDED_PLANNER + _PLANNER,
        ),
    }
)


def measured_upper_bound(rate: float, replicates: int) -> float:
    """The one-sided upper confidence bound, at :data:`UPPER_BOUND_LEVEL`, on a rate a simulation measured as ``rate``
    over ``replicates`` replicates: the Wilson score bound, which stays above zero for a rate measured at zero. It is the
    simulation's own uncertainty about its reference rate, never a bound on a campaign's error. A claim that a reference
    rate is at or under a level is shown only where this is.

    :param rate: The measured rate, as a share.
    :ptype rate: float
    :param replicates: The replicates it was measured on.
    :ptype replicates: int
    :return: The upper bound, as a share.
    :rtype: float
    """
    z = NormalDist().inv_cdf(UPPER_BOUND_LEVEL)
    n = replicates
    centre = rate + z * z / (2 * n)
    spread = z * sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
    return min(1.0, (centre + spread) / (1 + z * z / n))


#: Why a caveat on looks quotes no reference rate for a test: no row measured that test, plan and count
#: (``not_measured``), or the campaign's own launch or extend sizes, from stored facts, fall outside the setting the rows
#: simulated (``outside_setting``).
Unquoted = Literal["not_measured", "outside_setting"]


class MeasuredRate(NamedTuple):
    """The reference false-separation rate a caveat on looks quotes for one test (:func:`looks_false_separation`).

    :ivar test: The test it is measured for.
    :ivar rate: The reference rate, as a share; None where none is quoted (``unquoted`` says why).
    :ivar upper: The simulation's uncertainty on it (:func:`measured_upper_bound`); None with no rate.
    :ivar row: The :data:`FALSE_SEPARATION` row it is read from; None with no rate.
    :ivar looks: The looks the rate is for.
    :ivar setting: The reference setting, in one line; None where no row measured the situation.
    :ivar launched: The first-look cases the rows simulated; empty where none did.
    :ivar largest_step: The most cases one later look added in the rows; None where none did.
    :ivar unquoted: Why no rate is quoted; None with one.
    """

    test: SeparationTest
    rate: float | None
    upper: float | None
    row: str | None
    looks: int
    setting: str | None
    launched: tuple[int, ...]
    largest_step: int | None
    unquoted: Unquoted | None


def _rows(test: SeparationTest, planned: int | None) -> dict[LooksProcedure, tuple[str, MeasuredRow]]:
    """The measured row of each procedure for ``test`` under a plan of ``planned``."""
    return {
        row.procedure: (name, row)
        for name, row in FALSE_SEPARATION.items()
        if row.test == test and row.planned == planned
    }


def looks_false_separation(
    test: SeparationTest,
    planned: int | None,
    taken: int,
    *,
    launched: int | None = None,
    largest_step: int | None = None,
) -> MeasuredRate:
    """The reference false-separation rate for a contrast read by ``test``, looked at ``taken`` times under a plan of
    ``planned``, in the setting the rows simulated: what a caveat on looks quotes, with that setting.

    It is a reference figure, never the campaign's own rate, which is not known: a campaign launched smaller, extended
    by more cases a step, or reading several claims at once can err more, and for a claim of held (one tail) it is the
    two-sided separation rate. How the looks after the first were added (in batches fixed in advance, or sized by the
    planner from the reading before) is not stored, so the rate quoted is the larger of the two procedures' rows for
    this test, plan and count, named by its row. A count that is only a lower count is no situation a row measured
    (more looks may have been taken, and each row is the largest over its settings, so no row is a floor): the caller
    quotes nothing for it.

    No rate is quoted where either procedure has no row for this test and plan or measured fewer looks
    (``not_measured``), or where the campaign's own first-look cases (``launched``) or largest step
    (``largest_step``), known from stored facts, fall outside the setting the two rows simulated (``outside_setting``).

    :param test: The test the contrast's claims are read by.
    :ptype test: SeparationTest
    :param planned: The looks the plan in force allows; None where none is.
    :ptype planned: int | None
    :param taken: The looks taken, at least one.
    :ptype taken: int
    :param launched: The cases the campaign's contrast read at its first look; None where not known.
    :ptype launched: int | None
    :param largest_step: The most cases one of its later looks added; None where not known.
    :ptype largest_step: int | None
    :return: The rate with its row and setting, or why none is quoted.
    :rtype: MeasuredRate
    """
    rows = _rows(test, planned)
    if set(rows) != set(LOOKS_PROCEDURES) or any(taken > len(row.rates) for _, row in rows.values()):
        return MeasuredRate(test, None, None, None, taken, None, (), None, "not_measured")
    name, row = max(rows.values(), key=lambda named: named[1].rates[taken - 1])
    simulated = tuple(sorted({cases for _, each in rows.values() for cases in each.launched}))
    step = max(each.largest_step for _, each in rows.values())
    outside = (launched is not None and not simulated[0] <= launched <= simulated[-1]) or (
        largest_step is not None and largest_step > step
    )
    if outside:
        return MeasuredRate(test, None, None, None, taken, row.setting, simulated, step, "outside_setting")
    rate = row.rates[taken - 1]
    return MeasuredRate(
        test,
        rate,
        measured_upper_bound(rate, row.replicates),
        name,
        taken,
        row.setting,
        simulated,
        step,
        None,
    )


def plan_kept(test: SeparationTest, planned: int, taken: int) -> bool:
    """Whether a contrast read by ``test``, looked at ``taken`` times within a plan of ``planned``, is shown to err at
    most α over them.

    For batches fixed in advance the union bound shows it (the module). For extends sized by the planner from the
    reading before, no theorem does: it is shown only where the row measured for that procedure has an upper bound
    (:func:`measured_upper_bound`) at most α at that count. Which procedure added the looks is not stored, so both must
    be shown; a plan, a test or a count no planner-sized row measured is not.

    :param test: The test the contrast's claims are read by.
    :ptype test: SeparationTest
    :param planned: The looks the plan in force allows.
    :ptype planned: int
    :param taken: The looks taken, at most ``planned``.
    :ptype taken: int
    :return: Whether keeping the plan is shown to hold α.
    :rtype: bool
    """
    if taken <= 1:
        return True
    extends = _rows(test, planned).get("planner_extends")
    if extends is None or taken > len(extends[1].rates):
        return False
    _, row = extends
    return measured_upper_bound(row.rates[taken - 1], row.replicates) <= SIGNIFICANCE_ALPHA


def looks_taken(
    deliveries: Iterable[tuple[datetime, str, str, Hashable]],
    *,
    arms: tuple[str, str],
    changes: Iterable[tuple[datetime, Hashable]] = (),
    departed: Iterable[tuple[datetime, Hashable]] = (),
    rejudges: Iterable[Iterable[datetime]] = (),
    uncounted: Collection[Hashable] = (),
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
      leaving, a re-measure) and ``departed`` (when a departed run's evidence first reached the contrast) are occasions that change every contrast.
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
    :param departed: When the evidence of each run that has since left first reached the contrast, each with the occasion that brought it.
    :ptype departed: Iterable[tuple[datetime, Hashable]]
    :param rejudges: Each result's re-judges, as when each was made.
    :ptype rejudges: Iterable[Iterable[datetime]]
    :param uncounted: Occasions that changed the evidence but whose being a look is not recorded (a run that joined
        before membership was recorded, launched as it was stored): they date the first look where they made it, and
        are not counted after it, so the count is the looks recorded since, a lower count.
    :ptype uncounted: Collection[Hashable]
    :return: ``(looks, lower_bound)``; None where the two arms never both held a result, so there is no contrast.
    :rtype: tuple[int, bool] | None
    """
    walked = _walk(deliveries, arms)
    if walked is None:
        return None
    first, occasions, _ = walked
    reached = list(departed)
    first = min([first, *(at for at, _ in reached)])
    for at, occasion in [*reached, *changes]:
        occasions[occasion] = min(at, occasions.get(occasion, at))
    later = sum(1 for occasion, at in occasions.items() if at > first and occasion not in uncounted)
    passes = max((sum(1 for at in each if at > first) for each in rejudges), default=0)
    return 1 + later + passes, passes > 0


def look_sizes(
    deliveries: Iterable[tuple[datetime, str, str, Hashable]], *, arms: tuple[str, str]
) -> tuple[int, int] | None:
    """How many cases one contrast's test read at its first look, and the most cases one later occasion added: what a
    reference rate's setting is checked against (:func:`looks_false_separation`).

    A contrast's cases are those both arms hold where they share two or more (the test pairs over them), else the fewer
    one arm holds. Each occasion's cases are counted once all its results are in, occasions taken in the order of their
    first change; one that adds trials to held cases adds no cases.

    :param deliveries: Each result of the two arms, as :func:`looks_taken` takes them.
    :ptype deliveries: Iterable[tuple[datetime, str, str, Hashable]]
    :param arms: The contrast's two arms.
    :ptype arms: tuple[str, str]
    :return: ``(cases at the first look, largest step)``; None where there is no contrast.
    :rtype: tuple[int, int] | None
    """
    walked = _walk(deliveries, arms)
    if walked is None:
        return None
    _, occasions, cases = walked
    order = sorted(occasions, key=lambda occasion: occasions[occasion])
    launch = order[0]
    reached, step = cases[launch], 0
    for occasion in order[1:]:
        step = max(step, cases[occasion] - reached)
        reached = max(reached, cases[occasion])
    return cases[launch], step


def _walk(
    deliveries: Iterable[tuple[datetime, str, str, Hashable]], arms: tuple[str, str]
) -> tuple[datetime, dict[Hashable, datetime], dict[Hashable, int]] | None:
    """The first look, each occasion that changed what the contrast's test reads with when it first did, and the cases
    the test read once its results were in; None where the arms never both held a result."""
    held: dict[str, set[str]] = {arm: set() for arm in arms}
    first: datetime | None = None
    occasions: dict[Hashable, datetime] = {}
    cases: dict[Hashable, int] = {}
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
        cases[occasion] = len(shared) if len(shared) >= 2 else min(len(held[arms[0]]), len(held[arms[1]]))
    if first is None:
        return None
    return first, occasions, cases


__all__ = [
    "cross_subject_disclosure",
    "looks_false_separation",
    "format_significance",
    "looks_alpha",
    "looks_confidence",
    "look_sizes",
    "looks_taken",
    "plan_kept",
    "NOT_SIGNIFICANT_LABEL",
    "NOT_TESTED_LABEL",
    "PAIRED_EFFECT_LABEL",
    "PAIRED_HEDGES_LABEL",
    "significance_read",
    "SIGNIFICANT_LABEL",
    "UNPAIRED_EFFECT_LABEL",
    "UNPAIRED_HEDGES_LABEL",
]
