"""Significance: the three-way read of a test, and the disclosure that rides with it.

:func:`significance_read` decides whether a significance flag may be stated at all, :func:`format_significance`
spells the read with its effect size, and :func:`significance_disclosure` / :func:`cross_subject_disclosure`
say what the test did and did not cover.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from threetears.evals.analysis.numbers import format_number

if TYPE_CHECKING:
    pass


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

    Args:
        significant: The verdict, or ``None`` when no test was run.
        p: The p-value the verdict was thresholded against, when one exists.
        effect: The effect size, when one exists — paired or not, since which
            test produced it changes what it is CALLED but not whether one ran.

    Returns:
        :data:`SIGNIFICANT_LABEL`, :data:`NOT_SIGNIFICANT_LABEL`, or
        :data:`NOT_TESTED_LABEL`.
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
    significance verdict calls this one: a host's compare table, its history
    table's regression flags, and a delta-table chart's values table
    (:mod:`threetears.evals.analysis.viz.intents.delta_table`). The same branch written by hand
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

    Args:
        significant: The verdict, or ``None`` when no test was run.
        paired: Whether the test that produced ``effect`` paired its samples.
            Decides only the effect size's NAME (:data:`PAIRED_EFFECT_LABEL` vs
            :data:`UNPAIRED_EFFECT_LABEL`) — a caller must pass what actually
            ran, not what the surface is called.
        p: The p-value the verdict was thresholded against.
        effect: The effect size.
        n: The sample size the test would have run over.
        hedges: Whether ``effect`` is Hedges' g (the engine's tests, ``hedges_g``) rather than Cohen's d (a
            stored delta-table row's historical ``d_z``). Decides the name with ``paired``.

    Returns:
        e.g. ``"significant (p=0.0123, d_z=1.42, n=8)"`` or ``"not tested (n=1)"``.
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

    Args:
        paired: Whether the comparison paired its samples by test case.

    Returns:
        The disclosure sentence.
    """
    from threetears.evals.analysis.stats import PAIRED_TEST_NAME, UNPAIRED_TEST_NAME

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

    There is no undecidable case. This used to carry one — two runs whose subject nobody recorded
    are not evidence of sameness, and comparing two blanks for equality would manufacture exactly
    the claim this refuses — but a subject key can no longer be blank, so the pair is not
    constructible and the branch would answer about nothing.

    Args:
        subject_a: Run A's subject key.
        subject_b: Run B's subject key.

    Returns:
        The disclosure, or ``None`` when both sides name the same subject and
        the composites are therefore comparable.
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


__all__ = [
    "cross_subject_disclosure",
    "format_significance",
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
