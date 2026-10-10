"""A host measure with a margin and no declared range is never read ``equivalent`` (#695).

``equivalent`` is the one verdict that says two arms are alike. On a measure that declares ``value_range`` the
equivalence test is a bounded test by betting, which holds α at every n; with no range no test of a mean does,
and the paired t TOST the engine fell back on claimed ``equivalent`` 11-13% of the time against 5% on skewed
coarse values. No engine measure declares a margin, so this binds the measures a host declares. Pinned here
through the bundle a writer reads and the report a reader reads:

- **No range, no equivalence test**: arms that agree closely on every case read ``not_separated``, never
  ``equivalent``; the comparison carries no margin and no TOST p, and its ``equivalence_untested_reason`` names
  the remedy, declare value_range.
- **The same measure on a declared range is tested**, and the same arms read ``equivalent``.
- **The code-only report names the measure once**, with the remedy.
- **A history step carries the same refusal** on its ``RegressionFlag``, and the history text states it.
- **A floor is not a range**: a measure declared ``nonnegative`` is bounded at one end only, so it is refused
  the same way.
"""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.report import build_code_only_report
from threetears.evals.analysis.report.model import DisclosureBlock
from threetears.evals.analysis.reporting import METRIC_COST_USD
from threetears.evals.analysis.lenses.history import HistoryResult, RegressionFlag, compute_history
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, paired_change
from threetears.evals.kernel.host import HostProfile, MeasureRegistry
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.ops import history_text
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.guardrail_support import two_arm_bundle

MEASURE = "extraction_score"

#: Cases each arm ran: enough that the bounded test shows two agreeing arms inside the margin.
N = 30


def _profile(*, value_range: tuple[float, float] | None, nonnegative: bool = False) -> HostProfile:
    """The toy host with a quality measure declaring a margin of 0.3, and the range given."""
    profile = toyhost_profile()
    family = next(d.family for d in TOYHOST_MEASURES if d.name == "field_accuracy")
    measure = MetricDescriptor(
        name=MEASURE,
        reader_name="Extraction score",
        data_type="numeric",
        family=family,
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="How much of the document the candidate extracted, as a share.",
        higher_is_better=True,
        merit_axis="quality",
        materiality_threshold=0.3,
        value_range=value_range,
        nonnegative=nonnegative,
    )
    return replace(profile, measures=MeasureRegistry((*TOYHOST_MEASURES, measure), families=profile.measures.families))


def _bundle(*, value_range: tuple[float, float] | None, nonnegative: bool = False) -> AnalysisContextBundle:
    """Two arms that agree on every case to within a hundredth."""
    control = [{MEASURE: 0.5 + 0.01 * (case % 3)} for case in range(N)]
    contrast = [{MEASURE: 0.5 + 0.01 * ((case + 1) % 3)} for case in range(N)]
    profile = _profile(value_range=value_range, nonnegative=nonnegative)
    return two_arm_bundle((), host_measures=(control, contrast), profile=profile)


def _comparison(bundle: AnalysisContextBundle) -> FamilyComparison:
    (comparison,) = [c for f in bundle.multiple_comparisons.families for c in f.comparisons if c.name == MEASURE]
    return comparison


def test_with_no_declared_range_close_arms_are_not_equivalent_and_the_reason_names_the_remedy() -> None:
    comparison = _comparison(_bundle(value_range=None))

    assert comparison.test == "paired" and comparison.verdict == "not_separated"
    assert comparison.equivalence_margin is None and comparison.equivalence_p_raw is None
    assert comparison.equivalence_p_adjusted is None
    assert comparison.equivalence_untested_reason == EQUIVALENCE_NEEDS_RANGE
    assert comparison.equivalence_untested_reason.startswith("declare value_range")


def test_a_nonnegative_measure_has_a_floor_and_no_range_so_it_is_refused_the_same_way() -> None:
    """``nonnegative`` bounds one end; the bounded test needs both, so it does not switch the test on."""
    comparison = _comparison(_bundle(value_range=None, nonnegative=True))

    assert comparison.verdict == "not_separated"
    assert comparison.equivalence_p_raw is None
    assert comparison.equivalence_untested_reason == EQUIVALENCE_NEEDS_RANGE


def test_on_a_declared_range_the_same_arms_are_tested_and_read_equivalent() -> None:
    comparison = _comparison(_bundle(value_range=(0.0, 1.0)))

    assert comparison.verdict == "equivalent"
    assert comparison.equivalence_margin == 0.3 and comparison.equivalence_untested_reason is None


def test_the_code_only_report_names_the_measure_once_with_the_remedy() -> None:
    bundle = _bundle(value_range=None)

    report = build_code_only_report(bundle, measures=_profile(value_range=None).measures, assembled_at="2026-10-10")

    said = [b.text for b in report.blocks if isinstance(b, DisclosureBlock) and EQUIVALENCE_NEEDS_RANGE in b.text]
    assert said == [f"Equivalence untested on Extraction score: {EQUIVALENCE_NEEDS_RANGE}."]
    ranged = _bundle(value_range=(0.0, 1.0))
    report = build_code_only_report(ranged, measures=_profile(value_range=(0.0, 1.0)).measures, assembled_at="x")
    assert not [b for b in report.blocks if isinstance(b, DisclosureBlock) and EQUIVALENCE_NEEDS_RANGE in b.text]


def test_the_history_carries_the_refusal_on_each_step_and_its_text_states_it() -> None:
    """A history step reads its change by ``paired_change``, which refuses the same margin; the flag keeps the reason.

    No history measure can carry a host margin today (each is an engine core measure, and a host cannot
    re-declare one), so the text is pinned on a history read for a margin with no range.
    """
    verdict = paired_change(
        [0.5] * 6,
        [0.51] * 6,
        min_absolute_change=0.05,
        min_relative_change=0.1,
        higher_is_better=True,
        equivalence_margin=0.3,
    )
    assert verdict.equivalence_untested_reason == EQUIVALENCE_NEEDS_RANGE
    assert "equivalence_untested_reason" in RegressionFlag.model_fields

    runs = [make_eval_run(test_case_ids=["c1", "c2"], created_at=f"2026-07-0{day}T00:00:00Z") for day in (1, 2)]
    results = [
        make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, test_case_id=case, cost_usd=0.01)
        for run in runs
        for case in ("c1", "c2")
    ]
    out = compute_history(runs, results, metric=METRIC_COST_USD, profile=toyhost_profile(), archived_run_ids=None)
    flag = out.series[0].points[1].regression
    assert flag is not None and flag.equivalence_untested_reason is None, "no margin declared, nothing refused"

    refused = HistoryResult.model_validate(
        {
            **out.model_dump(),
            "measure": out.measure.model_copy(update={"materiality_threshold": 0.3, "value_range": None}),
            "equivalence_margin": 0.3,
        }
    )
    text = history_text(refused)
    assert "equivalence margin: ±0.3 (the measure's declared materiality threshold), but no step can read " in text
    assert EQUIVALENCE_NEEDS_RANGE in text
