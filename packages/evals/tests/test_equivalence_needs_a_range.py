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
"""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.analysis.bundle import AnalysisContextBundle, FamilyComparison
from threetears.evals.analysis.report import build_code_only_report
from threetears.evals.analysis.report.model import DisclosureBlock
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE
from threetears.evals.contracts.host import HostProfile, MeasureRegistry
from threetears.evals.contracts.metrics import MetricDescriptor
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.guardrail_support import two_arm_bundle

MEASURE = "extraction_score"

#: Cases each arm ran: enough that the bounded test shows two agreeing arms inside the margin.
N = 30


def _profile(*, value_range: tuple[float, float] | None) -> HostProfile:
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
    )
    return replace(profile, measures=MeasureRegistry((*TOYHOST_MEASURES, measure), families=profile.measures.families))


def _bundle(*, value_range: tuple[float, float] | None) -> AnalysisContextBundle:
    """Two arms that agree on every case to within a hundredth."""
    control = [{MEASURE: 0.5 + 0.01 * (case % 3)} for case in range(N)]
    contrast = [{MEASURE: 0.5 + 0.01 * ((case + 1) % 3)} for case in range(N)]
    return two_arm_bundle((), host_measures=(control, contrast), profile=_profile(value_range=value_range))


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
