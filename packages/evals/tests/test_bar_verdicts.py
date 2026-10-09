"""A bar's verdict is decided by the cell's interval against the measure's declared margin, never its mean (#593).

The toy campaign's cells are held to a registered ``field_accuracy`` bar, set here relative to the
higher cell's own interval so each test places the threshold where the rule under test decides
differently from the mean rule it replaced.

Mutations that turn this file red: comparing the cell's mean with the threshold again; folding a
straddling interval into cleared (or missed); reading the wrong end of the interval for either
outcome; ignoring the declared margin; deciding a cell with one observation.
"""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.contracts.analysis_measures import BarVerdict, MeasureSummary
from threetears.evals.contracts.host import MeasureRegistry
from threetears.evals.contracts.host.bars import BarRegistry
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_BARS, TOYHOST_MEASURES, toyhost_profile

ACCURACY = "field_accuracy"


def _bundle(
    *, threshold: float | None = None, margin: float | None = None, observations_per_run: int | None = None
) -> AnalysisContextBundle:
    """The toy campaign's bundle under a ``field_accuracy`` bar at ``threshold`` and a declared ``margin``.

    Args:
        threshold: Where the registered bar sits, or None for the toy host's own.
        margin: The measure's declared materiality threshold, or None for none.
        observations_per_run: Cut each run to its first results, or None to keep them all.
    """
    campaign, storage = toyhost_campaign()
    profile = toyhost_profile()
    if margin is not None:
        measures = tuple(
            descriptor.model_copy(update={"materiality_threshold": margin})
            if descriptor.name == ACCURACY
            else descriptor
            for descriptor in TOYHOST_MEASURES
        )
        profile = replace(profile, measures=MeasureRegistry(measures, families=profile.measures.families))
    if threshold is not None:
        bars = tuple(replace(bar, threshold=threshold) if bar.measure == ACCURACY else bar for bar in TOYHOST_BARS)
        profile = replace(profile, bars=BarRegistry(bars))
    if observations_per_run is not None:
        runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
        results = {
            run.id: storage.query_eval_results_by_run(run.id, campaign.scope_id)[:observations_per_run] for run in runs
        }
        storage = ToyhostStorage(runs, results)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _higher_cell() -> MeasureSummary:
    """The ``field_accuracy`` summary of the cell measuring the higher mean, with its interval."""
    summaries = [
        summary for cell in _bundle().cell_measures for summary in cell.measures.measures if summary.name == ACCURACY
    ]
    best = max(summaries, key=lambda summary: summary.mean or 0.0)
    assert best.mean is not None and best.ci_low is not None and best.ci_high is not None
    return best


def _verdict_on(bundle: AnalysisContextBundle, mean: float) -> BarVerdict:
    (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == ACCURACY]
    assert bar.direction == "higher_is_better" and bar.state == "adjudicated"
    (verdict,) = [verdict for verdict in bar.verdicts if verdict.value == mean]
    return verdict


class TestTheIntervalDecides:
    """Three outcomes, by where the cell's whole interval sits against the line."""

    def test_an_interval_straddling_the_bar_is_undecided_not_cleared(self) -> None:
        """The mean under the bar and the interval reaching past it: the data cannot say, and the verdict says so."""
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_high is not None
        threshold = (cell.mean + cell.ci_high) / 2

        verdict = _verdict_on(_bundle(threshold=threshold), cell.mean)

        assert cell.mean < threshold <= cell.ci_high
        assert verdict.cleared is None and verdict.decision == "undecided"
        assert verdict.margin is None, "the host declared no margin on the measure"

    def test_a_cell_whose_whole_interval_is_at_or_above_the_bar_clears(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_low is not None

        verdict = _verdict_on(_bundle(threshold=cell.ci_low), cell.mean)

        assert verdict.cleared is True and verdict.decision == "cleared"

    def test_a_cell_whose_whole_interval_is_under_the_bar_misses(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_high is not None

        verdict = _verdict_on(_bundle(threshold=cell.ci_high + 0.01), cell.mean)

        assert verdict.cleared is False and verdict.decision == "missed"

    def test_the_verdict_carries_the_cells_own_interval_and_serves_its_decision(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None

        verdict = _verdict_on(_bundle(), cell.mean)

        assert (verdict.ci_low, verdict.ci_high) == (cell.ci_low, cell.ci_high)
        assert not verdict.decided_on_the_mean
        assert verdict.model_dump(mode="json")["decision"] == verdict.decision
        assert BarVerdict.model_validate(verdict.model_dump(mode="json")) == verdict, "the echo is re-derived"


class TestTheDeclaredMargin:
    """The margin moves the line toward the bad side by the shortfall too small to act on."""

    def test_an_interval_wholly_inside_the_margin_clears(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_low is not None

        verdict = _verdict_on(_bundle(threshold=cell.ci_low + 0.05, margin=0.06), cell.mean)

        assert verdict.margin == 0.06
        assert verdict.cleared is True

    def test_without_the_margin_the_same_bar_is_not_cleared(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_low is not None

        verdict = _verdict_on(_bundle(threshold=cell.ci_low + 0.05, margin=0.04), cell.mean)

        assert verdict.cleared is not True

    def test_a_shortfall_past_the_margin_still_misses(self) -> None:
        cell = _higher_cell()
        assert cell.mean is not None and cell.ci_high is not None

        verdict = _verdict_on(_bundle(threshold=cell.ci_high + 0.05, margin=0.04), cell.mean)

        assert verdict.cleared is False


class TestTooFewObservations:
    def test_one_observation_has_a_value_and_no_verdict(self) -> None:
        """A single value against a threshold would be the mean comparison again, so nothing is decided."""
        (bar,) = [bar for bar in _bundle(observations_per_run=1).bar_adjudications if bar.measure_id == ACCURACY]

        assert bar.verdicts
        for verdict in bar.verdicts:
            assert verdict.n == 1 and verdict.value is not None
            assert (verdict.ci_low, verdict.ci_high, verdict.cleared) == (None, None, None)
            assert verdict.decision == "no_interval"
            assert not verdict.decided_on_the_mean
