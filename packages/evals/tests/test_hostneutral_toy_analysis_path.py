"""The engine's bundle behaviour and what a stored analysis keeps of it, driven over the toy host.

Bundle assembly over a toy bundle is on the toy host's surface, so each engine behaviour there owes a
toy test. Every test below assembles the toy corpus campaign under the toy host's profile, or builds the
analysis a report document over it is linked by.

Pinned here — each class is a property a mutation of the engine turned red:

- ``analysis/bundle/``: each cell's facts over its non-faulted observations, with its run notes;
  the judged half of the surface; an insight's standing; retraction read off the archived flag
  alone; intervals at the one level; the shared remainder predicate and measure description; the
  lever spread's spelling.
- ``kernel/declaration.py``: a refused bar quoted as declared.
- ``analysis/doc_analysis.py``: the surface a stored analysis freezes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.analysis import stats
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.surface import cell_dimension_facts
from threetears.evals.analysis.bundle.insights import insight_standing
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from threetears.evals.kernel.numbers import WRITTEN_FROM, format_number
from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight
from threetears.evals.kernel.declaration import BarOverride, refuse_an_undeclarable_design
from threetears.evals.kernel.metrics import WITHHELD_PARTITION_INCOMPLETE
from threetears.evals.schema.models import LatencyMetrics, RunCompleteness
from packages.evals.tests.fixtures.toyhost.campaign import (
    toyhost_aspect_design,
    TOYHOST_NARROW,
    toyhost_bundle,
    toyhost_campaign,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_JUDGED_DIMENSION,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import cell_at
from packages.evals.tests.writer_host_support import doc_analysis_over
from packages.evals.tests.aspect_support import with_control


def _bundle() -> Any:
    profile = toyhost_profile()
    return toyhost_bundle(profile=profile)


# =============================================================================
# analysis/bundle/
# =============================================================================


def _assemble_with(
    *, results: dict[str, list[Any]] | None = None, runs: list[Any] | None = None, **campaign_kwargs: Any
) -> Any:
    """Assemble the toy campaign over a store whose batches or observations a test has altered."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(**campaign_kwargs)
    stored_runs = (
        runs if runs is not None else [storage.load_eval_run(run_id, TOYHOST_SCOPE) for run_id in campaign.run_ids]
    )
    stored_results = (
        results
        if results is not None
        else {run_id: storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE) for run_id in campaign.run_ids}
    )
    return campaign, assemble_context_bundle(
        campaign,
        storage=ToyhostStorage(stored_runs, stored_results, storage.query_insights(TOYHOST_SCOPE)),
        profile=profile,
    )


def _cell_of_run(bundle: Any, run_id: str) -> Any:
    (cell,) = [cell for cell in bundle.cell_measures if run_id in cell.run_ids]
    return cell


def _measure(cell: Any, name: str) -> Any:
    return next(m for m in cell.measures.measures if m.name == name)


class TestACellsFactsAreItsNonFaultedObservations:
    """The population every bar is adjudicated over, with the faulted count beside it."""

    def test_a_faulted_observation_is_counted_out_and_named(self) -> None:
        campaign, storage = toyhost_campaign()
        narrow_id = campaign.run_ids[0]
        results = {run_id: storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE) for run_id in campaign.run_ids}
        results[narrow_id] = [
            results[narrow_id][0].model_copy(update={"infra_error": "grader timed out"}),
            *results[narrow_id][1:],
        ]
        _campaign, bundle = _assemble_with(results=results)

        narrow, wide = _cell_of_run(bundle, narrow_id), _cell_of_run(bundle, campaign.run_ids[1])
        assert (narrow.n_observations, narrow.n_infra_excluded) == (36, 1)
        assert _measure(narrow, "total_ms").n == 35
        assert (wide.n_infra_excluded, _measure(wide, "total_ms").n) == (0, 36)

    def test_a_short_run_and_an_unfinished_run_are_named_on_their_own_cells(self) -> None:
        campaign, storage = toyhost_campaign()
        narrow, wide = (storage.load_eval_run(run_id, TOYHOST_SCOPE) for run_id in campaign.run_ids)
        short = narrow.model_copy(
            update={
                "completeness": RunCompleteness(
                    expected_cells=40,
                    produced_cells=36,
                    persisted_cells=36,
                    infra_excluded_cells=0,
                    counted_from="run_loop",
                )
            }
        )
        unfinished = wide.model_copy(update={"status": "running"})
        _campaign, bundle = _assemble_with(runs=[short, unfinished])

        narrow_cell, wide_cell = _cell_of_run(bundle, short.id), _cell_of_run(bundle, unfinished.id)
        assert narrow_cell.short_runs == {short.id: bundle.short_runs[short.id]}
        assert narrow_cell.incomplete_runs == {}
        assert wide_cell.incomplete_runs == {unfinished.id: "running"}
        assert wide_cell.short_runs == {}


class TestTheJudgedHalfOfTheSurface:
    """Each judged dimension a cell scored is described by the bundle's own judged measure."""

    def test_every_scored_dimension_carries_its_declared_polarity_and_scale(self) -> None:
        bundle = _bundle()
        (judged,) = bundle.judged_measures

        facts = cell_dimension_facts(bundle)

        assert list(facts) == [TOYHOST_JUDGED_DIMENSION]
        assert (facts[TOYHOST_JUDGED_DIMENSION].higher_is_better, facts[TOYHOST_JUDGED_DIMENSION].value_range) == (
            judged.higher_is_better,
            judged.value_range,
        )
        assert judged.value_range is not None, "a scale the fixture never declared would make the comparison vacuous"

    def test_a_lower_is_better_dimension_is_described_as_one(self) -> None:
        """The toy host declares its dimension higher-is-better, so polarity is read off a bundle that says otherwise."""
        bundle = _bundle()
        (judged,) = bundle.judged_measures
        flipped = bundle.model_copy(update={"judged_measures": [judged.model_copy(update={"higher_is_better": False})]})

        assert cell_dimension_facts(flipped)[TOYHOST_JUDGED_DIMENSION].higher_is_better is False


class TestAnInsightsStandingIsReadOffItsMintingAnalysis:
    """Archived retracts; unresolvable orphans; each analysis is asked once."""

    def test_the_three_standings_are_told_apart(self) -> None:
        def insight(name: str, source: str) -> EvalInsight:
            return EvalInsight(
                scope_id=TOYHOST_SCOPE,
                id=name,
                subject_id=TOYHOST_SUBJECT.subject_id,
                subject_kind="extractor_config",
                statement=f"observed {name}",
                confidence="medium",
                source_analysis_id=source,
                invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
            )

        asked: list[str] = []
        archived = {"an-archived": True, "an-live": False}

        def lookup(analysis_id: str) -> bool | None:
            asked.append(analysis_id)
            return archived.get(analysis_id)

        standing = insight_standing(
            [
                insight("i-4", "an-gone"),
                insight("i-1", "an-archived"),
                insight("i-2", "an-live"),
                insight("i-3", ""),
                insight("i-5", "an-archived"),
            ],
            lookup,
        )

        assert standing.retracted == {"i-1": "an-archived", "i-5": "an-archived"}
        assert standing.orphaned == {"i-4": "an-gone"}
        assert sorted(asked) == ["an-archived", "an-gone", "an-live"]


class _FlagOnlyStorage(ToyhostStorage):
    """A store that can say whether an analysis is archived but cannot load one — the port as declared."""

    def __init__(self, base: ToyhostStorage, archived: dict[str, bool]) -> None:
        self.__dict__.update(base.__dict__)
        self._archived = archived

    def load_analysis(self, analysis_id: str, scope_id: str) -> EvalAnalysis | None:
        raise AssertionError(f"assembly hydrated analysis {analysis_id!r}; it may ask only whether it is archived")

    def analysis_archived(self, analysis_id: str, scope_id: str) -> bool | None:
        return self._archived.get(analysis_id) if scope_id == TOYHOST_SCOPE else None

    def analyses_archived(self, analysis_ids: Sequence[str], scope_id: str) -> dict[str, bool]:
        return {
            analysis_id: archived
            for analysis_id in analysis_ids
            if (archived := self.analysis_archived(analysis_id, scope_id)) is not None
        }


class TestRetractionReadsOnlyTheArchivedFlag:
    """One stored analysis a newer validator rejects cannot abort assembly."""

    def test_an_archived_minting_analysis_retracts_without_being_loaded(self) -> None:
        retracted = EvalInsight(
            scope_id=TOYHOST_SCOPE,
            id="i-retracted",
            subject_id=TOYHOST_SUBJECT.subject_id,
            subject_kind="extractor_config",
            statement="the wide chunk is cheaper",
            confidence="low",
            source_analysis_id="an-archived",
            invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
        )
        profile = toyhost_profile()
        campaign, storage = toyhost_campaign(insights=[retracted])
        bundle = assemble_context_bundle(
            campaign, storage=_FlagOnlyStorage(storage, {"an-archived": True}), profile=profile
        )

        assert bundle.retracted_insights == {"i-retracted": "an-archived"}
        assert all(insight.id != "i-retracted" for insight in bundle.prior_insights)


class TestEveryIntervalIsAtTheOneLevel:
    """The bundle's intervals are computed at ``stats.INTERVAL_LEVEL``, not a level of its own."""

    @pytest.mark.parametrize("name", ["total_ms", "cost_usd", "field_accuracy"])
    def test_moving_the_level_moves_every_cell_interval(self, name: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A score on a declared range takes the t interval: its half-width is the multiplier at the moved level
        times its standard error. A time or a spend takes the skewed interval (2026-11: the t interval alone
        covered a skewed mean far under its level, ``tests/test_simulated_floored_mean.py``): its ends are exactly
        ``observed_mean_interval(..., skewed=True)``'s over the cell's observations at the moved level, and not the
        t interval's, so a bundle that stopped asking for it fails here."""
        at_95 = _measure(_bundle().cell_measures[0], name)
        monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.80)
        cell = _bundle().cell_measures[0]
        at_80 = _measure(cell, name)

        # The interval is read on the cell's cases, not its observations (#590).
        half = stats.t_critical_two_sided(0.80, at_80.n_independent - 1) * at_80.sem
        if name == "field_accuracy":
            assert at_80.ci_high - at_80.mean == pytest.approx(half)
        else:
            _, storage = toyhost_campaign()
            (run_id,) = cell.run_ids
            results = storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)
            values = [result.latency.total_ms if name == "total_ms" else result.cost_usd for result in results]
            cases = [result.test_case_id for result in results]
            expected = stats.observed_mean_interval(values, cases=cases, floor=0.0, skewed=True)
            assert expected is not None
            assert (at_80.ci_low, at_80.ci_high) == pytest.approx(expected, rel=1e-12)
            assert at_80.ci_high - at_80.mean > half * (1 + 1e-6), "wider above than the t interval it contains"
        assert at_95.ci_low <= at_80.ci_low and at_80.ci_high <= at_95.ci_high
        assert at_80.ci_high - at_80.ci_low < at_95.ci_high - at_95.ci_low


class TestARemainderIsWithheldByTheSharedRule:
    """The divergence lens asks ``metrics.remainder_withheld_reason``."""

    def test_one_component_of_a_partition_earns_no_remainder(self) -> None:
        campaign, storage = toyhost_campaign()
        partitioned = {
            run_id: [
                result.model_copy(
                    update={
                        "latency": LatencyMetrics(
                            total_ms=result.latency.total_ms + 400.0,
                            llm_ms=result.latency.total_ms,
                            tool_ms=300.0 + result.k_iteration,
                        )
                    }
                )
                for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)
            ]
            for run_id in campaign.run_ids
        }
        _campaign, bundle = _assemble_with(results=partitioned)

        divergence = next(
            d for d in bundle.scope_divergences if d.end_to_end.name == "total_ms" and d.subsystem.name == "tool_ms"
        )
        assert divergence.unattributed_delta is None
        assert WITHHELD_PARTITION_INCOMPLETE in divergence.unattributed_withheld


class TestAPhaseTimingIsCataloguedAsAPhase:
    """The catalog describes a reported measure through ``metrics.describe_reported_measure``."""

    def test_a_phase_timing_reaching_the_bundle_is_described_as_measured_wall_clock(self) -> None:
        campaign, storage = toyhost_campaign()
        timed = {
            run_id: [
                result.model_copy(update={"phase_timings": {"ocr_page_ms": 40.0 + result.k_iteration}})
                for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)
            ]
            for run_id in campaign.run_ids
        }
        _campaign, bundle = _assemble_with(results=timed)

        described = bundle.measure_catalog["ocr_page_ms"]
        assert (described.family, described.unit, described.attribution_scope) == ("mechanical", "ms", "subsystem")


class TestTheLeverSpreadIsSpelledByTheNumberRule:
    """The within-level spread is written as every other reader-facing number is."""

    def test_the_spread_reads_back_as_its_own_number_rule_spelling(self) -> None:
        (lever,) = _bundle().coverage
        assert lever.dispersion.startswith("±")
        spelled = lever.dispersion[1:]
        # A spread below one millionth reads as the bound it is under, which the number rule writes for every
        # value there; anything above it reads back as its own spelling.
        bound = format_number(WRITTEN_FROM / 2)
        assert spelled == bound or spelled == format_number(float(spelled))


class TestARefusedBarIsQuotedAsDeclared:
    """``kernel/declaration.py``: the refusal echoes the author's value.

    The threshold is spelled with ``repr``, which keeps the kernel closed, rather than with
    ``:g`` or the reader-facing number rule. A value ``:g`` rounds and the number rule separates is
    what tells the three apart.
    """

    def test_the_threshold_is_echoed_exactly_as_the_author_declared_it(self) -> None:
        profile = toyhost_profile()
        campaign, _storage = toyhost_campaign()
        design = toyhost_aspect_design(
            bars=[BarOverride(measure_id="extraction_vibes", threshold=12345678.5, direction="higher_is_better")]
        )
        with pytest.raises(ValueError, match=r"extraction_vibes") as raised:
            refuse_an_undeclarable_design(design, behavior=campaign.behavior, template=None, profile=profile)

        assert "the bar at 12345678.5 names 'extraction_vibes'" in str(raised.value)


# =============================================================================
# analysis/doc_analysis.py
# =============================================================================


class TestAStoredAnalysisFreezesTheBundlesSurface:
    """The stored surface is the bundle's cells, bars and control, never recomputed."""

    def test_the_surface_carries_the_bundles_cells_and_every_bar(self) -> None:
        bundle = _bundle()
        analysis, _ = doc_analysis_over(campaign_bundle(bundle))

        surface = analysis.aspects[0].decision_surface
        assert surface is not None
        assert surface.cells == bundle.cell_measures
        assert surface.bars == bundle.bar_adjudications
        assert len(surface.bars) == 3, "the toy campaign holds three bars; an empty list would match an empty bundle"

    def test_the_surface_names_the_declared_control(self) -> None:
        # Declared on the campaign and assembled, as a control is: the bundle matches it to its arm once
        # (`design.control_arm`), and the surface reads that match, so patching the declaration onto an
        # already-assembled bundle would reach no reader.
        profile = toyhost_profile()
        campaign, storage = toyhost_campaign(profile=profile)
        narrow_key = cell_at(_bundle(), TOYHOST_NARROW).split(":")[0]
        controlled = assemble_context_bundle(with_control(campaign, narrow_key), storage=storage, profile=profile)

        analysis, _ = doc_analysis_over(campaign_bundle(controlled))

        assert analysis.aspects[0].decision_surface.control_variant_key == narrow_key
