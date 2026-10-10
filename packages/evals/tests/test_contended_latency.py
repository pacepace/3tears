"""Latency read under concurrency is never compared, ranked, barred or pooled as if it were clean (#701).

A result stamped ``execution_mode = "concurrent"`` recorded its wall-clock while other cells or runs executed
beside it. These build a two-arm campaign whose latency differs sharply between the arms, and read every
surface that consumes latency: with the latency read serially it is contrasted, Holm-corrected, charted and
ranked; read under concurrency it is in none of them, and the bundle and the report say so in one line. A
cell pooling a serial run with a concurrent one reads its latency from the serial run alone, and a history
series never steps across a contended reading. Every other measure of a contended result still counts.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``withhold_contended_latency`` in the bundle's load: the contended arm's latency is contrasted and ranked.
- ``latency_contended``: reading any value but ``concurrent`` as contended, or none — the serial arms lose
  their latency, or the concurrent ones keep it.
- ``compute_frontier`` / ``compute_history``: dropping the withhold — the frontier ranks a contended latency,
  a latency series steps across one.
- ``_latency_contended``: the bundle stops naming the cells, or the report stops saying it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.report.build import build_code_only_report
from threetears.evals.analysis.report.model import DisclosureBlock
from threetears.evals.analysis.reporting import METRIC_TOTAL_MS, compute_frontier
from threetears.evals.analysis.lenses.history import compute_history
from threetears.evals.kernel import EvalCampaign, Question
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import LatencyMetrics
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

CONTROL = "control-model"
CONTRAST = "contrast-model"
#: Per-arm turn times: the contrast is about 600 ms faster on every case, a separation no correction removes. The
#: gap varies by a few ms between cases: latency declares no range, so a gap of one amount on every case is never
#: called separated (#597), and this file is about concurrency, not that.
_BASE_MS = {CONTROL: 1000.0, CONTRAST: 400.0}
_CASES = 12
_PROFILE = toyhost_profile()


def _results(run: EvalRun, model: str, mode: str, *, offset_ms: float = 0.0) -> list[EvalResult]:
    return [
        make_eval_result(
            id=f"{run.id}-{case}",
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=model,
            test_case_id=f"tc-{case:02d}",
            goal_state_outcomes=[],
            latency=LatencyMetrics(
                total_ms=_BASE_MS[model] + offset_ms + 10.0 * case + (7.0 * (case % 3) if model == CONTRAST else 0.0)
            ),
            covariates={"execution_mode": mode},
        )
        for case in range(_CASES)
    ]


def _campaign(
    arms: Sequence[tuple[str, str, float]],
) -> tuple[EvalCampaign, list[EvalRun], dict[str, list[EvalResult]]]:
    """One run per ``(model, execution_mode, offset_ms)``; the control declared, one question on latency."""
    runs: list[EvalRun] = []
    results: dict[str, list[EvalResult]] = {}
    for model, mode, offset in arms:
        run = make_eval_run(status="completed", candidate_model=model)
        runs.append(run)
        results[run.id] = _results(run, model, mode, offset_ms=offset)
    declaration = minimal_declaration(control=fixture_variant_key(CONTROL)).model_copy(
        update={"questions": [Question(id="q-faster", text="is the contrast faster?", merit_axes=["latency"])]}
    )
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="latency under concurrency",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id for run in runs],
        declared_design=declaration,
        created_by="test:fixture",
    )
    return campaign, runs, results


def _bundle(arms: Sequence[tuple[str, str, float]]) -> AnalysisContextBundle:
    campaign, runs, results = _campaign(arms)
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=_PROFILE)


def _latency_contrasts(bundle: AnalysisContextBundle) -> list[Any]:
    return [
        comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if comparison.name == "total_ms"
    ]


def _cell_latency(bundle: AnalysisContextBundle, model: str) -> Any:
    (cell,) = [cell for cell in bundle.cell_measures if cell.variant_key == fixture_variant_key(model)]
    return next((measure for measure in cell.measures.measures if measure.name == "total_ms"), None)


def _point(bundle: AnalysisContextBundle, model: str) -> Any:
    (point,) = [point for subject in bundle.frontier.subjects for point in subject.points if point.model == model]
    return point


def _disclosures(bundle: AnalysisContextBundle) -> list[str]:
    report = build_code_only_report(bundle, measures=_PROFILE.measures, assembled_at="2026-10-10T00:00:00+00:00")
    return [block.text for block in report.blocks if isinstance(block, DisclosureBlock)]


class TestLatencyReadSeriallyIsCompared:
    """The control: the same campaign with every latency read serially reaches every surface."""

    def test_it_is_contrasted_held_in_the_family_and_ranked(self) -> None:
        bundle = _bundle([(CONTROL, "serial", 0.0), (CONTRAST, "serial", 0.0)])

        (contrast,) = _latency_contrasts(bundle)
        assert contrast.verdict == "improved", "the control arm must show the latency difference is readable"
        assert _cell_latency(bundle, CONTRAST) is not None and _cell_latency(bundle, CONTRAST).n == _CASES
        assert _point(bundle, CONTRAST).mean_total_ms is not None
        assert bundle.latency_contended is None and bundle.latency_contended_cells == []
        assert not any("concurrency" in text for text in _disclosures(bundle))


class TestLatencyReadUnderConcurrencyIsKeptOutOfEverySurface:
    def test_it_is_in_no_contrast_no_family_no_cell_and_no_ranking(self) -> None:
        bundle = _bundle([(CONTROL, "serial", 0.0), (CONTRAST, "concurrent", 0.0)])

        # Listed, so the question's latency reading is not silently missing — untested, carrying no p into the
        # Holm correction, and naming why.
        (contrast,) = _latency_contrasts(bundle)
        assert contrast.verdict == "untested" and contrast.p_raw is None and contrast.p_adjusted is None
        assert contrast.untested_reason is not None and "latency_contended" in contrast.untested_reason
        assert _cell_latency(bundle, CONTRAST) is None, "a contended latency was summarised on its cell"
        assert _point(bundle, CONTRAST).mean_total_ms is None, "the frontier ranked a contended latency"
        assert _cell_latency(bundle, CONTROL) is not None, "the serial arm keeps its latency"

    def test_every_other_measure_of_a_contended_result_still_counts(self) -> None:
        bundle = _bundle([(CONTROL, "serial", 0.0), (CONTRAST, "concurrent", 0.0)])

        (cell,) = [cell for cell in bundle.cell_measures if cell.variant_key == fixture_variant_key(CONTRAST)]
        assert cell.n_observations == _CASES
        # The mark itself is reported on the cell: a reader sees why its latency is absent.
        assert any(measure.name == "execution_mode" for measure in cell.measures.measures)

    def test_the_bundle_names_the_cell_and_the_report_says_so_in_one_line(self) -> None:
        bundle = _bundle([(CONTROL, "serial", 0.0), (CONTRAST, "concurrent", 0.0)])

        assert [cell.variant_key for cell in bundle.latency_contended_cells] == [fixture_variant_key(CONTRAST)]
        assert bundle.latency_contended is not None
        assert f"the latency of {_CASES} of {2 * _CASES} results in 1 of 2 cells" in bundle.latency_contended
        assert "measure_latency=True" in bundle.latency_contended
        (line,) = [text for text in _disclosures(bundle) if text.startswith("Latency is not compared here")]
        assert "Read under concurrency:" in line, "the report names the arm whose latency it left out"

    def test_a_run_with_no_latency_says_nothing_about_withholding_it(self) -> None:
        campaign, runs, results = _campaign([(CONTROL, "concurrent", 0.0), (CONTRAST, "concurrent", 0.0)])
        untimed = {
            run_id: [result.model_copy(update={"latency": None}) for result in members]
            for run_id, members in results.items()
        }
        bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(runs, untimed), profile=_PROFILE)

        assert bundle.latency_contended is None and bundle.latency_contended_cells == []


class TestPoolingSerialAndConcurrentRunsKeepsThemApart:
    def test_a_cell_pooling_both_reads_its_latency_from_the_serial_run_alone(self) -> None:
        # The control was measured twice: once serially, once under concurrency 4 s slower. Pooled, the
        # contended run would drag the control's latency up; kept apart, the cell reads the serial run alone.
        bundle = _bundle([(CONTROL, "serial", 0.0), (CONTROL, "concurrent", 4000.0), (CONTRAST, "serial", 0.0)])

        control = _cell_latency(bundle, CONTROL)
        assert control is not None and control.n == _CASES, "the contended run's latency pooled into the cell"
        assert control.mean == sum(_BASE_MS[CONTROL] + 10.0 * case for case in range(_CASES)) / _CASES
        (contrast,) = _latency_contrasts(bundle)
        assert contrast.verdict == "improved", "the serial latency on both arms is still compared"
        assert bundle.latency_contended is not None and "in 1 of 2 cells" in bundle.latency_contended


class TestTheScopeLenses:
    def test_the_frontier_never_ranks_a_contended_latency_and_says_so(self) -> None:
        _campaign_, runs, results = _campaign([(CONTROL, "serial", 0.0), (CONTRAST, "concurrent", 0.0)])
        everything = [result for members in results.values() for result in members]

        frontier = compute_frontier(runs, everything, archived_run_ids=None, profile=_PROFILE)

        points = {point.model: point for subject in frontier.subjects for point in subject.points}
        assert points[CONTRAST].mean_total_ms is None and points[CONTROL].mean_total_ms is not None
        assert frontier.contended_latency_disclosure is not None
        assert f"the latency of {_CASES} of {2 * _CASES} results" in frontier.contended_latency_disclosure

    def test_a_latency_series_never_steps_across_a_contended_reading(self) -> None:
        _campaign_, runs, results = _campaign([(CONTROL, "serial", 0.0), (CONTROL, "concurrent", 4000.0)])
        everything = [result for members in results.values() for result in members]

        history = compute_history(runs, everything, metric=METRIC_TOTAL_MS, archived_run_ids=None, profile=_PROFILE)

        (series,) = history.series
        values = [point.value for point in series.points]
        assert values[0] is not None and values[1] is None, "a contended run's latency entered the series"
        assert series.points[1].regression is None or series.points[1].regression.delta is None
        assert history.contended_latency_disclosure is not None

    def test_a_series_of_any_other_measure_is_untouched(self) -> None:
        _campaign_, runs, results = _campaign([(CONTROL, "serial", 0.0), (CONTROL, "concurrent", 4000.0)])
        everything = [result for members in results.values() for result in members]

        history = compute_history(runs, everything, archived_run_ids=None, profile=_PROFILE)

        assert history.contended_latency_disclosure is None


class TestACampaignDeclaringLatencyUnderTest:
    def test_its_runs_read_under_concurrency_are_named_with_the_remedy(self) -> None:
        campaign, runs, results = _campaign([(CONTROL, "serial", 0.0), (CONTRAST, "concurrent", 0.0)])
        assert campaign.declared_design is not None
        declared = campaign.model_copy(
            update={"declared_design": campaign.declared_design.model_copy(update={"measure_latency": True})}
        )

        bundle = assemble_context_bundle(declared, storage=ToyhostStorage(runs, results), profile=_PROFILE)

        assert bundle.latency_contended is not None
        assert "This campaign declares latency under test" in bundle.latency_contended
        assert bundle.latency_contended.count("measure_latency=True") == 1
