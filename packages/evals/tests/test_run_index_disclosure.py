"""The run index a generated analysis stores says what it left out, and reads each tail at its worse end (#656).

The index reports at most a handful of measures per run, chosen by which ones moved, and names every
measure it left out in each entry's ``key_metrics["measures_omitted"]`` — the disclosure that stops a
narrow table being read as complete. These drive the real build path: a generation over a toy bundle
whose run summaries a test has shaped, read back off the stored analysis.
"""

from __future__ import annotations

import json
from typing import Any

from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.schema.models import utc_now_iso
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload

#: Comfortably past any reasonable column cap, so the capped path is reached whatever the cap is tuned to.
_MANY = 30


def _with_measures(bundle: Any, per_run: list[list[MeasureSummary]]) -> Any:
    """The bundle with each run summary's measures replaced, in run order."""
    summaries = [
        summary.model_copy(
            update={
                "measures": summary.measures.model_copy(update={"measures": sorted(measures, key=lambda m: m.name)})
            }
        )
        for summary, measures in zip(bundle.run_summaries, per_run, strict=True)
    ]
    return bundle.model_copy(update={"run_summaries": summaries})


def _measure(bundle: Any, name: str, *, run: int = 0) -> MeasureSummary:
    return next(m for m in bundle.run_summaries[run].measures.measures if m.name == name)


async def _run_index(bundle: Any) -> list[Any]:
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(memo_payload(bundle))),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis.run_index


def _reported(entry: Any) -> set[str]:
    """The measure names an entry carries a column for, whatever tail key names them."""
    names = set()
    for key in entry.key_metrics:
        for suffix in ("_p95", "_p05", "_max"):
            if key.endswith(suffix):
                names.add(key.removesuffix(suffix))
    return names


def _omitted(entry: Any) -> set[str]:
    listed = entry.key_metrics.get("measures_omitted")
    return set(listed.split(", ")) if listed else set()


class TestAMeasureThatDidNotMoveIsNamed:
    async def test_a_measure_identical_across_runs_is_left_out_and_named(self) -> None:
        bundle = toyhost_bundle()
        same = _measure(bundle, "cost_usd", run=0)
        shaped = _with_measures(
            bundle,
            [
                [m if m.name != "cost_usd" else same for m in summary.measures.measures]
                for summary in bundle.run_summaries
            ],
        )

        index = await _run_index(shaped)

        assert len(index) == len(bundle.run_summaries)
        every = {m.name for m in bundle.run_summaries[0].measures.measures}
        for entry in index:
            assert "cost_usd" not in _reported(entry)
            assert "cost_usd" in _omitted(entry)
            assert _reported(entry) | _omitted(entry) == every, "every measure is either a column or named"
            assert "total_ms" in _reported(entry), "a measure that moved keeps its column"


class TestAWideCampaignIsCappedAndSaysSo:
    async def test_every_measure_not_reported_is_named_in_every_entry(self) -> None:
        bundle = toyhost_bundle()
        template = _measure(bundle, "total_ms")
        shaped = _with_measures(
            bundle,
            [
                [
                    template.model_copy(update={"name": f"phase_{i:02d}_ms", "p95": 100.0 + i * (1 + run)})
                    for i in range(_MANY)
                ]
                for run in range(len(bundle.run_summaries))
            ],
        )
        every = {f"phase_{i:02d}_ms" for i in range(_MANY)}

        index = await _run_index(shaped)

        reported = _reported(index[0])
        assert 0 < len(reported) < _MANY, "the index is capped"
        for entry in index:
            assert _reported(entry) == reported, "every row reports the same columns"
            assert _omitted(entry) == every - reported
            assert _reported(entry) | _omitted(entry) == every

    async def test_the_measures_that_moved_most_are_the_ones_kept(self) -> None:
        bundle = toyhost_bundle()
        template = _measure(bundle, "total_ms")
        # phase_i moves by i percent between the two runs, so the largest i moved most.
        shaped = _with_measures(
            bundle,
            [
                [
                    template.model_copy(update={"name": f"phase_{i:02d}_ms", "p95": 100.0 * (1 + run * i / 100)})
                    for i in range(_MANY)
                ]
                for run in range(len(bundle.run_summaries))
            ],
        )

        index = await _run_index(shaped)

        kept = sorted(_reported(index[0]))
        assert kept == [f"phase_{i:02d}_ms" for i in range(_MANY - len(kept), _MANY)]


class TestATailIsReadAtItsWorseEnd:
    async def test_lower_is_better_reads_p95_and_higher_is_better_reads_p05(self) -> None:
        bundle = toyhost_bundle()

        (entry, *_rest) = await _run_index(bundle)

        assert entry.key_metrics["total_ms_p95"] == _measure(bundle, "total_ms").p95
        assert entry.key_metrics["field_accuracy_p05"] == _measure(bundle, "field_accuracy").p05
        assert "total_ms_p05" not in entry.key_metrics
        assert "field_accuracy_p95" not in entry.key_metrics

    async def test_too_few_observations_for_a_p95_reads_the_slowest_seen_under_its_own_name(self) -> None:
        bundle = toyhost_bundle()
        # Nothing moves, so every measure is a column; the latency has no p95 to give.
        first = bundle.run_summaries[0].measures.measures
        shaped = _with_measures(
            bundle,
            [
                [m.model_copy(update={"p95": None, "max": 1500.0 + run}) if m.name == "total_ms" else m for m in first]
                for run in range(len(bundle.run_summaries))
            ],
        )

        index = await _run_index(shaped)

        assert [entry.key_metrics.get("total_ms_max") for entry in index] == [1500.0, 1501.0]
        assert all("total_ms_p95" not in entry.key_metrics for entry in index)


class TestBothCostSidesAreCarried:
    async def test_the_program_floor_and_the_production_figures_travel_together(self) -> None:
        bundle = toyhost_bundle()

        index = await _run_index(bundle)

        for entry, summary in zip(index, bundle.run_summaries, strict=True):
            metrics = entry.key_metrics
            assert metrics["program_cost_usd_floor"] == summary.cost_usd
            assert summary.prod_cost_usd is not None, "the toy corpus prices its candidate"
            assert metrics["total_prod_cost_usd"] == summary.prod_cost_usd
            assert metrics["mean_prod_cost_usd"] == summary.mean_prod_cost_usd
            assert metrics["n_prod_cost_usd"] == summary.n_prod_cost_usd
            assert metrics["n_prod_cost_unmeasured"] == summary.n_results - summary.n_prod_cost_usd
