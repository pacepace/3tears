"""The engine's bundle and generator behaviour, driven over the toy host.

Bundle assembly and generation over a toy bundle are on the toy host's surface, so each engine
behaviour there owes a toy test. Every test below assembles the toy corpus campaign under the toy
host's profile, or generates a memo over it with a fixtured completion. The fixtured memo and the
coordinates it cites live in the toy host's own ``memo`` module, which every toy-host generation
test shares.

Pinned here — each class is a property a mutation of the engine turned red:

- ``analysis/bundle.py``: each cell's facts over its non-faulted observations, with its run notes;
  the judged half of the surface; an insight's standing; retraction read off the archived flag
  alone; intervals at the one level; the shared remainder predicate and measure description; the
  lever spread's spelling.
- ``kernel/declaration.py``: a refused bar quoted as declared.
- ``analysis/generator.py``: the surface a generation freezes; the prompt version over the
  assembled prompt; the billed-cost spelling; one parser for every reference site; prose is not
  refused; the two functions made public for the reporter's judge.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.analysis import stats
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.surface import cell_dimension_facts
from threetears.evals.analysis.bundle.insights import insight_standing
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal
from threetears.evals.analysis.generator import (
    build_user_message,
    first_request,
    generate_analysis,
    prompt_content_version,
    refuse_an_undescribable_arm_table,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.kernel.campaign import EvalAnalysis, EvalInsight
from threetears.evals.kernel.declaration import BarOverride, refuse_an_undeclarable_design
from threetears.evals.kernel.metrics import WITHHELD_PARTITION_INCOMPLETE
from threetears.evals.schema.models import LatencyMetrics, RunCompleteness, utc_now_iso
from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_NARROW,
    TOYHOST_WIDE,
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
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    alias_at,
    cell_at,
    memo_payload,
)


def _bundle() -> Any:
    profile = toyhost_profile()
    return toyhost_bundle(profile=profile)


async def _generate(
    bundle: Any, payload: dict[str, Any] | None = None, *, client: Any = None
) -> tuple[EvalAnalysis, Any]:
    client = (
        client
        if client is not None
        else FixturedClient(json.dumps(payload if payload is not None else memo_payload(bundle)))
    )
    profile = toyhost_profile()
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=profile,
    )
    return analysis, client


def _finding(payload: dict[str, Any]) -> dict[str, Any]:
    (finding,) = payload["findings"]
    return finding


# =============================================================================
# analysis/bundle.py
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

    def test_moving_the_level_moves_every_cell_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
        at_95 = _measure(_bundle().cell_measures[0], "total_ms")
        monkeypatch.setattr(stats, "INTERVAL_LEVEL", 0.80)
        at_80 = _measure(_bundle().cell_measures[0], "total_ms")

        # The interval is read on the cell's cases, not its observations (#590).
        half = stats.t_critical_two_sided(0.80, at_80.n_independent - 1) * at_80.sem
        assert at_80.ci_high - at_80.mean == pytest.approx(half)
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
        assert lever.dispersion == f"±{format_number(float(lever.dispersion[1:]))}"


class TestARefusedBarIsQuotedAsDeclared:
    """``kernel/declaration.py``: the refusal echoes the author's value.

    The threshold is spelled with ``repr``, which keeps the kernel closed, rather than with
    ``:g`` or the reader-facing number rule. A value ``:g`` rounds and the number rule separates is
    what tells the three apart.
    """

    def test_the_threshold_is_echoed_exactly_as_the_author_declared_it(self) -> None:
        profile = toyhost_profile()
        campaign, _storage = toyhost_campaign()
        design = campaign.declared_design.model_copy(
            update={
                "bars": [BarOverride(measure_id="extraction_vibes", threshold=12345678.5, direction="higher_is_better")]
            }
        )
        with pytest.raises(ValueError, match=r"extraction_vibes") as raised:
            refuse_an_undeclarable_design(design, behavior=campaign.behavior, template=None, profile=profile)

        assert "the bar at 12345678.5 names 'extraction_vibes'" in str(raised.value)


# =============================================================================
# analysis/generator.py
# =============================================================================


class TestAGenerationFreezesTheBundlesSurface:
    """The stored surface is the bundle's cells, bars and control, never recomputed."""

    async def test_the_surface_carries_the_bundles_cells_and_every_bar(self) -> None:
        bundle = _bundle()
        analysis, _client = await _generate(bundle)

        surface = analysis.decision_surface
        assert surface is not None
        assert surface.cells == bundle.cell_measures
        assert surface.bars == bundle.bar_adjudications
        assert len(surface.bars) == 3, "the toy campaign holds three bars; an empty list would match an empty bundle"

    async def test_the_surface_names_the_declared_control(self) -> None:
        bundle = _bundle()
        narrow_key = cell_at(bundle, TOYHOST_NARROW).split(":")[0]
        controlled = bundle.model_copy(
            update={"declared_design": bundle.declared_design.model_copy(update={"control": narrow_key})}
        )

        analysis, _client = await _generate(controlled)

        assert analysis.decision_surface.control_variant_key == narrow_key


class TestTheGeneratorsReadersArePublished:
    """The reporter's judge reads what the generator sends, through the generator's own functions.

    Both were renames to a public name; reverting either fails these by ``ImportError`` alone.
    """

    async def test_the_user_message_is_the_one_the_generator_sent(self) -> None:
        bundle = _bundle()
        _analysis, client = await _generate(bundle)

        (call,) = client.calls
        assert build_user_message(bundle) == call["user"]

    def test_a_bundle_with_one_describable_arm_is_not_refused(self) -> None:
        refuse_an_undescribable_arm_table(_bundle())

    def test_a_bundle_whose_every_arm_is_undescribable_is_refused_before_any_call(self) -> None:
        bundle = _bundle()
        opaque = [
            entry.model_copy(update={"levels_unavailable": "recorded before levels were kept"})
            for entry in bundle.variant_index
        ]

        with pytest.raises(GenerationError, match=r"has no describable arm") as raised:
            refuse_an_undescribable_arm_table(bundle.model_copy(update={"variant_index": opaque}))
        assert not isinstance(raised.value, SoundnessRefusal)


class TestThePromptVersionIsWhatTheModelWasTold:
    """The version hashes what the model was told, with the host's register included."""

    async def test_the_recorded_version_is_the_hash_of_what_the_model_was_told(self) -> None:
        """The assembled system prompt AND the contract sent with it — two schemas under one prompt differ."""
        analysis, client = await _generate(_bundle())

        (call,) = client.calls
        _, _, sent = first_request(_bundle(), PROMPT, toyhost_profile())
        contract = json.dumps(sent, sort_keys=True, separators=(",", ":"))
        told = hashlib.sha256((call["system"] + "\n" + contract).encode("utf-8")).hexdigest()[:12]
        assert analysis.generation.prompt_version == told
        assert analysis.generation.prompt_version != hashlib.sha256(call["system"].encode("utf-8")).hexdigest()[:12]
        assert analysis.generation.prompt_version != hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:12]

    def test_the_one_derivation_agrees_with_what_the_generator_sends(self) -> None:
        profile = toyhost_profile()
        system, _, contract = first_request(_bundle(), PROMPT, profile)
        told = system + "\n" + json.dumps(contract, sort_keys=True, separators=(",", ":"))
        assert _bundle().time_axis is None, "the toy campaign runs on one day under no build label"
        assert (
            prompt_content_version(PROMPT, profile, time_axis=False)
            == hashlib.sha256(told.encode("utf-8")).hexdigest()[:12]
        )


@dataclass
class _Completion:
    content: str
    cost_usd: float | None
    stop_reason: str = "end_turn"
    input_tokens: int = 1000
    output_tokens: int = 700
    model: str = MODEL
    tool_calls: list[Any] = field(default_factory=list)
    reasoning_tokens: int | None = None


class _BilledClient:
    """Returns the same billed completion for every call."""

    def __init__(self, completion: _Completion) -> None:
        self._completion = completion
        self.calls: list[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        return MODEL

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> _Completion:
        self.calls.append({"system": system, "user": user})
        return self._completion


class TestTheBilledCostIsSpelledByTheNumberRule:
    """The cost carried on a failed generation is written as every other reader-facing number is."""

    async def test_a_twice_refused_generation_states_both_calls_cost(self) -> None:
        client = _BilledClient(_Completion(content="this is not an analysis", cost_usd=1234.5))
        with pytest.raises(SoundnessRefusal) as raised:
            await _generate(_bundle(), client=client)

        assert len(client.calls) == 2
        assert f"${format_number(2469.0)} was billed across both calls" in str(raised.value)

    async def test_a_cut_short_generation_states_its_one_calls_cost(self) -> None:
        client = _BilledClient(_Completion(content="{", cost_usd=1234.5, stop_reason="max_tokens"))
        with pytest.raises(GenerationError) as raised:
            await _generate(_bundle(), client=client)

        assert len(client.calls) == 1
        assert f"${format_number(1234.5)} was billed on the one call" in str(raised.value)


def _delta_chart(bundle: Any) -> dict[str, Any]:
    """A drawable chart over the two toy arms: narrow first, then wide, on wall-clock — cells as the writer names them."""
    narrow, wide = alias_at(bundle, TOYHOST_NARROW), alias_at(bundle, TOYHOST_WIDE)
    return {
        "type": "delta_table",
        "cells": [narrow, wide],
        "measures": [{"measure_id": "total_ms", "reading": "measure"}],
        "axis": "",
        "note": "",
        "caption": "Wall-clock per document at each width.",
    }


def _misspelt_reading_at(site: str, bundle: Any) -> dict[str, Any]:
    """The fixtured memo with a reading at ``site`` spelt ``raeding`` — a key that decides what is read."""
    payload = copy.deepcopy(memo_payload(bundle))
    finding = _finding(payload)
    if site == "evidence":
        readings = finding["evidence"]
    elif site == "chart measure":
        finding["chart"] = _delta_chart(bundle)
        readings = finding["chart"]["measures"]
    for reading in readings:
        reading["raeding"] = reading.pop("reading")
    return payload


class TestEveryReadingSiteIsParsedByTheOneContract:
    """A key that is no field is refused at every site, and a chart's numbers are compiled by code."""

    @pytest.mark.parametrize("site", ["evidence", "chart measure"])
    async def test_a_misspelt_field_is_refused_rather_than_read_as_the_default(self, site: str) -> None:
        bundle = _bundle()
        with pytest.raises(SoundnessRefusal, match=r"raeding\s+Extra inputs are not permitted"):
            await _generate(bundle, _misspelt_reading_at(site, bundle))

    async def test_the_same_memo_spelt_correctly_is_accepted_in_one_call(self) -> None:
        bundle = _bundle()
        payload = _misspelt_reading_at("evidence", bundle)
        for reading in _finding(payload)["evidence"]:
            reading["reading"] = reading.pop("raeding")

        analysis, client = await _generate(bundle, payload)

        assert len(client.calls) == 1
        assert [row.reading for row in analysis.resolutions[0].evidence] == ["measure", "measure"]

    async def test_a_payload_typed_beside_a_chart_is_refused(self) -> None:
        """A chart carries cells and readings; a number typed into it is off-contract, never drawn."""
        bundle = _bundle()
        payload = copy.deepcopy(memo_payload(bundle))
        _finding(payload)["chart"] = {
            **_delta_chart(bundle),
            "payload": {"rows": [{"metric": "total_ms", "a": 1.0, "b": 2.0}]},
        }

        with pytest.raises(SoundnessRefusal, match=r"payload\s+Extra inputs are not permitted"):
            await _generate(bundle, payload)

    async def test_the_charts_numbers_are_the_surfaces(self) -> None:
        bundle = _bundle()
        payload = copy.deepcopy(memo_payload(bundle))
        _finding(payload)["chart"] = _delta_chart(bundle)
        narrow, wide = cell_at(bundle, TOYHOST_NARROW), cell_at(bundle, TOYHOST_WIDE)

        analysis, client = await _generate(bundle, payload)

        assert len(client.calls) == 1
        chart = analysis.resolutions[0].chart
        assert chart is not None and chart.type == "delta_table"
        (row,) = chart.payload["rows"]
        means = {
            cell_ref(c.variant_key, c.apparatus_class_id): _measure(c, "total_ms").mean for c in bundle.cell_measures
        }
        assert (row["a"], row["b"]) == (means[narrow], means[wide])


class TestProseIsNotRefused:
    """Code checks the structure of the memo, never its sentences."""

    async def test_a_memo_whose_prose_the_removed_gates_refused_is_stored_in_one_call(self) -> None:
        bundle = _bundle()
        payload = copy.deepcopy(memo_payload(bundle))
        # A headline naming a level, a rendered table in a finding's body, and a figure the bundle
        # never computed: the headline gate, the rendered-table gate and the prose fact gate each
        # would refuse one of these.
        payload["headline"] = f"{TOYHOST_NARROW}tok beats {TOYHOST_WIDE}tok on wall-clock by 99999 ms."
        _finding(payload)["body"] = "| width | ms |\n|---|---|\n| 256tok | 920 |\n| 1024tok | 1420 |"

        analysis, client = await _generate(bundle, payload)

        assert len(client.calls) == 1
        assert analysis.document.headline.startswith(f"{TOYHOST_NARROW}tok beats")
        assert "|---|" in analysis.document.findings[0].body
        assert analysis.document.findings[0].title == _finding(payload)["title"]
