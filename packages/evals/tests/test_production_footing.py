"""A production-replicating cost names what its run moved off the subject's production configuration (#571).

The figure is what production would spend only for a run that moved nothing: a swept model, an overlay
set away from its default, or an apparatus setting that strips the candidate's learned state each make it
the cost of something production does not run, and the error has no reliable sign. These tests pin the
disclosure the cost surfaces carry, read off the host's sweepable declarations alone:

- a run that set a lever names it, with its level, on every surface that shows the figure;
- a run that set nothing says so, and only when every input was checked;
- a lever a host newly declares reaches the disclosure with no cost surface edited;
- a host can mark an apparatus input as one that moves the candidate off its production footing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest
from pydantic import BaseModel, Field

from threetears.evals.analysis import DisclosureBlock, build_code_only_report
from threetears.evals.analysis.bundle import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.reads import run_summary
from threetears.evals.analysis.reporting import FrontierPoint, compute_frontier
from threetears.evals.contracts.host import SHARED_CORE, HostProfile, Sweepable
from threetears.evals.contracts.host.kinds import KindContract
from threetears.evals.contracts.host.sweepables import RegistrationError
from threetears.evals.contracts.host.values import PooledProductionFooting, ProductionFooting
from threetears.evals.contracts.models import EvalResult, EvalRun, utc_now_iso
from threetears.evals.quick import Answer, compare
from packages.evals.tests.factories import as_listed, make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload

_INHERITED = {"candidate": "inherited"}


def _payload(key: str) -> Any:
    def read(run: EvalRun, _results: Sequence[EvalResult]) -> Any:
        return (run.host_payload or {}).get("probe", {}).get(key)

    return read


class TestTheEnginesDefaultReading:
    def test_a_run_at_the_kinds_own_model_that_set_nothing_moved_nothing(self) -> None:
        footing = SHARED_CORE.production_footing(make_eval_run(model_role_provenance=_INHERITED))

        assert footing.moved_nothing
        assert footing.held == ["candidate_kind", "model"]
        assert "moved none of the 2 input(s) checked" in footing.sentence()

    def test_a_launch_that_named_its_model_names_the_model_beside_the_cost(self) -> None:
        run = make_eval_run(candidate_model="cheap-model", model_role_provenance={"candidate": "chosen"})

        footing = SHARED_CORE.production_footing(run)

        assert footing.moved == {"model": "cheap-model"}
        assert "model=cheap-model" in footing.sentence()

    def test_a_run_stored_before_the_candidate_origin_is_unchecked_not_held(self) -> None:
        """No flag must be told apart from nobody checked: a run that recorded nothing claims nothing."""
        footing = SHARED_CORE.production_footing(make_eval_run())

        assert footing.unchecked == {"model": "sonnet"}
        assert not footing.moved_nothing

    def test_a_witnessed_run_ran_at_production_by_construction(self) -> None:
        footing = SHARED_CORE.production_footing(make_eval_run(apparatus_provenance="witnessed"))

        assert footing.moved_nothing


class TestANewlyDeclaredLeverReachesTheDisclosure:
    """The Done-when: adding a lever to a toy host's registry puts it on the cost axis, no surface edited."""

    def test_a_lever_added_to_the_toy_registry_appears_when_a_run_sets_it(self) -> None:
        registry = toyhost_profile().sweepables.extend(
            (Sweepable(name="probe_depth", role="lever", read=_payload("depth"), reader_prose="probe depth"),)
        )
        run = make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"depth": 7}})

        footing = registry.production_footing(run)

        assert footing.unchecked == {"probe_depth": "7"}, "a newly declared lever never reached the disclosure"
        assert "probe_depth=7" in footing.sentence()

    def test_the_same_lever_unset_is_the_subjects_own_setting(self) -> None:
        registry = SHARED_CORE.extend(
            (Sweepable(name="probe_depth", role="lever", read=_payload("depth"), reader_prose="probe depth"),)
        )

        footing = registry.production_footing(make_eval_run(model_role_provenance=_INHERITED))

        assert "probe_depth" in footing.held and footing.moved_nothing

    def test_a_lever_declaring_its_departure_is_named_as_moved(self) -> None:
        registry = SHARED_CORE.extend(
            (
                Sweepable(
                    name="probe_depth",
                    role="lever",
                    read=_payload("depth"),
                    reader_prose="probe depth",
                    departs_production=lambda run, _r: _payload("depth")(run, _r) not in (None, 3),
                ),
            )
        )
        moved = registry.production_footing(
            make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"depth": 9}})
        )
        held = registry.production_footing(
            make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"depth": 3}})
        )

        assert moved.moved == {"probe_depth": "9"}
        assert held.moved_nothing


class TestAnApparatusInputThatMovesTheCandidate:
    def test_an_unmarked_apparatus_input_is_not_on_the_cost_axis(self) -> None:
        registry = SHARED_CORE.extend(
            (
                Sweepable(
                    name="fresh_state",
                    role="apparatus",
                    read=_payload("fresh"),
                    reader_prose="whether the candidate started with its learned state stripped",
                    confounds="a stripped candidate has no history to draw on",
                ),
            )
        )
        run = make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"fresh": True}})

        assert registry.production_footing(run).moved_nothing

    def test_a_marked_one_that_stripped_state_is_named_beside_the_cost(self) -> None:
        registry = SHARED_CORE.extend(
            (
                Sweepable(
                    name="fresh_state",
                    role="apparatus",
                    read=_payload("fresh"),
                    reader_prose="whether the candidate started with its learned state stripped",
                    confounds="a stripped candidate has no history to draw on",
                    departs_production=lambda run, _r: bool(_payload("fresh")(run, _r)),
                ),
            )
        )
        stripped = make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"fresh": True}})
        stateful = make_eval_run(model_role_provenance=_INHERITED, host_payload={"probe": {"fresh": False}})

        assert registry.production_footing(stripped).moved == {"fresh_state": "True"}
        assert registry.production_footing(stateful).moved_nothing

    def test_a_label_cannot_declare_a_departure(self) -> None:
        with pytest.raises(RegistrationError, match="departs_production"):
            SHARED_CORE.extend(
                (
                    Sweepable(
                        name="batch",
                        role="label",
                        read=_payload("batch"),
                        reader_prose="the batch",
                        departs_production=lambda _run, _r: True,
                    ),
                )
            )


class _Overlays(BaseModel):
    page_limit: int = Field(default=10, description="how many pages the candidate reads")


class TestAKindsOverlaysAreReadAgainstTheirDefaults:
    """The record holds every overlay field, defaults included, so a value is no sign the launch set it."""

    def _registry(self) -> Any:
        return SHARED_CORE.extend(KindContract("probe-kind", overlays=_Overlays).sweepables)

    def test_an_overlay_at_its_default_holds(self) -> None:
        run = make_eval_run(candidate_kind="probe-kind", overlays={"page_limit": 10}, model_role_provenance=_INHERITED)

        assert self._registry().production_footing(run).moved_nothing

    def test_an_overlay_set_off_its_default_moved(self) -> None:
        run = make_eval_run(candidate_kind="probe-kind", overlays={"page_limit": 2}, model_role_provenance=_INHERITED)

        assert self._registry().production_footing(run).moved == {"probe-kind.page_limit": "2"}


class TestTheDisclosureIsReadOffAWholeRun:
    def test_a_listed_copy_with_its_payload_elided_is_refused(self) -> None:
        run = make_eval_run(host_payload={"toyhost": {"chunk_tokens": 512}})
        (listed,) = as_listed([run], frozenset({"toyhost.chunk_tokens"}))

        with pytest.raises(ValueError, match="elided"):
            toyhost_profile().sweepables.production_footing(listed)


class TestEverySurfaceThatShowsTheFigureCarriesIt:
    """The toy campaign sweeps chunk width, which its payload records and nothing says is production's."""

    def _bundle(self) -> AnalysisContextBundle:
        profile = toyhost_profile()
        campaign, storage = toyhost_campaign(profile=profile)
        return assemble_context_bundle(campaign, storage=storage, profile=profile)

    def test_the_bundles_run_summary_names_the_swept_lever(self) -> None:
        bundle = self._bundle()

        for summary in bundle.run_summaries:
            assert summary.prod_cost_usd is not None
            assert summary.production_footing is not None
            assert "chunk_tokens" in summary.production_footing.unchecked, summary.production_footing

    async def _key_metrics(self, bundle: AnalysisContextBundle) -> list[dict[str, Any]]:
        analysis, _insights = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(memo_payload(bundle))),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=toyhost_profile(),
        )
        return [entry.key_metrics for entry in analysis.run_index]

    async def test_the_generators_key_metrics_name_it_beside_the_prod_cost(self) -> None:
        for metrics in await self._key_metrics(self._bundle()):
            assert "mean_prod_cost_usd" in metrics
            assert "chunk_tokens" in metrics["prod_cost_unchecked_against_production"]
            assert "chunk_tokens=" in metrics["prod_cost_footing"]

    async def test_a_run_that_moved_nothing_says_so_and_one_nobody_checked_does_not(self) -> None:
        bundle = self._bundle()
        first, *rest = bundle.run_summaries
        summaries = [
            first.model_copy(update={"production_footing": ProductionFooting(held=["model"])}),
            *(summary.model_copy(update={"production_footing": None}) for summary in rest),
        ]

        moved_nothing, *unchecked = await self._key_metrics(bundle.model_copy(update={"run_summaries": summaries}))

        assert "moved none" in moved_nothing["prod_cost_footing"]
        assert "prod_cost_moved_off_production" not in moved_nothing
        assert unchecked and all(m["prod_cost_footing"].startswith("nobody checked") for m in unchecked)


class TestTheScoreRollupRowCarriesIt:
    """``run_summary``'s rows are the score rollup a host renders; each prod-cost row carries the footing."""

    async def test_a_quick_comparison_row_names_the_model_its_launch_named(self) -> None:
        async def arm(case: Mapping[str, Any]) -> Answer:
            return Answer(case["label"], model="m-cheap", input_tokens=10, output_tokens=1, cost_usd=0.001)

        cases = [{"text": "a cat", "label": "animal"}, {"text": "an oak", "label": "plant"}]
        comparison = await compare(
            cases, {"a": arm, "b": arm}, expected=lambda case: str(case["label"]), control="a", scope_id="footing", k=1
        )
        host = comparison.host
        run_id = comparison.arms["a"].run_id

        summary = run_summary(
            host.storage,
            run_id,
            "footing",
            load_run_listed=lambda rid, scope: host.storage.load_eval_run(rid, scope),
            row_columns=lambda _results: {},
            profile=host.profile,
        )

        run = host.storage.load_eval_run(run_id, "footing")
        assert run is not None
        (row,) = summary["rows"]
        assert row["mean_prod_cost_usd"] is not None
        footing = row["prod_cost_footing"]
        # A quick comparison's launch names each arm's model (here the shared placeholder its arms run on), and
        # the engine cannot see that it is the subject's own, so the model is named beside the cost.
        assert footing["moved"] == {"model": run.candidate_model}
        assert footing["moved_nothing"] is False
        assert f"model={run.candidate_model}" in footing["sentence"]


def _toy_frontier(profile: HostProfile | None) -> list[FrontierPoint]:
    """The toy campaign's two arms ranked on the frontier, read against ``profile``'s declarations."""
    campaign, storage = toyhost_campaign(profile=toyhost_profile())
    runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
    results = [result for run in runs for result in storage.query_eval_results_by_run(run.id, run.scope_id)]
    (subject,) = compute_frontier(runs, results, bar=0.0, archived_run_ids=None, profile=profile).subjects
    return subject.points


def _with_probe_depth() -> HostProfile:
    """The toy host with one more lever, which every run carries at 7 and nothing says is production's."""
    profile = toyhost_profile()
    probe = Sweepable(
        name="probe_depth",
        role="lever",
        read=lambda _run, _results: 7,
        reader_prose="probe depth",
        no_own_coordinate="the toy variant map predates it; every run carries one level of it",
    )
    return replace(profile, host_sweepables=profile.host_sweepables.extend([probe]))


class TestAPooledCostCarriesEachRunsFooting:
    """A frontier point's cost axis and an analysis arm's cost pool runs, and carry every run's footing."""

    def test_each_frontier_point_names_what_its_runs_left_unchecked(self) -> None:
        for point in _toy_frontier(toyhost_profile()):
            assert point.production_replicating_cost is not None
            assert point.production_footing is not None
            footings = list(point.production_footing.runs.values())
            assert footings and all(f is not None and "chunk_tokens" in f.unchecked for f in footings)
            assert "chunk_tokens=" in point.production_footing.sentence()

    def test_the_verdict_carries_the_picks_footing(self) -> None:
        campaign, storage = toyhost_campaign()
        runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)
        results = [result for run in runs for result in storage.query_eval_results_by_run(run.id, run.scope_id)]
        (subject,) = compute_frontier(runs, results, bar=0.0, archived_run_ids=None, profile=toyhost_profile()).subjects

        assert subject.verdict is not None
        (pick,) = [p for p in subject.points if p.variant_key == subject.verdict.variant_key]
        assert subject.verdict.production_footing == pick.production_footing is not None

    def test_a_frontier_given_no_declarations_says_nobody_checked(self) -> None:
        assert all(point.production_footing is None for point in _toy_frontier(None))

    def test_each_analysis_arm_names_its_runs_footings_and_the_report_says_so(self) -> None:
        profile = toyhost_profile()
        campaign, storage = toyhost_campaign(profile=profile)
        bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
        run_ids = {summary.run_id for summary in bundle.run_summaries}

        assert bundle.arm_production_footings
        assert {run_id for pooled in bundle.arm_production_footings.values() for run_id in pooled.runs} == run_ids
        for pooled in bundle.arm_production_footings.values():
            assert all(f is not None and "chunk_tokens" in f.unchecked for f in pooled.runs.values())
        disclosures = [
            block.text
            for block in build_code_only_report(
                bundle, measures=profile.measures, assembled_at="2026-10-10T00:00:00+00:00"
            ).blocks
            if isinstance(block, DisclosureBlock)
        ]
        assert any("production-replicating cost" in text and "chunk_tokens" in text for text in disclosures)


class TestALeverAddedToTheToyHostReachesThePooledCost:
    """The Done-when on the pooled surfaces: a lever newly declared reaches them with no surface edited."""

    def test_the_frontier_names_it_only_once_it_is_declared(self) -> None:
        before = [p.production_footing for p in _toy_frontier(toyhost_profile())]
        after = [p.production_footing for p in _toy_frontier(_with_probe_depth())]

        assert all(f is not None and "probe_depth" not in f.sentence() for f in before)
        for footing in after:
            assert footing is not None
            assert all(f is not None and f.unchecked.get("probe_depth") == "7" for f in footing.runs.values())
            assert "probe_depth=7" in footing.sentence()

    def test_the_analysis_arms_name_it_too(self) -> None:
        profile = _with_probe_depth()
        campaign, storage = toyhost_campaign(profile=profile)
        bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)

        assert bundle.arm_production_footings
        for pooled in bundle.arm_production_footings.values():
            assert "probe_depth=7" in pooled.sentence()


class TestAPooledFootingThatMovedNothingSaysSo:
    def test_runs_that_all_moved_nothing_say_so(self) -> None:
        pooled = PooledProductionFooting(runs={"r1": ProductionFooting(held=["model"]), "r2": ProductionFooting()})

        assert pooled.moved_nothing
        assert "moved none" in pooled.sentence()

    def test_a_run_nobody_checked_is_never_moved_nothing(self) -> None:
        pooled = PooledProductionFooting(runs={"r1": ProductionFooting(held=["model"]), "r2": None})

        assert not pooled.moved_nothing
        assert "nobody checked run(s) r2" in pooled.sentence()

    def test_runs_at_different_footings_are_named_apart(self) -> None:
        pooled = PooledProductionFooting(
            runs={"r1": ProductionFooting(held=["model"]), "r2": ProductionFooting(moved={"model": "cheap"})}
        )

        sentence = pooled.sentence()
        assert not pooled.moved_nothing
        assert "2 different production footings" in sentence
        assert "run(s) r2: this run set model=cheap" in sentence
