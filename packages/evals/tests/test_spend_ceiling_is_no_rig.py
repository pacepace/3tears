"""The spend ceiling is recorded and disclosed, and is no dimension of the rig.

A cap decides when a run may stop, never what a measured case scored, so two launches that differ only in the cap
they were given are one test setup: their runs share an apparatus class, pool in one cell per arm and pair, and no
confound list names the cap. The core still declares ``max_cost_usd`` as apparatus, because kinds seat it, and the
bisection still reports a difference in it as a plain fact; what takes it out of the rig is
``Sweepable.no_rig_coordinate``. A real apparatus dimension (who judged) still splits the rig.

The quick path's runs carry it too: an ``Eval`` shares its cap equally between its arms, so the same arm run in a
comparison of two arms and in one of three is capped at a half and at a third of the same figure.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping
from typing import Any

import pytest

from threetears.evals import Answer, Eval, Result
from threetears.evals.analysis import (
    AnalysisContextBundle,
    assemble_context_bundle,
    aspect_view,
    campaign_readings,
    get_campaign,
)
from threetears.evals.analysis.depth import Depth
from threetears.evals.analysis.reads import bisect_runs
from threetears.evals.kernel import EvalStorage
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import CampaignArm
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.sweepables import (
    CORE_SWEEPABLES,
    RegistrationError,
    RolePins,
    Sweepable,
    SweepableRegistry,
)
from threetears.evals.run import get_run
from threetears.evals.schema.models import EvalRun
from threetears.evals.storage import InMemoryDocumentStore

from packages.evals.tests.factories import make_aspect_campaign, make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop

#: The cases each arm runs, and the repeats of each.
CASES = 20
REPEATS = 3

#: The two caps the launches were given, per arm.
_LOW_CAP, _HIGH_CAP = 0.75, 3.0

#: The campaign's question, asked before every run.
_QUESTION = "Does the candidate pass more support requests than production?"
_ASKED_AT = "2000-01-01T00:00:00+00:00"


def _passing(right: int) -> Callable[[Mapping[str, Any]], Coroutine[Any, Any, Answer]]:
    """A runner answering right on every trial of the first ``right`` cases and wrong on the rest, at 0.2 cents."""

    async def run(case: Mapping[str, Any]) -> Answer:
        index = int(case["text"].rsplit(" ", 1)[1])
        return Answer(
            case["label"] if index < right else "skipped", model="m", input_tokens=100, output_tokens=10, cost_usd=0.002
        )

    return run


_CASES = [{"text": f"support request {index}", "label": "queued"} for index in range(CASES)]


async def _launch(store: InMemoryDocumentStore, arms: Mapping[str, int], *, max_cost_usd: float) -> Result:
    """One launch of the eval over :data:`_CASES` into ``store``: each arm passing its first so many cases, capped as
    given. Every launch on one store runs on one template, since an eval's template is named by its cases."""
    runners = {name: _passing(right) for name, right in arms.items()}
    return await Eval(_CASES, runners, ["label"], k=REPEATS, name="support", max_cost_usd=max_cost_usd).run(store=store)


def _campaign(host: EvalHost, launched: Result, run_ids: list[str]) -> EvalCampaign:
    """``launched``'s campaign, holding ``run_ids`` (from any launch on its template), its question asked first."""
    view = aspect_view(get_campaign(host.storage, launched.campaign_id, launched.scope_id))
    declared = view.declared
    assert declared is not None and declared.control is not None
    return make_aspect_campaign(
        label="support",
        template_id=view.template_id,
        run_ids=run_ids,
        design=declared.design.model_copy(update={"merit_axes": []}),
        question=_QUESTION,
        asked_at=_ASKED_AT,
        control=declared.control.model_copy(update={"named_at": _ASKED_AT, "label": "production"}),
        margin=10,
        id=view.campaign_id,
        scope_id=view.scope_id,
        name=view.name,
        subject_id=view.subject_id,
        subject_kind=view.subject_kind,
        behavior=view.behavior,
    )


def _bundle(host: EvalHost, campaign: EvalCampaign) -> AnalysisContextBundle:
    return assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)


def _key(bundle: AnalysisContextBundle, arm: str) -> str:
    return next(entry.variant_key for entry in bundle.variant_index if entry.levers["candidate"].display == arm)


def _named(bundle: AnalysisContextBundle) -> set[str]:
    """Every dimension any confound list, the catalog or the aliasing read of ``bundle`` names."""
    named = {confound.dimension for confound in bundle.apparatus_confounds}
    named |= {confound.dimension for entry in bundle.coverage for confound in entry.confounded_by}
    named |= {confound.dimension for divergence in bundle.scope_divergences for confound in divergence.confounded_by}
    named |= {factor for group in bundle.aliased_factors for factor in group.factors}
    return named | set(bundle.confound_catalog)


class TestTwoLaunchesThatDifferOnlyInTheirCapAreOneTestSetup:
    async def test_the_control_and_the_candidate_pair_and_read_an_interval(self) -> None:
        """The reported case: production launched at 75 cents, the candidate at 3 dollars, on one template."""
        store = InMemoryDocumentStore()
        # Each launch's cap is shared between its two arms: 1.50 is 0.75 an arm, 6.00 is 3.00.
        low = await _launch(store, {"production": CASES, "candidate": 6}, max_cost_usd=2 * _LOW_CAP)
        high = await _launch(store, {"production": CASES, "candidate": 6}, max_cost_usd=2 * _HIGH_CAP)
        host = low.host
        control_run, candidate_run = low.arms["production"].run_id, high.arms["candidate"].run_id
        caps = {
            off_loop(get_run, host.storage, run_id, low.scope_id).max_cost_usd
            for run_id in (control_run, candidate_run)
        }
        assert caps == {_LOW_CAP, _HIGH_CAP}, "the two runs must differ in their cap"
        campaign = off_loop(_campaign, host, low, [control_run, candidate_run])
        bundle = off_loop(_bundle, host, campaign)

        (rig,) = {cell.apparatus_class_id for cell in bundle.cells}
        assert len(bundle.cells) == 2, "one cell per arm, both under the one rig"
        assert "max_cost_usd" not in _named(bundle)
        assert campaign.control is not None
        arms = [campaign.control, CampaignArm(label="v2", role="candidate", variant_key=_key(bundle, "candidate"))]
        (row,) = campaign_readings(campaign, {"support": bundle}, arms=arms).readings
        assert "no test setup in common" not in row.reason
        # Production passes all twenty cases and the candidate the first six: 30% - 100% is -70 points.
        assert row.change == pytest.approx(-70.0)
        assert row.reading == "worse"
        assert row.drop_interval is not None and row.drop_interval[0] <= -70.0 <= row.drop_interval[1]
        assert rig


class TestTheQuickPathsPerArmShareSplitsNothing:
    async def test_one_arm_capped_at_a_half_and_at_a_third_of_one_figure_is_one_cell(self) -> None:
        store = InMemoryDocumentStore()
        # A dollar between two arms is 50 cents an arm; between three, a third of a dollar.
        pair = await _launch(store, {"production": CASES, "candidate": 6}, max_cost_usd=1.0)
        trio = await _launch(store, {"production": CASES, "candidate": 6, "v3": 12}, max_cost_usd=1.0)
        host = pair.host
        run_ids = [summary.run_id for result in (pair, trio) for name, summary in result.arms.items() if name != "v3"]
        runs = [off_loop(get_run, host.storage, run_id, pair.scope_id) for run_id in run_ids]
        assert sorted({run.max_cost_usd for run in runs}) == pytest.approx([1.0 / 3, 0.5])
        bundle = off_loop(_bundle, host, off_loop(_campaign, host, pair, run_ids))

        assert len({cell.apparatus_class_id for cell in bundle.cells}) == 1
        # Two arms, each run twice: every cell pools both runs' twenty cases at three repeats.
        assert sorted(cell.variant_key for cell in bundle.cells) == sorted(
            {_key(bundle, "production"), _key(bundle, "candidate")}
        )
        assert [cell.n_observations for cell in bundle.cells] == [2 * CASES * REPEATS] * 2
        assert "max_cost_usd" not in _named(bundle)


def _toy_bundle(*updates: dict[str, object], grader_versions: tuple[str, str] | None = None) -> AnalysisContextBundle:
    """The toy campaign's bundle, its two runs' fields rewritten as ``updates`` in order."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile, grader_versions=grader_versions)
    for run_id, update in zip(campaign.run_ids, updates, strict=True):
        run: EvalRun | None = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert run is not None
        for field, value in update.items():
            setattr(run, field, value)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _capped(cap: float) -> dict[str, object]:
    return {"max_cost_usd": cap, "max_cost_usd_origin": "chosen"}


class TestTheConfoundScanNamesNoCapDifference:
    def test_the_toy_arms_at_two_caps_share_the_rig_of_the_toy_arms_at_one(self) -> None:
        alike = _toy_bundle(_capped(_LOW_CAP), _capped(_LOW_CAP))
        apart = _toy_bundle(_capped(_LOW_CAP), _capped(_HIGH_CAP))

        assert len({cell.apparatus_class_id for cell in apart.cells}) == 1
        assert {cell.apparatus_class_id for cell in apart.cells} == {cell.apparatus_class_id for cell in alike.cells}
        assert "max_cost_usd" not in _named(apart)
        assert apart.apparatus_confounds == alike.apparatus_confounds

    def test_a_judge_that_differs_still_splits_the_rig_and_is_named_beside_a_cap_that_is_not(self) -> None:
        """The toy host's grader version is pinned into the engine's judge role: a real dimension of the rig."""
        judged_apart = _toy_bundle(_capped(_LOW_CAP), _capped(_HIGH_CAP), grader_versions=("narrow-v1", "wide-v2"))

        assert len({cell.apparatus_class_id for cell in judged_apart.cells}) == 2
        named = {confound.dimension for confound in judged_apart.apparatus_confounds}
        assert named and "max_cost_usd" not in named


class TestTheBisectionStillReportsTheCapAsAFact:
    def test_two_caps_differ(self) -> None:
        storage = EvalStorage(InMemoryDocumentStore())
        run_a = make_eval_run(id="run-a", **_capped(_LOW_CAP))
        run_b = make_eval_run(id="run-b", **_capped(_HIGH_CAP))
        for run in (run_a, run_b):
            storage.save_eval_run(run)

        bisected = bisect_runs(storage, run_a.id, run_b.id, run_a.scope_id, profile=toyhost_profile())

        assert "max_cost_usd" in bisected["differs"]
        assert bisected["details"]["max_cost_usd"] == {"a": _LOW_CAP, "b": _HIGH_CAP}


def _apparatus(**fields: Any) -> Sweepable:
    return Sweepable(
        name="bench_lamp",
        role=fields.pop("role", "apparatus"),
        read=lambda _run, _results: "on",
        reader_prose="whether the bench lamp was lit",
        **fields,
    )


class TestNoRigCoordinateIsRegisteredOnApparatusOnly:
    def test_the_core_spend_ceiling_is_apparatus_out_of_the_rig(self) -> None:
        (declared,) = [d for d in CORE_SWEEPABLES if d.name == "max_cost_usd"]
        assert declared.role == "apparatus"
        assert declared.no_rig_coordinate
        assert declared.confounds is None

    def test_an_apparatus_input_out_of_the_rig_needs_no_confound_reason(self) -> None:
        registry = SweepableRegistry([_apparatus(no_rig_coordinate="the lamp lights the bench, never the scan")])
        assert registry.names == ("bench_lamp",)

    def test_one_that_also_states_a_confound_reason_is_refused(self) -> None:
        with pytest.raises(RegistrationError, match="declares no_rig_coordinate and a confound reason"):
            SweepableRegistry([_apparatus(no_rig_coordinate="never the scan", confounds="the scan reads otherwise")])

    def test_a_blank_reason_is_refused(self) -> None:
        with pytest.raises(RegistrationError, match="declares no_rig_coordinate blank"):
            SweepableRegistry([_apparatus(no_rig_coordinate="  ")])

    @pytest.mark.parametrize("role", ["lever", "label"])
    def test_a_lever_or_a_label_declaring_it_is_refused(self, role: str) -> None:
        with pytest.raises(RegistrationError, match=f"is a {role} and declares no_rig_coordinate"):
            SweepableRegistry([_apparatus(role=role, no_rig_coordinate="never the scan")])

    def test_a_role_pin_declaring_it_is_refused(self) -> None:
        """A role pin names who filled a seat in the rig, so taking it out of the rig contradicts the pin."""
        lamp = _apparatus(no_rig_coordinate="the lamp lights the bench, never the scan")
        with pytest.raises(RegistrationError, match="is named as a role pin but declares no_rig_coordinate"):
            SweepableRegistry([lamp], roles=[RolePins(name="bench", pins=("bench_lamp",))])

    def test_only_a_rig_dimension_is_one(self) -> None:
        inside = _apparatus()
        outside = _apparatus(no_rig_coordinate="the lamp lights the bench, never the scan")

        assert inside.rig_dimension and not outside.rig_dimension
        assert SweepableRegistry([outside]).rig_dimensions == ()


class TestALowCapStopsItsArmShortAndTheReadingsDepthShowsIt:
    async def test_a_budget_stopped_arm_pools_with_a_higher_cap_arm_and_states_its_truncated_depth(self) -> None:
        store = InMemoryDocumentStore()
        # Each trial costs 0.2 cents, so a whole arm of 20 cases at 3 repeats costs 12 cents. The low launch gives each
        # arm 5 cents, which stops it part-way; the high launch gives each arm 3 dollars, which never binds.
        low = await _launch(store, {"production": CASES, "candidate": 6}, max_cost_usd=2 * 0.05)
        high = await _launch(store, {"production": CASES, "candidate": 6}, max_cost_usd=2 * _HIGH_CAP)
        host = low.host
        control_run, candidate_run = high.arms["production"].run_id, low.arms["candidate"].run_id
        stopped = off_loop(get_run, host.storage, candidate_run, low.scope_id)
        assert stopped.status == "budget_stopped"
        campaign = off_loop(_campaign, host, low, [control_run, candidate_run])
        bundle = off_loop(_bundle, host, campaign)

        assert len({cell.apparatus_class_id for cell in bundle.cells}) == 1, "the two caps are one rig"
        assert campaign.control is not None
        arms = [campaign.control, CampaignArm(label="v2", role="candidate", variant_key=_key(bundle, "candidate"))]
        (row,) = campaign_readings(campaign, {"support": bundle}, arms=arms).readings
        assert row.control_depth == Depth(
            cases=CASES, trials=CASES * REPEATS, repeats_min=REPEATS, repeats_median=float(REPEATS), repeats_max=REPEATS
        )
        depth = row.arm_depth
        assert depth is not None
        # The cap stopped the candidate short of its matrix: fewer trials, so some case ran short or not at all.
        assert depth.trials < CASES * REPEATS
        assert depth.cases < CASES or depth.repeats_min < REPEATS
