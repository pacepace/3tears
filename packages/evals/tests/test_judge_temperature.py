"""#633 (owner ruling): every judge call is requested at temperature 0 unless a config says otherwise, and the
temperature actually sent is part of the judge's identity.

Before, a dimension with a ``JudgeConfig`` was judged at its 0.0 and one without at the provider's default (around
1.0 on some), in one run, because nobody chose otherwise. Now:

* a dimension with no config is requested at :data:`DEFAULT_JUDGE_TEMPERATURE`, the config's own default;
* every score records what its call was SENT at, as the client reports it — ``model_default`` for a model that
  refuses a temperature and was sent none, ``None`` when the client reports nothing (not recorded);
* a different temperature is a different judge: it never shares an agreement group or a tier, it moves the
  ``judge_temperature`` apparatus input, a repeat sampled at another temperature is not paired, and a run's
  requested temperature is in its measurement context — a run that recorded none is partial, never equal.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.agreement import judge_agreement, judge_self_agreement
from threetears.evals.contracts import ValidationFailedError
from threetears.evals.contracts.host.sweepables import CORE_ROLES, CORE_SWEEPABLES, SHARED_CORE, SweepableRegistry
from threetears.evals.contracts.identity import derive_context_identity
from threetears.evals.contracts.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    EvalRun,
    MODEL_DEFAULT_TEMPERATURE,
    JudgeConfig,
    JudgedArtifact,
    JudgeEvidence,
    JudgeRepeat,
    RepeatedScore,
    RubricDim,
    RubricScore,
)
from threetears.evals.contracts.provider import withhold_failure_detail
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeContext, JudgeService
from threetears.evals.run.rejudge import recorded_judge_pins
from packages.evals.tests.factories import make_calibration_rating, make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_ROLES, TOYHOST_SWEEPABLES

_TONE = "conversation.tone"

#: The client reports a temperature attribute only when this is not ``_UNREPORTED``.
_UNREPORTED = object()


def _dim_of(system: str) -> str:
    match = re.search(r'the single key "(.+?)"', system)
    return match.group(1) if match else "?"


class _Client:
    """A judge client that reports, on each completion, the temperature it says its request carried."""

    def __init__(self, sent: object) -> None:
        self._sent = sent

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        fields: dict[str, Any] = {
            "content": json.dumps({"reasoning": "ok", "criteria_scores": {_dim_of(system): 4}}),
            "served_model": "judge-a",
            "stop_reason": "end_turn",
            "input_tokens": 1,
            "output_tokens": 1,
            "cost_usd": 0.0,
            "model": "judge-a",
        }
        if self._sent is not _UNREPORTED:
            fields["temperature"] = self._sent
        return SimpleNamespace(**fields)


def _service(*, configs: dict[str, JudgeConfig] | None = None, refuses: bool = False, reports: bool = True):
    """A judge service whose clients report what they were asked for — or none, for a model that refuses one."""
    asked: list[tuple[str | None, float | None]] = []

    def factory(model: str | None, temperature: float | None) -> _Client:
        asked.append((model, temperature))
        return _Client(_UNREPORTED if not reports else None if refuses else temperature)

    return JudgeService(client_factory=factory, configs=configs, failure_describer=withhold_failure_detail), asked


def _context() -> JudgeContext:
    return JudgeContext(
        case_id="tc-1",
        intent="answer politely",
        variation={},
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        judge_evidence=JudgeEvidence(subject=None, case_material="m", artifact="Candidate: hello"),
    )


def _dim() -> RubricDim:
    return RubricDim(name=_TONE, description="is it polite", scale="ordinal")


class TestEveryDimIsRequestedAtZeroUnlessAConfigSaysOtherwise:
    async def test_a_dim_with_no_config_is_requested_at_the_configs_own_default(self) -> None:
        service, asked = _service()
        outcome = await service.score_dimension(_dim(), _context())
        assert asked == [(None, DEFAULT_JUDGE_TEMPERATURE)]
        assert DEFAULT_JUDGE_TEMPERATURE == JudgeConfig.model_fields["temperature"].default == 0.0
        assert outcome.score is not None and outcome.score.judge_temperature == 0.0

    async def test_a_configured_and_an_unconfigured_dim_are_sampled_alike_by_default(self) -> None:
        config = JudgeConfig(scope_id="s", name="tone", rubric_dim_id=_TONE, prompt_template="Judge the tone.")
        service, asked = _service(configs={_TONE: config})
        await service.score_dimension(_dim(), _context())
        await service.score_transcript(_context())
        assert {temperature for _model, temperature in asked} == {0.0}

    async def test_a_config_that_says_otherwise_is_honoured_and_recorded(self) -> None:
        config = JudgeConfig(
            scope_id="s", name="tone", rubric_dim_id=_TONE, prompt_template="Judge the tone.", temperature=0.7
        )
        service, asked = _service(configs={_TONE: config})
        outcome = await service.score_dimension(_dim(), _context())
        assert asked == [(None, 0.7)]
        assert outcome.score is not None and outcome.score.judge_temperature == 0.7


class TestTheTemperatureRecordedIsWhatWasSent:
    async def test_a_model_refusing_a_temperature_is_recorded_as_sent_none(self) -> None:
        service, _asked = _service(refuses=True)
        outcome = await service.score_dimension(_dim(), _context())
        assert outcome.score is not None and outcome.score.judge_temperature == MODEL_DEFAULT_TEMPERATURE

    async def test_a_client_reporting_nothing_is_not_recorded_never_the_request(self) -> None:
        service, _asked = _service(reports=False)
        outcome = await service.score_dimension(_dim(), _context())
        assert outcome.score is not None and outcome.score.judge_temperature is None


def _scored(result_id: str, score: int, temperature: float | str | None) -> Any:
    return make_eval_result(
        id=result_id,
        rubric_scores=[
            RubricScore(dim=_TONE, scale="ordinal", score=score, served_model="judge-a", judge_temperature=temperature)
        ],
    )


class TestADifferentTemperatureIsADifferentJudge:
    def test_calibration_never_pools_two_temperatures(self) -> None:
        results = [_scored("cold", 4, 0.0), _scored("warm", 4, 1.0), _scored("old", 4, None)]
        ratings = [make_calibration_rating(result_id=result.id, rubric_dim=_TONE, score=4) for result in results]
        groups = judge_agreement(ratings, results).dimensions
        assert sorted((group.judge_temperature is None, str(group.judge_temperature), group.n) for group in groups) == [
            (False, "0.0", 1),
            (False, "1.0", 1),
            # Not recorded is a group of its own: never a match for a recorded temperature.
            (True, "None", 1),
        ]

    @pytest.mark.parametrize(("first", "again"), [(0.0, 1.0), (None, 0.0), (0.0, MODEL_DEFAULT_TEMPERATURE)])
    def test_a_repeat_sampled_at_another_temperature_is_not_paired(self, first: Any, again: Any) -> None:
        repeat = JudgeRepeat(
            judge_model="judge-a",
            scores=[
                RepeatedScore(
                    dim=_TONE,
                    scale="ordinal",
                    first_score=4,
                    first_served_model="judge-a",
                    first_judge_config_id=None,
                    first_judge_temperature=first,
                    repeat=RubricScore(
                        dim=_TONE, scale="ordinal", score=4, served_model="judge-a", judge_temperature=again
                    ),
                )
            ],
            judge_config_ids={},
        )
        read = judge_self_agreement([_scored("r", 4, first).model_copy(update={"judge_repeats": [repeat]})])
        assert [(u.result_id, u.reason) for u in read.unpaired] == [("r", "temperature_changed")]
        assert read.dimensions == []

    def test_the_apparatus_input_reads_what_was_sent(self) -> None:
        run = make_eval_run(judge_model="judge-a")
        (declared,) = [d for d in SHARED_CORE.declarations if d.name == "judge_temperature"]
        cold = declared.read(run, [_scored("a", 4, 0.0)])
        refused = declared.read(run, [_scored("b", 4, MODEL_DEFAULT_TEMPERATURE)])
        unrecorded = declared.read(run, [_scored("c", 4, 0.0), _scored("d", 4, None)])
        assert (cold, refused, unrecorded) == (["0.0"], [MODEL_DEFAULT_TEMPERATURE], None)
        assert SHARED_CORE.comparability("judge_temperature", [cold, refused]) == "differs"
        assert SHARED_CORE.comparability("judge_temperature", [cold, ["0.0"]]) == "same"
        assert SHARED_CORE.comparability("judge_temperature", [cold, unrecorded]) == "unknown"


def _judged(**overrides: Any) -> EvalRun:
    """A judged run as today's launch records it: its judge, simulator and request settings all stamped."""
    return make_eval_run(
        judge_model="judge-a", simulator_model="sim-a", judge_request_settings=JUDGE_REQUEST_SETTINGS, **overrides
    )


class TestTheRequestedTemperatureIsInTheMeasurementContext:
    def test_two_runs_judged_at_different_temperatures_are_two_conditions(self) -> None:
        profile = toyhost_profile()
        cold = derive_context_identity(_judged(judge_temperature=0.0), profile)
        warm = derive_context_identity(_judged(judge_temperature=1.0), profile)
        assert not cold.partial and not warm.partial
        assert cold.context_key != warm.context_key
        assert cold.context_components.roles != warm.context_components.roles

    def test_a_judged_run_that_recorded_none_is_partial_never_equal(self) -> None:
        profile = toyhost_profile()
        cold = derive_context_identity(_judged(judge_temperature=0.0), profile)
        old = derive_context_identity(_judged(judge_temperature=None), profile)
        assert old.partial and "roles" in old.missing_components
        assert old.context_key != cold.context_key

    def test_an_unjudged_run_has_no_judge_temperature_to_record(self) -> None:
        identity = derive_context_identity(make_eval_run(simulator_model="sim-a"), toyhost_profile())
        assert not identity.partial


class TestARunJudgedAtAnotherTemperatureIsNotReJudgedAtToday:
    def test_a_run_that_recorded_no_temperature_is_refused(self) -> None:
        run = _judged(judge_temperature=None)
        with pytest.raises(ValidationFailedError, match="recorded no judge temperature"):
            recorded_judge_pins(run, request_settings="today")

    def test_a_run_judged_at_another_temperature_is_refused(self) -> None:
        with pytest.raises(ValidationFailedError, match="temperature 0.7"):
            recorded_judge_pins(_judged(judge_temperature=0.7), request_settings="today")

    def test_a_run_judged_at_todays_temperature_is_reproduced(self) -> None:
        run = _judged()
        assert run.judge_temperature == DEFAULT_JUDGE_TEMPERATURE
        assert recorded_judge_pins(run, request_settings="today") == "judge-a"


def _profile_before_the_temperature_joined() -> Any:
    """The toy profile as it was before ``judge_temperature`` joined the core apparatus."""
    core = tuple(declared for declared in CORE_SWEEPABLES if declared.name != "judge_temperature")
    roles = tuple(
        replace(role, pins=tuple(pin for pin in role.pins if pin != "judge_temperature")) for role in CORE_ROLES
    )
    registry = SweepableRegistry(core, roles=roles).extend(TOYHOST_SWEEPABLES, roles=TOYHOST_ROLES)
    return replace(toyhost_profile(), host_sweepables=registry)


def _judging(profile: Any) -> Any:
    """``profile`` with its toy kind seating the judge — the toy host grades with code and seats none of its own."""
    return replace(profile, kinds=tuple(replace(kind, seats=kind.seats | {"judge"}) for kind in profile.kinds))


def _cells(bundle: Any) -> set[tuple[str, str]]:
    return {(cell.variant_key, cell.apparatus_class_id) for cell in bundle.cells}


def _with_temperature(campaign: Any, storage: Any, temperature: float, *, runs: int | None = None) -> Any:
    """The toy campaign's store, with every score of its first ``runs`` runs (all, by default) sent at ``temperature``."""
    loaded = storage.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    sent = {run.id for run in loaded[: len(loaded) if runs is None else runs]}

    def stamped(result: Any) -> Any:
        if result.eval_run_id not in sent:
            return result
        scores = [score.model_copy(update={"judge_temperature": temperature}) for score in result.rubric_scores]
        return result.model_copy(update={"rubric_scores": scores})

    return ToyhostStorage(
        loaded,
        {run.id: [stamped(r) for r in storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)] for run in loaded},
    )


def _judged_campaign() -> tuple[Any, ToyhostStorage]:
    """The toy campaign with its runs judged by a named judge, as stored before temperatures were recorded."""
    campaign, storage = toyhost_campaign()
    loaded = storage.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    judged = [run.model_copy(update={"judge_model": "toy-judge", "judge_config_ids": {}}) for run in loaded]
    results = {run.id: storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE) for run in loaded}
    return campaign, ToyhostStorage(judged, results)


class TestAStoredCellKeepsItsIdAcrossTheTemperatureJoining:
    """A stored analysis cites ``<variant_key>:<apparatus_class_id>``; a dimension its runs never recorded must not move it."""

    def test_a_judged_runs_unrecorded_temperature_leaves_its_cell_id_where_it_was(self) -> None:
        campaign, storage = _judged_campaign()
        before = assemble_context_bundle(
            campaign, storage=storage, profile=_judging(_profile_before_the_temperature_joined())
        )
        after = assemble_context_bundle(campaign, storage=storage, profile=_judging(toyhost_profile()))
        assert all("judge_temperature" in cell.unknown_dimensions for cell in after.cells), (
            "the runs must be judged with the temperature unrecorded, or this asserts nothing"
        )
        assert _cells(after) == _cells(before)

    def test_an_unjudged_runs_unseated_temperature_leaves_its_cell_id_where_it_was(self) -> None:
        assert _cells(toyhost_bundle()) == _cells(toyhost_bundle(profile=_profile_before_the_temperature_joined()))

    def test_a_recorded_temperature_is_a_cell_of_its_own(self) -> None:
        campaign, storage = toyhost_campaign()
        recorded = assemble_context_bundle(
            campaign, storage=_with_temperature(campaign, storage, 0.0), profile=toyhost_profile()
        )
        assert _cells(recorded).isdisjoint(_cells(toyhost_bundle()))

    def test_a_recorded_and_an_unrecorded_run_of_one_arm_never_pool(self) -> None:
        campaign, storage = _judged_campaign()
        loaded = storage.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
        # Both runs of the first arm: a re-run judged at 0 beside the stored one that recorded nothing.
        rerun = loaded[0].model_copy(update={"id": "0f6a4c2e-9b1d-4e8f-a3c5-7d2b1e9f4a60"})
        results = {run.id: storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE) for run in loaded}
        results[rerun.id] = [
            result.model_copy(
                update={
                    "id": f"{result.id}-rerun",
                    "eval_run_id": rerun.id,
                    "rubric_scores": [s.model_copy(update={"judge_temperature": 0.0}) for s in result.rubric_scores],
                }
            )
            for result in results[loaded[0].id]
        ]
        mixed = ToyhostStorage([*loaded, rerun], results)
        bundle = assemble_context_bundle(
            campaign.model_copy(update={"run_ids": [*campaign.run_ids, rerun.id]}),
            storage=mixed,
            profile=_judging(toyhost_profile()),
        )
        arm = results[loaded[0].id][0].variant_key
        assert len({cell.apparatus_class_id for cell in bundle.cells if cell.variant_key == arm}) == 2
        assert any(merge.variant_key == arm for merge in bundle.refused_merges)
        assert any(c.dimension == "judge_temperature" and c.status == "undecided" for c in bundle.apparatus_confounds)
