"""Stored judge profiles: recorded from a judge campaign, read by a second campaign that judge scored (#628).

The deliverable's done-condition: a judge campaign's measurement of a judge is recorded as a profile per (judge,
criterion); a second campaign scored by that judge, whose own evidence decides no tier, reads its tier from the
profile and says so — and changing the judge's prompt (its config) or its temperature, or rewording the criterion,
makes it read none.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import engine_actions
from threetears.evals.analysis import (
    JudgeKey,
    assemble_context_bundle,
    judge_kind_readings,
    judge_profiles_of,
    judged_criteria,
)
from threetears.evals.analysis.report import DisclosureBlock, build_code_only_report
from threetears.evals.kernel import (
    CALIBRATION_MIN_RESULTS,
    JUDGE_KIND,
    EvalJudgeProfile,
    JudgeCaseLabel,
    JudgeTrial,
    NotFoundError,
    ValidationFailedError,
    judge_criterion_digest,
)
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import JudgeProfilesRecord, judge_profiles_list, judge_profiles_record
from threetears.evals.run import rate_result
from threetears.evals.schema import DEFAULT_JUDGE_TEMPERATURE, EvalResult, EvalTemplate, RubricDim, RubricScore
from threetears.evals.storage import InMemoryDocumentStore
from threetears.evals.kernel.storage import EvalStorage
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The judge both campaigns are scored by: a named model, the built-in prompt, the default temperature.
_JUDGE = "judge/model-a"

#: How many frozen cases the judge campaign measured: enough labelled cases to decide calibration.
_CASES = 30

#: The criterion as the second campaign's template words it, and as the judge campaign's cases froze it.
_CRITERION = RubricDim(
    name=TOYHOST_JUDGED_DIMENSION,
    description="The extracted record keeps the invoice's layout: fields in their sections, in order.",
    scale="ordinal",
)

#: Before every toy run was created, so the template is the one they were judged against.
_TEMPLATE_AT = "2026-03-01T00:00:00+00:00"


def _template(criterion: RubricDim = _CRITERION, *, updated_at: str = _TEMPLATE_AT) -> EvalTemplate:
    return make_template(
        id="toy-judged",
        scope_id=TOYHOST_SCOPE,
        rubric=[criterion],
        created_at=_TEMPLATE_AT,
        updated_at=updated_at,
    )


def _trial(case: int, *, model: str | None = _JUDGE, agree: bool = True) -> dict[str, Any]:
    """One judge trial of case ``case``: the judge scores the label's score (or one off it) at the default temperature."""
    label = 2 + case % 4
    score = label if agree else (5 if label < 4 else 1)
    return JudgeTrial(
        test_case_id=f"jc-{case}",
        dim=TOYHOST_JUDGED_DIMENSION,
        scale="ordinal",
        criterion_digest=judge_criterion_digest(TOYHOST_JUDGED_DIMENSION, "ordinal", _CRITERION),
        case_digest=f"digest-{case}",
        source_result_id=f"src-{case}",
        labels=[JudgeCaseLabel(rater="ana", score=label, reason="read it", rating_id=f"rating-{case}")],
        outcome="scored",
        score=score,
        reasoning="read against the invoice",
        served_model=model,
        judge_temperature=DEFAULT_JUDGE_TEMPERATURE,
    ).model_dump(mode="json")


def _judge_campaign(host: EvalHost, *, cases: int = _CASES, model: str | None = _JUDGE, agree: bool = True) -> str:
    """A finished judge-kind run over ``cases`` frozen cases, each asked twice; returns its id."""
    run = make_eval_run(
        id="judge-arm",
        scope_id=TOYHOST_SCOPE,
        template_id="judge-template",
        candidate_kind=JUDGE_KIND,
        candidate_model=_JUDGE,
        k_runs=2,
        test_case_ids=[f"jc-{case}" for case in range(cases)],
        status="completed",
    )
    host.storage.save_eval_run(run)
    for case in range(cases):
        for repeat in (1, 2):
            host.storage.save_eval_result(
                make_eval_result(
                    id=f"trial-{case}-{repeat}",
                    scope_id=TOYHOST_SCOPE,
                    eval_run_id=run.id,
                    test_case_id=f"jc-{case}",
                    k_iteration=repeat,
                    candidate_kind=JUDGE_KIND,
                    model=_JUDGE,
                    rubric_scores=[],
                    goal_state_outcomes=[],
                    kind_payload=_trial(case, model=model, agree=agree),
                    scored_at=f"2026-10-0{repeat}T12:00:00+00:00",
                )
            )
    return run.id


def _recorded_profile(**judge_campaign: Any) -> EvalJudgeProfile:
    """The one profile a judge campaign records, through the operation every surface calls."""
    host = toyhost_host()
    run_id = _judge_campaign(host, **judge_campaign)
    recording = judge_profiles_record(host, JudgeProfilesRecord(judge_run_ids=[run_id]), TOYHOST_SCOPE)
    (entry,) = recording.profiles
    return entry.profile


def _scored_by(
    result: EvalResult,
    *,
    model: str = _JUDGE,
    temperature: float = DEFAULT_JUDGE_TEMPERATURE,
    config: str | None = None,
) -> EvalResult:
    """``result`` with its judged score served by ``model`` at ``temperature``, asked by ``config``."""
    scores = [
        RubricScore(**{**score.model_dump(), "served_model": model, "judge_temperature": temperature})
        if score.dim == TOYHOST_JUDGED_DIMENSION
        else score
        for score in result.rubric_scores
    ]
    configs = {TOYHOST_JUDGED_DIMENSION: config} if config is not None else {}
    return result.model_copy(update={"rubric_scores": scores, "judge_config_ids": configs})


def _second_campaign(
    profile: EvalJudgeProfile | None,
    *,
    template: EvalTemplate | None = None,
    rated: int = 0,
    **judge: Any,
) -> Any:
    """The toy campaign, judged by ``judge`` against ``template``, assembled over a store holding ``profile``."""
    campaign, toy = toyhost_campaign()
    template = template or _template()
    runs = [
        run.model_copy(update={"template_id": template.id})
        for run in toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    ]
    results_by_run = {
        run.id: [_scored_by(result, **judge) for result in toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE)]
        for run in runs
    }
    storage = ToyhostStorage(runs, results_by_run, templates=[template])
    if profile is not None:
        storage.save_judge_profile(profile)
    ordered = [result for run in runs for result in results_by_run[run.id]]
    for result in ordered[:rated]:
        score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
        assert score is not None
        rate_result(
            storage,
            result_id=result.id,
            scope_id=TOYHOST_SCOPE,
            rubric_dim=TOYHOST_JUDGED_DIMENSION,
            rater="reviewer-1",
            rater_kind="person",
            score=score.score,
            reason="read it against the source",
        )
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def _tier(bundle: Any) -> Any:
    (tier,) = [t for t in bundle.judge_evidence_tiers if t.rubric_dim == TOYHOST_JUDGED_DIMENSION]
    return tier


# --- recording ------------------------------------------------------------------------------------


def test_a_judge_campaign_records_one_profile_per_judge_and_criterion_with_what_it_was_measured_on() -> None:
    profile = _recorded_profile()

    assert (profile.judge_model, profile.judge_config_id, profile.judge_temperature) == (
        _JUDGE,
        None,
        DEFAULT_JUDGE_TEMPERATURE,
    )
    assert profile.criterion_digest == judge_criterion_digest(TOYHOST_JUDGED_DIMENSION, "ordinal", _CRITERION)
    assert (profile.cases, profile.trials, profile.run_ids) == (_CASES, 2 * _CASES, ["judge-arm"])
    assert profile.measured_at == "2026-10-02T12:00:00+00:00", "the latest trial's scored_at"
    assert profile.label_agreement is not None and profile.label_agreement.results == _CASES
    assert profile.parse_validity == 1.0
    assert profile.tier == "calibrated"


def test_recording_again_replaces_the_profile_and_says_when_the_replaced_one_was_measured() -> None:
    host = toyhost_host()
    run_id = _judge_campaign(host)
    record = JudgeProfilesRecord(judge_run_ids=[run_id])
    first = judge_profiles_record(host, record, TOYHOST_SCOPE)
    again = judge_profiles_record(host, record, TOYHOST_SCOPE)

    assert first.profiles[0].replaced_measured_at is None
    assert again.profiles[0].replaced_measured_at == first.profiles[0].profile.measured_at
    (listed,) = judge_profiles_list(host, TOYHOST_SCOPE).profiles
    assert listed.profile.id == first.profiles[0].profile.id and listed.tier == "calibrated"


def test_a_judge_whose_served_model_was_not_recorded_is_skipped_never_profiled() -> None:
    host = toyhost_host()
    run_id = _judge_campaign(host, model=None)
    results = host.storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)

    drafts = judge_profiles_of(judge_kind_readings(results), scope_id=TOYHOST_SCOPE)
    assert not drafts.profiles
    (skip,) = drafts.skipped
    assert "named no served model" in skip.reason
    with pytest.raises(ValidationFailedError, match="no judge profile could be recorded"):
        judge_profiles_record(host, JudgeProfilesRecord(judge_run_ids=[run_id]), TOYHOST_SCOPE)


def test_only_a_finished_judge_kind_run_records_a_profile() -> None:
    host = toyhost_host()
    host.storage.save_eval_run(make_eval_run(id="not-a-judge", scope_id=TOYHOST_SCOPE, status="completed"))
    with pytest.raises(ValidationFailedError, match="not 'judge'"):
        judge_profiles_record(host, JudgeProfilesRecord(judge_run_ids=["not-a-judge"]), TOYHOST_SCOPE)
    with pytest.raises(NotFoundError):
        judge_profiles_record(host, JudgeProfilesRecord(judge_run_ids=["missing"]), TOYHOST_SCOPE)


def test_a_profile_survives_a_round_trip_through_the_store_and_refuses_an_id_its_fields_do_not_derive() -> None:
    profile = _recorded_profile()
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_judge_profile(profile)

    assert storage.load_judge_profile(profile.id, TOYHOST_SCOPE) == profile
    assert storage.query_judge_profiles(TOYHOST_SCOPE, rubric_dim="another.dim") == []
    with pytest.raises(ValueError, match="is not the one the judge and criterion derive"):
        EvalJudgeProfile.model_validate({**profile.model_dump(), "judge_temperature": 0.7})


def test_the_actions_record_and_list_profiles_and_render_their_receipts() -> None:
    host = toyhost_host()
    run_id = _judge_campaign(host)
    actions = {action.name: action for action in engine_actions()}
    record, listing = actions["judge_profiles_record"], actions["judge_profiles_list"]
    assert (record.permission, listing.permission) == ("write", "read")

    text = record.render(judge_profiles_record(host, JudgeProfilesRecord(judge_run_ids=[run_id]), TOYHOST_SCOPE))
    assert text.startswith("1 judge profile(s) recorded")
    assert f"{TOYHOST_JUDGED_DIMENSION} ({_JUDGE}, temperature 0)" in text and ": calibrated" in text
    assert "on 30 cases" in text and "runs judge-arm" in text
    assert listing.render(judge_profiles_list(host, TOYHOST_SCOPE)).startswith("1 judge profile(s)")


# --- reading: the done-condition --------------------------------------------------------------------


def test_a_second_campaign_scored_by_the_judge_reads_its_tier_from_the_profile_and_says_so() -> None:
    profile = _recorded_profile()
    without = _second_campaign(None)
    bundle = _second_campaign(profile)

    assert _tier(without).tier == "undetermined" and _tier(without).from_profile is None
    tier = _tier(bundle)
    assert tier.tier == "calibrated"
    assert tier.from_profile is not None
    assert tier.from_profile.profile_id == profile.id
    assert tier.from_profile.measured_at == profile.measured_at
    assert tier.from_profile.case_set_fingerprint == profile.case_set_fingerprint
    assert tier.from_profile.own_calibration.n == 0, "the campaign's own evidence is kept beside the profile's"
    # Every judged reading the judge served carries the profile's tier.
    assert {arm.evidence_tier for measure in bundle.judged_measures for arm in measure.arms} == {"calibrated"}
    # The profile used is in the fingerprint's pre-image.
    assert bundle.fingerprint() != without.fingerprint()
    recorded_later = profile.model_copy(update={"recorded_at": "2026-10-11T00:00:00+00:00"})
    assert _second_campaign(recorded_later).fingerprint() != bundle.fingerprint()

    report = build_code_only_report(bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00Z")
    (sentence,) = [
        block.text
        for block in report.blocks
        if isinstance(block, DisclosureBlock) and block.text.startswith("Judged evidence tier:")
    ]
    assert f"calibrated, read from the judge's stored profile {profile.id}" in sentence
    assert f"measured at {profile.measured_at} on 30 frozen cases" in sentence
    assert "This campaign's own evidence decided no tier" in sentence


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"temperature": 0.7}, id="temperature"),
        pytest.param({"config": "faithfulness-strict-v2"}, id="prompt"),
        pytest.param({"model": "judge/model-b"}, id="model"),
    ],
)
def test_a_changed_judge_reads_no_profile(changed: dict[str, Any]) -> None:
    tier = _tier(_second_campaign(_recorded_profile(), **changed))
    assert tier.tier == "undetermined" and tier.from_profile is None


def test_a_reworded_criterion_reads_no_profile() -> None:
    reworded = _CRITERION.model_copy(update={"description": "Fields sit where the invoice put them."})
    tier = _tier(_second_campaign(_recorded_profile(), template=_template(reworded)))
    assert tier.from_profile is None


def test_a_template_edited_after_the_runs_leaves_the_criterion_unknown_and_reads_no_profile() -> None:
    edited = _template(updated_at="2026-09-01T00:00:00+00:00")
    assert _tier(_second_campaign(_recorded_profile(), template=edited)).from_profile is None


def test_a_campaign_whose_own_evidence_decided_a_tier_keeps_it() -> None:
    # The profile measured a judge in perfect agreement; the campaign's own ratings decide calibrated too, from its
    # own outputs, and the tier is its own.
    tier = _tier(_second_campaign(_recorded_profile(), rated=CALIBRATION_MIN_RESULTS))
    assert tier.tier == "calibrated" and tier.from_profile is None


def test_a_profile_that_decides_no_tier_is_not_read() -> None:
    thin = _recorded_profile(cases=CALIBRATION_MIN_RESULTS - 1)
    assert thin.tier == "undetermined"
    assert _tier(_second_campaign(thin)).from_profile is None


def test_a_profile_whose_disagreement_decides_no_tier_is_not_read() -> None:
    # The judge campaign's labels disagree with the judge, but with self-agreement short of its floor the profile
    # shows only one criterion missed, so it decides no tier — and a profile must decide one to stand in.
    wanting = _recorded_profile(agree=False)
    assert wanting.tier == "undetermined"
    assert _tier(_second_campaign(wanting)).tier == "undetermined"


def test_the_criterion_of_each_judge_is_read_off_the_template_its_runs_were_judged_against() -> None:
    campaign, toy = toyhost_campaign()
    runs = [
        run.model_copy(update={"template_id": "toy-judged"})
        for run in toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    ]
    results_by_run = {
        run.id: [_scored_by(result) for result in toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE)] for run in runs
    }
    key = JudgeKey(TOYHOST_JUDGED_DIMENSION, "ordinal", _JUDGE, None, DEFAULT_JUDGE_TEMPERATURE)

    criteria = judged_criteria(runs, results_by_run, {"toy-judged": _template()})
    assert criteria[key] == judge_criterion_digest(TOYHOST_JUDGED_DIMENSION, "ordinal", _CRITERION)
    assert judged_criteria(runs, results_by_run, {})[key] is None, "a template that did not load says nothing"
