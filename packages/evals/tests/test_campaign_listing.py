"""A campaign listing carries what a host's list page shows of each campaign without opening it.

Contracts held here (``threetears.evals.ops.campaigns_list``, ``CampaignLine``):

- **declared fields**: ``created_at``, the question and its short form, the declared arms (label, short name, role)
  and the recorded call, as stored;
- **the short question** is cut at a word boundary to at most ``QUESTION_SHORT_CHARS`` characters, an ellipsis
  marking the cut, and is the question whole when it fits;
- **the report**: ``report_state`` is ``none`` with no live analysis and ``report`` is then None; ``written`` for one
  this build reads over the runs the campaign holds, with code's verdict as its headline and its attempt's cost;
  ``needs_regeneration``, with why, once the campaign holds other runs than it was written over (a run id that never
  resolved is not a move), or where its analysis does not read back together, which fails no listing or write;
- **one read per analysis**: the frozen bundle is put back together once per line, and a write of a declared campaign
  reads no other campaign;
- **the name** is held to ``CAMPAIGN_NAME_CHARS`` when written, the refusal saying the detail goes in the question or
  description; renaming moves no fingerprint;
- **verdict_short** is code's: ``Keep``/``Adopt`` and the arm from the recorded call, else from the report's
  recommendation, ``No call`` for a report recommending none, None with neither; the arm found by its variant key and
  printed as the title prints it;
- **outcome_short** is code's, read off the report's frozen readings and distinct from the decision: the answer
  leads (a change: which candidate improved, else none did; a choice: which arm leads; a measurement: measured;
  `No readings yet` for no rows) and the floors the answer's arms broke and the bars they miss only qualify it, floor
  first, counted past a length; a warning on an arm the answer does not name changes nothing; None with no report;
- **the title** is code's: the arms (declared, else the runs' candidate models; judges counted, with a report or
  without; a sweep counted by each model's variants), control first, then
  the aspects, then the name in parentheses, counting past three arms and two aspects;
- **feeds**: an exploratory campaign names the declared campaigns holding any of its runs; a declared one feeds
  none;
- **aspect_runs**: each aspect's label to the ids of its runs, in the campaign's order, the same runs ``feeds`` reads.
"""

from __future__ import annotations

from typing import Any

from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, empty_bundle, make_analysis
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import (
    CALLER,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    generate,
    ops_fixture,
    settled,
)
import pytest

from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.analysis.campaigns import list_campaigns
from threetears.evals.analysis.campaign_readings import (
    AspectReading,
    BarStatus,
    CampaignReadings,
    MeasureReading,
    PairReading,
    ShownReading,
)
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.report.decision import FloorStatus
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.declaration import CampaignCall, printed_aspect_name
from threetears.evals.schema.models import DECLARED_NAME_CHARS
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.ops import analysis as ops_analysis
from threetears.evals.ops import (
    CampaignChanges,
    CampaignDefinition,
    analyses_list,
    campaign_create,
    campaign_update,
    campaigns_list,
)
from threetears.evals.ops.analysis import (
    CAMPAIGN_NAME_CHARS,
    QUESTION_SHORT_CHARS,
    CampaignLine,
    campaign_title,
    question_short,
)


def _line(lines: list[CampaignLine], campaign_id: str) -> CampaignLine:
    (line,) = [line for line in lines if line.id == campaign_id]
    return line


def test_a_question_that_fits_is_listed_whole_and_a_long_one_is_cut_at_a_word() -> None:
    assert question_short("Which  model\nanswers best?") == "Which model answers best?"
    long = "Does the faster model hold its quality " + "on every aspect we measure " * 5
    cut = question_short(long)
    assert len(cut) <= QUESTION_SHORT_CHARS and cut.endswith("…")
    assert long.startswith(cut[:-1]) and long[len(cut) - 1] == " ", "cut where a word ends"
    assert question_short("x" * 300) == "x" * (QUESTION_SHORT_CHARS - 1) + "…"
    assert question_short("") == ""


def test_a_campaign_with_no_report_lists_its_declared_fields_and_no_report() -> None:
    fixture = ops_fixture()
    stored = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None

    line = _line(campaigns_list(fixture.host.eval_host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert (line.created_at, line.question) == (stored.created_at, stored.question)
    assert line.question_short == question_short(stored.question)
    assert [(arm.label, arm.short_name, arm.role) for arm in line.arms] == [
        (arm.label, arm.short_name, arm.role) for arm in stored.arms
    ]
    assert (line.call, line.report_state, line.report, line.feeds) == (None, "none", None, [])


async def test_a_written_report_lists_its_headline_and_cost_and_reads_stale_once_the_runs_move() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)
    host = fixture.host.eval_host
    await run_blocking(host.blocking_executor, _written_then_stale, host, fixture.campaign.id, status.analysis_id)


def _written_then_stale(host: EvalHost, campaign_id: str, analysis_id: str | None) -> None:
    (attempt,) = [
        attempt
        for attempt in host.storage.list_analysis_attempts_by_campaign(campaign_id, TOYHOST_SCOPE)
        if attempt.analysis_id == analysis_id
    ]

    line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, campaign_id)

    assert line.report_state == "written" and line.report is not None
    assert line.report.analysis_id == analysis_id and line.report.why_regenerate is None
    (analysis_line,) = [a for a in analyses_list(host, campaign_id, TOYHOST_SCOPE).analyses if a.id == analysis_id]
    assert line.title == analysis_line.title, "a campaign declaring no arm is titled by the arms its report resolved"
    assert line.report.draft == analysis_line.draft, "the list says a draft is one, as the analysis line does"
    assert line.report.headline and line.report.generator_model
    expected_cost = attempt.token_cost if attempt.unpriced_calls == 0 else None
    assert (line.report.cost_usd, line.report.unpriced_calls) == (expected_cost, attempt.unpriced_calls)

    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    first = stored.aspects[0]
    fewer = first.model_copy(update={"run_ids": first.run_ids[:-1]})
    host.storage.save_campaign(stored.model_copy(update={"aspects": [fewer, *stored.aspects[1:]]}))
    moved = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, campaign_id)

    assert moved.report_state == "needs_regeneration" and moved.report is not None
    assert moved.report.why_regenerate is not None and "other runs" in moved.report.why_regenerate


def test_an_exploratory_campaign_names_the_declared_campaign_holding_its_runs() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    sweep = campaign_create(
        host,
        CampaignDefinition(
            name="sweep",
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior="speed",
            aspects=[{"label": "speed", "run_ids": fixture.campaign.run_ids}],
            exploratory=True,
        ),
        TOYHOST_SCOPE,
        created_by=CALLER.identity,
    )

    lines = campaigns_list(host, TOYHOST_SCOPE).campaigns

    declared = _line(lines, fixture.campaign.id)
    assert not declared.exploratory, "the fixture's campaign is declared"
    assert _line(lines, sweep.id).feeds == [fixture.campaign.id]
    assert declared.feeds == []


def test_a_line_maps_each_aspect_to_its_runs_in_the_campaigns_order_and_feeds_reads_the_same_runs() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None
    runs = fixture.campaign.run_ids
    sweep = campaign_create(
        host,
        CampaignDefinition(
            name="sweep",
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior="speed",
            aspects=[{"label": "speed", "run_ids": list(reversed(runs))}],
            exploratory=True,
        ),
        TOYHOST_SCOPE,
        created_by=CALLER.identity,
    )

    lines = campaigns_list(host, TOYHOST_SCOPE).campaigns

    declared = _line(lines, fixture.campaign.id)
    assert declared.aspect_runs == {aspect.label: list(aspect.run_ids) for aspect in stored.aspects}
    assert [run for ids in declared.aspect_runs.values() for run in ids] == runs
    assert (
        list(declared.aspect_runs) == declared.aspects
        and sum(map(len, declared.aspect_runs.values())) == declared.run_count
    )
    assert sweep.aspect_runs == {"speed": list(reversed(runs))}, "the order is the campaign's, as stored"
    assert _line(lines, sweep.id).aspect_runs == sweep.aspect_runs
    assert _line(lines, sweep.id).feeds == [fixture.campaign.id], "feeds reads the runs aspect_runs lists"


def test_a_name_over_the_cap_is_refused_at_create_and_update_saying_where_the_detail_goes() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    long = "n" * (CAMPAIGN_NAME_CHARS + 1)
    definition = CampaignDefinition(
        name=long,
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior="speed",
        aspects=[{"label": "speed", "run_ids": fixture.campaign.run_ids}],
        exploratory=True,
    )

    with pytest.raises(ValidationFailedError) as created:
        campaign_create(host, definition, TOYHOST_SCOPE, created_by=CALLER.identity)
    with pytest.raises(ValidationFailedError) as renamed:
        campaign_update(host, fixture.campaign.id, CampaignChanges(name=long), TOYHOST_SCOPE, updated_by="op")

    for refused in (created.value, renamed.value):
        assert refused.remedy is not None and "question or description" in refused.remedy
    fits = "n" * CAMPAIGN_NAME_CHARS
    line = campaign_update(host, fixture.campaign.id, CampaignChanges(name=fits), TOYHOST_SCOPE, updated_by="op")
    assert line.name == fits


def test_a_recorded_call_reads_as_keep_for_the_control_and_adopt_for_a_candidate() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    derived = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE).arms
    assert stored is not None and len(derived) >= 2
    arms = [derived[0].model_copy(update={"role": "control"}), *derived[1:]]
    stored = stored.model_copy(update={"arms": list(arms)})
    for arm in arms:
        called = CampaignCall(arm=arm.label, variant_key=arm.variant_key, called_by="op")
        host.storage.save_campaign(stored.model_copy(update={"call": called}))

        line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

        assert line.verdict_short == f"{'Keep' if arm.role == 'control' else 'Adopt'} {arm.label}"


async def test_with_no_call_a_report_s_recommendation_is_the_short_verdict() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    status = await settled(fixture.host, job.job_id)
    host = fixture.host.eval_host
    await run_blocking(host.blocking_executor, _recommendation_read, host, fixture.campaign.id, status.analysis_id)


def _recommendation_read(host: EvalHost, campaign_id: str, analysis_id: str | None) -> None:
    analysis = host.storage.load_reported_analysis(campaign_id, TOYHOST_SCOPE)
    assert analysis is not None and analysis.id == analysis_id

    line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, campaign_id)

    if analysis.recommended_variant_key is None:
        assert line.verdict_short == "No call"
    else:
        assert line.verdict_short is not None and line.verdict_short.split(" ", 1)[0] in {"Keep", "Adopt"}


def test_a_campaign_with_neither_a_call_nor_a_report_has_no_short_verdict() -> None:
    fixture = ops_fixture()

    line = _line(campaigns_list(fixture.host.eval_host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert line.verdict_short is None


def test_renaming_a_campaign_moves_no_fingerprint() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    before = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE)

    campaign_update(host, fixture.campaign.id, CampaignChanges(name="renamed"), TOYHOST_SCOPE, updated_by="op")
    after = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE)

    assert (before.name, after.name) != ("renamed", "renamed") and after.name == "renamed"
    assert after.fingerprint() == before.fingerprint()


def test_a_title_names_the_declared_arms_control_first_and_the_aspects_then_the_name() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    derived = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE).arms
    assert stored is not None and len(derived) >= 2
    control = derived[0].model_copy(update={"role": "control", "short_name": "ctl"})
    declared = stored.model_copy(update={"arms": [control, *derived[1:2]]})
    aspect = printed_aspect_name(stored.aspects[0].label, stored.aspects[0].short_name)

    assert campaign_title(declared) == f"ctl vs {derived[1].short_name or derived[1].label} · {aspect} ({stored.name})"


def test_past_three_arms_a_title_counts_the_candidates_and_past_two_aspects_counts_them() -> None:
    fixture = ops_fixture()
    stored = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None
    arm = assemble_campaign_bundle(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE).arms[0]
    arms = [arm.model_copy(update={"label": "base", "role": "control"})]
    arms += [arm.model_copy(update={"label": f"c{n}", "role": "candidate"}) for n in range(4)]
    aspects = [stored.aspects[0].model_copy(update={"label": f"a{n}", "short_name": None}) for n in range(3)]
    many = stored.model_copy(update={"arms": arms, "aspects": aspects})

    assert campaign_title(many) == f"base vs 4 candidates · 3 aspects ({stored.name})"


def test_a_campaign_declaring_no_arm_is_titled_by_its_runs_models_and_a_judge_one_counts_its_judges() -> None:
    fixture = ops_fixture()
    stored = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and not stored.arms
    measure = printed_aspect_name(stored.aspects[0].label, stored.aspects[0].short_name)

    titled = campaign_title(stored, candidate_models=["openrouter/qwen/qwen3.8-flash", "anthropic/haiku"])
    judged = campaign_title(stored, candidate_models=[f"m{n}" for n in range(8)], judges=True)

    assert titled == f"qwen3.8-flash vs haiku · {measure} ({stored.name})"
    assert judged == f"8 judges · {measure} ({stored.name})"
    assert (
        campaign_title(stored, candidate_models=["openrouter/x/m1"], variants=3)
        == f"m1, 3 variants · {measure} ({stored.name})"
    )
    line = _line(campaigns_list(fixture.host.eval_host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)
    assert line.title.endswith(f" · {measure} ({stored.name})")


def _change_row(aspect: str, key: str, label: str, reading: ShownReading) -> AspectReading:
    return AspectReading(
        aspect=aspect,
        arm=key,
        arm_label=label,
        change=1.0,
        gain_interval=None,
        gain_confidence=None,
        drop_interval=None,
        drop_confidence=None,
        gain_p=None,
        gain_p_holm=None,
        drop_p=None,
        drop_p_holm=None,
        reading=reading,
        margin=None,
        control_cases=1,
        arm_cases=1,
        control_passed=0,
        control_trials=1,
        arm_passed=0,
        arm_trials=1,
        projection=None,
    )


def _pair(aspect: str, first: tuple[str, str], second: tuple[str, str], ahead: str | None) -> PairReading:
    return PairReading(
        aspect=aspect,
        first=first[0],
        first_label=first[1],
        second=second[0],
        second_label=second[1],
        change=1.0,
        interval=None,
        confidence=None,
        p=None,
        p_holm=None,
        ahead=ahead,
        first_cases=1,
        second_cases=1,
        first_passed=0,
        first_trials=1,
        second_passed=0,
        second_trials=1,
    )


def _floor(aspect: str, key: str, label: str) -> FloorStatus:
    return FloorStatus(
        floor="f", aspect=aspect, arm=key, arm_label=label, status="broken", reading=None, interval=None, reason="r"
    )


def _bar(aspect: str, key: str, label: str) -> BarStatus:
    return BarStatus(
        aspect=aspect,
        arm=key,
        arm_label=label,
        measure="passed",
        bar=0.5,
        direction="higher_is_better",
        status="missed",
        value=0.1,
        interval=None,
        margin=None,
    )


def _measure(aspect: str, key: str, label: str) -> MeasureReading:
    return MeasureReading(
        aspect=aspect,
        arm=key,
        arm_label=label,
        measure="passed",
        name="pass rate",
        value=0.5,
        interval=None,
        cases=1,
    )


async def test_the_short_outcome_is_read_from_the_reports_readings_by_kind() -> None:
    fixture = off_loop(ops_fixture)
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await settled(fixture.host, job.job_id)
    host = fixture.host.eval_host
    await run_blocking(host.blocking_executor, _outcomes_read, host, fixture.campaign.id)


def _outcomes_read(host: EvalHost, campaign_id: str) -> None:
    analysis = host.storage.load_reported_analysis(campaign_id, TOYHOST_SCOPE)
    assert analysis is not None
    bundle = analysis_campaign_bundle(analysis)
    first, second = bundle.arms[:2]
    named = bundle.model_copy(update={"arms": [first, second.model_copy(update={"short_name": "fast"})]})
    aspect = bundle.readings.aspects[0]
    one = (first.variant_key, first.label)
    two = (second.variant_key, second.label)
    third = ("other-key", "third")
    empty = {"readings": [], "pairs": [], "measures": [], "floors": [], "bars": []}

    def read(kind: str, **rows: list[object]) -> str | None:
        readings: CampaignReadings = named.readings.model_copy(update={**empty, "kind": kind, **rows})
        host.storage.save_analysis(
            analysis.model_copy(update={"campaign_bundle": named.model_copy(update={"readings": readings}).frame()})
        )
        return _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, campaign_id).outcome_short

    assert read("change") == "No readings yet"
    assert read("change", readings=[_change_row(aspect, *two, "unclear")]) == "No candidate improved"
    assert read("change", readings=[_change_row(aspect, *two, "worse")]) == "No candidate improved"
    assert read("change", readings=[_change_row(aspect, *two, "within_margin")]) == "No candidate improved"
    assert read("change", readings=[_change_row(aspect, *two, "better")]) == "fast improved"
    both = [_change_row(aspect, *two, "better"), _change_row(aspect, *third, "better")]
    assert read("change", readings=both) == "2 candidates improved"
    assert read("change", readings=[_change_row(aspect, *third, "better")]) == "third improved", "label, no short name"
    assert read("choice") == "No readings yet"
    assert read("choice", pairs=[_pair(aspect, one, two, None)]) == "Not separated"
    assert read("choice", pairs=[_pair(aspect, one, two, two[0])]) == "fast ahead"
    cycle = [_pair(aspect, one, two, one[0]), _pair(aspect, two, third, two[0]), _pair(aspect, third, one, third[0])]
    assert read("choice", pairs=cycle) == "Arms differ", "an arm ahead of one and behind another leads nowhere"
    assert read("measurement") == "No readings yet"
    assert read("measurement", measures=[_measure(aspect, *one)]) == "Measured"
    missed = {"bars": [_bar(aspect, *two)]}
    broke = {"floors": [_floor(aspect, *two)]}
    win = [_change_row(aspect, *two, "better")]
    assert read("measurement", measures=[_measure(aspect, *one)], **missed) == "Measured, fast missed a bar"
    assert read("change", **missed, readings=win) == "fast improved, missed a bar"
    assert read("change", **broke, readings=win) == "fast improved, broke a floor"
    assert read("change", **broke, **missed, readings=win) == "fast improved, broke a floor and missed a bar"
    nothing = [_change_row(aspect, *two, "worse")]
    assert read("change", **broke, **missed, readings=nothing) == "No candidate improved", "a warning never answers"
    assert read("change", **broke, readings=[_change_row(aspect, *third, "better")]) == "third improved"
    wins = [_change_row(aspect, *two, "better"), _change_row(aspect, *third, "better")]
    assert read("change", **broke, readings=wins) == "2 candidates improved, 1 broke a floor"
    assert read("change", floors=[_floor(aspect, *two)], bars=[_bar(aspect, *third)], readings=wins) == (
        "2 candidates improved, 2 warnings"
    ), "past the length, the warnings are counted"
    assert read("choice", **broke, pairs=[_pair(aspect, one, two, two[0])]) == "fast ahead, broke a floor"
    assert read("choice", **broke, pairs=[_pair(aspect, one, two, one[0])]) == f"{one[1]} ahead"
    lots = {"floors": [_floor(aspect, *two), _floor(aspect, *two)], "bars": [_bar(aspect, *two)]}
    long_win = [_change_row(aspect, "k", "an arm with a very long printed name", "better")]
    long = {"floors": [_floor(aspect, "k", "an arm with a very long printed name")], "bars": [_bar(aspect, "k", "x")]}
    assert read("change", **long, readings=long_win) == "an arm with a very long printed name improved, 2 warnings"
    assert read("change", **lots, readings=win) == "fast improved, broke a floor and missed a bar"


def test_a_campaign_with_no_report_has_no_short_outcome() -> None:
    fixture = ops_fixture()

    line = _line(campaigns_list(fixture.host.eval_host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert line.outcome_short is None


def test_a_recorded_call_finds_its_arm_by_variant_key_not_by_label() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    derived = assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE).arms
    assert stored is not None and len(derived) >= 2
    control = derived[0].model_copy(update={"role": "control", "label": "base"})
    # The call names the control by its key under a label no arm carries now (the arm was relabelled since).
    called = CampaignCall(arm="old name", variant_key=control.variant_key, called_by="op")
    host.storage.save_campaign(stored.model_copy(update={"arms": [control, *derived[1:]], "call": called}))

    line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert line.verdict_short == "Keep base"


def _frozen_with_computed_label(host: EvalHost, campaign_id: str) -> tuple[EvalAnalysis, str]:
    """An analysis frozen over the campaign's bundle, its candidate labelled as code spells a lever's level
    (``chunk_tokens=1024tok``), recommending that candidate; and the candidate's key."""
    bundle = assemble_campaign_bundle(host, campaign_id, TOYHOST_SCOPE)
    control, candidate, *rest = bundle.arms
    computed = candidate.model_copy(update={"label": "chunk_tokens=1024tok"})
    frozen = bundle.model_copy(update={"arms": [control, computed, *rest]})
    analysis = make_analysis(
        campaign_id=campaign_id,
        scope_id=TOYHOST_SCOPE,
        aspects=[analysed_aspect(aspect) for aspect in frozen.aspects],
        campaign_bundle=frozen.frame(),
        observation_refs=frozen.run_ids,
        recommended_variant_key=candidate.variant_key,
    )
    return analysis, candidate.variant_key


def test_a_short_verdict_prints_a_computed_arm_as_the_title_does_never_axis_equals_level() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    analysis, key = _frozen_with_computed_label(host, fixture.campaign.id)
    host.storage.save_analysis(analysis)

    recommended = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None
    called = CampaignCall(arm="chunk_tokens=1024tok", variant_key=key, called_by="op")
    host.storage.save_campaign(stored.model_copy(update={"call": called}))
    decided = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert recommended.verdict_short == "Adopt 1024tok"
    assert decided.verdict_short == "Adopt 1024tok"


def _corrupt_analysis(campaign_id: str) -> EvalAnalysis:
    """An analysis whose frame names another aspect bundle than it holds: it does not read back together."""
    held = empty_bundle(campaign_id=campaign_id, scope_id=TOYHOST_SCOPE)
    named = empty_bundle(campaign_id=campaign_id, scope_id=TOYHOST_SCOPE, behavior="another")
    return make_analysis(
        campaign_id=campaign_id,
        scope_id=TOYHOST_SCOPE,
        aspects=[analysed_aspect(held)],
        campaign_bundle=campaign_bundle(named).frame(),
    )


def test_one_corrupt_analysis_lists_its_report_as_needing_regeneration_and_fails_no_listing_or_write() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    corrupt = _corrupt_analysis(fixture.campaign.id)
    host.storage.save_analysis(corrupt)

    line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)
    written = campaign_update(
        host, fixture.campaign.id, CampaignChanges(name="renamed"), TOYHOST_SCOPE, updated_by="op"
    )
    (listed,) = [
        one for one in analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE).analyses if one.id == corrupt.id
    ]

    for one in (line, written):
        assert one.report_state == "needs_regeneration" and one.report is not None
        assert one.report.why_regenerate is not None and "does not read back together" in one.report.why_regenerate
        assert "analysis_generate" in one.report.why_regenerate and one.outcome_short is None
    assert listed.needs_regeneration is not None and listed.verdict.startswith("Needs regeneration: ")


def test_a_listing_puts_each_frozen_bundle_back_together_once_and_a_declared_write_lists_no_other_campaign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    analysis, _ = _frozen_with_computed_label(host, fixture.campaign.id)
    host.storage.save_analysis(analysis)
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and not stored.exploratory
    built: list[str] = []
    listed: list[str] = []

    def counted_bundle(held: EvalAnalysis) -> Any:
        built.append(held.id)
        return analysis_campaign_bundle(held)

    def counted_list(*args: Any, **kwargs: Any) -> Any:
        listed.append("listed")
        return list_campaigns(*args, **kwargs)

    monkeypatch.setattr(ops_analysis, "analysis_campaign_bundle", counted_bundle)
    monkeypatch.setattr(ops_analysis, "list_campaigns", counted_list)

    campaigns_list(host, TOYHOST_SCOPE)
    assert built == [analysis.id], "one bundle per analysis, whatever the line reads off it"
    built.clear()
    listed.clear()
    campaign_update(host, stored.id, CampaignChanges(name="renamed"), TOYHOST_SCOPE, updated_by="op")
    assert built == [analysis.id] and listed == [], "a declared campaign feeds none, so no other campaign is read"


async def test_a_run_id_that_never_resolved_does_not_read_as_the_runs_having_moved() -> None:
    fixture = off_loop(ops_fixture)
    host = fixture.host.eval_host
    await run_blocking(host.blocking_executor, _hold_a_missing_run, host, fixture.campaign.id)
    (job,) = (await generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    await settled(fixture.host, job.job_id)
    line = await run_blocking(
        host.blocking_executor, lambda: _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)
    )

    assert line.report is not None and line.report.why_regenerate is None
    assert line.report_state == "written"


def _hold_a_missing_run(host: EvalHost, campaign_id: str) -> None:
    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    first = stored.aspects[0]
    holding = first.model_copy(update={"run_ids": [*first.run_ids, "no-such-run"]})
    host.storage.save_campaign(stored.model_copy(update={"aspects": [holding, *stored.aspects[1:]]}))


def test_a_judge_campaign_counts_its_judges_with_a_report_or_without() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and not stored.arms
    runs = host.storage.load_eval_runs(stored.run_ids, TOYHOST_SCOPE)
    for run in runs:
        host.storage.save_eval_run(run.model_copy(update={"candidate_kind": JUDGE_KIND}))
    judged = campaign_title(stored, candidate_models=sorted({run.candidate_model for run in runs}), judges=True)
    before = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id).title
    analysis, _ = _frozen_with_computed_label(host, fixture.campaign.id)
    host.storage.save_analysis(analysis)

    after = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id).title

    assert before == after == judged


def test_two_models_each_run_at_two_settings_title_as_a_sweep() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    stored = host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert stored is not None and not stored.arms
    (run, *_) = host.storage.load_eval_runs(stored.run_ids, TOYHOST_SCOPE)
    # Two settings of one lever, as the campaign's own arms record them: each model is run at both.
    settings = [arm.levers for arm in assemble_campaign_bundle(host, fixture.campaign.id, TOYHOST_SCOPE).arms[:2]]
    assert all(settings) and settings[0] != settings[1]
    swept = [
        run.model_copy(update={"id": f"sweep-{model}-{at}", "candidate_model": model, "variant_levers": levers})
        for model in ("a/one", "b/two")
        for at, levers in enumerate(settings)
    ]
    for one in swept:
        host.storage.save_eval_run(one)
    first = stored.aspects[0]
    host.storage.save_campaign(
        stored.model_copy(update={"aspects": [first.model_copy(update={"run_ids": [one.id for one in swept]})]})
    )
    measure = printed_aspect_name(first.label, first.short_name)

    line = _line(campaigns_list(host, TOYHOST_SCOPE).campaigns, fixture.campaign.id)

    assert line.title == f"4 variants · {measure} ({stored.name})", "two lever maps over two models are four variants"


def test_the_schemas_state_the_name_caps_a_host_form_reads() -> None:
    from threetears.evals.kernel.declaration import Aspect, CampaignArm

    assert CampaignDefinition.model_json_schema()["properties"]["name"]["maxLength"] == CAMPAIGN_NAME_CHARS
    assert CampaignChanges.model_json_schema()["properties"]["name"]["maxLength"] == CAMPAIGN_NAME_CHARS
    for declared in (CampaignArm, Aspect):
        assert declared.model_json_schema()["properties"]["short_name"]["maxLength"] == DECLARED_NAME_CHARS
