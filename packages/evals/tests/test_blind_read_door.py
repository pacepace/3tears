"""The blind read's host door: the actions a host's agent or page drives, onto the one core.

Pinned here, each as the refusal it is and not only as a happy path:

- the draw takes no seed: the parameter does not exist, two draws over one population take two orders, and the line
  says how many were drawn before;
- the sheet and every line while the read is open give a rater nothing that unblinds: no arm, run, result or case
  id, no judge score, no label key, no count of who preferred what;
- a person's answer is recorded only when the host attests the caller relaying it for that person: with no check, or
  a check that declines, it is refused (403) and nothing is written; an attested one is the host's name for them;
- a model's answers enter a read only from inside the read's own review job: outside any job, or inside another,
  they are refused (403) and nothing is written; the review job keeps them, named by the declared reviewer;
- a reviewer that is the judge's own model is refused at the draw;
- close writes each answer as a rating (a person's as theirs, naming the read and the relaying caller), refuses
  later answers, and a read closed before its planned length reads its preference by the anytime test;
- no response field is named with a word kept internal (anchor, standing, tier, ordinal).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel.job_context import job_running
from threetears.evals.kernel.errors import EvalServiceError
from threetears.evals.ops.analysis import CampaignChanges, campaign_update
from threetears.evals.ops.blind_reads import blind_read_create, blind_read_get, keep_review
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema import JudgedArtifact, JudgeEvidence, RubricDim, label_key_of
from threetears.evals.schema.blind_read import BlindAnswer, BlindReview, BlindVerdict
from threetears.evals.quick.blind import blind_read_html
from threetears.evals.testing.sheet_conformance import SheetConformanceFailure, check_sheet_renderer
from packages.evals.tests.factories import make_eval_trace, make_rubric_dim
from packages.evals.tests.fixtured_client import FixturedClient
from packages.evals.tests.fixtures.courierhost import (
    COURIER_DISPATCH_DESK,
    COURIER_SCOPE,
    courier_attests,
    courier_sheet_page,
)
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled

PERSON = "ana@example.com"
REVIEWER = "reviewer/other-model"
JUDGE_SERVED = "judge/served-toy"
DIM = RubricDim(
    name=TOYHOST_JUDGED_DIMENSION,
    description="The extracted fields keep the invoice's layout.",
    scale="ordinal",
    scoring_guide={"5": "every field where the invoice has it"},
)

#: Words a response field never carries (IA 2.1): kept for internal names, or ruled to mean something else.
INTERNAL_WORDS = ("anchor", "standing", "tier", "ordinal_", "null")


def _attests(*, scope_id: str, caller: str, person: str) -> str | None:
    """The host's check: it vouches for :data:`CALLER` relaying anyone, under its own spelling of their name."""
    return person.lower() if caller == CALLER.identity else None


def _with(host: OpsHost, **changes: Any) -> OpsHost:
    launch = host.launch
    return replace(host, launch=replace(launch, eval_host=replace(launch.eval_host, **changes)))


def _world(*, reviewer_reply: str | None = None) -> tuple[OpsHost, OpsFixture, list[FixturedClient]]:
    """The toy corpus campaign made readable blind: its results judged with label keys and stored evidence, its
    criterion catalogued, its first arm the control; a host attesting :data:`CALLER` for anyone.

    ``reviewer_reply`` is what every model call answers, where a test reads with a model.
    """
    fixture = off_loop(ops_fixture)
    eval_host = fixture.host.eval_host
    storage = eval_host.storage
    off_loop(storage.save_rubric_dim, make_rubric_dim(scope_id=TOYHOST_SCOPE, key=DIM.name, dim=DIM))
    sorted_cases: dict[str, int] = {}
    arms: dict[str, int] = {}
    for run_id in fixture.campaign.run_ids:
        for result in off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE):
            case = sorted_cases.setdefault(result.test_case_id, len(sorted_cases) + 1)
            arm = arms.setdefault(result.variant_key, len(arms) + 1)
            evidence = JudgeEvidence(
                case_material=f"an invoice of {case} lines",
                artifact=f"layout {'kept' if arm == 1 else 'lost'} on {case} lines, try {result.k_iteration}",
            )
            key = label_key_of(evidence, DIM)
            scores = [
                score.model_copy(
                    update={
                        "output_fingerprint": key.output_fingerprint,
                        "criterion_fingerprint": key.criterion_fingerprint,
                        "served_model": JUDGE_SERVED,
                    }
                )
                if score.dim == DIM.name
                else score
                for score in result.rubric_scores
            ]
            trace = make_eval_trace(
                result_id=result.id,
                scope_id=TOYHOST_SCOPE,
                eval_run_id=run_id,
                judge_evidence=evidence,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            off_loop(
                storage.save_eval_result, result.model_copy(update={"rubric_scores": scores, "has_trace": True}), trace
            )
    off_loop(
        campaign_update,
        eval_host,
        fixture.campaign.id,
        CampaignChanges(control_from_run_id=fixture.campaign.run_ids[0]),
        TOYHOST_SCOPE,
        updated_by=CALLER.identity,
    )
    built: list[FixturedClient] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> FixturedClient:
        client = FixturedClient(reviewer_reply or "{}")
        built.append(client)
        return client

    host = _with(fixture.host, attests_person=_attests, clients=clients)
    return host, fixture, built


def _candidate(host: OpsHost, fixture: OpsFixture) -> str:
    read = off_loop(
        blind_read_create, host.eval_host, TOYHOST_SCOPE, created_by="probe", campaign_id=fixture.campaign.id
    )
    control = read.arms[0].key
    held = {
        result.variant_key
        for run_id in fixture.campaign.run_ids
        for result in off_loop(host.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)
    }
    (candidate,) = held - {control}
    return candidate


def _evals() -> MountedTool:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]


async def _call(host: OpsHost, action: str, caller: Caller = CALLER, **params: Any) -> Any:
    return await _evals().call({"action": action, **params}, host=host, caller=caller)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


async def _pair_read(host: OpsHost, fixture: OpsFixture, **params: Any) -> dict[str, Any]:
    outcome = await _call(
        host, "blind_read_create", campaign_id=fixture.campaign.id, candidate=_candidate(host, fixture), **params
    )
    assert not outcome.is_error, outcome.text
    assert outcome.structured is not None
    return dict(outcome.structured)


# --- the draw --------------------------------------------------------------------------------------------------


async def test_the_draw_takes_no_seed_and_a_second_draw_over_the_same_outputs_takes_another_order() -> None:
    host, fixture, _ = _world()

    seeded = await _call(
        host, "blind_read_create", campaign_id=fixture.campaign.id, candidate=_candidate(host, fixture), seed=7
    )
    first = await _pair_read(host, fixture)
    second = await _pair_read(host, fixture)

    assert seeded.is_error, "a caller cannot choose the order"
    assert (first["earlier_reads"], second["earlier_reads"]) == (0, 1)
    assert first["read_id"] != second["read_id"]
    one = off_loop(host.eval_host.storage.load_blind_read_with_etag, first["read_id"], TOYHOST_SCOPE)[0]
    two = off_loop(host.eval_host.storage.load_blind_read_with_etag, second["read_id"], TOYHOST_SCOPE)[0]
    assert one is not None and two is not None
    assert one.draw.population_digest == two.draw.population_digest
    assert [item.case_id for item in one.items] != [item.case_id for item in two.items]


async def test_a_reviewer_that_is_the_judge_s_own_model_is_refused_at_the_draw() -> None:
    host, fixture, _ = _world()

    outcome = await _call(host, "blind_read_create", run_ids=[fixture.campaign.run_ids[0]], reviewer=JUDGE_SERVED)

    assert outcome.is_error and "judge's own model" in outcome.text, outcome.text
    assert off_loop(host.eval_host.storage.query_blind_reads, TOYHOST_SCOPE) == []


# --- the sheet gives nothing away ------------------------------------------------------------------------------


async def test_the_sheet_holds_no_id_score_label_key_or_tally_that_would_unblind_a_rater() -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    first = read.items[0]
    answered = await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=first.item,
        person=PERSON,
        verdicts={f"{output.slot}/{DIM.name}": "4" for output in first.outputs},
        prefer="A",
    )
    assert not answered.is_error, answered.text

    sheet = await _call(host, "blind_read_sheet", read_id=read.id, rater=PERSON)

    assert not sheet.is_error, sheet.text
    text = json.dumps(sheet.structured)
    forbidden = {
        *(arm.key for arm in read.arms),
        *(run_id for arm in read.arms for run_id in arm.run_ids),
        *(output.result_id for item in read.items for output in item.outputs),
        *(item.case_id for item in read.items),
        *(output.output_fingerprint for item in read.items for output in item.outputs),
        *(fp for item in read.items for output in item.outputs for fp in output.criterion_fingerprints.values()),
    }
    assert not [value for value in forbidden if value in text], (
        "the sheet names nothing that pairs an output with its arm"
    )
    assert {"fingerprint", "score", "prefer", "control", "candidate"}.isdisjoint(_keys(sheet.structured))
    assert sheet.structured["answered"] == [first.item], "the rater's own progress, and nobody's answers"
    for open_line in (answered.structured, (await _call(host, "blind_read_get", read_id=read.id)).structured):
        assert open_line["checks"] == [] and open_line["preferences"] == [], "no reading while the read is open"
        assert open_line["raters"] == [
            {"rater": PERSON, "made_by": "person", "sample": "random", "answered": 1, "skipped": 0, "partial": 0}
        ]


async def test_no_response_field_carries_a_word_kept_internal() -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    answered = await _call(
        host, "blind_read_answer", read_id=read.id, item=read.items[0].item, person=PERSON, skip=True
    )
    closed = await _call(host, "blind_read_close", read_id=line["read_id"])
    sheet = await _call(host, "blind_read_sheet", read_id=line["read_id"])
    listing = await _call(host, "blind_reads_list")

    fields = (
        _keys(line)
        | _keys(answered.structured)
        | _keys(closed.structured)
        | _keys(sheet.structured)
        | _keys(listing.structured)
    )

    assert "sample" in fields
    assert not [field for field in fields if any(word in field for word in INTERNAL_WORDS)], fields


# --- who may answer --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("check", ["none", "declines"])
async def test_an_unattested_person_s_answer_is_refused_and_nothing_is_written(check: str) -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    refusing = _with(host, attests_person=None if check == "none" else (lambda **_: None))

    outcome = await _call(
        refusing,
        "blind_read_answer",
        read_id=read.id,
        item=read.items[0].item,
        person=PERSON,
        verdicts={f"A/{DIM.name}": "4", f"B/{DIM.name}": "2"},
        prefer="A",
    )

    assert outcome.is_error and outcome.refusal is not None
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 403)
    after = off_loop(host.eval_host.storage.load_blind_read_with_etag, read.id, TOYHOST_SCOPE)[0]
    assert after is not None and after.answers == []


async def test_an_attested_answer_is_the_host_s_name_for_the_person_and_an_answer_the_item_does_not_ask_is_refused() -> (
    None
):
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    item = read.items[0].item

    off_scale = await _call(
        host, "blind_read_answer", read_id=read.id, item=item, person="Ana@Example.com", verdicts={f"A/{DIM.name}": "9"}
    )
    kept = await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=item,
        person="Ana@Example.com",
        verdicts={f"A/{DIM.name}": "4", f"B/{DIM.name}": "2"},
        prefer="B",
        answer_reason="B keeps the totals block",
    )
    as_model = await _call(
        host, "blind_read_answer", read_id=read.id, item=item, person=PERSON, made_by="model", skip=True
    )

    assert off_scale.is_error and "scale" in off_scale.text
    assert not kept.is_error, kept.text
    assert as_model.is_error, "no action takes a model's answer"
    stored = off_loop(host.eval_host.storage.load_blind_read_with_etag, read.id, TOYHOST_SCOPE)[0]
    assert stored is not None
    (answer,) = stored.answers
    assert (answer.rater, answer.rater_kind, answer.collected_by, answer.prefer) == (
        PERSON,
        "person",
        CALLER.identity,
        "B",
    )


# --- a model's answers ------------------------------------------------------------------------------------------


def _reply() -> str:
    return json.dumps({"verdicts": {"A": {DIM.name: "4"}}, "prefer": "none", "reason": "keeps the layout"})


async def test_a_model_s_answers_are_kept_only_from_inside_the_read_s_own_review_job() -> None:
    host, fixture, built = _world(reviewer_reply=_reply())
    created = await _call(host, "blind_read_create", run_ids=[fixture.campaign.run_ids[0]], reviewer=REVIEWER)
    assert not created.is_error, created.text
    read_id = created.structured["read_id"]
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, read_id, TOYHOST_SCOPE)[0]
    assert read is not None
    forged = [
        BlindAnswer(
            item=read.items[0].item,
            rater=REVIEWER,
            rater_kind="reference",
            verdicts=[BlindVerdict(slot="A", dim=DIM.name, score=5)],
        )
    ]
    review = BlindReview(reviewer=REVIEWER, asked=1)

    with pytest.raises(EvalServiceError) as outside:
        off_loop(keep_review, host.eval_host, read_id, TOYHOST_SCOPE, forged, review)
    with job_running("someone-else"), pytest.raises(EvalServiceError) as other_job:
        # A thread carrying the job's context, as the job's own store calls run off the loop.
        await asyncio.to_thread(keep_review, host.eval_host, read_id, TOYHOST_SCOPE, forged, review)
    started = await _call(host, "blind_read_review", read_id=read_id)
    assert not started.is_error, started.text
    (handle,) = started.structured["jobs"]
    status = await settled(host, handle["job_id"])

    assert outside.value.status_code == 403 and other_job.value.status_code == 403
    assert status.status == "completed", status
    assert status.result is not None and status.result["read_id"] == read_id, "the job hands back the read's line"
    stored = off_loop(host.eval_host.storage.load_blind_read_with_etag, read_id, TOYHOST_SCOPE)[0]
    assert stored is not None and stored.review_job_id is not None
    assert {(answer.rater, answer.rater_kind) for answer in stored.answers} == {(REVIEWER, "reference")}
    assert len(stored.answers) == len(stored.items), "every item answered, by the declared reviewer"
    assert [client.calls for client in built if client.calls], "the job made the calls"
    line = off_loop(blind_read_get, host.eval_host, read_id, TOYHOST_SCOPE)
    assert [(rater.rater, rater.made_by) for rater in line.raters] == [(REVIEWER, "model")]


async def test_a_read_with_no_declared_reviewer_starts_no_review() -> None:
    host, fixture, built = _world(reviewer_reply=_reply())
    line = await _pair_read(host, fixture)

    outcome = await _call(host, "blind_read_review", read_id=line["read_id"])

    assert outcome.is_error and "declared no reviewer" in outcome.text
    assert not [client for client in built if client.calls]


# --- close -----------------------------------------------------------------------------------------------------


async def test_close_writes_each_answer_as_the_person_s_rating_and_refuses_later_answers() -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    first = read.items[0]
    await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=first.item,
        person=PERSON,
        verdicts={f"A/{DIM.name}": "4", f"B/{DIM.name}": "2"},
        prefer="A",
    )

    closed = await _call(host, "blind_read_close", read_id=read.id)
    late = await _call(host, "blind_read_answer", read_id=read.id, item=read.items[1].item, person=PERSON, skip=True)
    again = await _call(host, "blind_read_close", read_id=read.id)

    assert not closed.is_error, closed.text
    assert closed.structured["ratings_written"] == 2 and closed.structured["ratings_not_written"] == []
    assert late.is_error and late.refusal is not None and late.refusal.status == 409
    assert again.is_error and again.refusal is not None and again.refusal.status == 409
    ratings = [
        rating
        for output in first.outputs
        for rating in off_loop(
            host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=output.result_id
        )
    ]
    assert {(r.rater, r.rater_kind, r.source, r.collected_by) for r in ratings} == {
        (PERSON, "person", f"blind_read:{read.id}", CALLER.identity)
    }


async def test_an_early_close_reads_the_preference_by_the_anytime_test() -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture, planned_items=12)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None
    candidate = next(arm.key for arm in read.arms if arm.role == "candidate")
    pairs = [item for item in read.items[: read.draw.planned_items] if len(item.outputs) == 2]
    assert len(pairs) >= 9, "the toy corpus holds enough pairs for a read"
    for item in pairs[:9]:
        slot = next(output.slot for output in item.outputs if output.arm == candidate)
        outcome = await _call(host, "blind_read_answer", read_id=read.id, item=item.item, person=PERSON, prefer=slot)
        assert not outcome.is_error, outcome.text
    still_open = await _call(host, "blind_read_get", read_id=read.id)

    closed = await _call(host, "blind_read_close", read_id=read.id)

    assert still_open.structured["preferences"] == [], "withheld while open"
    assert not closed.is_error, closed.text
    assert closed.structured["closed_early"] is True
    (preference,) = closed.structured["preferences"]
    assert (preference["made_by"], preference["test"], preference["prefer_candidate"]) == ("person", "anytime", 9)
    assert preference["verdict"] in ("improved", "not_separated") and preference["p"] is not None


# --- the sheet conformance check -------------------------------------------------------------------------------


def test_the_quick_path_s_page_keeps_the_sheet_s_order_next_and_slots() -> None:
    check_sheet_renderer(blind_read_html)


@pytest.mark.parametrize(
    ("render", "rule"),
    [
        (lambda sheet: "\n".join(sorted(item.case_material for item in sheet.items)), "in order"),
        (lambda sheet: "\n".join(item.case_material for item in sheet.items[2:]), "next first"),
        (
            lambda sheet: "\n".join(
                f"{item.case_material}\n" + "\n".join(sorted(output.text for output in item.outputs))
                for item in sheet.items
            ),
            "slots as given",
        ),
    ],
    ids=["re-sorted items", "opens past next", "re-sorted slots"],
)
def test_a_renderer_that_re_sorts_or_skips_ahead_is_refused_by_the_rule_it_breaks(render: Any, rule: str) -> None:
    with pytest.raises(SheetConformanceFailure, match=rule):
        check_sheet_renderer(render)


def test_a_renderer_that_shows_from_next_on_keeps_the_sheet() -> None:
    check_sheet_renderer(
        lambda sheet: "\n".join(
            f"{item.case_material}\n" + "\n".join(f"{output.slot}: {output.text}" for output in item.outputs)
            for item in sheet.items
            if item.item not in sheet.answered
        )
    )


# --- the courier example: a host's side of the door ------------------------------------------------------------


def test_the_courier_s_sheet_page_keeps_the_sheet() -> None:
    check_sheet_renderer(courier_sheet_page)


async def test_the_courier_vouches_for_its_dispatch_desk_alone() -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read = off_loop(host.eval_host.storage.load_blind_read_with_etag, line["read_id"], TOYHOST_SCOPE)[0]
    assert read is not None

    assert courier_attests(scope_id=COURIER_SCOPE, caller=COURIER_DISPATCH_DESK, person=" Dana ") == "dana"
    assert courier_attests(scope_id=COURIER_SCOPE, caller="agent:someone", person="dana") is None
    assert courier_attests(scope_id=TOYHOST_SCOPE, caller=COURIER_DISPATCH_DESK, person="dana") is None
    refused = await _call(
        _with(host, attests_person=courier_attests),
        "blind_read_answer",
        read_id=read.id,
        item=read.items[0].item,
        person="dana",
        skip=True,
    )
    assert refused.is_error and refused.refusal is not None and refused.refusal.status == 403, "not its scope or desk"
