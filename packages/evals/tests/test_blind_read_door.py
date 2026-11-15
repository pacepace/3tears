"""The blind read's host door: the actions a host's agent or page drives, onto the one core.

Pinned here, each as the refusal it is and not only as a happy path:

- the draw takes no seed: the parameter does not exist, two draws over one population take two orders, and the line
  says how many were drawn before;
- the sheet and every line while the read is open give a rater nothing that unblinds: no arm, run, result or case
  id, no judge score, no label key, no count of who preferred what;
- a person's answer is recorded only when the host attests the caller relaying it for that person: with no check, or
  a check that declines, it is refused (403) and nothing is written; an attested one is the host's name for them;
- a model's answers enter a read only from inside the read's own review job: outside any job, inside another, or
  naming a model the read did not declare, they are refused (403) and nothing is written; the review job keeps them,
  named by the declared reviewer; a job that fails to start leaves the read free for the next review;
- a reviewer that is the judge's own model is refused at the draw;
- close stores the closed read before it writes a rating, so an answer landing meanwhile refuses it and no rating
  exists for a read still open; a rating write that fails reaches the caller, and closing again writes the rest;
- close writes each answer as a rating (a person's as theirs, naming the read and the relaying caller), refuses
  later answers, and a read closed before its planned length reads its preference by the anytime test;
- no response field is named with a word kept internal (anchor, standing, tier, ordinal), read off every response
  populated at every depth and off every field its model declares.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel.job_context import job_running
from threetears.evals.kernel.errors import EvalServiceError, StorageError
from threetears.evals.ops.analysis import CampaignChanges, campaign_update
from threetears.evals.ops.blind_reads import (
    BlindReadLine,
    BlindReadListing,
    blind_read_answer,
    blind_read_close,
    blind_read_create,
    blind_read_get,
    keep_review,
)
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.jobs import JobsStarted
from threetears.evals.schema import CalibrationRating, JudgedArtifact, JudgeEvidence, RubricDim, label_key_of
from threetears.evals.schema.blind_read import BlindAnswer, BlindRead, BlindReview, BlindSheet, BlindVerdict
from threetears.evals.quick.blind import blind_read_html
from packages.evals.tests.sheet_conformance_support import SheetConformanceFailure, check_sheet_renderer
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


def _schema_fields(model: type[Any]) -> set[str]:
    """Every field name a response model can carry, at every depth: its JSON schema's properties and its $defs'."""
    schema = model.model_json_schema()
    shapes = [schema, *schema.get("$defs", {}).values()]
    return {name for shape in shapes for name in shape.get("properties", {})}


def _stored(host: OpsHost, read_id: str) -> tuple[BlindRead, str]:
    read, etag = off_loop(host.eval_host.storage.load_blind_read_with_etag, read_id, TOYHOST_SCOPE)
    assert read is not None and etag is not None
    return read, etag


def _ratings(host: OpsHost, read: BlindRead) -> list[CalibrationRating]:
    return [
        rating
        for result_id in dict.fromkeys(output.result_id for item in read.items for output in item.outputs)
        for rating in off_loop(host.eval_host.storage.query_calibration_ratings, TOYHOST_SCOPE, result_id=result_id)
    ]


def _unblinding(read: BlindRead, sheet: Any) -> list[str]:
    """What in ``sheet`` (the action's JSON, read at every depth) would pair an output with its arm."""
    text = json.dumps(sheet)
    forbidden = {
        *(arm.key for arm in read.arms),
        *(run_id for arm in read.arms for run_id in arm.run_ids),
        *(output.result_id for item in read.items for output in item.outputs),
        *(item.case_id for item in read.items),
        *(output.output_fingerprint for item in read.items for output in item.outputs),
        *(fp for item in read.items for output in item.outputs for fp in output.criterion_fingerprints.values()),
    }
    leaked = [value for value in forbidden if value in text]
    keys = _keys(sheet)
    leaked += sorted(key for key in keys if any(word in key for word in ("fingerprint", "tally")))
    leaked += sorted(key for key in keys if key.endswith("_id") and key != "read_id")
    leaked += sorted({"score", "scores", "prefer", "control", "candidate", "arm", "arms", "rater", "raters"} & keys)
    return leaked


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
    assert sheet.structured["items"] and all(item["outputs"] for item in sheet.structured["items"])
    assert _unblinding(read, sheet.structured) == [], "the sheet names nothing that pairs an output with its arm"
    assert sheet.structured["answered"] == [first.item], "the rater's own progress, and nobody's answers"
    for open_line in (answered.structured, (await _call(host, "blind_read_get", read_id=read.id)).structured):
        assert open_line["checks"] == [] and open_line["preferences"] == [], "no reading while the read is open"
        assert open_line["raters"] == [
            {"rater": PERSON, "made_by": "person", "sample": "random", "answered": 1, "skipped": 0, "partial": 0}
        ]


async def test_no_response_field_carries_a_word_kept_internal() -> None:
    """Every response the door returns, each populated at every depth it has (a sheet with items and outputs, a closed
    line with raters, checks, preferences and a model's reading), and every field its model could carry."""
    host, fixture, _ = _world(reviewer_reply=_reply())
    created = await _pair_read(host, fixture, reviewer=REVIEWER)
    read, _ = _stored(host, created["read_id"])
    candidate = next(arm.key for arm in read.arms if arm.role == "candidate")
    pairs = [item for item in read.items[: read.draw.planned_items] if len(item.outputs) == 2]
    answers = []
    for item in pairs[:6]:
        slot = next(output.slot for output in item.outputs if output.arm == candidate)
        answers.append(
            await _call(
                host,
                "blind_read_answer",
                read_id=read.id,
                item=item.item,
                person=PERSON,
                verdicts={f"{output.slot}/{DIM.name}": "4" if output.slot == slot else "2" for output in item.outputs},
                prefer=slot,
            )
        )
    sheet = await _call(host, "blind_read_sheet", read_id=read.id, rater=PERSON)
    started = await _call(host, "blind_read_review", read_id=read.id)
    assert not started.is_error, started.text
    (handle,) = started.structured["jobs"]
    status = await settled(host, handle["job_id"])
    assert status.status == "completed", status
    got = await _call(host, "blind_read_get", read_id=read.id)
    closed = await _call(host, "blind_read_close", read_id=read.id)
    listing = await _call(host, "blind_reads_list")
    responses = [created, *(answer.structured for answer in answers), sheet.structured, started.structured]
    responses += [status.result, got.structured, closed.structured, listing.structured]

    assert not closed.is_error, closed.text
    line = closed.structured
    assert sheet.structured["items"] and all(item["outputs"] for item in sheet.structured["items"])
    assert {rater["made_by"] for rater in line["raters"]} == {"person", "model"}
    assert line["checks"] and line["preferences"] and line["reviews"], "the nested shapes are walked populated"
    assert _unblinding(read, sheet.structured) == [], "a populated sheet names nothing that pairs an output with an arm"
    fields = {key for response in responses for key in _keys(response)}
    assert "sample" in fields
    assert not [field for field in fields if any(word in field for word in INTERNAL_WORDS)], sorted(fields)
    # A field a response could carry but this read left empty (a pass/fail criterion's table) is read off its model.
    declared = {
        field for model in (BlindReadLine, BlindReadListing, BlindSheet, JobsStarted) for field in _schema_fields(model)
    }
    assert "table" in declared and "reviewer_checks" in declared
    assert not [field for field in declared if any(word in field for word in INTERNAL_WORDS)], sorted(declared)


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
    assert (answer.rater, answer.made_by, answer.collected_by, answer.prefer) == (
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
            made_by="model",
            verdicts=[BlindVerdict(slot="A", dim=DIM.name, score=5)],
        )
    ]
    review = BlindReview(reviewer=REVIEWER, asked=1)

    _, etag = _stored(host, read_id)

    with pytest.raises(EvalServiceError) as outside:
        off_loop(keep_review, host.eval_host, read_id, TOYHOST_SCOPE, forged, review)
    with job_running("someone-else"), pytest.raises(EvalServiceError) as other_job:
        # A thread carrying the job's context, as the job's own store calls run off the loop.
        await asyncio.to_thread(keep_review, host.eval_host, read_id, TOYHOST_SCOPE, forged, review)
    refused, refused_etag = _stored(host, read_id)

    assert (refused, refused_etag) == (read, etag), "nothing was written: the read is as drawn, its etag unmoved"
    assert (refused.answers, refused.reviews, _ratings(host, refused)) == ([], [], [])
    started = await _call(host, "blind_read_review", read_id=read_id)
    assert not started.is_error, started.text
    (handle,) = started.structured["jobs"]
    status = await settled(host, handle["job_id"])

    assert outside.value.status_code == 403 and other_job.value.status_code == 403
    assert status.status == "completed", status
    assert status.result is not None and status.result["read_id"] == read_id, "the job hands back the read's line"
    stored = off_loop(host.eval_host.storage.load_blind_read_with_etag, read_id, TOYHOST_SCOPE)[0]
    assert stored is not None and stored.review_job_id is not None
    assert {(answer.rater, answer.made_by) for answer in stored.answers} == {(REVIEWER, "model")}
    assert len(stored.answers) == len(stored.items), "every item answered, by the declared reviewer"
    assert [client.calls for client in built if client.calls], "the job made the calls"
    line = off_loop(blind_read_get, host.eval_host, read_id, TOYHOST_SCOPE)
    assert [(rater.rater, rater.made_by) for rater in line.raters] == [(REVIEWER, "model")]

    # Inside the read's own job, an answer naming any model but the declared reviewer is refused, and nothing written.
    reviewed, reviewed_etag = _stored(host, read_id)
    assert reviewed.review_job_id is not None
    stray = [forged[0].model_copy(update={"rater": "reviewer/a-stray-model"})]
    with job_running(reviewed.review_job_id), pytest.raises(EvalServiceError) as from_a_stray:
        await asyncio.to_thread(keep_review, host.eval_host, read_id, TOYHOST_SCOPE, stray, review)
    assert from_a_stray.value.status_code == 403 and "a-stray-model" in str(from_a_stray.value)
    assert _stored(host, read_id) == (reviewed, reviewed_etag)


async def test_a_review_whose_job_fails_to_start_leaves_the_read_free_for_the_next_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, fixture, _ = _world(reviewer_reply=_reply())
    created = await _call(host, "blind_read_create", run_ids=[fixture.campaign.run_ids[0]], reviewer=REVIEWER)
    assert not created.is_error, created.text
    read_id = created.structured["read_id"]
    storage = host.eval_host.storage
    save_job_record = storage.save_job_record
    refused: list[str] = []

    def refuse_the_first(document: dict[str, Any], /, *, if_match: str | None, create_only: bool = False) -> None:
        if not refused:
            refused.append(str(document.get("id")))
            raise StorageError("the job table is read-only for maintenance")
        save_job_record(document, if_match=if_match, create_only=create_only)

    monkeypatch.setattr(storage, "save_job_record", refuse_the_first)

    failed = await _call(host, "blind_read_review", read_id=read_id)
    after_failure, _ = _stored(host, read_id)
    started = await _call(host, "blind_read_review", read_id=read_id)

    assert failed.is_error and "read-only for maintenance" in failed.text, "the failed start reaches the caller"
    assert refused and after_failure.review_job_id is None, "the claim on a job that never started is released"
    assert not started.is_error, started.text
    (handle,) = started.structured["jobs"]
    status = await settled(host, handle["job_id"])
    assert status.status == "completed", status
    stored, _ = _stored(host, read_id)
    assert len(stored.answers) == len(stored.items), "the next review read every item"


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


async def test_an_answer_landing_as_the_read_closes_refuses_the_close_and_no_rating_exists_for_the_open_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read, _ = _stored(host, line["read_id"])
    first, second = read.items[0], read.items[1]
    answered = await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=first.item,
        person=PERSON,
        verdicts={f"{output.slot}/{DIM.name}": "4" for output in first.outputs},
    )
    assert not answered.is_error, answered.text
    storage = host.eval_host.storage
    load = storage.load_blind_read_with_etag
    landed: list[str] = []

    def load_then_another_answer_lands(read_id: str, scope_id: str) -> tuple[BlindRead | None, str | None]:
        loaded = load(read_id, scope_id)
        if not landed:
            # Between the close's load and its save, another person's answer is written.
            landed.append(second.item)
            blind_read_answer(
                host.eval_host,
                read_id,
                scope_id,
                collected_by=CALLER.identity,
                person="bo@example.com",
                item=second.item,
                verdicts={f"{output.slot}/{DIM.name}": "3" for output in second.outputs},
            )
        return loaded

    monkeypatch.setattr(storage, "load_blind_read_with_etag", load_then_another_answer_lands)
    refused = await _call(host, "blind_read_close", read_id=read.id)
    monkeypatch.setattr(storage, "load_blind_read_with_etag", load)
    still_open, _ = _stored(host, read.id)

    assert landed, "the answer landed inside the close"
    assert refused.is_error and refused.refusal is not None and refused.refusal.status == 409, refused.text
    assert still_open.closed_at is None and len(still_open.answers) == 2
    assert _ratings(host, still_open) == [], "no rating exists for a read that is still open"

    closed = await _call(host, "blind_read_close", read_id=read.id)

    assert not closed.is_error, closed.text
    assert closed.structured["ratings_written"] == len(first.outputs) + len(second.outputs), "asking again closes it"


async def test_a_close_whose_rating_write_fails_is_completed_by_closing_again(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The recovery path: the read is closed before its ratings are written; a rating write the store refuses reaches
    the caller as a 503 naming the remedy (close it again), the store's failure attached and logged; the next close
    writes what the first did not, then a further close is refused."""
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read, _ = _stored(host, line["read_id"])
    first = read.items[0]
    assert len(first.outputs) == 2
    await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=first.item,
        person=PERSON,
        verdicts={f"A/{DIM.name}": "4", f"B/{DIM.name}": "2"},
        prefer="A",
    )
    storage = host.eval_host.storage
    save_rating = storage.save_calibration_rating
    writes: list[str] = []

    def the_second_and_third_writes_fail(rating: CalibrationRating) -> None:
        writes.append(rating.id)
        if len(writes) in (2, 3):
            raise StorageError("the ratings table is read-only for maintenance")
        save_rating(rating)

    monkeypatch.setattr(storage, "save_calibration_rating", the_second_and_third_writes_fail)
    failed = await _call(host, "blind_read_close", read_id=read.id)
    after_failure, _ = _stored(host, read.id)
    partly = _ratings(host, after_failure)
    with pytest.raises(StorageError) as again_refused:
        off_loop(blind_read_close, host.eval_host, read.id, TOYHOST_SCOPE, closed_by=CALLER.identity)
    completed = await _call(host, "blind_read_close", read_id=read.id)
    again = await _call(host, "blind_read_close", read_id=read.id)

    assert failed.is_error and failed.refusal is not None
    assert (failed.refusal.kind, failed.refusal.status) == ("storage_failed", 503)
    assert "read-only for maintenance" in failed.text, "the store's own reason"
    assert "close the read again (blind_read_close)" in failed.text, "the refusal names the call that finishes it"
    assert after_failure.closed_at is not None and len(partly) == 1, "closed, with one of its two ratings written"
    cause = again_refused.value.__cause__
    assert isinstance(cause, StorageError) and "read-only for maintenance" in cause.message, "the cause is attached"
    assert [r for r in caplog.records if "ratings_unwritten" in r.getMessage() and r.exc_info], "and recorded"
    assert not completed.is_error, completed.text
    assert (completed.structured["ratings_written"], completed.structured["ratings_not_written"]) == (1, [])
    assert completed.structured["closed_at"] == after_failure.closed_at, "the read was closed once"
    assert again.is_error and again.refusal is not None and again.refusal.status == 409, "nothing is left to write"
    ratings = _ratings(host, after_failure)
    assert sorted((r.rater, r.rater_kind, r.source, r.score) for r in ratings) == [
        (PERSON, "person", f"blind_read:{read.id}", 2),
        (PERSON, "person", f"blind_read:{read.id}", 4),
    ]


async def test_a_closed_read_whose_missing_ratings_are_all_refused_says_why_and_not_to_draw_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host, fixture, _ = _world()
    line = await _pair_read(host, fixture)
    read, _ = _stored(host, line["read_id"])
    first = read.items[0]
    await _call(
        host,
        "blind_read_answer",
        read_id=read.id,
        item=first.item,
        person=PERSON,
        verdicts={f"{output.slot}/{DIM.name}": "4" for output in first.outputs},
    )
    storage = host.eval_host.storage
    save_rating = storage.save_calibration_rating
    lost: list[str] = []

    def the_first_write_fails(rating: CalibrationRating) -> None:
        if not lost:
            lost.append(rating.result_id)
            raise StorageError("the ratings table is read-only for maintenance")
        save_rating(rating)

    monkeypatch.setattr(storage, "save_calibration_rating", the_first_write_fails)
    failed = await _call(host, "blind_read_close", read_id=read.id)
    assert failed.is_error and failed.refusal is not None and failed.refusal.status == 503
    assert _ratings(host, read) == [], "the first refusal stopped the close before any rating was written"
    # The results the unwritten ratings are of are deleted before the read is closed again.
    gone = [output.result_id for output in first.outputs]
    assert all(off_loop(storage.delete_eval_result, result_id, TOYHOST_SCOPE) for result_id in gone)

    refused = await _call(host, "blind_read_close", read_id=read.id)

    assert refused.is_error and refused.refusal is not None and refused.refusal.status == 409, refused.text
    assert "draw another read" not in refused.text, "the read is not the problem: each rating is"
    assert "nothing is left to close" in refused.text
    assert all(result_id in refused.text for result_id in gone), "each refused rating, with why"


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
    """The courier renders the sheet as its page receives it: the blind_read_sheet action's JSON."""
    check_sheet_renderer(lambda sheet: courier_sheet_page(sheet.model_dump(mode="json")))


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
