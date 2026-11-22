"""The claim read's host door: the actions a host's agent or page drives, onto the one core.

Pinned here, each as the refusal it is and not only as a happy path:

- the draw takes no seed: the parameter does not exist, two draws over one population take two seeds and two orders,
  and the line says which seed and how many were drawn before; the plan is the caller's, fixed before the draw;
- the sheet and every open line give a rater nothing that unblinds: no verdict, stratum or arm, no run, result or case
  id, and no figure;
- a person's answer is recorded only when the host attests the caller relaying it: otherwise refused (403) and nothing
  is written; a question the read does not ask, or an answer that is not pass or fail, is refused;
- the read is looked at once: close serves every figure and check, the arms the lean test could not read and the
  claims the draw left out, and a closed read refuses answers and a second close;
- a repeat grades the drawn claims' cells again on a judge, priced at its worst case before any call: over the cap it
  is refused with nothing spent; it stores its readings on the read, the run's judge as the instrument against itself
  and another as a second judge, each call ledgered under its purpose; a judge's repeat enters a read only from the
  read's own repeat job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel.claims.grading import criterion_fingerprint, criterion_steps, grade_tally
from threetears.evals.kernel.errors import EvalServiceError
from threetears.evals.ops.claim_reads import keep_repeats
from threetears.evals.ops.host import OpsHost
from threetears.evals.schema import ClaimItemRepeat, ClaimRead
from threetears.evals.schema.models import ClaimCriterion, EvalTrace, PageBudget, RepairBudget, eval_trace_doc_id
from packages.evals.tests.claim_read_support import graded_cell
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled

PERSON = "Ana@Example.com"
JUDGE = "scripted/claims-judge"
SECOND = "scripted/second-judge"
CRITERION = ClaimCriterion(
    name="almanac.claims",
    grounded_in=("run",),
    pages=PageBudget(per_document_chars=2000, total_chars=4000, window_chars=1000),
    repairs=RepairBudget(rounds=1, max_items_per_cell=2),
    judge="builtin",
)

#: Each case's claim reads this verdict; the doc number picks it, so every arm holds every stratum.
VERDICTS = ("supported", "contradicted", "unverifiable", "supported", "true_ungrounded", "supported")


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = JUDGE
    served_model: str | None = JUDGE
    input_tokens: int | None = 300
    output_tokens: int | None = 60
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None
    sent_settings: Any = None


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeClaimsJudge:
    """Answers every truth call with no citation and every ground call with none: each claim reads unverifiable and
    ungrounded. Records each call's reply schema."""

    model_name: str = JUDGE
    usd_per_call: float | None = 0.002
    asked: list[str] = field(default_factory=list)
    closed: int = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del system, user, response_format
        return self.usd_per_call

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del system, user
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        if name == "claim_truth":
            return _Reply(json.dumps({"claims": [{"claim": 0, "citations": [], "suspect": False}]}))
        return _Reply(json.dumps({"claims": [{"claim": 0, "citations": []}]}))

    async def aclose(self) -> None:
        self.closed += 1


def _attests(*, scope_id: str, caller: str, person: str) -> str | None:
    """The host's check: it vouches for :data:`CALLER` relaying anyone, under its own spelling of their name."""
    return person.lower() if caller == CALLER.identity else None


def _with(host: OpsHost, **changes: Any) -> OpsHost:
    launch = host.launch
    return replace(host, launch=replace(launch, eval_host=replace(launch.eval_host, **changes)))


def _world(*, usd_per_call: float | None = 0.002) -> tuple[OpsHost, OpsFixture, list[_FakeClaimsJudge]]:
    """The toy corpus campaign graded under a claim criterion: its template declares it, each run records the judge,
    and every result holds one claim, its verdict set by its case, its trace the grade and the evidence; two cells are
    graded under another definition. A host attesting :data:`CALLER` for anyone, its judges scripted."""
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    template = toyhost_template().model_copy(update={"claim_criteria": [CRITERION]})
    off_loop(storage.save_template, template)
    fingerprint = criterion_fingerprint(CRITERION, *criterion_steps(CRITERION, _FakeClaimsJudge()))
    redefined = shelves = 0
    for run_id in fixture.campaign.run_ids:
        run = off_loop(storage.load_eval_run, run_id, TOYHOST_SCOPE)
        assert run is not None
        off_loop(
            storage.save_eval_run,
            run.model_copy(update={"template_id": template.id, "judge_model": JUDGE, "judge_temperature": 0.0}),
        )
        for result in off_loop(storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE):
            verdict = VERDICTS[int(result.test_case_id.rsplit("-", 1)[1]) % len(VERDICTS)]
            other = redefined < 2 and verdict == "supported"
            redefined += int(other)
            # The built text names its cell; a shelf number stands for it, so no stored id reaches the sheet.
            shelves += 1
            cell = graded_cell(
                run_id,
                f"shelf {shelves}",
                [verdict],
                case_id=result.test_case_id,
                grounded=[verdict != "unverifiable"],
                fingerprint="b" * 64 if other else fingerprint,
            )
            tally = grade_tally(cell.grade, cell.evidence, CRITERION, derivation=None)
            trace = EvalTrace(
                id=eval_trace_doc_id(result.id),
                scope_id=TOYHOST_SCOPE,
                result_id=result.id,
                eval_run_id=run_id,
                claim_evidence=cell.evidence,
                claim_grades={CRITERION.name: cell.grade},
            )
            stored = result.model_copy(
                update={
                    "claim_tallies": {CRITERION.name: tally},
                    "claim_fingerprints": {CRITERION.name: cell.grade.criterion_fingerprint},
                    "has_trace": True,
                }
            )
            off_loop(storage.save_eval_result, stored, trace)
    built: list[_FakeClaimsJudge] = []

    def clients(role: str, model: str | None, *, temperature: float | None = None) -> _FakeClaimsJudge:
        client = _FakeClaimsJudge(model_name=model or JUDGE, usd_per_call=usd_per_call)
        built.append(client)
        return client

    return _with(fixture.host, attests_person=_attests, clients=clients), fixture, built


def _evals() -> MountedTool:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]


async def _call(host: OpsHost, action: str, caller: Caller = CALLER, **params: Any) -> Any:
    return await _evals().call({"action": action, **params}, host=host, caller=caller)


def _stored(host: OpsHost, read_id: str) -> ClaimRead:
    read = off_loop(host.eval_host.storage.load_claim_read_with_etag, read_id, TOYHOST_SCOPE)[0]
    assert read is not None
    return read


async def _drawn(host: OpsHost, fixture: OpsFixture, **params: Any) -> dict[str, Any]:
    outcome = await _call(
        host, "claim_read_create", claim_criterion=CRITERION.name, campaign_id=fixture.campaign.id, **params
    )
    assert not outcome.is_error, outcome.text
    return dict(outcome.structured)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


# --- the draw --------------------------------------------------------------------------------------------------


async def test_the_draw_takes_no_seed_and_a_second_draw_over_the_same_claims_takes_another() -> None:
    host, fixture, _ = _world()

    seeded = await _call(
        host, "claim_read_create", claim_criterion=CRITERION.name, campaign_id=fixture.campaign.id, seed=7
    )
    first = await _drawn(host, fixture)
    second = await _drawn(host, fixture)

    assert seeded.is_error, "a caller cannot choose the draw"
    assert (first["earlier_reads"], second["earlier_reads"]) == (0, 1)
    assert first["seed"] != second["seed"] and first["read_id"] != second["read_id"]
    one, two = _stored(host, first["read_id"]), _stored(host, second["read_id"])
    assert one.draw.population_digest == two.draw.population_digest
    assert [item.item for item in one.items] != [item.item for item in two.items]
    assert len(first["arms"]) == 2, "one arm per variant key the graded cells carry"
    assert first["questions"] == ["true", "grounded", "is_claim"], "grounded asked where a ground step read claims"
    assert first["excluded"] == {"redefined": 2}, "the cells of another definition are left out, counted"


async def test_a_plan_per_stratum_is_fixed_before_the_draw_and_one_past_its_stratum_is_refused() -> None:
    host, fixture, _ = _world()

    line = await _drawn(host, fixture, per_stratum={"contradicted": 3, "supported": 4})
    past = await _call(
        host,
        "claim_read_create",
        claim_criterion=CRITERION.name,
        campaign_id=fixture.campaign.id,
        per_stratum={"contradicted": 999},
    )
    both = await _call(
        host,
        "claim_read_create",
        claim_criterion=CRITERION.name,
        campaign_id=fixture.campaign.id,
        per_stratum={"contradicted": 3},
        claim_items=10,
    )
    unknown = await _call(host, "claim_read_create", claim_criterion="no.such", campaign_id=fixture.campaign.id)

    planned = {stratum["stratum"]: stratum["planned"] for stratum in line["strata"]}
    assert planned == {"contradicted": 3, "true_ungrounded": 0, "unverifiable": 0, "supported": 4}
    assert past.is_error and "planned past" in past.text, past.text
    assert both.is_error and "one of the two" in both.text, both.text
    assert unknown.is_error and "holds a stored grade" in unknown.text, unknown.text


# --- the sheet gives nothing away ------------------------------------------------------------------------------


async def test_the_sheet_and_the_open_line_hold_nothing_that_would_unblind_a_rater() -> None:
    host, fixture, _ = _world()
    line = await _drawn(host, fixture)
    read = _stored(host, line["read_id"])
    first = read.items[0]
    answered = await _call(
        host,
        "claim_read_answer",
        claim_read_id=read.id,
        item=first.item,
        person=PERSON,
        claim_answers={"true": "pass", "is_claim": "pass"},
    )
    assert not answered.is_error, answered.text

    sheet = await _call(host, "claim_read_sheet", claim_read_id=read.id, rater="ana@example.com")

    assert not sheet.is_error, sheet.text
    text = json.dumps(sheet.structured)
    forbidden = {
        *(arm.key for arm in read.arms),
        *(run_id for arm in read.arms for run_id in arm.run_ids),
        *(item.result_id for item in read.items),
        *(item.case_id for item in read.items),
        *(item.view_digest for item in read.items),
    }
    assert [value for value in forbidden if value in text] == []
    keys = _keys(sheet.structured)
    assert not keys & {"verdict", "stratum", "arm", "arms", "grounded", "inclusion", "raters"}
    assert sorted(key for key in keys if key.endswith("_id")) == ["read_id"]
    assert [item["item"] for item in sheet.structured["items"]] == [item.item for item in read.items]
    assert sheet.structured["answered"] == [first.item] and sheet.structured["next"] == read.items[1].item
    for open_line in (answered.structured, (await _call(host, "claim_read_get", claim_read_id=read.id)).structured):
        assert open_line["figures"] == [], "no figure while the read is open"
        assert open_line["raters"] == [{"rater": "ana@example.com", "made_by": "person", "answered": 1, "skipped": 0}]


# --- who may answer ----------------------------------------------------------------------------------------------


async def test_an_answer_the_host_does_not_attest_is_refused_and_nothing_is_written() -> None:
    host, fixture, _ = _world()
    line = await _drawn(host, fixture)
    item = _stored(host, line["read_id"]).items[0].item
    stranger = Caller(identity="someone-else", scope_id=TOYHOST_SCOPE)

    refused = await _call(
        host,
        "claim_read_answer",
        caller=stranger,
        claim_read_id=line["read_id"],
        item=item,
        person=PERSON,
        claim_answers={"true": "fail"},
    )
    unasked = await _call(
        host, "claim_read_answer", claim_read_id=line["read_id"], item=item, person=PERSON, claim_answers={"why": "x"}
    )
    worded = await _call(
        host, "claim_read_answer", claim_read_id=line["read_id"], item=item, person=PERSON, claim_answers={"true": "4"}
    )
    stray = await _call(
        host, "claim_read_answer", claim_read_id=line["read_id"], item="nope", person=PERSON, claim_answers={}
    )

    assert refused.is_error and "refused" in refused.text, refused.text
    assert unasked.is_error and "not one this read asks" in unasked.text, unasked.text
    assert worded.is_error and "pass or fail" in worded.text, worded.text
    assert stray.is_error and "no item" in stray.text, stray.text
    assert _stored(host, line["read_id"]).answers == []


# --- one look --------------------------------------------------------------------------------------------------


async def test_close_reads_every_figure_and_check_once_and_refuses_every_later_answer() -> None:
    host, fixture, _ = _world()
    line = await _drawn(host, fixture)
    read = _stored(host, line["read_id"])
    # People agree with the instrument on every item but the last, which nobody answers.
    for item in read.items[:-1]:
        fails = item.verdict == "contradicted"
        outcome = await _call(
            host,
            "claim_read_answer",
            claim_read_id=read.id,
            item=item.item,
            person=PERSON,
            claim_answers={
                "true": "fail" if fails else "pass",
                "grounded": "pass" if item.grounded else "fail",
                "is_claim": "pass",
            },
        )
        assert not outcome.is_error, outcome.text

    closed = await _call(host, "claim_read_close", claim_read_id=read.id)
    again = await _call(host, "claim_read_close", claim_read_id=read.id)
    late = await _call(
        host,
        "claim_read_answer",
        claim_read_id=read.id,
        item=read.items[-1].item,
        person=PERSON,
        claim_answers={"true": "pass"},
    )

    assert not closed.is_error, closed.text
    shown = closed.structured
    assert shown["closed_by"] == CALLER.identity and shown["closed_early"], "an item nobody answered closes it early"
    assert [figure["question"] for figure in shown["figures"]] == ["true", "grounded", "is_claim"]
    truth = shown["figures"][0]
    assert truth["items"] == len(read.items) - 1
    assert truth["kappa"]["value"] == 1.0, "people agree with the instrument on every answered item"
    assert (
        truth["check"]["instrument"]
        == shown["instrument"]
        == f"claims:{CRITERION.name}@{read.criterion_fingerprint[:12]}"
    )
    assert truth["check"]["criterion"] == f"{CRITERION.name}.true" and truth["check"]["read_id"] == read.id
    assert truth["lean_untested"] == truth["check"].get("lean_untested", []), "the check names the arms not tested"
    assert {arm["arm"] for arm in truth["arms"]} == {arm.key for arm in read.arms}
    assert shown["figures"][2]["check"] is None, "is_claim is people's alone"
    assert shown["excluded"] == {"redefined": 2}
    assert again.is_error and "closed" in again.text, again.text
    assert late.is_error and "closed" in late.text, late.text


# --- repeats ---------------------------------------------------------------------------------------------------


async def test_a_repeat_the_cap_cannot_pay_for_at_its_worst_case_is_refused_with_nothing_spent() -> None:
    host, fixture, built = _world(usd_per_call=50.0)
    line = await _drawn(host, fixture)

    refused = await _call(host, "claim_read_repeat", claim_read_id=line["read_id"])

    assert refused.is_error and "out-of-run cap" in refused.text, refused.text
    assert all(not client.asked for client in built), "no call is made"
    assert all(client.closed for client in built), "every client built is released"
    assert off_loop(host.eval_host.storage.query_out_of_run_spend, TOYHOST_SCOPE) == []
    assert _stored(host, line["read_id"]).repeat_job_id is None


async def test_a_repeat_stores_each_judges_reading_on_the_read_and_ledgers_each_call_under_its_purpose() -> None:
    host, fixture, _ = _world()
    line = await _drawn(host, fixture, per_stratum={"contradicted": 4, "supported": 4})
    read = _stored(host, line["read_id"])

    started = await _call(host, "claim_read_repeat", claim_read_id=read.id)
    assert not started.is_error, started.text
    status = await settled(host, started.structured["jobs"][0]["job_id"])
    second = await _call(host, "claim_read_repeat", claim_read_id=read.id, repeat_judge=SECOND)
    assert not second.is_error, second.text
    await settled(host, second.structured["jobs"][0]["job_id"])
    again = await _call(host, "claim_read_repeat", claim_read_id=read.id)

    assert status.done
    stored = _stored(host, read.id)
    assert len(stored.repeats) == 2 * len(read.items)
    by = {(repeat.judge_model, repeat.against) for repeat in stored.repeats}
    assert by == {(JUDGE, "itself"), (SECOND, "model")}
    assert all(repeat.verdict == "unverifiable" and repeat.grounded is False for repeat in stored.repeats)
    shown = (await _call(host, "claim_read_get", claim_read_id=read.id)).structured
    itself = next(reading for reading in shown["repeats"] if reading["against"] == "itself")
    truth = next(question for question in itself["questions"] if question["question"] == "true")
    contradicted = sum(1 for item in read.items if item.verdict == "contradicted")
    assert (truth["first_only"], truth["repeat_only"]) == (contradicted, 0)
    assert truth["agreed"] == len(read.items) - contradicted
    spends = off_loop(host.eval_host.storage.query_out_of_run_spend, TOYHOST_SCOPE)
    assert {spend.purpose for spend in spends} == {"judge", "second_judge"}
    assert all(spend.priced_ceiling_usd is not None for spend in spends), "every call priced as it was sent"
    assert again.is_error and "already" in again.text, "a judge that repeated every item is not asked again"


async def test_a_judges_repeat_enters_a_read_only_from_its_own_repeat_job() -> None:
    host, fixture, _ = _world()
    line = await _drawn(host, fixture)
    read = _stored(host, line["read_id"])
    forged = ClaimItemRepeat(
        item=read.items[0].item, repeat_id="forged", judge_model=JUDGE, against="itself", verdict="supported"
    )

    try:
        off_loop(keep_repeats, host.eval_host, read.id, TOYHOST_SCOPE, [forged])
    except EvalServiceError as refused:
        assert refused.status_code == 403
    else:
        raise AssertionError("a repeat from outside the read's repeat job was kept")
    assert _stored(host, read.id).repeats == []
