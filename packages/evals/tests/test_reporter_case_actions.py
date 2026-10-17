"""The reporter case bank's operations and actions, driven over the toy host (#566).

A reporter run's cases are never generated: each is one campaign's analysis bundle frozen into a case of
an ``analysis_reporter`` template, so freezing is the step every reporter run starts from. Pinned here,
through the operations and through :meth:`MountedTool.call`, the one path every transport takes:

- **A freeze pins a campaign's bundle**, alone or with the memo the campaign got: the receipt names the
  stored case, the fingerprint of the bundle it froze and the limits the freeze recorded, and labels are
  stamped with the template's criterion. A second identical freeze answers with the same case.
- **Every refusal fires**: an unknown template, a template of another kind (before anything is stored),
  a template or campaign of another scope, labels with no memo to be about.
- **The bank is curatable**: a listing says which case each pair launches, what superseded or retired the
  rest, which cases this build cannot read (listed, never refused) and which pairs hold two live cases; a
  case is retired and restored through its own action.
- **A reporter run counts the turns it delivered**: each generator call that returned. A memo that failed
  validation after its billed calls returned keeps their cost; one whose first call was refused delivered
  none, and its cell reads as one where no result took a turn.
- **A memo titling an arm its readings omit fails its case** (#829): the generator refuses it, naming the arm, and
  the cell records that refusal as the candidate's own failure.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.actions import Caller, eval_catalogue, standard_tools
from threetears.evals.analysis import REPORTER_KIND, ReporterKind, assemble_context_bundle
from threetears.evals.analysis.reporter_kind import REPORTER_CASE_KEY, LabelCriterion, reporter_case_of
from threetears.evals.schema import EvalResult
from threetears.evals.run.budget import BudgetStoppedError, EvalRunCostCap
from threetears.evals.kernel import NotFoundError, ValidationFailedError, delivered_a_turn
from threetears.evals.schema.models import EvalTestCase
from threetears.evals.ops import (
    FrozenReporterCase,
    ReporterCaseListing,
    analysis_generate,
    reporter_case_archive,
    reporter_cases_list,
)
from threetears.evals.run.runner import RunnerOptions, execute_run
from packages.evals.tests.factories import make_campaign, make_eval_run
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, alias_at, memo_payload
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.reporter_case_support import (
    GROUNDED,
    freeze_corpus_campaign,
    reporter_fixture,
    reporter_template,
)

ELSEWHERE = "another-scope"


async def _analysis_id(fixture: OpsFixture) -> str:
    """Generate the corpus campaign's analysis through its operation, and return the stored analysis."""
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
    assert analysis_id is not None
    return analysis_id


def _stored_cases(fixture: OpsFixture, template_id: str) -> list[EvalTestCase]:
    return fixture.host.eval_host.storage.query_test_cases(TOYHOST_SCOPE, template_id=template_id)


def _tools() -> dict[str, Any]:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}


# =============================================================================
# Freezing
# =============================================================================


def test_a_freeze_without_an_analysis_pins_the_bundle_alone() -> None:
    fixture = reporter_fixture()
    eval_host = fixture.host.eval_host
    receipt = freeze_corpus_campaign(fixture)

    (stored,) = _stored_cases(fixture, reporter_template().id)
    assert receipt.test_case_id == stored.id and stored.scope_id == TOYHOST_SCOPE
    case = reporter_case_of(stored)
    assert case is not None
    bundle = assemble_context_bundle(fixture.campaign, storage=eval_host.storage, profile=eval_host.profile)
    assert bundle.run_ids, "a bundle resolving no runs is refused, so the fixture must carry some"
    assert receipt.bundle_fingerprint == case.bundle_fingerprint == bundle.fingerprint()
    assert receipt.limits == case.limits
    assert (receipt.template_id, receipt.source_campaign_id) == (reporter_template().id, fixture.campaign.id)
    assert (receipt.recorded_analysis_id, receipt.writer_message_check, receipt.labels) == (None, None, [])

    again = freeze_corpus_campaign(fixture)
    assert again.test_case_id == receipt.test_case_id, "an identical freeze answers with the pair's live case"
    assert len(_stored_cases(fixture, reporter_template().id)) == 1


async def test_a_freeze_with_an_analysis_pins_its_memo_and_stamps_each_label_with_the_criterion() -> None:
    fixture = off_loop(reporter_fixture)
    analysis_id = await _analysis_id(fixture)
    # The reader's words verbatim, surrounding whitespace included: nothing on the way tidies a quote.
    label = {"dimension": GROUNDED, "direction": "high", "quote": "  every number traces to a run\n"}
    receipt = off_loop(freeze_corpus_campaign, fixture, recorded_analysis_id=analysis_id, labels=[label])

    (stored,) = off_loop(_stored_cases, fixture, reporter_template().id)
    case = reporter_case_of(stored)
    assert case is not None and case.recorded_memo, "the memo the campaign got is frozen with the case"
    assert receipt.test_case_id == stored.id and receipt.recorded_analysis_id == analysis_id
    assert receipt.bundle_fingerprint == case.bundle_fingerprint
    assert receipt.limits == case.limits
    assert receipt.writer_message_check is not None, "a pinned memo freezes the message its writer read"
    (frozen_label,) = receipt.labels
    assert (frozen_label.dimension, frozen_label.direction, frozen_label.quote) == (
        GROUNDED,
        "high",
        label["quote"],
    )
    assert frozen_label.criterion == LabelCriterion.of(reporter_template().rubric[0])


# =============================================================================
# Refusals
# =============================================================================


def test_an_unknown_template_is_refused() -> None:
    with pytest.raises(NotFoundError, match="template"):
        freeze_corpus_campaign(reporter_fixture(), template_id="no-such-template")


def test_a_template_of_another_kind_is_refused_before_anything_is_stored() -> None:
    fixture = reporter_fixture()
    other = toyhost_template()
    assert other.candidate_kind != REPORTER_KIND
    with pytest.raises(ValidationFailedError, match=f"is of kind '{other.candidate_kind}', not '{REPORTER_KIND}'"):
        freeze_corpus_campaign(fixture, template_id=other.id)
    assert _stored_cases(fixture, other.id) == []


def test_a_campaign_of_another_scope_does_not_resolve() -> None:
    fixture = reporter_fixture()
    storage = fixture.host.eval_host.storage
    elsewhere = fixture.campaign.model_copy(update={"id": "campaign-elsewhere", "scope_id": ELSEWHERE})
    storage.save_campaign(elsewhere)
    assert storage.load_campaign(elsewhere.id, ELSEWHERE) is not None, "it exists — in its own scope"

    with pytest.raises(NotFoundError, match="campaign"):
        freeze_corpus_campaign(fixture, campaign_id=elsewhere.id)
    assert _stored_cases(fixture, reporter_template().id) == []


def test_a_template_of_another_scope_does_not_resolve() -> None:
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    storage.save_template(reporter_template(ELSEWHERE))
    assert storage.load_template(reporter_template().id, ELSEWHERE) is not None, "it exists — in its own scope"

    with pytest.raises(NotFoundError, match=f"template '{reporter_template().id}' not found"):
        freeze_corpus_campaign(fixture)


def test_labels_with_no_memo_to_be_about_are_refused() -> None:
    with pytest.raises(ValidationFailedError, match="pass the analysis_id"):
        freeze_corpus_campaign(
            reporter_fixture(), labels=[{"dimension": GROUNDED, "direction": "low", "quote": "vague"}]
        )


# =============================================================================
# Through the catalogue
# =============================================================================


async def test_an_agent_freezes_a_case_through_the_action_and_reads_its_receipt() -> None:
    fixture = off_loop(reporter_fixture)
    tools = _tools()
    action = tools["evals"].action("reporter_case_freeze")
    assert action is not None and action.permission == "write"

    outcome = await tools["evals"].call(
        {"action": "reporter_case_freeze", "template_id": reporter_template().id, "campaign_id": fixture.campaign.id},
        host=fixture.host,
        caller=CALLER,
    )
    assert not outcome.is_error, outcome.text
    receipt = FrozenReporterCase.model_validate(outcome.structured)
    (stored,) = off_loop(_stored_cases, fixture, reporter_template().id)
    assert receipt.test_case_id == stored.id
    assert {
        f"test_case_id: {stored.id}",
        f"template_id: {reporter_template().id}",
        f"bundle_fingerprint: {receipt.bundle_fingerprint}",
        f"limits: [{', '.join(receipt.limits)}]",
    } <= set(outcome.text.splitlines()), "the receipt's text carries its facts"


async def test_the_actions_refusals_teach() -> None:
    fixture = off_loop(reporter_fixture)
    tools = _tools()
    other_kind = await tools["evals"].call(
        {"action": "reporter_case_freeze", "template_id": toyhost_template().id, "campaign_id": fixture.campaign.id},
        host=fixture.host,
        caller=CALLER,
    )
    assert other_kind.is_error
    assert other_kind.text.startswith("refused: reporter_case_freeze: template")
    assert f"not '{REPORTER_KIND}'" in other_kind.text

    # The scope is the caller's: an agent in another scope reaches neither the template nor the campaign.
    another = await tools["evals"].call(
        {"action": "reporter_case_freeze", "template_id": reporter_template().id, "campaign_id": fixture.campaign.id},
        host=fixture.host,
        caller=Caller(scope_id=ELSEWHERE, identity="agent:elsewhere"),
    )
    assert another.is_error and f"template '{reporter_template().id}' not found" in another.text
    assert off_loop(_stored_cases, fixture, reporter_template().id) == []


# =============================================================================
# The bank: listed, retired, restored
# =============================================================================


async def test_the_listing_says_which_case_launches_and_what_replaced_or_retired_the_rest() -> None:
    fixture = off_loop(reporter_fixture)
    eval_host = fixture.host.eval_host
    analysis_id = await _analysis_id(fixture)
    bundle_only = off_loop(freeze_corpus_campaign, fixture)
    first = off_loop(
        freeze_corpus_campaign,
        fixture,
        recorded_analysis_id=analysis_id,
        labels=[{"dimension": GROUNDED, "direction": "high", "quote": "traces"}],
    )
    with pytest.raises(ValidationFailedError, match="supersedes"):
        off_loop(
            freeze_corpus_campaign,
            fixture,
            recorded_analysis_id=analysis_id,
            labels=[{"dimension": GROUNDED, "direction": "low", "quote": "on reflection, no"}],
        )
    revised = off_loop(
        freeze_corpus_campaign,
        fixture,
        recorded_analysis_id=analysis_id,
        labels=[{"dimension": GROUNDED, "direction": "low", "quote": "on reflection, no"}],
        supersedes=[first.test_case_id],
    )

    listing = off_loop(reporter_cases_list, eval_host, reporter_template().id, TOYHOST_SCOPE)
    states = {entry.case.test_case_id: (entry.live, entry.superseded_by) for entry in listing.cases}
    assert states == {
        bundle_only.test_case_id: (True, []),
        first.test_case_id: (False, [revised.test_case_id]),
        revised.test_case_id: (True, []),
    }
    assert (listing.unreadable, listing.ambiguous) == ([], [])

    retired = off_loop(
        reporter_case_archive,
        eval_host,
        bundle_only.test_case_id,
        TOYHOST_SCOPE,
        archived=True,
        reason="bundle orphaned",
    )
    assert (retired.archived, retired.archived_reason) == (True, "bundle orphaned")
    active = off_loop(reporter_cases_list, eval_host, reporter_template().id, TOYHOST_SCOPE)
    assert bundle_only.test_case_id not in {entry.case.test_case_id for entry in active.cases}
    everything = off_loop(reporter_cases_list, eval_host, reporter_template().id, TOYHOST_SCOPE, include_archived=True)
    (entry,) = [entry for entry in everything.cases if entry.case.test_case_id == bundle_only.test_case_id]
    assert not entry.live and entry.case.archived

    restored = off_loop(reporter_case_archive, eval_host, bundle_only.test_case_id, TOYHOST_SCOPE, archived=False)
    assert (restored.archived, restored.archived_reason) == (False, None)


def test_a_case_this_build_cannot_read_is_listed_not_refused_and_two_live_cases_of_a_pair_are_named() -> None:
    fixture = reporter_fixture()
    eval_host = fixture.host.eval_host
    storage = eval_host.storage
    live = freeze_corpus_campaign(fixture)
    (stored,) = _stored_cases(fixture, reporter_template().id)
    # A second live case of the same pair, as two racing supersessions leave one.
    twin = stored.model_copy(update={"id": "case-twin"})
    storage.save_test_case(twin)
    listing = reporter_cases_list(eval_host, reporter_template().id, TOYHOST_SCOPE)
    (pair,) = listing.ambiguous
    assert (pair.campaign_id, pair.recorded_analysis_id) == (fixture.campaign.id, None)
    assert pair.live_case_ids == sorted([live.test_case_id, twin.id])

    unreadable = EvalTestCase(
        id="case-unreadable",
        scope_id=TOYHOST_SCOPE,
        template_id=reporter_template().id,
        host_payload={REPORTER_CASE_KEY: {"bundle_fingerprint": "fp-only"}},
    )
    storage.save_test_case(unreadable)
    listing = reporter_cases_list(eval_host, reporter_template().id, TOYHOST_SCOPE)
    assert [row.test_case_id for row in listing.unreadable] == ["case-unreadable"]
    # The freeze decides from liveness, so it refuses where the listing reads on.
    with pytest.raises(ValidationFailedError, match="cannot read"):
        freeze_corpus_campaign(fixture)


def test_the_listing_refuses_a_template_that_is_not_a_reporters() -> None:
    fixture = reporter_fixture()
    with pytest.raises(NotFoundError, match="template"):
        reporter_cases_list(fixture.host.eval_host, "no-such-template", TOYHOST_SCOPE)
    with pytest.raises(ValidationFailedError, match=f"not '{REPORTER_KIND}'"):
        reporter_cases_list(fixture.host.eval_host, toyhost_template().id, TOYHOST_SCOPE)


async def test_an_agent_lists_retires_and_restores_through_the_actions() -> None:
    fixture = off_loop(reporter_fixture)
    tools = _tools()
    assert tools["evals"].action("reporter_cases_list") is not None
    archive = tools["evals"].action("reporter_case_archive")
    assert archive is not None and archive.permission == "write"
    case_id = off_loop(freeze_corpus_campaign, fixture).test_case_id

    listed = await tools["evals"].call(
        {"action": "reporter_cases_list", "template_id": reporter_template().id}, host=fixture.host, caller=CALLER
    )
    assert not listed.is_error, listed.text
    assert [entry.case.test_case_id for entry in ReporterCaseListing.model_validate(listed.structured).cases] == [
        case_id
    ]
    (line,) = [line for line in listed.text.splitlines() if case_id in line]
    assert "live=True" in line and fixture.campaign.id in line

    retired = await tools["evals"].call(
        {"action": "reporter_case_archive", "reporter_case_id": case_id, "archive_reason": "orphaned"},
        host=fixture.host,
        caller=CALLER,
    )
    assert not retired.is_error, retired.text
    assert FrozenReporterCase.model_validate(retired.structured).archived
    assert {"archived: True", "archived_reason: orphaned"} <= set(retired.text.splitlines())

    listed = await tools["evals"].call(
        {"action": "reporter_cases_list", "template_id": reporter_template().id, "include_archived": True},
        host=fixture.host,
        caller=CALLER,
    )
    (line,) = [line for line in listed.text.splitlines() if case_id in line]
    assert "live=False" in line and '"archived": true' in line and '"archived_reason": "orphaned"' in line


# =============================================================================
# A memo that failed after its generator calls returned keeps their cost
# =============================================================================


class _RefusedWriter(FixturedClient):
    """A generator client whose provider refuses every call: nothing returns, nothing is billed."""

    async def generate(self, *, system: str, user: str, response_format: Any = None, tools: Any = None) -> Any:
        """Refuse the call."""
        raise RuntimeError("the provider refused the request")


async def _reporter_result(client: FixturedClient, *, cap: EvalRunCostCap | None = None) -> tuple[EvalResult, Any]:
    """Run the reporter kind once over a frozen case with ``client``, and return its result and a bundle over it.

    Under ``cap`` the run is held to it as a launch holds one, and a stop it causes is left to the caller to see on
    the result.
    """
    fixture = off_loop(reporter_fixture)
    host = fixture.host.eval_host
    receipt = off_loop(freeze_corpus_campaign, fixture)
    case = off_loop(host.storage.load_test_case, receipt.test_case_id, TOYHOST_SCOPE)
    template = off_loop(host.storage.load_template, reporter_template().id, TOYHOST_SCOPE)
    assert case is not None and template is not None
    kind = ReporterKind(prompt=PROMPT, prompt_id=PROMPT_ID, prompt_version=None, client=client, model=MODEL, host=host)
    run = make_eval_run(
        id="reporter-run",
        scope_id=TOYHOST_SCOPE,
        template_id=template.id,
        candidate_model=MODEL,
        k_runs=1,
        test_case_ids=[case.id],
    )
    off_loop(host.storage.save_eval_run, run)
    options = RunnerOptions(candidate_kinds={REPORTER_KIND: lambda _cell: kind})
    try:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=None,
            options=options,
            budget_gate=cap.check if cap is not None else None,
            on_cost=cap.record if cap is not None else None,
        )
    except BudgetStoppedError:
        assert cap is not None
    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, TOYHOST_SCOPE)
    campaign = make_campaign(scope_id=TOYHOST_SCOPE, run_ids=[run.id])
    return result, off_loop(assemble_context_bundle, campaign, storage=host.storage, profile=host.profile)


async def test_a_memo_refused_after_its_billed_calls_returned_keeps_their_cost() -> None:
    """Both generator calls returned and were billed, then the memo failed validation: two turns delivered."""
    client = FixturedClient("this is not a memo")
    client.completion.cost_usd = 0.15
    result, bundle = await _reporter_result(client)

    assert result.candidate_error is not None and result.infra_error is None
    assert (result.turns_delivered, result.cost_usd) == (2, pytest.approx(0.30))
    assert delivered_a_turn(result), "the calls that returned are the memo's time and spend"
    (cell,) = bundle.cell_measures
    assert (cell.n_candidate_failed, cell.n_no_turn, cell.all_failed) == (1, 0, False)
    cost = next(summary for summary in cell.measures.measures if summary.name == "cost_usd")
    # Measuring spend, read over every result billed (the pivot's, the history's and a run summary's rule).
    assert (cost.mean, cost.n, cost.population) == (pytest.approx(0.30), 1, "all_observed")


async def test_a_generator_completion_whose_sent_settings_is_mistyped_keeps_its_cost_and_is_never_read() -> None:
    """The reporter's generator is the candidate, whose rows carry no sent settings: what its completion says it was
    sent with is never read, so a wrong-typed value there neither fails the cell nor loses the billed calls."""
    client = FixturedClient("this is not a memo")
    client.completion.cost_usd = 0.15
    client.completion.sent_settings = "medium"  # type: ignore[attr-defined]  # a host's completion filling it wrong
    result, _bundle = await _reporter_result(client)

    assert result.infra_error is None, "the candidate's completion is not read for what it was sent with"
    assert (result.turns_delivered, result.cost_usd) == (2, pytest.approx(0.30))
    (row,) = [row for row in result.usage if row.role == "candidate"]
    assert (row.call_count, row.sent_settings) == (2, None)


async def test_a_memo_whose_first_call_was_refused_delivered_no_turn() -> None:
    result, bundle = await _reporter_result(_RefusedWriter(""))

    assert result.candidate_error is not None
    assert result.turns_delivered == 0
    assert not delivered_a_turn(result)
    (cell,) = bundle.cell_measures
    assert (cell.n_candidate_failed, cell.n_no_turn, cell.all_failed) == (1, 1, True)


async def test_a_generator_repair_past_the_runs_cap_is_sent_and_both_calls_keep_their_cost_for_the_cap_to_read() -> (
    None
):
    """The repair's $0.20 ceiling, on top of the $0.15 the first call cost, is past a $0.30 cap, and it is sent anyway.

    The cap is read between cells, against what has landed; inside a cell it refuses nothing, so the generator's
    repair goes out, the memo fails on its own merits, and the cap counts both calls once the cell has landed.
    """
    client = FixturedClient("this is not a memo")
    client.completion.cost_usd = 0.15
    client.ceiling_usd = 0.20
    cap = EvalRunCostCap("reporter-run", 0.30, enabled=True)

    result, _bundle = await _reporter_result(client, cap=cap)

    assert result.turns_delivered == 2, "the repair call was sent: nothing inside a cell is refused"
    assert result.cost_usd == pytest.approx(0.30)
    assert result.candidate_error is not None, "the memo failed validation, the candidate's own failure"
    assert result.infra_error is None, "the cap refused nothing, so there is no apparatus fault to record"
    assert cap.accumulated_usd == pytest.approx(0.30), "the cap counts both calls once the cell has landed"


# =============================================================================
# A memo titling an arm its readings omit fails its case
# =============================================================================


async def test_a_memo_titling_an_arm_its_readings_omit_fails_its_case_naming_the_arm() -> None:
    """The corpus case's memo titles the wide arm by its level and cites the narrow arm's reading only (#829).

    The production instance: a finding titled for two arms' cost over readings at two others. The reporter's
    generator refuses it on both calls, so the case's cell is the candidate's failure, and the failure names the arm
    as the title wrote it.
    """
    fixture = off_loop(reporter_fixture)
    host = fixture.host.eval_host
    bundle = off_loop(assemble_context_bundle, fixture.campaign, storage=host.storage, profile=host.profile)
    payload = memo_payload(bundle)
    payload["findings"][0] |= {
        "title": "The 1024tok chunk is the slower one",
        "evidence": [{"cell": alias_at(bundle, TOYHOST_NARROW), "measure_id": "total_ms", "reading": "measure"}],
    }

    result, _bundle = await _reporter_result(FixturedClient(json.dumps(payload)))

    assert result.turns_delivered == 2, "refused once, repaired once, refused again"
    error = result.candidate_error
    assert error is not None and result.infra_error is None
    assert 'findings[0].title names "1024tok" (the arm chunk_tokens=1024tok at cell(s) ' in error
    assert "none of that finding's evidence rows or trials is taken at that arm" in error
