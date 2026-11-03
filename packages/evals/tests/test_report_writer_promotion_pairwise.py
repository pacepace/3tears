"""The promotion read under its pre-registered primary reading: the pairwise preference, gated by every other on
regression only.

Each campaign is the promotion read's scripted one (a control and a candidate over the toy host's twelve held-out
documents, scored by one judge), with every result's page stored as its judge evidence and the scoring judge scripted
by a policy over which slot holds the candidate's page. The pairwise pass runs as its job, and the read is handed that
job's status, as the action hands it what ``job_poll`` answers. Every primary verdict a branch rests on is checked
against the bounded test and interval on the shares the test computes from the policy, so the branch is the
computation's, not the pass's word. No model is called.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_DOCUMENTS, TOYHOST_SCOPE
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, ops_fixture
from packages.evals.tests.writer_promotion_support import (
    BETTER,
    CANDIDATE,
    CONTROL,
    HELD_OUT,
    SCORING_JUDGE,
    WriterMeasures,
    promotion_campaign,
)
from threetears.evals.actions import eval_catalogue, read_only_tools, standard_tools
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_separation_p,
)
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.ops import JobsStarted, JobStatus, OpsHost, await_job
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND
from threetears.evals.ops.report_writer.writer_pairwise import WriterPairwiseReport, writer_pairwise
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion, writer_promotion
from threetears.evals.run.judge import CANNOT_TELL
from threetears.evals.schema.models import EvalTemplate, EvalTrace, JudgedArtifact, JudgeEvidence, eval_trace_doc_id

#: The template every run is pointed at, so the judge's prompt carries its intent.
_TEMPLATE = "writer-template"
#: The share range the primary reading is on.
_RANGE = (0.0, 1.0)
#: The judge's answer, given the document and the slot (A or B) holding the candidate's page: a 1-5 score, or
#: CANNOT_TELL.
Policy = Callable[[str, str], int | str]

#: A candidate shown more accurate and shown worse on cost and time: a gated reading regressed.
_TRADE: dict[str, Any] = {"cost_usd": 0.03, "total_ms": 1400.0, "field_accuracy": 0.91, "layout_fidelity": None}

#: A guardrail measure: the share of the snapshot's private rows a page leaked, fewer is better, on a declared range
#: and with a margin twelve cases can decide either way: nothing leaked on either arm is held, and every row leaked by
#: the candidate is breached.
_LEAKS = MetricDescriptor(
    name="leaked_rows",
    reader_name="Private rows leaked",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="The share of the snapshot's private rows the page leaked.",
    higher_is_better=False,
    materiality_threshold=0.35,
    guardrail=True,
    value_range=(0.0, 1.0),
)


def _leaks(control: float, candidate: float) -> tuple[WriterMeasures, WriterMeasures]:
    """Each arm's share of private rows leaked on every document, control first."""
    return (
        {document: {_LEAKS.name: control} for document in TOYHOST_DOCUMENTS},
        {document: {_LEAKS.name: candidate} for document in TOYHOST_DOCUMENTS},
    )


def _prefers_the_candidate(_document: str, slot: str) -> int:
    return 1 if slot == "A" else 5


def _prefers_the_control(_document: str, slot: str) -> int:
    return 5 if slot == "A" else 1


def _ties(_document: str, _slot: str) -> int:
    return 3


#: The documents a judge that can tell on few of them scores: one under the cases a rate needs.
_TOLD = frozenset(sorted(TOYHOST_DOCUMENTS)[: MIN_CASES - 1])


def _tells_on_few(document: str, slot: str) -> int | str:
    """Prefers the candidate on the documents in :data:`_TOLD`, and cannot tell on every other."""
    return _prefers_the_candidate(document, slot) if document in _TOLD else CANNOT_TELL


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    temperature: float | None = None
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = SCORING_JUDGE
    served_model: str | None = SCORING_JUDGE


@dataclass
class _PairJudge:
    """The scoring judge: reads which slot holds the candidate's page off the pages it is shown, answers by policy."""

    policy: Policy
    calls: list[tuple[str, str]] = field(default_factory=list)
    model_name: str = SCORING_JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        return 0.01

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> _Completion:
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        first = re.search(r"=== Report A ===\n(.*?)\n\n=== Report B ===", user, re.S).group(1)  # type: ignore[union-attr]
        document = re.search(r" page on (\S+),", first).group(1)  # type: ignore[union-attr]
        slot = "A" if first.startswith("# candidate") else "B"
        self.calls.append((document, slot))
        return _Completion(
            content=json.dumps({"reasoning": "read both", "criteria_scores": {dim: self.policy(document, slot)}})
        )

    async def aclose(self) -> None:
        """Nothing to release."""


def _campaign(policy: Policy, candidate: dict[str, Any] = BETTER, **campaign: Any) -> tuple[EvalHost, str]:
    """The promotion read's campaign with every result's page stored and the judge answering by ``policy``."""
    host, campaign_id = off_loop(promotion_campaign, candidate, **campaign)
    off_loop(_store_pages, host, campaign_id)
    judge = _PairJudge(policy=policy)
    return dataclasses.replace(host, clients=lambda _role, _model, *, temperature=None: judge), campaign_id


def _store_pages(host: EvalHost, campaign_id: str) -> None:
    storage = host.storage
    storage.save_template(
        EvalTemplate(
            id=_TEMPLATE, scope_id=TOYHOST_SCOPE, name="writer", intent="Write the report.", candidate_kind=WRITER_KIND
        )
    )
    stored = storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    for arm, run_id in zip(("control", "candidate"), stored.run_ids, strict=True):
        run = storage.load_eval_run(run_id, TOYHOST_SCOPE)
        assert run is not None
        storage.save_eval_run(run.model_copy(update={"template_id": _TEMPLATE}))
        for result in storage.query_eval_results(TOYHOST_SCOPE, run_id=run_id):
            page = f"# {arm} page on {result.test_case_id}, repeat {result.k_iteration}\n\nThe {arm} arm's findings."
            evidence = JudgeEvidence(case_material=f"Computed facts of {result.test_case_id}.", artifact=page)
            trace = EvalTrace(
                id=eval_trace_doc_id(result.id),
                scope_id=TOYHOST_SCOPE,
                result_id=result.id,
                eval_run_id=run_id,
                judge_evidence=evidence,
                judged_artifact=JudgedArtifact.DOCUMENT,
            )
            storage.save_eval_result(result, trace)


def _apart(model: str) -> str:
    """A host's reading of a model's family: the scoring judge in one, every writer in another."""
    return "judges" if model == SCORING_JUDGE else "writers"


def _ops(host: EvalHost) -> OpsHost:
    launch = off_loop(ops_fixture, generation=False).host.launch
    return OpsHost(launch=replace(launch, eval_host=host), model_family=_apart)


async def _pass(host: EvalHost, campaign_id: str) -> JobStatus:
    """Run the pass as its job and return the job's status once it completed, as ``job_poll`` answers it."""
    ops = _ops(host)
    (status,) = await await_job(
        ops, await writer_pairwise(ops, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE), TOYHOST_SCOPE
    )
    assert status.state == "completed" and status.result is not None, status.detail
    return status


def _read(host: EvalHost, campaign_id: str, status: JobStatus | None) -> WriterPromotion:
    return off_loop(writer_promotion, host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE, pairwise_job=status)


def _computed(policy: Policy) -> tuple[str, str]:
    """The primary verdict's outcome and reason, from the bounded test and interval on the policy's shares.

    A document's share is the mean of its two orders' shares, (5-s)/4 with the candidate as Report A and (s-1)/4 as
    B; a document the judge cannot tell on has none. Its repeats all score alike, so they leave the mean as it is.
    """
    shares = []
    for document in sorted(TOYHOST_DOCUMENTS):
        scored = [(policy(document, slot), slot) for slot in ("A", "B")]
        told = [
            Fraction(5 - int(s), 4) if slot == "A" else Fraction(int(s) - 1, 4)
            for s, slot in scored
            if s != CANNOT_TELL
        ]
        if told:
            shares.append(float(sum(told, Fraction(0)) / len(told)))
    control = [1.0 - share for share in shares]
    p = bounded_separation_p(control, shares, paired=True, value_range=_RANGE)
    interval = bounded_difference_interval(control, shares, paired=True, value_range=_RANGE, confidence=INTERVAL_LEVEL)
    assert p is not None and interval is not None
    separated = p < SIGNIFICANCE_ALPHA and (interval[0] > 0 or interval[1] < 0)
    outcome = ("improved" if interval[0] > 0 else "regressed") if separated else "not_separated"
    reason = "too_few_cases" if len(shares) < MIN_CASES else ("separated" if separated else "no_margin")
    return outcome, reason


def _outcomes(read: WriterPromotion, kind: str) -> Mapping[str, str]:
    return {verdict.name: verdict.outcome for verdict in read.verdicts if verdict.kind == kind}


# --- the four branches that decide ---------------------------------------------------------------------------------


async def test_a_preferred_candidate_with_no_gated_regression_and_every_guardrail_held_is_promoted() -> None:
    assert _computed(_prefers_the_candidate) == ("improved", "separated")
    host, campaign_id = _campaign(_prefers_the_candidate, writer=_leaks(0.0, 0.0), measures=[_LEAKS])
    status = await _pass(host, campaign_id)
    read = _read(host, campaign_id, status)

    assert read.decision == "promote", read.why
    report = WriterPairwiseReport.model_validate(status.result)
    assert read.primary == report.whole.verdict and read.pairwise_job == status.job_id
    assert _outcomes(read, "guardrail") == {_LEAKS.name: "held"}
    assert "regressed" not in _outcomes(read, "contrast").values()
    stored = off_loop(host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
    assert stored is not None and stored.call is None, "the read never writes"


async def test_a_primary_reading_regressed_keeps_the_control_beside_every_gated_reading_improved() -> None:
    assert _computed(_prefers_the_control) == ("regressed", "separated")
    host, campaign_id = _campaign(_prefers_the_control)
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert read.decision == "keep", read.why
    assert set(_outcomes(read, "contrast").values()) == {"improved"}
    assert "on the primary reading." in read.why


async def test_a_gated_reading_regressed_keeps_the_control_beside_an_improved_primary() -> None:
    assert _computed(_prefers_the_candidate)[0] == "improved"
    host, campaign_id = _campaign(_prefers_the_candidate, _TRADE)
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert read.primary is not None and read.primary.outcome == "improved"
    regressed = [verdict for verdict in read.verdicts if verdict.kind == "contrast" and verdict.outcome == "regressed"]
    assert regressed
    assert read.decision == "keep", read.why
    assert all(verdict.heading in read.why for verdict in regressed)
    assert "the primary reading" not in read.why


async def test_a_breached_guardrail_keeps_the_control_beside_an_improved_primary() -> None:
    host, campaign_id = _campaign(_prefers_the_candidate, writer=_leaks(0.0, 1.0), measures=[_LEAKS])
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert read.primary is not None and read.primary.outcome == "improved"
    assert _outcomes(read, "guardrail") == {_LEAKS.name: "breached"}
    assert "regressed" not in _outcomes(read, "contrast").values()
    assert read.decision == "keep", read.why
    assert "(breached)" in read.why


async def test_a_guardrail_not_shown_held_leaves_an_improved_primary_undecided() -> None:
    """Half the private rows leaked is neither within the margin nor shown beyond it on twelve cases."""
    host, campaign_id = _campaign(_prefers_the_candidate, writer=_leaks(0.0, 0.5), measures=[_LEAKS])
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert read.primary is not None and read.primary.outcome == "improved"
    assert _outcomes(read, "guardrail") == {_LEAKS.name: "undecided"}
    assert read.decision == "undecided", read.why
    assert "1 of 1 guardrails are not shown held" in read.why


# --- undecided, named by the primary's reason ---------------------------------------------------------------------


async def test_a_primary_reading_not_separated_is_undecided_however_the_gated_readings_improve() -> None:
    assert _computed(_ties) == ("not_separated", "no_margin")
    host, campaign_id = _campaign(_ties)
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert set(_outcomes(read, "contrast").values()) == {"improved"}
    assert read.primary is not None and (read.primary.outcome, read.primary.reason) == ("not_separated", "no_margin")
    assert read.decision == "undecided", read.why
    assert "does not separate" in read.why


async def test_a_primary_reading_on_too_few_cases_is_undecided_and_named_by_its_reason() -> None:
    """Under the cases a rate needs, the primary is read by its reason, whatever outcome its test came to."""
    outcome, reason = _computed(_tells_on_few)
    assert reason == "too_few_cases"
    host, campaign_id = _campaign(_tells_on_few)
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    assert read.primary is not None
    assert (read.primary.outcome, read.primary.reason) == (outcome, reason)
    assert read.decision == "undecided", read.why
    detail = f"{len(_TOLD)} cases, under the {MIN_CASES} a rate needs"
    assert read.primary.reason_detail == detail
    assert f"too few cases to decide ({detail})" in read.why
    assert "| primary | too few cases |" in read.readout


def test_with_no_pairwise_pass_the_read_never_promotes_and_says_the_primary_is_absent() -> None:
    host, campaign_id = promotion_campaign(BETTER)
    read = writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)
    assert set(_outcomes(read, "contrast").values()) == {"improved"}
    assert read.decision == "undecided" and read.primary is None and read.pairwise_job is None
    assert "no pairwise pass was read" in read.why
    assert "| report_writer.preferred | primary | not read |" in read.readout


# --- the readout ---------------------------------------------------------------------------------------------------


async def test_the_readout_is_one_table_row_per_reading_with_its_role_and_numbers() -> None:
    host, campaign_id = _campaign(_prefers_the_candidate, writer=_leaks(0.0, 0.0), measures=[_LEAKS])
    read = _read(host, campaign_id, await _pass(host, campaign_id))

    header, rule, *rows = read.readout.splitlines()
    assert header == "| Reading | Role | Reads | Difference | Interval | Level |" and rule.startswith("|---|")
    assert len(rows) == 1 + len(read.verdicts)
    assert read.primary is not None and read.primary.interval is not None
    assert rows[0].startswith(f"| {read.primary.heading} | primary | improved | ")
    assert [row.split(" | ")[1] for row in rows[1:]] == [
        {"contrast": "gated", "guardrail": "guardrail", "bar": "reported"}[verdict.kind] for verdict in read.verdicts
    ]
    assert not re.search(r"\w=\S", read.readout) and read.why.count(". ") == 0 and read.why.endswith(".")


# --- what the read refuses -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("named", "other"),
    [("campaign", "campaign_id"), ("candidate", "candidate_key"), ("judge", "judge_model")],
)
async def test_a_pass_naming_another_campaign_candidate_or_judge_than_the_read_is_refused(
    named: str, other: str
) -> None:
    host, campaign_id = _campaign(_prefers_the_candidate)
    status = await _pass(host, campaign_id)
    assert status.result is not None
    moved = status.model_copy(update={"result": {**status.result, other: "another"}})
    with pytest.raises(ValidationFailedError, match=f"names {named} another, where the read's is"):
        _read(host, campaign_id, moved)


async def test_a_job_that_is_no_completed_pass_is_refused() -> None:
    host, campaign_id = _campaign(_prefers_the_candidate)
    status = await _pass(host, campaign_id)
    with pytest.raises(ValidationFailedError, match=r"\(action, failed\) is not a completed writer_pairwise pass"):
        _read(host, campaign_id, status.model_copy(update={"state": "failed", "result": None}))
    with pytest.raises(ValidationFailedError, match="is no writer_pairwise report"):
        _read(host, campaign_id, status.model_copy(update={"result": {"agreement": 0.9}}))


@pytest.mark.parametrize(
    ("campaign", "refusal"),
    [
        pytest.param({"judges": (SCORING_JUDGE, "openai/gpt-mini")}, "scored under 2 judges", id="a moved judge"),
        pytest.param({"case_sets": (HELD_OUT, None)}, "named no case set", id="a run with no case set"),
        pytest.param({"side": "dev"}, "is on the dev side of its split", id="a case off the held-out side"),
    ],
)
async def test_a_campaign_the_read_refuses_is_refused_with_a_pass_in_hand(
    campaign: dict[str, Any], refusal: str
) -> None:
    """A completed pass does not lift a refusal: the campaign's own refusals come first."""
    host, campaign_id = _campaign(_prefers_the_candidate)
    status = await _pass(host, campaign_id)
    refused, refused_id = off_loop(promotion_campaign, BETTER, **campaign)
    with pytest.raises(ValidationFailedError, match=refusal):
        _read(refused, refused_id, status)


def test_a_campaign_run_the_store_cannot_return_names_the_action_that_removes_it() -> None:
    host, campaign_id = promotion_campaign(CONTROL)
    stored = host.storage.load_campaign(campaign_id, TOYHOST_SCOPE)
    assert stored is not None
    (aspect,) = stored.aspects
    host.storage.save_campaign(
        stored.model_copy(update={"aspects": [aspect.model_copy(update={"run_ids": [*aspect.run_ids, "run-gone"]})]})
    )
    with pytest.raises(NotFoundError, match="run-gone") as refused:
        writer_promotion(host, campaign_id, TOYHOST_SCOPE, candidate=CANDIDATE)
    assert "campaign_runs_remove" in refused.value.remedy and "campaign_update" not in refused.value.remedy


# --- the action ----------------------------------------------------------------------------------------------------


async def test_the_action_reads_the_pass_it_is_named_and_promotes_on_both_tools() -> None:
    eval_host, campaign_id = _campaign(_prefers_the_candidate)
    host = _ops(eval_host)
    tool = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]
    started = await tool.call(
        {"action": "writer_pairwise", "campaign_id": campaign_id, "candidate": CANDIDATE}, host=host, caller=CALLER
    )
    assert not started.is_error, started.text
    (job,) = JobsStarted.model_validate(started.structured).jobs
    await await_job(host, JobsStarted.model_validate(started.structured), CALLER.scope_id)

    (read_only,) = eval_catalogue().mount_all(read_only_tools())
    for mounted in (tool, read_only):
        read = await mounted.call(
            {
                "action": "writer_promotion",
                "campaign_id": campaign_id,
                "candidate": CANDIDATE,
                "pairwise_job": job.job_id,
            },
            host=host,
            caller=CALLER,
        )
        assert not read.is_error, read.text
        assert read.structured is not None
        assert read.structured["decision"] == "promote" and read.structured["pairwise_job"] == job.job_id
        assert read.structured["primary"]["name"] == "report_writer.preferred"
