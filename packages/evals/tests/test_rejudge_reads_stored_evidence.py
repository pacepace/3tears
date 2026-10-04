"""A re-judge sends the judge exactly what the first judge read, for every judged kind.

The evidence a judge reads — the subject, the case material and the artifact — is rendered by the
candidate's kind and stored on the cell's :class:`~threetears.evals.contracts.models.EvalTrace` beside
the kind's declaration. A re-judge reads both back rather than re-rendering anything, so:

* a DOCUMENT result is re-scored as the document it was — on its rubric, under the document
  wording — and never as a transcript (the defect this file was written against: the re-judge path
  once built its judge context with no evidence at all, which is a conversation by default);
* every re-asked dim's prompt is byte-identical to the one the first judge was sent;
* a result whose trace stores no evidence is refused before any judge call is paid for;
* the launch attributes a judge to exactly the dims the kind's judge is asked — a document's
  attribution names no conversation axis — and a re-judge refuses a run whose attribution does.

Every cell here runs through the real runner and the real storage, so the evidence a re-judge reads
is the one the runner wrote, not one a fixture placed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.contracts.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.contracts.cassettes import CellCassettes
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    EvalResult,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    eval_trace_doc_id,
    scored_dim_ids,
)
from threetears.evals.contracts.provider import withhold_failure_detail
from threetears.evals.run.judge import CANNOT_TELL, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.lifecycle import rejudge_result
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.runner import RunnerOptions, run_one_result
from threetears.evals.contracts.identity import resolve_variant_identity
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

_SCOPE = "scope-rejudge"
_KIND = "rejudge-probe"
_JUDGE = "judge/scripted"
_DIM = "doc.faithful"

#: Evidence a kind would render, with whitespace that is part of what a judge reads, and a fact
#: the candidate's interlocutors never saw — the case a game master's judge needs.
_EVIDENCE = JudgeEvidence(
    subject="  The game master.\n  Rules as written; no fudged rolls.",
    case_material="GM ONLY: the third flagstone is a pressure-plate trap (DC 15 to spot).",
    artifact="Player (Ivo): I step forward.\nPlayer (Ivo, whispered): I check the floor.\nGM: Roll Perception.",
)


@dataclass
class _ScriptedJudge:
    """A judge client: answers each dim with a score, except the dims told to fail, and records every call.

    Failing is answering unparseable text, which the judge service records as that dim's error —
    the state a re-judge exists to repair.
    """

    failing: set[str] = field(default_factory=set)
    #: The dims answered "can't tell" rather than scored.
    cannot_tell: set[str] = field(default_factory=set)
    #: ``(dim, system, user)`` for every call, in call order.
    calls: list[tuple[str, str, str]] = field(default_factory=list)
    model_name: str = _JUDGE

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        """Record the call; answer a score of 4, or unparseable text for a failing dim.

        Args:
            system: The system prompt, which names the dim's JSON key.
            user: The evidence as placed for the judge.
            response_format: Ignored.

        Returns:
            A completion in the shape the provider port reads.
        """
        match = re.search(r'the single key "(.+?)"', system)
        dim = match.group(1) if match else "?"
        self.calls.append((dim, system, user))
        answer: Any = CANNOT_TELL if dim in self.cannot_tell else 4
        reply = {"reasoning": "the evidence never reaches it", "criteria_scores": {dim: answer}}
        content = "not json" if dim in self.failing else json.dumps(reply)
        return _Completion(content=content)

    async def aclose(self) -> None:
        """Nothing to release."""


@dataclass(frozen=True)
class _Completion:
    content: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = _JUDGE
    served_model: str | None = None


class _JudgedKind:
    """A kind declaring ``judged_artifact`` and rendering :data:`_EVIDENCE` for its one output."""

    def __init__(self, judged_artifact: JudgedArtifact) -> None:
        """Declare what a judge reads of this kind.

        Args:
            judged_artifact: The declaration.
        """
        self.judged_artifact = judged_artifact

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> None:
        """Nothing to build.

        Args:
            subject_snapshot: Ignored.
            variant_config: Ignored.
            world_seed: Ignored.
            span_window: Ignored.
            cassettes: Ignored.
        """

    async def invoke(self, instance: None, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Hand back one output and the evidence a judge reads of it.

        Args:
            instance: Ignored.
            test_case: Ignored.
            sink: Ignored.

        Returns:
            The output and its evidence.
        """
        return CandidateOutput(output=[{"rendered": "elsewhere"}], judge_evidence=_EVIDENCE)


async def _judged_and_stored(
    judged_artifact: JudgedArtifact, judge: _ScriptedJudge, **run_fields: Any
) -> tuple[EvalHost, EvalResult]:
    """Run one judged cell through the real runner, judge it with ``judge``, and store it as a run would.

    Args:
        judged_artifact: The kind's declaration.
        judge: The judge the cell's judge phase calls, and the host's client for a re-judge.
        **run_fields: Fields the run records beyond the defaults below.

    Returns:
        The host holding the stored records, and the stored result.
    """
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge)
    template = EvalTemplate(
        scope_id=_SCOPE,
        name="rejudge",
        intent="run a fair encounter",
        candidate_kind=_KIND,
        rubric=[RubricDim(name=_DIM, description="the GM rules what the dice say", scale="ordinal")],
    )
    case = EvalTestCase(template_id=template.id, scope_id=_SCOPE, variation_params={"party": "two"})
    host.storage.save_template(template)
    host.storage.save_test_case(case)
    # Attributed the way a launch attributes it: through the launch's own judge build, for the kind.
    built = build_judge_service(host, template, _JUDGE, judged_artifact=judged_artifact)
    run_fields = {"effective_judges": built.effective_judges, **run_fields}
    run = make_eval_run(
        scope_id=_SCOPE,
        template_id=template.id,
        candidate_kind=_KIND,
        candidate_model="candidate/m",
        test_case_ids=[case.id],
        status="completed",
        judge_model=_JUDGE,
        judge_request_settings=JUDGE_REQUEST_SETTINGS,
        **run_fields,
    )
    host.storage.save_eval_run(run)
    outcome = await run_one_result(
        host,
        template=template,
        test_case=case,
        subject_id="subject-1",
        model=run.candidate_model,
        k_iteration=1,
        eval_run_id=run.id,
        scope_id=_SCOPE,
        judge_service=JudgeService(
            client_factory=lambda _model, _temperature: judge, failure_describer=withhold_failure_detail
        ),
        judge_model=_JUDGE,
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _JudgedKind(judged_artifact)}),
        variant=resolve_variant_identity(run=run, profile=host.profile),
    )
    host.storage.save_eval_result(outcome.result, outcome.trace)
    return host, outcome.result


def _users_by_dim(calls: list[tuple[str, str, str]]) -> dict[str, set[str]]:
    users: dict[str, set[str]] = {}
    for dim, _system, user in calls:
        users.setdefault(dim, set()).add(user)
    return users


async def test_a_rejudged_document_is_scored_as_the_document_it_was():
    """The document wording, the rubric alone, and the evidence its first judge read.

    Fails without the stored evidence being read back: a re-judge with no evidence has nothing to
    show the judge, and one that guesses a transcript asks the two conversation axes of a document
    and frames its rubric dim as a side of a conversation.
    """
    judge = _ScriptedJudge(failing={_DIM})
    host, result = await _judged_and_stored(JudgedArtifact.DOCUMENT, judge)
    assert result.judge_error is not None and result.rubric_scores == []
    first = list(judge.calls)
    judge.failing.clear()
    judge.calls.clear()

    rejudged = await rejudge_result(host, result.id, _SCOPE)

    assert [score.dim for score in rejudged.rubric_scores] == [_DIM]
    assert rejudged.judge_error is None
    assert rejudged.transcript_score is None and rejudged.outcome_score is None
    assert [dim for dim, _, _ in judge.calls] == [_DIM], "a document is never asked the conversation axes"
    ((_, system, user),) = judge.calls
    assert "You are scoring a candidate's output" in system
    assert "passages of the output and of the case material" in system
    assert "transcript" not in system and "conversation" not in system
    assert user.endswith(f"# Output under review\n{_EVIDENCE.artifact}")
    assert "# Transcript" not in user
    # What the first judge read, byte for byte.
    assert {user} == _users_by_dim(first)[_DIM]


async def test_a_rejudged_conversation_resends_every_failed_axis_exactly_as_first_sent():
    """The transcript a re-judge sends is the kind's stored rendering — the engine renders none."""
    judge = _ScriptedJudge(failing={TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID, _DIM})
    host, result = await _judged_and_stored(JudgedArtifact.TRANSCRIPT, judge)
    first = _users_by_dim(judge.calls)
    judge.failing.clear()
    judge.calls.clear()

    rejudged = await rejudge_result(host, result.id, _SCOPE)

    assert rejudged.judge_error is None
    assert rejudged.transcript_score is not None and rejudged.outcome_score is not None
    assert [dim for dim, _, _ in judge.calls] == [TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID, _DIM]
    again = _users_by_dim(judge.calls)
    for dim in (TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID, _DIM):
        assert again[dim] == first[dim], f"{dim} was asked something other than what its first judge read"
    (transcript_user,) = again[TRANSCRIPT_DIM_ID]
    assert _EVIDENCE.case_material in transcript_user, "the judge-only facts reach the re-judge too"
    assert transcript_user.endswith(f"# Transcript\n{_EVIDENCE.artifact}")


async def test_a_rejudge_asks_only_the_dims_that_errored_and_keeps_a_recorded_cannot_tell():
    """A "can't tell" is the judge's answer, so a re-judge triggered by another dim leaves it standing.

    Re-asking it would pay for a call nothing failed, and storing a fresh sample over it would file
    a second measurement under the first judge's record. The judge below would now SCORE the
    can't-tell dim if asked, so a re-ask is visible as both a call and a changed record.
    """
    judge = _ScriptedJudge(failing={OUTCOME_DIM_ID}, cannot_tell={_DIM})
    host, result = await _judged_and_stored(JudgedArtifact.TRANSCRIPT, judge)
    assert result.judge_error is not None and OUTCOME_DIM_ID in result.judge_error
    assert set(result.judge_cannot_tell) == {_DIM}
    recorded = dict(result.judge_cannot_tell)
    judge.failing.clear()
    judge.cannot_tell.clear()
    judge.calls.clear()

    rejudged = await rejudge_result(host, result.id, _SCOPE)

    assert [dim for dim, _, _ in judge.calls] == [OUTCOME_DIM_ID]
    assert rejudged.outcome_score is not None and rejudged.judge_error is None
    assert rejudged.judge_cannot_tell == recorded
    assert [score.dim for score in rejudged.rubric_scores] == []
    assert rejudged.judge_rescores[-1].dims == [OUTCOME_DIM_ID]


async def test_a_launch_attributes_a_judge_only_to_the_dims_the_kinds_judge_is_asked():
    """A document's attribution names its rubric alone; a conversation's names every axis.

    Built through the launch's own judge build, so this is the record a launched run carries — and
    the record the re-judge below reads its dim set from.
    """
    judge = _ScriptedJudge()
    host, _ = await _judged_and_stored(JudgedArtifact.DOCUMENT, judge)
    template = host.storage.query_templates(_SCOPE)[0]

    document = build_judge_service(host, template, _JUDGE, judged_artifact=JudgedArtifact.DOCUMENT).effective_judges
    conversation = build_judge_service(
        host, template, _JUDGE, judged_artifact=JudgedArtifact.TRANSCRIPT
    ).effective_judges

    assert document == {_DIM: _JUDGE}, "a document is never asked the conversation axes, so none is attributed"
    assert conversation == {TRANSCRIPT_DIM_ID: _JUDGE, OUTCOME_DIM_ID: _JUDGE, _DIM: _JUDGE}
    with pytest.raises(ValueError, match="unjudged kind"):
        build_judge_service(host, template, _JUDGE, judged_artifact=JudgedArtifact.UNJUDGED)


async def test_a_document_rejudge_refuses_a_run_attributing_an_axis_its_cell_was_never_asked():
    """A run recording a judge for a conversation axis on a document cell is refused, naming the axis.

    The launch attributes only the dims a kind's judge is asked, so such a record is not one a
    launch makes; a re-judge reading it would be reproducing an attribution nobody's judge acted on.
    """
    judge = _ScriptedJudge(failing={_DIM})
    host, result = await _judged_and_stored(
        JudgedArtifact.DOCUMENT, judge, effective_judges={TRANSCRIPT_DIM_ID: _JUDGE, _DIM: _JUDGE}
    )
    judge.calls.clear()

    with pytest.raises(ValidationFailedError, match=rf"judged {TRANSCRIPT_DIM_ID}, which a document cell"):
        await rejudge_result(host, result.id, _SCOPE)

    assert judge.calls == []


def test_a_document_is_judged_on_its_rubric_and_a_conversation_on_every_axis():
    """Both directions of the one rule, on one rubric."""
    assert scored_dim_ids([_DIM], JudgedArtifact.DOCUMENT) == [_DIM]
    assert scored_dim_ids([_DIM], JudgedArtifact.TRANSCRIPT) == [TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID, _DIM]
    assert scored_dim_ids([_DIM], JudgedArtifact.UNJUDGED) == []


async def test_the_runner_stores_the_evidence_with_the_declaration_that_picked_its_axes():
    judge = _ScriptedJudge()
    host, result = await _judged_and_stored(JudgedArtifact.DOCUMENT, judge)

    trace = host.storage.load_eval_trace(result.id, _SCOPE)

    assert trace is not None
    assert trace.judge_evidence == _EVIDENCE, "the evidence round-trips through storage, whitespace intact"
    assert trace.judged_artifact is JudgedArtifact.DOCUMENT


async def test_a_result_whose_trace_stores_no_evidence_is_refused_before_any_judge_call():
    """Nothing to resend is a refusal, never a re-render and never a guess at a transcript."""
    judge = _ScriptedJudge(failing={_DIM})
    host, result = await _judged_and_stored(JudgedArtifact.DOCUMENT, judge)
    # The trace as a cell whose payload write lost its evidence would leave it.
    host.storage.save_eval_result(
        result,
        EvalTrace(
            id=eval_trace_doc_id(result.id),
            scope_id=_SCOPE,
            result_id=result.id,
            eval_run_id=result.eval_run_id,
            trace=[{"rendered": "elsewhere"}],
        ),
    )
    judge.calls.clear()

    with pytest.raises(ValidationFailedError, match="stores no judge evidence"):
        await rejudge_result(host, result.id, _SCOPE)

    assert judge.calls == []


def _trace(**fields: Any) -> EvalTrace:
    return EvalTrace(id="r:trace", scope_id=_SCOPE, result_id="r", eval_run_id="run", **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"judge_evidence": _EVIDENCE},
        {"judged_artifact": JudgedArtifact.DOCUMENT},
        {"judge_evidence": _EVIDENCE, "judged_artifact": JudgedArtifact.UNJUDGED},
    ],
    ids=["evidence-without-its-declaration", "a-declaration-without-evidence", "evidence-for-an-unjudged-kind"],
)
def test_a_trace_refuses_evidence_a_rejudge_could_not_resend(fields):
    with pytest.raises(ValueError, match="judge_evidence and judged_artifact|unjudged kind"):
        _trace(**fields)


def test_a_trace_carries_evidence_with_its_declaration_or_neither():
    """The accepting side of the refusals above, on the same fixture."""
    assert _trace().judge_evidence is None
    assert _trace(judge_evidence=_EVIDENCE, judged_artifact=JudgedArtifact.TRANSCRIPT).judge_evidence == _EVIDENCE
