"""The candidate-kind seam: one dispatch, and nothing kind-shaped below it.

What lives here is the seam's own evidence, and the case that matters most is the one
no other file can state: **a cell that runs with no conversational subject, no simulator,
no world and no judge model and still produces an ``EvalResult``.** Every non-conversational
kind builds on exactly that shape, so it is the test that says the seam is real rather than
decorative.

The fake kind below is deliberately minimal — a dict for an instance and a canned
output — because anything richer would start re-testing the conversational turn
through a second implementation.
"""

from __future__ import annotations

import ast
import logging
import sys
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.candidate_kind import (
    CandidateKindDefect,
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    JudgedArtifact,
    UnknownCandidateKind,
    VariantConfig,
)
from threetears.evals.contracts.cassettes import CellCassettes
from threetears.evals.contracts.host.apparatus import ApparatusError
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.traces import CellTrace, TraceSink
from threetears.evals.contracts.models import (
    AsyncDelivery,
    ConversationStopCause,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    GoalStateOutcome,
    PreconditionOutcome,
    RoleUsage,
)
from threetears.evals.contracts.provider import withhold_failure_detail
from threetears.evals.run.metering import MeteredCallLedger
from threetears.evals.run.runner import EveryCellApparatusFailedError, RunnerOptions, execute_run, run_one_result
from threetears.evals.contracts.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.contracts.host.values import SweepableValue
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.memory_store import memory_storage

#: The level map a cell stamped without a run in hand is keyed by: its model, and nothing else.
_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}

#: The package's source root; the import/dispatch-site scans below walk the tree under it.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
_EVAL_ROOT = _SRC_ROOT / "threetears" / "evals"

#: The kind name the fakes below register under. Not a real kind and never will be — it
#: exists so the dispatch has something to resolve that the runner cannot have built.
_FAKE_KIND = "fake-single-shot"


# parity-with: threetears.evals.contracts.candidate_kind.CandidateKind
class _FakeSingleShotKind:
    """A kind with no subject to mint, no turns to drive and nothing to tear down.

    The shape a classifier or an artifact generator has: ``prepare`` unpacks a
    payload, ``invoke`` returns one document. It records what it was handed so the seam's
    own arguments can be asserted rather than assumed.
    """

    def __init__(self, output: CandidateOutput, *, judged_artifact: JudgedArtifact = JudgedArtifact.UNJUDGED) -> None:
        """Bind the canned output this kind reports.

        Args:
            output: What :meth:`invoke` hands back.
            judged_artifact: What this kind declares a judge reads. Unjudged by default, which is
                what a single-shot kind graded by code is.
        """
        self.output = output
        self.judged_artifact = judged_artifact
        self.prepared: dict[str, Any] | None = None
        self.invoked_case: EvalTestCase | None = None

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> dict[str, Any]:
        """Record ``prepare``'s arguments and hand back an opaque instance.

        Args:
            subject_snapshot: The engine's half of the subject pair.
            variant_config: This cell's contestant stack.
            world_seed: The world the scenario presumes.
            span_window: This cell's tracing windows. Recorded and deliberately never
                opened — this kind is the population the harvest's disclosure is about.
            cassettes: This cell's cassettes; ``None`` for every run here, which has them off.

        Returns:
            The instance, which is just what was handed over.
        """
        self.prepared = {
            "subject_snapshot": subject_snapshot,
            "variant_config": variant_config,
            "world_seed": world_seed,
            "span_window": span_window,
        }
        return self.prepared

    async def invoke(self, instance: dict[str, Any], test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Report the canned output.

        Args:
            instance: What :meth:`prepare` returned.
            test_case: The cell's case.
            sink: The cell's sink; a single shot that awaits nothing has no progress to report.

        Returns:
            The canned output.

        Raises:
            AssertionError: The instance handed back is not the one prepared.
        """
        assert instance is self.prepared, "the engine must hand back the instance the kind prepared, untouched"
        self.invoked_case = test_case
        return self.output


class _FailingPrepareKind:
    """A kind whose preparation refuses, naming its own termination arm."""

    def __init__(self, failure: CandidatePreparationFailed) -> None:
        """Bind the refusal this kind raises.

        Args:
            failure: What :meth:`prepare` raises.
        """
        self._failure = failure
        self.judged_artifact = JudgedArtifact.UNJUDGED

    async def prepare(self, **_kwargs: Any) -> Any:
        """Refuse to build a candidate.

        Args:
            **_kwargs: ``prepare``'s arguments, ignored.

        Raises:
            CandidatePreparationFailed: Always — that is the case under test.
        """
        raise self._failure

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Never reached.

        Args:
            instance: Ignored.
            test_case: Ignored.
            sink: Ignored.

        Raises:
            AssertionError: Always — a refused preparation must not be invoked.
        """
        raise AssertionError("invoke must not run after prepare refused")


def _template(**overrides: Any) -> EvalTemplate:
    """A template naming the fake kind, with nothing a conversational path would need.

    Args:
        **overrides: Field overrides.

    Returns:
        The template.
    """
    fields: dict[str, Any] = {
        "scope_id": "uni-1",
        "name": "seam",
        "intent": "prove the seam",
        "candidate_kind": _FAKE_KIND,
    }
    fields.update(overrides)
    return EvalTemplate(**fields)


def _case(template_id: str) -> EvalTestCase:
    """One case for the template.

    Args:
        template_id: The template this case belongs to.

    Returns:
        The case.
    """
    return EvalTestCase(template_id=template_id, scope_id="u", variation_params={"tone": "warm"})


async def _run(kind: Any, template: EvalTemplate, *, sink: TraceSink | None = None, **overrides: Any) -> Any:
    """Drive one cell through ``run_one_result`` with the given kind wired, for the toy host.

    The runner takes no subject factory and no simulator of its own, so a cell that completes
    with only this kind wired is direct evidence the seam needs neither.

    Args:
        kind: The kind to register under :data:`_FAKE_KIND`.
        template: The template whose ``candidate_kind`` the dispatch reads.
        sink: The host's trace sink, or ``None`` for a host that wires none.
        **overrides: Extra ``run_one_result`` arguments.

    Returns:
        The cell outcome.
    """
    host = toyhost_host(trace_sink=sink)
    call: dict[str, Any] = {
        "template": template,
        "test_case": _case(template.id),
        "subject_id": "subj-1",
        "model": "test/model",
        "k_iteration": 1,
        "eval_run_id": "run-1",
        "scope_id": "u",
        "judge_service": None,
        "options": RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
        # No run is in hand here, so the cell is stamped with the key of its model alone. A test about
        # the key passes its own.
        "variant": DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    }
    call.update(overrides)
    return await run_one_result(host, **call)


# =============================================================================
# The seam is real: a cell with no subject, no simulator and no judge model
# =============================================================================


async def test_a_single_shot_kind_produces_a_result_with_no_subject_and_no_judge():
    """The shape every non-conversational kind builds on, end to end through the real runner."""
    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"turn": 0, "role": "candidate", "content": "label=on_topic"}],
            mechanical_facts=[GoalStateOutcome(expression="label == expected", passed=True, detail="on_topic")],
            telemetry=CandidateTelemetry(
                usage=[
                    RoleUsage(role="candidate", model="test/model", call_count=1, cost_usd=0.25, price_source="script")
                ]
            ),
            candidate_instance_id="sandbox-42",
        )
    )

    result, trace = await _run(kind, _template())

    assert result.termination == "completed"
    assert result.runner_error is None
    # The mechanical tier reached the record without a judge being wired at all.
    assert [o.expression for o in result.goal_state_outcomes] == ["label == expected"]
    assert result.goal_state_outcomes[0].passed is True
    assert result.rubric_scores == []
    assert result.transcript_score is None and result.outcome_score is None
    assert result.judge_model is None
    # The candidate's own output is what the trace document stores.
    assert trace.trace == [{"turn": 0, "role": "candidate", "content": "label=on_topic"}]
    assert result.cost_usd == 0.25
    assert [row.role for row in result.usage] == ["candidate"]
    assert result.candidate_instance_id == "sandbox-42"


async def test_the_kind_receives_the_engines_half_of_the_subject_and_nothing_of_the_hosts():
    """The subject split, asserted at the call: ``prepare`` is handed the engine's snapshot alone.

    The host's rich half never passes through the engine — a kind that needs it has it bound into
    its factory by its host — so the call carries the engine's key/label/hashes view, the variant
    and the seed, and no argument a host's object could ride in on.
    """
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))
    template = _template()
    snapshot = SubjectSnapshot(subject_id="subj-1", subject_label="subject one", state=None)

    await _run(kind, template, subject_snapshot=snapshot)

    assert kind.prepared is not None
    assert set(kind.prepared) == {"subject_snapshot", "variant_config", "world_seed", "span_window"}
    assert kind.prepared["subject_snapshot"] is snapshot, "the engine's half reaches the kind unchanged"
    assert kind.prepared["variant_config"].candidate_model == "test/model"
    assert kind.prepared["world_seed"] is template.world_seed
    assert kind.invoked_case is not None and kind.invoked_case.template_id == template.id


async def test_every_result_of_a_run_names_the_model_the_run_was_launched_for():
    """``EvalResult.model`` is the run's ``candidate_model``, as the runner writes it.

    Driven through :func:`execute_run`, the entrypoint that reads the model off the run —
    ``run_one_result`` is handed a model directly, so a test of it cannot see that read go wrong.
    And through a kind that never reads the model it is given, so a runner stamping the wrong name
    leaves the kind nothing to fail on: only this assertion can see it.
    """
    host = toyhost_host()
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))
    template = _template()
    cases = [_case(template.id), _case(template.id)]
    run = make_eval_run(
        scope_id="u",
        template_id=template.id,
        candidate_kind=_FAKE_KIND,
        candidate_model="vendor/model-under-test",
        k_runs=2,
        test_case_ids=[case.id for case in cases],
    )

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
    )

    results = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert len(results) == len(cases) * run.k_runs
    assert [result.model for result in results] == [run.candidate_model] * len(results)
    assert kind.prepared is not None
    assert kind.prepared["variant_config"].candidate_model == run.candidate_model


class _ApparatusFaultOnInvokeKind(_FakeSingleShotKind):
    """A kind whose candidate is built and whose rig then fails — the cut-short, excluded arm."""

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Fail as the apparatus, after preparation succeeded.

        Args:
            instance: Ignored.
            test_case: Ignored.
            sink: Ignored.

        Raises:
            ApparatusError: Always — that is the case under test.
        """
        raise ApparatusError("the delivery corpus for this case does not rebuild")


@pytest.mark.parametrize(
    ("make_kind", "raised"),
    [
        pytest.param(
            lambda: _FailingPrepareKind(
                CandidatePreparationFailed("subject_factory: no", termination="factory_failed")
            ),
            nullcontext(),
            id="refused-preparation",
        ),
        # The run's only cell excluded by its rig: the loop saves it and then refuses the run.
        pytest.param(
            lambda: _ApparatusFaultOnInvokeKind(CandidateOutput(output=[])),
            pytest.raises(EveryCellApparatusFailedError),
            id="apparatus-fault-on-invoke",
        ),
    ],
)
async def test_a_cell_that_did_not_complete_still_names_the_model_the_run_was_launched_for(
    make_kind: Any, raised: AbstractContextManager[Any]
):
    """Every arm that writes a result writes the run's ``candidate_model`` — not only the one that completes.

    The runner builds an :class:`EvalResult` on each exit separately, so the completed arm above says
    nothing about these two; a cell excluded for its apparatus pools with its run's model or nowhere.
    """
    host = toyhost_host()
    kind = make_kind()
    template = _template()
    case = _case(template.id)
    run = make_eval_run(
        scope_id="u",
        template_id=template.id,
        candidate_kind=_FAKE_KIND,
        candidate_model="vendor/model-under-test",
        k_runs=1,
        test_case_ids=[case.id],
    )

    with raised:
        await execute_run(
            host,
            run=run,
            template=template,
            test_cases=[case],
            judge_service=None,
            options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
        )

    (result,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert result.infra_error is not None, "the control: this cell took an excluded arm, not the completed one"
    assert result.model == run.candidate_model


async def test_the_candidates_errors_reach_the_result_split_the_way_scoring_reads_them():
    """A candidate error FAILS a cell and an infra error EXCLUDES it, so they stay apart."""
    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"content": "x"}],
            candidate_errors=["candidate turn LLM error: boom"],
            infra_errors=["harness: nope"],
        )
    )

    result, _ = await _run(kind, _template())

    assert result.candidate_error == "candidate turn LLM error: boom"
    assert result.infra_error == "harness: nope"
    assert result.runner_error == "candidate turn LLM error: boom; harness: nope"


async def test_telemetry_the_kind_reports_lands_on_the_result():
    """Everything below the dispatch reads telemetry, and never the kind that made it."""
    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"content": "x"}],
            telemetry=CandidateTelemetry(
                usage=[
                    RoleUsage(role="candidate", model="test/model", call_count=2, cost_usd=1.5, price_source="script")
                ],
                metered_calls_refused=3,
                phase_timings={"search": 12.5},
                async_wait_ms=44.0,
            ),
        )
    )

    result, _ = await _run(kind, _template())

    assert result.cost_usd == 1.5
    assert result.metered_calls_refused == 3
    assert result.phase_timings == {"search": 12.5}
    assert result.latency is not None and result.latency.async_wait_ms == 44.0
    # No sink was wired, so the span-derived buckets stay unmeasured rather than zeroed.
    assert result.latency.total_ms is None


async def test_a_kinds_own_refusals_are_added_to_the_runs_and_neither_is_dropped():
    """The case ``CandidateTelemetry.metered_calls_refused`` now exists for, driven end to end.

    The count on the result is the RUNNER's, taken against the run's ledger around the whole
    dispatch, because a kind that never thought to count is the default case and copying a
    kind's report is how a classifier cell came to store ``None`` under a ledger that was in
    force. What the field on the telemetry still carries is the other thing: refusals a kind
    made against a ledger of its OWN, which the run's cannot see.

    The two are ADDED. Either one winning would publish a count lower than what happened —
    the run's ledger refused nothing here and the kind refused two, so a result reading 0 or
    reading nothing would both be false. This is the only one of the four input combinations
    no other test reaches, and it is the one the whole narrowing turns on.
    """
    # A ceiling of zero, so the one call this cell makes against the RUN's ledger is refused
    # and the runner's own slice reads 1. Both sides must be non-zero and DIFFERENT, or the
    # assertion below cannot tell a sum from either side winning outright.
    ledger = MeteredCallLedger("run-1", ceiling=0)

    class _MetersItsOwnProviderToo(_FakeSingleShotKind):
        """A kind that reaches the run's ledger AND keeps a second one the run cannot see."""

        async def invoke(self, instance: dict[str, Any], test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
            """Spend against the run's ledger, then report this kind's own refusals.

            Args:
                instance: What ``prepare`` returned.
                test_case: The cell's case.
                sink: The cell's sink.

            Returns:
                The canned output.
            """
            assert not ledger.admit(tool="web_search", action="search", spend=None), "the ceiling was not in force"
            return await super().invoke(instance, test_case, sink)

    kind = _MetersItsOwnProviderToo(
        CandidateOutput(
            output=[{"content": "x"}],
            telemetry=CandidateTelemetry(metered_calls_refused=2),
        )
    )

    result, _ = await _run(
        kind,
        _template(),
        options=RunnerOptions(
            candidate_kinds={_FAKE_KIND: lambda _cell: kind},
            metered_calls=ledger,
        ),
    )

    assert result.metered_calls_refused == 3, (
        "1 refused by the run's ledger plus 2 the kind metered itself — a 1 or a 2 here is one side winning"
    )


async def test_a_kind_that_reports_no_async_deliveries_observed_nothing():
    """The record's absence is unobserved — not a crash, and not an empty that claims a watch.

    ``[]`` would say the kind watched for background work and none was started; a kind that
    reports ``None`` never watched. The positive cases follow: a kind's entries land verbatim.
    """
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))

    result, _ = await _run(kind, _template())

    assert result.async_deliveries is None
    assert result.kind_payload is None
    # Nor does a kind that does not converse have a conversation to have stopped.
    assert result.stop_cause is None


async def test_a_kinds_async_deliveries_land_on_the_result_verbatim():
    """A scout acknowledged on one turn and delivered two turns later is stored as the kind reported it."""
    scouted = AsyncDelivery(
        tool="scout_ahead",
        requested_by="gm",
        status="delivered",
        model="scout/model",
        acknowledged_turn=1,
        delivered_turn=3,
        elapsed_ms=4200.0,
        delivered_items=2,
        summary="two goblins behind the ridge",
        substituted=False,
    )
    seeded = AsyncDelivery(
        tool="scout_ahead", status="delivered", acknowledged_turn=4, delivered_turn=4, substituted=True
    )
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], async_deliveries=[scouted, seeded]))

    result, _ = await _run(kind, _template())

    assert result.async_deliveries == [scouted, seeded]


def _live_scout(**overrides: Any) -> AsyncDelivery:
    """A scout delivered live on its own model — work that necessarily spent tokens.

    Args:
        **overrides: Field overrides.

    Returns:
        The entry.
    """
    fields: dict[str, Any] = {
        "tool": "scout_ahead",
        "status": "delivered",
        "model": "scout/model",
        "acknowledged_turn": 0,
        "delivered_turn": 2,
        "substituted": False,
    }
    fields.update(overrides)
    return AsyncDelivery(**fields)


_SILENT_CARRIER = "report no spend at all"


async def test_live_background_work_with_no_spend_row_is_logged_as_a_broken_carrier(caplog):
    """A delivery that ran on a model spent tokens there; one reporting none lost its spend on the way."""
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], async_deliveries=[_live_scout()]))

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.runner"):
        await _run(kind, _template())

    assert any(_SILENT_CARRIER in r.getMessage() for r in caplog.records), caplog.text


@pytest.mark.parametrize(
    "delivery",
    [
        pytest.param(_live_scout(cost_usd=0.01), id="it-reported-its-spend"),
        pytest.param(_live_scout(input_tokens=12), id="it-reported-tokens-and-no-dollars"),
        pytest.param(_live_scout(model=None, substituted=True), id="a-seed-spent-nothing"),
        pytest.param(_live_scout(model=None), id="no-model-did-the-work"),
        pytest.param(
            _live_scout(status="failed", error="timed out", delivered_turn=None), id="it-failed-before-delivering"
        ),
        pytest.param(_live_scout(status="undelivered", delivered_turn=None), id="it-was-still-in-flight"),
    ],
)
async def test_background_work_that_honestly_spent_nothing_measurable_is_not_logged(caplog, delivery):
    """The detector's condition is narrow on purpose: one that cries wolf trains its reader to ignore it."""
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], async_deliveries=[delivery]))

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.runner"):
        await _run(kind, _template())

    assert not any(_SILENT_CARRIER in r.getMessage() for r in caplog.records), caplog.text


async def test_a_conversing_kind_reports_its_stop_cause_onto_the_result():
    """The stop cause a kind reports is the one the stored result carries."""
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], stop_cause=ConversationStopCause.USER_DONE))

    result, _ = await _run(kind, _template())

    assert result.stop_cause is ConversationStopCause.USER_DONE


class _WatchingButUnreachedSink:
    """A ``TraceSink`` that is wired and never asked for a window.

    Not a stub for convenience — it IS the situation under test. The kind below is handed
    this cell's windows like every other kind and opens neither, so a perfectly good sink
    sits there collecting nothing and the cell reports no latency while looking exactly like
    a cell that emitted no spans. What makes the omission visible is the warning, not the
    absence of a reach: every kind alike receives ``prepare``'s ``span_window``.
    """

    def __init__(self) -> None:
        self.windows_opened = 0

    @contextmanager
    def cell_identity(self, cell: Any) -> Iterator[None]:
        """Yield the identity scope, recording nothing.

        Unopened on this path, and that is the point rather than an omission: the kind under
        test opens neither of the two windows it was handed. Only the collection one is
        counted below, because only its absence costs a measurement.
        """
        yield

    @contextmanager
    def cell_spans(self, cell: Any) -> Iterator[Any]:
        """Yield a collection window, and count it, so the test can assert none was asked for."""
        self.windows_opened += 1
        yield CellTrace()


async def test_a_kind_that_opens_no_span_window_says_so_while_a_sink_is_watching(caplog):
    """The disclosure exercised, so the warning is held by a test rather than by prose.

    The harvest warns rather than letting a cell leave the cost-vs-latency comparison silently.
    Delete the branch and that promise becomes false with nothing going red, which is the same shape of defect the warning exists to
    disclose. The assertion is on the message naming the KIND, because that is what tells an
    operator which population lost its latency.

    It is a kind's own omission now rather than a gap in the seam: the windows arrive as an
    argument, and the test below drives the same fake kind through one that opens them.
    """
    sink = _WatchingButUnreachedSink()
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.runner"):
        result, _ = await _run(
            kind,
            _template(),
            sink=sink,
            options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
        )

    assert sink.windows_opened == 0, "the fake kind opens neither window it was handed — that is the case under test"
    assert result.latency is None or result.latency.total_ms is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(_FAKE_KIND in message and "no span collection window" in message for message in warnings), warnings


async def test_a_kind_that_says_it_did_no_timed_work_is_not_blamed_for_the_missing_window(caplog):
    """A replay has no latency; the cell leaves the axis, and the kind is not accused of an omission.

    Same fake kind, same watching sink as the test above — the only difference is the kind's own
    ``untimed_reason``, so the warning's absence here can only be that field's doing.
    """
    sink = _WatchingButUnreachedSink()
    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"content": "x"}], telemetry=CandidateTelemetry(untimed_reason="a replay of a stored artifact")
        )
    )

    with caplog.at_level(logging.INFO, logger="threetears.evals.run.runner"):
        result, _ = await _run(
            kind,
            _template(),
            sink=sink,
            options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
        )

    assert sink.windows_opened == 0
    assert result.latency is None or result.latency.total_ms is None, "an untimed cell still has no latency"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("no span collection window" in message for message in warnings), warnings
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("did no timed work" in m and "a replay of a stored artifact" in m for m in infos), infos


async def test_a_cell_with_no_sink_wired_stays_quiet_about_its_missing_window(caplog):
    """Nobody was watching, so there is nothing to disclose — the third state, kept distinct.

    `trace_sink=None` is a decision at the call site, not a mechanism that stopped working.
    Warning here would train an operator to ignore the line that matters.
    """
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.runner"):
        await _run(kind, _template())

    assert not [r for r in caplog.records if "no span collection window" in r.getMessage()]


class _CollectingSink:
    """A ``TraceSink`` whose collection window hands back the record it was built with.

    What fills a record's three buckets is the host's own span summing
    (``summarize_span_durations_ms``), which is tested where it lives. Handing back a
    prepared record keeps the subject here the PORT: a kind built outside this module opens
    the window it was handed, and what the sink recorded reaches the result's latency.
    """

    def __init__(self, record: CellTrace) -> None:
        """Bind the record this sink's collection window yields.

        Args:
            record: What a closing window leaves behind.
        """
        self._record = record
        self.windows_opened = 0
        self.identities_opened = 0

    @contextmanager
    def cell_identity(self, cell: Any) -> Iterator[None]:
        """Yield the identity scope, counting that it was asked for."""
        self.identities_opened += 1
        yield

    @contextmanager
    def cell_spans(self, cell: Any) -> Iterator[Any]:
        """Yield the prepared record as this cell's collection window."""
        self.windows_opened += 1
        yield self._record


class _WindowOpeningKind(_FakeSingleShotKind):
    """The same single-shot kind, opening the windows it was handed.

    The shape a kind built outside this module has after the port: it carries ``prepare``'s
    ``span_window`` out on its instance and opens it around its own work in ``invoke``,
    which is where the candidate's execution actually is.
    """

    async def invoke(self, instance: dict[str, Any], test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Report the canned output from inside both of this cell's windows.

        Args:
            instance: What :meth:`prepare` returned, carrying the window.
            test_case: The cell's case.
            sink: The cell's sink.

        Returns:
            The canned output.
        """
        window = instance["span_window"]
        with window.identity(), window.collecting():
            return await super().invoke(instance, test_case, sink)


async def test_a_kind_built_outside_this_module_times_its_own_cell_through_the_window():
    """The port: a kind the runner could not have built opens the cell's window and is timed.

    This is the case that could not be written before the window was an argument, and it is
    the whole reason it is one — a classifier or a toy extractor is constructed by its host
    and wired onto ``RunnerOptions.candidate_kinds``, so every cell it drives used to leave
    the cost-vs-latency comparison. Asserted through to ``result.latency`` rather than at the
    sink, because the sink being asked for a window proves only half of it: the harvest reads
    the record by three named fields, and a bucket it mis-keys is silently dropped.
    """
    sink = _CollectingSink(CellTrace(spans=[{"name": "candidate.work"}], total_ms=42.0, llm_ms=30.0, tool_ms=5.0))
    kind = _WindowOpeningKind(CandidateOutput(output=[{"content": "x"}]))

    result, trace = await _run(
        kind,
        _template(),
        sink=sink,
        options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
    )

    assert (sink.windows_opened, sink.identities_opened) == (1, 1)
    assert result.latency is not None, "a kind that opened the window must not report a cell that timed nothing"
    assert (result.latency.total_ms, result.latency.llm_ms, result.latency.tool_ms) == (42.0, 30.0, 5.0)
    assert trace.otel_trace == [{"name": "candidate.work"}], "the harvested spans ride out on the trace document"


async def test_a_window_that_closed_on_no_turn_root_span_is_disclosed_as_that(caplog):
    """The second disclosure, kept apart from the first because the repair is the opposite.

    A kind can open the window and still be untimed — nothing it called emitted the
    ``agent.invoke`` span the total is summed across, which is exactly what a scripted client
    does. Reporting that as "opened no window" would send an operator to add a window that is
    already there.
    """
    sink = _CollectingSink(CellTrace())
    kind = _WindowOpeningKind(CandidateOutput(output=[{"content": "x"}]))

    with caplog.at_level(logging.WARNING, logger="threetears.evals.run.runner"):
        result, _ = await _run(
            kind,
            _template(),
            sink=sink,
            options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
        )

    assert sink.windows_opened == 1
    assert result.latency is None or result.latency.total_ms is None
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(_FAKE_KIND in message and "no turn-root span landed in it" in message for message in warnings), warnings
    assert not any("opened no span collection window" in message for message in warnings), warnings


async def test_a_kind_payload_is_stored_verbatim_and_never_unpacked_by_the_runner():
    """``kind_payload`` is opaque: the engine stores it as the kind wrote it and reads nothing off it.

    The record the runner does read arrives on its own declared field, ``async_deliveries``. A
    dispatch site that also looked inside the payload would be reading a kind's object it is not
    entitled to read, and nothing static would catch it — so the payload here even holds an
    ``async_deliveries`` key of its own, and it still does not land on the result's field.
    """
    payload = {
        "labels_emitted": 7,
        "async_deliveries": [{"tool": "not the runner's to read"}],
        "nested": {"stop_reason": "budget", "phases": [1.5, None, True]},
    }
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], kind_payload=payload))

    result, _ = await _run(kind, _template())

    assert result.kind_payload == payload
    assert result.async_deliveries is None


async def test_a_kind_payload_round_trips_through_storage():
    """What a kind reported is what a later reader gets back from the store, field for field."""
    payload = {"router": {"label": "attack", "confidence": 0.82, "runner_up": None}, "turns": [1, 2]}
    delivery = AsyncDelivery(
        tool="scout_ahead", status="failed", acknowledged_turn=0, error="timed out", substituted=False
    )
    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"content": "x"}],
            kind_payload=payload,
            async_deliveries=[delivery],
            candidate_instance_id="gm-7",
        )
    )
    result, trace = await _run(kind, _template())
    storage, _ = memory_storage()

    storage.save_eval_result(result, trace)
    loaded = storage.load_eval_result(result.id, result.scope_id)

    assert loaded is not None
    assert loaded.kind_payload == payload
    assert loaded.async_deliveries == [delivery]
    assert loaded.candidate_instance_id == "gm-7"


#: A payload whose whitespace is the content: keys that differ only by surrounding space, and
#: values whose leading and trailing space a kind meant. Every model these pass through strips
#: ``str`` fields, so a payload that never carried any whitespace could not tell verbatim storage
#: from a trimmed copy.
_WHITESPACE_PAYLOAD = {
    "note": "plain",
    "  note ": "  indented narration\n",
    "nested": {" key ": [" spaced ", "\ttabbed"]},
}


async def test_a_kind_payload_keeps_its_whitespace_from_the_kind_to_the_store():
    """Verbatim means the kind's own bytes: no key trimmed, no two keys collapsed, no value trimmed."""
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}], kind_payload=_WHITESPACE_PAYLOAD))
    result, trace = await _run(kind, _template())
    storage, _ = memory_storage()

    storage.save_eval_result(result, trace)
    loaded = storage.load_eval_result(result.id, result.scope_id)

    assert result.kind_payload == _WHITESPACE_PAYLOAD
    assert loaded is not None and loaded.kind_payload == _WHITESPACE_PAYLOAD


def test_a_host_payload_keeps_its_keys_verbatim_on_a_run_and_a_case():
    """The host's own opaque payloads are stored as the host wrote them, top-level keys included."""
    payload = {" ns ": {" key ": " value "}, "ns": {"key": "value"}}
    run = make_eval_run(host_payload=payload)
    case = EvalTestCase(scope_id=run.scope_id, template_id="t-1", host_payload=payload)

    assert EvalRun.from_dict(run.to_dict()).host_payload == payload
    assert EvalTestCase.from_dict(case.to_dict()).host_payload == payload


@pytest.mark.parametrize(
    "carrier",
    [object(), "a lone \ud800 surrogate"],
    ids=["not-a-json-value", "a-lone-surrogate"],
)
def test_a_kind_payload_that_storage_cannot_encode_is_refused_at_construction(carrier):
    """Refused where the kind builds its output, not at the write after the judging was paid for.

    A lone surrogate is refused for the same reason: no UTF-8 store can hold it as written, and the
    payload is stored verbatim or not at all.
    """
    with pytest.raises(ValidationError, match="kind_payload"):
        CandidateOutput(output=[{"content": "x"}], kind_payload={"carrier": carrier})


# =============================================================================
# Preparation refusals, and a kind this host does not have
# =============================================================================


@pytest.mark.parametrize(
    ("termination", "error"),
    [
        ("factory_failed", "subject_factory: no such model"),
        ("seed_failed", "apparatus: malformed seed"),
    ],
)
async def test_a_refused_preparation_records_one_cleanly_excluded_cell(termination: str, error: str):
    """The termination the kind names is the arm the record carries — it is how an operator navigates."""
    kind = _FailingPrepareKind(CandidatePreparationFailed(error, termination=termination))

    result, trace = await _run(kind, _template())

    assert result.termination == termination
    assert result.runner_error == error
    assert result.infra_error == error
    assert result.candidate_error is None
    assert result.cost_usd == 0.0
    assert trace.trace == [], "a cell that never ran a turn recorded no transcript"
    # No turn ran, so no conversation stopped — and the capture that did run saw no delivery.
    assert result.stop_cause is None
    assert result.async_deliveries == []
    assert result.kind_payload is None


async def test_a_refused_precondition_carries_its_outcomes_onto_the_record():
    """The arm that HAS outcomes keeps them: an exclusion is read by asking what was presumed."""
    outcome = PreconditionOutcome(expression="call_count('x') == 0", presumes="a clean world", held=False, detail="1")
    kind = _FailingPrepareKind(
        CandidatePreparationFailed("precondition: nope", termination="precondition_failed", preconditions=[outcome])
    )

    result, _ = await _run(kind, _template())

    assert result.termination == "precondition_failed"
    assert [o.expression for o in result.precondition_outcomes] == ["call_count('x') == 0"]


async def test_an_unwired_kind_fails_the_run_rather_than_the_cell():
    """Every cell would fail identically, so this is a misconfigured run, not a partial one."""
    with pytest.raises(UnknownCandidateKind) as caught:
        await _run(_FakeSingleShotKind(CandidateOutput()), _template(candidate_kind="artifact-generator"))

    assert "artifact-generator" in str(caught.value)
    # The message names what this host DOES have, so an operator can see the typo — and only that:
    # no kind is built in, so nothing but the wired kind is named.
    assert _FAKE_KIND in str(caught.value)


# =============================================================================
# No default kind
# =============================================================================


def test_a_template_that_names_no_kind_is_refused():
    """A template states its kind: no host's kind name is assumed for one that does not.

    The default was one host's word, which every other host would have inherited without choosing
    it — the reason ``EvalAnalysis.subject_kind`` is un-defaulted too.
    """
    with pytest.raises(ValidationError, match="candidate_kind"):
        EvalTemplate(scope_id="uni-1", name="t", intent="i")
    assert EvalTemplate(scope_id="uni-1", name="t", intent="i", candidate_kind="any-kind").candidate_kind == "any-kind"


async def test_every_kind_can_open_its_windows_without_asking_whether_anyone_is_watching():
    """The engine side of the port, and the one promise it makes to a kind.

    The object every kind is handed as ``prepare``'s ``span_window`` must, per
    :class:`~threetears.evals.contracts.candidate_kind.CellSpanWindow`, be enterable with no sink
    wired. A kind that had to ask would branch on the host's tracing from inside the seam, which
    is the coupling the port exists to avoid -- and the run that wires no sink is every fixture
    run there is. Driven through the runner: a kind that opens both windows on a cell with no
    ``trace_sink`` completes, and nothing was harvested, so the cell reports no latency.
    """
    kind = _WindowOpeningKind(CandidateOutput(output=[{"content": "x"}]))

    result, trace = await _run(kind, _template())

    assert result.candidate_error is None and result.infra_error is None
    assert result.latency is None or result.latency.total_ms is None, "nothing watched, so nothing was timed"
    assert not trace.otel_trace, "nothing watched, so nothing was harvested -- not an empty harvest"


# =============================================================================
# The one-dispatch-site norm, and the kind classes' blast radius
# =============================================================================


def _eval_modules() -> list[Path]:
    """Every module of the package, ``threetears/evals/``.

    Returns:
        The paths, sorted.
    """
    return sorted(_EVAL_ROOT.rglob("*.py"))


def test_the_runner_tier_reads_a_templates_candidate_kind_exactly_once():
    """The RUN tier dispatches once, at the top of a cell, and nowhere else.

    A second reader within a tier is how a kind acquires a second meaning — one branch
    here, another three modules away — which is the creep the seam exists to stop.

    **Scoped to the run tier on purpose, and that scope is the whole amendment.** This
    began as a product-wide "exactly one reader" and was true while the only way to reach
    a kind was the runner. A LAUNCH cannot avoid reading the field: the runner's dispatch
    is a table over collaborators and can only look up a kind somebody already built,
    which is what ``RunnerOptions.candidate_kinds`` is for. So the product now has one
    reader per TIER, and that per-tier property is the one worth enforcing — a relaxation
    to "at most two anywhere" would let both sit in the runner.
    A host's own launch path holds the product-wide count; this holds the run tier's half. It
    carries THREE tiers — dispatch, construction and authoring — and asserts
    the node count beside the tier set, because the tier key alone cannot see a second read
    inside one function.
    """
    readers = [
        f"{path.relative_to(_SRC_ROOT).as_posix()}:{node.lineno}"
        for path in sorted(_EVAL_ROOT.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "candidate_kind"
    ]
    run_tier = [site for site in readers if site.startswith("threetears/evals/run/runner.py:")]

    assert len(run_tier) == 1, (
        f"``.candidate_kind`` is read at {len(run_tier)} site(s) in the runner: {run_tier}. The seam "
        "dispatches ONCE, at the top of run_one_result; everything below it reads a CandidateOutput."
    )


def test_no_engine_module_names_the_conversational_kinds_implementation():
    """The conversational kind is a host's adapter; no module of the engine may name it.

    The protocol module is the opposite and is imported freely — that is what a declaration is
    for. What must not spread is ``ConversationalTurnKind`` and what builds it: an engine module
    reaching one would be the engine knowing what a conversational subject is. The dispatch table is the runner's
    own, so ``_candidate_kind_for`` is named in the dispatch site and nowhere else.
    """
    conversational = {
        "ConversationalTurnKind",
        "ConversationalKindFactory",
        "ConversationalOptions",
        "_PreparedConversationalTurn",
    }
    offenders: list[str] = []
    for path in _eval_modules():
        rel = path.relative_to(_SRC_ROOT).as_posix()
        # Every module here is engine: the package holds the four engine trees and no adapter.
        forbidden = set(conversational)
        if rel != "threetears/evals/run/runner.py":
            forbidden = forbidden | {"_candidate_kind_for"}
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name) and node.id in forbidden:
                offenders.append(f"{rel}:{node.lineno}: {node.id}")
            if isinstance(node, ast.ImportFrom) and any(alias.name in forbidden for alias in node.names):
                offenders.append(f"{rel}:{node.lineno}: from {node.module} import …")

    assert not offenders, (
        "the conversational kind or the dispatch table is reachable where it must not be:\n" + "\n".join(offenders)
    )


def test_the_protocol_module_reaches_no_host_package():
    """The seam's declaration travels with the engine, so it imports nothing host-side.

    The repo-wide form of this is ``test_extraction_import_boundary.py``; the point of
    restating it here is that this module's whole reason for existing is to be liftable, so
    the day it acquires a host import is the day the seam stopped being one.
    """
    tree = ast.parse((_EVAL_ROOT / "contracts" / "candidate_kind.py").read_text(encoding="utf-8"))
    imported = {
        alias.name if isinstance(node, ast.Import) else f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
        if node is not None
    }
    host = sorted(
        name
        for name in imported
        if (name.split(".", 1)[0] not in sys.stdlib_module_names | {"pydantic", "threetears"})
        or (name.startswith("threetears.") and not name.startswith("threetears.evals.") and name != "threetears.evals")
    )
    assert not host, f"threetears/evals/contracts/candidate_kind.py reaches host packages: {host}"


def test_the_seam_declares_no_trace_field():
    """The no-trace disposition, as a test rather than a paragraph.

    The objection to this boundary is that ``CandidateOutput.trace`` would leak a host's
    shapes across it. The disposition taken was to keep the rich turn/span payload off the
    seam entirely: spans reach storage through the trace-sink port and the transcript IS the
    output. A ``trace`` field reappearing here is that decision being reversed by accident.
    """
    assert "trace" not in CandidateOutput.model_fields
    assert "trace" not in CandidateTelemetry.model_fields
    assert "spans" not in CandidateTelemetry.model_fields


# =============================================================================
# The kind is not part of a contestant's identity
# =============================================================================


def test_the_candidate_kind_is_a_coordinate_of_every_variant():
    """A kind is what the candidate IS, so two kinds at one model are two variants.

    This used to assert the opposite — that the kind was a property of the template and never
    entered a key. That held while the template was the only record of the kind; the run now
    carries it, and a campaign may hold runs of any template, so a key without the kind pooled two
    kinds into one arm whenever nothing else told them apart. The predicate reads it off the run,
    which is still all it is handed.
    """
    import inspect

    from threetears.evals.contracts import identity as identity_module
    from threetears.evals.contracts.host import CANDIDATE_KIND_LEVER, SHARED_CORE, HostProfile, MeasureRegistry

    params = set(inspect.signature(identity_module.derive_variant_identity).parameters)
    assert params == {"run", "profile"}, (
        f"the variant predicate takes a run and the host's vocabulary; got {sorted(params)}"
    )

    profile = HostProfile(host_id="kinds", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()))
    levers = identity_module.derive_variant_identity(
        run=make_eval_run(candidate_kind=_FAKE_KIND), profile=profile
    ).levers
    assert levers[CANDIDATE_KIND_LEVER].display == _FAKE_KIND


# =============================================================================
# The measurement conditions survive a kind that measures none
# =============================================================================


async def test_a_kind_that_samples_no_conditions_does_not_erase_the_runners_own_reading():
    """The measurement-condition covariate is a high-water mark across the cell, and the seam must not be a hole.

    The runner samples around the kind and a conversational turn keeps sampling through its
    turns; a single-shot kind samples nothing and reports ``None``. ``None`` means nobody
    looked, so it must never displace a reading that happened — which is the one way this
    covariate could have gone quiet without anything failing.
    """
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))

    result, _ = await _run(
        kind,
        _template(),
        options=RunnerOptions(
            candidate_kinds={_FAKE_KIND: lambda _cell: kind},
            concurrent_eval_jobs_probe=lambda: 2,
        ),
    )

    assert result.covariates.get("execution_mode") == "concurrent"


async def test_a_kind_that_samples_busier_conditions_wins_over_the_runners_reading():
    """The other direction of the same rule: a busier observation replaces a quieter one."""
    kind = _FakeSingleShotKind(
        CandidateOutput(output=[{"content": "x"}], telemetry=CandidateTelemetry(concurrent_eval_jobs=3))
    )

    result, _ = await _run(
        kind,
        _template(),
        options=RunnerOptions(candidate_kinds={_FAKE_KIND: lambda _cell: kind}),
    )

    assert result.covariates.get("execution_mode") == "concurrent", "the kind's own sample must reach the covariate"


# =============================================================================
# A document candidate is judged against its case material, on its rubric alone
# =============================================================================


class _RecordingJudgeLLM:
    """A judge client that records every prompt it is sent and scores each dim 4."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        """Record the call and answer the dim the system prompt names.

        Args:
            system: The system prompt, which names the dim's JSON key.
            user: The evidence rendered for the judge.
            response_format: Ignored.

        Returns:
            A completion scoring the named dim 4.
        """
        import json
        import re
        from types import SimpleNamespace

        self.calls.append((system, user))
        match = re.search(r'the single key "(.+?)"', system)
        dim_id = match.group(1) if match else "?"
        payload = json.dumps({"reasoning": f"because {dim_id}", "criteria_scores": {dim_id: 4}})
        return SimpleNamespace(
            served_model=None,
            stop_reason="end_turn",
            content=payload,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            model="judge/m",
        )


async def test_a_document_candidate_is_judged_on_its_rubric_against_the_material_it_was_given():
    """The seam a report, an extraction or a summary needs, end to end through the real runner.

    Such a candidate's output is one document, which the transcript renderer cannot show, and
    whose quality means nothing apart from what it was produced FROM. So the kind hands the
    judge both, and the engine must (a) judge even though the stored output holds no
    conversation turn, (b) put the case material and the rendered output in every prompt, and
    (c) skip the transcript and outcome axes, which have no turns or goals to read here.
    """
    from threetears.evals.contracts.models import JudgeEvidence
    from threetears.evals.contracts.models import RubricDim
    from threetears.evals.run.judge_service import JudgeService

    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"document": "report", "headline": "Keep prod's settings."}],
            judge_evidence=JudgeEvidence(
                case_material="CASE-MATERIAL: arm A 0.9, arm B 0.7", artifact="ARTIFACT: Keep prod's."
            ),
        ),
        judged_artifact=JudgedArtifact.DOCUMENT,
    )
    template = _template(
        rubric=[RubricDim(name="doc.groundedness", description="agrees with the material", scale="ordinal")]
    )
    judge = _RecordingJudgeLLM()

    result, _ = await _run(
        kind,
        template,
        judge_service=JudgeService(
            client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
        ),
    )

    assert [score.dim for score in result.rubric_scores] == ["doc.groundedness"]
    assert result.transcript_score is None and result.outcome_score is None, "a document has no conversation axes"
    assert len(judge.calls) == 1, f"only the rubric dim may be called, got {len(judge.calls)} calls"
    system, user = judge.calls[0]
    assert "CASE-MATERIAL: arm A 0.9, arm B 0.7" in user
    assert "ARTIFACT: Keep prod's." in user
    assert "# Transcript" not in user, "a document candidate's evidence must not be rendered as a transcript"
    assert "candidate's output" in system and "candidate persona" not in system
    assert "passages of the output and of the case material" in system and "transcript" not in system


async def test_a_conversational_candidate_gets_both_axes_and_the_transcript_its_kind_rendered():
    """The other side of the switch, on the same fixture: a transcript declaration, all three axes.

    The transcript is the KIND's rendering — the engine never reads the turns in ``output`` — so
    what every call carries is the artifact string, under the transcript heading.
    """
    from threetears.evals.contracts.models import JudgeEvidence, RubricDim
    from threetears.evals.run.judge_service import JudgeService

    kind = _FakeSingleShotKind(
        CandidateOutput(
            output=[{"turn": 0, "role": "candidate", "content": "a turn the engine never reads"}],
            judge_evidence=JudgeEvidence(
                subject="SUBJECT: a terse concierge",
                case_material="CASE-MATERIAL: one table left",
                artifact="Candidate: hello there",
            ),
        ),
        judged_artifact=JudgedArtifact.TRANSCRIPT,
    )
    template = _template(
        rubric=[RubricDim(name="doc.groundedness", description="agrees with the material", scale="ordinal")]
    )
    judge = _RecordingJudgeLLM()

    result, _ = await _run(
        kind,
        template,
        judge_service=JudgeService(
            client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
        ),
    )

    assert len(judge.calls) == 3, "transcript + outcome + one rubric dim"
    assert result.transcript_score is not None and result.outcome_score is not None
    for _, user in judge.calls:
        assert user.endswith("# Transcript\nCandidate: hello there")
        assert "SUBJECT: a terse concierge" in user and "CASE-MATERIAL: one table left" in user
        assert "a turn the engine never reads" not in user
    assert all("moments in the transcript" in system for system, _ in judge.calls)


def _judge_evidence() -> Any:
    """A document's evidence, for the declaration tests.

    Returns:
        Judge evidence with recognisable strings.
    """
    from threetears.evals.contracts.models import JudgeEvidence

    return JudgeEvidence(case_material="CASE-MATERIAL", artifact="ARTIFACT")


@pytest.mark.parametrize(
    ("declared", "evidence"),
    [
        (JudgedArtifact.DOCUMENT, False),
        (JudgedArtifact.TRANSCRIPT, False),
        (JudgedArtifact.UNJUDGED, True),
    ],
    ids=["a-document-with-no-evidence", "a-transcript-with-no-evidence", "an-unjudged-kind-with-evidence"],
)
async def test_a_cell_contradicting_its_kinds_declaration_is_refused_before_any_judge_call(declared, evidence):
    """The axes follow the kind's declaration, and the evidence is the kind's to render.

    The engine renders nothing a judge reads, so a judged kind — transcript or document — that
    handed back output without evidence has left its judge nothing to read; an unjudged kind that
    rendered some is contradicting its own declaration. Either is the kind's defect and names it.
    """
    from threetears.evals.contracts.models import RubricDim
    from threetears.evals.run.judge_service import JudgeService

    kind = _FakeSingleShotKind(
        CandidateOutput(output=[{"document": "extraction"}], judge_evidence=_judge_evidence() if evidence else None),
        judged_artifact=declared,
    )
    template = _template(
        rubric=[RubricDim(name="doc.groundedness", description="agrees with the material", scale="ordinal")]
    )
    judge = _RecordingJudgeLLM()
    judge_service = (
        None
        if declared is JudgedArtifact.UNJUDGED
        else JudgeService(
            client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
        )
    )

    with pytest.raises(CandidateKindDefect) as caught:
        await _run(kind, template, judge_service=judge_service)

    assert caught.value.name == _FAKE_KIND and _FAKE_KIND in str(caught.value)
    assert (caught.value.declared, caught.value.has_evidence) == (declared, evidence)
    assert judge.calls == []


@pytest.mark.parametrize("declared", [JudgedArtifact.DOCUMENT, JudgedArtifact.TRANSCRIPT])
async def test_a_judged_kind_that_produced_nothing_owes_no_evidence(declared):
    """An empty output is a cell that produced nothing to judge — a failed generation — not a defect."""
    from threetears.evals.contracts.models import RubricDim
    from threetears.evals.run.judge_service import JudgeService

    kind = _FakeSingleShotKind(CandidateOutput(candidate_errors=["generation refused"]), judged_artifact=declared)
    template = _template(
        rubric=[RubricDim(name="doc.groundedness", description="agrees with the material", scale="ordinal")]
    )
    judge = _RecordingJudgeLLM()

    result, _ = await _run(
        kind,
        template,
        judge_service=JudgeService(
            client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
        ),
    )

    assert result.candidate_error == "generation refused"
    assert judge.calls == []


async def test_a_judge_wired_to_an_unjudged_kind_is_refused_before_the_candidate_is_built():
    """Nothing a judge could read, so nothing is spent finding that out: ``prepare`` is never reached."""
    from threetears.evals.run.judge_service import JudgeService

    kind = _FakeSingleShotKind(CandidateOutput(output=[{"label": "relevant"}]), judged_artifact=JudgedArtifact.UNJUDGED)
    judge = _RecordingJudgeLLM()

    with pytest.raises(ValueError, match=f"{_FAKE_KIND}.*judged_artifact='unjudged'"):
        await _run(
            kind,
            _template(),
            judge_service=JudgeService(
                client_factory=lambda model, temperature: judge, configs={}, failure_describer=withhold_failure_detail
            ),
        )

    assert kind.prepared is None
    assert judge.calls == []


@pytest.mark.parametrize("declared", [None, "document"], ids=["undeclared", "a-bare-string"])
async def test_a_kind_without_a_declaration_is_refused_by_name_before_it_is_built(declared):
    """A host kind written to the two-operation seam gets a refusal naming the contract, not an AttributeError."""
    kind = _FakeSingleShotKind(CandidateOutput(output=[{"content": "x"}]))
    if declared is None:
        del kind.judged_artifact
    else:
        kind.judged_artifact = declared

    with pytest.raises(TypeError, match=f"{_FAKE_KIND}.*declares no judged_artifact"):
        await _run(kind, _template())

    assert kind.prepared is None


def test_every_shipped_kind_declares_what_a_judge_reads_of_it():
    """The declaration is what picks the axes, so each kind this package ships is pinned.

    The engine ships the reporter kinds; a host's own kinds (a conversational turn, a classifier)
    are pinned by that host's suite.
    """
    from threetears.evals.analysis.reporter_kind import AsRecordedReporterKind, ReporterKind

    assert ReporterKind.judged_artifact is JudgedArtifact.DOCUMENT
    assert AsRecordedReporterKind.judged_artifact is JudgedArtifact.DOCUMENT


def test_an_account_refusal_must_record_the_apparatus_fault_it_is():
    """``account_refused`` without an ``infra_errors`` entry is refused at construction.

    Without the entry nothing excludes the refused cell — it would be scored — and the run would
    stop quoting a refusal the cell never recorded.
    """
    with pytest.raises(ValueError, match="account_refused is set but infra_errors is empty"):
        CandidateOutput(account_refused=True)

    accepted = CandidateOutput(account_refused=True, infra_errors=["apparatus: refused for the calling account"])
    assert accepted.account_refused is True
    assert CandidateOutput().account_refused is False
