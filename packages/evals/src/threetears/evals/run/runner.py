"""The generic trial loop: every cell of a run, whatever kind of candidate it measures.

:func:`execute_run` runs a run's matrix — each ``(test_case × model × k_iteration)`` once, in a
shuffled order derived from the run's id (:func:`cell_execution_order`) — and :func:`run_one_result`
runs one cell: it asks the template's candidate kind for a candidate, judges what it produced, and
builds the :class:`~threetears.evals.contracts.models.EvalResult` and its sibling
:class:`~threetears.evals.contracts.models.EvalTrace`. Nothing here knows what a subject is.

The candidate-kind seam
-----------------------

Everything that differs between one evaluable subject and another lives behind one seam.
:func:`run_one_result` dispatches on ``template.candidate_kind``
**exactly once**, at the top of the cell, and everything after that point — judging, goal-state
recording, usage capture, the cell summary, storage — reads a
:class:`~threetears.evals.contracts.candidate_kind.CandidateOutput`. The protocol is
:mod:`threetears.evals.contracts.candidate_kind`; every implementation belongs to whoever owns its
subject, and reaches the runner as a :class:`KindFactory` on :attr:`RunnerOptions.candidate_kinds`
with its collaborators already bound. A host's kinds live in its own adapter; the one this
package ships is the reporter kind, :mod:`threetears.evals.analysis.reporter_kind`.

The host reaches this loop only through the :class:`~threetears.evals.contracts.host.eval_host.EvalHost`
it is handed — its storage, trace sink, cell timeout, failure describer, blocking-I/O executor and
vocabulary — and through the run's options: the kind factories and values such as the rate table, the
metered-call ledger and the judge concurrency. That is what lets any host run a trial through the same
function every other host does.

Result lifecycle
----------------

For each cell:

1. Ask the template's kind's factory for the cell's kind, and refuse a judged run of a kind no judge
   can read before anything is spent.
2. ``prepare`` the candidate. A kind that cannot build one names the termination arm, and the cell is
   recorded as one cleanly-excluded result.
3. ``invoke`` it. The kind reports its output, its mechanical facts, its telemetry and its errors —
   the candidate's own failures and the rig's apart — and, through the cell's sink, what it is waiting
   on and how to read what it has spent so far, so a cell its deadline cancels is still recorded.
   An :class:`~threetears.evals.contracts.host.apparatus.ApparatusError` out of ``prepare`` or
   ``invoke`` is recorded the same way, as that one cell excluded, and the run goes on.
4. Run the rubric judge, when a judge service is wired and the cell produced something to judge.
5. Build one :class:`~threetears.evals.contracts.models.EvalResult`, and one trace document beside it.

A cell cut off before step 5 — by its deadline, its run's cancel, or an apparatus fault — is built
from its sink instead (``_cut_short_cell``): the spend it had reported, its errors with the cut's own
charge, and, when the cut struck the judge phase, the evidence and every score that came back.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping, Sequence
from contextlib import (
    AbstractContextManager,
    contextmanager,
    nullcontext,
)
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol

from threetears.evals.contracts.candidate_kind import (
    CandidateKind,
    CandidateKindDefect,
    CandidateOutput,
    CandidatePreparationFailed,
    CellPending,
    UnknownCandidateKind,
    VariantConfig,
)
from threetears.evals.contracts.covariates import (
    count_dropped_tool_calls,
    count_refused_tool_attaches,
    count_truncated_rounds,
    derive_covariates,
)
from threetears.evals.contracts.dsl import evaluate_with_detail
from threetears.evals.contracts.errors import StorageError
from threetears.evals.contracts.host.apparatus import ApparatusError
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.host.spend import ExternalSpend
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.timeouts import EvalCellTimeout
from threetears.evals.contracts.host.traces import CellIdentity, CellTrace, TraceSink
from threetears.evals.contracts.identity import DerivedVariantIdentity, resolve_variant_identity
from threetears.evals.contracts.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    AsyncDelivery,
    CellTermination,
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    LatencyMetrics,
    PreconditionOutcome,
    RoleUsage,
    RubricScore,
    eval_trace_doc_id,
    scored_dim_ids,
    utc_now_iso,
)
from threetears.evals.contracts.scoring import CellSummary
from threetears.evals.contracts.call_ledger import CallLedger
from threetears.evals.contracts.world_events import WorldEvent
from threetears.evals.contracts.world_session import WorldSession
from threetears.evals.contracts.usage_capture import (
    ExternalRateTable,
    RoleUsageLedger,
    async_delivery_usage,
    blended_cost_roles,
    cell_cost,
)
from threetears.evals.run.cassette_proxy import CassetteCell, CassetteLane
from threetears.evals.run.judge_service import JudgeContext, JudgeOutcome, JudgeService, fold_judge_outcomes
from threetears.evals.run.metering import MeteredCallLedger, MeteredCallTally
from threetears.evals.run.offload import run_blocking, wait_through_cancellation
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.world import WorldRegistry
    from threetears.evals.run.budget import CapBreach

log = get_logger(__name__)


#: The per-cell wall-clock ceiling a caller building options by hand gets — see
#: :attr:`RunnerOptions.cell_timeout_s`. A launch whose kind can legitimately run longer raises it.
DEFAULT_CELL_TIMEOUT_S = 600.0


#: Judge calls in flight at once while one result is scored — the code default behind
#: ``eval.judge_concurrency``. Stated here rather than read from the section because the
#: runner takes configuration as values through :class:`RunnerOptions`;
#: a host with a configured setting copies it onto each run's options. See the field's docstring for why 4 and why the phase gathers at all.
DEFAULT_JUDGE_CONCURRENCY = 4


@dataclass(kw_only=True)
class RunnerOptions:
    """The run's own knobs that don't fit on the run document — per-run values, never host wiring.

    What every run of a host shares — its storage, tracing, cell timeout, executor and failure
    describer — is the :class:`~threetears.evals.contracts.host.eval_host.EvalHost` the loop is
    handed. What is here is decided per run: the kinds a launch bound for it, its ceilings and
    rates, its ledger, its probe of the job it runs in, and a test's pinned cell order.
    """

    # Judge calls in flight at once while one result is scored. The 2 + N single-dim calls
    # (transcript axis, outcome axis, one per declared rubric dim) are independent, so the
    # phase gathers them under a semaphore of this width and assembles the scores in
    # dimension order afterwards — see ``judge_dims``. ``1`` reproduces the serial phase
    # exactly. ``start_run`` copies the live ``eval.judge_concurrency`` here per run, which
    # is what makes that setting hot; a caller building options by hand gets the default.
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY
    # Per-cell hard ceiling. Bounds the WHOLE per-result execution (the kind's own work +
    # judging) through the host's cell timeout, so one hung cell is capped here instead of stalling
    # the run to the job's backstop (matrix-adaptive, ~N × this ceiling). When it
    # fires the run proceeds to the rest, and the cell keeps what it had spent and is charged by
    # what it was waiting on: the candidate's turn or its background work FAILS it, the
    # simulator, the judge or the rig EXCLUDES it (``_DEADLINE_CHARGE``). A kind whose own work
    # can legitimately run longer than the default has its launch raise this to match.
    cell_timeout_s: float = DEFAULT_CELL_TIMEOUT_S
    # Measurement-condition probe: returns how many eval jobs are executing
    # right now, INCLUDING this one. Cells inside a run are structurally serial, so the
    # contention that corrupts a latency pool comes from a concurrent JOB — which today means
    # a second RUN, and the job manager is the only thing that can see one — so the launch that
    # handed the run to a job manager supplies it, per run. The example that
    # used to stand here, a template-generation job spending on the same provider account, is
    # an instance the tree no longer has: that path went with the v6 eval surface. The probe
    # is unchanged, because what it measures is jobs, not runs, and the next non-run job
    # would contend exactly as that one would have. ``None`` (a direct or test invocation with nothing wired)
    # records no execution_mode at all rather than asserting "serial": nobody looked.
    concurrent_eval_jobs_probe: Callable[[], int] | None = None
    # Rate table for the run's counted external calls — one entry per (provider, unit),
    # resolved ONCE at launch from the operator's declared rates and the providers this
    # run can actually reach. It rides here rather than being read per cell because the
    # rates are hot-reloadable: an operator editing one mid-run would otherwise leave one
    # run's cells measured against two, with only the dollars to tell them apart.
    # ``None`` — the default for any caller that resolved nothing — leaves external calls
    # counted and unpriced, which every cost surface discloses; it is never a claim that
    # the calls were free.
    external_rates: ExternalRateTable | None = None
    # The run's metered-call ledger — the same object a kind's collaborators declare on every
    # subject they build, so what the dispatcher counts and what the runner reports are one
    # tally rather than two that can disagree. It rides here for the reason
    # ``external_rates`` does, plus one of its own: the ceiling bounds the RUN, and a
    # per-cell object would silently turn it into a per-cell ceiling N times larger.
    # ``None`` — every caller that built its own options, and every non-eval path —
    # leaves metered calls uncounted, which is what those callers already had.
    metered_calls: MeteredCallLedger | None = None
    # Injection seam for the within-run cell shuffle. ``None`` — every production
    # caller — means the order is derived from the run's own id (see
    # :func:`cell_execution_order`), so nothing has to be threaded through the
    # service to make a run reproducible. A test supplies its own
    # :class:`random.Random` when it needs to pin a specific permutation; it is a
    # generator rather than an integer seed so a caller cannot accidentally share
    # one stream between the shuffle and some future use.
    cell_order_rng: random.Random | None = None
    # The candidate kinds this run wires, keyed by the name a template carries, each as the
    # factory that builds the kind for one cell (:class:`KindFactory`), with the collaborators the
    # launch bound for this run. Every kind arrives this way: the runner has no way to build a
    # collaborator it has never heard of, and a factory is how a kind is handed the cell it is
    # driving. An empty table wires nothing, so a template naming any kind raises rather than
    # being quietly re-routed.
    candidate_kinds: Mapping[str, KindFactory] = field(default_factory=dict)


@dataclass(frozen=True)
class CellContext:
    """What the runner knows about one cell when it asks a kind's factory for the cell's kind.

    Handed to a :class:`KindFactory` once per cell, before ``prepare``. Everything on it is the
    engine's own vocabulary: the host the cell runs for (its failure describer, its world), the
    scenario and the case, the cell's coordinates, its opening measurement-conditions sample and
    the run's options — the rate table, the metered-call ledger and the conditions probe a kind's
    own measurement reads.

    Attributes:
        host: The host the run belongs to.
        template: The scenario being run.
        test_case: The cell's case.
        eval_run_id: The run this cell belongs to.
        k_iteration: Which repetition this cell is.
        concurrent_eval_jobs: The cell's opening measurement-conditions sample, which a kind that
            keeps sampling through its own work samples against.
        options: The run's options.
    """

    host: EvalHost
    template: EvalTemplate
    test_case: EvalTestCase
    eval_run_id: str
    k_iteration: int
    concurrent_eval_jobs: int | None
    options: RunnerOptions


class KindFactory(Protocol):
    """Builds the candidate kind for one cell.

    The one shape a kind reaches the runner in (:attr:`RunnerOptions.candidate_kinds`). A host
    binds the kind's collaborators when it builds the factory — a client, a subject factory, the
    host's own subject object — and the runner supplies the cell. A kind that needs nothing per
    cell returns the same instance every time, which :class:`~threetears.evals.contracts.candidate_kind.CandidateKind`
    permits because it holds no state between cells.
    """

    def __call__(self, context: CellContext) -> CandidateKind:
        """Return the kind that drives this cell.

        Args:
            context: The cell, as the runner knows it.
        """
        ...  # pragma: no cover — protocol


class CellOutcome(NamedTuple):
    """The two documents one eval cell produces.

    They are separate because ``trace`` and ``otel_trace`` are most of a stored
    result's bytes and no list, aggregate or cost path reads them — so the analysis
    record stays narrow and the payload lives in a sibling
    :class:`~threetears.evals.contracts.models.EvalTrace` keyed by the result's id.

    A :class:`~typing.NamedTuple` rather than a dataclass so a caller that only wants
    the result can unpack and ignore the rest (``result, _ = await run_one_result(...)``)
    while the fields stay named and typed.
    """

    result: EvalResult
    #: Always present, even when empty — a cell that recorded nothing still produced a
    #: definite answer about what it recorded. :meth:`EvalStorage.save_eval_result`
    #: declines to write an empty one rather than making that decision here.
    trace: EvalTrace


class EveryCellApparatusFailedError(RuntimeError):
    """Raised by :func:`execute_run` when an apparatus fault excluded every cell of the run.

    Each cell is recorded and persisted first, as any excluded cell is; this is raised after the
    last one, so the job manager files the run ``failed`` with the faults as its error. A rig that
    failed every cell measured nothing, and an operator has to repair it before a relaunch can.

    Attributes:
        total: The cells the run produced, every one excluded.
        faults: Each cell's apparatus message, in execution order.
    """

    def __init__(self, total: int, *, faults: Sequence[str]) -> None:
        """Record how many cells the rig excluded and why.

        Args:
            total: The cells the run produced.
            faults: Each cell's apparatus message.
        """
        self.total = total
        self.faults = list(faults)
        distinct = list(dict.fromkeys(self.faults))
        shown = "; ".join(distinct[:3]) + (f"; and {len(distinct) - 3} more" if len(distinct) > 3 else "")
        super().__init__(
            f"the eval apparatus failed every one of the run's {total} cell(s), so the run measured nothing — "
            f"repair the rig before relaunching: {shown}"
        )


#: Whose failure a cell deadline is, by what was pending when it struck, and how the message
#: names it: a failure is charged to whoever caused it.
#:
#: The candidate's own turn and the background work it started are the configuration under
#: test — the model a background tool runs on is a setting of it, as the candidate's model is
#: — so a
#: deadline spent waiting on either FAILS the cell. The simulator, the judge and the apparatus
#: are the rig, so a deadline spent on one of them EXCLUDES it. The charge lands on the error
#: field ``classify_result`` already reads, so there is one classification, not two.
#:
#: ``background_work`` charges the candidate because the background work a kind reports in flight
#: runs on a model of the configuration's: a tool whose background work runs no model the
#: configuration sets would need its own arm.
#:
#: ``judge`` is the rig too, and lands on the judge's own field rather than ``infra_error``: by then
#: the candidate's work has finished and been paid for, so the deadline is charged to each dim it
#: cut off (``judge_error``, which excludes the cell exactly as an infra error does) and the cell
#: keeps its evidence and the scores that came back. That is what lets a re-judge recover it — an
#: infra error is permanent, and a re-judge clears only the judge's.
_DEADLINE_CHARGE: dict[CellPending, tuple[Literal["candidate", "infra", "judge"], str]] = {
    "apparatus": ("infra", "the eval apparatus"),
    "candidate": ("candidate", "the candidate's turn"),
    "background_work": ("candidate", "background work the candidate started"),
    "simulator": ("infra", "the simulated user"),
    "judge": ("judge", "the judge"),
}


@dataclass
class ErrorLedger:
    """Accumulates a cell's errors, categorized candidate-vs-infra at the source.

    Replaces the flat ``error_details`` list so scoring can classify structurally:
    a **candidate** error is a model the candidate's configuration
    runs failing — its own turn model, or the model its background work runs on (→ fail); an **infra**
    error is a harness/delivery failure (→ exclude). A cell's deadline lands on whichever
    side :data:`_DEADLINE_CHARGE` names for what it struck. Categorizing here, at each append site, avoids re-parsing a joined
    string later (fragile-parse rule). ``combined`` reproduces the old
    ``runner_error`` display string (candidate details first, then infra).
    """

    candidate: list[str] = field(default_factory=list)
    infra: list[str] = field(default_factory=list)
    #: An apparatus fault was recorded — the measuring rig broke, so nothing this cell
    #: produces from here on is evidence about the candidate. Read by the cell loop to
    #: stop driving turns; see :meth:`add_apparatus` for why it is a flag and not a
    #: substring search over ``infra``.
    apparatus_failed: bool = False
    #: The apparatus fault was the calling ACCOUNT being refused. Carried out of the cell as
    #: ``CandidateOutput.account_refused``, which stops the whole run — see
    #: :meth:`add_account_refusal`.
    account_refused: bool = False

    @classmethod
    def of(cls, candidate: CandidateOutput | None) -> ErrorLedger:
        """The ledger a kind's report amounts to — empty when it reported nothing.

        Args:
            candidate: What the kind returned, or a reading of what it had so far, or ``None``.

        Returns:
            A fresh ledger holding the report's errors and its account refusal.
        """
        if candidate is None:
            return cls()
        return cls(
            candidate=list(candidate.candidate_errors),
            infra=list(candidate.infra_errors),
            account_refused=candidate.account_refused,
        )

    def add_candidate(self, msg: str) -> None:
        """Record a candidate-attributable error: a model the configuration runs failed, not its account.

        The candidate's own turn model, or a model its configuration sets for a tool it
        drives in the background.
        """
        self.candidate.append(msg)

    def add_infra(self, msg: str) -> None:
        """Record a harness/infra error (factory, simulator, delivery, a timeout charged to the rig)."""
        self.infra.append(msg)

    def add_apparatus(self, msg: str) -> None:
        """Record an apparatus fault: infra for scoring, plus a flag that stops the cell.

        Infra, because the scoring question is the same one — a broken rig is not the
        candidate's failure and must not be scored as one. The extra flag is what makes
        it *stop*, and that matters for a reason the classification alone does not cover:
        :func:`~threetears.evals.contracts.result_condition.classify_result` gives ``candidate_error``
        precedence over ``infra_error``, so a cell that keeps driving turns after the rig
        broke can pick up a candidate LLM error and be relabelled a candidate failure —
        a harness fault attributed to the candidate. It also stops paying for turns whose
        output cannot be scored, and stops a deterministic fault (a malformed seed, a
        cassette miss) re-firing every remaining turn.

        A flag rather than a prefix search over ``infra``: the parse-the-string version is
        the fragile-parse rule this ledger's categories exist to retire.
        """
        self.infra.append(f"apparatus: {msg}")
        self.apparatus_failed = True

    def add_account_refusal(self, msg: str) -> None:
        """Record an apparatus fault that was the calling account being refused.

        Everything :meth:`add_apparatus` does — excluded, and the cell stops — plus the flag the
        run reads: the key is refused or the balance is gone for every cell after this one too,
        so the run stops rather than paying to exclude them one by one.
        """
        self.add_apparatus(msg)
        self.account_refused = True

    @property
    def account_refusal(self) -> str | None:
        """The cell's infra errors when the calling account was refused, else None — what stops the run.

        Never ``None`` for a refused account: both ways the flag is set carry an infra entry with
        it (:meth:`add_account_refusal` records one, and ``CandidateOutput`` refuses the flag
        without one), so the run loop reads this one property rather than the flag and the
        message separately.
        """
        return self.infra_error if self.account_refused else None

    @property
    def candidate_error(self) -> str | None:
        """Joined candidate errors, or None if there were none."""
        return "; ".join(self.candidate) or None

    @property
    def infra_error(self) -> str | None:
        """Joined infra errors, or None if there were none."""
        return "; ".join(self.infra) or None

    @property
    def combined(self) -> str | None:
        """All errors joined for human display (``EvalResult.runner_error``)."""
        return "; ".join(self.candidate + self.infra) or None


@dataclass(frozen=True)
class _JudgePhase:
    """One cell's judge phase as it began: what the judge reads, and every dim it asks.

    Attributes:
        judged_artifact: The kind's declaration, which picked the axes.
        evidence: What the kind rendered for its judge.
        dims: Every dim the phase asks, in the order it asks them (:func:`scored_dim_ids`).
    """

    judged_artifact: JudgedArtifact
    evidence: JudgeEvidence
    dims: tuple[str, ...]


class _JudgeRecord(NamedTuple):
    """What a cell's judge phase leaves on its result, however the phase ended."""

    transcript_score: RubricScore | None
    outcome_score: RubricScore | None
    rubric_scores: list[RubricScore]
    #: Mirrors the outcome axis's reasoning — the holistic "did it work" signal.
    judge_reasoning: str
    judge_config_ids: dict[str, str]
    #: ``"<dim>: <error>"`` for each dim that failed or did not finish, joined; ``None`` when none.
    judge_error: str | None
    judge_cannot_tell: dict[str, str]
    #: The judge role's rows, for the calls that returned — what the result's cost reads them from.
    usage: list[RoleUsage]


def _judge_record(
    phase: _JudgePhase, outcomes: Sequence[tuple[str, JudgeOutcome]], *, unfinished: str | None = None
) -> _JudgeRecord:
    """Fold a judge phase's returned calls into what the result stores, naming any dim that did not finish.

    One fold for every way the phase ends — every call returned, the phase raised, or the cell was
    cut off mid-phase — so a paid score is kept whichever it was. Outcomes are put in the phase's
    own dim order first, whatever order they arrived in, so a cut-off phase's scores sit where a
    finished one's would.

    Args:
        phase: The phase as it began.
        outcomes: ``(dim_id, outcome)`` for each call that returned.
        unfinished: Why a dim with no returned call did not finish. ``None`` asserts every call
            returned.

    Returns:
        The record.

    Raises:
        RuntimeError: ``unfinished`` is ``None`` and a dim the phase asks returned no call — the
            judge phase dropped one, which no record may hide.
    """
    first_index = {dim: index for index, dim in reversed(list(enumerate(phase.dims)))}
    settled = sorted(outcomes, key=lambda pair: first_index.get(pair[0], len(phase.dims)))
    returned = Counter(dim for dim, _ in settled)
    cut: list[str] = []
    for dim in phase.dims:
        if returned[dim]:
            returned[dim] -= 1
        else:
            cut.append(dim)
    if cut and unfinished is None:
        raise RuntimeError(f"the judge phase returned no call for {', '.join(cut)}")
    folded = fold_judge_outcomes(settled)
    axes = {dim: outcome for dim, outcome in settled if dim in (TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID)}
    transcript = axes.get(TRANSCRIPT_DIM_ID)
    outcome = axes.get(OUTCOME_DIM_ID)
    outcome_score = outcome.score if outcome is not None else None
    errors = [*folded.errors, *((dim, unfinished) for dim in cut if unfinished is not None)]
    return _JudgeRecord(
        transcript_score=transcript.score if transcript is not None else None,
        outcome_score=outcome_score,
        rubric_scores=[oc.score for dim, oc in settled if dim not in axes and oc.score is not None],
        judge_reasoning=outcome_score.reasoning if outcome_score is not None else "",
        judge_config_ids=folded.config_ids,
        judge_error="; ".join(f"{dim}: {error}" for dim, error in errors) or None,
        judge_cannot_tell=folded.cannot_tell,
        usage=folded.usage,
    )


@dataclass
class _CellSink:
    """The engine's record of one cell as it runs — everything a deadline's record is built from.

    :func:`execute_run` bounds each cell with the host's ``cell_timeout``, which cancels it from
    outside; every local of the cancelled frame dies with it, so a record of what the cell had
    spent, or of what it was waiting on, cannot be taken from its return value.
    The run loop builds one of these per cell, hands it down through :func:`run_one_result` to
    the kind's ``invoke`` — the kind-facing half is
    :class:`~threetears.evals.contracts.candidate_kind.CellSink` — and reads it in its own timeout
    arm, which reads nothing else about the cell.

    It holds two things the kind reports — :attr:`pending` and how to read the candidate side
    so far — and three the engine observes for itself: the judge calls that returned, the kind
    dispatched to, and what ``invoke`` returned once it has. **Nothing may assign
    :attr:`pending` in a ``finally``** — a cancel unwinds through those, and a write there would
    overwrite the answer with whatever the unwind was doing.
    """

    #: What the cell is waiting on right now — the component a deadline striking at this
    #: moment would be waiting for. See :data:`_DEADLINE_CHARGE` for whose failure each is.
    pending: CellPending = "apparatus"
    #: Each judge call that has returned, in completion order — appended as each settles,
    #: so a deadline mid-phase keeps the calls already paid for.
    judged: list[tuple[str, JudgeOutcome]] = field(default_factory=list)
    #: The kind the cell dispatched to, once it has, for the record the deadline arm writes.
    kind_name: str | None = None
    #: What the kind's ``invoke`` returned, whole, once it has — the one source both the
    #: completed cell's record and a deadline in judging read the candidate side from.
    returned: CandidateOutput | None = None
    #: The cell's errors as the run loop reads them after the cell, however it ended: built
    #: from what ``invoke`` returned, or at a deadline from the kind's reading plus the
    #: deadline's own charge.
    errors: ErrorLedger = field(default_factory=ErrorLedger)
    #: How to read the candidate side so far, as the kind registered it — see
    #: :meth:`~threetears.evals.contracts.candidate_kind.CellSink.report_progress`.
    progress: Callable[[], CandidateOutput] | None = None
    #: The judge phase, once it has begun — what it reads and every dim it asks — so a cell cut
    #: off while it is judged keeps the evidence and can name the dims that did not finish.
    judging: _JudgePhase | None = None
    #: The cell's handle on the run's cassette lane, once it has one — ``None`` for a run with
    #: cassettes off. Kept here so every exit, a cut-short one included, closes it and holds the
    #: kind's report to what the cell replayed.
    cassettes: CassetteCell | None = None
    #: The cell's handle on the host's world — ``None`` for a host that declares none. Kept here, and
    #: built by the engine rather than the kind, so every exit records what fired before it and the
    #: end state if it was read: the session survives a cancel that destroys the kind's frame.
    world: WorldSession | None = None
    #: The one reading taken, once :meth:`side` has taken it.
    _read: CandidateOutput | None = None

    @property
    def world_events(self) -> list[WorldEvent] | None:
        """What moved the cell's world, or ``None`` when no world was opened for it.

        Returns:
            The session's events once the kind seeded through it; ``None`` for a host with no world and
            for a kind that never seeded — nobody opened the world, which is not "nothing moved it".
        """
        if self.world is None or not self.world.opened:
            return None
        return list(self.world.events)

    @property
    def end_state(self) -> dict[str, Any] | None:
        """The world the cell left, once it was read; ``None`` when it never was.

        Returns:
            The session's one end-state reading, or ``None``.
        """
        return self.world.end_state_read if self.world is not None else None

    def waiting_on(self, pending: CellPending) -> None:
        """Name what the cell is waiting on from here.

        Args:
            pending: The component the next await is waiting for.
        """
        self.pending = pending

    def report_progress(self, read: Callable[[], CandidateOutput]) -> None:
        """Register how to read the candidate side so far.

        Args:
            read: What ``invoke`` would return if the cell ended now.
        """
        self.progress = read

    def begin_judging(self, phase: _JudgePhase) -> None:
        """Mark the judge phase begun: the cell now waits on the judge, and the phase is on record.

        One method for both, so a cell cut off while it waits on the judge always has the phase
        whose dims it names as unfinished.

        Args:
            phase: What the judge reads and the dims it asks.
        """
        self.judging = phase
        self.pending = "judge"

    def adopt(self, candidate: CandidateOutput) -> None:
        """Hold what the kind's ``invoke`` returned as the cell's candidate side.

        Any kind: once ``invoke`` returns, its output is the authority on what the candidate
        side spent, produced and failed at, and a cell cancelled while it is judged still has
        all of it — kept whole, so a field the completed record stores cannot be missing from
        the deadline's record for want of a second copy here.

        Args:
            candidate: What the kind's ``invoke`` returned.
        """
        self.returned = candidate
        self.errors = ErrorLedger.of(candidate)

    def side(self) -> CandidateOutput | None:
        """The candidate side as the cell left it: what ``invoke`` returned, else the kind's reading.

        Reads the kind's registered reading at most once, and only when ``invoke`` never
        returned — called by the deadline arm after the cancel has unwound, so what the unwind
        itself recorded is in it. ``None`` when the kind returned nothing and registered no
        reading: it reported nothing, which the record then says.

        Returns:
            The candidate side, or ``None``.
        """
        if self.returned is not None:
            return self.returned
        if self._read is None and self.progress is not None:
            self._read = self.progress()
        return self._read

    @property
    def account_refusal(self) -> str | None:
        """What stops the run after this cell: a call the provider refused for the calling account.

        The candidate side and the simulator report through :attr:`errors`; a judge call reports
        on its outcome in :attr:`judged`, which is kept however the cell ended. Either way the
        cell is excluded, and every later cell would meet the same key.

        Returns:
            The refusal's message, or ``None`` when no call of this cell was refused for the account.
        """
        if (refusal := self.errors.account_refusal) is not None:
            return refusal
        judged = [f"judge {dim_id}: {outcome.error}" for dim_id, outcome in self.judged if outcome.account_refused]
        return "; ".join(judged) or None


@dataclass
class _CellTraceScopes:
    """The two trace-sink windows one cell opens, and the record they leave behind.

    **The runner owns the harvest and a kind owns only the extents**, and that split is the
    no-host-shapes rule made structural: the host-shaped span payload never crosses the
    candidate-kind seam (see :mod:`threetears.evals.contracts.candidate_kind`), while the decision of
    *where* each window opens stays with the code that knows which work is a turn and which
    is apparatus.

    This is the engine side of :class:`~threetears.evals.contracts.candidate_kind.CellSpanWindow`, which
    is the whole of what a kind is handed — structurally, so the two scopes are all a kind
    can reach, and the sink, the cell identity and the record stay on this side of the seam.

    Two scopes rather than one, and the difference between their extents is a measurement
    decision: identity covers the slot startup and teardown, because that work is the
    cell's; collection does not, because it is apparatus rather than a turn and belongs in
    no bucket the candidate is compared on.
    """

    #: The host's tracing, or ``None`` — a decision made at the call site, never a mechanism
    #: that stopped working. A sink that is present and broken is the other case entirely.
    sink: TraceSink | None
    #: The identity both scopes are opened against — one object per cell, handed to both.
    cell: CellIdentity
    #: The sink's record, once a collection window has opened and closed. Read through
    #: rather than checked: a sink yielding anything but the record is a programming error,
    #: and :attr:`harvested` is what separates that from nobody having watched.
    collected: CellTrace | None = None
    #: A collection window completed. ``False`` covers both "no sink" and "the kind opened
    #: none", and keeps either from being read as a sink that collected nothing.
    harvested: bool = False

    def identity(self) -> AbstractContextManager[None]:
        """Mark everything inside as this cell's work, apparatus included.

        Returns:
            The sink's identity scope, or a no-op when nothing is watching.
        """
        return self.sink.cell_identity(self.cell) if self.sink is not None else nullcontext()

    @contextmanager
    def collecting(self) -> Iterator[None]:
        """Collect the spans the candidate's own work emits.

        The record is read after the scope closes, never inside it: the sink narrows to this
        cell as it exits, so reading early asks for an answer it has not worked out yet. A
        body that raises leaves :attr:`harvested` false and the cell recording no trace —
        the same answer as a cell that never ran its turns.

        Yields:
            Nothing. The record lands on :attr:`collected` once the window closes.
        """
        if self.sink is None:
            yield
            return
        with self.sink.cell_spans(self.cell) as collected:
            yield
        self.collected = collected
        self.harvested = True


def _candidate_kind_for(name: str, *, context: CellContext) -> CandidateKind:
    """Build the kind a template names — the table behind the seam's one dispatch.

    Every kind arrives as a factory on :attr:`RunnerOptions.candidate_kinds`, because the runner
    has no way to build a collaborator it has never heard of; the factory is handed the cell.

    Args:
        name: ``template.candidate_kind`` — read at exactly one place, by the caller.
        context: The cell, carrying the run's options and so the kinds this host wired.

    Returns:
        The kind to prepare and invoke.

    Raises:
        UnknownCandidateKind: No kind by that name is wired on this host.
    """
    wired = context.options.candidate_kinds
    if (factory := wired.get(name)) is None:
        raise UnknownCandidateKind(name, known=list(wired))
    return factory(context)


def _busier_conditions_sample(previous: int | None, observed: int | None) -> int | None:
    """Keep the busier of two measurement-condition samples (R4).

    The runner samples around the kind and the kind samples through its own work, and the
    covariate is the high-water mark across all of it. ``None`` on either side is "nobody
    looked", which never displaces a real reading and never becomes one.

    Args:
        previous: The busiest reading so far, or ``None``.
        observed: A further reading, or ``None``.

    Returns:
        The busier of the two, or ``None`` when neither was measured.
    """
    if observed is None:
        return previous
    return observed if previous is None else max(previous, observed)


async def run_one_result(
    host: EvalHost,
    *,
    template: EvalTemplate,
    test_case: EvalTestCase,
    subject_id: str,
    model: str,
    k_iteration: int,
    eval_run_id: str,
    scope_id: str,
    judge_service: JudgeService | None,
    options: RunnerOptions,
    variant: DerivedVariantIdentity,
    cassettes: CassetteLane | None = None,
    subject_snapshot: SubjectSnapshot | None = None,
    judge_model: str | None = None,
    sink: _CellSink | None = None,
) -> CellOutcome:
    """Run one ``(test_case, model, k)`` and return the two documents it produced.

    Pure-function-style: no persistence here, caller decides when to save. Returns a
    :class:`CellOutcome` — the analysis record and its debug payload are separate
    documents, so a function that runs a cell hands back both rather than one object
    that pretends to be both.

    **The candidate is built and driven behind one seam**: this
    function dispatches on ``template.candidate_kind`` exactly once, and everything after
    that point — judging, goal-state recording, usage capture, the cell summary, storage —
    reads a :class:`~threetears.evals.contracts.candidate_kind.CandidateOutput` and knows nothing about
    subjects, simulators or worlds. That is what makes one refactor cover every kind of
    subject rather than one.

    The HOST half of the subject pair is not an argument here: a kind that needs the host's
    rich subject has it bound into its factory by the host that built it, once per run, so the
    engine never holds it.

    ``subject_snapshot`` is the ENGINE half of that pair — a key, a label and content
    hashes, and nothing else. ``None`` for a caller that resolved none, which is every
    direct and fixture invocation; the run loop passes the run's own. It is deliberately
    NOT the host's own record of its subject: a snapshot that retained what the candidate was
    made of would scale a run's retention with what it produced rather than with its
    matrix.

    ``subject_id`` is the run's subject key, passed separately rather than
    read off the host's subject — which rides on the run's optional host
    payload, and an ad-hoc run legitimately carries none, while the key is
    required and non-empty. Reading it off that payload stamped a blank
    subject onto results of runs that had one.

    ``cassettes`` is the run's cassette lane (:meth:`CassetteLane.for_run`), ``None`` for a run with
    cassettes off. The cell's own handle on it is handed to the kind's ``prepare``, which must wire
    its candidate's seams through it; a kind that does not is refused before ``invoke``, since its
    tools would run live under a replay. The kind's report of the background work the cell replayed
    is held to what the cell actually served.

    ``variant`` is this cell's resolved contestant identity, computed once per
    model by the caller (which holds the run) and stamped onto every result
    this function returns, successful or not. Required: every result carries the key
    (:func:`~threetears.evals.contracts.identity.resolve_variant_identity`).

    ``judge_model`` is the judge the run records, stamped onto the result on every exit. The run
    loop passes ``run.judge_model``, the one carrier of that fact; a direct caller passes the
    judge it wired, or ``None`` when nothing judges the cell.

    ``sink`` is the cell's :class:`_CellSink`, owned by the caller so it survives this function
    being cancelled: the run loop passes one, this function hands it to the kind's ``invoke``,
    and the run loop reads it when the cell's deadline strikes, which is the only way a
    cancelled cell's spend and pending work reach its record. ``None`` — a direct caller that
    bounds nothing — builds a fresh one. It is also how a refusal for the calling account
    (``CandidateOutput.account_refused``) reaches the run: the cell is recorded as any other
    apparatus fault is, and :func:`execute_run` reads ``sink.account_refusal`` after the
    cell however it ended.
    """
    if sink is None:
        sink = _CellSink()
    # 1. The cell's own bookkeeping, above the seam and independent of what kind of
    # candidate this is.
    # Attribution metadata: the REAL subject the run was launched
    # against rides on the result as metadata only — the candidate's runtime identity is
    # whatever the kind minted, and comes back on its output.
    #
    # Taken from the RUN's subject key, which is required and non-empty, rather than off the
    # subject recovered from the host payload — that payload is optional and yields an empty
    # snapshot for an ad-hoc or fixture run, so reading it stamped ``subject_id=""`` onto results
    # of a run that had a perfectly good key. Those results then fell out of every
    # subject-partitioned surface, and the counter that would have disclosed it is gone: a blank
    # subject is unrepresentable ON THE RUN, which is what retired the counter, and was not yet
    # unrepresentable on the RESULT.
    real_subject_id = subject_id
    # R4: measurement conditions are sampled around the work, not read off it afterwards —
    # by the time the result is built, a concurrent run that contended for the provider
    # quota may already have finished.
    concurrent_eval_jobs = sample_concurrent_eval_jobs(options)

    def _excluded_cell(
        *, error: str, termination: CellTermination, preconditions: Sequence[PreconditionOutcome] = ()
    ) -> CellOutcome:
        """Record one cleanly-excluded cell for a fault that happened before any turn.

        Every fault that reaches this is a kind failing to PREPARE a candidate — for the
        conversational turn, the factory, either half of the seeding step, or a precondition
        that did not hold — and they share everything except which one it was. The
        ``termination`` the kind raised with is what tells an operator where to look, and
        telling them wrongly sends them to the wrong half of the apparatus.

        No candidate turn ran, but the CELL did — see :func:`_degraded_capture_fields` for why
        every capture field then says "measured nothing" rather than "not captured". Nothing was
        spent on any of these paths, so the sink holds nothing and the zero total is a real sum.

        Args:
            error: The message, already prefixed with what kind of fault it was.
            termination: Which arm the kind named.
            preconditions: The t=0 assertion's outcomes, for the arm that has them.

        Returns:
            The cell, with an empty trace.
        """
        result = EvalResult(
            scope_id=scope_id,
            eval_run_id=eval_run_id,
            test_case_id=test_case.id,
            model=model,
            # Stamped on the excluded arm too: a cell whose candidate could never be built is
            # still a cell of a run that knows which kind it was asking for. Read off the
            # dispatch's own local, which is bound before this closure's single call site.
            candidate_kind=candidate_kind_name,
            k_iteration=k_iteration,
            subject_id=real_subject_id,
            runner_error=error,
            infra_error=error,
            precondition_outcomes=list(preconditions),
            **_degraded_capture_fields(
                sink=sink,
                variant=variant,
                concurrent_eval_jobs=concurrent_eval_jobs,
                termination=termination,
                judge_model=judge_model,
                rate_table=options.external_rates,
            ),
        )
        return CellOutcome(result, _empty_trace(result_id=result.id, scope_id=scope_id, eval_run_id=eval_run_id))

    def _apparatus_cell(fault: ApparatusError, *, during: Literal["prepare", "invoke"]) -> CellOutcome:
        """Record one cell an apparatus fault ended: excluded, with what it had spent and why.

        The candidate side is whatever the kind had reported through the cell's sink when the fault
        unwound it — its reading registered with ``report_progress`` — so the spend already billed
        is on the record, and the fault is recorded through :meth:`ErrorLedger.add_apparatus`,
        which excludes the cell and is what :func:`execute_run` reads to tell a rig that failed
        every cell from one that failed some.

        Args:
            fault: What reached the engine.
            during: Which of the kind's operations raised it, for the operator's log line.

        Returns:
            The cell, with whatever output the kind had produced as its trace.
        """
        log.warning(
            "Eval cell excluded: an apparatus fault reached the engine from %s's %s (test_case=%s model=%s k=%s "
            "run=%s): %s: %s",
            candidate_kind_name,
            during,
            test_case.id,
            model,
            k_iteration,
            eval_run_id,
            type(fault).__name__,
            fault,
        )
        errors = sink.errors = ErrorLedger.of(sink.side())
        errors.add_apparatus(f"{type(fault).__name__}: {fault}")
        side = sink.side()
        return _cut_short_cell(
            sink=sink,
            scope_id=scope_id,
            eval_run_id=eval_run_id,
            test_case_id=test_case.id,
            model=model,
            k_iteration=k_iteration,
            subject_id=real_subject_id,
            variant=variant,
            concurrent_eval_jobs=sample_concurrent_eval_jobs(options, concurrent_eval_jobs),
            termination="apparatus_failed",
            judge_model=judge_model,
            rate_table=options.external_rates,
            # The judge phase follows ``invoke``, so it never began; nothing is left unfinished.
            unfinished_judging=None,
            metered_calls_refused=_cell_metered_refusals(
                metered_cell_tally(options.metered_calls, metered_baseline),
                side.telemetry.metered_calls_refused if side is not None else None,
            ),
        )

    # A NONCE rather than the cell's coordinates, and that is the whole point:
    # this run's own next model and next k iteration reuse the same
    # (run, test case) pair, so coordinates would pool a cell with its own
    # siblings exactly when a concurrent job keeps the shared buffer alive
    # across them. Minted per execution, so nothing else can carry it.
    trace_scopes = _CellTraceScopes(
        sink=host.trace_sink,
        cell=CellIdentity(
            run_id=eval_run_id,
            test_case_id=test_case.id,
            cell_id=str(uuid.uuid7()),
            subject_id=subject_id,
        ),
    )

    # =========================================================================
    # 2. THE DISPATCH — the one site that reads ``template.candidate_kind``
    #
    # Everything above is the cell's own bookkeeping and everything below sees a
    # ``CandidateOutput``: an ``output`` to judge, ``mechanical_facts`` to record, and the
    # telemetry the cell measured. No code below this line drives a turn, builds a subject,
    # or reads a world — which is the property that makes one refactor cover every kind.
    #
    # **What a judge reads arrives the same way**: the subject, the case material and the
    # artifact are the kind's to render (``CandidateOutput.judge_evidence``), so the runner hands
    # them to the judge's context unread and holds no subject or transcript shape of its own.
    # =========================================================================
    # Read ONCE, into a name the rest of the cell uses: the seam's whole point is that
    # ``template.candidate_kind`` has exactly one reader, and a second read — even a log
    # line's — is a second place that knows what a subject is. ``test_candidate_kind.py``
    # caught exactly that when the no-window warning below reached for the template again.
    candidate_kind_name = template.candidate_kind
    kind = _candidate_kind_for(
        candidate_kind_name,
        context=CellContext(
            host=host,
            template=template,
            test_case=test_case,
            eval_run_id=eval_run_id,
            k_iteration=k_iteration,
            concurrent_eval_jobs=concurrent_eval_jobs,
            options=options,
        ),
    )
    sink.kind_name = candidate_kind_name
    # What a judge reads of this kind's output is the kind's own declaration, read before
    # ``prepare`` so a judged run of a kind no judge can read is refused before anything is
    # spent. Like an unknown kind, it would fail every cell identically.
    # A kind that omits the declaration would otherwise fail on a bare AttributeError here;
    # name the missing contract instead.
    judged_artifact = getattr(kind, "judged_artifact", None)
    if not isinstance(judged_artifact, JudgedArtifact):
        raise TypeError(
            f"candidate kind {candidate_kind_name!r} declares no judged_artifact (got {judged_artifact!r}); every "
            "CandidateKind states what a judge reads of it — JudgedArtifact.TRANSCRIPT, DOCUMENT or UNJUDGED"
        )
    if judge_service is not None and judged_artifact is JudgedArtifact.UNJUDGED:
        raise ValueError(
            f"candidate kind {candidate_kind_name!r} declares judged_artifact={judged_artifact.value!r}, so there is "
            "nothing for a judge to read, and this run wires a judge service; pass no judge service, or wire a kind "
            "that renders what a judge reads"
        )
    # Baseline for this CELL's slice of the run's metered-call ledger, taken before the kind
    # builds anything and read again after it returns. It lives at the dispatch site rather
    # than inside a kind because the ledger belongs to the RUN: a kind that never thought to
    # count is the default case, and a per-kind count is a line every future kind has to
    # remember. ``ClassifierKind`` did not, and every classifier cell stored ``None`` under a
    # ledger that was in force — the value ``EvalResult.metered_calls_refused`` documents as
    # meaning nobody counted. Cells are structurally serial, so a before/after pair around the
    # whole dispatch is exactly what this cell did, ``prepare`` included.
    metered_baseline = options.metered_calls.tally() if options.metered_calls is not None else None
    # The cell's handle on the run's cassette lane, already bound to its corpus, template and case,
    # so one kind instance can drive every cell without being told which it is driving.
    cell_cassettes = (
        cassettes.cell(template_id=template.id, test_case_id=test_case.id, model=model)
        if cassettes is not None
        else None
    )
    sink.cassettes = cell_cassettes
    # The cell's handle on the host's world, built here rather than by the kind so that what the
    # kind does to its world — the seed, the triggers, the end state — is on the sink whatever way
    # the cell ends. ``None`` for a host that declares no world. Built over the profile's registry, which
    # is the declaration; a host whose world is per-cell state binds this cell's own table on it
    # (``WorldSession.bind``) from ``prepare``, so no two cells — of this run or a concurrent one — share
    # a path to one world, and a host declaring ``binds_per_cell`` that forgets is refused at seed.
    world_session = WorldSession(host.profile.world) if host.profile.world is not None else None
    sink.world = world_session
    try:
        # The subject arrives as a PAIR and the split is the point: the engine's snapshot is a
        # key, a label and content hashes, handed over here, while the host's rich object was
        # bound into the kind's factory by its host and is read by the kind alone.
        instance = await kind.prepare(
            subject_snapshot=subject_snapshot,
            variant_config=VariantConfig(candidate_model=model),
            world_seed=template.world_seed,
            # Every kind alike, and the reason this is an argument rather than something a
            # kind reaches for: the windows are this CELL's, minted above with its nonce, and
            # a kind wired for a whole run would otherwise have to be told which cell it is
            # driving to find them.
            span_window=trace_scopes,
            cassettes=cell_cassettes,
            world=world_session,
        )
    except CandidatePreparationFailed as failure:
        # A kind that cannot build its candidate records one cleanly excluded cell, under the
        # arm the kind itself named — because naming the wrong half of the apparatus sends an
        # operator to the wrong place, and only the kind knows which half it was.
        return _excluded_cell(
            error=failure.error,
            termination=failure.termination,
            preconditions=failure.preconditions,
        )
    except ApparatusError as fault:
        return _apparatus_cell(fault, during="prepare")
    if cell_cassettes is not None and not cell_cassettes.wired:
        # Refused before anything runs, and the RUN rather than the cell: a kind that does not wire
        # the handle it was given is the kind's own code, so every cell would do the same, and its
        # tools would run live under a replay, spending exactly what the replay exists not to spend.
        raise ValueError(
            f"candidate kind {candidate_kind_name!r} was handed this cell's cassettes in a "
            f"{cell_cassettes.mode} run and did not wire them; a kind wires its candidate's CassetteSeams "
            "through prepare's cassettes before returning, or the run launches with cassette_mode='off'"
        )
    # The candidate is what the cell now waits on; a kind refines this through the sink as its
    # work alternates with other components' — the conversational kind's with the simulator and the rig.
    sink.waiting_on("candidate")
    try:
        candidate = await kind.invoke(instance, test_case, sink)
        if world_session is not None and world_session.opened:
            # The world the cell LEFT, read back through every attached dimension's ``read`` handle once
            # ``invoke`` has returned — the reading the kind took to grade, if it took one, since the
            # session reads once. Here rather than in each kind, so a kind that never thought to read
            # its world still stores what its candidate left behind, and none can store the seed instead.
            sink.waiting_on("apparatus")
            await world_session.end_state()
    except ApparatusError as fault:
        # The rig broke under the candidate — a replay miss, a corrupt recording, a harness fault
        # a kind's tool boundary re-raised as the seam tells it to. That is THIS cell's exclusion,
        # never the run's: every later cell may meet a working rig. The cell's record is built
        # from its sink, which closes the cassette handle and folds its background work's spend.
        return _apparatus_cell(fault, during="invoke")
    finally:
        # However ``invoke`` ended — returned, raised, or cancelled by the deadline — the capture's
        # session ended here, and background work it left in flight is recorded as such. The
        # cut-short record closes it too (idempotently), for the exits that are built before this runs.
        if cell_cassettes is not None:
            cell_cassettes.close()
    sink.adopt(candidate)
    if cell_cassettes is not None:
        cell_cassettes.hold(candidate.async_deliveries, complete=True)
    # Held to the declaration before anything below reads the output — see :func:`hold_to_declaration`.
    hold_to_declaration(candidate_kind_name, judged_artifact, candidate)
    metered_cell = metered_cell_tally(options.metered_calls, metered_baseline)

    # 3. Below the dispatch: ``output`` + ``mechanical_facts`` + telemetry, and nothing
    # kind-shaped. ``trace`` is the judged artifact AND the stored transcript — one document,
    # because for every kind so far they are the same one.
    telemetry = candidate.telemetry
    trace = candidate.output
    goal_outcomes = candidate.mechanical_facts
    # Refused HERE, before the judge phase is paid for, though the rows are folded only when the
    # cell is assembled: a kind double-reporting its background work's spend is its own code, so
    # every cell would do it, and judging the cell first would spend on a record that cannot be built.
    refuse_inner_agent_usage(telemetry.usage)

    # The trace sink's record of the candidate's own work, handed to the assembly whole: it reads
    # the spans and the three named latency buckets off it. ``None`` for a cell that wired no sink,
    # and for one whose kind opened no collection window.
    spans: CellTrace | None = None
    # Branching on whether a window CLOSED, not on what it yielded. ``collected is None``
    # would also be what a sink yielding None produces, and recording that as "nobody was
    # watching" would file a broken sink under the one state the design says must stay
    # distinguishable from it. Read through the record instead and let such a sink raise
    # where it can be attributed.
    if trace_scopes.harvested:
        collected = trace_scopes.collected
        if collected is None:
            raise TypeError(
                f"trace sink {type(host.trace_sink).__name__} closed a collection window having yielded "
                "None from cell_spans rather than a CellTrace"
            )
        spans = collected
        if collected.total_ms is None and host.trace_sink is not None:
            # The window opened, closed, and nothing the total is summed from landed in it.
            # The cell still leaves the cost-vs-latency comparison, so it is still disclosed —
            # separately, because the repair is the opposite one. "Opened no window" is fixed
            # by opening one; this is fixed on the candidate's own call path, which has to
            # emit the turn-root span the total is summed across (``summarize_span_durations_ms``
            # sums ``agent.invoke``). A kind driving a scripted client emits none by nature,
            # and that is a fixture reading an honest zero-coverage answer rather than a bug.
            log.warning(
                "Eval run %s: candidate kind %r opened a span collection window for test_case=%s and no "
                "turn-root span landed in it, so this cell reports no total latency and drops out of the "
                "cost-vs-latency comparison — the candidate's call path emits none",
                eval_run_id,
                candidate_kind_name,
                test_case.id,
            )
    elif host.trace_sink is not None and telemetry.untimed_reason is not None:
        # The kind SAYS there was nothing to time. The cell still leaves the latency axis — it
        # has no latency — but the warning below would blame the kind for an omission it did
        # not make, once per cell of every such run.
        log.info(
            "Eval run %s: candidate kind %r did no timed work for test_case=%s (%s); this cell reports no latency",
            eval_run_id,
            candidate_kind_name,
            test_case.id,
            telemetry.untimed_reason,
        )
    elif host.trace_sink is not None:
        # A sink was wired, the kind was handed this cell's windows, and none ever closed — so
        # the cell reports no latency at all, and `decompose_total_ms` and the frontier read
        # that as "timed nothing" and drop it from the axis every campaign compares cost
        # against. Said out loud because the alternative is a cell that silently leaves the
        # comparison. It is the kind's own omission, not a reach it lacks: the windows arrive
        # as ``prepare``'s ``span_window``, which every kind alike is given.
        log.warning(
            "Eval run %s: candidate kind %r opened no span collection window for test_case=%s, so this cell "
            "reports no latency at all and drops out of the cost-vs-latency comparison — every kind is handed "
            "this cell's windows as prepare's span_window and this one opened neither",
            eval_run_id,
            candidate_kind_name,
            test_case.id,
        )

    # 4. Judge — dual-score axes (transcript + outcome) are universal; per-dim
    # rubric scores only when the template declares dimensions. Judging needs a
    # judge service and a non-empty transcript; factory / immediate-error slots
    # (no trace) skip it and leave the scores None.
    judged = _unjudged_record()
    # Timed off a monotonic clock, not harvested: the judge emits no spans (llm.call
    # spans come from the candidate's own call path alone), and it runs after the harvest
    # closes, so it sits outside every turn-root span the total sums. None until the
    # phase actually runs — see LatencyMetrics.judge_ms.
    judge_ms: float | None = None
    # ``trace`` alone gates the phase whatever the kind: empty output means nothing was
    # produced to judge. Which axes are judged is the kind's declaration; what each reads is the
    # evidence the kind rendered, which the declaration check above guarantees is here.
    judge_evidence = candidate.judge_evidence
    if judge_service is not None and trace and judge_evidence is not None:
        phase = _JudgePhase(
            judged_artifact=judged_artifact,
            evidence=judge_evidence,
            dims=tuple(scored_dim_ids([dim.name for dim in template.rubric], judged_artifact)),
        )
        judge_started = time.monotonic()
        sink.begin_judging(phase)
        try:
            outcomes = await judge_dims(
                template=template,
                judge_service=judge_service,
                concurrency=options.judge_concurrency,
                settled=sink.judged,
                context=build_judge_context(
                    template=template,
                    test_case=test_case,
                    goal_outcomes=goal_outcomes,
                    judged_artifact=judged_artifact,
                    judge_evidence=judge_evidence,
                ),
            )
        except Exception as e:  # prawduct:ok-broad-except — judge boundary; a judge failure must not wedge the slot
            log.exception("Judge service failed during eval run %s", eval_run_id)
            # The calls that returned before it raised were paid for and keep their scores; the
            # rest are named as failed with the cause.
            judged = _judge_record(phase, sink.judged, unfinished=f"the judge phase raised {type(e).__name__}: {e}")
        else:
            judged = _judge_record(phase, outcomes)
        finally:
            # Stamped on the way out of EVERY exit, not just the scoring one. A judge
            # that raised after two minutes spent those two minutes, and leaving the
            # phase untimed there would file real wall-clock under the same absence as
            # a run that never had a judge at all — judge_error is what says it failed.
            judge_ms = (time.monotonic() - judge_started) * 1000.0

    # Second (and final) conditions sample — see sample_concurrent_eval_jobs. Taken after
    # the judging that closes the cell, so the whole window is covered. The assembly folds
    # in the busiest count the kind itself observed.
    concurrent_eval_jobs = sample_concurrent_eval_jobs(options, concurrent_eval_jobs)

    # 5. Build the EvalResult and its trace — through the one assembly every completed cell takes,
    # the host-observed ones (:func:`~threetears.evals.run.witnessed.record_witnessed_cell`) included. The ids are this cell's own,
    # minted here, which is the one thing a host recording a cell after the fact supplies instead.
    return assemble_completed_cell(
        scope_id=scope_id,
        eval_run_id=eval_run_id,
        test_case_id=test_case.id,
        model=model,
        candidate_kind=candidate_kind_name,
        k_iteration=k_iteration,
        subject_id=real_subject_id,
        variant=variant,
        judge_model=judge_model,
        output=candidate,
        judged_artifact=judged_artifact,
        rate_table=options.external_rates,
        result_id=str(uuid.uuid7()),
        scored_at=utc_now_iso(),
        judged=judged,
        judge_ms=judge_ms,
        spans=spans,
        concurrent_eval_jobs=concurrent_eval_jobs,
        metered_cell=metered_cell,
        world_events=sink.world_events,
        end_state=sink.end_state,
    )


def hold_to_declaration(kind_name: str, judged_artifact: JudgedArtifact, output: CandidateOutput) -> None:
    """Refuse output that contradicts the kind's own declaration of what a judge reads.

    A judge reads nothing but the evidence the kind renders — the engine has no transcript or
    subject renderer to fall back on — so a judged kind owes it for every output it produced, and an
    unjudged kind rendering some is contradicting its own declaration. An EMPTY output owes none
    whatever the kind: it is a cell that produced nothing, and nothing is judged.

    One check for both callers that record a completed cell: the runner, before its judge phase
    reads the output, and :func:`~threetears.evals.run.witnessed.record_witnessed_cell`, whose host declares the same thing for the
    kind that produced what it observed.

    Args:
        kind_name: The kind, for the refusal's message.
        judged_artifact: What the kind declares a judge reads.
        output: What the kind produced.

    Raises:
        CandidateKindDefect: The output contradicts the declaration.
    """
    has_evidence = output.judge_evidence is not None
    if judged_artifact is JudgedArtifact.UNJUDGED:
        contradicts = has_evidence
    else:
        contradicts = bool(output.output) and not has_evidence
    if contradicts:
        raise CandidateKindDefect(kind_name, declared=judged_artifact, has_evidence=has_evidence)


def refuse_inner_agent_usage(usage: Sequence[RoleUsage]) -> None:
    """Refuse a kind's telemetry that reports its background work's spend a second time.

    Background work reports its spend on its :class:`~threetears.evals.contracts.models.AsyncDelivery`,
    which the engine folds (:func:`_candidate_side_usage`); a kind reporting ``inner_agent`` rows of
    its own would be counted beside it. Its own function so the runner can refuse before its judge
    phase pays for anything, while the fold itself happens once, when the cell is assembled.

    Args:
        usage: The rows the kind reported on its telemetry.

    Raises:
        ValueError: The kind reported ``inner_agent`` rows on its telemetry.
    """
    if any(row.role == "inner_agent" for row in usage):
        raise ValueError(
            "a candidate kind reported inner_agent usage on its telemetry; background work reports its spend on "
            "its AsyncDelivery entry (cost_usd, tokens, llm_calls, external_spend), which the runner folds — "
            "reporting it on the telemetry too would count it twice"
        )


def _unjudged_record() -> _JudgeRecord:
    """The record of a cell nothing judged: every score absent, nothing failed.

    A function rather than a constant so no two cells share its lists.

    Returns:
        The record.
    """
    return _JudgeRecord(
        transcript_score=None,
        outcome_score=None,
        rubric_scores=[],
        judge_reasoning="",
        judge_config_ids={},
        judge_error=None,
        judge_cannot_tell={},
        usage=[],
    )


def assemble_completed_cell(
    *,
    scope_id: str,
    eval_run_id: str,
    test_case_id: str,
    model: str,
    candidate_kind: str,
    k_iteration: int,
    subject_id: str,
    variant: DerivedVariantIdentity,
    judge_model: str | None,
    output: CandidateOutput,
    judged_artifact: JudgedArtifact,
    rate_table: ExternalRateTable | None,
    result_id: str,
    scored_at: str,
    judged: _JudgeRecord | None,
    judge_ms: float | None,
    spans: CellTrace | None,
    concurrent_eval_jobs: int | None,
    metered_cell: MeteredCallTally | None,
    world_events: list[WorldEvent] | None,
    end_state: dict[str, Any] | None,
) -> CellOutcome:
    """Build one completed cell's :class:`EvalResult` and :class:`EvalTrace` from what its candidate produced.

    **The one assembly path a completed cell takes**, whoever ran it: :func:`run_one_result` calls it
    once its judge phase is over, and :func:`~threetears.evals.run.witnessed.record_witnessed_cell` calls it for a cell a host observed
    rather than ran. Everything derived from the output — the candidate side's usage rows with its
    background work folded in, the error taxonomy, the blended cost, the covariates, the latency record,
    the trace document's id — is derived here and nowhere else, so a host-recorded cell and a run's cell
    over the same output are the same record. Pure: it persists nothing, and mints nothing — the ids and
    the timestamp are the caller's.

    Args:
        scope_id: The run's partition.
        eval_run_id: The run.
        test_case_id: The case.
        model: The candidate model.
        candidate_kind: The kind that produced the output.
        k_iteration: The repeat.
        subject_id: The run's subject key.
        variant: The run's resolved contestant identity.
        judge_model: The judge the run records, or ``None``.
        output: What the candidate produced.
        judged_artifact: What the kind declares a judge reads, kept on the trace beside the evidence.
        rate_table: The run's rate table, which prices external calls.
        result_id: The result's id; the trace's id is derived from it.
        scored_at: When the result was recorded.
        judged: What the judge phase left, or ``None`` for a cell nothing judged.
        judge_ms: The judge phase's wall clock, or ``None`` when it did not run.
        spans: The trace sink's record of the candidate's own work, or ``None`` when nothing was collected.
        concurrent_eval_jobs: The busiest count of concurrent eval jobs the caller sampled; the kind's
            own is folded in here.
        metered_cell: The cell's slice of the run's metered-call ledger, or ``None`` when nothing counted.
        world_events: What moved the cell's world, or ``None`` when no world was opened.
        end_state: The world the cell left, or ``None`` when it was never read.

    Returns:
        The cell.
    """
    if judged is None:
        judged = _unjudged_record()
    telemetry = output.telemetry
    trace = output.output
    async_deliveries = output.async_deliveries
    # The candidate side's rows: the kind's own, and what its background work spent.
    candidate_usage = _candidate_side_usage(telemetry.usage, async_deliveries, rate_table=rate_table)
    concurrent_eval_jobs = _busier_conditions_sample(concurrent_eval_jobs, telemetry.concurrent_eval_jobs)
    # The ledger the kind's report amounts to — the same one the runner's sink built on ``adopt``, and
    # the same convention: "candidate details first, then infra" is the order an operator reads
    # ``runner_error`` in, and it is one convention in one place.
    errors = ErrorLedger.of(output)

    # Span-derived latency buckets, read off the trace sink's record by this module, from the
    # record's named fields, rather than splatting whatever mapping a sink chose to return:
    # LatencyMetrics ignores unknown keys with no log, so one host misspelling a bucket would store
    # a measured cell as unmeasured.
    otel_spans: list[dict[str, Any]] = spans.spans if spans is not None else []
    spans_ms: dict[str, float | None] = (
        {"total_ms": spans.total_ms, "llm_ms": spans.llm_ms, "tool_ms": spans.tool_ms} if spans is not None else {}
    )
    # Built here rather than from the harvest alone because not every source is a span: the
    # drain wait falls between spans and the judge phase runs after them, so a cell whose
    # spans never arrived can still have measured both. Any one measured COMPONENT is
    # enough — discarding a real measurement for want of a span would defeat the point of
    # timing the windows nothing else covers.
    #
    # The test is on the values, not on whether a span was harvested, and the difference
    # is a fact rather than a style: a harvest can return spans that land in no bucket at
    # all (a background tool's enqueue, its background task's own root), and summarizing those yields the three keys with
    # every value None. Building on that would persist a carrier whose every field is
    # unmeasured — a record asserting a measurement happened, on which the documented
    # reading of `latency is None` ("this cell timed nothing whatever") would be a
    # one-way claim rather than the contract callers are told to check.
    latency: LatencyMetrics | None = None
    measured = {name: value for name, value in spans_ms.items() if value is not None}
    if measured or telemetry.async_wait_ms is not None or judge_ms is not None:
        latency = LatencyMetrics(**spans_ms, async_wait_ms=telemetry.async_wait_ms, judge_ms=judge_ms)

    # ``usage`` is a list on EVERY exit of a running cell — never None, which means no observation
    # was made at all. It holds a row per (role, model, provider, unit) actually observed. An external
    # row reports call volume, plus the provider's own units and dollars when the run resolved a rate
    # for them; `cost_usd` below stays the authoritative blended spend either way.
    #
    # The candidate's own rows came back on its telemetry, already folded across its roles
    # and its metered third-party calls — that arithmetic belongs where the ledgers are.
    # What is added here is the one role the candidate did not incur: scoring it.
    usage: list[RoleUsage] = [*candidate_usage, *judged.usage]
    # The total is derived from the rows, never reported beside them: one statement of the spend,
    # and an unpriced call in it makes the total unknown rather than quietly smaller.
    cost_roles = blended_cost_roles(rate_table)
    _warn_on_unreported_async_spend(async_deliveries, eval_run_id=eval_run_id)
    result = EvalResult(
        id=result_id,
        scope_id=scope_id,
        eval_run_id=eval_run_id,
        test_case_id=test_case_id,
        model=model,
        # Stamped from the dispatch's single read rather than from the template: a template is
        # store-mastered, so a reader following this cell back to one would get whatever the
        # kind field says TODAY.
        candidate_kind=candidate_kind,
        # The host's own mechanical grade, carried verbatim and interpreted nowhere here: the
        # measure registry is what knows a name's population, range and merit axis, and the
        # analysis bundle reads these through it. ``{}`` from a kind that graded nothing is
        # "measured none", which the bundle treats as the empty set and not as a gap.
        host_measures=output.host_measures,
        k_iteration=k_iteration,
        candidate_instance_id=output.candidate_instance_id,
        subject_id=subject_id,
        goal_state_outcomes=output.mechanical_facts,
        rubric_scores=judged.rubric_scores,
        transcript_score=judged.transcript_score,
        outcome_score=judged.outcome_score,
        judge_config_ids=judged.judge_config_ids,
        judge_reasoning=judged.judge_reasoning,
        judge_model=judge_model,
        cost_usd=cell_cost(usage, async_deliveries=async_deliveries, rate_table=rate_table),
        cost_roles=list(cost_roles),
        usage=usage,
        # The cell-level half of the ceiling's disclosure. Zero and None say different
        # things — "the ceiling was in force and never bound here" versus "nothing
        # counted" — so the None is carried through rather than coerced.
        metered_calls_refused=_cell_metered_refusals(metered_cell, telemetry.metered_calls_refused),
        latency=latency,
        covariates=derive_covariates(
            usage=usage,
            concurrent_eval_jobs=concurrent_eval_jobs,
            dropped_tool_calls=count_dropped_tool_calls(trace),
            refused_tool_attaches=count_refused_tool_attaches(trace),
            truncated_rounds=count_truncated_rounds(trace),
            turns_ended_by_budget=telemetry.turns_ended_by_budget,
        ),
        phase_timings=telemetry.phase_timings,
        async_deliveries=async_deliveries,
        # The kind's own report, stored verbatim: nothing below the dispatch reads inside it.
        kind_payload=output.kind_payload,
        world_events=world_events,
        runner_error=errors.combined,
        candidate_error=errors.candidate_error,
        infra_error=errors.infra_error,
        judge_error=judged.judge_error,
        judge_cannot_tell=judged.judge_cannot_tell,
        variant_key=variant.variant_key,
        identity_version=variant.identity_version,
        # Reached the end of the cell. Stated on the success path too, and not only on the
        # exits that lost something, because the point of the field is that "nothing was
        # cancelled" is an answer rather than the absence of one. It says nothing about whether
        # the cell ERRORED: a candidate whose LLM
        # failed still completed, and the error taxonomy above is what carries that.
        termination="completed",
        # Why the conversation inside it stopped, as the kind reported it — a separate fact
        # from termination, and None from a kind that does not converse.
        stop_cause=output.stop_cause,
        scored_at=scored_at,
    )
    judge_evidence = output.judge_evidence
    return CellOutcome(
        result,
        EvalTrace(
            id=eval_trace_doc_id(result.id),
            scope_id=scope_id,
            result_id=result.id,
            eval_run_id=eval_run_id,
            trace=trace,
            otel_trace=otel_spans,
            # What the judge reads, kept with the declaration that picked its axes, so a re-judge
            # sends the same evidence rather than re-rendering it through a kind that may have
            # changed since. ``None`` for both on an unjudged kind's cell and on an empty output.
            judge_evidence=judge_evidence,
            judged_artifact=judged_artifact if judge_evidence is not None else None,
            # What the candidate did, as the kind recorded it — the input a re-check re-grades from.
            call_ledger=output.call_ledger,
            # What the world held when the cell ended — the other input a re-check re-grades from.
            end_state=end_state,
        ),
    )


def _empty_trace(*, result_id: str, scope_id: str, eval_run_id: str) -> EvalTrace:
    """The trace document for a cell that recorded no turns and harvested no spans.

    Built rather than returning ``None`` so every exit of ``run_one_result`` answers the
    same shape; :meth:`EvalStorage.save_eval_result` is the one place that decides an
    empty payload is not worth a row.
    """
    return EvalTrace(
        id=eval_trace_doc_id(result_id),
        scope_id=scope_id,
        result_id=result_id,
        eval_run_id=eval_run_id,
    )


def _cut_short_cell(
    *,
    sink: _CellSink,
    scope_id: str,
    eval_run_id: str,
    test_case_id: str,
    model: str,
    k_iteration: int,
    subject_id: str,
    variant: DerivedVariantIdentity,
    concurrent_eval_jobs: int | None,
    termination: CellTermination,
    judge_model: str | None,
    rate_table: ExternalRateTable | None,
    unfinished_judging: str | None,
    ungraded_goal_checks: Sequence[GoalStateOutcome] = (),
    metered_calls_refused: int | None = None,
) -> CellOutcome:
    """The two documents of a cell cut off before :func:`run_one_result` returned them.

    Its deadline, its run's cancel and an apparatus fault all end a cell this way, and all three
    build its record from the one thing that survives the cut: the cell's sink. The candidate side
    and its spend are :func:`_degraded_capture_fields`'s; the errors are whatever the caller put on
    ``sink.errors``, its own charge included; and a judge phase the cut struck keeps everything it
    had — the evidence on the trace, so a re-judge can send it again, the scores that came back,
    and each dim that did not finish named on ``judge_error`` with ``unfinished_judging`` as its
    cause. That is the state a re-judge repairs: the candidate's finished, paid-for work is not
    thrown away for want of a judge's reply.

    Args:
        sink: The cell's sink, its ``errors`` already holding the cut's own charge where it has one.
        scope_id: The run's scope.
        eval_run_id: The run.
        test_case_id: The cell's case.
        model: The cell's candidate model.
        k_iteration: Which repetition.
        subject_id: The run's subject key.
        variant: The cell's resolved contestant identity.
        concurrent_eval_jobs: The cell's measurement-conditions sample.
        termination: How the cell ended.
        judge_model: The judge the run pins, or ``None``.
        rate_table: The run's rate card, or ``None``.
        unfinished_judging: Why a judge dim that did not return did not finish — the cut's own
            message — or ``None`` for a cut that cannot strike the judge phase.
        ungraded_goal_checks: See :func:`_degraded_capture_fields`.
        metered_calls_refused: The cell's refused metered calls, where the caller still holds the
            dispatch's baseline; ``None`` where it does not.

    Returns:
        The result and its trace.
    """
    judged = (
        _judge_record(sink.judging, sink.judged, unfinished=unfinished_judging) if sink.judging is not None else None
    )
    errors = sink.errors
    side = sink.side()
    judge_fields: dict[str, Any] = (
        {
            "transcript_score": judged.transcript_score,
            "outcome_score": judged.outcome_score,
            "rubric_scores": judged.rubric_scores,
            "judge_reasoning": judged.judge_reasoning,
            "judge_config_ids": judged.judge_config_ids,
            "judge_error": judged.judge_error,
            "judge_cannot_tell": judged.judge_cannot_tell,
        }
        if judged is not None
        else {}
    )
    result = EvalResult(
        scope_id=scope_id,
        eval_run_id=eval_run_id,
        test_case_id=test_case_id,
        model=model,
        k_iteration=k_iteration,
        subject_id=subject_id,
        candidate_kind=sink.kind_name,
        runner_error=errors.combined,
        candidate_error=errors.candidate_error,
        infra_error=errors.infra_error,
        metered_calls_refused=metered_calls_refused,
        **judge_fields,
        **_degraded_capture_fields(
            sink=sink,
            variant=variant,
            concurrent_eval_jobs=concurrent_eval_jobs,
            termination=termination,
            judge_model=judge_model,
            rate_table=rate_table,
            ungraded_goal_checks=ungraded_goal_checks,
        ),
    )
    # The output the cell delivered before it was cut, for whoever reads why; empty when it was
    # cut before any, which storage declines to write. The judge's evidence rides with it once the
    # judge phase began.
    trace = EvalTrace(
        id=eval_trace_doc_id(result.id),
        scope_id=scope_id,
        result_id=result.id,
        eval_run_id=eval_run_id,
        trace=list(side.output) if side is not None else [],
        judge_evidence=sink.judging.evidence if sink.judging is not None else None,
        judged_artifact=sink.judging.judged_artifact if sink.judging is not None else None,
        # Read only once ``invoke`` returned, so a cell cut off before that stores none; one cut off
        # while judged keeps what was read.
        end_state=sink.end_state,
    )
    return CellOutcome(result, trace)


def metered_cell_tally(
    ledger: MeteredCallLedger | None,
    baseline: MeteredCallTally | None,
) -> MeteredCallTally | None:
    """One cell's slice of the run's metered-call tally, or ``None`` when nothing counted.

    Cells inside a run execute serially, so the difference between a tally taken before
    the cell and one taken after IS the cell's contribution — no per-cell ledger, and
    therefore no way for a per-run ceiling to become a per-cell one N times larger.

    ``None`` when the run carries no ledger (every caller that built its own
    ``RunnerOptions``, and every non-eval path), which the caller must not read as
    zero: it means nobody counted, where zero would mean the ceiling was in force and
    this cell never touched it.

    The runner reads it for the cell's refusal count, and a kind that settles its own usage
    rows reads it for :func:`fold_metered_cell`; both ask here so the two cannot disagree
    about what one cell metered.

    Args:
        ledger: The run's ledger, or ``None``.
        baseline: The tally taken at the top of this cell, or ``None``.

    Returns:
        The cell's slice, or ``None``.
    """
    if ledger is None or baseline is None:
        return None
    return ledger.tally().delta_from(baseline)


def fold_metered_cell(external_usage: RoleUsageLedger, cell: MeteredCallTally | None) -> None:
    """Fold one cell's action-seam tally into the external role's rows.

    The tally's per-provider spends go in one at a time rather than summed: two providers'
    weighted units are not one quantity, so the row keying is what keeps them apart.

    **A tool that named no provider still made its calls**, and they land in a row of their
    own rather than disappearing. The tally counts every metered call in ``calls`` but only
    an attributed one produces a spend entry, so folding the spends alone would drop those
    calls out of ``usage`` silently — a cell that made ten unattributable metered calls
    would read as having made none. A tool should declare its provider; this is what keeps the
    volume honest when one does not, and it is a function rather than an inline block so a
    test can drive the real fold instead of asserting a copy of it.

    The runner's fold rather than the metering module's: metering counts calls and knows
    nothing about role rows, and folding a tally into one is the eval accumulation layer's
    work. Engine API: a kind that builds its own usage rows calls it with its external ledger
    when it settles.

    Args:
        external_usage: The cell's external-role ledger.
        cell: This cell's slice of the run tally, or ``None`` when nothing counted.
    """
    if cell is None or not cell.calls:
        return
    for spend in cell.spends:
        external_usage.add_external_unpriced(spend)
    attributed = sum(spend.calls for spend in cell.spends)
    if attributed < cell.calls:
        external_usage.add_external_unpriced(ExternalSpend(provider=None, calls=cell.calls - attributed))


def _cell_metered_refusals(cell: MeteredCallTally | None, reported: int | None) -> int | None:
    """This cell's refused metered calls: the runner's own count, plus anything a kind added.

    The runner's slice is the authority, because the ledger being refused against is the
    RUN's and the dispatch site is the only place that holds it. A kind reports here only
    what that ledger could not see — a provider it meters itself — so the two are added
    rather than one winning: dropping either would publish a refusal count lower than what
    happened.

    ``None`` from both means nobody counted, which is not zero. Zero means the ceiling was
    in force and never bound this cell, and ``EvalResult.metered_calls_refused`` says the
    two are never collapsed — so a present count on either side makes the answer a number,
    and only a total absence stays absent.

    Args:
        cell: This cell's slice of the run's ledger, or ``None`` when the run carries none.
        reported: What the kind counted itself, or ``None`` when it counted nothing.

    Returns:
        The cell's refusals, or ``None`` when neither side counted.
    """
    if cell is None and reported is None:
        return None
    return (cell.refused if cell is not None else 0) + (reported or 0)


def _degraded_capture_fields(
    *,
    sink: _CellSink,
    variant: DerivedVariantIdentity,
    concurrent_eval_jobs: int | None,
    termination: CellTermination,
    judge_model: str | None,
    rate_table: ExternalRateTable | None,
    ungraded_goal_checks: Sequence[GoalStateOutcome] = (),
) -> dict[str, Any]:
    """Capture fields for a cell that did not reach the end of :func:`run_one_result`.

    A subject-factory failure, an apparatus fault seeding the world, an unmet
    precondition, an apparatus fault out of the kind, a cell timeout and a cancel all build their
    result here, and all must say what they captured the same way — because every tri-state here
    reads its empty arm and its absent arm differently:

    - ``usage`` is always a list: the rows the cell's candidate side reported plus the judge
      calls that returned, which is ``[]`` when capture ran and attributed no roles. ``None``
      means no per-role observation exists at all, so handing it to a cell that DID run would
      report the apparatus as never having looked.
    - ``covariates`` is a real (if sparse) mapping for the same reason, and it is worth
      having: a cell that failed or timed out under concurrent load is exactly the
      observation an operator wants to stratify.
    - ``phase_timings`` is what the candidate side reported — ``{}`` when it reported none, as
      distinct from nobody looking.
    - the variant is stamped even here: the cell still observed a specific contestant, and
      dropping the key would bias any grouping toward the cells that succeeded.
    - ``judge_model`` is stamped for the same reason: a blank on this field means the run
      pinned no judge, and every surface reads it that way. This is what the cell WOULD have
      been judged by; that no judging happened is carried by the scores being absent, not by
      pretending nobody pinned a judge.

    Collected in one place so the contract has a single definition rather than one copy per
    degraded exit — a new capture field added to only some of them is the silent failure.

    **Every exit reads the cell's sink, so the dollars and the rows are derived, not chosen**.
    The candidate side is :meth:`_CellSink.side`: what ``invoke`` returned, else the
    reading the kind registered, else nothing. The pre-turn exits have nothing, so their total
    is a real zero over the run's roles. A cell timeout holds what its kind had reported when
    the deadline struck — for the conversational kind the turns and deliveries recorded and the
    simulator's replies, for the reporter the generator calls that returned — plus the judge
    calls that returned, because :func:`execute_run` owns the sink and reads it after the
    cancel, rather than taking the record from a frame the cancel destroyed. Any kind
    alike: the sink is on the seam. ``cost_roles`` therefore names the run's roles on every
    exit, timeout included: they are the roles the total sums. What a timed-out total cannot
    hold is the call that was in flight when the deadline struck, whose spend no response ever
    reported — for a single-call kind that is the whole of it; ``termination`` is what tells a
    reader the total stops there (see
    :func:`~threetears.evals.contracts.usage_capture.resolve_result_usage`).

    **Once the kind's ``invoke`` has returned, its whole output is the candidate side's record**
    (:attr:`_CellSink.returned`): a deadline in judging keeps the goal-state facts, host
    measures, stop cause, instance id, kind payload, phase timings, async deliveries and conditions sample
    the completed record would have stored, read from the same object. A reading taken
    mid-``invoke`` is read the same way, except for the goal state: graded only at the end of
    ``invoke``, so before that the record carries ``ungraded_goal_checks``. ``latency`` is not
    rebuilt — the span record and the judge's clock lived in the cancelled frame.

    **``metered_calls_refused`` is deliberately NOT here.** Its baseline is the dispatch
    site's, taken around ``prepare`` and ``invoke``, and none of these exits reaches the
    line that reads it: a factory or seed failure in the kind's preparation precedes any turn, so
    there was no call for a ceiling to refuse, and a timed-out cell's count is the gap this
    leaves. **A kind's own ``prepare`` refusing** (:exc:`CandidatePreparationFailed`) is
    the one where "precedes any turn" is one kind's shape standing in for every kind's: a
    kind that meters inside ``prepare`` and then refuses records ``None`` while its calls are
    in the run's ledger. No kind in the tree does, so this is stated rather than fixed. The
    run-level record (``EvalRun.metered_calls_refused``) is unaffected either way — the
    ledger belongs to the run, not to the cell.

    ``termination`` is the branch the caller took, recorded because the record cannot
    recover it. It was once a parallel ``spend_measured`` boolean from which the cost arm
    was derived; with every exit summing what its sink holds, no arm is derived from it any more.

    ``rate_table`` is the run's rate card, and it is required rather than defaulted: which
    roles the run's totals sum is a property of the run, so a degraded exit that omitted it
    would quietly file a priced run's dead cell under the unpriced composition.

    Args:
        sink: The cell's sink — holding nothing on a pre-turn exit.
        variant: The cell's resolved contestant identity.
        concurrent_eval_jobs: The cell's measurement-conditions sample.
        termination: How the cell ended.
        judge_model: The judge the run pins, or ``None``.
        rate_table: The run's rate card, or ``None`` when it priced nothing.
        ungraded_goal_checks: What the goal state reads as when the cell ended before its kind's
            ``invoke`` returned — the template's checks, failed and unevaluated, for a deadline
            charged to the candidate (:func:`_unevaluated_goal_checks`); nothing on every other exit.
            Ignored once ``invoke`` has returned, since the kind's own checks are then the record.

    Returns:
        The keyword arguments every degraded ``EvalResult`` spreads.
    """
    side = sink.side()
    judged = fold_judge_outcomes(sink.judged)
    if sink.cassettes is not None:
        # The cell ended here, wherever it was — inside ``prepare`` included, which the close after
        # ``invoke`` never reaches. Idempotent, so a cell already closed is unchanged.
        sink.cassettes.close()
    if side is None:
        # The kind reported nothing: a pre-turn exit, or a kind cut off inside the one call it
        # makes. Every tri-state reads its empty arm — see the list above.
        usage: list[RoleUsage] = [*judged.usage]
        async_deliveries: list[AsyncDelivery] | None = []
        phase_timings: dict[str, float] = {}
        trace: list[dict[str, Any]] = []
    else:
        if sink.cassettes is not None:
            sink.cassettes.hold(side.async_deliveries, complete=False)
        usage = [
            *_candidate_side_usage(side.telemetry.usage, side.async_deliveries, rate_table=rate_table),
            *judged.usage,
        ]
        async_deliveries = side.async_deliveries
        phase_timings = dict(side.telemetry.phase_timings)
        trace = side.output
        concurrent_eval_jobs = _busier_conditions_sample(concurrent_eval_jobs, side.telemetry.concurrent_eval_jobs)
    returned = sink.returned
    cost_roles = blended_cost_roles(rate_table)
    return {
        "usage": usage,
        "cost_usd": cell_cost(usage, async_deliveries=async_deliveries, rate_table=rate_table),
        # ``{}`` when the kind reported none: the host's grader never ran.
        "host_measures": dict(side.host_measures) if side is not None else {},
        "goal_state_outcomes": list(returned.mechanical_facts) if returned is not None else list(ungraded_goal_checks),
        "stop_cause": side.stop_cause if side is not None else None,
        "candidate_instance_id": side.candidate_instance_id if side is not None else None,
        "kind_payload": side.kind_payload if side is not None else None,
        # What fired before the cell ended, from the session the engine owns — so a cut-off cell
        # keeps it, as it keeps its spend.
        "world_events": sink.world_events,
        "cost_roles": list(cost_roles),
        "covariates": derive_covariates(
            usage=usage,
            concurrent_eval_jobs=concurrent_eval_jobs,
            dropped_tool_calls=count_dropped_tool_calls(trace),
            refused_tool_attaches=count_refused_tool_attaches(trace),
            truncated_rounds=count_truncated_rounds(trace),
            turns_ended_by_budget=side.telemetry.turns_ended_by_budget if side is not None else None,
        ),
        "phase_timings": phase_timings,
        "async_deliveries": async_deliveries,
        "variant_key": variant.variant_key,
        "identity_version": variant.identity_version,
        "judge_model": judge_model,
        "termination": termination,
    }


def _candidate_side_usage(
    usage: Sequence[RoleUsage],
    async_deliveries: Sequence[AsyncDelivery] | None,
    *,
    rate_table: ExternalRateTable | None,
) -> list[RoleUsage]:
    """The candidate side's rows: the kind's own, plus what its background work spent.

    Background work reports its spend on its :class:`~threetears.evals.contracts.models.AsyncDelivery`,
    so the runner folds it (:func:`~threetears.evals.contracts.usage_capture.async_delivery_usage`)
    rather than each kind — work still in flight when the cell ended included, and a substituted
    entry, which can report none, contributing nothing. That is the one path such spend takes: a kind
    reporting ``inner_agent`` rows of its own would be counted beside it, so it is refused. The
    dollars are not summed here: the cell's cost is derived from these rows once, by
    :func:`~threetears.evals.contracts.usage_capture.cell_cost`.

    Args:
        usage: The rows the kind reported on its telemetry.
        async_deliveries: The kind's background work.
        rate_table: The run's rate table, which prices the work's external calls.

    Returns:
        The rows.

    Raises:
        ValueError: The kind reported ``inner_agent`` rows on its telemetry.
    """
    refuse_inner_agent_usage(usage)
    return [*usage, *async_delivery_usage(async_deliveries, rate_table=rate_table)]


def _warn_on_unreported_async_spend(async_deliveries: list[AsyncDelivery] | None, *, eval_run_id: str) -> None:
    """Log when background work that delivered from a model reported nothing it spent.

    A live entry — not substituted — that delivered and names the model its work ran on necessarily
    spent on that model, so one reporting no tokens, no calls and no dollars is a kind that forgot to
    carry its spend, not work that spent nothing; the cell's production-replicating cost is then short
    by it. Work that failed or was still in flight may honestly have spent nothing yet, so it is not
    held to this — a detector that cries wolf trains its reader to ignore it. A log rather
    than a raise: ``None`` is also an honest "the tool could not say", and a cell that produced real
    scores must not be failed over its telemetry.
    """
    if async_deliveries is None:
        return
    # ``model is not None`` already excludes a substituted entry: AsyncDelivery refuses one that names a model.
    silent = [
        entry
        for entry in async_deliveries
        if entry.status == "delivered" and entry.model is not None and not entry.reported_spend
    ]
    if not silent:
        return
    log.warning(
        "Eval run %s: %d live async delivery(ies) ran on a model and report no spend at all — the kind is likely "
        "not carrying its background work's spend on AsyncDelivery, and this cell's cost is short by it",
        eval_run_id,
        len(silent),
    )


def sample_concurrent_eval_jobs(options: RunnerOptions, previous: int | None = None) -> int | None:
    """Sample how many eval jobs are executing, keeping the busiest observation so far (R4).

    ``None`` when no probe is wired — recorded as no ``execution_mode`` at all rather than
    as "serial", because nothing looked. Sampling repeatedly through the cell and keeping
    the max is what makes the covariate honest across a cell that runs for minutes: a
    second job starting halfway through contaminated this measurement just as surely as
    one that was already running, and boundary samples alone would report it as clean.

    The reading is still a floor, not a guarantee: a job that starts and ends entirely
    between two samples is invisible. Sampling every turn narrows that window to a turn;
    closing it entirely would need the job manager to push, not the cell to poll.

    The probe is caller-supplied, so it is a boundary: a raising one degrades this covariate
    to unmeasured rather than taking the cell down. That matters most at the cell-timeout
    call site, whose entire purpose is preserving a degraded observation — losing the
    result to a telemetry probe would destroy the very evidence the timeout path exists to
    keep. The failure is logged, never swallowed silently.
    """
    if options.concurrent_eval_jobs_probe is None:
        return previous
    try:
        observed = options.concurrent_eval_jobs_probe()
    except Exception:  # prawduct:ok-broad-except — caller-supplied probe; a telemetry read must not fail a cell
        log.warning("Concurrent-eval-job probe raised; recording execution_mode as unmeasured", exc_info=True)
        return previous
    return observed if previous is None else max(previous, observed)


def assert_preconditions(
    template: EvalTemplate, test_case: EvalTestCase, seeded: Mapping[str, Any], *, world: WorldRegistry | None
) -> list[PreconditionOutcome]:
    """Assert at t=0 that the world this template presumes actually holds.

    A precondition is runnable, so the moment the world is seeded is the moment to ask. What comes
    back is empty when every presumption held — which is the common case and the one that must
    cost nothing on the record — and the WHOLE set when any did not, because an exclusion is read
    by asking what was presumed and which part of it the world failed.

    **Evaluated against the same state the goal judge reads**, and against the world at t=0 rather
    than at the end: this is the question of whether the subject was ever placed in the state the
    probe was written for, and asking it after the run would ask whether the subject left it.

    **An expression that cannot be evaluated counts as not holding.** The alternative is to treat
    an unevaluable presumption as satisfied, which scores a cell whose world nobody checked — the
    missing-value semantics this whole contract exists to end. Parse errors cannot reach here
    (``Precondition`` refuses them where the template is written) and a path naming no declared
    dimension cannot either (``resolve_preconditions`` refuses that where the template is used),
    so what is left is a genuine evaluation fault and it belongs on the excluded record.

    **Engine API, for every kind that seeds a world.**
    ``EvalTemplate.preconditions`` is a field of the engine's own template, evaluated by the
    engine's DSL against the engine's world state, so any kind that seeds a world has the same
    presumptions to check and should check them the same way. Moving this beside one kind would
    leave every other world-bearing kind to re-derive what "a presumption did not hold" means.

    Args:
        template: The template whose presumptions these are.
        test_case: The case, for its variation parameters — an expression may read
            ``variation.tone`` beside the world.
        seeded: The seeded world, immediately after seeding and before any turn, keyed by declared
            dimension name — read back through the dimensions' ``read`` handles, or named from the
            seed by :meth:`~threetears.evals.contracts.host.world.WorldRegistry.named`. Read with an
            empty call ledger: at t=0 the candidate has made no call.
        world: The host's world registry (``profile.world``), so a path reads the dimension the
            authoring gate resolved it to.

    Returns:
        Every outcome when at least one did not hold; empty when they all did.
    """
    outcomes: list[PreconditionOutcome] = []
    for precondition in template.preconditions:
        try:
            held, detail = evaluate_with_detail(
                precondition.expression,
                end_state=seeded,
                ledger=CallLedger(),
                world=world,
                variation=dict(test_case.variation_params),
            )
        except (
            Exception
        ) as e:  # prawduct:ok-broad-except — DSL evaluation boundary; an unevaluable presumption excludes
            held, detail = False, f"<error: {e}>"
        outcomes.append(
            PreconditionOutcome(
                expression=precondition.expression, presumes=precondition.presumes, held=held, detail=detail
            )
        )
    return outcomes if any(not outcome.held for outcome in outcomes) else []


def precondition_failure_text(outcomes: Sequence[PreconditionOutcome]) -> str:
    """One line naming what was presumed and what the world held instead.

    Engine API beside :func:`assert_preconditions`, for the same reason: the outcome is the
    engine's ``PreconditionOutcome``, and every world-bearing kind that excludes a cell on a
    failed presumption reports it in this one form.

    Args:
        outcomes: The t=0 outcomes, including the ones that held.

    Returns:
        The text for ``runner_error`` / ``infra_error``. Names only the presumptions that FAILED —
        the satisfied ones are on the record for a reader who wants them, and repeating them in a
        one-line summary would bury the ones that matter.
    """
    return "; ".join(
        f"{outcome.presumes!r} did not hold ({outcome.expression} — {outcome.detail})"
        for outcome in outcomes
        if not outcome.held
    )


class GoalCheckUnevaluable(RuntimeError):
    """A goal check raised while being evaluated — a fault of the rig, not a verdict on the candidate.

    The check is a claim about the world, and a check that cannot be evaluated has made no claim.
    Scoring it as failed would put a harness or DSL fault on the candidate's record, so a kind that
    catches it raises :class:`~threetears.evals.contracts.host.ApparatusError` from ``invoke``, which
    the runner records as an apparatus fault that excludes the cell.

    Engine API beside :func:`evaluate_goal_state`, which raises it: a kind catches it to tell a
    rig fault from a verdict, whichever kind evaluates the template's goal checks.
    """

    def __init__(self, expression: str, error: BaseException) -> None:
        """Name the expression that could not be evaluated and why.

        Args:
            expression: The goal check's text.
            error: What its evaluation raised.
        """
        super().__init__(f"goal check {expression!r} could not be evaluated: {type(error).__name__}: {error}")
        self.expression = expression


def evaluate_goal_state(
    *,
    template: EvalTemplate,
    test_case: EvalTestCase,
    ledger: CallLedger,
    end_state: Mapping[str, Any],
    fired: Collection[str] | None,
    world: WorldRegistry | None,
) -> list[GoalStateOutcome]:
    """Run every expression in ``template.goal_state_checks`` and capture outcomes.

    **Engine API, for every kind that leaves a world behind.**
    ``goal_state_checks`` is a field of the engine's own template, read by the engine's DSL
    against the engine's world state, so any kind that leaves a world behind grades it here rather
    than re-implementing the evaluation and the rig-fault rule beside itself.

    Args:
        template: The template whose goal checks to run.
        test_case: The case, for its variation parameters.
        ledger: The calls the candidate made that succeeded.
        end_state: The world the cell left behind, keyed by declared dimension name.
        fired: The triggered dimensions that fired (:attr:`WorldSession.fired`), or ``None`` when no
            world events were recorded.
        world: The host's world registry (``profile.world``).

    Returns:
        One outcome per goal check, in the template's order.

    Raises:
        GoalCheckUnevaluable: An expression raised while being evaluated.
    """
    return grade_goal_checks(
        template.goal_state_checks,
        ledger=ledger,
        end_state=end_state,
        fired=fired,
        variation=test_case.variation_params,
        world=world,
    )


def grade_goal_checks(
    expressions: Sequence[str],
    *,
    ledger: CallLedger,
    end_state: Mapping[str, Any],
    fired: Collection[str] | None,
    variation: Mapping[str, Any],
    world: WorldRegistry | None,
) -> list[GoalStateOutcome]:
    """Grade goal checks against a call ledger, an end state and what fired: the one evaluation every caller uses.

    **Callable by every kind.** A kind fills a :class:`~threetears.evals.contracts.call_ledger.CallLedger`
    as its candidate acts, reads its end state, and grades here; :func:`evaluate_goal_state` is the
    same call over a template's checks. The check-controls gate
    (:mod:`threetears.evals.run.check_controls`) grades a template's control end states through here,
    so a check proven to discriminate at authoring is proven under the rule a run will grade it by,
    and the re-check (:mod:`threetears.evals.run.recheck`) re-grades a stored result through here,
    so a ledger rule's change reaches results stored before it under that same rule.

    Args:
        expressions: The goal checks, in order.
        ledger: The calls the candidate made that succeeded — a cell's, or a control's stated calls.
        end_state: The world to read, keyed by declared dimension name — a cell's end state, or a
            control end state.
        fired: The triggered dimensions that fired, read by ``fired()`` — a cell's
            (:attr:`~threetears.evals.contracts.world_session.WorldSession.fired`), or a control's stated
            set. Required rather than defaulted: ``None`` says no world events were recorded, and a check
            reading ``fired()`` then raises rather than scoring "nothing fired" for a cell nobody watched.
        variation: The case parameters a check may read as ``variation.*``.
        world: The host's world registry (``profile.world``), so a path reads the dimension the
            authoring gate resolved it to.

    Returns:
        One outcome per expression, in order.

    Raises:
        GoalCheckUnevaluable: An expression raised while being evaluated.
    """
    outcomes: list[GoalStateOutcome] = []
    for expr in expressions:
        try:
            passed, detail = evaluate_with_detail(
                expr, end_state=end_state, ledger=ledger, world=world, variation=dict(variation), fired=fired
            )
        except Exception as e:  # prawduct:ok-broad-except — DSL evaluation boundary; re-raised as a named rig fault
            log.warning("Goal-state expression failed to evaluate: %r → %s", expr, e)
            raise GoalCheckUnevaluable(expr, e) from e
        outcomes.append(GoalStateOutcome(expression=expr, passed=passed, detail=detail))
    return outcomes


def _unevaluated_goal_checks(template: EvalTemplate, *, waiting_on: str) -> list[GoalStateOutcome]:
    """The template's goal-state checks, as a candidate-charged deadline leaves them: unevaluated, failed.

    A deadline that strikes while the candidate's turn or its background work is pending cancels
    the cell before the goal state is graded, and fails the candidate (``_DEADLINE_CHARGE``). A
    candidate failure counts every goal-state check as failed, and
    :func:`~threetears.evals.contracts.result_condition.counted_goal_verdicts` is where every per-check rate
    applies that — but only to the checks a result carries. Carrying them here is what puts this
    cell in those rates; left empty, an arm whose candidate keeps hitting its deadline would show
    per-check pass rates over only the cells it finished, while pass^k counted it as failing.

    ``passed`` is ``False`` because nothing held: no check was evaluated, and ``detail`` says so.

    Args:
        template: The scenario the cell ran, whose checks are the ones a finished cell grades.
        waiting_on: What the deadline struck while waiting on, as ``_DEADLINE_CHARGE`` names it.

    Returns:
        One failed, unevaluated outcome per ``template.goal_state_checks`` entry, in order.
    """
    detail = f"not evaluated: the cell's deadline struck while waiting on {waiting_on}, which fails the candidate"
    return [GoalStateOutcome(expression=expr, passed=False, detail=detail) for expr in template.goal_state_checks]


def build_judge_context(
    *,
    template: EvalTemplate,
    test_case: EvalTestCase,
    goal_outcomes: list[GoalStateOutcome],
    judged_artifact: JudgedArtifact,
    judge_evidence: JudgeEvidence,
) -> JudgeContext:
    """The evidence a result's judge reads, built from the cell's own records.

    One construction shared by the cell's judge phase and a later re-judge, so a re-score
    is handed exactly the inputs the phase was: the template's intent, the case's
    variation, the goal-state outcomes, and the kind's declaration and rendered evidence —
    which the re-judge reads back off the cell's stored trace.

    Args:
        template: The run's template, which supplies the intent.
        test_case: The case the cell ran.
        goal_outcomes: The cell's goal-state outcomes.
        judged_artifact: The kind's declaration, which picks the axes.
        judge_evidence: What the kind rendered for its judge.

    Returns:
        The context every judge call for the result shares.
    """
    return JudgeContext(
        case_id=test_case.id,
        intent=template.intent,
        variation=dict(test_case.variation_params),
        goal_outcomes=goal_outcomes,
        judged_artifact=judged_artifact,
        judge_evidence=judge_evidence,
    )


async def judge_dims(
    *,
    template: EvalTemplate,
    judge_service: JudgeService,
    context: JudgeContext,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
    only: frozenset[str] | None = None,
    settled: list[tuple[str, JudgeOutcome]] | None = None,
) -> list[tuple[str, JudgeOutcome]]:
    """Make the judge calls for one result and return each outcome, in dimension order.

    The one place that decides which calls a result's judging consists of, shared by the
    cell's judge phase (:func:`run_one_result`) and by a later re-judge of the dims whose
    judging failed, so a re-score asks exactly the question the phase asked.

    **The calls run concurrently, bounded by ``concurrency``, and nothing in the result
    depends on which finished first.** Each call is one prompt scoring one dimension and
    no call reads another's output. The judge client builds each request locally, which
    is what makes gathering on the one client the service caches per ``(model,
    temperature)`` sound; the port states the requirement
    (:class:`~threetears.evals.contracts.provider.CompletionClient`). ``concurrency=1`` reproduces a
    serial phase exactly.

    The order is DIMENSION order — transcript axis, outcome axis, then ``template.rubric``
    in declaration order — never completion order: ``gather`` returns positionally.

    The transcript and outcome axes are conversation-shaped: a document candidate
    (``context.judged_artifact`` is ``DOCUMENT``) has no turns to read for the first and no goal
    to have reached for the second, so they are not called for one and are absent from the
    return.

    A call that RAISES (a client factory refusing, a prompt builder failing) is caught per
    dimension and returned as an outcome carrying the exception's type as its error, so one
    dimension's crash does not discard the scores its siblings paid for.

    Args:
        template: The run's template, whose ``rubric`` names the per-dim calls.
        judge_service: The judge; one cached client per ``(model, temperature)``.
        context: The finished transcript and its evidence, shared by every call.
        concurrency: Calls in flight at once.
        only: Restrict the calls to these dim ids; ``None`` makes every call.
        settled: A caller-owned list each call's outcome is appended to as it returns, in
            completion order — so a caller whose deadline cancels this phase still holds the
            calls already paid for, which the return value cannot give it.

    Returns:
        ``(dim_id, outcome)`` for every call made, in dimension order.
    """
    gate = asyncio.Semaphore(concurrency)

    async def _judge_one(dim_id: str, call: Callable[[], Awaitable[JudgeOutcome]]) -> JudgeOutcome:
        async with gate:
            try:
                outcome = await call()
            except (
                Exception
            ) as e:  # prawduct:ok-broad-except — judge boundary; one dim's crash must not discard its siblings
                log.exception("Judge dim=%s raised for case=%s", dim_id, context.case_id)
                outcome = JudgeOutcome(score=None, error=f"{type(e).__name__}: {e}")
        if settled is not None:
            settled.append((dim_id, outcome))
        return outcome

    conversational = context.judged_artifact is JudgedArtifact.TRANSCRIPT
    calls: list[tuple[str, Callable[[], Awaitable[JudgeOutcome]]]] = [
        *(
            [
                (TRANSCRIPT_DIM_ID, partial(judge_service.score_transcript, context)),
                (OUTCOME_DIM_ID, partial(judge_service.score_outcome, context)),
            ]
            if conversational
            else []
        ),
        *[(dim.name, partial(judge_service.score_dimension, dim, context)) for dim in template.rubric],
    ]
    if only is not None:
        calls = [(dim_id, call) for dim_id, call in calls if dim_id in only]
    # ``return_exceptions`` stays False on purpose: every Exception is already caught per
    # call above, so anything that reaches gather is a BaseException — a cell timeout's
    # cancellation — and must propagate, not be recorded as a score.
    outcomes = await asyncio.gather(*(_judge_one(dim_id, call) for dim_id, call in calls))
    return [(dim_id, oc) for (dim_id, _), oc in zip(calls, outcomes, strict=True)]


@dataclass
class RunCallbacks:
    """Optional progress / persistence hooks for the run loop."""

    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None
    on_result: Callable[[EvalResult], Awaitable[None]] | None = None


def cell_execution_order(
    run: EvalRun,
    test_cases: list[EvalTestCase],
    *,
    rng: random.Random | None = None,
) -> list[tuple[int, EvalTestCase]]:
    """The order a run's cells are executed in — the full matrix, shuffled.

    Cells run one after another over hours, and the box they run on does not hold
    still: another job starts, a provider slows down, a leak degrades the process.
    Executed in nested ``k -> test case`` order, every such drift lands on the
    matrix as *structure* — the last case measured always sits later in the
    degradation than the first, so a case-level comparison silently carries the clock.
    Shuffling turns that same drift into noise spread across the coordinates, which
    is the difference between a confound and a wider error bar. It cannot remove
    the drift, and nothing here claims to: what it removes is the alignment between
    drift and the thing being compared.

    **The order is derived from the run's own id, so it is recorded by
    construction** — a run document is all anyone needs to reproduce which cell ran
    when, with no extra stored field to keep in sync. Note precisely what that
    reproducibility is: *this run document* always yields *this* order. Re-launching
    the same matrix mints a new run with a new id and therefore a new order, which
    is the point — two runs of one configuration should not share a position
    effect.

    **The seed is deliberately NOT part of the measurement-context key.** The
    context key names the conditions two runs must share to pool as repetitions,
    and this shuffle exists precisely so that cell position stops being one of
    them. Hashing a per-run-unique value would give every run a unique key and make
    nothing comparable with anything — inverting the key's purpose to record a
    quantity the shuffle was added to render irrelevant.

    Args:
        run: The run whose matrix is being ordered; supplies ``k_runs`` and the id the
            order is derived from.
        test_cases: The resolved cases, one per matrix column.
        rng: Generator to shuffle with. ``None`` derives one from ``run.id``.

    Returns:
        Every ``(k_iteration, test_case)`` cell exactly once, in the order the run will
        execute them. Every cell is at the run's one candidate model.
    """
    cells = [(k, tc) for k in range(1, run.k_runs + 1) for tc in test_cases]
    # ``random.Random(str)`` seeds off a hash of the string that does not depend on
    # PYTHONHASHSEED, so the same run id gives the same order in every process.
    (rng or random.Random(run.id)).shuffle(cells)
    return cells


async def execute_run(
    host: EvalHost,
    *,
    run: EvalRun,
    template: EvalTemplate,
    test_cases: list[EvalTestCase],
    judge_service: JudgeService | None,
    options: RunnerOptions,
    callbacks: RunCallbacks | None = None,
    budget_gate: Callable[[], CapBreach | None] | None = None,
    on_cost: Callable[[float | None], None] | None = None,
    cell_sink: list[CellSummary] | None = None,
) -> list[CellSummary]:
    """Execute an entire :class:`EvalRun` end-to-end.

    Produces one :class:`EvalResult` per ``(test_case × model × k_iteration)``.
    Persists each via ``host.storage.save_eval_result``; the run document is
    *not* mutated here — the job manager owns status transitions. Each cell runs
    under ``host.cell_timeout``, its spans go to ``host.trace_sink``, and its
    write runs on ``host.blocking_executor``.

    Cells are executed in the shuffled order :func:`cell_execution_order` derives
    from the run's id, not in nested ``k × model × test case`` order. They remain
    strictly serial, and the produced set is the identical matrix; only the
    sequence moves, so that drift over the run's duration spreads across the
    coordinates instead of aligning with one of them.

    Returns one :class:`CellSummary` per cell, **not** the results themselves.
    Storage is where a result lives; a caller wanting a transcript, scores or
    spans loads them back by ``result_id``, having checked ``persisted``.
    Accumulating the full records here as well made a run's peak memory scale
    with its total trace volume, which is unbounded in the matrix size — a
    large sweep could exhaust the container before it finished. Callers that
    need each result *as it lands* have ``callbacks.on_result``, which sees the
    whole record and retains nothing.

    Every candidate kind arrives as a factory on ``options.candidate_kinds``, with whatever
    collaborators it needs bound by the host that built it, so this loop takes no subject
    factory and no simulator of its own. ``judge_service`` is the stateless single-dim judge;
    ``None`` skips all scoring (results carry empty score lists / None axes).

    ``cell_sink`` is the same list this returns, handed in by a caller that needs
    to see what was delivered even when this function does not return at all.
    Every way a run stops early destroys the frame before the ``return``: the
    cost cap raises, the job timeout cancels, an operator's cancel cancels. A
    caller owing a record of the run's shortfall — which is exactly the situation
    where there IS one — cannot get it from a return value that never arrives, so
    it passes a list, reads it in its own ``finally``, and this loop appends to it
    rather than to one of its own. Omitted, the loop keeps its own list and
    behaves as before.

    Cost cap: ``budget_gate`` is checked BEFORE each cell — when it
    returns a :class:`~threetears.evals.run.budget.CapBreach` (this run's accumulated cost
    has exceeded its cap, or a result's spend could not be priced) the loop stops GRACEFULLY by raising
    :class:`~threetears.evals.run.budget.BudgetStoppedError`, carrying that breach so the
    stop reports the ceiling and the spend that crossed it rather than asserting
    that one did. ``None`` from the gate means the run may proceed.
    Every already-run cell is saved before its successor's check, so nothing
    produced is lost; the job manager translates the raise into the
    ``budget_stopped`` status. ``on_cost`` is invoked with each saved result's
    ``cost_usd`` so the cap accumulates this run's spend in memory and trips on
    the next cell once it's over — ``None`` for a result whose spend went unpriced, which
    the cap counts as such. Both default ``None`` (no gating / no
    recording) — the pure-function contract for tests and cap-disabled contexts.

    Apparatus faults: a cell an :class:`~threetears.evals.contracts.host.apparatus.ApparatusError`
    ended is recorded excluded and the loop goes on. Only when EVERY cell of the run ended that
    way does the loop raise :class:`EveryCellApparatusFailedError`, after the last cell is saved:
    the rig never worked, so the run measured nothing, and the job manager files it ``failed``.

    Cancellation: a cancel that lands while a cell runs is delivered once that cell is recorded —
    excluded under the ``cancelled`` termination, with the spend its sink holds — and saved, so
    the money a cancelled run spent on its last cell is on the record with the rest.

    Account exhaustion: when a cell's candidate, simulator or judge call was refused for the
    calling ACCOUNT (``_CellSink.account_refusal`` — out of credit, or the key refused), that
    cell is saved and reported like any other, and then the loop stops by raising
    :class:`~threetears.evals.run.budget.AccountExhaustedError`: every model behind the key would get the
    same answer, so no further cell is launched. The error carries how far the run got and what
    its delivered results cost, summed from the cells this loop recorded; the job manager
    translates it into the ``exhausted`` status. Unlike the cost cap this needs no caller wiring —
    the refusal is observed, not configured.

    Raises:
        ValueError: ``judge_service`` and ``run.judge_model`` disagree about whether this run is
            judged — one is set and the other is not. Refused before any cell runs, so nothing
            has been spent.
        BudgetStoppedError: The cost cap tripped before a cell.
        AccountExhaustedError: A cell's call was refused for the calling account.
        EveryCellApparatusFailedError: An apparatus fault excluded every cell the run produced.
        asyncio.CancelledError: The run was cancelled; the cell it struck is recorded first.
    """
    from threetears.evals.run.budget import AccountExhaustedError, BudgetStoppedError

    # Every result is stamped with ``run.judge_model``, and every read surface takes a blank there
    # to mean "this run was not judged". So a judged run with no judge model would file its scores
    # with their judge lost, and an unjudged run naming one claims a judge that scored nothing. The launch that resolves the judge
    # lives with the host, so the runner is the one point every host's judged run passes through,
    # and the one place the pairing can be held.
    if (judge_service is None) != (run.judge_model is None):
        raise ValueError(
            "a judged run must name its judge and an unjudged run must name none: "
            f"judge_service is {'set' if judge_service is not None else 'None'} but "
            f"run.judge_model is {run.judge_model!r}. Record on the run the model the judge service "
            "scores with, or pass no judge service."
        )

    callbacks = callbacks or RunCallbacks()
    total = len(test_cases) * run.k_runs
    done = 0
    # Each cell an apparatus fault excluded, by its fault — see the check after the loop.
    apparatus_failures: list[str] = []
    # The caller's list when it supplied one, so what it can read mid-flight is
    # the identical object this returns — never a copy that could disagree.
    summaries: list[CellSummary] = cell_sink if cell_sink is not None else []

    # One lane for the whole run, so every cell records into — or replays — the one corpus the run
    # names: its own id for a capture, the capture it names for a replay. ``None`` with cassettes off.
    cassettes = CassetteLane.for_run(run, host.storage)

    # The variant is the resolved contestant stack, which a run holds fixed — it is one
    # arm — so resolve it once rather than per cell. Every result the loop produces carries it, timeouts and stubs
    # included: a cell that failed still observed a specific variant, and
    # dropping the key there would silently bias any grouping toward successes.
    #
    # READ from what the launch stamped on the run, never derived a second time here. The run
    # carries the lever map its own launch resolved, and digesting that map is what makes this
    # loop's keys the same keys the run says it has. A second independent derivation would agree
    # today and is exactly the drift this pre-image was added to remove: two readers of one
    # question, authored apart, where either can move without the other hearing about it. The
    # run is authoritative; a run that carries no map (one its host assembled without the launch)
    # falls through to the derivation inside.
    variant = resolve_variant_identity(run=run, profile=host.profile)
    model = run.candidate_model

    # Shuffled rather than nested k -> model -> test case: within-run drift
    # (a box that slows, a neighbouring job, a leak) would otherwise land on the
    # matrix aligned with the coordinate being compared. See
    # :func:`cell_execution_order` for the order's derivation and why its seed is
    # not a measurement condition.
    for k, tc in cell_execution_order(run, test_cases, rng=options.cell_order_rng):
        # Cost cap: stop gracefully BEFORE spending on the next
        # cell if this run's accumulated cost has exceeded its cap. Results
        # already saved below are preserved; the raise becomes a
        # ``budget_stopped`` run.
        breach = budget_gate() if budget_gate is not None else None
        if breach is not None:
            log.warning(
                "Eval run %s stopping: $%.4f priced spend and %d unpriced result(s) against its $%.4f cap "
                "after %d/%d results",
                run.id,
                breach.accumulated_usd,
                breach.unpriced_results,
                breach.max_cost_usd,
                done,
                total,
            )
            raise BudgetStoppedError(done, total, breach)
        # Per-cell timeout: bound the WHOLE result execution (candidate turn + delivery drain +
        # judging) so a hung cell is capped here and the run proceeds to the rest, instead of
        # stalling to the matrix-adaptive eval_job backstop.
        #
        # The deadline cancels the cell from outside, so everything its frame held dies with it.
        # What the record needs survives on ``sink``, which this loop owns, the kind reports
        # into, and the timeout arm reads: the spend already billed, the
        # per-role rows, the output so far, and what the cell was waiting on when it struck.
        sink = _CellSink()
        cancelled_in_cell = False
        try:
            async with host.cell_timeout(options.cell_timeout_s):
                result, cell_trace = await run_one_result(
                    host,
                    template=template,
                    test_case=tc,
                    subject_id=run.subject_snapshot.subject_id,
                    model=model,
                    k_iteration=k,
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    judge_service=judge_service,
                    options=options,
                    judge_model=run.judge_model,
                    cassettes=cassettes,
                    variant=variant,
                    subject_snapshot=run.subject_snapshot,
                    sink=sink,
                )
        except EvalCellTimeout as timed_out:
            charged_to, waiting_on = _DEADLINE_CHARGE[sink.pending]
            log.error(
                "Eval cell timed out after %.0fs waiting on %s (test_case=%s model=%s k=%s run=%s attribution=%s)",
                timed_out.elapsed_s,
                waiting_on,
                tc.id,
                model,
                k,
                run.id,
                timed_out.attribution,
            )
            # Whose failure the deadline is follows what it struck: the candidate's own turn or
            # the background work it started fails the cell; the simulator, the judge or the rig
            # excludes it. Recorded beside the errors the kind had
            # reported, so one it recorded before the deadline keeps its own category and
            # ``classify_result`` reads one set of fields for every exit. Kept on the sink, where
            # the account-refusal check below reads it however the cell ended. A deadline in the
            # judge phase is charged to each dim it cut off instead (``_cut_short_cell``); a kind
            # naming the judge outside that phase is waiting on nothing the engine runs, so on the rig.
            errors = sink.errors = ErrorLedger.of(sink.side())
            msg = f"cell timeout while waiting on {waiting_on}: {timed_out}"
            if charged_to == "candidate":
                errors.add_candidate(msg)
            elif charged_to == "infra" or sink.judging is None:
                errors.add_infra(msg)
            # Checks the kind already reported are kept as it reported them (``_degraded_capture_fields``).
            # A candidate-charged deadline before that cancelled the grading, so the checks a candidate
            # failure counts as failed are the template's own — stamped as unevaluated failures, so
            # ``counted_goal_verdicts`` counts them as it counts every other candidate failure's.
            # A rig-charged deadline is in no per-check rate and stamps none.
            result, cell_trace = _cut_short_cell(
                sink=sink,
                scope_id=run.scope_id,
                eval_run_id=run.id,
                test_case_id=tc.id,
                model=model,
                k_iteration=k,
                subject_id=run.subject_snapshot.subject_id,
                variant=variant,
                # The cell ran — long enough to time out. A timed-out cell is the single most
                # valuable observation for "was something else running?", so the conditions are
                # sampled here rather than nulled.
                concurrent_eval_jobs=sample_concurrent_eval_jobs(options),
                termination="cell_timeout",
                judge_model=run.judge_model,
                rate_table=options.external_rates,
                unfinished_judging=f"did not finish: {msg}",
                ungraded_goal_checks=(
                    _unevaluated_goal_checks(template, waiting_on=waiting_on) if charged_to == "candidate" else []
                ),
            )
        except asyncio.CancelledError:
            # The run was cancelled while this cell ran. The cancel still ends the run — it is
            # re-raised below, once the cell is accounted for — but the cell's spend was real, and
            # it lives only on this sink, so it is recorded as an excluded cell first rather than
            # dropped with the frame. Excluded on the rig's side: nothing the candidate did ended it.
            _, waiting_on = _DEADLINE_CHARGE[sink.pending]
            log.warning(
                "Eval cell cancelled with its run while waiting on %s (test_case=%s model=%s k=%s run=%s); "
                "recording what it had spent",
                waiting_on,
                tc.id,
                model,
                k,
                run.id,
            )
            errors = sink.errors = ErrorLedger.of(sink.side())
            msg = f"cell cancelled with its run while waiting on {waiting_on}"
            if sink.judging is None:
                errors.add_infra(msg)
            result, cell_trace = _cut_short_cell(
                sink=sink,
                scope_id=run.scope_id,
                eval_run_id=run.id,
                test_case_id=tc.id,
                model=model,
                k_iteration=k,
                subject_id=run.subject_snapshot.subject_id,
                variant=variant,
                concurrent_eval_jobs=sample_concurrent_eval_jobs(options),
                termination="cancelled",
                judge_model=run.judge_model,
                rate_table=options.external_rates,
                unfinished_judging=f"did not finish: {msg}",
            )
            cancelled_in_cell = True
        # A write failure loses the cell — this loop holds no second copy that
        # could stand in for the lost record. Say so loudly and carry the answer
        # on the summary; the run still proceeds, because one unwritable cell is
        # not a reason to discard the rest of a sweep that has already been paid for.
        #
        # Off the loop, and waited for to its end even through a cancel: the write cannot be
        # interrupted once a worker has it, so abandoning the wait would record this cell as lost
        # while its row lands anyway. The cancel is delivered once the cell is accounted for below.
        saving = asyncio.ensure_future(
            run_blocking(host.blocking_executor, host.storage.save_eval_result, result, cell_trace)
        )
        cancelled_while_saving = await wait_through_cancellation(saving)
        try:
            saving.result()
            persisted = True
        except StorageError:
            persisted = False
            log.error(
                "Eval result NOT persisted — cell is lost (result=%s run=%s test_case=%s model=%s k=%s)",
                result.id,
                run.id,
                tc.id,
                model,
                k,
            )
        summaries.append(CellSummary.from_result(result, persisted=persisted))
        done += 1
        # Accumulate this cell's spend into the per-run cost cap so the
        # run trips its own cap on the next cell once it's over.
        # Passed as it stands: ``None`` is unpriced spend, which the cap counts
        # as such and an enforcing cap stops on.
        if on_cost is not None:
            on_cost(result.cost_usd)
        if cancelled_in_cell or cancelled_while_saving:
            raise asyncio.CancelledError
        if sink.errors.apparatus_failed:
            apparatus_failures.append(sink.errors.infra_error or "an apparatus fault")
        if callbacks.on_result is not None:
            await callbacks.on_result(result)
        if callbacks.on_progress is not None:
            await callbacks.on_progress({"completed": done, "total": total, "current_model": model})
        # The account paying for the run refused one of this cell's calls. The cell is already
        # saved, counted and reported above; every later cell would be refused the same way, so
        # none is launched. Checked after the cell rather than before the next one so a refusal
        # on the LAST cell still ends the run ``exhausted`` instead of ``completed``. Read off the
        # sink, which answers it however the cell ended: its ledger (``adopt`` built it from what
        # the kind returned, the timeout arm from the kind's reading when its deadline struck
        # first) and the judge calls that returned.
        if (refusal := sink.account_refusal) is not None:
            priced = [summary.cost_usd for summary in summaries if summary.cost_usd is not None]
            accumulated_usd = sum(priced)
            unpriced_results = len(summaries) - len(priced)
            log.warning(
                "Eval run %s stopping: the provider account refused a call after %d/%d results "
                "($%.4f priced spend, %d unpriced result(s))",
                run.id,
                done,
                total,
                accumulated_usd,
                unpriced_results,
            )
            raise AccountExhaustedError(
                done, total, accumulated_usd=accumulated_usd, unpriced_results=unpriced_results, detail=refusal
            )

    # Every cell was excluded by an apparatus fault: the rig never worked, so the run measured
    # nothing at all. Ending it ``completed`` would publish a run with no measurement as a
    # finished one; it fails instead, naming the faults, because an operator has to repair the
    # rig — re-capture a corpus, fix a seam — before any relaunch can measure. A run where SOME
    # cells measured completes, and its completeness record counts the ones the rig excluded.
    if total and len(apparatus_failures) == total:
        raise EveryCellApparatusFailedError(total, faults=apparatus_failures)

    return summaries


__all__ = [
    "DEFAULT_CELL_TIMEOUT_S",
    "DEFAULT_JUDGE_CONCURRENCY",
    "CellContext",
    "CellOutcome",
    "ErrorLedger",
    "EveryCellApparatusFailedError",
    "GoalCheckUnevaluable",
    "KindFactory",
    "RunCallbacks",
    "RunnerOptions",
    "assert_preconditions",
    "build_judge_context",
    "cell_execution_order",
    "evaluate_goal_state",
    "execute_run",
    "fold_metered_cell",
    "grade_goal_checks",
    "judge_dims",
    "metered_cell_tally",
    "precondition_failure_text",
    "run_one_result",
    "sample_concurrent_eval_jobs",
]
