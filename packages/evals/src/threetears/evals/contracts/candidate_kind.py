"""The candidate-kind seam — the one thing that varies between evaluable subjects.

The seam is two operations, and everything else in the runner is subject-agnostic:

* ``prepare(subject_snapshot, variant_config, world_seed) -> instance``
* ``invoke(instance, test_case, sink) -> CandidateOutput``

This module is the engine half of that seam: the protocol, the object a kind hands
back, and the two failures the dispatch site has to tell apart. **It is not where a
kind lives.** A kind is implemented by whoever owns its subject — a host's in its own
adapter, the reporter kind in analysis — and reaches the runner as a factory, so this
module imports nothing host-side and travels with the engine.

**Which engine package it travels in is a separate question, and the answer is contracts,
by elimination.** The extraction is four packages, and a kind is implemented on both sides of
the run/analysis line: a host's kinds in its adapter, and the reporter kind in analysis. A
Protocol living in run would give analysis an edge into run, which the package matrix forbids;
contracts is the one package every implementer may import (a judgement call, recorded here).
The cost is that the lingua franca UIs and
exporters bind to carries an interface only a runner drives, which is why this module holds
the declaration and nothing that executes it.

Why the subject arrives as a PAIR
---------------------------------

``subject_snapshot`` is a key, a label and content hashes — deliberately not the host's
own record of its subject. The rich host object never passes through the engine at
all: the host binds it into the factory a kind arrives through, once per run, and the
engine hands ``prepare`` only the first half. Putting the rich object back inside the
snapshot is the move the model exists to forbid, because
a snapshot that retains what the candidate was made of scales a run's retention with
what it produced rather than with its matrix.

Why ``trace`` and ``telemetry`` do not cross
--------------------------------------------

The design review of this seam warned that it is *not* a good library boundary, because a kind
like ``conversational-turn`` drags in the simulator, the world and the turn/trace
model — "so ``CandidateOutput.trace``/``telemetry`` would leak the first host's shapes across
it". That objection is about SHAPE, not about imports, so a clean import scan does
not answer it. It is answered here by what the seam DECLINES to carry:

* **``trace`` is off ``CandidateOutput`` entirely.** The rich turn/span payload never
  crosses. A cell's spans reach storage through the trace-sink port
  (:mod:`threetears.evals.contracts.host.traces`). The runner OWNS those windows and harvests them;
  the kind chooses only where each one opens, because their extents are a measurement
  decision rather than a transport one. The assembled
  :class:`~threetears.evals.contracts.models.EvalTrace` then rides beside the result on the runner's own
  ``CellOutcome``. Nothing about a span shape is expressible through this seam, so nothing
  about a span shape can leak across it. **Every kind is handed its cell's two windows as
  ``prepare``'s ``span_window``** (:class:`CellSpanWindow`) — two context managers, no span
  shape — so a kind built anywhere opens the same windows the built-in one does. A kind that
  opens none, and one whose window closed with no turn-root span in it, are each disclosed by
  a warning at the harvest rather than left to read as a cell that timed nothing.
* **``telemetry`` is the measurement, and every field on it is already engine
  vocabulary** — roles and dollars (:class:`~threetears.evals.contracts.models.RoleUsage`), the
  metered-call ceiling's own disclosure, the one wall-clock window no span covers.
  **The span-derived buckets are deliberately NOT on it**: they come off the trace-sink
  port, which the runner harvests, so the only timings a kind reports are the ones it
  measured off a clock of its own. It names no tool, no transport and no subject.
* **What only the kind can name goes on ``kind_payload``, opaque.** A classifier's
  per-cell outcome facts, the internals of a tool's background run: the engine has no word for
  them, so it stores them verbatim on the result, unread, rather than declaring a field whose
  name would be one kind's. The host that owns the kind is their only reader.
* **Background work crosses in engine vocabulary.** A tool that acknowledges a call at once and
  delivers its work turns later is reported as one
  :class:`~threetears.evals.contracts.models.AsyncDelivery` per piece of work — who asked, when it
  was acknowledged and delivered, what it ran on, what it delivered, and whether a harness
  supplied the payload. Those are properties of any such tool; what the tool's work consisted
  of is the kind's, and goes on ``kind_payload``.
* **``output`` is a list of JSON documents, and the engine never looks inside one.**
  For ``conversational-turn`` it is the delivered turns; for a classifier it is one
  label record; for an artifact generator it is the artifact. The trace document stores
  it and no judge reads it: what a judge reads is the kind's own rendering
  (``judge_evidence``), because only the kind knows how its output reads as text.

What that leaves crossing is ``{output, mechanical_facts, telemetry, host_measures}`` — what
the seam's design lets downstream JUDGE, plus the registered grade a code-graded kind computes itself,
minus the trace payload objected to above; ``judge_evidence``, every judged kind's own rendering of what
its judge reads — the subject, the case material and the artifact, so the engine renders no
subject, transcript or document shape of its own; ``stop_cause``, why a conversing kind
stopped; ``async_deliveries``, the background work above — and, as the cell's own bookkeeping
rather than anything the scoring machinery interprets: ``candidate_errors`` and
``infra_errors``, whose split IS the fail-versus-exclude decision (a kind that joined them
into one string would make the runner parse a scoring decision back out of prose);
``candidate_instance_id``, the host identity the result is joinable on; and ``kind_payload``,
the deliberately opaque carrier, which is the disposition rather than an exception to it.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal, Protocol

from pydantic import model_validator

from threetears.evals.contracts.base import EvalBaseModel, VerbatimJsonObject
from threetears.evals.contracts.call_ledger import CallLedger
from threetears.evals.contracts.cassettes import CellCassettes
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.models import (
    AsyncDelivery,
    CellTermination,
    ConversationStopCause,
    EvalTestCase,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    PreconditionOutcome,
    RoleUsage,
    WorldSeed,
)

__all__ = [
    "CandidateKind",
    "CandidateKindDefect",
    "CandidateOutput",
    "CandidatePreparationFailed",
    "CandidateTelemetry",
    "CellPending",
    "CellSink",
    "CellSpanWindow",
    "UnknownCandidateKind",
    "VariantConfig",
]


class VariantConfig(EvalBaseModel):
    """The A/B'd stack one cell runs, as every kind alike receives it.

    The third ``prepare`` argument. The candidate model is the part every kind needs —
    it is what a conversing candidate is built against, what a classifier calls, what an artifact
    generator generates with.

    **The derived contestant identity used to ride beside it and is gone.** Every kind in the
    tree reads ``candidate_model`` alone. This model is registered shared contract, so an
    unread field here is surface a second host inherits and kind authors are taught to
    populate; a kind that needs to know which contestant it is takes an argument for it,
    which is a requirements change and not a field left standing in case.
    """

    candidate_model: str
    """The model this cell binds the candidate to (``EvalResult.model``)."""


class CandidateTelemetry(EvalBaseModel):
    """What one candidate's execution cost, and the windows nothing else timed.

    Every field is a measurement the engine already stores on
    :class:`~threetears.evals.contracts.models.EvalResult`; nothing here describes HOW the
    candidate was driven. See this module's docstring for why that is the property
    keeping host shapes off this seam rests on.
    """

    usage: list[RoleUsage] = []
    """One row per ``(role, model, provider, unit)`` this candidate actually observed.

    A list on every exit of a running candidate — never ``None``, which would mean no
    observation was made at all.

    **These rows are the candidate side's spend, and the only statement of it.** The engine
    derives the result's ``cost_usd`` from them, the judge's rows added, over the run's cost
    roles (:func:`~threetears.evals.contracts.usage_capture.blended_cost`) — so a kind reports
    each call's dollars as its client priced them, ``None`` where it priced nothing, and never a
    total of its own beside them that could disagree.
    """

    metered_calls_refused: int | None = None
    """Metered third-party calls this kind refused ITSELF, against a ledger the run has not.

    Leave it ``None``, which is what every kind in the tree does. The run's metered-call
    ledger is the engine's, and the dispatch site counts this cell's refusals against it
    around ``prepare``/``invoke`` — so a kind that reports nothing here still gets an
    honest count, and a kind that reports its own slice of the RUN's ledger would be
    counted twice. It exists for the other case: a kind metering a provider of its own,
    whose refusals the run's ledger never sees. Those are added to the engine's count.

    ``None`` means this kind counted nothing of its own — never that nobody counted, which
    is a statement only the dispatch site is in a position to make.
    """

    untimed_reason: str | None = None
    """Why this candidate did no work there was any latency to measure, when it did none.

    A replay returning a stored artifact, say. The cell still reports no latency and still
    leaves the cost-vs-latency comparison — that is correct, since there is nothing to compare
    — but the runner stops warning that the kind FAILED to open a timing window, which would be
    false. ``None`` for every kind that did timed work, which is every kind that calls anything.
    """

    phase_timings: dict[str, float] = {}
    """Per-phase wall-clock the candidate's own work reported, folded per source."""

    async_wait_ms: float | None = None
    """Wall-clock spent waiting on in-flight background work, which no span covers.

    ``None`` until something actually waited — a different fact from having waited
    zero milliseconds.
    """

    concurrent_eval_jobs: int | None = None
    """Busiest observation of how many eval jobs ran alongside this candidate.

    ``None`` when no probe was wired: nobody looked, which is not the same claim as
    "nothing else was running".
    """

    turns_ended_by_budget: int | None = None
    """How many of the candidate's turns the host's turn budget ended before they finished.

    A host that bounds a turn in production must bound it in the cell too, or a turn production
    would have cancelled completes and scores. An ended turn delivered nothing, so the runner lands
    this on the result's covariates (:data:`~threetears.evals.contracts.covariates.TURN_BUDGET_ENDED_KEY`)
    and a non-zero count makes the result a candidate failure. ``0``: the turns ran under a budget and none outlived it.
    ``None``: no budget bounded them — a kind, or a host, with none.
    """


class CandidateOutput(EvalBaseModel):
    """What one candidate produced, as everything below the dispatch sees it.

    The seam's whole output contract in one object. Rubric, judge and goal-state recording read the judged
    artifact and the judge-free facts; usage capture reads the measurement; the cell summary
    reads the two error lists and persistence reads the candidate's host identity. **None of
    them reads anything kind-shaped** — that is the invariant, and it is about shape rather
    than about how many fields cross.
    """

    output: list[dict[str, Any]] = []
    """What the candidate produced, as an ordered list of JSON documents.

    :class:`~threetears.evals.contracts.models.EvalTrace` stores it; a judge reads the kind's
    ``judge_evidence`` instead. The engine never looks inside a document: for ``conversational-turn`` these
    are the delivered turns, for a classifier one label record, for an artifact
    generator the artifact. Empty means the candidate produced nothing to judge,
    which is what suppresses the judge phase.
    """

    mechanical_facts: list[GoalStateOutcome] = []
    """The judge-free tier: what held about the candidate's execution, and what did not.

    The cheap regression gate, reported by every kind — goal-state checks for a
    conversational turn, ``label == expected`` for a classifier, schema validity for an
    artifact. Lands on :attr:`~threetears.evals.contracts.models.EvalResult.goal_state_outcomes` and
    is handed to the judge as context.
    """

    call_ledger: CallLedger | None = None
    """Every action the candidate took that succeeded, as the kind recorded it.

    The ledger the kind graded its goal checks against
    (:func:`~threetears.evals.run.runner.grade_goal_checks`), stored on the cell's
    :class:`~threetears.evals.contracts.models.EvalTrace` so a re-check
    (:mod:`threetears.evals.run.recheck`) re-grades the stored result from exactly what the
    candidate did, under today's rule, without re-running it. ``None`` for a kind that keeps no
    ledger — nobody recorded, which is a different fact from an empty ledger, where the kind
    recorded and the candidate called nothing.
    """

    telemetry: CandidateTelemetry = CandidateTelemetry()
    """What the execution cost and how long its phases took."""

    candidate_errors: list[str] = []
    """Errors attributable to the candidate itself — its own model failing.

    Kept apart from ``infra_errors`` at the source rather than parsed back out of a
    joined string later: a candidate error FAILS the cell, an infra error EXCLUDES it.
    """

    infra_errors: list[str] = []
    """Errors attributable to the harness — the rig, the delivery, a timeout."""

    account_refused: bool = False
    """The candidate's call was refused for the calling ACCOUNT — out of credit, or the key refused.

    Set by the kind at the one site that classified the failure, alongside the apparatus entry
    it puts on ``infra_errors`` (which excludes the cell). A flag rather than a reading of that
    message: it is what the RUN acts on — every model behind the same key gets the same answer,
    so the runner stops launching cells and the run ends ``exhausted`` rather than excluding the
    rest of the matrix one paid-for cell at a time. ``False`` for every kind that cannot tell an
    account's refusal from a model's.

    Requires an ``infra_errors`` entry: the refusal is an apparatus fault first, and the entry is
    what excludes the cell and what the run's own stop reason quotes.
    """

    host_measures: dict[str, float] = {}
    """What the host's own grader measured about this cell, by registered measure name.

    The mechanical grade a kind computes itself, for the kinds whose scoring is code rather
    than a model: ``field_accuracy`` for an extractor, ``classifier_accuracy`` and
    ``parse_failure_rate`` for a classifier. Lands verbatim on
    :attr:`~threetears.evals.contracts.models.EvalResult.host_measures`, which the analysis bundle reads
    through the measure registry.

    **A name and a float is engine vocabulary, which is why this crosses in the open while a
    host-named stream does not.** The engine already owns what a measure IS — the registry
    declares its population, range and merit axis, and the descriptor lookup resolves the
    name — so carrying one names no tool, no transport and no subject. What it deliberately
    does not do is interpret: an unregistered name is the registry's defect to report, not
    this seam's.

    ``{}`` is "this host measured none", which is every kind whose scoring is a judge model.
    """

    judge_evidence: JudgeEvidence | None = None
    """Everything the judge reads about this cell, as the kind renders it.

    Required with every non-empty ``output`` of a judged kind — :attr:`JudgedArtifact.TRANSCRIPT`
    and :attr:`JudgedArtifact.DOCUMENT` alike — and refused from an unjudged one: the runner holds
    each cell to its kind's declaration. The engine renders no subject, transcript or document
    of its own, so what a judge sees — including facts the candidate's interlocutors never saw —
    is the kind's to decide. Stored on the cell's trace, so a re-judge sends exactly this again.
    """

    stop_cause: ConversationStopCause | None = None
    """Why the candidate's conversation stopped taking turns, for a kind that converses.

    Lands on :attr:`~threetears.evals.contracts.models.EvalResult.stop_cause`. ``None`` for a kind that does
    not converse — there is no conversation to have stopped.
    """

    candidate_instance_id: str | None = None
    """The host identity this candidate ran under, joinable to the host's own artifacts.

    Lands on :attr:`~threetears.evals.contracts.models.EvalResult.candidate_instance_id`. ``None`` for a
    candidate with no persisted identity to join to, which is every kind that mints none.
    """

    async_deliveries: list[AsyncDelivery] | None = None
    """One entry per piece of background work the candidate started, in the order it was acknowledged.

    Lands verbatim on :attr:`~threetears.evals.contracts.models.EvalResult.async_deliveries`, and is what
    the engine reads to tell a seeded or replayed delivery from a live one. ``None`` for a kind that
    starts no background work — nobody watched, which is a different fact from an empty list, where
    the kind watched and nothing was started.
    """

    kind_payload: VerbatimJsonObject | None = None
    """What only this kind can name about the cell, carried to storage unread.

    The return-direction twin of ``EvalRun.host_payload`` and ``EvalTestCase.host_payload``, and the
    same discipline: the engine stores it verbatim on
    :attr:`~threetears.evals.contracts.models.EvalResult.kind_payload` and reads no field of it, and the
    host that owns the kind is the only thing that unpacks it. ``None`` for a kind with none.

    **This is where the line the seam draws is precise.** A measurement the engine has words for —
    dollars, roles, a phase's wall-clock, a refused call, a piece of background work — is declared on
    :class:`CandidateTelemetry` or :attr:`async_deliveries`, as is a registered measure on
    ``host_measures``. A measurement only one kind can name travels here, unnamed and unread. Renaming
    such a stream into a neutral-looking engine field would pass every check the engine has and leave
    the kind's shape on the seam anyway, which is the failure this field makes unnecessary.

    JSON values only, refused at construction: it is persisted as it stands, and a value storage cannot
    encode would otherwise surface at the write, after the cell's judging had been paid for.
    """

    @model_validator(mode="after")
    def _account_refusal_is_an_apparatus_fault(self) -> CandidateOutput:
        """Refuse an account refusal that records no apparatus fault.

        Without the ``infra_errors`` entry the refused cell would be SCORED (nothing excludes it),
        and the run would stop citing a refusal the cell never recorded.
        """
        if self.account_refused and not self.infra_errors:
            raise ValueError(
                "account_refused is set but infra_errors is empty; an account refusal is an apparatus fault"
            )
        return self


class CellSpanWindow(Protocol):
    """One cell's two tracing windows, as the kind driving that cell sees them.

    The kind-facing half of the trace-sink port (:mod:`threetears.evals.contracts.host.traces`): the
    engine owns the sink, mints the cell's identity and harvests the record, and hands a kind
    only these two scopes. That split is what lets a kind time its own work without the
    host-shaped span payload crossing the seam — *where* each window opens is a measurement
    decision only the kind can make, and what lands inside one is the host's shape, which the
    kind never sees through this handle.

    A kind is handed one per cell as ``prepare``'s ``span_window`` and uses it in
    :meth:`CandidateKind.invoke`, so it belongs on the instance ``prepare`` returns rather
    than on the kind — a kind wired for a whole run is reached for every cell of it.

    When the run wired no sink both scopes are no-ops, so a kind opens them unconditionally
    and never asks whether anything is watching.
    """

    def identity(self) -> AbstractContextManager[None]:
        """Mark everything inside as this cell's work, apparatus included.

        Returns:
            The identity scope — a no-op when nothing is watching.
        """
        ...  # pragma: no cover — protocol

    def collecting(self) -> AbstractContextManager[None]:
        """Collect the spans the candidate's own work emits — this cell's latency.

        Opened around the candidate's own execution and nothing else: the buckets it
        produces are what the cost-vs-latency comparison reads, so apparatus work inside it
        is time the candidate is charged for and did not spend.

        Returns:
            The collection scope — a no-op when nothing is watching.
        """
        ...  # pragma: no cover — protocol


#: What a running cell is waiting on — the component its deadline, striking now, would be
#: waiting for. A kind names it through :meth:`CellSink.waiting_on`; the runner names the rest.
#:
#:   ``apparatus``        the eval's own machinery: building and seeding the subject, starting
#:                        and stopping its tools, grading the world.
#:   ``candidate``        the candidate's own work — its model's calls, and the actions it takes.
#:   ``background_work``  background tool work the candidate started and the cell is waiting
#:                        out — an async tool's run, on whatever model the host drives it with.
#:   ``simulator``        the simulated user's next reply.
#:   ``judge``            the judge scoring the finished cell.
CellPending = Literal["apparatus", "candidate", "background_work", "simulator", "judge"]


class CellSink(Protocol):
    """One cell's record of where it stands, as the kind driving that cell reports into it.

    The runner bounds each cell with a deadline that cancels it from outside, and every local
    of the cancelled frame dies with it — so a cell's spend and what it was waiting on cannot be
    taken from what :meth:`CandidateKind.invoke` returns, because on that exit it returns nothing.
    The runner owns one of these per cell, hands it to ``invoke``, and reads it after the cancel.
    It is the only thing the deadline's record is built from.

    A kind is handed one per cell, like ``span_window``, and uses it for that cell alone.
    """

    def waiting_on(self, pending: CellPending) -> None:
        """Name what the cell is waiting on from here — whose failure a deadline now would be.

        The runner marks ``candidate`` before ``invoke``; a kind that alternates the candidate's
        work with other components' names each as it awaits it. **Never call it in a**
        ``finally``: a cancel unwinds through those, and the write would replace the answer with
        whatever the unwind was doing.

        Args:
            pending: The component the next await is waiting for.
        """
        ...  # pragma: no cover — protocol

    def report_progress(self, read: Callable[[], CandidateOutput]) -> None:
        """Register how to read the candidate side as it stands, for a cell cut off before it returns.

        ``read`` returns what ``invoke`` would return if the cell ended now: the spend already
        billed and its usage rows, the errors recorded, the output produced so far. The runner
        calls it once, after a deadline has cancelled ``invoke`` and the cancel has unwound —
        so what the unwind itself recorded is in it — and never once ``invoke`` has returned,
        whose output is then the record.

        A reading rather than a snapshot pushed at each change, because a kind's spend and
        errors change at every await, the unwind included, and a push at each is a list of sites
        that fails by omission. ``read`` must be synchronous and read only what the kind has
        accumulated; nothing it reaches is still running. A kind registers one reading per cell,
        and a kind that registers none reported nothing — a call in flight at the deadline has
        no response to report, so a kind whose only call is that one has nothing to register.

        Args:
            read: The candidate side so far.
        """
        ...  # pragma: no cover — protocol


class CandidateKind(Protocol):
    """The two operations that vary between one evaluable subject and another.

    A kind is constructed with its collaborators — a subject factory, a completion
    client, whatever it builds its subject through — as explicit typed arguments, never
    by reaching a registry from inside :meth:`invoke`.

    Both operations are ``async``, not synchronous: building a subject and
    driving it both cross awaits in every host that has more than a function call
    behind them, and a synchronous protocol would force each kind to run its own loop.

    **What a failure costs.** :meth:`prepare` may raise :class:`CandidatePreparationFailed` and
    it costs ONE cell: the runner records a cleanly-excluded result under the termination arm
    the kind named and goes on to the next cell. Either operation may raise
    :class:`~threetears.evals.contracts.host.apparatus.ApparatusError` — a replay miss, a corrupt
    recording, any fault of the rig rather than of the world under test — and that too costs
    ONE cell: the runner records it excluded under ``apparatus_failed``, with whatever spend the
    kind had reported through its sink, and goes on. A kind's tool boundary re-raises one rather
    than absorbing it into an ordinary tool failure, which is what the arm exists for. Every
    other fault a host can attribute once a candidate is running is the candidate's own or the
    rig's, and is *reported* on the way out through ``candidate_errors`` and ``infra_errors``.
    **Any other exception out of either operation is not a cell-level event: it leaves the
    runner and ends the run** — it is the kind's own code failing, so every cell would do it.

    **Lifetime and reentrancy.** A kind instance may be used for more than one cell — a
    kind reaches the runner through a factory on ``RunnerOptions.candidate_kinds``, which is
    asked for every cell of the run and may hand back the same instance each time — so it
    must carry no state from one
    ``prepare``/``invoke`` pair to the next; a cell's own state belongs on the instance
    ``prepare`` returns. Cells within a run execute serially, so a kind need not be
    concurrency-safe against itself; two runs in one process are two callers and two
    instances.
    """

    judged_artifact: JudgedArtifact
    """What a judge reads of this kind's output, and so which axes a judged cell scores.

    A declaration about the kind, never about one cell: the runner reads it before ``prepare``
    — refusing a judged run of an :attr:`~JudgedArtifact.UNJUDGED` kind before anything is
    spent — and holds every cell's ``judge_evidence`` to it after ``invoke``.
    """

    async def prepare(
        self,
        *,
        subject_snapshot: SubjectSnapshot | None,
        variant_config: VariantConfig,
        world_seed: WorldSeed,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
    ) -> Any:
        """Build one candidate, ready to be invoked.

        Args:
            subject_snapshot: The engine's view of the subject — key, label and content
                hashes. ``None`` for a caller that resolved none, which is every ad-hoc
                or fixture run. A kind that needs the subject's substance has it bound
                into its factory by its host, never through this.
            variant_config: The A/B'd stack this cell runs.
            world_seed: The world state the scenario presumes, before any turn.
            span_window: This cell's tracing windows, to open around the candidate's own
                work in :meth:`invoke`. Keep it on the returned instance: it is the cell's,
                not the kind's, and a kind wired for a run is reached for every cell.
            cassettes: This cell's handle on the run's cassette lane, or ``None`` for a run with
                cassettes off. Handed one, the kind calls its ``wire`` once with the candidate's
                :class:`~threetears.evals.contracts.cassettes.CassetteSeams` before returning; the
                engine refuses a cell whose kind does not, since its tools would run live under a
                replay. The cell's, like ``span_window``: it already names the corpus and the case.

        Returns:
            The kind's own prepared instance, opaque to the engine and handed straight
            back to :meth:`invoke`.

        Raises:
            CandidatePreparationFailed: The candidate could not be built. Carries the
                termination arm that tells an operator which half of the apparatus to
                look at; the runner records one cleanly-excluded cell and moves on.
            ApparatusError: The rig failed while building it. One cell excluded under
                ``apparatus_failed``; the runner moves on.
        """
        ...  # pragma: no cover — protocol

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Run one prepared candidate against one case.

        A kind owns everything between these two calls — the turn loop or the single
        shot, its own teardown, and the mechanical tier it reports. What comes back is
        the only thing anything downstream sees on a cell that finishes; ``sink`` is what
        a cell cut off at its deadline is recorded from.

        Args:
            instance: Whatever :meth:`prepare` returned.
            test_case: The concrete stimulus — variation parameters and, for a kind that
                carries one, its own host payload.
            sink: This cell's sink, for what the kind is waiting on and how to read the
                spend it has billed so far. The cell's, like ``span_window``, not the kind's.

        Returns:
            The candidate's output, its judge-free facts and its telemetry.

        Raises:
            ApparatusError: The rig failed under the candidate. One cell excluded under
                ``apparatus_failed``, recorded with what the kind had reported through ``sink``;
                the runner moves on.
        """
        ...  # pragma: no cover — protocol


class CandidatePreparationFailed(Exception):
    """A candidate could not be built, and the cell is cleanly excluded rather than scored.

    Raised by :meth:`CandidateKind.prepare` and caught at the single dispatch site. It
    carries a termination rather than leaving the runner to infer one, because the arms
    are what an operator navigates by: a malformed seed, a subject factory that raised
    and a presumption that did not hold send them to three different places.

    No candidate turn ran, but the CELL did — the runner records every capture field as
    "measured nothing" rather than "not captured", which is honest precisely because
    nothing was spent on any of these paths.
    """

    def __init__(
        self,
        error: str,
        *,
        termination: CellTermination,
        preconditions: list[PreconditionOutcome] | None = None,
    ) -> None:
        """Record what failed, and which half of the apparatus it was.

        Args:
            error: The message, already prefixed with what kind of fault it was —
                it reaches an operator as ``EvalResult.runner_error`` verbatim.
            termination: Which arm this is.
            preconditions: The t=0 outcomes, for the arm that has them. ``None`` for the
                arms that do not, which is not an empty set of assertions but an absence
                of the question.
        """
        super().__init__(error)
        self.error = error
        self.termination: CellTermination = termination
        self.preconditions: list[PreconditionOutcome] = list(preconditions or [])


class CandidateKindDefect(ValueError):
    """A kind handed back output that contradicts its own :attr:`CandidateKind.judged_artifact`.

    A judged kind — transcript or document — with no :class:`JudgeEvidence` for a non-empty
    output, or an unjudged kind with some. Raised at the dispatch site before any judge call, and
    not degraded into an excluded cell, for :class:`UnknownCandidateKind`'s reason: it is the
    kind's code, so every cell of the run would do it, and the engine has nothing of its own to
    show a judge in its place.
    """

    def __init__(self, name: str, *, declared: JudgedArtifact, has_evidence: bool) -> None:
        """Name the kind, what it declared, and what its cell carried.

        Args:
            name: The template's ``candidate_kind``.
            declared: The kind's ``judged_artifact``.
            has_evidence: Whether the cell's output carried ``judge_evidence``.
        """
        carried = "carried judge evidence" if has_evidence else "carried no judge evidence"
        super().__init__(
            f"candidate kind {name!r} declares judged_artifact={declared.value!r} but a cell's output {carried}; "
            "a judged kind renders JudgeEvidence for every non-empty output, and an unjudged kind renders none"
        )
        self.name = name
        self.declared = declared
        self.has_evidence = has_evidence


class UnknownCandidateKind(ValueError):
    """A template names a kind this host did not wire.

    Raised at the dispatch site, and deliberately not degraded into an excluded cell:
    every cell of the run would fail identically, so the run is misconfigured rather
    than partly unmeasurable, and burning the matrix to record N copies of one
    configuration fault buys nothing an operator can use.
    """

    def __init__(self, name: str, *, known: list[str]) -> None:
        """Name the kind and what this host does have.

        Args:
            name: The unresolvable ``template.candidate_kind``.
            known: Every kind name this host can build, for the message.
        """
        super().__init__(
            f"no candidate kind {name!r} is wired on this host; known kinds: {', '.join(sorted(known)) or '(none)'}"
        )
        self.name = name
        self.known = list(known)
