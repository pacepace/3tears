"""The launch: refusing what cannot run, dispatching on the template's kind, and handing each arm's run to its job.

What every candidate kind's launch shares, as functions whose dependencies are parameters. A launch
loads its template, refuses what its arguments cannot honour, admits its runs, and dispatches each
arm to its kind's launcher, found in a registry the host supplies (:class:`LaunchableKind`) rather
than by name. The kind's launcher builds what only that kind has — its subject, its case set, its
clients — and hands the shared tail (:func:`launch_run`) a typed :class:`KindWiring`; the tail stamps
everything the request and template already say, assembles and bounds the run and gives its work
function to the :class:`LaunchGroup` the launch starts together. The universal battery (:func:`start_universal_battery`) and the judge-service
build (:func:`build_judge_service`) are here for the same reason: neither names a kind.

What only a host can answer arrives on the :class:`LaunchHost` every entrypoint here takes: the
:class:`~threetears.evals.contracts.host.eval_host.EvalHost` the rest of the engine reads (vocabulary,
storage, tracing, cell timeout, executor), composed with what starting runs needs — the registry of
kinds this host launches, its launch settings (read when the launch needs them rather than once at
construction, because they hot-reload), how a run is placed in its world, and the job manager it
builds over the host's own storage. Each kind's
launcher is the host's own code, because it composes this package with whatever builds that kind's
subject.

The scope a launch runs in is ``scope_id`` here — the engine's word for a partition it never
interprets. The host chooses what a scope is and names it on every launch: the template is read
there, and the launch's runs live there, because a run lives in its template's scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, Protocol, TypeVar, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.host.kinds import freeze
from threetears.evals.contracts.host.world import WorldPlacement

from threetears.evals.contracts.arguments import normalize_blank
from threetears.evals.contracts.errors import NotFoundError, ValidationFailedError
from threetears.evals.contracts.host.sweepables import CORE_SWEEPABLES
from threetears.evals.contracts.identity import derive_context_identity, variant_levers_of_run
from threetears.evals.contracts.models import (
    DEFAULT_LAUNCH_K_RUNS,
    ApparatusSettingValue,
    EvalRun,
    JudgedArtifact,
    ModelRoleOrigin,
    resolve_effective_judges,
    scored_dim_ids,
)
from threetears.evals.contracts.out_of_run import OutOfRunBudget
from threetears.evals.run.authoring import validated_kind_spec
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.contracts.cassettes import CassetteMode
from threetears.evals.run.jobs import MAX_CONCURRENT_JOBS, EvalJobManager, JobTimeoutFactory, adaptive_job_timeout_s
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.run.metering import MeteredCallLedger
from threetears.evals.run.offload import run_blocking, wait_through_cancellation
from threetears.evals.run.runner import DEFAULT_CELL_TIMEOUT_S, KindFactory, RunCallbacks, RunnerOptions, execute_run
from threetears.evals.run.simulator import SIMULATOR_REQUEST_SETTINGS
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.subject import SubjectSnapshot
    from threetears.evals.contracts.host.sweepables import SweepableRegistry
    from threetears.evals.contracts.models import EvalTemplate, EvalTestCase, JudgeConfig, VariationCounts
    from threetears.evals.contracts.scoring import CellSummary
    from threetears.evals.contracts.storage import DefinitionStore
    from threetears.evals.contracts.usage_capture import ExternalRateTable
    from threetears.evals.run.jobs import AdmissionTicket, WorkFn

log = get_logger(__name__)

#: Every cassette mode a launch accepts; ``'off'`` is also what a blank mode spells.
_CASSETTE_MODES: tuple[CassetteMode, ...] = get_args(CassetteMode)

_Formed = TypeVar("_Formed")
_Validated = TypeVar("_Validated", bound=BaseModel)


class LaunchSettings(BaseModel):
    """The host's launch settings, as one snapshot of values.

    Read through :attr:`LaunchHost.settings` at the moment a launch needs them — the arm and admission
    ceilings when a launch is admitted, the run ceilings and judge concurrency once per run when it is
    assembled — rather than once at construction, because a host's settings hot-reload. One snapshot
    per read, so the figures a run records and the objects that enforce them come from the same read
    and a reload cannot leave a run recording a ceiling it did not run under.

    Attributes:
        max_launch_arms: How many runs one launch may start together. A group starts every member
            at once in one job slot, so this is the concurrency one launch adds.
        max_admitted_runs: How many runs may be admitted and unfinished at once across every launch
            — the ceiling admission refuses past rather than queueing behind.
        judge_concurrency: How many judge calls one cell makes at once.
        enforcement_enabled: Whether the cost and metered-call ceilings are enforced at all.
        max_cost_usd: The run cost ceiling a run inherits when its launch names none.
        max_metered_calls: The metered-call ceiling a run inherits when its launch names none.
        max_out_of_run_cost_usd: The most a launch's out-of-run calls — its case generation, which runs
            before any run exists and so under no run's cap — may together be priced at before they are
            made (:class:`~threetears.evals.contracts.out_of_run.OutOfRunBudget`). Enforced exactly when
            ``enforcement_enabled`` is.
        setting_names: What the host calls each of the fields above, keyed by field name, so a
            refusal names the knob an operator turns. A field the host does not name here is
            called by its own name; the engine names no host setting of its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_launch_arms: int = Field(gt=0)
    max_admitted_runs: int = Field(gt=0)
    judge_concurrency: int = Field(gt=0)
    enforcement_enabled: bool
    max_cost_usd: float = Field(gt=0)
    max_metered_calls: int = Field(gt=0)
    max_out_of_run_cost_usd: float = Field(gt=0)
    setting_names: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_only_its_own_settings(self) -> LaunchSettings:
        """Refuse a name for a setting this snapshot does not have — a typo would never be read."""
        if unknown := sorted(set(self.setting_names) - (set(type(self).model_fields) - {"setting_names"})):
            raise ValueError(f"setting_names names settings a launch does not have: {', '.join(unknown)}")
        return self

    def name_of(self, setting: str) -> str:
        """What the host calls ``setting``, for a refusal to name.

        Args:
            setting: A field of this snapshot.

        Returns:
            The host's name for it, or the field's own name when the host gave none.
        """
        return self.setting_names.get(setting, setting)


@dataclass(frozen=True, kw_only=True)
class ArmPlan:
    """What one arm of a generating launch will run, as its kind says before it generates.

    A generating launch's cases do not exist until its kind's launcher has paid for them, so the
    engine asks the kind (:attr:`LaunchableKind.plan_arm`) what the arm will run before calling the
    launcher, prices that, and refuses an arm its cap cannot pay for before any call. The plan is a
    promise the launch tail holds the launcher to: a run freezing more cases than planned, or on
    another model, is refused, since it was priced as something it is not.

    Attributes:
        case_count: The most cases the arm will run — an upper bound, since generation de-duplicates
            and can keep fewer.
        candidate_model: The model the arm will run on: the one the launch named, or the kind's role
            default for an arm that named none.
    """

    case_count: int
    candidate_model: str

    def __post_init__(self) -> None:
        """Refuse a plan of no cases, or one naming no model.

        Raises:
            ValueError: ``case_count`` is below one, or ``candidate_model`` is blank.
        """
        if self.case_count < 1:
            raise ValueError(f"an arm plan runs at least one case; got case_count={self.case_count}")
        if not self.candidate_model.strip():
            raise ValueError("an arm plan names the model the arm runs on; candidate_model is blank")


@dataclass(frozen=True, kw_only=True)
class ArmQuote:
    """One arm of a generating launch, as the engine asks the host's pricer to price it.

    Attributes:
        scope_id: The scope the arm's run will live in, whose history a history pricer reads.
        template_id: The template the arm runs.
        subject_id: The subject it measures.
        candidate_model: The model it runs on, as its plan named it.
        k_runs: Repeats of every case.
        case_count: The cases it runs, as its plan bounded them.
        cassette_mode: Its cassette mode, normalised.
    """

    scope_id: str
    template_id: str
    subject_id: str
    candidate_model: str
    k_runs: int
    case_count: int
    cassette_mode: CassetteMode


@dataclass(frozen=True, kw_only=True)
class ArmPrice:
    """What the host's pricer predicts one arm will cost, and how it knows.

    Attributes:
        predicted_usd: The predicted cost of the arm's whole run, in dollars, or ``None`` when the pricer
            cannot predict it — unknown, which the launch never reads as $0.
        basis: How the prediction was made, or why there is none, in words a refusal quotes ("from 12
            past results, method usage-history").
    """

    predicted_usd: float | None
    basis: str

    def __post_init__(self) -> None:
        """Refuse a negative or non-finite prediction, and a blank basis.

        Raises:
            ValueError: ``predicted_usd`` is negative or not finite, or ``basis`` is blank.
        """
        if self.predicted_usd is not None and not (math.isfinite(self.predicted_usd) and self.predicted_usd >= 0):
            raise ValueError(f"a predicted cost is a finite amount, 0 or more; got {self.predicted_usd!r}")
        if not self.basis.strip():
            raise ValueError("an arm price says how it was made, or why there is none; basis is blank")


class LaunchPricer(Protocol):
    """The host's prediction of what one arm of a generating launch will cost, before anything is paid for.

    A port on :class:`LaunchHost` because the prediction is a read over history — an analysis — and the
    run package imports no analysis: the host composes one. ``threetears.evals.ops.history_launch_pricer``
    is the engine's own, over the scope's per-observation usage history; a host with a rate card of its
    own prices from that instead. Called off the event loop, through the host's blocking executor.
    """

    def __call__(self, quote: ArmQuote, /) -> ArmPrice:
        """Predict ``quote``'s cost.

        Args:
            quote: The arm.

        Returns:
            The prediction, or a price with no prediction and the reason there is none.
        """
        ...  # pragma: no cover — protocol


@dataclass(frozen=True, kw_only=True)
class LaunchHost:
    """An :class:`~threetears.evals.contracts.host.eval_host.EvalHost`, and what starting its runs needs.

    What :func:`start_run`, :func:`start_universal_battery` and the launch tail take. It COMPOSES the
    host rather than extending it: the host is built once and handed in whole, so no field of it can
    be dropped on the way, and the analysis side is handed the same :attr:`eval_host` this launches
    through.

    Attributes:
        eval_host: The host every run this launches belongs to — its vocabulary, its storage, its
            clients, its tracing, its executor and its cell timeout.
        kinds: The launch registry: every candidate kind this host launches, keyed by the name a
            template declares, with its launcher. A template naming a kind with no entry is refused
            before anything is read.
        settings: Reads the host's :class:`LaunchSettings` — a callable because they hot-reload.
        job_timeout_factory: What bounds each run's job, as
            :class:`~threetears.evals.run.jobs.EvalJobManager` takes it. No default, for the reason the
            host's cell timeout has none: :func:`~threetears.evals.run.jobs.default_job_timeout` is
            right for a host with no timeout layer of its own and silently wrong for one with.
        world_placements: What one assembled run did with every world dimension the host declares,
            stamped on the run before the identity that hashes it. Required exactly when the
            profile declares a world, and refused when it declares none — a placement over a world
            the host does not have describes nothing. A host with no world records ``{}``.
        launch_pricer: Predicts what each arm of a GENERATING launch will cost, so an arm its cap cannot
            pay for is refused before the launch's paid generation call (:class:`LaunchPricer`). ``None``
            for a host that prices no launch, whose generating launch under an enforced cap is refused
            saying so — never run unpriced.
        max_concurrent_jobs: How many runs' jobs execute at once in this process.
        on_job_progress: Called with ``(run id, progress)`` on every progress write — typically a
            broadcast to an operator's view — or ``None``.
        job_manager: The process's job manager, built here over :attr:`eval_host`'s storage and
            executor, so the store a run's status is written to is the store its results are. It is
            the one a host hands :func:`~threetears.evals.run.lifecycle.cancel_run`, the boot reclaim
            and its shutdown.
    """

    eval_host: EvalHost
    kinds: Mapping[str, LaunchableKind]
    settings: Callable[[], LaunchSettings]
    job_timeout_factory: JobTimeoutFactory
    world_placements: Callable[[EvalRun], dict[str, WorldPlacement]] | None = None
    launch_pricer: LaunchPricer | None = None
    max_concurrent_jobs: int = MAX_CONCURRENT_JOBS
    on_job_progress: Callable[[str, dict[str, Any]], None] | None = None
    job_manager: EvalJobManager = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a world placement that disagrees with the profile, and an apparatus setting it does not declare; build the jobs.

        Raises:
            ValueError: The profile declares a world and no placement was supplied, or declares
                none and one was; or a kind honours an apparatus setting the profile does not declare as
                one of the host's own apparatus dimensions.
        """
        profile = self.eval_host.profile
        settable = settable_apparatus(profile.sweepables)
        for kind, launchable in self.kinds.items():
            if undeclared := sorted(launchable.apparatus_settings - settable):
                raise ValueError(
                    f"kind {kind!r} honours apparatus setting(s) {', '.join(undeclared)}, which host "
                    f"{profile.host_id!r} does not declare as apparatus of its own; a launch sets only a "
                    f"host-declared apparatus dimension (declared: {', '.join(sorted(settable)) or 'none'}) — the "
                    "engine's own apparatus has launch arguments of its own"
                )
        if profile.world is not None and self.world_placements is None:
            raise ValueError(
                f"host {profile.host_id!r} declares a world, so a launch must stamp where each run sits in "
                "it: build the LaunchHost with world_placements=<the host's placement of one run>"
            )
        if profile.world is None and self.world_placements is not None:
            raise ValueError(
                f"host {profile.host_id!r} declares no world, so there is nothing to place a run in: "
                "drop world_placements, or declare the world on the profile"
            )
        object.__setattr__(
            self,
            "job_manager",
            EvalJobManager(
                self.eval_host.storage,
                self.max_concurrent_jobs,
                self.on_job_progress,
                job_timeout_factory=self.job_timeout_factory,
                blocking_executor=self.eval_host.blocking_executor,
            ),
        )

    def place(self, run: EvalRun) -> dict[str, WorldPlacement]:
        """Where ``run`` sits in this host's world: one placement per declared dimension.

        Args:
            run: The assembled run.

        Returns:
            The placements, or ``{}`` for a host with no world — the recording that the run placed
            nothing, which is a different fact from a run that recorded no placements at all.
        """
        return {} if self.world_placements is None else self.world_placements(run)


#: The engine's own apparatus dimensions, which have launch arguments of their own (the judge and
#: simulator pins, the cost ceiling) and so are never an apparatus SETTING.
_ENGINE_APPARATUS: frozenset[str] = frozenset(
    declared.name for declared in CORE_SWEEPABLES if declared.role == "apparatus"
)


def settable_apparatus(registry: SweepableRegistry) -> frozenset[str]:
    """The apparatus dimensions a launch may set: every one the host declares, but none of the engine's.

    Args:
        registry: The host's sweepable registry (``HostProfile.sweepables``).

    Returns:
        The names of the host's own ``apparatus`` declarations.
    """
    return frozenset(
        declared.name
        for declared in registry.declarations
        if declared.role == "apparatus" and declared.name not in _ENGINE_APPARATUS
    )


def _refuse_oversized_launch(n_runs: int, ceiling: int, setting: str) -> None:
    """Refuse a launch that would start more runs at once than one launch may.

    A group starts every member at once in a single job slot, so its run count is the concurrency the
    launch adds to the process — counted in runs, one per arm, because that is what executes
    concurrently; a model sweep crossed with settings multiplies.

    Args:
        n_runs: How many runs the launch would start.
        ceiling: The host's :attr:`LaunchSettings.max_launch_arms`.
        setting: The host's name for that setting.

    Raises:
        ValidationFailedError: ``n_runs`` exceeds ``ceiling``.
    """
    if n_runs > ceiling:
        raise ValidationFailedError(
            f"this launch would start {n_runs} arms (one run each: every model at every setting), more than one "
            f"launch may start together ({ceiling}, {setting}); launch fewer models or settings at "
            "once, or raise the ceiling"
        )


def _normalized_cassette_mode(cassette_mode: str | None) -> CassetteMode:
    """Read a launch's cassette mode, blank spelling ``'off'``, refusing anything that is not a mode.

    Blank is normalised through :func:`~threetears.evals.contracts.arguments.normalize_blank`, the helper
    the cost estimate reads this field through: it exists because two surfaces once disagreed about
    what blank meant, so a second spelling of that rule here would recreate the divergence.

    Args:
        cassette_mode: The mode the launch was given.

    Returns:
        ``'off'``, ``'capture'`` or ``'replay'``.

    Raises:
        ValidationFailedError: ``cassette_mode`` is not blank and not a mode.
    """
    mode = normalize_blank(cassette_mode, "off")
    for known in _CASSETTE_MODES:
        if mode == known:
            return known
    raise ValidationFailedError(
        f"cassette_mode={cassette_mode!r} is not a mode: expected 'off', 'capture' or 'replay'."
    )


def _refuse_a_corpus_the_mode_cannot_use(cassette_mode: CassetteMode, cassette_corpus_id: str | None) -> None:
    """Refuse a replay that names no corpus, and a corpus named for a launch that replays nothing.

    A replay serves the corpus one capture run recorded, and only the launch can say which: a
    replay with no corpus would have nothing to serve, and a corpus named beside another mode would
    be recorded as a condition nothing ran under. Read from the arguments alone, so it is refused
    before admission and before any read.

    Args:
        cassette_mode: The launch's cassette mode, normalised.
        cassette_corpus_id: The capture run whose corpus the launch replays, or ``None``.

    Raises:
        ValidationFailedError: A replay with no corpus, or a corpus with any other mode.
    """
    if cassette_mode == "replay" and cassette_corpus_id is None:
        raise ValidationFailedError(
            "cassette_mode='replay' names no corpus: pass cassette_corpus_id=<the id of the capture run whose "
            "recording this launch replays>"
        )
    if cassette_mode != "replay" and cassette_corpus_id is not None:
        raise ValidationFailedError(
            f"cassette_corpus_id={cassette_corpus_id!r} is a corpus to replay, and this launch's cassette_mode is "
            f"{cassette_mode!r}; name a corpus only with cassette_mode='replay'"
        )


def _refuse_a_corpus_that_cannot_serve(
    corpus_run: EvalRun | None, cassette_corpus_id: str, *, template: EvalTemplate, scope_id: str
) -> None:
    """Refuse a replay corpus that is not a capture of this template in this scope.

    Args:
        corpus_run: The run the corpus id names, read in the launch's scope, or ``None``.
        cassette_corpus_id: The id the launch named.
        template: The template the launch runs.
        scope_id: The launch's scope.

    Raises:
        ValidationFailedError: No run of that id in the scope (a missing run and one in another scope
            read alike, since a run is read only in its own scope), one that did not capture, or one
            that captured another template's cases.
    """
    if corpus_run is None:
        raise ValidationFailedError(
            f"cassette_corpus_id={cassette_corpus_id!r} names no run in scope {scope_id!r}; a replay serves the "
            "corpus of a capture run in its own scope"
        )
    if corpus_run.cassette_mode != "capture":
        raise ValidationFailedError(
            f"run {cassette_corpus_id!r} ran with cassette_mode={corpus_run.cassette_mode!r}, so it recorded no "
            "corpus to replay; name a capture run"
        )
    if corpus_run.template_id != template.id:
        raise ValidationFailedError(
            f"run {cassette_corpus_id!r} captured template {corpus_run.template_id!r}, and this launch runs "
            f"{template.id!r}; its corpus holds no recording of this template's cases"
        )


class LaunchGroup:
    """The runs of one launch, prepared together and started together — or not at all.

    **A run carries exactly one arm**, and an arm is a complete configuration with the candidate model
    among its settings, so a launch naming three models is three runs. Every launch is a group, of
    one run or of many: a run launched into it is fully prepared (every refusal made, every client
    built) but not started — :func:`launch_run` hands its work function here — and the launch then
    starts every member in one concurrency slot, or, when any member is refused, releases what the
    prepared ones built, so a refusal on the last arm leaves nothing running and nothing billed.

    **What sibling arms share is resolved once, here** (:meth:`resolve_once`): the subject snapshot,
    the case set and the judge. Resolving them per run would let two arms of one launch answer
    different generated questions, capture a live subject at two moments, or be scored by different
    judges — each a difference between arms that nobody chose, which the analysis would then have to
    report as a confound of the comparison the launch existed to make.

    **Two runs of one arm are refused before the group forms**: the launch names each model once
    (:func:`start_run` refuses a repeat), and every run of one launch carries the same overlays, so
    no two members can be one arm under two names.

    Attributes:
        id: The group's id, stamped on every member run.
        candidate_models: Every candidate model the launch names, across all its arms — the set the
            judge is chosen against, so a candidate that is also the default judge moves every
            sibling's judge rather than its own run's alone.
        members: The prepared runs, their work functions and their job timeouts.
    """

    def __init__(self, candidate_models: Sequence[str]) -> None:
        """Mint the group's id; it is stamped on every member run.

        Args:
            candidate_models: Every candidate model the launch names. Required, because the judge is
                chosen against it: a group built without it could hand the launch a judge that is one
                of its own candidates.
        """
        self.id = str(uuid.uuid7())
        self.candidate_models = list(candidate_models)
        self.members: list[tuple[EvalRun, WorkFn, float]] = []
        self._teardowns: list[contextlib.AsyncExitStack] = []
        self._resolved: dict[tuple[str, ...], Any] = {}

    async def resolve_once(self, key: tuple[str, ...], compute: Callable[[], Awaitable[Any]]) -> Any:
        """Resolve a launch-wide input the first time an arm asks for it, and hand every later arm the same.

        A computation that raises stores nothing, so the refusal reaches whichever arm asked, and the
        launch abandons as it does for any refusal.

        Args:
            key: What is being resolved, and every input it depends on.
            compute: Resolves it; awaited at most once per key.

        Returns:
            The resolved value.
        """
        if key not in self._resolved:
            self._resolved[key] = await compute()
        return self._resolved[key]

    def add(
        self,
        run: EvalRun,
        work: WorkFn,
        job_timeout_s: float,
        teardown: contextlib.AsyncExitStack,
    ) -> None:
        """Hold one prepared run, and what it must release if the group never starts."""
        self.members.append((run, work, job_timeout_s))
        self._teardowns.append(teardown)

    async def abandon(self) -> None:
        """Release every prepared member's clients; called when the group will not start.

        Called from inside the handler of the refusal that abandoned the group, so a teardown that
        fails is logged and the rest still run: raising here would leak every later member's clients
        and replace the refusal the caller is about to re-raise.
        """
        for teardown in self._teardowns:
            try:
                await teardown.aclose()
            # prawduct:allow prawduct/broad-except -- one member's failed release must not leak the others' or mask the launch refusal
            except Exception:
                log.exception("eval.launch_group group=%s a member's teardown failed while abandoning", self.id)
        self.members.clear()
        self._teardowns.clear()


@dataclass(frozen=True)
class LaunchRequest:
    """One arm's launch as :func:`start_run` resolved it, handed to the kind's own launcher.

    :func:`start_run` loads the template, validates what every kind shares — the overlays against the
    kind's model among it — and finds the kind's entry in the launch registry; the entry is the
    launcher, and the launcher reads what its kind honours from here. One shape for every launcher,
    so the dispatch is a lookup rather than a per-kind argument list. A launch argument the kind
    cannot honour never reaches it: the dispatch refuses the launch first
    (:attr:`LaunchableKind.unhonoured_launch_arguments`). The launcher hands this same request to
    :func:`launch_run`, which stamps everything it carries onto the run — the launcher restates none
    of it.

    Attributes:
        template: The template, loaded.
        kind: The kind the dispatch read off the template, carried to the launch tail for the run
            to record rather than read again. Not named ``candidate_kind``, because every attribute
            of that name is a reader the once-per-tier canary counts, and this is a carried value.
        subject_id: The launch's subject, as the caller named it.
        candidate_model: This arm's candidate model as the caller named it, or ``None`` when the caller
            named none and the kind has a role default to run on.
        k_runs: Repeats per case.
        scope_id: The scope the template was read in, which the launch's runs live in.
        n_variations: New cases to generate.
        variation_model: The model that writes the template's ``llm`` variation axes' values when the
            launch generates (``n_variations`` > 0), as the caller named it. The dispatch has already
            made it present exactly when the template has an ``llm`` axis and the launch generates, so
            ``None`` means no model writes this launch's cases. A launcher builds the writer through its
            host's client factory in the ``variation`` role (never the simulator's — a kind with no
            simulated user still generates) and hands it to
            :func:`~threetears.evals.gen.generate_variations`, whose counts record the model it resolved
            to and which :func:`launch_run` checks against this pin. Generation runs before the run
            exists, once per launch for every arm (:meth:`LaunchGroup.resolve_once`), so its calls are
            outside every run's cost cap and metered-call ceiling.
        judge_model: The run-level judge pin, unresolved.
        judge_config_ids: The per-dim judge-configuration selection.
        simulator_model: The simulator pin, unresolved.
        cassette_mode: The cassette mode, already normalised to ``'off'``, ``'capture'`` or ``'replay'``.
        cassette_corpus_id: For a replay, the capture run whose corpus it serves, checked to be a capture
            of this template in this scope; ``None`` for any other mode.
        overlays: The launch's overlays as the kind's model validated them, every field resolved; ``None``
            for a kind that declares no overlays. Read it through :meth:`overlays_as`. The run records
            exactly this model's JSON form.
        kind_spec: The template's kind spec as the kind's spec model validated it at this launch; ``None``
            for a kind that declares no spec. Read it through :meth:`kind_spec_as`. The run records
            exactly this model's JSON form.
        max_cost_usd: The per-run cost-cap override, already checked positive.
        max_metered_calls: The per-run metered-call ceiling override, already checked positive.
        apparatus_settings: The host-declared apparatus values this launch sets, each one the kind
            declares it honours (:attr:`LaunchableKind.apparatus_settings`) and already validated; ``{}``
            when it sets none. The launcher sets its rig up from them, and the run records exactly
            these (``EvalRun.apparatus_settings``).
        generation_budget: For a launch that generates (``n_variations`` > 0), the out-of-run budget its
            generation's calls are priced against and ledgered through — one for the whole launch, shared
            by every arm, capped at the host's ``max_out_of_run_cost_usd``. A launcher hands it to
            :func:`~threetears.evals.gen.generate_variations` as ``budget``; a generation a model wrote
            whose calls this budget never ledgered is refused at the launch tail. ``None`` for a launch
            that generates nothing.
        arm_plan: For a launch that generates, what the kind planned this arm to run
            (:attr:`LaunchableKind.plan_arm`) — the plan the arm was priced at, and the launch tail holds
            its run to. ``None`` for a launch that generates nothing.
        launch_group: The launch this run is prepared into; the launch starts it with its siblings.
    """

    template: EvalTemplate
    kind: str
    subject_id: str
    candidate_model: str | None
    k_runs: int
    scope_id: str
    n_variations: int
    variation_model: str | None
    judge_model: str | None
    judge_config_ids: dict[str, str] | None
    simulator_model: str | None
    cassette_mode: CassetteMode
    cassette_corpus_id: str | None
    overlays: BaseModel | None
    kind_spec: BaseModel | None
    max_cost_usd: float | None
    max_metered_calls: int | None
    apparatus_settings: Mapping[str, ApparatusSettingValue]
    generation_budget: OutOfRunBudget | None
    arm_plan: ArmPlan | None
    launch_group: LaunchGroup

    def overlays_as(self, model: type[_Validated]) -> _Validated:
        """The launch's overlays as the kind's own overlay model, typed.

        Args:
            model: The overlay model the kind's contract declares.

        Returns:
            The validated overlays.

        Raises:
            TypeError: The kind's contract declares another model, or none — a launcher reading a
                model its kind does not declare, which no caller input can cause.
        """
        return _typed(self.kind, "overlays", self.overlays, model)

    def kind_spec_as(self, model: type[_Validated]) -> _Validated:
        """The template's kind spec as the kind's own spec model, typed.

        Args:
            model: The spec model the kind's contract declares.

        Returns:
            The validated spec.

        Raises:
            TypeError: The kind's contract declares another model, or none.
        """
        return _typed(self.kind, "kind spec", self.kind_spec, model)


def _typed(kind: str, what: str, validated: BaseModel | None, model: type[_Validated]) -> _Validated:
    """``validated`` as ``model``, or a refusal naming both.

    Args:
        kind: The kind, for the refusal.
        what: What ``validated`` is, for the refusal.
        validated: The value the dispatch validated.
        model: The model the launcher reads it as.

    Returns:
        ``validated``, typed.

    Raises:
        TypeError: ``validated`` is not a ``model``.
    """
    if not isinstance(validated, model):
        raise TypeError(
            f"kind {kind!r}'s {what} are {type(validated).__name__}, not {model.__name__}; a launcher reads the "
            "model its kind's contract declares on the host profile"
        )
    return validated


#: One kind's launcher: builds that kind's collaborators for one arm and hands :func:`launch_run` its run.
KindLauncher = Callable[[LaunchRequest], Awaitable[EvalRun]]

#: A launch argument a kind may declare it cannot honour. A launch naming one for such a kind is
#: refused at the dispatch, before anything is read or built, rather than handed to a launcher that
#: might ignore it — a cassette mode ignored is a requested replay run live, and paid for.
LaunchArgument = Literal["n_variations", "judge_model", "judge_config_ids", "simulator_model", "cassette_mode"]


@dataclass(frozen=True)
class LaunchableKind:
    """One entry of the launch registry a host hands :func:`start_run`: how a kind launches, and what it refuses.

    A kind with no entry is refused at the dispatch (:func:`no_launcher_for`), before anything is
    read or built, so a kind the runner could execute but no host can build collaborators for never
    reaches a cell.

    What a launch may turn on a kind's runs is not here: it is the kind's overlay model, declared on
    the host profile (:attr:`~threetears.evals.contracts.host.profile.HostProfile.kinds`), and the
    dispatch refuses an overlay that model refuses.

    Attributes:
        launch: The kind's launcher.
        unhonoured_launch_arguments: The engine's launch arguments a run of this kind cannot honour; a
            launch supplying one is refused at the dispatch, naming it. Empty for a kind that honours
            every one.
        plan_arm: What one arm of a GENERATING launch of this kind will run, asked before the launcher
            is — the arm's case count and model (:class:`ArmPlan`) — so the engine can price the arm and
            refuse one its cap cannot pay for before the launch's paid generation call. Required of a kind
            that generates: a generating launch of a kind with none is refused at the dispatch. ``None``
            for a kind that declines ``n_variations``, and refused beside it, since nothing would ask it.
        apparatus_settings: The host-declared apparatus dimensions this kind's launcher sets its rig up
            from, read off :attr:`LaunchRequest.apparatus_settings`. A launch setting any other is
            refused at the dispatch; the :class:`LaunchHost` refuses a name the profile does not declare as
            one of the host's own apparatus dimensions. Empty for a kind whose rig a launch cannot set.
    """

    launch: KindLauncher
    unhonoured_launch_arguments: frozenset[LaunchArgument] = frozenset()
    plan_arm: Callable[[LaunchRequest], ArmPlan] | None = None
    apparatus_settings: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Refuse a name that is not a launch argument, and an arm plan for a kind that never generates.

        Raises:
            ValueError: An entry of ``unhonoured_launch_arguments`` is not a :data:`LaunchArgument`, or
                ``plan_arm`` is supplied for a kind that declines ``n_variations``.
        """
        if unknown := sorted(set(self.unhonoured_launch_arguments) - set(get_args(LaunchArgument))):
            raise ValueError(
                f"unhonoured_launch_arguments names {', '.join(unknown)}, which are not launch arguments; "
                f"the arguments a kind can decline are {', '.join(get_args(LaunchArgument))}"
            )
        if self.plan_arm is not None and "n_variations" in self.unhonoured_launch_arguments:
            raise ValueError(
                "plan_arm plans the arms of a generating launch, and this kind declines n_variations, so nothing "
                "would ever ask it; drop plan_arm, or honour n_variations"
            )


@dataclass(frozen=True, kw_only=True)
class RunJudge:
    """A judged run's judge, as :func:`build_judge_service` resolved it from one config load.

    Handed to :func:`launch_run` on :attr:`KindWiring.judge`, which stamps the run's judge pin, its
    per-dim attribution and the config set from here — so what the run records and what scores it
    are one value.

    Attributes:
        service: The judge service the runner drives.
        model: The run-level judge pin, resolved.
        selection: The configs the launch named, ``{dim_id: config_id}``; empty when it named none.
        effective_judges: The model each scored dim is requested from.
        configs: The configs the service was built from, ``{dim_id: config}``, over the dims that have one.
    """

    service: JudgeService
    model: str
    selection: dict[str, str]
    effective_judges: dict[str, str]
    configs: dict[str, JudgeConfig]

    @property
    def config_ids(self) -> dict[str, str]:
        """The config set the run commits to, ``{dim_id: config_id}``."""
        return {dim_id: config.id for dim_id, config in self.configs.items()}

    @property
    def config_provenance(self) -> dict[str, ModelRoleOrigin]:
        """How each config was arrived at: ``chosen`` when the launch named it, ``inherited`` from the active lookup."""
        return {dim_id: "chosen" if dim_id in self.selection else "inherited" for dim_id in self.configs}


@dataclass(frozen=True, kw_only=True)
class KindWiring:
    """What one kind's launcher resolved for its arm — everything :func:`launch_run` cannot know itself.

    Everything the request and the template already say — the scope, the template, the kind, the
    candidate model the launch named, the repeats, the cassette mode, the overlays and spec, the
    world seed and tool bound the template states, the ceilings — the launch tail stamps from them.
    What is here is the kind's: the candidate it built, the subject it captured, the cases it froze,
    and the apparatus it resolved. Every field the request also speaks to is checked against it.

    Attributes:
        kind_factory: Builds the kind for each cell, with the collaborators the launcher bound.
        subject: The subject the launcher captured for the request's ``subject_id``.
        test_cases: The frozen case set, in the order the runner walks it — every case in the
            request's scope and of its template.
        default_candidate_model: The model the kind ran on for an arm that named none — its role
            default. ``None`` when the request named one, which the run then records.
        judge: The run's judge, from :func:`build_judge_service`, for a judged kind; ``None`` for a
            kind whose grade is mechanical.
        simulator_model: The model that drives the simulated user, resolved, for a kind that runs
            one; ``None`` for a kind with none.
        payload: The host's opaque payload for the run, stored as its ``host_payload`` and never read.
        variation_counts: How many cases generation was asked for and froze, and the model that wrote
            the ``llm`` axes' values, for a launch that generated — the counts
            :func:`~threetears.evals.gen.generate_variations` returned; ``None`` for one that reused
            stored cases.
        turn_budget_s: The per-turn budget the kind runs its candidate's turns under, or ``None``.
        cell_timeout_s: A per-cell ceiling for a kind whose work legitimately outlasts the default,
            or ``None`` for the default.
        external_rates: The rate table for the run's counted external calls, or ``None`` to leave
            them counted and unpriced.
        teardown: What the run owns and must release when it ends, or ``None`` for nothing. Owned by
            the launch tail from the moment it is handed over.
    """

    kind_factory: KindFactory
    subject: SubjectSnapshot
    test_cases: Sequence[EvalTestCase]
    default_candidate_model: str | None = None
    judge: RunJudge | None = None
    simulator_model: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    variation_counts: VariationCounts | None = None
    turn_budget_s: float | None = None
    cell_timeout_s: float | None = None
    external_rates: ExternalRateTable | None = None
    teardown: contextlib.AsyncExitStack | None = None


def no_launcher_for(template_id: str, candidate_kind: str, launchable: Mapping[str, Any]) -> ValidationFailedError:
    """The refusal for a template whose kind this host cannot launch — one wording, wherever it fires.

    Args:
        template_id: The template.
        candidate_kind: The kind it declares.
        launchable: The kinds the host launches, keyed by kind, in the order the refusal lists them.

    Returns:
        The error to raise.
    """
    return ValidationFailedError(
        f"template {template_id!r} declares candidate_kind={candidate_kind!r}, which this "
        f"host has no launcher for; launchable kinds: {', '.join(launchable)}. "
        "A kind the runner can be wired with still needs a launch that builds its collaborators, "
        "because the runner cannot build a client it has never heard of."
    )


def _unhonoured_launch_arguments(
    unhonoured: frozenset[LaunchArgument],
    *,
    n_variations: int,
    judge_model: str | None,
    judge_config_ids: dict[str, str] | None,
    simulator_model: str | None,
    cassette_mode: str,
) -> dict[LaunchArgument, Any]:
    """Collect the launch arguments a launch SUPPLIED that its kind cannot honour.

    Args:
        unhonoured: The kind's :attr:`LaunchableKind.unhonoured_launch_arguments`.
        n_variations: New cases the launch asked to generate.
        judge_model: The launch's judge pin.
        judge_config_ids: The launch's per-dim judge-configuration selection.
        simulator_model: The launch's simulator pin.
        cassette_mode: The launch's cassette mode, already normalised by
            :func:`_normalized_cassette_mode`; ``'off'`` is the absence of one and so is not reported.

    Returns:
        ``{argument: the value the launch supplied}`` for each supplied argument the kind cannot
        honour, and ``{}`` when there is none. The values are carried so a refusal can echo what it
        is refusing rather than only naming the field.
    """
    supplied: dict[LaunchArgument, Any] = {
        "n_variations": n_variations or None,
        "judge_model": judge_model,
        "judge_config_ids": judge_config_ids,
        "simulator_model": simulator_model,
        # Blank has already been read as ``'off'``, which spells "no cassette".
        "cassette_mode": None if cassette_mode == "off" else cassette_mode,
    }
    return {name: value for name, value in supplied.items() if value and name in unhonoured}


def _llm_axes(template: EvalTemplate) -> list[str]:
    """The template's variation axes whose values a model writes, by name.

    Args:
        template: The template.

    Returns:
        The names of its ``llm``-generated axes, in declaration order; empty when none is.
    """
    return [axis.name for axis in template.variation_axes if axis.generator == "llm"]


def _refuse_a_variation_model_the_launch_cannot_use(
    template: EvalTemplate, *, n_variations: int, variation_model: str | None
) -> None:
    """Refuse a generation that needs a model and names none, and a model nothing would call.

    A template's ``llm`` axes are written by a model, so a launch that generates for one must say
    which: no other launch argument names it, and borrowing the simulator's would record a
    simulated user on a kind that has none. A model named for a launch that generates nothing, or
    for a template with no ``llm`` axis, would be recorded as the writer of cases it never wrote —
    the same unhonoured argument :attr:`LaunchableKind.unhonoured_launch_arguments` refuses.
    Read from the arguments and the template alone, so it is refused before any arm is prepared
    and before a single generation call is paid for.

    Args:
        template: The loaded template.
        n_variations: New cases the launch asked to generate, already checked non-negative.
        variation_model: The model the launch named to write the ``llm`` axes' values, or ``None``.

    Raises:
        ValidationFailedError: The launch generates for a template with an ``llm`` axis and names
            no variation model, or names one while generating nothing or for a template with no
            ``llm`` axis.
    """
    llm_axes = _llm_axes(template)
    if variation_model is None:
        if n_variations > 0 and llm_axes:
            raise ValidationFailedError(
                f"template {template.id!r} has its {', '.join(repr(name) for name in llm_axes)} axis values written "
                f"by a model, and this launch asks for {n_variations} generated case(s) without naming one: pass "
                "variation_model=<the model that writes them>. It is its own role (the host's 'variation' client, "
                "never the simulator's), and the runs record it beside their generation counts"
            )
        return
    if n_variations == 0:
        raise ValidationFailedError(
            f"variation_model={variation_model!r} names the model that writes generated cases, and this launch "
            "generates none (n_variations=0); launch it without variation_model, or ask for n_variations"
        )
    if not llm_axes:
        raise ValidationFailedError(
            f"variation_model={variation_model!r} names the model that writes a template's llm-generated axis "
            f"values, and template {template.id!r} has no llm axis, so no model writes its cases; launch it "
            "without variation_model"
        )


#: Validates a launch's apparatus settings as the run will store them.
_APPARATUS_SETTINGS = TypeAdapter(dict[str, ApparatusSettingValue])


def _validated_apparatus_settings(
    template: EvalTemplate, launchable: LaunchableKind, apparatus_settings: Mapping[str, Any] | None
) -> dict[str, ApparatusSettingValue]:
    """Refuse an apparatus setting the template's kind does not honour, and a value the run could not store.

    Args:
        template: The loaded template, whose kind is the one asked.
        launchable: That kind's registry entry, which names the settings its launcher reads.
        apparatus_settings: The launch's settings, unvalidated, or ``None`` for none.

    Returns:
        The settings as the run stores them; ``{}`` when the launch set none.

    Raises:
        ValidationFailedError: A setting the kind does not honour, or a value that is not a string, a bool
            or a finite number.
    """
    if not apparatus_settings:
        return {}
    if unhonoured := sorted(set(apparatus_settings) - launchable.apparatus_settings):
        honoured = ", ".join(sorted(launchable.apparatus_settings)) or "none"
        raise ValidationFailedError(
            f"template {template.id!r} is a {template.candidate_kind!r} template, and that kind's launcher sets its rig "
            f"up from no apparatus setting named {', '.join(repr(name) for name in unhonoured)} (it reads: {honoured}); "
            "a setting nothing reads would be recorded as a rig nobody built"
        )
    try:
        return _APPARATUS_SETTINGS.validate_python(dict(apparatus_settings))
    except ValidationError as e:
        raise ValidationFailedError(
            f"invalid apparatus_settings: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})"
        ) from e


class _Launchable(NamedTuple):
    """What the refusals over a template and its kind resolved.

    Attributes:
        kind: The kind the template declares, read once.
        launchable: Its registry entry.
        overlays: The launch's overlays, validated by the kind's model.
        kind_spec: The template's kind spec, validated by the kind's spec model.
        apparatus_settings: The launch's apparatus settings, validated against the kind.
    """

    kind: str
    launchable: LaunchableKind
    overlays: BaseModel | None
    kind_spec: BaseModel | None
    apparatus_settings: dict[str, ApparatusSettingValue]


def _launchable(
    host: LaunchHost,
    template: EvalTemplate,
    *,
    n_variations: int,
    variation_model: str | None,
    judge_model: str | None,
    judge_config_ids: dict[str, str] | None,
    simulator_model: str | None,
    cassette_mode: str,
    overlays: Mapping[str, Any] | None,
    apparatus_settings: Mapping[str, Any] | None,
) -> _Launchable:
    """Make every refusal that reads the template and its kind's registry entry — the dispatch's, and the battery's pre-flight.

    One function for both callers, so a battery refuses before launching anything exactly what each
    of its launches would refuse once it had begun.

    Args:
        host: The host, whose launch registry and profile declare the kind.
        template: The loaded template.
        n_variations: New cases the launch asked to generate.
        variation_model: The model the launch named to write the template's ``llm`` axes' values.
        judge_model: The launch's judge pin.
        judge_config_ids: The launch's per-dim judge-configuration selection.
        simulator_model: The launch's simulator pin.
        cassette_mode: The launch's cassette mode, already normalised.
        overlays: The launch's overlays, unvalidated.
        apparatus_settings: The launch's apparatus settings, unvalidated.

    Returns:
        The kind the template declares, read once; its registry entry; the overlays and the template's
        kind spec as the kind's models validated them; and the apparatus settings as the run stores them.

    Raises:
        ValidationFailedError: The template names a kind with no launcher, the launch supplies an
            argument the kind cannot honour, a negative ``n_variations``, a variation model the
            generation needs and the launch does not name or one nothing would call, a generating launch
            of a kind that plans no arm, an apparatus setting the kind does not honour, or the kind's
            models refuse the overlays or the spec.
    """
    if n_variations < 0:
        # Every surface reaches here; one without a wire-level bound would otherwise read a negative as
        # "generate nothing" and run the stored cases while the caller believed it asked for new ones.
        raise ValidationFailedError(f"n_variations must be 0 (reuse the stored cases) or more; got {n_variations}")
    candidate_kind = template.candidate_kind
    launchable = host.kinds.get(candidate_kind)
    if launchable is None:
        raise no_launcher_for(template.id, candidate_kind, host.kinds)
    # Refused here rather than handed to the kind's launcher to refuse: a launcher that ignored one
    # would run a requested replay live, or record a judge nobody scored with.
    if unusable := _unhonoured_launch_arguments(
        launchable.unhonoured_launch_arguments,
        n_variations=n_variations,
        judge_model=judge_model,
        judge_config_ids=judge_config_ids,
        simulator_model=simulator_model,
        cassette_mode=cassette_mode,
    ):
        raise ValidationFailedError(
            f"template {template.id!r} is a {candidate_kind!r} template, and that kind cannot honour "
            + ", ".join(f"{name}={value!r}" for name, value in unusable.items())
            + "; launch it without them"
        )
    _refuse_a_variation_model_the_launch_cannot_use(
        template, n_variations=n_variations, variation_model=variation_model
    )
    # A generating launch's arms are priced before its generation is paid for, from what the kind plans
    # each to run — so a kind that cannot say is refused here, before anything is read or built.
    if n_variations > 0 and launchable.plan_arm is None:
        raise ValidationFailedError(
            f"template {template.id!r} is a {candidate_kind!r} template, and that kind plans no arm of a generating "
            f"launch (LaunchableKind.plan_arm), so this launch's {n_variations} generated case(s) cannot be priced "
            "before the generation is paid for; launch it with n_variations=0, or declare the kind's plan_arm"
        )
    settings = _validated_apparatus_settings(template, launchable, apparatus_settings)
    # What the launch may turn, and what the template states for its kind, are the kind's models'
    # answers, made here — before any arm is prepared, so a refusal leaves nothing built and no run
    # created. The spec was validated when the template was authored; it is validated again because
    # the kind's model is code and may have moved since, and a run freezes what it validates now.
    validated = host.eval_host.profile.kind_contract(candidate_kind).validate_overlays(overlays)
    kind_spec = validated_kind_spec(template, profile=host.eval_host.profile)
    return _Launchable(candidate_kind, launchable, validated, kind_spec, settings)


class _Dispatched(NamedTuple):
    """What a launch's dispatch resolved: the template, the kind it declares, and how that kind launches.

    Attributes:
        template: The template, loaded.
        kind: The kind it declares, read once.
        launchable: The kind's registry entry: its launcher, and its arm plan.
        overlays: The launch's overlays, validated by the kind's model.
        kind_spec: The template's kind spec, validated by the kind's spec model.
        apparatus_settings: The launch's apparatus settings, validated against the kind.
    """

    template: EvalTemplate
    kind: str
    launchable: LaunchableKind
    overlays: BaseModel | None
    kind_spec: BaseModel | None
    apparatus_settings: dict[str, ApparatusSettingValue]


async def _dispatch(
    host: LaunchHost,
    template_id: str,
    scope_id: str,
    *,
    models: list[str],
    n_variations: int,
    variation_model: str | None,
    judge_model: str | None,
    judge_config_ids: dict[str, str] | None,
    simulator_model: str | None,
    cassette_mode: CassetteMode,
    cassette_corpus_id: str | None,
    overlays: Mapping[str, Any] | None,
    apparatus_settings: Mapping[str, Any] | None,
) -> _Dispatched:
    """Load a launch's template, find its kind's launcher, and make the refusals that read them.

    Args:
        host: The host, whose launch registry names the kind's launcher.
        template_id: The template to run.
        scope_id: The scope the template is read in.
        models: The launch's candidate models.
        n_variations: New cases the launch asked to generate.
        variation_model: The model the launch named to write the template's ``llm`` axes' values.
        judge_model: The launch's judge pin.
        judge_config_ids: The launch's per-dim judge-configuration selection.
        simulator_model: The launch's simulator pin.
        cassette_mode: The launch's cassette mode, already normalised.
        cassette_corpus_id: The capture run a replay serves, already checked against the mode.
        overlays: The launch's overlays, unvalidated.
        apparatus_settings: The launch's apparatus settings, unvalidated.

    Returns:
        The template, its kind, that kind's registry entry, the overlays and the template's kind spec as
        the kind's models validated them, and the apparatus settings validated against the kind.

    Raises:
        NotFoundError: The template is not found.
        ValidationFailedError: The template names a kind with no launcher, the launch supplies an
            argument the kind cannot honour, an overlay the kind's model refuses, a template kind spec
            the kind's spec model refuses, a negative ``n_variations``, a variation model the generation
            needs and the launch does not name or one nothing would call, a generating launch of a kind
            that plans no arm, an apparatus setting the kind does not honour, a model named twice, or a
            replay corpus that is not a capture of this template in this scope.
    """
    eval_host = host.eval_host
    template = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_template, template_id, scope_id)
    if template is None:
        raise NotFoundError("template", template_id)

    # THE LAUNCH DISPATCH — ``template.candidate_kind`` is read HERE, once per tier: the
    # tiers and their one reader each are declared by
    # ``test_classifier_launch.py::test_a_templates_candidate_kind_is_read_once_per_tier``,
    # which is the list to read rather than one kept here. This tier exists beside the
    # runner's own dispatch because the runner cannot build a
    # collaborator it has never heard of: a kind's clients are assembled by the host,
    # at the launch, and wired onto ``RunnerOptions.candidate_kinds``. Read ONCE here
    # and carried as a value for the same reason the runner reads it once — a second
    # read in the launch is a second place that decides what a subject is.
    #
    # It is ahead of every collaborator build because each kind's launcher builds only its
    # own: a conversational candidate needs a simulator client, a judge service, a candidate
    # factory and a resolved world, and a single-shot kind needs none of them. Building them
    # anyway would charge an operator for apparatus their run cannot use — and, for the
    # judge, would leave a code-graded run carrying a resolved judge model that scored nothing.
    #
    # An unknown kind is refused HERE rather than left to the runner. The runner's own
    # ``UnknownCandidateKind`` is raised at the dispatch inside the first cell, by
    # which point the subject is snapshotted, the clients are built and the run is
    # persisted — and every cell would fail identically, so the run is misconfigured
    # rather than partly unmeasurable.
    resolved = _launchable(
        host,
        template,
        n_variations=n_variations,
        variation_model=variation_model,
        judge_model=judge_model,
        judge_config_ids=judge_config_ids,
        simulator_model=simulator_model,
        cassette_mode=cassette_mode,
        overlays=overlays,
        apparatus_settings=apparatus_settings,
    )
    if cassette_corpus_id is not None:
        corpus_run = await run_blocking(
            eval_host.blocking_executor, eval_host.storage.load_eval_run, cassette_corpus_id, scope_id
        )
        _refuse_a_corpus_that_cannot_serve(corpus_run, cassette_corpus_id, template=template, scope_id=scope_id)
    if repeated := sorted({model for model in models if models.count(model) > 1}):
        raise ValidationFailedError(
            f"{', '.join(repr(model) for model in repeated)} named more than once; each model is one arm, so "
            "a second mention would measure that arm twice — raise k_runs for more repeats"
        )
    # The registry entry is the launcher, so a kind the host registered is a kind this launch
    # dispatches — there is no chain of kind comparisons here for a new kind to be missing from.
    return _Dispatched(
        template,
        resolved.kind,
        resolved.launchable,
        resolved.overlays,
        resolved.kind_spec,
        resolved.apparatus_settings,
    )


async def start_run(
    host: LaunchHost,
    *,
    template_id: str,
    subject_id: str,
    models: list[str],
    k_runs: int = DEFAULT_LAUNCH_K_RUNS,
    n_variations: int = 0,
    variation_model: str | None = None,
    judge_model: str | None = None,
    judge_config_ids: dict[str, str] | None = None,
    simulator_model: str | None = None,
    cassette_mode: str | None = "off",
    cassette_corpus_id: str | None = None,
    overlays: Mapping[str, Any] | None = None,
    apparatus_settings: Mapping[str, Any] | None = None,
    max_cost_usd: float | None = None,
    max_metered_calls: int | None = None,
    scope_id: str,
    launch_group: LaunchGroup | None = None,
    admission: AdmissionTicket | None = None,
) -> list[EvalRun]:
    """Refuse what no kind can run, admit the launch, and dispatch each arm to its kind's launcher.

    **A run carries exactly one arm**, and the candidate model is one of an arm's settings, so naming
    several models launches several runs: one :class:`LaunchGroup`, one run per model, every run
    prepared before any starts and all of them started together in one concurrency slot. What the
    arms share is resolved once for the group (:meth:`LaunchGroup.resolve_once`), so they answer the
    same questions under the same instruments and differ only in the model.

    **The kind arrives on the TEMPLATE, and this function dispatches on it once.** A template
    declares ``candidate_kind`` and the runner reads that same field at the top of every cell, so a
    launch parameter would be a second source of truth for one fact. What the launch adds is the
    collaborators: the runner cannot build a client it has never heard of, so a kind's launcher —
    ``host.kinds[kind].launch`` — builds its kind factory and hands it to :func:`launch_run` on its
    :class:`KindWiring`.

    Args:
        host: The host: its storage, its launch registry of kinds, its settings and its job manager.
        template_id: Template to run.
        subject_id: The subject the launch measures, as the host names it.
        models: Candidate models, one arm and one run each. A kind with a role default of its own
            runs one arm on that default when this is empty, and a kind with none refuses it.
        k_runs: Repeats per case for ``pass^k``.
        n_variations: New test cases to generate (0 = reuse existing).
        variation_model: The model that writes the template's ``llm`` variation axes' values —
            required when ``n_variations`` > 0 and the template has an ``llm`` axis, and refused
            otherwise. Asked of the host in the ``variation`` role and recorded on each run's
            ``variation_counts``. Its calls run before any run starts, once for every arm, and are
            outside every run's cost cap and metered-call ceiling.
        judge_model: The judge pin, unresolved.
        judge_config_ids: Optional per-dim judge-configuration selection, ``{dim_id: config_id}``.
        simulator_model: The simulator pin, unresolved.
        cassette_mode: ``'off'`` (the default, and what blank spells), ``'capture'`` or ``'replay'``.
        cassette_corpus_id: For ``'replay'``, and only for it: the id of the capture run whose corpus
            the launch replays — a capture of this template in ``scope_id``.
        overlays: The knobs this launch turns on its runs, by field of the kind's overlay model
            (:attr:`~threetears.evals.contracts.host.profile.HostProfile.kinds`). Validated before any arm is
            prepared; every run records the validated model, defaults included.
        apparatus_settings: Host-declared apparatus values this launch sets its runs' rig up with, by
            apparatus dimension — each one the kind declares it honours
            (:attr:`LaunchableKind.apparatus_settings`), valued a string, a bool or a finite number. So one
            template can be run at two values of, say, an adjudicator's seat and the runs compared. Every
            run records them (``EvalRun.apparatus_settings``), and they are part of its measurement context.
        max_cost_usd: Optional per-run cost-cap override; must be ``> 0``.
        max_metered_calls: Optional per-run metered-call ceiling override; must be ``> 0``.
        scope_id: The scope the template is read in and the launch's runs live in.
        launch_group: Prepare the runs into this launch instead of starting them; the caller starts
            the group once every arm is prepared. When omitted, this call is the whole launch: it
            forms the group and starts it through :func:`launch_as_group`.
        admission: A reservation the caller already holds, of which this launch takes its runs'
            share instead of asking for room of its own. Ignored with ``launch_group``, whose
            caller's admission covers it. When omitted, a launch that owns its group is admitted here.

    **A launch that generates is priced before it pays for anything.** Its generation runs inside the
    kind's launcher and before any run exists, so before calling the launcher this asks the kind what
    each arm will run (:attr:`LaunchableKind.plan_arm`), asks the host's pricer what that will cost
    (:attr:`LaunchHost.launch_pricer`) and refuses an arm predicted above the cap its run will be held
    to — or one nothing can predict whose cap the run would merely inherit, since unknown is not $0 and
    an inherited cap is nobody's decision about this run (a cap the launch named is that decision, and
    the run goes ahead under it). The generation's own calls are priced in turn before the first is
    made, against the host's out-of-run cap (:attr:`LaunchRequest.generation_budget`).

    Returns:
        The persisted runs, one per model in the order given (status ``pending``).

    Raises:
        NotFoundError: The template is not found.
        AdmissionRefusedError: The runs would pass the host's admission ceiling — raised before the
            template is read, so the refused launch prepared nothing.
        ValidationFailedError: A model named twice, more runs than one launch may start, a negative
            ``n_variations``, a variation model the generation needs and the launch does not name or one
            nothing would call, a ``k_runs`` outside the run's bounds, a non-positive ``max_cost_usd``
            or ``max_metered_calls``, a ``cassette_mode`` that is not a mode, a replay naming no corpus
            or a corpus that is no capture of this template in this scope, a template naming a
            kind with no launcher, a launch argument that kind cannot honour, an overlay the kind's
            model refuses (named by field), an apparatus setting the kind does not honour, a generating
            launch of a kind that plans no arm, a generating arm predicted above its cap or unpriceable
            under an inherited one (or any generating launch under an enforced cap on a host with no
            pricer), or any refusal the kind's launcher makes.
    """
    # NOTE: the "at least one model" refusal is NOT here, because it is not the same
    # refusal for every kind — each kind's launcher makes its own.
    # Per-run cost-cap override: a non-positive cap would stop the
    # run before the first result — reject it as a run-parameter error up
    # front, before any snapshot / generation spend.
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValidationFailedError(f"max_cost_usd must be > 0 (got {max_cost_usd})")
    # Same reasoning one currency over: a ceiling of zero or less would refuse the
    # candidate's first metered call and measure a candidate that could not search.
    if max_metered_calls is not None and max_metered_calls <= 0:
        raise ValidationFailedError(f"max_metered_calls must be > 0 (got {max_metered_calls})")
    # The run model bounds k_runs, but it is built after the cases are resolved — so an out-of-range
    # value from a surface that does not bound it at the wire would be refused after generation had
    # been paid for. Checked against the model's own field, so the bound is stated once.
    k_field = EvalRun.model_fields["k_runs"]
    try:
        TypeAdapter(Annotated[int, *k_field.metadata]).validate_python(k_runs)
    except ValidationError as e:
        raise ValidationFailedError(f"invalid k_runs: {e.errors()[0]['msg']} (got {k_runs})") from e
    mode = _normalized_cassette_mode(cassette_mode)
    _refuse_a_corpus_the_mode_cannot_use(mode, cassette_corpus_id)
    dispatch = partial(
        _dispatch,
        host,
        template_id,
        scope_id,
        models=models,
        n_variations=n_variations,
        variation_model=variation_model,
        judge_model=judge_model,
        judge_config_ids=judge_config_ids,
        simulator_model=simulator_model,
        cassette_mode=mode,
        cassette_corpus_id=cassette_corpus_id,
        overlays=overlays,
        apparatus_settings=apparatus_settings,
    )

    async def prepare(group: LaunchGroup, dispatched: _Dispatched) -> list[EvalRun]:
        """Price a generating launch's arms, then hand each arm, one per model, to the kind's launcher.

        Args:
            group: The launch the runs are prepared into.
            dispatched: What the dispatch resolved.

        Returns:
            The prepared runs, one per model in the order given.
        """
        requests = _arm_requests(
            host,
            dispatched,
            group,
            subject_id=subject_id,
            models=models,
            k_runs=k_runs,
            scope_id=scope_id,
            n_variations=n_variations,
            variation_model=variation_model,
            judge_model=judge_model,
            judge_config_ids=judge_config_ids,
            simulator_model=simulator_model,
            cassette_mode=mode,
            cassette_corpus_id=cassette_corpus_id,
            max_cost_usd=max_cost_usd,
            max_metered_calls=max_metered_calls,
        )
        if n_variations > 0:
            # Every arm priced before the first launcher runs: the launcher is what pays for the
            # generation the arms share, so pricing an arm after it would refuse a launch already billed.
            requests = await _priced_generating_arms(host, dispatched.launchable, requests)
        return [await dispatched.launchable.launch(request) for request in requests]

    # Preparing arms into a caller's group: that caller admitted the launch, and it starts the
    # group once every arm is prepared or abandons it on a refusal, so this call only prepares.
    if launch_group is not None:
        return await prepare(launch_group, await dispatch())

    async def form() -> tuple[LaunchGroup, _Dispatched]:
        """Make the template's refusals, then form the launch's group: one arm per model.

        Returns:
            The group, and what the dispatch resolved.
        """
        dispatched = await dispatch()
        return LaunchGroup(candidate_models=models), dispatched

    # This call is the whole launch. A caller holding a ticket for several launches (the battery)
    # hands this one its share rather than letting it compete for room with launches that arrived
    # after the caller was admitted.
    return await launch_as_group(
        host, max(1, len(models)), form=form, prepare=prepare, admission=admission, event="eval.start_run"
    )


def _arm_requests(
    host: LaunchHost,
    dispatched: _Dispatched,
    group: LaunchGroup,
    *,
    subject_id: str,
    models: Sequence[str],
    k_runs: int,
    scope_id: str,
    n_variations: int,
    variation_model: str | None,
    judge_model: str | None,
    judge_config_ids: dict[str, str] | None,
    simulator_model: str | None,
    cassette_mode: CassetteMode,
    cassette_corpus_id: str | None,
    max_cost_usd: float | None,
    max_metered_calls: int | None,
) -> list[LaunchRequest]:
    """One request per arm — per model, or one on the kind's default when the launch names none.

    A launch that generates gets ONE out-of-run budget, shared by every arm's request: the arms share
    one generation (:meth:`LaunchGroup.resolve_once`), so its calls are admitted against one cap and
    ledgered under the launch's group.

    Args:
        host: The host, whose settings cap the generation and whose store ledgers it.
        dispatched: What the dispatch resolved.
        group: The launch the arms are prepared into.
        subject_id: The launch's subject.
        models: The launch's candidate models.
        k_runs: Repeats per case.
        scope_id: The scope the runs live in.
        n_variations: New cases the launch generates.
        variation_model: The model that writes the ``llm`` axes' values.
        judge_model: The judge pin.
        judge_config_ids: The per-dim judge-configuration selection.
        simulator_model: The simulator pin.
        cassette_mode: The cassette mode, normalised.
        cassette_corpus_id: The corpus a replay serves.
        max_cost_usd: The per-run cost-cap override.
        max_metered_calls: The per-run metered-call ceiling override.

    Returns:
        The requests, in arm order, with no arm plan yet.
    """
    budget: OutOfRunBudget | None = None
    if n_variations > 0:
        settings = host.settings()
        budget = OutOfRunBudget(
            host.eval_host.storage,
            scope_id=scope_id,
            cap_usd=settings.max_out_of_run_cost_usd if settings.enforcement_enabled else None,
            template_id=dispatched.template.id,
            subject_id=subject_id,
            launch_group_id=group.id,
        )
    # One arm per model. Naming none is one arm too — on the kind's role default where it has one,
    # and refused by the launcher of a kind that has none.
    arm_models: list[str | None] = [*models] or [None]
    return [
        LaunchRequest(
            template=dispatched.template,
            kind=dispatched.kind,
            subject_id=subject_id,
            candidate_model=arm_model,
            k_runs=k_runs,
            scope_id=scope_id,
            n_variations=n_variations,
            variation_model=variation_model,
            judge_model=judge_model,
            judge_config_ids=judge_config_ids,
            simulator_model=simulator_model,
            cassette_mode=cassette_mode,
            cassette_corpus_id=cassette_corpus_id,
            overlays=dispatched.overlays,
            kind_spec=dispatched.kind_spec,
            max_cost_usd=max_cost_usd,
            max_metered_calls=max_metered_calls,
            apparatus_settings=MappingProxyType(dict(dispatched.apparatus_settings)),
            generation_budget=budget,
            arm_plan=None,
            launch_group=group,
        )
        for arm_model in arm_models
    ]


def _planned(launchable: LaunchableKind, request: LaunchRequest) -> ArmPlan:
    """Ask the kind what one arm of a generating launch will run, and refuse a plan the arm contradicts.

    Args:
        launchable: The kind's registry entry; its ``plan_arm`` was checked present at the dispatch.
        request: The arm.

    Returns:
        The plan.

    Raises:
        ValueError: The plan names another model than the one the arm named — a kind defect.
    """
    plan_arm = launchable.plan_arm
    assert plan_arm is not None, "the dispatch refuses a generating launch of a kind that plans no arm"
    plan = plan_arm(request)
    if request.candidate_model is not None and plan.candidate_model != request.candidate_model:
        raise ValueError(
            f"kind {request.kind!r} planned an arm the launch named {request.candidate_model!r} on "
            f"{plan.candidate_model!r}; an arm runs on the model the launch named"
        )
    return plan


async def _priced_generating_arms(
    host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]
) -> list[LaunchRequest]:
    """Plan and price every arm of a generating launch, refusing before the generation is paid for.

    An arm is refused when its predicted cost is above the cap its run will be held to, or when nothing
    predicts it and that cap is one the run would merely inherit (the host's configured ceiling, which
    nobody chose for this run): unknown is not $0. An arm with a cap the launch named and no prediction
    goes ahead — the named cap is the most the operator chose to risk on a run nobody could price, and
    the run's own cost cap enforces it as the spend arrives. With the host's enforcement off no cap is in
    force, and nothing is priced.

    Args:
        host: The host: its settings resolve the cap, and its pricer predicts.
        launchable: The kind's registry entry, whose ``plan_arm`` plans each arm.
        requests: The launch's arms, in order; every one carries the same ``max_cost_usd``.

    Returns:
        The requests, each carrying its plan.

    Raises:
        ValidationFailedError: An arm predicted above its cap, one unpredicted under an inherited cap, or
            any arm under an enforced cap on a host with no pricer.
        ValueError: The kind planned an arm on another model than the one it named.
    """
    planned = [replace(request, arm_plan=_planned(launchable, request)) for request in requests]
    settings = host.settings()
    max_cost_usd = planned[0].max_cost_usd
    cap = EvalRunCostCap.resolve_effective_ceiling(
        max_cost_usd, configured_max_cost_usd=settings.max_cost_usd, enforcement_enabled=settings.enforcement_enabled
    )
    if cap is None:
        return planned
    origin = EvalRunCostCap.resolve_ceiling_origin(max_cost_usd, enforcement_enabled=settings.enforcement_enabled)
    pricer = host.launch_pricer
    first = planned[0]
    if pricer is None:
        raise ValidationFailedError(
            f"this launch generates {first.n_variations} case(s) before its runs start, and host "
            f"{host.eval_host.profile.host_id!r} prices no launch (LaunchHost.launch_pricer), so its arms cannot be "
            f"held to their ${cap:.2f} cap before the generation is paid for; build the host with a launch pricer, "
            "or launch with n_variations=0. Nothing was called"
        )
    for request in planned:
        plan = request.arm_plan
        assert plan is not None
        price = await run_blocking(
            host.eval_host.blocking_executor,
            pricer,
            ArmQuote(
                scope_id=request.scope_id,
                template_id=request.template.id,
                subject_id=request.subject_id,
                candidate_model=plan.candidate_model,
                k_runs=request.k_runs,
                case_count=plan.case_count,
                cassette_mode=request.cassette_mode,
            ),
        )
        what = (
            f"the arm on {plan.candidate_model!r} ({plan.case_count} generated case(s) x {request.k_runs} repeat(s) "
            f"of template {request.template.id!r})"
        )
        if price.predicted_usd is None:
            if origin == "chosen":
                continue
            raise ValidationFailedError(
                f"{what} cannot be priced: {price.basis}. Its cost is unknown, not $0, and its ${cap:.2f} cap is "
                f"inherited from {settings.name_of('max_cost_usd')}, which nobody chose for it. Launch it naming "
                "max_cost_usd — the most you will risk on a run nobody can price; what it costs then prices the next "
                "launch. Refused before the generation was paid for"
            )
        if price.predicted_usd > cap:
            raise ValidationFailedError(
                f"{what} is predicted to cost ${price.predicted_usd:.2f} ({price.basis}), above its ${cap:.2f} cap. "
                "Fewer generated cases, a smaller k_runs or a cheaper model brings it under; a larger max_cost_usd "
                "raises the cap. Refused before the generation was paid for"
            )
    return planned


async def launch_as_group(
    host: LaunchHost,
    n_runs: int,
    *,
    form: Callable[[], Awaitable[tuple[LaunchGroup, _Formed]]],
    prepare: Callable[[LaunchGroup, _Formed], Awaitable[list[EvalRun]]],
    event: str,
    admission: AdmissionTicket | None = None,
    attach: Callable[[list[EvalRun]], Awaitable[None]] | None = None,
    detach: Callable[[list[EvalRun]], Awaitable[None]] | None = None,
) -> list[EvalRun]:
    """Launch one group of runs: admit it, prepare every arm, and start them together — or start none.

    The one sequence every launch that owns its group goes through, in this order:

    1. Refuse a launch of more runs than one launch may start together. It reads only the run
       count, and it is the more specific refusal, so it comes before admission.
    2. Reserve room for the runs against the host's admission ceiling, or take their share of a
       reservation the caller already holds — before anything is read, so a launch this process
       has no room for costs nothing.
    3. ``form``: the caller's own reads and refusals, which form the group.
    4. ``prepare``: every arm prepared into the group (every refusal made, every client built).
    5. ``attach``, when given: record the prepared runs somewhere before they start. Waited for
       through a cancel, because the write cannot be interrupted once a worker has it, and
       ``detach`` must know whether it landed.
    6. Start every member in one concurrency slot.

    A failure anywhere in 4-6, a cancel included, abandons the group — every prepared member's
    clients are released — and undoes ``attach`` through ``detach`` if it had landed. The
    reservation is released however the launch ends: started runs are live tasks that count on
    their own, and a refused launch's share goes back.

    Args:
        host: The host; its arm ceiling and its admission are read here and nowhere else.
        n_runs: How many runs the launch starts, one per arm.
        form: Makes the caller's refusals under the reservation and returns the group, with
            whatever ``prepare`` needs of what it read.
        prepare: Prepares every arm into the group and returns the prepared runs.
        event: How the group's log lines begin, e.g. ``eval.start_run``.
        admission: A reservation the caller holds, of which this launch takes ``n_runs``.
        attach: Records the prepared runs before they start.
        detach: Undoes ``attach`` when the group is abandoned after it landed.

    Returns:
        The started runs, as ``prepare`` returned them.

    Raises:
        ValidationFailedError: ``n_runs`` is more than one launch may start together, or any
            refusal ``form`` or ``prepare`` makes.
        AdmissionRefusedError: The runs would pass the host's admission ceiling.
    """
    job_manager = host.job_manager
    settings = host.settings()
    _refuse_oversized_launch(n_runs, settings.max_launch_arms, settings.name_of("max_launch_arms"))
    ticket = (
        admission.split(n_runs)
        if admission is not None
        else job_manager.admit(
            n_runs, limit=settings.max_admitted_runs, limit_name=settings.name_of("max_admitted_runs")
        )
    )
    try:
        group, formed = await form()
        runs: list[EvalRun] = []
        attached = False
        try:
            runs = await prepare(group, formed)
            if attach is not None:
                attaching = asyncio.ensure_future(attach(runs))
                cancelled_while_attaching = await wait_through_cancellation(attaching)
                attaching.result()
                attached = True
                if cancelled_while_attaching:
                    raise asyncio.CancelledError
            await job_manager.start_group(group.members)
        except BaseException:
            log.warning("%s group=%s abandoned before starting", event, group.id)
            await group.abandon()
            if attached and detach is not None:
                # The runs never started, so whatever recorded them must not list them.
                await detach(runs)
            raise
        log.info("%s group=%s started runs=%s", event, group.id, [run.id for run in runs])
        return runs
    finally:
        ticket.release()


def _refuse_wiring_the_request_contradicts(request: LaunchRequest, wiring: KindWiring) -> str:
    """Check what a kind's launcher resolved against what the launch asked for, and name the candidate model.

    The launch tail stamps the request onto the run; the wiring carries only what the kind resolved.
    Where the two speak to one fact they must agree, or the run would record one thing and run
    another — so each disagreement is refused, before anything is built.

    Args:
        request: The arm as the dispatch resolved it.
        wiring: What the kind's launcher resolved for it.

    Returns:
        The arm's candidate model: the one the request named, or the kind's default.

    Raises:
        ValidationFailedError: The request named no model and the kind supplies no default.
        ValueError: The wiring contradicts the request.
    """
    kind = request.kind
    if wiring.subject.subject_id != request.subject_id:
        raise ValueError(
            f"kind {kind!r}'s launcher captured subject {wiring.subject.subject_id!r} for a launch naming "
            f"{request.subject_id!r}; a launcher captures the subject the launch names"
        )
    if strays := sorted(
        case.id
        for case in wiring.test_cases
        if case.scope_id != request.scope_id or case.template_id != request.template.id
    ):
        raise ValueError(
            f"kind {kind!r}'s launcher froze case(s) {', '.join(strays)} from outside template "
            f"{request.template.id!r} in scope {request.scope_id!r}; a run's cases are its template's, in its scope, "
            "or its results name cases nothing in that scope resolves"
        )
    if request.candidate_model is not None and wiring.default_candidate_model is not None:
        raise ValueError(
            f"kind {kind!r}'s launcher supplied a default model ({wiring.default_candidate_model!r}) for an arm the "
            f"launch named ({request.candidate_model!r}); a default is for an arm that named none"
        )
    candidate_model = request.candidate_model or wiring.default_candidate_model
    if candidate_model is None:
        raise ValidationFailedError(f"kind {kind!r} has no default candidate model; name one")
    judge = wiring.judge
    if request.judge_model is not None and (judge is None or judge.model != request.judge_model):
        raise ValueError(
            f"the launch pinned judge {request.judge_model!r} and kind {kind!r}'s launcher wired "
            f"{judge.model if judge is not None else 'no judge'!r}; a pinned judge is the judge"
        )
    if request.judge_config_ids and (judge is None or judge.selection != request.judge_config_ids):
        raise ValueError(
            f"the launch selected judge configs {request.judge_config_ids} and kind {kind!r}'s launcher built its "
            "judge from another selection; build it from the request's"
        )
    if request.simulator_model is not None and wiring.simulator_model != request.simulator_model:
        raise ValueError(
            f"the launch pinned simulator {request.simulator_model!r} and kind {kind!r}'s launcher resolved "
            f"{wiring.simulator_model!r}; a pinned simulator is the simulator"
        )
    counts = wiring.variation_counts
    if (counts is None) != (request.n_variations == 0) or (
        counts is not None and counts.requested != request.n_variations
    ):
        raise ValueError(
            f"the launch asked for {request.n_variations} generated case(s) and kind {kind!r}'s launcher recorded "
            f"{counts.requested if counts is not None else 'no generation'}; a launch that generates records what it "
            "asked for, and one that reuses stored cases records none"
        )
    if counts is not None and counts.variation_model != request.variation_model:
        raise ValueError(
            f"the launch named variation model {request.variation_model!r} and kind {kind!r}'s launcher's generation "
            f"recorded {counts.variation_model!r} as the model that wrote its cases; build the writer in the "
            "'variation' role from the request's variation_model and hand it to generate_variations, whose counts "
            "name the model it called"
        )
    budget = request.generation_budget
    if counts is not None and counts.variation_model is not None and (budget is None or not budget.recorded):
        raise ValueError(
            f"kind {kind!r}'s launcher recorded cases written by {counts.variation_model!r}, and the launch's "
            "generation budget ledgered no call: hand generate_variations budget=request.generation_budget, so the "
            "calls are priced against the launch's out-of-run cap before they are made and ledgered once they are"
        )
    plan = request.arm_plan
    if plan is not None:
        if candidate_model != plan.candidate_model:
            raise ValueError(
                f"kind {kind!r} planned this arm on {plan.candidate_model!r} and its launcher ran it on "
                f"{candidate_model!r}; the arm was priced as the plan, so it runs as the plan"
            )
        if len(wiring.test_cases) > plan.case_count:
            raise ValueError(
                f"kind {kind!r} planned this arm at most {plan.case_count} case(s) and its launcher froze "
                f"{len(wiring.test_cases)}; the arm was priced at the plan's count, so it may run fewer, never more"
            )
    return candidate_model


async def launch_run(host: LaunchHost, request: LaunchRequest, wiring: KindWiring) -> EvalRun:
    """Assemble, stamp, bound and hand over a run — the launch tail every candidate kind shares.

    Each kind's launcher makes its own refusals, captures its own subject, resolves its own case
    set and builds its own bound kind; what is left is the same for all of them, and its ORDER is an
    invariant rather than a convention, so it lives here once:

    1. Check the wiring against the request: the subject the launcher captured is the one the
       launch named, the cases live in the run's scope and template, every model the request
       names is the one the wiring resolved, a model-written generation was ledgered through the
       request's budget, and a planned arm runs within its plan.
    2. Read the host's budget settings once, and assemble the run. Everything the request and the
       template already say is stamped from them — scope, template, kind, candidate model, repeats,
       cassette mode, overlays, spec, apparatus settings, world seed, tool bound, ceilings — and only what the kind
       resolved comes from the wiring.
    3. Stamp the world placements, then the context identity that hashes them, then the
       variant levers, all from the assembled run.
    4. Build the cost cap and the ONE metered-call ledger from the same settings read.
    5. Build the runner options and the work function — whose ``finally`` records completeness
       and then releases the wiring's ``teardown`` — and hand them to the request's group under a
       backstop sized to the matrix. The group starts them with the run's sibling arms.

    **The wiring's ``teardown`` is owned from the moment it is handed over.** A launch that raises
    anywhere in here releases it, because the work function that would otherwise release it never
    runs; a run handed to the group leaves it to the group (released on abandon) or to the
    work function.

    Args:
        host: The host: its settings, its world placement, its job manager and what every run reads.
        request: The arm as :func:`start_run` handed it to the kind's launcher, unchanged. The run is
            stamped with the request's group's id, and its work function, budget and teardown are
            handed to that group, which then owns them and starts them with the run's siblings.
        wiring: What the kind's launcher resolved for the arm.

    Returns:
        The run (status ``pending``), held by the request's group.

    Raises:
        ValidationFailedError: The request named no candidate model and the kind has no default, or
            the assembled run does not validate.
        ValueError: The wiring contradicts the request — a subject other than the one named, a case
            in another scope or of another template, a model other than the one the request pinned,
            a generation count other than the one it asked for, a generation written by a model
            other than the request's variation model or whose calls the request's generation budget
            never ledgered, or an arm that runs more cases than, or another model than, its plan. A
            launcher defect, never a caller's; nothing is built.
    """
    template = request.template
    launch_group = request.launch_group
    max_cost_usd = request.max_cost_usd
    max_metered_calls = request.max_metered_calls
    teardown = wiring.teardown if wiring.teardown is not None else contextlib.AsyncExitStack()
    try:
        candidate_model = _refuse_wiring_the_request_contradicts(request, wiring)
        test_cases = list(wiring.test_cases)
        judge = wiring.judge
        # Whether each pinned role was named by the launch or resolved from the role's default: the
        # request says which, so the run records it without the launcher restating it.
        role_provenance: dict[str, ModelRoleOrigin] = {}
        if judge is not None:
            role_provenance["judge"] = "chosen" if request.judge_model is not None else "inherited"
        if wiring.simulator_model is not None:
            role_provenance["simulator"] = "chosen" if request.simulator_model is not None else "inherited"
        # The eval ceilings arrive at the engine as values (R7): budget.py and metering.py
        # read no configuration of their own, and the host's settings are resolved through
        # the host's launch settings. Read ONCE here and used for both the figures stored on the
        # run and the objects that enforce them — and the judge concurrency below — so separate
        # reads cannot straddle a hot reload and leave a run recording a ceiling that is not the
        # one it ran under.
        settings = host.settings()
        eval_enforcement_enabled = settings.enforcement_enabled
        configured_max_cost_usd = settings.max_cost_usd
        configured_max_metered_calls = settings.max_metered_calls

        try:
            run = EvalRun(
                scope_id=request.scope_id,
                template_id=template.id,
                subject_snapshot=wiring.subject,
                host_payload=dict(wiring.payload),
                candidate_model=candidate_model,
                k_runs=request.k_runs,
                test_case_ids=[case.id for case in test_cases],
                variation_counts=wiring.variation_counts,
                candidate_kind=request.kind,
                # A launch is the act of fixing a rig and measuring against it, so every run it starts
                # was commissioned. A witnessed run never comes through here: its writer is the host
                # operation that captured traffic it did not control, and it says so itself.
                apparatus_provenance="commissioned",
                judge_model=judge.model if judge is not None else None,
                model_role_provenance=role_provenance,
                effective_judges=judge.effective_judges if judge is not None else None,
                effective_judges_source="recorded" if judge is not None else None,
                judge_config_ids=judge.config_ids if judge is not None else None,
                judge_config_provenance=judge.config_provenance if judge is not None else None,
                rubric_scales={dim.name: dim.scale for dim in template.rubric},
                overlays=freeze(request.overlays),
                kind_spec=freeze(request.kind_spec),
                apparatus_settings=dict(request.apparatus_settings),
                # The world and the tool bound the template states, frozen as this run launched them:
                # the template is editable, and the runner hands the candidate the template's seed.
                resolved_world_seed=dict(template.world_seed.namespaces),
                resolved_ambient_perturbation_turns=list(template.world_seed.ambient_perturbation_turns),
                resolved_tools_allowed=list(template.tools_allowed) if template.tools_allowed is not None else None,
                cassette_mode=request.cassette_mode,
                cassette_corpus_id=request.cassette_corpus_id,
                simulator_model=wiring.simulator_model,
                turn_budget_s=wiring.turn_budget_s,
                # How each apparatus role was ASKED, stamped from the one value the host's client
                # builder applies to it, and only for a role this kind ran: a resolved model is
                # recorded exactly when the role ran, so it is the predicate here too.
                judge_request_settings=JUDGE_REQUEST_SETTINGS if judge is not None else None,
                simulator_request_settings=SIMULATOR_REQUEST_SETTINGS if wiring.simulator_model is not None else None,
                max_cost_usd=EvalRunCostCap.resolve_effective_ceiling(
                    max_cost_usd,
                    configured_max_cost_usd=configured_max_cost_usd,
                    enforcement_enabled=eval_enforcement_enabled,
                ),
                # Resolution erases which tier supplied the ceiling, and only the
                # inherited one moves under a run's feet — so the origin is recorded
                # beside the number, exactly as the judge/simulator pins are. Read
                # through the same helper that resolved the number so the two can
                # never disagree about whether a cap was in force at all.
                max_cost_usd_origin=EvalRunCostCap.resolve_ceiling_origin(
                    max_cost_usd, enforcement_enabled=eval_enforcement_enabled
                ),
                # The other currency, resolved through its own helper for the same
                # reason: the number the run records must be the number that bounds it.
                max_metered_calls=MeteredCallLedger.resolve_effective_ceiling(
                    max_metered_calls,
                    configured_max_metered_calls=configured_max_metered_calls,
                    enforcement_enabled=eval_enforcement_enabled,
                ),
                max_metered_calls_origin=MeteredCallLedger.resolve_ceiling_origin(
                    max_metered_calls, enforcement_enabled=eval_enforcement_enabled
                ),
            )
        except ValidationError as e:
            raise ValidationFailedError(f"invalid run parameters: {e}") from e

        # What this run did with every dimension the host's world declares, derived from the
        # assembled run for the same reason the context key is: the run is the thing the record
        # describes. Stamped rather than left to a read, because the world registry is host CODE
        # and moves under stored runs — deriving placements later would report today's world for a
        # run measured under an older one. It has to land BEFORE the identity below, which hashes
        # it: the world a subject is placed in is stimulus, and two runs that placed it differently
        # are not repetitions of one condition.
        run.world_placements = host.place(run)

        # Measurement-context identity, stamped from the assembled run rather
        # than from the launch arguments: the run is the thing whose conditions
        # the key claims to describe, and reading it back is what makes a
        # stamped key and a read-time derived one provably the same predicate
        # over the same inputs. Stamped so `data->>'context_key'` groups
        # comparable runs in a query, with the components kept alongside so a
        # mismatch can name which condition moved.
        identity = derive_context_identity(run, host.eval_host.profile)
        run.context_key = identity.context_key
        run.context_components = identity.context_components
        run.identity_version = identity.identity_version
        # The VARIANT key's pre-image, one map per candidate model, stamped from the same
        # assembled run and for the same reason the components above are: a digest cannot say
        # what it was digested from, and reconstructing the map later replays whatever predicate
        # the reading build happens to carry. That reconstruction is only valid while the
        # predicate is frozen, and IDENTITY_VERSION exists because it is not — so a campaign
        # analysed across a bump lost the description of every one of its arms while their keys
        # stayed authoritative for pooling. Recorded here, nothing has to be reconstructed.
        run.variant_levers = variant_levers_of_run(run, host.eval_host.profile)

        # Per-run cost cap: bounds THIS run's spend only. Pure in-memory
        # arithmetic seeded from the ceilings resolved above (or the validated per-run
        # override) — no shared pool, no budget-layer / DB call in the gate path.
        cost_cap = EvalRunCostCap.for_run(
            run.id,
            max_cost_usd,
            configured_max_cost_usd=configured_max_cost_usd,
            enforcement_enabled=eval_enforcement_enabled,
        )

        # Per-run metered-call ceiling: bounds the third-party quota the cost cap cannot
        # see. ONE ledger for the whole run — handed to the runner, which hands it to the kind's
        # factory with every cell (a candidate factory declares it on every slot, so the
        # dispatcher can consult it) and has each cell report its slice — because a per-cell ledger would
        # silently turn a per-run ceiling into a per-cell one N times larger. Same
        # in-memory, no-I/O shape as the cost cap; the two are built through their own
        # `for_run` because their constructors take the ceiling in mirror-opposite forms,
        # and that difference is the engine's to hold, not this function's to remember.
        metered_calls = MeteredCallLedger.for_run(
            run.id,
            max_metered_calls,
            configured_max_metered_calls=configured_max_metered_calls,
            enforcement_enabled=eval_enforcement_enabled,
        )

        job_manager = host.job_manager
        options = RunnerOptions(
            # From the settings read above, per run, so a hot raise of the judge concurrency
            # lands on the next run rather than the next restart — the runner takes
            # configuration as values, so this copy is the only way the setting reaches it.
            # Inert for a kind with no judge.
            judge_concurrency=settings.judge_concurrency,
            # The kind the launch dispatched on, wired to the one factory its launcher built: the
            # dispatch's kind is the only key, so the factory cannot be registered under another.
            candidate_kinds={request.kind: wiring.kind_factory},
            external_rates=wiring.external_rates,
            cell_timeout_s=wiring.cell_timeout_s if wiring.cell_timeout_s is not None else DEFAULT_CELL_TIMEOUT_S,
            # Measurement-condition probe (R4): the job manager is the only thing that can
            # see a SECOND job executing beside this one, which is the contention that
            # corrupts a latency pool — cells within a run are serial. The run is this job
            # manager's job, so the probe is the run's rather than the host's: a run executed
            # outside a job manager records no execution_mode rather than a guessed one.
            concurrent_eval_jobs_probe=lambda: job_manager.active_count,
            # One tally rather than two: what the kind's collaborators count and what each
            # cell reports. A kind that calls no metered third party records zero refusals
            # under a ceiling that was in force, which is a different fact from nobody
            # having counted.
            metered_calls=metered_calls,
        )

        # The loop the work function runs, bound now rather than looked up when the job starts it:
        # the run is assembled against this launch's runner, and the job may start it after anything
        # module-level has moved.
        run_cells = execute_run

        async def work(progress_fn: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
            # Handed to the loop rather than taken from its return, so the record
            # is written however the loop ends. A cancel, the cost cap and the job
            # timeout each unwind the frame instead of returning, and those are
            # precisely the runs with a shortfall to disclose — a run killed after
            # one cell of five otherwise reported a perfect pass^k with no badge,
            # because the mechanism that badges short matrices had no record to
            # read.
            cells: list[CellSummary] = []
            try:
                await run_cells(
                    host.eval_host,
                    run=run,
                    template=template,
                    test_cases=test_cases,
                    judge_service=judge.service if judge is not None else None,
                    options=options,
                    callbacks=RunCallbacks(on_progress=progress_fn),
                    # Mid-run cap: accumulate each delivered result's cost
                    # in memory and stop gracefully once this run's total exceeds its
                    # cap. Pure arithmetic — no budget-layer / DB I/O in the gate path.
                    budget_gate=cost_cap.check,
                    on_cost=cost_cap.record,
                    cell_sink=cells,
                )
            finally:
                try:
                    # Runs before the job manager stamps a status over it, on every
                    # path. Nothing here can be re-derived from storage afterwards: a
                    # cell whose write failed left no row to count. Off the loop — it is a
                    # read-modify-write retried under contention — and waited for to its end
                    # even through a second cancel, so it still lands before the terminal
                    # status write reads the same document; the cancel is delivered after.
                    recording = asyncio.ensure_future(
                        run_blocking(
                            host.eval_host.blocking_executor,
                            record_completeness,
                            host.eval_host.storage,
                            run.id,
                            run.scope_id,
                            cells,
                            metered=metered_calls.tally(),
                        )
                    )
                    cancelled_while_recording = await wait_through_cancellation(recording)
                    # `record_completeness` logs its own failures, so an exception here is the
                    # hop itself failing — the pool gone — and nothing else would say so.
                    if recording.exception() is not None:
                        log.error(
                            "eval.completeness run=%s scope=%s NOT recorded — the write never ran: %s",
                            run.id,
                            run.scope_id,
                            recording.exception(),
                        )
                    if cancelled_while_recording:
                        raise asyncio.CancelledError
                finally:
                    # **Run teardown is where the launch-built clients are released**.
                    # They own httpx pools the collector does not close
                    # deterministically, and they are used across a run lasting
                    # minutes — so a `finally` at the mint would have closed them
                    # mid-use, and this frame is the first one that outlives their
                    # whole lifetime. It runs on the paths that matter most: a
                    # cancel, the mid-run cost cap and the job timeout each unwind
                    # here rather than returning. Nested inside the completeness
                    # record's `finally` so a failed record still releases, and vice
                    # versa. No release raises: each client's `__aexit__` logs a failed
                    # teardown instead, because the run's own outcome is the error
                    # worth reporting.
                    await teardown.aclose()

        # Size the job's wall-clock backstop to the matrix. Cells run
        # sequentially, each capped at options.cell_timeout_s, so a fixed 3600s
        # cap guillotines any matrix past ~6 cells; scale it with N instead.
        n_results = len(test_cases) * run.k_runs
        job_timeout_s = adaptive_job_timeout_s(n_results, per_result_s=options.cell_timeout_s)
        log.info(
            "eval.start_run run=%s group=%s job_timeout=%.0fs (%d results × %.0fs/result cell ceiling)",
            run.id,
            launch_group.id,
            job_timeout_s,
            n_results,
            options.cell_timeout_s,
        )
        run.launch_group_id = launch_group.id
        launch_group.add(run, work, job_timeout_s, teardown)
    except BaseException:
        # The work function never ran, so its teardown never will.
        await teardown.aclose()
        raise
    return run


def _battery_refusal(refused: Exception) -> str:
    """Render a per-template refusal as a battery refusal.

    The per-run message cannot know that nothing was launched, and an operator reading a
    battery failure has to. Punctuated rather than concatenated: the refusals it wraps end
    in prose, a bracket or a period depending on which one fired, and ``… names it wrongly
    Nothing was launched.`` is what naive concatenation produced.

    Args:
        refused: The refusal raised for one template.

    Returns:
        Its message with the battery's own suffix.
    """
    message = str(refused).rstrip()
    separator = "" if message.endswith((".", "!", "?")) else "."
    return f"{message}{separator} Nothing was launched."


#: Checks one of a battery's templates the way its launch would, before any template launches.
#: ``(template, cassette_mode) -> None``, raising the launch's own refusal.
TemplatePreflight = Callable[["EvalTemplate", str], Awaitable[None]]

#: Prepares a battery's pre-flight for one subject and the battery's models, once, and returns the
#: per-template check. A host's arms are built on a captured subject, so what every template's check
#: reads is captured here once rather than per template. ``(subject_id, models) -> TemplatePreflight``.
BatteryPreflight = Callable[[str, Sequence[str]], Awaitable[TemplatePreflight]]


async def start_universal_battery(
    host: LaunchHost,
    subject_id: str,
    *,
    scope_id: str,
    models: list[str],
    k_runs: int = DEFAULT_LAUNCH_K_RUNS,
    n_variations: int = 0,
    variation_model: str | None = None,
    judge_model: str | None = None,
    simulator_model: str | None = None,
    cassette_mode: str | None = "off",
    apparatus_settings: Mapping[str, Any] | None = None,
    preflight: BatteryPreflight,
) -> list[str]:
    """Launch the operator-curated boundary battery against one subject.

    Queries every active ``universal`` template in ``scope_id`` and launches a separate :func:`start_run` per
    template, all against the same subject and models. The battery deliberately respects the
    one-template-per-run model rather than unioning test cases: a run's rubric is its template's,
    so each universal template scores against its own boundary rubric.

    **No judge-configuration selection here, for the same reason.** A launch selects configs per
    scored DIM, and these templates score different dims — so one set would be refused by the first
    template that does not score a named dim, and a set narrowed to the intersection would silently
    mean something different per template. Each battery run therefore inherits its dims' active
    configs. Select per run by calling :func:`start_run` directly.

    **All or nothing.** Every refusal an arm of any template would make is made over the whole set
    before anything launches, through ``preflight`` — the host's check that calls what its launchers
    call, not a list of checks kept here.

    Args:
        host: The host, whose launch registry names the kind's launcher.
        subject_id: The subject every template's runs measure.
        scope_id: The scope the battery's templates are read in and its runs live in.
        models: Candidate models, one arm and one run each, per template.
        k_runs: Repeats per ``(test_case, model)`` for ``pass^k``.
        n_variations: New test cases to generate per run (0 = reuse existing).
        variation_model: The model that writes the ``llm`` variation axes' values, handed to each
            template's launch that has such an axis and to no other — a battery's templates differ, and
            one without an ``llm`` axis has no cases a model writes. Required when the battery generates
            and any of its templates has an ``llm`` axis; refused when it generates nothing or none of
            its templates has one.
        judge_model: Model for the rubric judge (role default if ``None``).
        simulator_model: Model driving the simulated user (role default if ``None``).
        cassette_mode: ``'off'`` (default, and what blank spells), ``'capture'`` or ``'replay'``,
            applied to every template in the battery.
        apparatus_settings: Host-declared apparatus values every template's runs are set up with, as
            :func:`start_run` takes them — refused before anything launches when any template's kind
            does not honour one.
        preflight: The host's pre-flight, prepared once for the subject and models and then asked
            of every template before any launches. A host whose launchers generate with a model checks
            each template's generation calls here
            (:func:`~threetears.evals.gen.price_variations`), since those are priced inside the launcher;
            each template's generating ARMS the battery prices itself, before any template launches.

    Returns:
        The ids of the launched runs: for each active universal template, one per model. An empty
        battery (no active universal templates) returns ``[]``.

    Raises:
        AdmissionRefusedError: Every template's runs together would pass the host's admission
            ceiling — raised once the set is listed and before any subject is read or arm prepared,
            so nothing launches.
        ValidationFailedError: An unrecognised ``cassette_mode``, ``'replay'`` (a corpus serves one
            template, and a battery runs many), or any refusal a template or its
            arm would make at launch — raised before any template launches, so nothing does.
    """
    templates = await run_blocking(
        host.eval_host.blocking_executor,
        host.eval_host.storage.query_templates,
        scope_id,
        universal=True,
        archived=False,
    )
    # Every template's runs are admitted together, here, before the pre-flight reads a subject or
    # prepares an arm — the count is known once the set is, and a battery admitted template by
    # template could be refused part-way, launching the partial set it promises never to.
    settings = host.settings()
    ticket = host.job_manager.admit(
        len(templates) * max(1, len(models)),
        limit=settings.max_admitted_runs,
        limit_name=settings.name_of("max_admitted_runs"),
    )
    try:
        # Pre-flight the whole set before launching any of it. Every other run-param
        # refusal is a property of the ARGUMENTS, so it fires on the first template and
        # the all-or-nothing promise above holds for free. The refusals that depend on the
        # TEMPLATE do not: a battery whose third template trips one would launch and pay
        # for the first two, then abort — the partial set the docstring says cannot happen.
        # So every template-dependent refusal the launch makes is pre-flighted here over
        # the whole set, through the host's check, so nothing here is a copy.
        # Read through the SAME helper start_run uses, not compared raw: blank spells 'off',
        # and a surface may accept a bare `str`, so a raw compare refuses `cassette_mode=""` —
        # a battery that would have launched every template cleanly with cassettes off — and
        # blames the operator's templates for it. An unrecognised mode is refused HERE, before
        # the seeding check: that guard tests `!= "off"`, which a typo also satisfies, and
        # because this pre-flight fires before the loop, start_run's own refusal is never
        # reached — so a typo would be answered by telling the operator to archive templates
        # that are fine.
        try:
            battery_cassette_mode = _normalized_cassette_mode(cassette_mode)
        except ValidationFailedError as refused:
            raise ValidationFailedError(_battery_refusal(refused)) from refused
        # A replay serves ONE capture run's corpus, and a capture run recorded one template; a battery
        # launches many templates, so no single corpus could serve it.
        if battery_cassette_mode == "replay":
            raise ValidationFailedError(
                _battery_refusal(
                    ValueError(
                        "a replay serves the corpus of one capture run, which recorded one template, and a battery "
                        "launches every universal template; replay each template with start_run and its own corpus"
                    )
                )
            )
        # A variation model reaches only the templates with an llm axis, so per template it is honoured
        # or absent — and one no template could use is refused here, since no template's launch would.
        if variation_model is not None and templates and not any(_llm_axes(t) for t in templates):
            raise ValidationFailedError(
                _battery_refusal(
                    ValueError(
                        f"variation_model={variation_model!r} names the model that writes llm-generated axis values, "
                        "and none of the battery's templates has an llm axis; launch it without variation_model"
                    )
                )
            )
        # The refusals the dispatch itself makes, per template — a kind with no launcher, a battery
        # argument the kind cannot honour, a stored spec its model now refuses — through the same
        # function the dispatch calls, so a battery cannot launch its first templates and then be
        # refused one of them.
        for universal_template in templates:
            try:
                resolved = _launchable(
                    host,
                    universal_template,
                    n_variations=n_variations,
                    variation_model=variation_model if _llm_axes(universal_template) else None,
                    judge_model=judge_model,
                    judge_config_ids=None,
                    simulator_model=simulator_model,
                    cassette_mode=battery_cassette_mode,
                    overlays=None,
                    apparatus_settings=apparatus_settings,
                )
                if n_variations > 0:
                    # Every template's generating arms priced now, through the function each launch prices
                    # them with, so a battery cannot launch and pay for its first templates' generations and
                    # then be refused a later template's arm. The group is provisional: nothing joins it.
                    dispatched = _Dispatched(
                        universal_template,
                        resolved.kind,
                        resolved.launchable,
                        resolved.overlays,
                        resolved.kind_spec,
                        resolved.apparatus_settings,
                    )
                    await _priced_generating_arms(
                        host,
                        resolved.launchable,
                        _arm_requests(
                            host,
                            dispatched,
                            LaunchGroup(candidate_models=models),
                            subject_id=subject_id,
                            models=models,
                            k_runs=k_runs,
                            scope_id=scope_id,
                            n_variations=n_variations,
                            variation_model=variation_model if _llm_axes(universal_template) else None,
                            judge_model=judge_model,
                            judge_config_ids=None,
                            simulator_model=simulator_model,
                            cassette_mode=battery_cassette_mode,
                            cassette_corpus_id=None,
                            max_cost_usd=None,
                            max_metered_calls=None,
                        ),
                    )
            except ValidationFailedError as refused:
                raise ValidationFailedError(_battery_refusal(refused)) from refused
        # Every refusal an arm of any template would make, made here over the whole set before anything
        # launches — through the host's check, which calls what its launchers call. The subject is read
        # once for the check and not shared with the launches, which each capture their own.
        if templates:
            check = await preflight(subject_id, models)
            for universal_template in templates:
                try:
                    await check(universal_template, battery_cassette_mode)
                except ValidationFailedError as refused:
                    raise ValidationFailedError(_battery_refusal(refused)) from refused
        run_ids: list[str] = []
        for template in templates:
            # One launch per template, of one run per model: a template's arms answer that
            # template's cases, so they are what shares a group.
            runs = await start_run(
                host,
                template_id=template.id,
                scope_id=scope_id,
                subject_id=subject_id,
                models=models,
                k_runs=k_runs,
                n_variations=n_variations,
                variation_model=variation_model if _llm_axes(template) else None,
                judge_model=judge_model,
                simulator_model=simulator_model,
                cassette_mode=cassette_mode,
                apparatus_settings=apparatus_settings,
                admission=ticket,
            )
            run_ids += [run.id for run in runs]
        log.info(
            "eval.start_universal_battery subject=%s launched=%d runs=%s",
            subject_id,
            len(run_ids),
            ",".join(run_ids),
        )
        return run_ids
    finally:
        ticket.release()


def build_judge_service(
    host: EvalHost,
    template: EvalTemplate,
    judge_model: str,
    selection: dict[str, str] | None = None,
    *,
    judged_artifact: JudgedArtifact,
) -> RunJudge:
    """Build the :class:`~threetears.evals.run.judge_service.JudgeService` for a run, with its judge attribution.

    Resolves one :class:`~threetears.evals.contracts.models.JudgeConfig` per dim the kind's judge
    scores (:func:`~threetears.evals.contracts.models.scored_dim_ids`: the two reserved dual-score
    axes for a conversation, then every declared ``template.rubric`` dim) — so per-result scoring is
    a dict lookup, and a document run's attribution names no conversation axis it is never asked. A dim named in ``selection``
    resolves to exactly that config; every other dim resolves to the active
    one, which is what a launch that selected nothing gets for all of them.

    **Judge model resolves as a cascade in specificity**, through the host's client factory
    (:func:`~threetears.evals.run.judge_service.judge_clients_for_run` binds the run's pin; the
    service supplies each dim's ``config.model or None``).
    Least to most specific:

    1. the judge role's default — what a ``None`` model resolves to;
    2. the run-level pin (``judge_model``) — governs every dim that does not
       override it, including both reserved axes, which carry no config;
    3. a per-dim ``JudgeConfig.model`` — the narrowest statement, so it wins.

    Tier 1 is a launch-time resolution rather than a tier any run reaches
    here: a launch resolves the pin before calling this, so the factory is
    never handed ``None``. The consequence that reads as a bug — a
    run scored by more than one model while its record stamps a single
    ``judge_model`` — is the cascade working: ``judge_model`` records the
    run's pin, and a dim whose config names its own model is scored by that one.

    Args:
        host: The host: where the judge configs are read, the judge clients it builds, and how a
            raised judge call reads.
        template: The run's template, whose rubric names the scored dims.
        judge_model: The run's run-level judge fallback, **already resolved** —
            the caller pins it so the model the run records is the model that
            scores.
        selection: Optional ``{dim_id: config_id}`` naming the configuration
            this run is judged by, per dim. Validated here rather than at the
            caller because this is where the dim set and the config records
            are both in hand: a key must be one of this template's scored
            dims, the id must load, and the loaded config's own
            ``rubric_dim_id`` must be the dim it was named under — without
            that last check, naming one dim's config under another's key
            scores it with the wrong prompt and records a set that reads
            correct. An **archived** config is accepted deliberately, since
            naming a superseded record by id is how an A/B's control arm runs
            without un-archiving state every other run reads.
        judged_artifact: What the kind's judge reads, which picks the dims it scores. Required:
            the kind's declaration is the only thing that knows, and a default would attribute a
            document run's judges for axes no judge is asked.

    Returns:
        The run's judge: the service, the pin, the selection, the model each scored dim is requested
        from, and the configs — all from ONE config load, so the attribution a run records cannot
        disagree with what scores it.

    Raises:
        ValidationFailedError: ``selection`` names an unscored dim, a config
            that does not load in the template's scope, or a config authored for
            a different dim.
        ValueError: The host supplies no completion clients, so nothing could score the run, or
            ``judged_artifact`` declares a kind no judge reads.
    """
    if judged_artifact is JudgedArtifact.UNJUDGED:
        raise ValueError("an unjudged kind has no judge to build; its grade is the kind's own measures")
    clients = host.completion_clients("a judged run")
    storage = host.storage
    dim_ids = scored_dim_ids([dim.name for dim in template.rubric], judged_artifact)
    chosen = _resolve_selected_judge_configs(storage, selection, dim_ids, template.scope_id)
    configs = {}
    for dim_id in dim_ids:
        # Membership, not truthiness: a selected config is a record, and asking
        # whether the record is truthy asks a different question than whether the
        # launch named one.
        cfg = chosen[dim_id] if dim_id in chosen else storage.load_active_judge_config(dim_id, template.scope_id)
        if cfg is not None:
            configs[dim_id] = cfg

    # Attribution is computed from the SAME config load that builds the service, and
    # returned with it, rather than re-derived by a second caller. Loading twice would
    # leave a window in which an operator re-authors a config between the two reads —
    # and the run would then record an attribution that disagrees with what actually
    # scored it, which is the precise failure this field exists to make impossible.
    # The cascade itself is ``resolve_effective_judges``.
    return RunJudge(
        service=JudgeService(
            client_factory=judge_clients_for_run(clients, judge_model),
            configs=configs,
            failure_describer=host.failure_describer,
        ),
        model=judge_model,
        selection=dict(selection or {}),
        effective_judges=resolve_effective_judges(dim_ids, configs.get, judge_model),
        configs=configs,
    )


def _resolve_selected_judge_configs(
    storage: DefinitionStore, selection: dict[str, str] | None, dim_ids: list[str], scope_id: str
) -> dict[str, JudgeConfig]:
    """Load the judge configs a launch named, refusing every way the naming can be wrong.

    Three refusals, all before any spend, because each one otherwise produces a run
    that looks correct: an unscored dim silently selects nothing, an unloadable id
    silently falls back to the active config, and a config authored for a different
    dim scores that dim with someone else's prompt while the recorded set reads right.

    Args:
        storage: Where the judge configs are read.
        selection: ``{dim_id: config_id}`` as passed to the launch, or ``None``.
        dim_ids: This template's scored dims, from
            :func:`~threetears.evals.contracts.models.scored_dim_ids`.
        scope_id: The template's scope, where its judge configs live.

    Returns:
        ``{dim_id: config}`` over exactly ``selection``'s keys — empty when nothing
        was named.

    Raises:
        ValidationFailedError: A key is not a scored dim of this template, an id
            does not load in the scope, or a loaded config's
            ``rubric_dim_id`` is not the dim it was named under.
    """
    if not selection:
        return {}
    scored = set(dim_ids)
    resolved: dict[str, JudgeConfig] = {}
    for dim_id, config_id in selection.items():
        if dim_id not in scored:
            raise ValidationFailedError(
                f"judge_config_ids names dim {dim_id!r}, which this template does not score. Scored dims: {', '.join(dim_ids)}."
            )
        config = storage.load_judge_config(config_id, scope_id)
        if config is None:
            raise ValidationFailedError(
                f"judge_config_ids names judge config {config_id!r} for dim {dim_id!r}, which does not exist in scope {scope_id!r}."
            )
        if config.rubric_dim_id != dim_id:
            raise ValidationFailedError(
                f"judge config {config_id!r} was authored for dim {config.rubric_dim_id!r}, so it cannot "
                f"be named under {dim_id!r} — that dim would be scored by another dim's prompt."
            )
        resolved[dim_id] = config
    return resolved


__all__ = [
    "ArmPlan",
    "ArmPrice",
    "ArmQuote",
    "BatteryPreflight",
    "KindLauncher",
    "KindWiring",
    "LaunchArgument",
    "LaunchHost",
    "LaunchGroup",
    "LaunchPricer",
    "LaunchRequest",
    "LaunchableKind",
    "LaunchSettings",
    "RunJudge",
    "TemplatePreflight",
    "build_judge_service",
    "launch_as_group",
    "launch_run",
    "no_launcher_for",
    "settable_apparatus",
    "start_run",
    "start_universal_battery",
]
