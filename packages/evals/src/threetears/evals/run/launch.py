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
    RoleModelOrigin,
    resolve_effective_judges,
    scored_dim_ids,
)
from threetears.evals.contracts.out_of_run import OutOfRunBudget, plan_variation_calls
from threetears.evals.run.authoring import validated_kind_spec
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.run.ceilings import CeilingRaisedError, refuse_raised_ceiling
from threetears.evals.contracts.cassettes import CassetteMode
from threetears.evals.run.jobs import MAX_CONCURRENT_JOBS, EvalJobManager, JobTimeoutFactory, adaptive_job_timeout_s
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.run.metering import MeteredCallLedger
from threetears.evals.contracts.offload import run_blocking, wait_through_cancellation
from threetears.evals.run.runner import DEFAULT_CELL_TIMEOUT_S, KindFactory, RunCallbacks, RunnerOptions, execute_run
from threetears.evals.run.simulator import SIMULATOR_REQUEST_SETTINGS
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.host.subject import SubjectSnapshot
    from threetears.evals.contracts.host.sweepables import SweepableRegistry
    from threetears.evals.contracts.models import EvalTemplate, EvalTestCase, JudgeConfig, VariationCounts
    from threetears.evals.contracts.provider import PricedCompletion
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

    Read through :attr:`LaunchHost.settings` ONCE per launch, when it begins, rather than once at
    construction, because a host's settings hot-reload — and not again during it. The one snapshot
    decides every refusal the launch makes before it pays for anything (the arm and admission ceilings,
    the out-of-run cap its generation is priced against, the run caps its arms are priced against) and
    travels to every arm's tail on its request (:attr:`LaunchRequest.settings`), which records and
    enforces the run's ceilings and judge concurrency from it. So a reload part-way through cannot turn
    a launch that was admitted under one ceiling into a run refused, after its generation was paid for,
    under another — nor leave a run recording a ceiling it did not run under. A battery reads once for
    every template it launches.

    Attributes:
        max_launch_arms: How many runs one launch may start together. A group starts every member
            at once in one job slot, so this is the concurrency one launch adds.
        max_admitted_runs: How many runs may be admitted and unfinished at once across every launch
            — the ceiling admission refuses past rather than queueing behind.
        judge_concurrency: How many judge calls one cell makes at once.
        enforcement_enabled: Whether the cost and metered-call ceilings are enforced at all.
        max_cost_usd: The run cost ceiling a run inherits when its launch names none.
        max_metered_calls: The metered-call ceiling a run inherits when its launch names none, or
            ``None`` for a host that declares it has NO metered tools: its runs record a ceiling of
            ``0`` (origin ``none_declared``), a metered call on one is refused and counted, and a
            launch naming a ceiling is refused, since it would bound nothing.
        max_out_of_run_cost_usd: The most a launch's out-of-run calls — its case generation, which runs
            before any run exists and so under no run's cap — may together be priced at before they are
            made (:class:`~threetears.evals.contracts.out_of_run.OutOfRunBudget`). Enforced exactly when
            ``enforcement_enabled`` is. Per LAUNCH, as ``max_cost_usd`` is per run: a battery is one launch
            per template, so a battery of N generating templates may spend up to N times this out of run,
            as its runs may spend up to their count times their cap. An analysis generation is held to it
            too, per generation (:func:`~threetears.evals.ops.analysis_generate`): its calls run after the
            runs it reads, under no run's cap.
        judge_alternate_model: The judge a launch's arms are scored by instead of the judge role's default
            when that default IS one of the launch's candidate models — a model grading its own output —
            provided it is itself none of them (:func:`resolve_judge_pin`), spelled as the host's clients name
            the model they resolve. It never overrides a judge the launch named, nor a model a judge config
            pins per dim: those are choices. ``None`` substitutes nothing, and a run judged on a candidate's
            model says so on every surface that lists its judges
            (:func:`~threetears.evals.contracts.judge_attribution.judges_sharing_a_candidate_model`).
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
    max_metered_calls: int | None = Field(gt=0)
    max_out_of_run_cost_usd: float = Field(gt=0)
    judge_alternate_model: str | None = Field(default=None, min_length=1)
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
class PlannedJudge:
    """The judge one arm will be scored by, resolved before anything is built: what :func:`plan_judge` returns.

    The model a run spends its judging on is not the run-level pin alone — a dim whose config names its own
    model is scored by that one, and a pin a launch left unnamed resolves to the role's default, which moves
    over time — so an arm is priced, and its launcher held, by the judges it will actually be scored by.

    Attributes:
        model: The run-level judge pin, resolved — the launch's, or the role's default for one that named none.
        effective_judges: The model each scored dim is requested from, ``{dim_id: model}``, as a run records it.
        config_ids: The config each dim with one is judged by, ``{dim_id: config_id}`` — the ones the launch
            named and the active ones it inherited — as a run records its config set.
    """

    model: str
    effective_judges: Mapping[str, str]
    config_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze both maps, so a plan cannot move after it was priced.

        Raises:
            ValueError: ``model`` is blank, or a dim is scored by no model.
        """
        if not self.model.strip():
            raise ValueError("a planned judge names the model it resolved to; model is blank")
        if blank := sorted(dim for dim, model in self.effective_judges.items() if not model.strip()):
            raise ValueError(f"a planned judge names a model for every scored dim; {', '.join(blank)} name none")
        object.__setattr__(self, "effective_judges", MappingProxyType(dict(self.effective_judges)))
        object.__setattr__(self, "config_ids", MappingProxyType(dict(self.config_ids)))

    def __eq__(self, other: object) -> bool:
        """Equal when every field is: the frozen maps compare by content."""
        if not isinstance(other, PlannedJudge):
            return NotImplemented
        return (self.model, dict(self.effective_judges), dict(self.config_ids)) == (
            other.model,
            dict(other.effective_judges),
            dict(other.config_ids),
        )

    def __hash__(self) -> int:
        """Hashed over the same content the equality reads."""
        return hash((self.model, tuple(sorted(self.effective_judges.items())), tuple(sorted(self.config_ids.items()))))


@dataclass(frozen=True, kw_only=True)
class ArmPlan:
    """What one arm of a launch will run, as its kind says before its launcher runs.

    Every arm is priced before any launcher runs — a generating launch's cases do not exist until its
    launcher has paid for them, and a stored-case arm's launcher builds its clients and captures its
    subject — so the engine asks the kind (:attr:`LaunchableKind.plan_arm`) what the arm will run,
    prices that, and refuses an arm its cap cannot pay for before any launcher is called. The plan is a
    promise the launch tail holds the launcher to: a run freezing more cases than planned, or on
    another model, scored by other judges or driven by another simulator, is refused, since it was priced
    as something it is not.

    Attributes:
        case_count: The most cases the arm will run. For an arm over stored cases (``n_variations=0``),
            how many of the template's stored cases the kind plays; for a generating arm, at most
            ``n_variations`` — an upper bound, since generation de-duplicates and can keep fewer.
        candidate_model: The model the arm will run on: the one the launch named, or the kind's role
            default for an arm that named none.
        judge: The judges the arm will be scored by (:func:`plan_judge`), resolved; ``None`` for a kind whose
            grade is mechanical. Required, so a judged kind cannot be priced from unjudged history by omission.
        simulator_model: The model that will drive the arm's simulated user, resolved — the launch's pin, or
            the role's default; ``None`` for a kind that runs no simulator.
    """

    case_count: int
    candidate_model: str
    judge: PlannedJudge | None
    simulator_model: str | None

    def __post_init__(self) -> None:
        """Refuse a plan of no cases, or one naming no model.

        Raises:
            ValueError: ``case_count`` is below one, or ``candidate_model`` or ``simulator_model`` is blank.
        """
        if self.case_count < 1:
            raise ValueError(f"an arm plan runs at least one case; got case_count={self.case_count}")
        if not self.candidate_model.strip():
            raise ValueError("an arm plan names the model the arm runs on; candidate_model is blank")
        if self.simulator_model is not None and not self.simulator_model.strip():
            raise ValueError("an arm plan names the model its simulator runs on, or None; simulator_model is blank")


@dataclass(frozen=True, kw_only=True)
class ArmQuote:
    """One arm of a launch, as the engine asks the host's pricer to price it — stored-case arms and generating ones alike.

    Attributes:
        scope_id: The scope the arm's run will live in, whose history a history pricer reads.
        template_id: The template the arm runs.
        subject_id: The subject it measures.
        candidate_model: The model it runs on, as its plan named it.
        k_runs: Repeats of every case.
        case_count: The cases it runs, as its plan bounded them: the template's stored cases the kind plays
            for an arm that generates none, or at most ``n_variations`` for one that generates.
        n_variations: The cases the launch generates for the arm, or ``0`` for an arm over the template's
            stored cases (:attr:`case_source`).
        cassette_mode: Its cassette mode, normalised.
        judge: The judges it will be scored by, as its plan resolved them — the run-level pin, the model each
            scored dim is requested from (a dim's config can override the pin) and the config set — or ``None``
            for a kind with no judge. Judging is part of what a run spends, so an arm priced from history judged
            by cheaper models would be predicted low.
        simulator_model: The model that will drive its simulated user, as its plan resolved it; ``None`` for a
            kind with no simulator.
        apparatus_settings: The apparatus values the arm's rig is set up with, resolved — every setting its
            kind honours, defaults filled in, as its run will record them.
    """

    scope_id: str
    template_id: str
    subject_id: str
    candidate_model: str
    k_runs: int
    case_count: int
    n_variations: int
    cassette_mode: CassetteMode
    judge: PlannedJudge | None
    simulator_model: str | None
    apparatus_settings: Mapping[str, ApparatusSettingValue]

    @property
    def case_source(self) -> Literal["stored", "generated"]:
        """Where the arm's cases come from: the template's ``stored`` cases, or ``generated`` by its launch."""
        return "generated" if self.n_variations > 0 else "stored"


@dataclass(frozen=True, kw_only=True)
class ArmPrice:
    """What the host's pricer predicts one arm will cost, and how it knows.

    Attributes:
        predicted_usd: The predicted cost of the arm's whole run, in dollars, or ``None`` when the pricer
            cannot predict it — unknown, which the launch never reads as $0. It is the figure the arm's cap
            is held to, so a pricer whose prediction is a range returns its UPPER end: an arm admitted
            because its central estimate fits is one that runs past its cap whenever the run lands above
            the centre, after its generation was paid for.
        basis: How the prediction was made, or why there is none, in words a refusal quotes ("from 12
            priced past results, method usage-history").
        central_usd: For a pricer whose prediction is a range, its central estimate; ``None`` for one that
            states a single figure. Never held to the cap — reported beside :attr:`predicted_usd`, and what a
            cost pivot sets beside the cost a run later observed.
        low_usd: For a range, its lower end; ``None`` otherwise.
        method_id: The estimator, so two predictions by different methods are never silently compared;
            ``None`` for a pricer that names none.
    """

    predicted_usd: float | None
    basis: str
    central_usd: float | None = None
    low_usd: float | None = None
    method_id: str | None = None

    def __post_init__(self) -> None:
        """Refuse a negative or non-finite figure, a range out of order, and a blank basis.

        Raises:
            ValueError: A figure is negative or not finite; ``low_usd`` or ``central_usd`` is given without a
                prediction or out of order (``low <= central <= predicted``); or ``basis`` is blank.
        """
        for name in ("predicted_usd", "central_usd", "low_usd"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"a predicted cost is a finite amount, 0 or more; got {name}={value!r}")
        if self.predicted_usd is None and (self.central_usd is not None or self.low_usd is not None):
            raise ValueError("a range has an upper end, which is the prediction; predicted_usd is None")
        ordered = [v for v in (self.low_usd, self.central_usd, self.predicted_usd) if v is not None]
        if ordered != sorted(ordered):
            raise ValueError(
                f"a range runs low <= central <= predicted; got {self.low_usd!r}, {self.central_usd!r}, "
                f"{self.predicted_usd!r}"
            )
        if not self.basis.strip():
            raise ValueError("an arm price says how it was made, or why there is none; basis is blank")


class LaunchPricer(Protocol):
    """The host's prediction of what one arm of a launch will cost, before any launcher runs.

    Asked of EVERY arm the engine prices — one over the template's stored cases and one whose cases its
    launch generates alike (:attr:`ArmQuote.case_source`) — so one launch never prices two arms by two rules.

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
        launch_pricer: Predicts what each arm of a launch will cost (:class:`LaunchPricer`), so an arm its cap
            cannot pay for is refused before any launcher runs — before a generating launch's paid generation
            call, and before a stored-case arm's launcher builds anything. Every arm of every launch is priced
            through it, under an enforced cap. ``None`` for a host that prices no launch: its every arm is then
            unpriceable, which an arm with a cap its launch named proceeds under and one with an inherited cap
            is refused for — never run unpriced under a cap nobody chose.
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
            if undeclared := sorted(set(launchable.apparatus_settings) - settable):
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
            judge is chosen against (:func:`resolve_judge_pin`, and the launch tail that holds a launcher to
            it), so a candidate that is also the default judge moves every sibling's judge rather than its
            own run's alone.
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
        apparatus_settings: The host-declared apparatus values this arm's rig is set up with: every one
            the kind honours (:attr:`LaunchableKind.apparatus_settings`), the launch's value where it set one
            and the kind's default where it did not, already validated; ``{}`` only for a kind whose rig a
            launch cannot set. The launcher sets its rig up from them, reading no default of its own, and the
            run records exactly these (``EvalRun.apparatus_settings``).
        generation_budget: For a launch that generates (``n_variations`` > 0), the out-of-run budget its
            generation's calls are priced against and ledgered through — one for the whole launch, shared
            by every arm, capped at the host's ``max_out_of_run_cost_usd``. A launcher hands it to
            :func:`~threetears.evals.gen.generate_variations` as ``budget``; a generation a model wrote
            whose calls this budget never ledgered is refused at the launch tail. ``None`` for a launch
            that generates nothing.
        arm_plan: What the kind planned this arm to run (:attr:`LaunchableKind.plan_arm`) — the plan the arm
            was priced at, and the launch tail holds its run to — for an arm over stored cases and a generating
            one alike. ``None`` for a kind that plans nothing.
        arm_price: What the host's pricer predicted for this arm when :func:`price_arms` admitted it under an
            enforced cap — the mark that the arm was priced. ``None`` when no cap is in force (nothing is priced)
            or before the arm is priced; under an enforced cap the launch tail refuses an arm that carries none,
            so a launch composed outside :func:`start_run` cannot run an arm nobody priced.
        launch_group: The launch this run is prepared into; the launch starts it with its siblings.
        settings: The host's launch settings as the launch read them, once, when it began — what its
            refusals were made under, and what :func:`launch_run` records and enforces the run's ceilings
            and judge concurrency from. A launcher reads the host's settings from here, never afresh.
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
    arm_price: ArmPrice | None
    launch_group: LaunchGroup
    settings: LaunchSettings

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
        plan_arm: What one arm of a launch of this kind will run, asked before the launcher is — the arm's
            case count, model, judges and simulator (:class:`ArmPlan`) — so the engine can price the arm and
            refuse one its cap cannot pay for before any launcher runs. Asked of EVERY arm: for one over stored
            cases it counts the template's stored cases the kind plays, and for a generating one it bounds the
            cases at ``n_variations``; a judged kind resolves its judges with :func:`plan_judge`. Called off the
            event loop, through the host's blocking executor, so it may read the host's store. **It is the home
            of the kind's request-level refusals**: every refusal that depends only on the request — an arm
            naming no model on a kind with no role default (:func:`require_candidate_model`, the refusal the
            launch tail makes too), a judge or simulator the kind needs and the launch cannot give it, a
            judge-config selection that does not fit the template, a template with no case to play — is a
            ``ValidationFailedError`` raised here, since every arm is planned before any is priced: an operator
            who forgot something hears that rather than "cannot be priced". The launcher keeps only the
            refusals that need what it builds. ``None`` for a kind that plans nothing: under an enforced cap its
            every arm is unpriceable — refused under an inherited cap, saying the launch may also be missing
            something its kind needs, and run under one its launch named — and nothing holds its launcher to a
            plan.
        apparatus_settings: The host-declared apparatus dimensions this kind's launcher sets its rig up
            from, each with the value its rig takes when a launch sets none — the kind's standing rig. A
            launch setting any other is refused at the dispatch; the :class:`LaunchHost` refuses a name the
            profile does not declare as one of the host's own apparatus dimensions. The dispatch RESOLVES a
            launch's settings against these defaults, so :attr:`LaunchRequest.apparatus_settings` names every
            one, the launcher reads them there with no default of its own, and the run records — and its
            context key hashes — the rig as it was set up: a launch naming a default and one leaving it out
            are one condition, as a ``kind_spec`` stating a default and one omitting it are. Empty for a kind
            whose rig a launch cannot set.
    """

    launch: KindLauncher
    unhonoured_launch_arguments: frozenset[LaunchArgument] = frozenset()
    plan_arm: Callable[[LaunchRequest], ArmPlan] | None = None
    apparatus_settings: Mapping[str, ApparatusSettingValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        """Refuse a name that is not a launch argument, and a default no run could store.

        Raises:
            ValueError: An entry of ``unhonoured_launch_arguments`` is not a :data:`LaunchArgument`, or an
                apparatus default is not a string, a bool or a finite number.
        """
        try:
            defaults = _APPARATUS_SETTINGS.validate_python(dict(self.apparatus_settings))
        except ValidationError as e:
            raise ValueError(
                f"apparatus_settings defaults a setting to a value no run could record: {e.errors()[0]['msg']} "
                f"({e.errors()[0]['loc']})"
            ) from e
        # Frozen as validated, so a caller's mapping changing afterwards cannot move a kind's standing rig.
        object.__setattr__(self, "apparatus_settings", MappingProxyType(defaults))
        if unknown := sorted(set(self.unhonoured_launch_arguments) - set(get_args(LaunchArgument))):
            raise ValueError(
                f"unhonoured_launch_arguments names {', '.join(unknown)}, which are not launch arguments; "
                f"the arguments a kind can decline are {', '.join(get_args(LaunchArgument))}"
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

    @property
    def planned(self) -> PlannedJudge:
        """This judge as a plan states one, so the launch tail can hold it to the plan its arm was priced at."""
        return PlannedJudge(model=self.model, effective_judges=self.effective_judges, config_ids=self.config_ids)


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
        The settings as the run stores them: every one the kind honours, the launch's value where it set
        one and the kind's default where it did not — ``{}`` only for a kind whose rig a launch cannot set.

    Raises:
        ValidationFailedError: A setting the kind does not honour, or a value that is not a string, a bool
            or a finite number.
    """
    if not apparatus_settings:
        return dict(launchable.apparatus_settings)
    if unhonoured := sorted(set(apparatus_settings) - set(launchable.apparatus_settings)):
        honoured = ", ".join(sorted(launchable.apparatus_settings)) or "none"
        raise ValidationFailedError(
            f"template {template.id!r} is a {template.candidate_kind!r} template, and that kind's launcher sets its rig "
            f"up from no apparatus setting named {', '.join(repr(name) for name in unhonoured)} (it reads: {honoured}); "
            "a setting nothing reads would be recorded as a rig nobody built"
        )
    try:
        # Resolved against the kind's standing rig: what the run records is the rig as it was set up.
        return {**launchable.apparatus_settings, **_APPARATUS_SETTINGS.validate_python(dict(apparatus_settings))}
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
            generation needs and the launch does not name or one nothing would call, an apparatus setting
            the kind does not honour, or the kind's models refuse the overlays or the spec.
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


def _refuse_repeated_models(models: Sequence[str]) -> None:
    """Refuse a model named twice: each model is one arm.

    Args:
        models: The launch's candidate models.

    Raises:
        ValidationFailedError: A model is named more than once.
    """
    if repeated := sorted({model for model in models if models.count(model) > 1}):
        raise ValidationFailedError(
            f"{', '.join(repr(model) for model in repeated)} named more than once; each model is one arm, so "
            "a second mention would measure that arm twice — raise k_runs for more repeats"
        )


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
            needs and the launch does not name or one nothing would call, an apparatus setting the kind does
            not honour, a model named twice, or a replay corpus that is not a capture of this template in this
            scope.
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
    _refuse_repeated_models(models)
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
        max_cost_usd: Optional per-run cost-cap override; must be ``> 0`` and at most the host's configured
            ``max_cost_usd`` — a launch may only lower the host's ceiling.
        max_metered_calls: Optional per-run metered-call ceiling override; must be ``> 0`` and at most the
            host's configured ``max_metered_calls``.
        scope_id: The scope the template is read in and the launch's runs live in.
        launch_group: Prepare the runs into this launch instead of starting them; the caller starts
            the group once every arm is prepared. When omitted, this call is the whole launch: it
            forms the group and starts it through :func:`launch_as_group`.
        admission: A reservation the caller already holds, of which this launch takes its runs'
            share instead of asking for room of its own. Ignored with ``launch_group``, whose
            caller's admission covers it. When omitted, a launch that owns its group is admitted here.

    **Every arm is priced before any launcher runs, by one rule.** Before calling the launcher this asks
    the kind what each arm will run (:attr:`LaunchableKind.plan_arm` — the template's stored cases it
    plays, or at most ``n_variations`` generated ones), asks the host's pricer what that will cost
    (:attr:`LaunchHost.launch_pricer`) and refuses an arm predicted above the cap its run will be held
    to — or one nothing can predict (no pricer, no plan, or no history the pricer can bound) whose cap
    the run would merely inherit, since unknown is not $0 and an inherited cap is nobody's decision
    about this run (a cap the launch named is that decision, and the run goes ahead under it). The same
    rule prices an arm over stored cases and one whose cases its launch generates, so a launch never
    prices two arms two ways, and a host prices no arm of its own. A generating launch's generation's
    own calls are priced in turn before the first is made, against the host's out-of-run cap
    (:attr:`LaunchRequest.generation_budget`).

    Returns:
        The persisted runs, one per model in the order given (status ``pending``).

    Raises:
        NotFoundError: The template is not found.
        AdmissionRefusedError: The runs would pass the host's admission ceiling — raised before the
            template is read, so the refused launch prepared nothing.
        ValidationFailedError: A model named twice, more runs than one launch may start, a negative
            ``n_variations``, a variation model the generation needs and the launch does not name or one
            nothing would call, a ``k_runs`` outside the run's bounds, a non-positive ``max_cost_usd``
            or ``max_metered_calls`` or one above the host's configured ceiling (or any ``max_metered_calls`` on a host declaring no metered tools),
            a ``cassette_mode`` that is not a mode, a replay naming no corpus
            or a corpus that is no capture of this template in this scope, a template naming a
            kind with no launcher, a launch argument that kind cannot honour, an overlay the kind's
            model refuses (named by field), an apparatus setting the kind does not honour, an arm
            predicted above its cap or unpriceable under an inherited one, a plan the kind refuses to make,
            or any refusal the kind's launcher makes.
    """
    # The host's settings, read ONCE for the whole launch: every refusal below and every arm's tail
    # reads this snapshot, so a hot reload part-way cannot refuse, after its generation was paid for, a
    # launch admitted under the figures it started with.
    return await _start_run(
        host,
        host.settings(),
        template_id=template_id,
        subject_id=subject_id,
        models=models,
        k_runs=k_runs,
        n_variations=n_variations,
        variation_model=variation_model,
        judge_model=judge_model,
        judge_config_ids=judge_config_ids,
        simulator_model=simulator_model,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
        overlays=overlays,
        apparatus_settings=apparatus_settings,
        max_cost_usd=max_cost_usd,
        max_metered_calls=max_metered_calls,
        scope_id=scope_id,
        launch_group=launch_group,
        admission=admission,
    )


def _refuse_raised_ceilings(
    settings: LaunchSettings, *, max_cost_usd: float | None, max_metered_calls: int | None
) -> None:
    """Refuse a launch whose per-run override would RAISE the host's ceiling, before anything is paid for.

    The rule is :func:`~threetears.evals.run.ceilings.refuse_raised_ceiling`, stated once for both
    currencies; this applies it to a launch's two overrides against its one settings snapshot and
    turns its refusal into the launch's own error. Every launch surface — a single launch, its quote and
    a battery — reaches it, so none can admit a raised ceiling another refuses.

    Args:
        settings: The launch's settings snapshot.
        max_cost_usd: The per-run cost-cap override.
        max_metered_calls: The per-run metered-call ceiling override.

    Raises:
        ValidationFailedError: Either override is above the host's configured ceiling.
    """
    try:
        refuse_raised_ceiling(
            max_cost_usd,
            configured=settings.max_cost_usd,
            name="max_cost_usd",
            configured_name=settings.name_of("max_cost_usd"),
        )
        if settings.max_metered_calls is not None:
            refuse_raised_ceiling(
                max_metered_calls,
                configured=settings.max_metered_calls,
                name="max_metered_calls",
                configured_name=settings.name_of("max_metered_calls"),
            )
    except CeilingRaisedError as refused:
        raise ValidationFailedError(str(refused)) from refused


def _refuse_launch_arguments(
    host: LaunchHost,
    settings: LaunchSettings,
    *,
    k_runs: int,
    max_cost_usd: float | None,
    max_metered_calls: int | None,
    cassette_mode: str | None,
    cassette_corpus_id: str | None,
) -> CassetteMode:
    """The refusals a launch makes of its arguments alone, before it reads anything — one place, for the launch and its quote.

    Args:
        host: The host, named in a refusal.
        settings: The launch's settings snapshot.
        k_runs: Repeats per case.
        max_cost_usd: The per-run cost-cap override.
        max_metered_calls: The per-run metered-call ceiling override.
        cassette_mode: The cassette mode, unnormalised.
        cassette_corpus_id: The corpus a replay serves.

    Returns:
        The cassette mode, normalised.

    Raises:
        ValidationFailedError: A non-positive ``max_cost_usd`` or ``max_metered_calls`` (or any
            ``max_metered_calls`` on a host declaring no metered tools), either one above the host's configured
            ceiling, a ``k_runs`` outside the run's bounds, a ``cassette_mode`` that is not a mode, or a corpus
            the mode cannot use.
    """
    # Per-run cost-cap override: a non-positive cap would stop the
    # run before the first result — reject it as a run-parameter error up
    # front, before any snapshot / generation spend.
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValidationFailedError(f"max_cost_usd must be > 0 (got {max_cost_usd})")
    # Same reasoning one currency over: a ceiling of zero or less would refuse the
    # candidate's first metered call and measure a candidate that could not search.
    if max_metered_calls is not None and max_metered_calls <= 0:
        raise ValidationFailedError(f"max_metered_calls must be > 0 (got {max_metered_calls})")
    # A host that declares no metered tools has nothing for a ceiling to bound, and recording one would
    # claim a bound the run never had.
    if max_metered_calls is not None and settings.max_metered_calls is None:
        raise ValidationFailedError(
            f"max_metered_calls={max_metered_calls} names a metered-call ceiling, and host "
            f"{host.eval_host.profile.host_id!r} declares no metered tools, so it would bound nothing; launch "
            "without max_metered_calls"
        )
    _refuse_raised_ceilings(settings, max_cost_usd=max_cost_usd, max_metered_calls=max_metered_calls)
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
    return mode


async def _start_run(
    host: LaunchHost,
    settings: LaunchSettings,
    *,
    template_id: str,
    subject_id: str,
    models: list[str],
    k_runs: int,
    n_variations: int,
    variation_model: str | None,
    judge_model: str | None,
    judge_config_ids: dict[str, str] | None,
    simulator_model: str | None,
    cassette_mode: str | None,
    cassette_corpus_id: str | None,
    overlays: Mapping[str, Any] | None,
    apparatus_settings: Mapping[str, Any] | None,
    max_cost_usd: float | None,
    max_metered_calls: int | None,
    scope_id: str,
    launch_group: LaunchGroup | None,
    admission: AdmissionTicket | None,
    priced: _PricedArms | None = None,
) -> list[EvalRun]:
    """:func:`start_run` under a settings snapshot its caller read — the launch's own, or a battery's.

    Args and Returns as :func:`start_run`; ``settings`` is the one read every refusal and every arm's
    tail is made under. ``priced`` is the battery's: its arms as its pre-flight already planned and priced
    them under the same snapshot, which this launch then carries rather than pricing a second time —
    ``None`` for a launch that prices its own arms.

    Raises:
        See :func:`start_run`.
    """
    # NOTE: the "at least one model" refusal is NOT here, because it is not the same
    # refusal for every kind — each kind's plan (or, for one that plans nothing, its launcher) makes its own.
    mode = _refuse_launch_arguments(
        host,
        settings,
        k_runs=k_runs,
        max_cost_usd=max_cost_usd,
        max_metered_calls=max_metered_calls,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
    )
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
        """Price every arm, then hand each arm, one per model, to the kind's launcher.

        Args:
            group: The launch the runs are prepared into.
            dispatched: What the dispatch resolved.

        Returns:
            The prepared runs, one per model in the order given.
        """
        requests = _arm_requests(
            host,
            settings,
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
        # Every arm priced before the first launcher runs, whether it generates or not: a launcher is what
        # pays for the generation the arms share and what builds an arm's clients, so pricing an arm after
        # it would refuse a launch already billed. Priced exactly once — here, or by the battery's
        # pre-flight, whose plans the arms then carry.
        requests = (
            await price_arms(host, dispatched.launchable, requests) if priced is None else priced.carried_by(requests)
        )
        return [await dispatched.launchable.launch(request) for request in requests]

    async def dispatched_once() -> _Dispatched:
        """The launch's dispatch — or, for a battery's launch, the one its pre-flight priced, template and all.

        Returns:
            What the dispatch resolved.
        """
        # A battery's launch runs the template its pre-flight loaded and priced, never one re-read since:
        # a template edited mid-battery would otherwise launch under plans made for another.
        return priced.dispatched if priced is not None else await dispatch()

    # Preparing arms into a caller's group: that caller admitted the launch, and it starts the
    # group once every arm is prepared or abandons it on a refusal, so this call only prepares.
    if launch_group is not None:
        return await prepare(launch_group, await dispatched_once())

    async def form() -> tuple[LaunchGroup, _Dispatched]:
        """Make the template's refusals, then form the launch's group: one arm per model.

        Returns:
            The group, and what the dispatch resolved.
        """
        dispatched = await dispatched_once()
        return LaunchGroup(candidate_models=models), dispatched

    # This call is the whole launch. A caller holding a ticket for several launches (the battery)
    # hands this one its share rather than letting it compete for room with launches that arrived
    # after the caller was admitted.
    return await launch_as_group(
        host,
        max(1, len(models)),
        settings=settings,
        form=form,
        prepare=prepare,
        admission=admission,
        event="eval.start_run",
    )


def _generation_budget(
    host: LaunchHost,
    settings: LaunchSettings,
    *,
    scope_id: str,
    template_id: str,
    subject_id: str,
    launch_group_id: str,
) -> OutOfRunBudget:
    """The out-of-run budget one launch's generation is admitted under: the host's cap, ledgered under the launch.

    One construction for both of its readers — the launch that hands it to its launcher, and the battery
    that prices every template's generation against it before any template launches — so the battery
    cannot pass a generation its launch would refuse.

    Args:
        host: The host, whose store ledgers the calls.
        settings: The launch's settings snapshot; the cap is enforced exactly when its enforcement is.
        scope_id: The scope the generation is for.
        template_id: The template whose cases it writes.
        subject_id: The launch's subject.
        launch_group_id: The launch's group, stamped on every ledger row.

    Returns:
        The budget, nothing yet admitted.
    """
    return OutOfRunBudget(
        host.eval_host.storage,
        scope_id=scope_id,
        cap_usd=settings.max_out_of_run_cost_usd if settings.enforcement_enabled else None,
        template_id=template_id,
        subject_id=subject_id,
        launch_group_id=launch_group_id,
        blocking_executor=host.eval_host.blocking_executor,
    )


def _arm_requests(
    host: LaunchHost,
    settings: LaunchSettings,
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
        host: The host, whose store ledgers the generation.
        settings: The launch's settings snapshot, which caps the generation and travels on every request.
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
        budget = _generation_budget(
            host,
            settings,
            scope_id=scope_id,
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
            arm_price=None,
            launch_group=group,
            settings=settings,
        )
        for arm_model in arm_models
    ]


async def _planned(host: LaunchHost, launchable: LaunchableKind, request: LaunchRequest) -> ArmPlan | None:
    """Ask the kind what one arm will run, and refuse a plan the arm contradicts.

    Args:
        host: The host, whose blocking executor the plan is asked through — a plan may read the store.
        launchable: The kind's registry entry.
        request: The arm.

    Returns:
        The plan, or ``None`` for a kind that plans nothing (:attr:`LaunchableKind.plan_arm`).

    Raises:
        ValidationFailedError: The kind cannot plan the arm (its own refusal).
        ValueError: The plan contradicts what the arm named — another model, another judge pin, a config
            selection it did not plan, or another simulator — a kind defect.
    """
    plan_arm = launchable.plan_arm
    if plan_arm is None:
        return None
    plan = await run_blocking(host.eval_host.blocking_executor, plan_arm, request)
    kind = request.kind
    if request.candidate_model is not None and plan.candidate_model != request.candidate_model:
        raise ValueError(
            f"kind {kind!r} planned an arm the launch named {request.candidate_model!r} on "
            f"{plan.candidate_model!r}; an arm runs on the model the launch named"
        )
    if request.judge_model is not None and (plan.judge is None or plan.judge.model != request.judge_model):
        raise ValueError(
            f"kind {kind!r} planned an arm the launch pinned to judge {request.judge_model!r} under "
            f"{plan.judge.model if plan.judge is not None else 'no judge'!r}; a pinned judge is the judge"
        )
    if request.judge_config_ids and (
        plan.judge is None
        or any(plan.judge.config_ids.get(dim) != config for dim, config in request.judge_config_ids.items())
    ):
        raise ValueError(
            f"kind {kind!r} planned an arm the launch selected judge configs {request.judge_config_ids} for under "
            "another config set; plan its judge with plan_judge from the request's judge_config_ids"
        )
    if request.simulator_model is not None and plan.simulator_model != request.simulator_model:
        raise ValueError(
            f"kind {kind!r} planned an arm the launch pinned to simulator {request.simulator_model!r} on "
            f"{plan.simulator_model!r}; a pinned simulator is the simulator"
        )
    return plan


def _arm_described(request: LaunchRequest) -> str:
    """One arm, as a pricing refusal names it: its model, its cases by source, its repeats and its template.

    Args:
        request: The arm, planned when its kind plans.

    Returns:
        The description.
    """
    plan = request.arm_plan
    if plan is None:
        model = repr(request.candidate_model) if request.candidate_model is not None else "its kind's default model"
        return f"the arm on {model} of template {request.template.id!r} (its kind plans no arm)"
    source = "generated" if request.n_variations > 0 else "stored"
    return (
        f"the arm on {plan.candidate_model!r} ({plan.case_count} {source} case(s) x {request.k_runs} repeat(s) "
        f"of template {request.template.id!r})"
    )


async def _arm_price(host: LaunchHost, request: LaunchRequest) -> ArmPrice:
    """What the host's pricer predicts ``request``'s arm will cost — or why nothing can.

    An arm is unpriceable when the host prices no launch, when its kind plans no arm (nothing says what
    it will run), or when the pricer predicts nothing; each is the same unknown to the caller, which is
    the point: one rule reads it.

    Args:
        host: The host, whose pricer predicts.
        request: The arm, planned when its kind plans.

    Returns:
        The price, or a price with no prediction and the reason there is none.
    """
    pricer = host.launch_pricer
    if pricer is None:
        return ArmPrice(
            predicted_usd=None,
            basis=f"host {host.eval_host.profile.host_id!r} prices no launch (LaunchHost.launch_pricer)",
        )
    plan = request.arm_plan
    if plan is None:
        return ArmPrice(
            predicted_usd=None,
            basis=f"kind {request.kind!r} plans no arm (LaunchableKind.plan_arm), so nothing says what it will run",
        )
    return await run_blocking(
        host.eval_host.blocking_executor,
        pricer,
        ArmQuote(
            scope_id=request.scope_id,
            template_id=request.template.id,
            subject_id=request.subject_id,
            candidate_model=plan.candidate_model,
            k_runs=request.k_runs,
            case_count=plan.case_count,
            n_variations=request.n_variations,
            cassette_mode=request.cassette_mode,
            judge=plan.judge,
            simulator_model=plan.simulator_model,
            apparatus_settings=request.apparatus_settings,
        ),
    )


#: What the one pricing rule made of an arm: admitted under its cap; refused (predicted above it, or
#: unpriceable under an inherited one); unpriceable and run under a cap its launch named; or not held to
#: any cap, the host enforcing none.
ArmOutcome = Literal["admitted", "refused", "unpriced-under-chosen-cap", "uncapped"]


@dataclass(frozen=True, kw_only=True)
class ArmVerdict:
    """One arm of a launch as the engine's one pricing rule judged it: its plan, its price, its cap and the outcome.

    What :func:`quote_launch` returns per arm, and what :func:`price_arms` refuses on — the same judgement,
    so a quote and the launch it quotes cannot disagree about an arm.

    Attributes:
        described: The arm, as a refusal names it.
        candidate_model: The model it runs on — its plan's, or the one the launch named when its kind plans
            nothing (``None`` there for an arm on the kind's unnamed default).
        plan: What its kind planned it to run, or ``None`` for a kind that plans nothing.
        price: What the host's pricer predicted, or why nothing could.
        cap_usd: The cap its run will be held to; ``None`` when the host enforces no cap.
        cap_origin: ``chosen`` when the launch named the cap, ``inherited`` from the host's; ``None`` uncapped.
        outcome: What the rule made of it (:data:`ArmOutcome`).
        refusal: The refusal the launch makes of it, word for word; ``None`` unless ``outcome`` is ``refused``.
    """

    described: str
    candidate_model: str | None
    plan: ArmPlan | None
    price: ArmPrice
    cap_usd: float | None
    cap_origin: Literal["chosen", "inherited"] | None
    outcome: ArmOutcome
    refusal: str | None


def _cap_of(request: LaunchRequest) -> tuple[float | None, Literal["chosen", "inherited"] | None]:
    """The cap ``request``'s run will be held to, and where it came from — from the launch's one settings read.

    Args:
        request: The arm.

    Returns:
        The cap and its origin, or ``(None, None)`` when the host enforces no cap.
    """
    settings = request.settings
    cap = EvalRunCostCap.resolve_effective_ceiling(
        request.max_cost_usd,
        configured_max_cost_usd=settings.max_cost_usd,
        enforcement_enabled=settings.enforcement_enabled,
    )
    origin = EvalRunCostCap.resolve_ceiling_origin(
        request.max_cost_usd, enforcement_enabled=settings.enforcement_enabled
    )
    if cap is None or origin == "uncapped":
        return None, None
    return cap, origin


def _judged(request: LaunchRequest, price: ArmPrice) -> ArmVerdict:
    """The one pricing rule, over one planned arm and its price.

    An arm is refused when its predicted cost is above the cap its run will be held to, or when nothing
    predicts it and that cap is one the run would merely inherit (the host's configured ceiling, which nobody
    chose for this run): unknown is not $0. An arm with a cap the launch named and no prediction goes ahead —
    the named cap is the most the operator chose to risk on a run nobody could price, and the run's own cost
    cap enforces it as the spend arrives.

    Args:
        request: The arm, planned when its kind plans.
        price: Its price.

    Returns:
        The verdict.
    """
    cap, origin = _cap_of(request)
    what = _arm_described(request)
    plan = request.arm_plan
    model = plan.candidate_model if plan is not None else request.candidate_model

    def verdict(outcome: ArmOutcome, refusal: str | None = None) -> ArmVerdict:
        return ArmVerdict(
            described=what,
            candidate_model=model,
            plan=plan,
            price=price,
            cap_usd=cap,
            cap_origin=origin,
            outcome=outcome,
            refusal=refusal,
        )

    if cap is None:
        return verdict("uncapped")
    refused = (
        "Refused before the generation was paid for" if request.n_variations > 0 else "Refused before any launcher ran"
    )
    if price.predicted_usd is None:
        if origin == "chosen":
            return verdict("unpriced-under-chosen-cap")
        # A kind that plans no arm also makes its request-level refusals only in its launcher, so an arm
        # refused here may be missing something besides a price; the operator hears that now rather than
        # after naming a cap.
        unplanned = (
            " The launch may also be missing something its kind needs: a kind that plans no arm refuses that "
            "only once its launcher runs."
            if plan is None
            else ""
        )
        return verdict(
            "refused",
            f"{what} cannot be priced: {price.basis}. Its cost is unknown, not $0, and its ${cap:.2f} cap is "
            f"inherited from {request.settings.name_of('max_cost_usd')}, which nobody chose for it. Launch it naming "
            f"max_cost_usd (at most ${request.settings.max_cost_usd:.2f}; a launch may only lower the host's "
            "ceiling) — the most you will risk on a run nobody can price; what it costs then prices the next "
            f"launch.{unplanned} {refused}",
        )
    if price.predicted_usd > cap:
        ceiling = request.settings.max_cost_usd
        raise_it = (
            f"a larger max_cost_usd, up to the host's ${ceiling:.2f} ceiling, raises the cap"
            if cap < ceiling
            else f"the cap is the host's ceiling, which a launch cannot raise — only the host's operator can, through "
            f"{request.settings.name_of('max_cost_usd')}"
        )
        return verdict(
            "refused",
            f"{what} is predicted to cost ${price.predicted_usd:.2f} ({price.basis}), above its ${cap:.2f} cap. "
            f"Fewer cases, a smaller k_runs or a cheaper model brings it under; {raise_it}. {refused}",
        )
    return verdict("admitted")


async def _planned_arms(
    host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]
) -> list[LaunchRequest]:
    """Every arm, carrying what its kind planned — all planned before any is priced.

    Planning first is what puts a kind's request-level refusals (:attr:`LaunchableKind.plan_arm`) ahead of
    every pricing refusal: an operator who forgot something the kind needs hears that, not "cannot be priced".

    Args:
        host: The host, whose executor each plan is asked through.
        launchable: The kind's registry entry.
        requests: The launch's arms, in order.

    Returns:
        The requests with their plans.

    Raises:
        ValidationFailedError: The kind cannot plan an arm.
        ValueError: A plan contradicts what its arm named.
    """
    return [replace(request, arm_plan=await _planned(host, launchable, request)) for request in requests]


async def price_arms(
    host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]
) -> list[LaunchRequest]:
    """Plan and price every arm of one launch by the engine's one rule, refusing before any launcher runs.

    The step every launch takes between building its arms' requests and calling their launchers —
    :func:`start_run` and the battery call it, and a host composing its own launch through
    :func:`launch_as_group` must call it too: under an enforced cap the launch tail refuses an arm that did
    not come through here (:attr:`LaunchRequest.arm_price`). Every arm is planned before any is priced, and
    every arm is priced before the first launcher runs. With the host's enforcement off no cap is in force
    and nothing is priced; every arm is still planned, so the launch tail holds its launcher to its plan.

    Args:
        host: The host: its pricer predicts.
        launchable: The kind's registry entry, whose ``plan_arm`` plans each arm.
        requests: The launch's arms, in order; every one carries the same ``max_cost_usd`` and the same
            settings snapshot.

    Returns:
        The requests, each carrying its plan (``None`` for a kind that plans nothing) and, under an enforced
        cap, its price.

    Raises:
        ValidationFailedError: The kind cannot plan an arm; an arm predicted above its cap, or one
            unpriceable under an inherited cap.
        ValueError: The kind planned an arm contradicting what it named, or ``requests`` is empty.
    """
    if not requests:
        raise ValueError("a launch prices at least one arm; requests is empty")
    planned = await _planned_arms(host, launchable, requests)
    cap, _origin = _cap_of(planned[0])
    if cap is None:
        return planned
    priced: list[LaunchRequest] = []
    for request in planned:
        price = await _arm_price(host, request)
        judged = _judged(request, price)
        if judged.refusal is not None:
            raise ValidationFailedError(judged.refusal)
        priced.append(replace(request, arm_price=price))
    return priced


@dataclass(frozen=True, kw_only=True)
class LaunchQuote:
    """What a launch would make of its arms' prices, made by the launch's own steps and launching nothing.

    Attributes:
        template_id: The template.
        subject_id: The subject.
        scope_id: The scope whose history a history pricer reads.
        cassette_mode: The cassette mode, normalised.
        k_runs: Repeats per case.
        n_variations: Cases the launch would generate first; ``0`` for stored cases.
        case_count: The case count the quote put in place of each plan's, for a hypothetical grid; ``None``
            when every arm is quoted at its plan, as the launch would price it.
        arms: Every arm's verdict, in arm order.
    """

    template_id: str
    subject_id: str
    scope_id: str
    cassette_mode: CassetteMode
    k_runs: int
    n_variations: int
    case_count: int | None
    arms: tuple[ArmVerdict, ...]


async def quote_launch(
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
    case_count: int | None = None,
) -> LaunchQuote:
    """What :func:`start_run` with the same arguments would make of its arms' prices — read-only.

    The launch's own steps, in its order, up to the point where it would refuse or call a launcher: its
    argument refusals, its dispatch, each arm's request, each arm's plan (the kind's request-level refusals
    with it) and each arm's price through the host's ``launch_pricer``, judged by the rule
    :func:`price_arms` refuses on. Nothing is admitted, no launcher is called, nothing is generated and nothing
    is spent. Where the launch would refuse an arm on its price, the quote reports the refusal instead of
    raising it; every refusal the launch makes before pricing is raised, as the launch raises it. With the
    host's enforcement off the launch prices nothing; the quote prices every arm anyway and reports it
    ``uncapped``.

    Args:
        host: The host.
        template_id: As :func:`start_run` takes it.
        subject_id: As :func:`start_run` takes it.
        models: As :func:`start_run` takes it.
        k_runs: As :func:`start_run` takes it.
        n_variations: As :func:`start_run` takes it.
        variation_model: As :func:`start_run` takes it.
        judge_model: As :func:`start_run` takes it.
        judge_config_ids: As :func:`start_run` takes it.
        simulator_model: As :func:`start_run` takes it.
        cassette_mode: As :func:`start_run` takes it.
        cassette_corpus_id: As :func:`start_run` takes it.
        overlays: As :func:`start_run` takes it.
        apparatus_settings: As :func:`start_run` takes it.
        max_cost_usd: As :func:`start_run` takes it.
        max_metered_calls: As :func:`start_run` takes it.
        scope_id: As :func:`start_run` takes it.
        case_count: A case count to quote every planned arm at in place of its plan's, for a hypothetical
            grid; ``None`` quotes each at its plan, as the launch prices it.

    Returns:
        The quote.

    Raises:
        NotFoundError: The template is not found.
        ValidationFailedError: Any refusal :func:`start_run` makes before it prices an arm, or a
            ``case_count`` below one.
        ValueError: The kind planned an arm contradicting what it named.
    """
    if case_count is not None and case_count < 1:
        raise ValidationFailedError(f"case_count prices a grid of at least one case; got {case_count}")
    settings = host.settings()
    mode = _refuse_launch_arguments(
        host,
        settings,
        k_runs=k_runs,
        max_cost_usd=max_cost_usd,
        max_metered_calls=max_metered_calls,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
    )
    _refuse_oversized_launch(max(1, len(models)), settings.max_launch_arms, settings.name_of("max_launch_arms"))
    dispatched = await _dispatch(
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
    # A provisional group: nothing joins it, and its id stamps only a budget nothing is admitted to.
    planned = await _planned_arms(
        host,
        dispatched.launchable,
        _arm_requests(
            host,
            settings,
            dispatched,
            LaunchGroup(candidate_models=models),
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
        ),
    )
    if case_count is not None:
        planned = [
            replace(request, arm_plan=replace(request.arm_plan, case_count=case_count))
            if request.arm_plan is not None
            else request
            for request in planned
        ]
    return LaunchQuote(
        template_id=dispatched.template.id,
        subject_id=subject_id,
        scope_id=scope_id,
        cassette_mode=mode,
        k_runs=k_runs,
        n_variations=n_variations,
        case_count=case_count,
        arms=tuple([_judged(request, await _arm_price(host, request)) for request in planned]),
    )


@dataclass(frozen=True)
class _PricedArms:
    """One launch's arms as a battery's pre-flight planned and priced them, carried into the launch.

    The battery prices every template's arms before launching any, so the launch it then makes must not
    price them again — a second pricing is a second read of history that could disagree with the first,
    and a second place the rule could drift. The launch carries these plans and prices instead, and its
    tail holds each launcher to them. It carries the dispatch too — the template as the pre-flight loaded
    and priced it — so the launch runs the template it priced, not one edited since.

    Attributes:
        dispatched: What the pre-flight's dispatch resolved: the template, its kind and its launcher.
        arms: Each arm's plan and price, in arm order (``None`` for a kind that plans nothing, and for a
            price where no cap is in force).
    """

    dispatched: _Dispatched
    arms: tuple[tuple[ArmPlan | None, ArmPrice | None], ...]

    @classmethod
    def of(cls, dispatched: _Dispatched, requests: Sequence[LaunchRequest]) -> _PricedArms:
        """The plans and prices ``requests`` carry, as :func:`price_arms` returned them.

        Args:
            dispatched: What the pre-flight's dispatch resolved.
            requests: One launch's priced arms.

        Returns:
            Their plans and prices, with the dispatch they were priced under.
        """
        return cls(dispatched, tuple((request.arm_plan, request.arm_price) for request in requests))

    def carried_by(self, requests: Sequence[LaunchRequest]) -> list[LaunchRequest]:
        """``requests``, each carrying the plan and price it was priced at.

        Args:
            requests: The launch's arms — built by :func:`_arm_requests` from :attr:`dispatched` and the same
                models the pre-flight priced, so in the same order and number.

        Returns:
            The requests with their plans and prices.

        Raises:
            ValueError: ``requests`` are not this template's arms, or not as many as were priced — an engine
                defect that would launch arms at prices made for others.
        """
        template_id = self.dispatched.template.id
        if len(requests) != len(self.arms) or any(request.template.id != template_id for request in requests):
            raise ValueError(
                f"a battery's launch of template {template_id!r} carries the {len(self.arms)} arm(s) it priced, and "
                f"was handed {len(requests)} of template(s) {sorted({request.template.id for request in requests})}"
            )
        return [
            replace(request, arm_plan=plan, arm_price=price)
            for request, (plan, price) in zip(requests, self.arms, strict=True)
        ]


async def launch_as_group(
    host: LaunchHost,
    n_runs: int,
    *,
    settings: LaunchSettings,
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

    **It prices nothing: a ``prepare`` composing its own arms prices them through :func:`price_arms`** before
    calling any launcher, as :func:`start_run`'s does. Under an enforced cap the launch tail refuses an arm that
    did not come through it, so this primitive cannot run an arm nobody priced.

    A failure anywhere in 4-6, a cancel included, abandons the group — every prepared member's
    clients are released — and undoes ``attach`` through ``detach`` if it had landed. The
    reservation is released however the launch ends: started runs are live tasks that count on
    their own, and a refused launch's share goes back.

    Args:
        host: The host; its job manager admits the launch.
        settings: The launch's settings snapshot, read once by the caller when the launch began; its arm
            and admission ceilings are applied here, and the caller's arms are prepared under the same read.
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


def resolve_judge_pin(request: LaunchRequest, role_default: str, *, candidate_model: str) -> str:
    """The run-level judge pin an arm is scored under: the launch's, or the role default stepped off a candidate.

    A launch that names a judge gets that judge, whatever it is: a choice is recorded, not overridden. One
    that names none inherits the judge role's default — unless that default is one of the launch's candidate
    models (every arm's, :attr:`LaunchGroup.candidate_models`, and this arm's own, which is the kind's default
    for an arm that named none), in which case :attr:`LaunchSettings.judge_alternate_model` scores instead,
    provided it is set and is itself none of the candidates. Otherwise the default stands, and the run's
    surfaces disclose the overlap
    (:func:`~threetears.evals.contracts.judge_attribution.judges_sharing_a_candidate_model`). A judged kind
    calls this in its ``plan_arm`` and in its launcher alike — the plan's judges and the launcher's must agree
    — and the launch tail refuses a launcher that kept a candidate's model where an alternate stood ready.

    Args:
        request: The arm.
        role_default: The judge role's default model, resolved as the host's clients resolve it.
        candidate_model: The model the arm runs on — the one it named, or the kind's default.

    Returns:
        The judge pin, resolved.
    """
    if request.judge_model is not None:
        return request.judge_model
    candidates = _candidates_of(request, candidate_model)
    alternate = request.settings.judge_alternate_model
    if role_default in candidates and alternate is not None and alternate not in candidates:
        return alternate
    return role_default


def _judge_origin(request: LaunchRequest, judge_model: str) -> RoleModelOrigin:
    """How an arm's judge was arrived at: named by the launch, its alternate stepped in, or the role default.

    ``alternate`` when the launch named none and the judge is the host's alternate — the pin
    :func:`resolve_judge_pin` steps to off a role default that is one of the launch's candidates (the
    tail has already refused a launcher that kept such a default where the alternate stood ready).

    Args:
        request: The arm.
        judge_model: The judge the launcher wired.

    Returns:
        The origin recorded on the run's ``model_role_provenance``.
    """
    if request.judge_model is not None:
        return "chosen"
    if judge_model == request.settings.judge_alternate_model:
        return "alternate"
    return "inherited"


def _candidates_of(request: LaunchRequest, candidate_model: str) -> frozenset[str]:
    """Every candidate model of ``request``'s launch: each arm's, and this arm's own.

    Args:
        request: The arm.
        candidate_model: The model it runs on.

    Returns:
        The candidates.
    """
    return frozenset({*request.launch_group.candidate_models, candidate_model})


def require_candidate_model(request: LaunchRequest, default: str | None) -> str:
    """The model ``request``'s arm runs on — the one it named, or the kind's role default — or the refusal.

    One refusal for the two places that ask: a kind's :attr:`LaunchableKind.plan_arm`, which makes it before
    the arm is priced, and the launch tail, which makes it for a kind that plans nothing. So an operator who
    named no model on a kind with no default hears exactly that, before anything is priced or built.

    Args:
        request: The arm.
        default: The kind's role default, or ``None`` for a kind with none.

    Returns:
        The model.

    Raises:
        ValidationFailedError: The arm named no model and the kind has no default.
    """
    model = request.candidate_model or default
    if model is None:
        raise ValidationFailedError(f"kind {request.kind!r} has no default candidate model; name one")
    return model


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
    candidate_model = require_candidate_model(request, wiring.default_candidate_model)
    judge = wiring.judge
    if request.judge_model is not None and (judge is None or judge.model != request.judge_model):
        raise ValueError(
            f"the launch pinned judge {request.judge_model!r} and kind {kind!r}'s launcher wired "
            f"{judge.model if judge is not None else 'no judge'!r}; a pinned judge is the judge"
        )
    alternate = request.settings.judge_alternate_model
    candidates = _candidates_of(request, candidate_model)
    if (
        request.judge_model is None
        and judge is not None
        and judge.model in candidates
        and alternate is not None
        and alternate not in candidates
    ):
        raise ValueError(
            f"kind {kind!r}'s launcher kept the inherited judge {judge.model!r}, one of the launch's candidates, "
            f"where {request.settings.name_of('judge_alternate_model')} names {alternate!r} to score instead; a "
            "candidate would grade its own output — resolve the pin with resolve_judge_pin"
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
        if (judge.planned if judge is not None else None) != plan.judge:
            raise ValueError(
                f"kind {kind!r} planned this arm judged by {plan.judge} and its launcher wired "
                f"{judge.planned if judge is not None else 'no judge'}; the arm was priced as judged by its plan, so "
                "it is scored by its plan's judges — plan them with plan_judge from what the launcher builds from"
            )
        if wiring.simulator_model != plan.simulator_model:
            raise ValueError(
                f"kind {kind!r} planned this arm's simulator on {plan.simulator_model!r} and its launcher resolved "
                f"{wiring.simulator_model!r}; the arm was priced as its plan, so it runs as the plan"
            )
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
    2. Take the launch's settings snapshot off the request, and assemble the run. Everything the request and the
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
        # Under an enforced cap every arm is priced before any launcher runs, by one rule; an arm that never
        # came through it (a launch composed outside start_run that skipped price_arms) is refused here rather
        # than run under a cap nothing checked it against.
        if _cap_of(request)[0] is not None and request.arm_price is None:
            raise ValueError(
                f"kind {request.kind!r}'s arm of template {template.id!r} reached the launch tail unpriced under an "
                "enforced cap; every arm is priced before any launcher runs — compose a launch's arms through "
                "price_arms"
            )
        candidate_model = _refuse_wiring_the_request_contradicts(request, wiring)
        test_cases = list(wiring.test_cases)
        judge = wiring.judge
        # Whether each pinned role was named by the launch or resolved from the role's default: the
        # request says which, so the run records it without the launcher restating it.
        role_provenance: dict[str, RoleModelOrigin] = {}
        if judge is not None:
            role_provenance["judge"] = _judge_origin(request, judge.model)
        if wiring.simulator_model is not None:
            role_provenance["simulator"] = "chosen" if request.simulator_model is not None else "inherited"
        # The eval ceilings arrive at the engine as values (R7): budget.py and metering.py
        # read no configuration of their own, and the host's settings are resolved through
        # the host's launch settings. Taken from the launch's ONE read, carried on the request, and
        # used for both the figures stored on the run and the objects that enforce them — and the
        # judge concurrency below — so no read here can straddle a hot reload: neither leave a run
        # recording a ceiling it did not run under, nor refuse, after its generation was paid for, a
        # launch whose refusals were all made under the figures it started with.
        settings = request.settings
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
                    max_metered_calls,
                    configured_max_metered_calls=configured_max_metered_calls,
                    enforcement_enabled=eval_enforcement_enabled,
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
                    # cap — checked between cells, and asked from inside a cell (a
                    # conversation's simulator spend) through the cell's sink. Pure
                    # arithmetic — no budget-layer / DB I/O in the gate path.
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
    max_cost_usd: float | None = None,
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

    **All or nothing.** Every template's arms are prepared — every launcher called, every tail's checks
    made — before any template's runs start, so a refusal any template's preparation makes starts nothing.
    Each template's launch runs the template its pre-flight loaded and priced, never one edited since. And
    every refusal an arm of any template would make is made over the whole set before anything is prepared: the engine's own — the dispatch's refusals, every template's arms priced
    against their run caps (by the one rule a launch prices its arms by, and only once: each template's
    launch carries the plans its arms were priced at), and every generating template's ``llm`` writer calls
    priced on the host's ``variation`` client against the out-of-run cap its launch will be held to — and
    then the host's, through ``preflight``, the check that calls what its launchers call. A battery that
    generates therefore pays for no template's cases until every template's have been priced.

    **Preparing a template whose cases a model writes pays for them** — its launcher generates — so those
    templates are prepared after every other: a refusal any stored-case or enumerated template's preparation
    makes is made before any generation is paid. A model-written template's preparation refusing after an
    earlier model-written template's generation was paid still loses that generation (held to that
    template's own out-of-run cap) and starts nothing; the pricing above is what keeps that rare — such a
    refusal is one only a launcher or its tail makes, against something that changed since the pre-flight.

    **Its bounds are per launch, as a launch's are.** Each template is one launch, so each template's
    generation is held to the host's ``max_out_of_run_cost_usd`` on its own — a battery of N generating
    templates may spend up to N times that out of run — and each of its runs to its own cost cap, exactly
    as N separate launches would be. ``max_cost_usd`` names that per-run cap for every run of the
    battery; without it every run inherits the host's, and a template whose arms no history can price is
    then refused (unknown is not $0, and an inherited cap is nobody's decision about it).
    The host's settings are read once, for the whole battery.

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
        max_cost_usd: Optional per-run cost-cap override for every run the battery launches, as
            :func:`start_run` takes it; must be ``> 0`` and at most the host's configured ceiling. Also the cap
            each template's arms are priced against before anything launches.
        preflight: The host's pre-flight, prepared once for the subject and models and then asked
            of every template before any launches, after the engine's own pricing. It checks what only the
            host's launchers know; a generation's calls and every template's arms the battery prices itself.

    Returns:
        The ids of the launched runs: for each active universal template, one per model. An empty
        battery (no active universal templates) returns ``[]``.

    Raises:
        AdmissionRefusedError: Every template's runs together would pass the host's admission
            ceiling — raised once the set is listed and before any subject is read or arm prepared,
            so nothing launches.
        ValidationFailedError: An unrecognised ``cassette_mode``, ``'replay'`` (a corpus serves one
            template, and a battery runs many), a non-positive ``max_cost_usd`` or one above the host's configured
            ceiling, a generating template whose
            writer's calls cannot be priced under an enforced out-of-run cap or are priced above it, or any
            refusal a template or its arm would make at launch — raised before any template launches, so
            nothing does.
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
    # The battery's ONE read of the host's settings: its admission, every template's pre-flight pricing
    # and every template's launch are made under it, so the pre-flight cannot pass a launch a reload then
    # refuses part-way through the set.
    settings = host.settings()
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValidationFailedError(_battery_refusal(ValueError(f"max_cost_usd must be > 0 (got {max_cost_usd})")))
    try:
        _refuse_raised_ceilings(settings, max_cost_usd=max_cost_usd, max_metered_calls=None)
    except ValidationFailedError as refused:
        raise ValidationFailedError(_battery_refusal(refused)) from refused
    # Every template is one launch of one run per model, so a launch's own argument refusals — a model named
    # twice, more arms than one launch may start — read the battery's arguments alone, and are made here once.
    try:
        _refuse_repeated_models(models)
        _refuse_oversized_launch(max(1, len(models)), settings.max_launch_arms, settings.name_of("max_launch_arms"))
    except ValidationFailedError as refused:
        raise ValidationFailedError(_battery_refusal(refused)) from refused
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
        # refused one of them. Then every template's arms are priced — by the one rule every launch's are,
        # stored-case and generating alike — and, for a generating battery, every template's writer calls,
        # so it cannot launch (or pay for the generations of) its first templates and then be refused a
        # later one. Each template's priced plans are carried into its launch, which prices nothing again.
        # The writer is the host's, built in the role each launch builds it in; here it is asked its prices
        # and never called.
        priced: dict[str, _PricedArms] = {}
        async with contextlib.AsyncExitStack() as pricing:
            writer: PricedCompletion | None = None
            for universal_template in templates:
                writes_with_a_model = bool(_llm_axes(universal_template))
                template_variation_model = variation_model if writes_with_a_model else None
                try:
                    resolved = _launchable(
                        host,
                        universal_template,
                        n_variations=n_variations,
                        variation_model=template_variation_model,
                        judge_model=judge_model,
                        judge_config_ids=None,
                        simulator_model=simulator_model,
                        cassette_mode=battery_cassette_mode,
                        overlays=None,
                        apparatus_settings=apparatus_settings,
                    )
                    # Through the functions each launch builds and prices its arms with. The group is
                    # provisional: nothing joins it, and its id stamps only a budget nothing is admitted to.
                    dispatched = _Dispatched(
                        universal_template,
                        resolved.kind,
                        resolved.launchable,
                        resolved.overlays,
                        resolved.kind_spec,
                        resolved.apparatus_settings,
                    )
                    requests = await price_arms(
                        host,
                        resolved.launchable,
                        _arm_requests(
                            host,
                            settings,
                            dispatched,
                            LaunchGroup(candidate_models=models),
                            subject_id=subject_id,
                            models=models,
                            k_runs=k_runs,
                            scope_id=scope_id,
                            n_variations=n_variations,
                            variation_model=template_variation_model,
                            judge_model=judge_model,
                            judge_config_ids=None,
                            simulator_model=simulator_model,
                            cassette_mode=battery_cassette_mode,
                            cassette_corpus_id=None,
                            max_cost_usd=max_cost_usd,
                            max_metered_calls=None,
                        ),
                    )
                    priced[universal_template.id] = _PricedArms.of(dispatched, requests)
                    if template_variation_model is None:
                        continue
                    # The calls this template's launch would make, quoted against the budget it would be
                    # admitted under — the same plan and the same budget construction the launch uses.
                    # One writer for every template: the battery names one variation model, and each
                    # launch builds its writer from the same factory, role and model.
                    quoted_on: PricedCompletion = (
                        writer
                        if writer is not None
                        else await pricing.enter_async_context(
                            host.eval_host.completion_clients("a battery's generation pricing")(
                                "variation", template_variation_model
                            )
                        )
                    )
                    writer = quoted_on
                    budget = requests[0].generation_budget
                    assert budget is not None, "a generating launch's every arm carries its budget"
                    existing = await run_blocking(
                        host.eval_host.blocking_executor,
                        partial(host.eval_host.storage.query_test_cases, scope_id, template_id=universal_template.id),
                    )
                    budget.quote(
                        quoted_on,
                        "variation",
                        list(plan_variation_calls(universal_template, n_variations, existing).values()),
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

        # Every template's arms PREPARED before any template's runs start — each launcher called, each case
        # set frozen, each tail's checks made against the plans priced above — and only then is every group
        # started. So a refusal any template's preparation makes (a stored case added since its pre-flight,
        # which its tail refuses as more cases than planned; a refusal only its launcher makes) abandons every
        # prepared group and starts nothing, where launching template by template would have left the
        # templates ahead of it running. Each template is still its own group, its runs sharing one slot.
        #
        # Preparing a template whose cases a model writes PAYS: its launcher generates them. So those are
        # prepared last — every template whose preparation spends nothing (stored cases, or cases enumerated
        # without a model) makes its refusals first, and a refusal there costs no generation. What ordering
        # cannot close: a later model-written template's preparation refusing after an earlier one's
        # generation was paid — that spend is spent, held to its own launch's out-of-run cap, and its runs are
        # abandoned with the rest. The ids are still returned, and the groups started, in the templates' order.
        def pays_to_prepare(template: EvalTemplate) -> bool:
            return n_variations > 0 and bool(_llm_axes(template))

        prepared_runs: dict[str, list[str]] = {}
        group_of: dict[str, LaunchGroup] = {}
        groups: list[LaunchGroup] = []
        started: set[str] = set()
        try:
            for template in sorted(templates, key=pays_to_prepare):
                group = LaunchGroup(candidate_models=models)
                groups.append(group)
                group_of[template.id] = group
                runs = await _start_run(
                    host,
                    settings,
                    template_id=template.id,
                    scope_id=scope_id,
                    subject_id=subject_id,
                    models=models,
                    k_runs=k_runs,
                    n_variations=n_variations,
                    variation_model=variation_model if _llm_axes(template) else None,
                    judge_model=judge_model,
                    judge_config_ids=None,
                    simulator_model=simulator_model,
                    cassette_mode=cassette_mode,
                    cassette_corpus_id=None,
                    overlays=None,
                    apparatus_settings=apparatus_settings,
                    max_cost_usd=max_cost_usd,
                    max_metered_calls=None,
                    # Prepare only: the battery admitted every run above, and starts the groups itself.
                    launch_group=group,
                    admission=None,
                    priced=priced[template.id],
                )
                prepared_runs[template.id] = [run.id for run in runs]
            for group in (group_of[template.id] for template in templates):
                await host.job_manager.start_group(group.members)
                started.add(group.id)
                log.info(
                    "eval.start_universal_battery group=%s started runs=%s",
                    group.id,
                    [r.id for r, _, _ in group.members],
                )
        except BaseException:
            # A started group's runs are live and release their own clients; only the unstarted are abandoned.
            # Starting is a storage write per group, so a store refusing a later group's save leaves the
            # earlier groups running — the one window this ordering does not close.
            for group in groups:
                if group.id not in started:
                    await group.abandon()
            raise
        run_ids = [run_id for template in templates for run_id in prepared_runs[template.id]]
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
    dim_ids, configs = _judge_configs(host.storage, template, selection, judged_artifact)

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


def plan_judge(
    host: EvalHost,
    template: EvalTemplate,
    judge_model: str,
    selection: Mapping[str, str] | None = None,
    *,
    judged_artifact: JudgedArtifact,
) -> PlannedJudge:
    """The judges one arm of ``template`` will be scored by, resolved as :func:`build_judge_service` resolves them — building nothing.

    What a judged kind's :attr:`LaunchableKind.plan_arm` puts on its :class:`ArmPlan`, so the arm is priced by
    the judges it will be scored by and its launcher is held to them. The same config load and cascade the
    judge service is built from (one config per scored dim — the selection's where it names one, the active
    one elsewhere — and each dim's effective model), with no client built and nothing called: it pays for
    nothing. It makes the selection's refusals too, so a launch naming a config wrongly hears that before its
    arm is priced.

    Args:
        host: The host, where the judge configs are read.
        template: The arm's template, whose rubric names the scored dims.
        judge_model: The run-level judge pin, **resolved** — the launch's, or the kind's role default.
        selection: The launch's ``judge_config_ids``, or ``None``.
        judged_artifact: What the kind's judge reads, which picks the dims it scores.

    Returns:
        The planned judge.

    Raises:
        ValidationFailedError: ``selection`` names an unscored dim, a config that does not load in the
            template's scope, or one authored for another dim.
        ValueError: ``judged_artifact`` declares a kind no judge reads.
    """
    dim_ids, configs = _judge_configs(host.storage, template, selection, judged_artifact)
    return PlannedJudge(
        model=judge_model,
        effective_judges=resolve_effective_judges(dim_ids, configs.get, judge_model),
        config_ids={dim_id: config.id for dim_id, config in configs.items()},
    )


def _judge_configs(
    storage: DefinitionStore,
    template: EvalTemplate,
    selection: Mapping[str, str] | None,
    judged_artifact: JudgedArtifact,
) -> tuple[list[str], dict[str, JudgeConfig]]:
    """The dims a kind's judge scores on ``template``, and the config each is judged by — ONE load for both readers.

    Args:
        storage: Where the judge configs are read.
        template: The template, whose rubric names the scored dims.
        selection: ``{dim_id: config_id}`` as the launch named it, or ``None``.
        judged_artifact: What the kind's judge reads.

    Returns:
        The scored dim ids, and ``{dim_id: config}`` over the dims that have one.

    Raises:
        ValidationFailedError: See :func:`_resolve_selected_judge_configs`.
        ValueError: ``judged_artifact`` declares a kind no judge reads.
    """
    if judged_artifact is JudgedArtifact.UNJUDGED:
        raise ValueError("an unjudged kind has no judge to build; its grade is the kind's own measures")
    dim_ids = scored_dim_ids([dim.name for dim in template.rubric], judged_artifact)
    chosen = _resolve_selected_judge_configs(storage, dict(selection or {}), dim_ids, template.scope_id)
    configs = {}
    for dim_id in dim_ids:
        # Membership, not truthiness: a selected config is a record, and asking
        # whether the record is truthy asks a different question than whether the
        # launch named one.
        cfg = chosen[dim_id] if dim_id in chosen else storage.load_active_judge_config(dim_id, template.scope_id)
        if cfg is not None:
            configs[dim_id] = cfg
    return dim_ids, configs


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
    "ArmOutcome",
    "ArmPlan",
    "ArmPrice",
    "ArmQuote",
    "ArmVerdict",
    "BatteryPreflight",
    "KindLauncher",
    "KindWiring",
    "LaunchArgument",
    "LaunchHost",
    "LaunchGroup",
    "LaunchPricer",
    "LaunchQuote",
    "LaunchRequest",
    "LaunchableKind",
    "LaunchSettings",
    "PlannedJudge",
    "RunJudge",
    "TemplatePreflight",
    "build_judge_service",
    "launch_as_group",
    "launch_run",
    "no_launcher_for",
    "plan_judge",
    "price_arms",
    "quote_launch",
    "require_candidate_model",
    "resolve_judge_pin",
    "settable_apparatus",
    "start_run",
    "start_universal_battery",
]
