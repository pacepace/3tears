"""The launch: refusing what cannot run, dispatching on the template's kind, and handing each arm's run to its job.

What every candidate kind's launch shares, as functions whose dependencies are parameters. A launch
loads its template, refuses what its arguments cannot honour, admits its runs, and dispatches each
arm to its kind's launcher, found in a registry the host supplies (:class:`LaunchableKind`) rather
than by name. The kind's launcher builds what only that kind has — its subject, its case set, its
clients — and hands the shared tail (:func:`launch_run`) a typed :class:`KindWiring`; the tail stamps
everything the request and template already say, assembles and bounds the run and gives its work
function to the :class:`LaunchGroup` the launch starts together. The judge-service build
(:func:`build_judge_service`) is here for the same reason: it names no kind.

What only a host can answer arrives on the :class:`LaunchHost` every entrypoint here takes: the
:class:`~threetears.evals.kernel.host.eval_host.EvalHost` the rest of the engine reads (vocabulary,
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
from typing import TYPE_CHECKING, Annotated, Any, Literal, NamedTuple, Protocol, TypeVar, cast, get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from threetears.evals.kernel import JobTimeoutFactory
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.schema.models import EvalTemplate, WorldPlacement

from threetears.evals.kernel.arguments import normalize_blank
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.sweepables import CORE_SWEEPABLES
from threetears.evals.kernel.metrics import RUN_MARGIN_MEASURES, declaration_of, run_margin_refusal
from threetears.evals.kernel.identity import derive_context_identity, variant_levers_of_run
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_LAUNCH_K_RUNS,
    GOAL_CHECK_PROOF_RULES,
    ApparatusSettingValue,
    CaseSet,
    CaseSetRef,
    KnownCellTimeoutOrigin,
    EvalRun,
    EvalTestCase,
    refused_goal_checks,
    JudgedArtifact,
    KnownModelRoleOrigin,
    KnownRoleModelOrigin,
    resolve_effective_judges,
    scored_dim_ids,
)
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.out_of_run import AdmittedGeneration, OutOfRunBudget, plan_variation_calls
from threetears.evals.run.authoring import (
    refuse_ignored_template_fields,
    refuse_undeclared_judges,
    validated_kind_spec,
)
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.run.case_sets import resolve_case_set
from threetears.evals.run.ceilings import CeilingRaisedError, refuse_raised_ceiling
from threetears.evals.kernel.cassettes import CassetteMode
from threetears.evals.run.jobs import (
    MAX_CONCURRENT_JOBS,
    EvalJobManager,
    RunEndHook,
    adaptive_job_timeout_s,
)
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService, judge_clients_for_run
from threetears.evals.kernel.job_leases import DEFAULT_JOB_LEASE_S, JobLeases
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.run.executor import DEFAULT_MAX_CONCURRENT_CELLS, CellExecutor
from threetears.evals.run.metering import MeteredCallLedger
from threetears.evals.kernel.offload import run_blocking, wait_through_cancellation
from threetears.evals.run.check_controls import goal_check_proofs
from threetears.evals.run.runner import (
    DEFAULT_CELL_TIMEOUT_S,
    KindFactory,
    RunCallbacks,
    RunnerOptions,
    execute_run,
    template_as_graded,
)
from threetears.evals.run.simulator import SIMULATOR_REQUEST_SETTINGS
from threetears.evals.schema.logs import get_logger
from threetears.evals.kernel.remedies import FIND_JUDGE_CONFIG, FIND_RUN, FIND_TEMPLATE

if TYPE_CHECKING:
    from threetears.evals.schema.subject import SubjectSnapshot
    from threetears.evals.kernel.host.sweepables import SweepableRegistry
    from threetears.evals.schema.models import JudgeConfig, VariationCounts
    from threetears.evals.schema.completion import BoundCompletionClient
    from threetears.evals.kernel.scoring import CellSummary
    from threetears.evals.kernel.storage import DefinitionStore
    from threetears.evals.kernel.usage_capture import ExternalRateTable
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
    under another — nor leave a run recording a ceiling it did not run under.

    :ivar max_launch_arms: How many runs one launch may start together. A group starts every member
        at once in one job slot, so this is the concurrency one launch adds.
    :ivar max_admitted_runs: How many runs may be admitted and unfinished at once across every launch
        — the ceiling admission refuses past rather than queueing behind.
    :ivar judge_concurrency: How many judge calls one cell makes at once.
    :ivar max_concurrent_cells: The most of a run's cells that execute at once when its launch does not declare
        latency under test (``measure_latency``) — a ceiling, never a width a kind is pushed to: a run executes
        at the narrower of this and what its kind declares it is safe at
        (:attr:`KindWiring.max_concurrent_cells`), and a kind that declares nothing runs serially. A launch
        that declares latency runs its cells one at a time whatever either says. Defaults to
        :data:`~threetears.evals.run.executor.DEFAULT_MAX_CONCURRENT_CELLS`; ``1`` runs every run serially.
    :ivar enforcement_enabled: Whether the cost and metered-call ceilings are enforced at all.
    :ivar max_cost_usd: The run cost ceiling a run inherits when its launch names none.
    :ivar max_metered_calls: The metered-call ceiling a run inherits when its launch names none, or
        ``None`` for a host that declares it has NO metered tools: its runs record a ceiling of
        ``0`` (origin ``none_declared``), a metered call on one is refused and counted, and a
        launch naming a ceiling is refused, since it would bound nothing.
    :ivar max_out_of_run_cost_usd: The most a launch's out-of-run calls — its case generation, which runs
        before any run exists and so under no run's cap — may together be priced at before they are
        made (:class:`~threetears.evals.kernel.out_of_run.OutOfRunBudget`). Enforced exactly when
        ``enforcement_enabled`` is. Per LAUNCH, as ``max_cost_usd`` is per run: N launches may spend up to N
        times this out of run, as their runs may spend up to their count times their cap. An analysis generation is held
        to it too, per generation (:func:`~threetears.evals.ops.analysis_generate`): its calls run after the
        runs it reads, under no run's cap.
    :ivar judge_alternate_model: The judge a launch's arms are scored by instead of the judge role's default
        when that default IS one of the launch's candidate models — a model grading its own output —
        provided it is itself none of them (:func:`resolve_judge_pin`), spelled as the host's clients name
        the model they resolve. It never overrides a judge the launch named, nor a model a judge config
        pins per dim: those are choices. ``None`` substitutes nothing, and a run judged on a candidate's
        model says so on every surface that lists its judges
        (:func:`~threetears.evals.schema.judge_attribution.judges_sharing_a_candidate_model`).
    :ivar max_cell_timeout_s: The longest per-cell deadline a launch may name (``cell_timeout_s``), in seconds.
        ``None`` declares no ceiling of the host's own: a launch may then only LOWER its kind's deadline (the
        one the kind's launcher wires, or :data:`~threetears.evals.run.runner.DEFAULT_CELL_TIMEOUT_S`), so a
        host that declares nothing cannot be overridden upward.
    :ivar setting_names: What the host calls each of the fields above, keyed by field name, so a
        refusal names the knob an operator turns. A field the host does not name here is
        called by its own name; the engine names no host setting of its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_launch_arms: int = Field(gt=0)
    max_admitted_runs: int = Field(gt=0)
    judge_concurrency: int = Field(gt=0)
    max_concurrent_cells: int = Field(default=DEFAULT_MAX_CONCURRENT_CELLS, gt=0)
    enforcement_enabled: bool
    max_cost_usd: float = Field(gt=0)
    max_metered_calls: int | None = Field(gt=0)
    max_out_of_run_cost_usd: float = Field(gt=0)
    judge_alternate_model: str | None = Field(default=None, min_length=1)
    max_cell_timeout_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    setting_names: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_only_its_own_settings(self) -> LaunchSettings:
        """Refuse a name for a setting this snapshot does not have — a typo would never be read."""
        if unknown := sorted(set(self.setting_names) - (set(type(self).model_fields) - {"setting_names"})):
            raise ValueError(f"setting_names names settings a launch does not have: {', '.join(unknown)}")
        return self

    def name_of(self, setting: str) -> str:
        """What the host calls ``setting``, for a refusal to name.

        :param setting: A field of this snapshot.
        :ptype setting: str
        :return: The host's name for it, or the field's own name when the host gave none.
        :rtype: str
        """
        return self.setting_names.get(setting, setting)


@dataclass(frozen=True, kw_only=True)
class PlannedJudge:
    """The judge one arm will be scored by, resolved before anything is built: what :func:`plan_judge` returns.

    The model a run spends its judging on is not the run-level pin alone — a dim whose config names its own
    model is scored by that one, and a pin a launch left unnamed resolves to the role's default, which moves
    over time — so an arm is priced, and its launcher held, by the judges it will actually be scored by.

    :ivar model: The run-level judge pin, resolved — the launch's, or the role's default for one that named none.
    :ivar effective_judges: The model each scored dim is requested from, ``{dim_id: model}``, as a run records it.
    :ivar config_ids: The config each dim with one is judged by, ``{dim_id: config_id}`` — the ones the launch
        named and the active ones it inherited — as a run records its config set.
    """

    model: str
    effective_judges: Mapping[str, str]
    config_ids: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze both maps, so a plan cannot move after it was priced.

        :raises ValueError: ``model`` is blank, or a dim is scored by no model.
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
class CallCeiling:
    """The largest reservation one call of an arm's cells will hold against its run's cap, priced before launch.

    A run's cap is read between cells and refuses no call inside one, so a call whose ceiling is above the cap on
    its own is made, and the run stops ``budget_stopped`` before its next cell having spent more than the cap on
    that one call. An arm is still refused before launch when its largest call's ceiling is above its cap
    (:func:`price_arms`, and :func:`quote_launch` with it): a cap smaller than one call cannot bound the run, and
    this is what that refusal reads.

    **Exact, or a floor.** For a kind whose calls are whole before launch (the judge kind's frozen cases, the
    reporter's frozen bundles), ``usd`` is the largest call's ceiling exactly, and ``lower_bound`` is ``False``.
    A subject kind's judge and simulator calls carry a transcript that does not exist before its run, so the most
    it can say is a floor (:func:`~threetears.evals.run.price_subject_call_floor`): ``lower_bound`` is ``True``
    and ``usd`` is what the named call is reserved at with every part its cell fills left empty — so the call,
    and the arm's largest call with it, is reserved at ``usd`` or more. The admission rule reads either the same
    way, and is sound for both: a floor above the cap means the real call is too. A floor at or under the cap
    admits nothing about the real call, which is still held to the cap as the run reserves it.

    :ivar usd: The ceiling in dollars: the client's ``price_ceiling`` for the call, times every attempt the engine
        makes of it — exactly what the run will reserve, or, when ``lower_bound`` is set, the least it will.
    :ivar call: Which call it is, in words a refusal quotes ("the judge call on case 'c-3' (2 requests: the
        first and its parse retry) on 'google/gemini-3.8-flash'").
    :ivar lowered_by: What lowers the ceiling, in words a remedy quotes — the levers besides a larger cap.
    :ivar lower_bound: Whether ``usd`` is a floor under the call's reservation rather than its exact figure.
    """

    usd: float
    call: str
    lowered_by: str
    lower_bound: bool = False

    def __post_init__(self) -> None:
        """Refuse an amount no reservation could hold, and blank words.

        :raises ValueError: ``usd`` is negative or not finite, or ``call`` or ``lowered_by`` is blank.
        """
        if isinstance(self.usd, bool) or not math.isfinite(self.usd) or self.usd < 0:
            raise ValueError(f"a call's ceiling is a finite amount, 0 or more; got usd={self.usd!r}")
        if not self.call.strip():
            raise ValueError("a call's ceiling names the call it bounds; call is blank")
        if not self.lowered_by.strip():
            raise ValueError("a call's ceiling says what lowers it; lowered_by is blank")


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

    :ivar case_count: The most cases the arm will run. For an arm over stored cases (``n_variations=0``),
        how many of the template's stored cases the kind plays; for a generating arm, at most
        ``n_variations`` — an upper bound, since generation de-duplicates and can keep fewer.
    :ivar candidate_model: The model the arm will run on: the one the launch named, or the kind's role
        default for an arm that named none.
    :ivar judge: The judges the arm will be scored by (:func:`plan_judge`), resolved; ``None`` for a kind whose
        grade is mechanical. Required, so a judged kind cannot be priced from unjudged history by omission.
    :ivar simulator_model: The model that will drive the arm's simulated user, resolved — the launch's pin, or
        the role's default; ``None`` for a kind that runs no simulator.
    :ivar call_ceiling: The largest reservation one call of the arm's cells will hold — or a floor under it, for a
        subject kind (:attr:`CallCeiling.lower_bound`) — as its kind priced it
        (:attr:`LaunchableKind.price_largest_call`), which the engine sets once the plan is made; ``None`` for
        a kind that does not price its calls before launch, whose calls are bounded only as the run reserves
        them.
    """

    case_count: int
    candidate_model: str
    judge: PlannedJudge | None
    simulator_model: str | None
    call_ceiling: CallCeiling | None = None

    def __post_init__(self) -> None:
        """Refuse a plan of no cases, or one naming no model.

        :raises ValueError: ``case_count`` is below one, or ``candidate_model`` or ``simulator_model`` is blank.
        """
        if self.case_count < 1:
            raise ValueError(f"an arm plan runs at least one case; got case_count={self.case_count}")
        if not self.candidate_model.strip():
            raise ValueError("an arm plan names the model the arm runs on; candidate_model is blank")
        if self.simulator_model is not None and not self.simulator_model.strip():
            raise ValueError("an arm plan names the model its simulator runs on, or None; simulator_model is blank")


@dataclass(frozen=True, kw_only=True)
class ArmQuote:
    """One arm of a launch, as the engine asks the host's pricer to price it — stored-case arms and generating ones
    alike.

    :ivar scope_id: The scope the arm's run will live in, whose history a history pricer reads.
    :ivar template_id: The template the arm runs.
    :ivar subject_id: The subject it measures.
    :ivar candidate_model: The model it runs on, as its plan named it.
    :ivar k_runs: Repeats of every case.
    :ivar case_count: The cases it runs, as its plan bounded them: the template's stored cases the kind plays
        for an arm that generates none, or at most ``n_variations`` for one that generates.
    :ivar n_variations: The cases the launch generates for the arm, or ``0`` for an arm over the template's
        stored cases (:attr:`case_source`).
    :ivar cassette_mode: Its cassette mode, normalised.
    :ivar judge: The judges it will be scored by, as its plan resolved them — the run-level pin, the model each
        scored dim is requested from (a dim's config can override the pin) and the config set — or ``None``
        for a kind with no judge. Judging is part of what a run spends, so an arm priced from history judged
        by cheaper models would be predicted low.
    :ivar simulator_model: The model that will drive its simulated user, as its plan resolved it; ``None`` for a
        kind with no simulator.
    :ivar apparatus_settings: The apparatus values the arm's rig is set up with, resolved — every setting its
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

    :ivar predicted_usd: The predicted cost of the arm's whole run, in dollars, or ``None`` when the pricer
        cannot predict it — unknown, which the launch never reads as $0. It is the figure the arm's cap
        is held to, so a pricer whose prediction is a range returns its UPPER end: an arm admitted
        because its central estimate fits is one that runs past its cap whenever the run lands above
        the centre, after its generation was paid for.
    :ivar basis: How the prediction was made, or why there is none, in words a refusal quotes ("from 12
        priced past results, method usage-history").
    :ivar central_usd: For a pricer whose prediction is a range, its central estimate; ``None`` for one that
        states a single figure. Never held to the cap — reported beside :attr:`predicted_usd`, and what a
        cost pivot sets beside the cost a run later observed.
    :ivar low_usd: For a range, its lower end; ``None`` otherwise.
    :ivar method_id: The estimator, so two predictions by different methods are never silently compared;
        ``None`` for a pricer that names none.
    """

    predicted_usd: float | None
    basis: str
    central_usd: float | None = None
    low_usd: float | None = None
    method_id: str | None = None

    def __post_init__(self) -> None:
        """Refuse a negative or non-finite figure, a range out of order, and a blank basis.

        :raises ValueError: A figure is negative or not finite; ``low_usd`` or ``central_usd`` is given without a
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

        :param quote: The arm.
        :ptype quote: ArmQuote
        :return: The prediction, or a price with no prediction and the reason there is none.
        :rtype: ArmPrice
        """
        ...  # pragma: no cover — protocol


@dataclass(frozen=True, kw_only=True)
class UnknownSubject:
    """A host's answer that it holds no subject of the id a launch named — and how its caller finds one it does.

    :ivar remedy: How the caller finds a subject this host holds, in the host's words (the call or page that lists
        them); stated beside the refusal, so it must say something.
    """

    remedy: str

    def __post_init__(self) -> None:
        """Refuse a blank remedy.

        :raises ValueError: ``remedy`` is blank.
        """
        if not self.remedy.strip():
            raise ValueError("an unknown subject's remedy says how to find a subject the host holds; it is blank")


#: How a kind answers whether its host holds the subject a launch of it names, before anything is priced:
#: ``(subject_id, scope_id) -> None`` when it does, :class:`UnknownSubject` when it does not. A kind's, not the
#: host's: one subject id means a live entity to one kind, a fixed prompt key to another and a free label to a third,
#: so a resolver asked without the kind could only refuse one of them or admit any string for all. Called off the
#: event loop, through the host's blocking executor, so it may read the host's own store.
SubjectResolver = Callable[[str, str], UnknownSubject | None]


@dataclass(frozen=True, kw_only=True)
class LaunchHost:
    """An :class:`~threetears.evals.kernel.host.eval_host.EvalHost`, and what starting its runs needs.

    What :func:`start_run` and the launch tail take. It COMPOSES the
    host rather than extending it: the host is built once and handed in whole, so no field of it can
    be dropped on the way, and the analysis side is handed the same :attr:`eval_host` this launches
    through.

    :ivar eval_host: The host every run this launches belongs to — its vocabulary, its storage, its
        clients, its tracing, its executor and its cell timeout, and the template checks whose ``kinds`` are the
        launch registry (:attr:`kinds`).
    :ivar settings: Reads the host's :class:`LaunchSettings` — a callable because they hot-reload.
    :ivar job_timeout_factory: What bounds each run's job, as
        :class:`~threetears.evals.run.jobs.EvalJobManager` takes it. No default, for the reason the
        host's cell timeout has none: :func:`~threetears.evals.kernel.default_job_timeout` is
        right for a host with no timeout layer of its own and silently wrong for one with.
    :ivar world_placements: What one assembled run did with every world dimension the host declares,
        stamped on the run before the identity that hashes it. Required exactly when the
        profile declares a world, and refused when it declares none — a placement over a world
        the host does not have describes nothing. A host with no world records ``{}``.
    :ivar launch_pricer: Predicts what each arm of a launch will cost (:class:`LaunchPricer`), so an arm its cap
        cannot pay for is refused before any launcher runs — before a generating launch's paid generation
        call, and before a stored-case arm's launcher builds anything. Every arm of every launch is priced
        through it, under an enforced cap. ``None`` for a host that prices no launch: its every arm is then
        unpriceable, which an arm with a cap its launch named proceeds under and one with an inherited cap
        is refused for — never run unpriced under a cap nobody chose.
    :ivar max_concurrent_jobs: How many runs' jobs execute at once in this process. A launch declaring latency
        under test takes all of them while its runs execute, one run at a time, so nothing executes
        beside a latency it measures.
    :ivar cell_executor: What runs each run's cells (:class:`~threetears.evals.run.executor.CellExecutor`) — a
        host's own pool or queue — or ``None`` for the in-process default, which runs them serially or at
        the width each run's launch decided.
    :ivar on_job_progress: Called with ``(run id, progress)`` on every progress write — typically a
        broadcast to an operator's view — or ``None``.
    :ivar on_run_end: Told each run's recorded terminal status
        (:class:`~threetears.evals.run.jobs.RunEndHook`) — a
        :class:`~threetears.evals.ops.RegressionWatch` to check a completed run against its contestant's
        history — or ``None`` for nobody.
    :ivar job_lease_s: How long a job's lease lasts unless renewed, in seconds
        (:data:`~threetears.evals.kernel.job_leases.DEFAULT_JOB_LEASE_S` by default); renewed every third of it. A
        job whose process stops renewing reads lost this long after its last renewal, in every process, so
        it bounds both how soon a dead process's jobs are answered lost and how long a cancel asked from
        another process waits (a third of it).
    :ivar job_leases: The process's job leases, built here over :attr:`eval_host`'s storage and executor: every
        job this host starts — a run, a sweep, an analysis generation — carries one on its record.
    :ivar job_manager: The process's job manager, built here over :attr:`eval_host`'s storage and
        executor and holding :attr:`job_leases`, so the store a run's status is written to is the store
        its results are. It is the one a host hands :func:`~threetears.evals.run.lifecycle.cancel_run`
        and its shutdown, which writes every job it holds lost.
    """

    eval_host: EvalHost
    settings: Callable[[], LaunchSettings]
    job_timeout_factory: JobTimeoutFactory
    world_placements: Callable[[EvalRun], dict[str, WorldPlacement]] | None = None
    launch_pricer: LaunchPricer | None = None
    max_concurrent_jobs: int = MAX_CONCURRENT_JOBS
    cell_executor: CellExecutor | None = None
    on_job_progress: Callable[[str, dict[str, Any]], None] | None = None
    on_run_end: RunEndHook | None = None
    job_lease_s: float = DEFAULT_JOB_LEASE_S
    job_leases: JobLeases = field(init=False, repr=False, compare=False)
    job_manager: EvalJobManager = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Refuse a world placement that disagrees with the profile, and an apparatus setting it does not declare; build
        the jobs.

        :raises ValueError: The profile declares a world and no placement was supplied, or declares
            none and one was; a kind honours an apparatus setting the profile does not declare as
            one of the host's own apparatus dimensions; or ``job_lease_s`` is not positive.
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
        leases = JobLeases(self.eval_host.storage, lease_s=self.job_lease_s)
        object.__setattr__(self, "job_leases", leases)
        object.__setattr__(
            self,
            "job_manager",
            EvalJobManager(
                self.eval_host.storage,
                self.max_concurrent_jobs,
                self.on_job_progress,
                job_timeout_factory=self.job_timeout_factory,
                blocking_executor=self.eval_host.blocking_executor,
                on_run_end=self.on_run_end,
                leases=leases,
            ),
        )

    @property
    def kinds(self) -> Mapping[str, LaunchableKind]:
        """The launch registry: every candidate kind this host launches, keyed by the name a template declares.

        Read off the host's template checks (``eval_host.template_checks.kinds``), where the host states it once, so a
        template is admitted against the registry its launch dispatches through. A host with no template checks
        launches no kind. A template naming a kind with no entry is refused before anything is read.

        :return: The registry, empty for a host with no template checks.
        :rtype: Mapping[str, LaunchableKind]
        """
        checks = self.eval_host.template_checks
        return {} if checks is None else checks.kinds

    def place(self, run: EvalRun) -> dict[str, WorldPlacement]:
        """Where ``run`` sits in this host's world: one placement per declared dimension.

        :param run: The assembled run.
        :ptype run: EvalRun
        :return: The placements, or ``{}`` for a host with no world — the recording that the run placed
            nothing, which is a different fact from a run that recorded no placements at all.
        :rtype: dict[str, WorldPlacement]
        """
        return {} if self.world_placements is None else self.world_placements(run)


#: The engine's own apparatus dimensions, which have launch arguments of their own (the judge and
#: simulator pins, the cost ceiling) and so are never an apparatus SETTING.
_ENGINE_APPARATUS: frozenset[str] = frozenset(
    declared.name for declared in CORE_SWEEPABLES if declared.role == "apparatus"
)


def settable_apparatus(registry: SweepableRegistry) -> frozenset[str]:
    """The apparatus dimensions a launch may set: every one the host declares, but none of the engine's.

    :param registry: The host's sweepable registry (``HostProfile.sweepables``).
    :ptype registry: SweepableRegistry
    :return: The names of the host's own ``apparatus`` declarations.
    :rtype: frozenset[str]
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

    :param n_runs: How many runs the launch would start.
    :ptype n_runs: int
    :param ceiling: The host's :attr:`LaunchSettings.max_launch_arms`.
    :ptype ceiling: int
    :param setting: The host's name for that setting.
    :ptype setting: str
    :raises ValidationFailedError: ``n_runs`` exceeds ``ceiling``.
    """
    if n_runs > ceiling:
        raise ValidationFailedError(
            f"this launch would start {n_runs} arms (one run each: every model at every setting), more than one "
            f"launch may start together ({ceiling}, {setting})",
            remedy=f"launch fewer models or settings at once, or raise {setting}",
        )


def _normalized_cassette_mode(cassette_mode: str | None) -> CassetteMode:
    """Read a launch's cassette mode, blank spelling ``'off'``, refusing anything that is not a mode.

    Blank is normalised through :func:`~threetears.evals.kernel.arguments.normalize_blank`, the helper
    the cost estimate reads this field through: it exists because two surfaces once disagreed about
    what blank meant, so a second spelling of that rule here would recreate the divergence.

    :param cassette_mode: The mode the launch was given.
    :ptype cassette_mode: str | None
    :return: ``'off'``, ``'capture'`` or ``'replay'``.
    :rtype: CassetteMode
    :raises ValidationFailedError: ``cassette_mode`` is not blank and not a mode.
    """
    mode = normalize_blank(cassette_mode, "off")
    for known in _CASSETTE_MODES:
        if mode == known:
            return known
    raise ValidationFailedError(
        f"cassette_mode={cassette_mode!r} is not a mode", remedy="name one of 'off', 'capture' or 'replay'"
    )


def _refuse_a_corpus_the_mode_cannot_use(cassette_mode: CassetteMode, cassette_corpus_id: str | None) -> None:
    """Refuse a replay that names no corpus, and a corpus named for a launch that replays nothing.

    A replay serves the corpus one capture run recorded, and only the launch can say which: a
    replay with no corpus would have nothing to serve, and a corpus named beside another mode would
    be recorded as a condition nothing ran under. Read from the arguments alone, so it is refused
    before admission and before any read.

    :param cassette_mode: The launch's cassette mode, normalised.
    :ptype cassette_mode: CassetteMode
    :param cassette_corpus_id: The capture run whose corpus the launch replays, or ``None``.
    :ptype cassette_corpus_id: str | None
    :raises ValidationFailedError: A replay with no corpus, or a corpus with any other mode.
    """
    if cassette_mode == "replay" and cassette_corpus_id is None:
        raise ValidationFailedError(
            "cassette_mode='replay' names no corpus",
            remedy="pass cassette_corpus_id=<the id of the capture run whose recording this launch replays>",
        )
    if cassette_mode != "replay" and cassette_corpus_id is not None:
        raise ValidationFailedError(
            f"cassette_corpus_id={cassette_corpus_id!r} is a corpus to replay, and this launch's cassette_mode is "
            f"{cassette_mode!r}",
            remedy="name a corpus only with cassette_mode='replay'",
        )


def _refuse_a_corpus_that_cannot_serve(
    corpus_run: EvalRun | None, cassette_corpus_id: str, *, template: EvalTemplate, scope_id: str
) -> None:
    """Refuse a replay corpus that is not a capture of this template in this scope.

    :param corpus_run: The run the corpus id names, read in the launch's scope, or ``None``.
    :ptype corpus_run: EvalRun | None
    :param cassette_corpus_id: The id the launch named.
    :ptype cassette_corpus_id: str
    :param template: The template the launch runs.
    :ptype template: EvalTemplate
    :param scope_id: The launch's scope.
    :ptype scope_id: str
    :raises ValidationFailedError: No run of that id in the scope (a missing run and one in another scope
        read alike, since a run is read only in its own scope), one that did not capture, or one
        that captured another template's cases.
    """
    if corpus_run is None:
        raise ValidationFailedError(
            f"cassette_corpus_id={cassette_corpus_id!r} names no run in scope {scope_id!r}; a replay serves the "
            "corpus of a capture run in its own scope",
            remedy=f"name a capture run of this scope ({FIND_RUN})",
        )
    if corpus_run.cassette_mode != "capture":
        raise ValidationFailedError(
            f"run {cassette_corpus_id!r} ran with cassette_mode={corpus_run.cassette_mode!r}, so it recorded no "
            "corpus to replay",
            remedy="name a run launched with cassette_mode='capture'",
        )
    if corpus_run.template_id != template.id:
        raise ValidationFailedError(
            f"run {cassette_corpus_id!r} captured template {corpus_run.template_id!r}, and this launch runs "
            f"{template.id!r}; its corpus holds no recording of this template's cases",
            remedy=f"name a capture run of template {template.id!r}",
        )


async def _refuse_an_unknown_subject(
    host: LaunchHost, launchable: LaunchableKind, subject_id: str, scope_id: str
) -> None:
    """Refuse a launch naming a subject its kind's host does not hold — at admission, before anything is priced.

    :raises NotFoundError: The kind's :attr:`LaunchableKind.resolve_subject` holds no such subject; its remedy is the
        kind's.
    """
    unknown = await run_blocking(host.eval_host.blocking_executor, launchable.resolve_subject, subject_id, scope_id)
    if unknown is not None:
        raise NotFoundError("subject", subject_id, remedy=unknown.remedy)


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
    (:func:`start_run` refuses a repeat), and every run one :func:`start_run` call launches carries the
    same overlays, so no two of its members can be one arm under two names. A caller composing one group
    from several calls (``launch_group=``), each at its own overlays — the arms of a factorial, which differ
    in them — holds its arms distinct itself.

    :ivar id: The group's id, stamped on every member run.
    :ivar candidate_models: Every candidate model the launch names, across all its arms — the set the
        judge is chosen against (:func:`resolve_judge_pin`, and the launch tail that holds a launcher to
        it), so a candidate that is also the default judge moves every sibling's judge rather than its
        own run's alone.
    :ivar members: The prepared runs, their work functions and their job timeouts.
    """

    def __init__(self, candidate_models: Sequence[str]) -> None:
        """Mint the group's id; it is stamped on every member run.

        :param candidate_models: Every candidate model the launch names. Required, because the judge is
            chosen against it: a group built without it could hand the launch a judge that is one
            of its own candidates.
        :ptype candidate_models: Sequence[str]
        """
        self.id = str(uuid.uuid7())
        self.candidate_models = list(candidate_models)
        #: Whether the group's runs declared latency under test — decided by its first member, and held of
        #: every other (:meth:`add`). The job manager then gives the group every slot and runs its members
        #: one at a time.
        self.measure_latency: bool | None = None
        self.members: list[tuple[EvalRun, WorkFn, float]] = []
        self._teardowns: list[contextlib.AsyncExitStack] = []
        self._resolved: dict[tuple[str, ...], Any] = {}

    async def resolve_once(self, key: tuple[str, ...], compute: Callable[[], Awaitable[Any]]) -> Any:
        """Resolve a launch-wide input the first time an arm asks for it, and hand every later arm the same.

        A computation that raises stores nothing, so the refusal reaches whichever arm asked, and the
        launch abandons as it does for any refusal.

        :param key: What is being resolved, and every input it depends on.
        :ptype key: tuple[str, ...]
        :param compute: Resolves it; awaited at most once per key.
        :ptype compute: Callable[[], Awaitable[Any]]
        :return: The resolved value.
        :rtype: Any
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
        """Hold one prepared run, and what it must release if the group never starts.

        :raises ValueError: The run declares latency under test and the group's other runs do not, or the other way
            round. Arms measured as one experiment are measured one way: side by side their latency is read
            under contention, one at a time it is not, and comparing the two would compare the conditions.
            Nothing is held; the caller releases ``teardown`` with the rest of its refusal.
        """
        declared = bool(run.measure_latency)
        if self.measure_latency is not None and declared != self.measure_latency:
            raise ValueError(
                f"launch group {self.id} mixes runs that declare latency under test with runs that do not; launch "
                "every arm of one group with the same measure_latency, so their latency is read under one condition"
            )
        self.measure_latency = declared
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
                # NOSILENT: the caller re-raises the refusal that abandoned the group, which never started, so no
                # member ran; what failed is releasing this process's clients,
                # which nothing a member's record measures, and an error_details entry would count as a run's fault.
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

    :ivar template: The template, loaded.
    :ivar kind: The kind the dispatch read off the template, carried to the launch tail for the run
        to record rather than read again. Not named ``candidate_kind``, because every attribute
        of that name is a reader the once-per-tier canary counts, and this is a carried value.
    :ivar subject_id: The launch's subject, as the caller named it.
    :ivar candidate_model: This arm's candidate model as the caller named it, or ``None`` when the caller
        named none and the kind has a role default to run on.
    :ivar k_runs: Repeats per case.
    :ivar scope_id: The scope the template was read in, which the launch's runs live in.
    :ivar n_variations: New cases to generate.
    :ivar variation_model: The model that writes the template's ``llm`` variation axes' values when the
        launch generates (``n_variations`` > 0), as the caller named it. The dispatch has already
        made it present exactly when the template has an ``llm`` axis and the launch generates, so
        ``None`` means no model writes this launch's cases. A launcher builds the writer through its
        engine's writer for it (:attr:`generation`), built from the host's client factory in the ``variation``
        role (never the simulator's — a kind with no simulated user still generates), makes the calls in
        :func:`~threetears.evals.gen.generate_variations`, whose counts record the model it resolved to and
        which :func:`launch_run` checks against this pin. Generation runs before the run
        exists, once per launch for every arm (:meth:`LaunchGroup.resolve_once`), so its calls are
        outside every run's cost cap and metered-call ceiling.
    :ivar judge_model: The run-level judge pin, unresolved.
    :ivar judge_config_ids: The per-dim judge-configuration selection.
    :ivar simulator_model: The simulator pin, unresolved.
    :ivar cassette_mode: The cassette mode, already normalised to ``'off'``, ``'capture'`` or ``'replay'``.
    :ivar cassette_corpus_id: For a replay, the capture run whose corpus it serves, checked to be a capture
        of this template in this scope; ``None`` for any other mode.
    :ivar overlays: The launch's overlays as the kind's model validated them, every field resolved; ``None``
        for a kind that declares no overlays. Read it through :meth:`overlays_as`. The run records
        exactly this model's JSON form.
    :ivar kind_spec: The template's kind spec as the kind's spec model validated it at this launch; ``None``
        for a kind that declares no spec. Read it through :meth:`kind_spec_as`. The run records
        exactly this model's JSON form.
    :ivar max_cost_usd: The per-run cost-cap override, already checked positive.
    :ivar max_metered_calls: The per-run metered-call ceiling override, already checked positive.
    :ivar apparatus_settings: The host-declared apparatus values this arm's rig is set up with: every one
        the kind honours (:attr:`LaunchableKind.apparatus_settings`), the launch's value where it set one
        and the kind's default where it did not, already validated; ``{}`` only for a kind whose rig a
        launch cannot set. The launcher sets its rig up from them, reading no default of its own, and the
        run records exactly these (``EvalRun.apparatus_settings``).
    :ivar generation_budget: For a launch that generates (``n_variations`` > 0), the out-of-run budget its
        generation's calls are priced against and ledgered through — one for the whole launch, shared
        by every arm, capped at the host's ``max_out_of_run_cost_usd``. The calls :attr:`generation` admitted
        were admitted under it, and are ledgered through it as they are made; a generation a model wrote
        whose calls this budget never ledgered is refused at the launch tail. ``None`` for a launch
        that generates nothing.
    :ivar generation: For a launch whose generation a model writes, its calls as the launch's call admitted them,
        before it returned: the writer they were priced on, :attr:`generation_budget`, and each ``llm`` axis's
        admitted call. A launcher hands it to :func:`~threetears.evals.gen.generate_variations` as
        ``generation`` and builds no writer of its own; the launch holds the writer open around its launchers
        and releases it. ``None`` for a launch that generates nothing, or whose template no model writes.
    :ivar arm_plan: What the kind planned this arm to run (:attr:`LaunchableKind.plan_arm`) — the plan the arm
        was priced at, and the launch tail holds its run to — for an arm over stored cases and a generating
        one alike. ``None`` for a kind that plans nothing.
    :ivar arm_price: What the host's pricer predicted for this arm when :func:`price_arms` admitted it under an
        enforced cap — the mark that the arm was priced. ``None`` when no cap is in force (nothing is priced)
        or before the arm is priced; under an enforced cap the launch tail refuses an arm that carries none,
        so a launch composed outside :func:`start_run` cannot run an arm nobody priced.
    :ivar launch_group: The launch this run is prepared into; the launch starts it with its siblings.
    :ivar settings: The host's launch settings as the launch read them, once, when it began — what its
        refusals were made under, and what :func:`launch_run` records and enforces the run's ceilings
        and judge concurrency from. A launcher reads the host's settings from here, never afresh.
    :ivar measure_latency: Whether the launch declared latency under test: its run executes its cells one at a time
        and records it, and its group holds every job slot. ``False``: the cells run up to the narrower of
        :attr:`LaunchSettings.max_concurrent_cells` and what the kind declares (:attr:`KindWiring.max_concurrent_cells`)
        at once, and their latency is stamped read under concurrency.
    :ivar refused_goal_checks: Each of the stored template's goal checks the current grammar refuses, with why
        (:func:`~threetears.evals.schema.models.refused_goal_checks`). ``template`` already leaves them
        out (:func:`~threetears.evals.run.runner.template_as_graded`), so every launcher and every cell
        grades the rest, and the run records these (``EvalRun.refused_goal_checks``).
    :ivar cell_timeout_s: The per-cell deadline the launch named, in seconds, already checked positive and within
        the host's declared ceiling (:attr:`LaunchSettings.max_cell_timeout_s`); ``None`` runs every cell under
        the kind's own deadline. The launch tail holds it to the kind's deadline when the host declares no
        ceiling, and records the deadline every cell ran under (``EvalRun.cell_timeout_s``).
    :ivar case_set: The named, versioned case set the launch targets, as stored, or ``None`` for a launch over the
        template's cases. Its cases are :attr:`case_set_cases`; the launcher freezes exactly those
        (:meth:`cases_or`), the launch tail refuses a wiring that froze any others, and the run records the set
        (``EvalRun.case_set``).
    :ivar case_set_cases: The set's cases, resolved in the set's order; empty when :attr:`case_set` is ``None``.
    :ivar margins: The margins the launch declared on core rate measures, already checked
        (:func:`~threetears.evals.kernel.metrics.run_margin_refusal`), which the run records
        (``EvalRun.declared_margins``); empty when it declared none.
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
    refused_goal_checks: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    cell_timeout_s: float | None = None
    case_set: CaseSet | None = None
    case_set_cases: tuple[EvalTestCase, ...] = ()
    margins: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    measure_latency: bool = False
    generation: AdmittedGeneration | None = None

    def cases_or(self, template_cases: Sequence[EvalTestCase]) -> list[EvalTestCase]:
        """The cases this arm freezes: the case set's, when the launch names one, else the kind's own.

        :param template_cases: The cases the kind plays when the launch names no set.
        :ptype template_cases: Sequence[EvalTestCase]
        :return: The cases, in the order the run freezes them.
        :rtype: list[EvalTestCase]
        """
        return list(self.case_set_cases) if self.case_set is not None else list(template_cases)

    def overlays_as(self, model: type[_Validated]) -> _Validated:
        """The launch's overlays as the kind's own overlay model, typed.

        :param model: The overlay model the kind's contract declares.
        :ptype model: type[_Validated]
        :return: The validated overlays.
        :rtype: _Validated
        :raises TypeError: The kind's contract declares another model, or none — a launcher reading a
            model its kind does not declare, which no caller input can cause.
        """
        return _typed(self.kind, "overlays", self.overlays, model)

    def kind_spec_as(self, model: type[_Validated]) -> _Validated:
        """The template's kind spec as the kind's own spec model, typed.

        :param model: The spec model the kind's contract declares.
        :ptype model: type[_Validated]
        :return: The validated spec.
        :rtype: _Validated
        :raises TypeError: The kind's contract declares another model, or none.
        """
        return _typed(self.kind, "kind spec", self.kind_spec, model)


def _typed(kind: str, what: str, validated: BaseModel | None, model: type[_Validated]) -> _Validated:
    """``validated`` as ``model``, or a refusal naming both.

    :param kind: The kind, for the refusal.
    :ptype kind: str
    :param what: What ``validated`` is, for the refusal.
    :ptype what: str
    :param validated: The value the dispatch validated.
    :ptype validated: BaseModel | None
    :param model: The model the launcher reads it as.
    :ptype model: type[_Validated]
    :return: ``validated``, typed.
    :rtype: _Validated
    :raises TypeError: ``validated`` is not a ``model``.
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

#: The fields of a template a kind may never act on: what it states about tools, the cases it generates, a
#: conversation, the world (its seed, what it presumes, the checks graded against it) and the rubric a judge scores.
#: A template carrying one its kind ignores would run as if it said nothing, so a kind's registry entry declares them
#: (:attr:`LaunchableKind.ignored_template_fields`) and the dispatch refuses the template. Its identity, its intent and
#: its kind spec — validated by the kind's own model — are every kind's.
TemplateField = Literal[
    "tools_required",
    "tools_allowed",
    "variation_axes",
    "conversation",
    "world_seed",
    "preconditions",
    "goal_state_checks",
    "goal_check_controls",
    "rubric",
]

#: Every :data:`TemplateField`, in the template's order.
TEMPLATE_FIELDS: tuple[TemplateField, ...] = get_args(TemplateField)


@dataclass(frozen=True)
class LaunchableKind:
    """One entry of the launch registry a host hands :func:`start_run`: how a kind launches, and what it refuses.

    A kind with no entry is refused at the dispatch (:func:`no_launcher_for`), before anything is
    read or built, so a kind the runner could execute but no host can build collaborators for never
    reaches a cell.

    What a launch may turn on a kind's runs is not here: it is the kind's overlay model, declared on
    the host profile (:attr:`~threetears.evals.kernel.host.profile.HostProfile.kinds`), and the
    dispatch refuses an overlay that model refuses.

    :ivar launch: The kind's launcher.
    :ivar resolve_subject: Whether the host holds the subject a launch of this kind names (:data:`SubjectResolver`).
        Asked at admission — by :func:`quote_launch` and by every launch, once the template's kind is known and
        before any arm is priced — so an estimate and a launch refuse one unknown subject alike, and a launcher
        never meets one. The kind's, because what a subject id names is: a live entity for one kind, a prompt
        key for another, any label for a third. No default: a kind that does not say would have every launch's
        estimate price a subject its launch then refuses.
    :ivar unhonoured_launch_arguments: The engine's launch arguments a run of this kind cannot honour; a
        launch supplying one is refused at the dispatch, naming it. Empty for a kind that honours
        every one.
    :ivar plan_arm: What one arm of a launch of this kind will run, asked before the launcher is — the arm's
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
    :ivar ignored_template_fields: The template fields this kind never acts on (:data:`TemplateField`); a template
        stating any of them — anything but the field's default — is refused at the dispatch, naming it, rather
        than run as if it had said nothing. Empty for a kind that reads every one.
    :ivar apparatus_settings: The host-declared apparatus dimensions this kind's launcher sets its rig up
        from, each with the value its rig takes when a launch sets none — the kind's standing rig. A
        launch setting any other is refused at the dispatch; the :class:`LaunchHost` refuses a name the
        profile does not declare as one of the host's own apparatus dimensions. The dispatch RESOLVES a
        launch's settings against these defaults, so :attr:`LaunchRequest.apparatus_settings` names every
        one, the launcher reads them there with no default of its own, and the run records — and its
        context key hashes — the rig as it was set up: a launch naming a default and one leaving it out
        are one condition, as a ``kind_spec`` stating a default and one omitting it are. Empty for a kind
        whose rig a launch cannot set.
    :ivar price_largest_call: The largest reservation one call of a planned arm's cells will hold
        (:class:`CallCeiling`), priced on the clients the cells will call through, or ``None`` when no call of
        the arm can be priced. A subject kind, whose judge and simulator prompts carry transcripts that do not
        exist yet, composes :func:`~threetears.evals.run.price_subject_call_floor` here: a floor under those
        calls (:attr:`CallCeiling.lower_bound`), which refuses the same way. Asked of every planned arm — the request
        carries its plan — right after
        :attr:`plan_arm`, by the estimate and the launch alike, and awaited on the event loop, so it may build
        and release clients. An arm whose largest call is above its cap is refused before anything is spent
        (:func:`price_arms`): its run could never make that call. ``None`` for a kind that does not price its
        calls before launch — the engine then checks nothing here, and a call whose ceiling is above the cap is
        refused only when the run reaches it, which stops the run ``budget_stopped`` with that case excluded.
        Ignored for a kind that plans nothing.
    """

    launch: KindLauncher
    resolve_subject: SubjectResolver
    unhonoured_launch_arguments: frozenset[LaunchArgument] = frozenset()
    ignored_template_fields: frozenset[TemplateField] = frozenset()
    plan_arm: Callable[[LaunchRequest], ArmPlan] | None = None
    apparatus_settings: Mapping[str, ApparatusSettingValue] = field(default_factory=lambda: MappingProxyType({}))
    price_largest_call: Callable[[LaunchRequest], Awaitable[CallCeiling | None]] | None = None

    def __post_init__(self) -> None:
        """Refuse a name that is not a launch argument, and a default no run could store.

        :raises ValueError: An entry of ``unhonoured_launch_arguments`` is not a :data:`LaunchArgument`, one of
            ``ignored_template_fields`` is not a :data:`TemplateField`, or an apparatus default is not a string,
            a bool or a finite number.
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
        if undeclared := sorted(set(self.ignored_template_fields) - set(TEMPLATE_FIELDS)):
            raise ValueError(
                f"ignored_template_fields names {', '.join(undeclared)}, which are not template fields a kind may "
                f"ignore; those are {', '.join(TEMPLATE_FIELDS)}"
            )


@dataclass(frozen=True, kw_only=True)
class RunJudge:
    """A judged run's judge, as :func:`build_judge_service` resolved it from one config load.

    Handed to :func:`launch_run` on :attr:`KindWiring.judge`, which stamps the run's judge pin, its
    per-dim attribution and the config set from here — so what the run records and what scores it
    are one value.

    :ivar service: The judge service the runner drives.
    :ivar model: The run-level judge pin, resolved.
    :ivar selection: The configs the launch named, ``{dim_id: config_id}``; empty when it named none.
    :ivar effective_judges: The model each scored dim is requested from.
    :ivar configs: The configs the service was built from, ``{dim_id: config}``, over the dims that have one.
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
    def config_provenance(self) -> dict[str, KnownModelRoleOrigin]:
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

    :ivar kind_factory: Builds the kind for each cell, with the collaborators the launcher bound.
    :ivar subject: The subject the launcher captured for the request's ``subject_id``.
    :ivar test_cases: The frozen case set, in the order the runner walks it — every case in the
        request's scope and of its template.
    :ivar default_candidate_model: The model the kind ran on for an arm that named none — its role
        default. ``None`` when the request named one, which the run then records.
    :ivar judge: The run's judge, from :func:`build_judge_service`, for a judged kind; ``None`` for a
        kind whose grade is mechanical.
    :ivar simulator_model: The model that drives the simulated user, resolved, for a kind that runs
        one; ``None`` for a kind with none.
    :ivar payload: The host's opaque payload for the run, stored as its ``host_payload`` and never read.
    :ivar variation_counts: How many cases generation was asked for and froze, and the model that wrote
        the ``llm`` axes' values, for a launch that generated — the counts
        :func:`~threetears.evals.gen.generate_variations` returned; ``None`` for one that reused
        stored cases.
    :ivar turn_budget_s: The per-turn budget the kind runs its candidate's turns under, or ``None``.
    :ivar cell_timeout_s: A per-cell ceiling for a kind whose work legitimately outlasts the default,
        or ``None`` for the default.
    :ivar external_rates: The rate table for the run's counted external calls, or ``None`` to leave
        them counted and unpriced.
    :ivar teardown: What the run owns and must release when it ends, or ``None`` for nothing. Owned by
        the launch tail from the moment it is handed over.
    :ivar max_concurrent_cells: The most cells the kind declares it can drive at once — the kind's own
        :attr:`~threetears.evals.kernel.candidate_kind.CandidateKind.max_concurrent_cells`, handed here
        because the launch sizes the run before any cell builds a kind. ``None`` declares no limit of the
        kind's own. A run not measuring latency executes at the smallest of this, the host's
        :attr:`LaunchSettings.max_concurrent_cells` and its matrix; the runner refuses a kind whose own
        declaration is narrower than the run, so stating more here than the kind does fails the run before
        anything is spent. Defaults to ``1``, the safe width: a launcher that says nothing runs serially.
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
    max_concurrent_cells: int | None = 1

    def __post_init__(self) -> None:
        """Refuse a width no run can execute at.

        :raises ValueError: ``max_concurrent_cells`` is neither ``None`` nor an int of at least 1.
        """
        width = self.max_concurrent_cells
        if width is not None and (isinstance(width, bool) or not isinstance(width, int) or width < 1):
            raise ValueError(
                f"a kind declares the most cells it can drive at once as an int of at least 1, or None for no limit "
                f"of its own; got max_concurrent_cells={width!r}"
            )


def no_launcher_for(template_id: str, candidate_kind: str, launchable: Mapping[str, Any]) -> ValidationFailedError:
    """The refusal for a template whose kind this host cannot launch — one wording, wherever it fires.

    :param template_id: The template.
    :ptype template_id: str
    :param candidate_kind: The kind it declares.
    :ptype candidate_kind: str
    :param launchable: The kinds the host launches, keyed by kind, in the order the refusal lists them.
    :ptype launchable: Mapping[str, Any]
    :return: The error to raise.
    :rtype: ValidationFailedError
    """
    return ValidationFailedError(
        f"template {template_id!r} declares candidate_kind={candidate_kind!r}, which this "
        "host has no launcher for. A kind the runner can be wired with still needs a launch that builds its "
        "collaborators, because the runner cannot build a client it has never heard of",
        remedy=(
            f"launch a template of a kind this host launches ({', '.join(launchable) or 'it launches none'}), or "
            f"register a launcher for {candidate_kind!r} in the host's TemplateChecks.kinds"
        ),
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

    :param unhonoured: The kind's :attr:`LaunchableKind.unhonoured_launch_arguments`.
    :ptype unhonoured: frozenset[LaunchArgument]
    :param n_variations: New cases the launch asked to generate.
    :ptype n_variations: int
    :param judge_model: The launch's judge pin.
    :ptype judge_model: str | None
    :param judge_config_ids: The launch's per-dim judge-configuration selection.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: The launch's simulator pin.
    :ptype simulator_model: str | None
    :param cassette_mode: The launch's cassette mode, already normalised by
        :func:`_normalized_cassette_mode`; ``'off'`` is the absence of one and so is not reported.
    :ptype cassette_mode: str
    :return: ``{argument: the value the launch supplied}`` for each supplied argument the kind cannot
        honour, and ``{}`` when there is none. The values are carried so a refusal can echo what it
        is refusing rather than only naming the field.
    :rtype: dict[LaunchArgument, Any]
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

    :param template: The template.
    :ptype template: EvalTemplate
    :return: The names of its ``llm``-generated axes, in declaration order; empty when none is.
    :rtype: list[str]
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

    :param template: The loaded template.
    :ptype template: EvalTemplate
    :param n_variations: New cases the launch asked to generate, already checked non-negative.
    :ptype n_variations: int
    :param variation_model: The model the launch named to write the ``llm`` axes' values, or ``None``.
    :ptype variation_model: str | None
    :raises ValidationFailedError: The launch generates for a template with an ``llm`` axis and names
        no variation model, or names one while generating nothing or for a template with no
        ``llm`` axis.
    """
    llm_axes = _llm_axes(template)
    if variation_model is None:
        if n_variations > 0 and llm_axes:
            raise ValidationFailedError(
                f"template {template.id!r} has its {', '.join(repr(name) for name in llm_axes)} axis values written "
                f"by a model, and this launch asks for {n_variations} generated case(s) without naming one. It is "
                "its own role (the host's 'variation' client, never the simulator's), and the runs record it beside "
                "their generation counts",
                remedy="pass variation_model=<the model that writes them>",
            )
        return
    if n_variations == 0:
        raise ValidationFailedError(
            f"variation_model={variation_model!r} names the model that writes generated cases, and this launch "
            "generates none (n_variations=0)",
            remedy="launch it without variation_model, or ask for n_variations",
        )
    if not llm_axes:
        raise ValidationFailedError(
            f"variation_model={variation_model!r} names the model that writes a template's llm-generated axis "
            f"values, and template {template.id!r} has no llm axis, so no model writes its cases",
            remedy="launch it without variation_model",
        )


#: Validates a launch's apparatus settings as the run will store them.
_APPARATUS_SETTINGS = TypeAdapter(dict[str, ApparatusSettingValue])


def _validated_apparatus_settings(
    template: EvalTemplate, launchable: LaunchableKind, apparatus_settings: Mapping[str, Any] | None
) -> dict[str, ApparatusSettingValue]:
    """Refuse an apparatus setting the template's kind does not honour, and a value the run could not store.

    :param template: The loaded template, whose kind is the one asked.
    :ptype template: EvalTemplate
    :param launchable: That kind's registry entry, which names the settings its launcher reads.
    :ptype launchable: LaunchableKind
    :param apparatus_settings: The launch's settings, unvalidated, or ``None`` for none.
    :ptype apparatus_settings: Mapping[str, Any] | None
    :return: The settings as the run stores them: every one the kind honours, the launch's value where it set
        one and the kind's default where it did not — ``{}`` only for a kind whose rig a launch cannot set.
    :rtype: dict[str, ApparatusSettingValue]
    :raises ValidationFailedError: A setting the kind does not honour, or a value that is not a string, a bool
        or a finite number.
    """
    if not apparatus_settings:
        return dict(launchable.apparatus_settings)
    if unhonoured := sorted(set(apparatus_settings) - set(launchable.apparatus_settings)):
        honoured = ", ".join(sorted(launchable.apparatus_settings)) or "none"
        raise ValidationFailedError(
            f"template {template.id!r} is a {template.candidate_kind!r} template, and that kind's launcher sets its rig "
            f"up from no apparatus setting named {', '.join(repr(name) for name in unhonoured)}; "
            "a setting nothing reads would be recorded as a rig nobody built",
            remedy=f"set only what it reads: {honoured}",
        )
    try:
        # Resolved against the kind's standing rig: what the run records is the rig as it was set up.
        return {**launchable.apparatus_settings, **_APPARATUS_SETTINGS.validate_python(dict(apparatus_settings))}
    except ValidationError as e:
        raise ValidationFailedError(
            f"invalid apparatus_settings: {e.errors()[0]['msg']} ({e.errors()[0]['loc']})",
            remedy="give each apparatus setting a string, a bool or a finite number",
        ) from e


class _Launchable(NamedTuple):
    """What the refusals over a template and its kind resolved.

    :ivar kind: The kind the template declares, read once.
    :ivar launchable: Its registry entry.
    :ivar overlays: The launch's overlays, validated by the kind's model.
    :ivar kind_spec: The template's kind spec, validated by the kind's spec model.
    :ivar apparatus_settings: The launch's apparatus settings, validated against the kind.
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
    """Make every refusal that reads the template and its kind's registry entry, before the launch begins.

    :param host: The host, whose launch registry and profile declare the kind.
    :ptype host: LaunchHost
    :param template: The loaded template.
    :ptype template: EvalTemplate
    :param n_variations: New cases the launch asked to generate.
    :ptype n_variations: int
    :param variation_model: The model the launch named to write the template's ``llm`` axes' values.
    :ptype variation_model: str | None
    :param judge_model: The launch's judge pin.
    :ptype judge_model: str | None
    :param judge_config_ids: The launch's per-dim judge-configuration selection.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: The launch's simulator pin.
    :ptype simulator_model: str | None
    :param cassette_mode: The launch's cassette mode, already normalised.
    :ptype cassette_mode: str
    :param overlays: The launch's overlays, unvalidated.
    :ptype overlays: Mapping[str, Any] | None
    :param apparatus_settings: The launch's apparatus settings, unvalidated.
    :ptype apparatus_settings: Mapping[str, Any] | None
    :return: The kind the template declares, read once; its registry entry; the overlays and the template's
        kind spec as the kind's models validated them; and the apparatus settings as the run stores them.
    :rtype: _Launchable
    :raises ValidationFailedError: The template names a kind with no launcher, the launch supplies an
        argument the kind cannot honour, a negative ``n_variations``, a variation model the
        generation needs and the launch does not name or one nothing would call, an apparatus setting
        the kind does not honour, or the kind's models refuse the overlays or the spec.
    """
    if n_variations < 0:
        # Every surface reaches here; one without a wire-level bound would otherwise read a negative as
        # "generate nothing" and run the stored cases while the caller believed it asked for new ones.
        raise ValidationFailedError(
            f"n_variations must be 0 (reuse the stored cases) or more; got {n_variations}",
            remedy="pass 0, or the number of new cases to generate",
        )
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
            + ", ".join(f"{name}={value!r}" for name, value in unusable.items()),
            remedy="launch it without them",
        )
    refuse_ignored_template_fields(template, launchable)
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

    :ivar template: The template, loaded.
    :ivar kind: The kind it declares, read once.
    :ivar launchable: The kind's registry entry: its launcher, and its arm plan.
    :ivar overlays: The launch's overlays, validated by the kind's model.
    :ivar kind_spec: The template's kind spec, validated by the kind's spec model.
    :ivar apparatus_settings: The launch's apparatus settings, validated against the kind.
    :ivar case_set: The case set the launch targets and its cases in order, resolved; ``None`` for none.
    """

    template: EvalTemplate
    kind: str
    launchable: LaunchableKind
    overlays: BaseModel | None
    kind_spec: BaseModel | None
    apparatus_settings: dict[str, ApparatusSettingValue]
    case_set: tuple[CaseSet, tuple[EvalTestCase, ...]] | None = None


def _refuse_repeated_models(models: Sequence[str]) -> None:
    """Refuse a model named twice: each model is one arm.

    :param models: The launch's candidate models.
    :ptype models: Sequence[str]
    :raises ValidationFailedError: A model is named more than once.
    """
    if repeated := sorted({model for model in models if models.count(model) > 1}):
        raise ValidationFailedError(
            f"{', '.join(repr(model) for model in repeated)} named more than once; each model is one arm, so "
            "a second mention would measure that arm twice",
            remedy="name each model once, and raise k_runs for more repeats",
        )


async def _dispatch(
    host: LaunchHost,
    template_id: str,
    scope_id: str,
    *,
    subject_id: str,
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
    case_set: CaseSetRef | None = None,
) -> _Dispatched:
    """Load a launch's template, find its kind's launcher, and make the refusals that read them.

    :param host: The host, whose launch registry names the kind's launcher.
    :ptype host: LaunchHost
    :param template_id: The template to run.
    :ptype template_id: str
    :param scope_id: The scope the template is read in.
    :ptype scope_id: str
    :param subject_id: The subject the launch measures, asked of the template's kind.
    :ptype subject_id: str
    :param models: The launch's candidate models.
    :ptype models: list[str]
    :param n_variations: New cases the launch asked to generate.
    :ptype n_variations: int
    :param variation_model: The model the launch named to write the template's ``llm`` axes' values.
    :ptype variation_model: str | None
    :param judge_model: The launch's judge pin.
    :ptype judge_model: str | None
    :param judge_config_ids: The launch's per-dim judge-configuration selection.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: The launch's simulator pin.
    :ptype simulator_model: str | None
    :param cassette_mode: The launch's cassette mode, already normalised.
    :ptype cassette_mode: CassetteMode
    :param cassette_corpus_id: The capture run a replay serves, already checked against the mode.
    :ptype cassette_corpus_id: str | None
    :param overlays: The launch's overlays, unvalidated.
    :ptype overlays: Mapping[str, Any] | None
    :param apparatus_settings: The launch's apparatus settings, unvalidated.
    :ptype apparatus_settings: Mapping[str, Any] | None
    :param case_set: The case set the launch targets, or ``None``.
    :ptype case_set: CaseSetRef | None
    :return: The template, its kind, that kind's registry entry, the overlays and the template's kind spec as
        the kind's models validated them, the apparatus settings validated against the kind, and the case set
        with its cases.
    :rtype: _Dispatched
    :raises NotFoundError: The template is not found, or its kind's host holds no such subject.
    :raises ValidationFailedError: The template names a kind with no launcher, the launch supplies an
        argument the kind cannot honour, an overlay the kind's model refuses, a template kind spec
        the kind's spec model refuses, a negative ``n_variations``, a variation model the generation
        needs and the launch does not name or one nothing would call, an apparatus setting the kind does
        not honour, a model named twice, or a replay corpus that is not a capture of this template in this
        scope.
    """
    eval_host = host.eval_host
    template = await run_blocking(eval_host.blocking_executor, eval_host.storage.load_template, template_id, scope_id)
    if template is None:
        raise NotFoundError("template", template_id, remedy=FIND_TEMPLATE)

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
    # The subject is the kind's to know, so it is refused here, the moment the kind is — at admission, where the
    # estimate refuses it too — rather than by a launcher inside the launch's job, after every arm was priced.
    await _refuse_an_unknown_subject(host, resolved.launchable, subject_id, scope_id)
    if cassette_corpus_id is not None:
        corpus_run = await run_blocking(
            eval_host.blocking_executor, eval_host.storage.load_eval_run, cassette_corpus_id, scope_id
        )
        _refuse_a_corpus_that_cannot_serve(corpus_run, cassette_corpus_id, template=template, scope_id=scope_id)
    _refuse_repeated_models(models)
    resolved_set: tuple[CaseSet, tuple[EvalTestCase, ...]] | None = None
    if case_set is not None:
        if n_variations > 0:
            raise ValidationFailedError(
                f"case_set={case_set.label!r} names the cases to run and n_variations={n_variations} asks for new "
                "ones; a launch runs one or the other",
                remedy="drop case_set to generate new cases, or n_variations to run the set",
            )
        stored, cases = await run_blocking(
            eval_host.blocking_executor,
            partial(resolve_case_set, eval_host.storage, case_set, template=template, scope_id=scope_id),
        )
        resolved_set = (stored, tuple(cases))
    # The registry entry is the launcher, so a kind the host registered is a kind this launch
    # dispatches — there is no chain of kind comparisons here for a new kind to be missing from.
    return _Dispatched(
        template,
        resolved.kind,
        resolved.launchable,
        resolved.overlays,
        resolved.kind_spec,
        resolved.apparatus_settings,
        resolved_set,
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
    cell_timeout_s: float | None = None,
    case_set: CaseSetRef | None = None,
    margins: Mapping[str, float] | None = None,
    measure_latency: bool = False,
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

    :param host: The host: its storage, its launch registry of kinds, its settings and its job manager.
    :ptype host: LaunchHost
    :param template_id: Template to run.
    :ptype template_id: str
    :param subject_id: The subject the launch measures, as the host names it.
    :ptype subject_id: str
    :param models: Candidate models, one arm and one run each. A kind with a role default of its own
        runs one arm on that default when this is empty, and a kind with none refuses it.
    :ptype models: list[str]
    :param k_runs: Repeats per case for ``pass^k``.
    :ptype k_runs: int
    :param n_variations: New test cases to generate (0 = reuse existing).
    :ptype n_variations: int
    :param variation_model: The model that writes the template's ``llm`` variation axes' values —
        required when ``n_variations`` > 0 and the template has an ``llm`` axis, and refused
        otherwise. Asked of the host in the ``variation`` role and recorded on each run's
        ``variation_counts``. Its calls run before any run starts, once for every arm, and are
        outside every run's cost cap and metered-call ceiling.
    :ptype variation_model: str | None
    :param judge_model: The judge pin, unresolved.
    :ptype judge_model: str | None
    :param judge_config_ids: Optional per-dim judge-configuration selection, ``{dim_id: config_id}``.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: The simulator pin, unresolved.
    :ptype simulator_model: str | None
    :param cassette_mode: ``'off'`` (the default, and what blank spells), ``'capture'`` or ``'replay'``.
    :ptype cassette_mode: str | None
    :param cassette_corpus_id: For ``'replay'``, and only for it: the id of the capture run whose corpus
        the launch replays — a capture of this template in ``scope_id``.
    :ptype cassette_corpus_id: str | None
    :param overlays: The knobs this launch turns on its runs, by field of the kind's overlay model
        (:attr:`~threetears.evals.kernel.host.profile.HostProfile.kinds`). Validated before any arm is
        prepared; every run records the validated model, defaults included.
    :ptype overlays: Mapping[str, Any] | None
    :param apparatus_settings: Host-declared apparatus values this launch sets its runs' rig up with, by
        apparatus dimension — each one the kind declares it honours
        (:attr:`LaunchableKind.apparatus_settings`), valued a string, a bool or a finite number. So one
        template can be run at two values of, say, an adjudicator's seat and the runs compared. Every
        run records them (``EvalRun.apparatus_settings``), and they are part of its measurement context.
    :ptype apparatus_settings: Mapping[str, Any] | None
    :param max_cost_usd: Optional per-run cost-cap override; must be ``> 0`` and at most the host's configured
        ``max_cost_usd`` — a launch may only lower the host's ceiling.
    :ptype max_cost_usd: float | None
    :param max_metered_calls: Optional per-run metered-call ceiling override; must be ``> 0`` and at most the
        host's configured ``max_metered_calls``.
    :ptype max_metered_calls: int | None
    :param scope_id: The scope the template is read in and the launch's runs live in.
    :ptype scope_id: str
    :param launch_group: Prepare the runs into this launch instead of starting them; the caller starts
        the group once every arm is prepared. When omitted, this call is the whole launch: it
        forms the group and starts it through :func:`launch_as_group`.
    :ptype launch_group: LaunchGroup | None
    :param admission: A reservation the caller already holds, of which this launch takes its runs'
        share instead of asking for room of its own. Ignored with ``launch_group``, whose
        caller's admission covers it. When omitted, a launch that owns its group is admitted here.
    :ptype admission: AdmissionTicket | None
    :param cell_timeout_s: The deadline each cell of the launch's runs runs under, in seconds, in place of the
        kind's own; ``None`` keeps the kind's. Bounded by the host's ceiling
        (:attr:`LaunchSettings.max_cell_timeout_s`, or the kind's own deadline when the host declares none,
        so an undeclared host can only be lowered). The run's job budget is sized from it, and the run
        records it with its origin (``EvalRun.cell_timeout_s``, ``cell_timeout_s_origin``).
    :ptype cell_timeout_s: float | None
    :param case_set: A named, versioned case set of the template to run in place of the kind's own choice of cases
        (:func:`~threetears.evals.run.case_sets.mint_case_set`). Every arm runs exactly its cases, in its
        order, and each run records it (``EvalRun.case_set``). Refused beside ``n_variations`` (a launch runs
        the set's cases or new ones, not both), and when the set is another template's or a case of it no
        longer resolves.
    :ptype case_set: CaseSetRef | None
    :param margins: Margins on core rate measures (``{"accuracy": 0.05}``), the most two arms may differ on one and
        still be alike, declared before any result exists. A core measure's descriptor declares none, so
        this is how a comparison of these runs can read ``equivalent`` on it. Every run records them
        (``EvalRun.declared_margins``), outside its measurement context; the analysis reads one only when
        every run of the campaign declares it alike.
    :ptype margins: Mapping[str, float] | None
    :param measure_latency: Declare latency under test. Each run then executes its cells one at a time, and the
        launch's runs execute one at a time with nothing else beside them, so the latency they record is
        read clean. ``False`` (the default) runs each run's cells up to the narrower of the host's
        :attr:`LaunchSettings.max_concurrent_cells` and what its kind declares it is safe at
        (:attr:`KindWiring.max_concurrent_cells`) at once: far faster, and the latency still recorded is
        stamped read under concurrency and kept out of every comparison, bar and ranking. Every run
        records the declaration (``EvalRun.measure_latency``) and the width it ran at
        (``EvalRun.cell_concurrency``).
    :ptype measure_latency: bool


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

    :return: The persisted runs, one per model in the order given (status ``pending``).
    :rtype: list[EvalRun]
    :raises NotFoundError: The template is not found.
    :raises AdmissionRefusedError: The runs would pass the host's admission ceiling — raised before the
        template is read, so the refused launch prepared nothing.
    :raises ValidationFailedError: A model named twice, more runs than one launch may start, a negative
        ``n_variations``, a variation model the generation needs and the launch does not name or one
        nothing would call, a ``k_runs`` outside the run's bounds, a non-positive ``max_cost_usd``
        or ``max_metered_calls`` or one above the host's configured ceiling (or any ``max_metered_calls`` on a host
        declaring no metered tools), a ``cassette_mode`` that is not a mode, a replay naming no corpus
        or a corpus that is no capture of this template in this scope, a template naming a
        kind with no launcher, a launch argument that kind cannot honour, an overlay the kind's
        model refuses (named by field), an apparatus setting the kind does not honour, an arm
        predicted above its cap or unpriceable under an inherited one, a plan the kind refuses to make,
        a margin on a measure that is not a core rate measure or that is not between 0 and 1, or any
        refusal the kind's launcher makes.
    """
    # Admitted, then started: the host's settings are read ONCE for the whole launch (in admit_launch), so a hot
    # reload part-way cannot refuse, after its generation was paid for, a launch admitted under the figures it
    # started with.
    return await (
        await admit_launch(
            host,
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
            cell_timeout_s=cell_timeout_s,
            case_set=case_set,
            margins=margins,
            measure_latency=measure_latency,
        )
    ).start()


async def admit_launch(
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
    cell_timeout_s: float | None = None,
    case_set: CaseSetRef | None = None,
    margins: Mapping[str, float] | None = None,
    measure_latency: bool = False,
) -> AdmittedLaunch:
    """The first half of :func:`start_run`: every refusal it makes and every arm priced — no launcher run, nothing
    spent.

    Takes :func:`start_run`'s arguments, reads the host's settings once, admits the launch against the host's
    admission ceiling (holding the room until it is started or discarded), loads and refuses the template, and plans
    and prices every arm by the launch's one rule. What it returns starts the launch
    (:meth:`AdmittedLaunch.start`): the kinds' launchers run there — a generating launch's case generation is
    paid for there — and the runs start together. A transport admits in its call and starts in a job
    (:func:`~threetears.evals.ops.run_launch`), so its call waits on no model.

    Its parameters and refusals are :func:`start_run`'s, except the refusals a kind's launcher makes, which
    :meth:`AdmittedLaunch.start` raises.

    :return: The admitted launch.
    :rtype: AdmittedLaunch
    """
    # The host's settings, read ONCE for the whole launch: every refusal and every arm's tail reads this snapshot.
    return await _admit_run(
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
        cell_timeout_s=cell_timeout_s,
        case_set=case_set,
        margins=margins,
        measure_latency=measure_latency,
    )


def _refused_margins_or(margins: Mapping[str, float] | None) -> dict[str, float]:
    """The launch's run-scoped margins, refusing one no run may declare, before anything is prepared.

    :raises ValidationFailedError: A margin on a measure that is not a core rate measure, or not between 0 and 1.
    """
    declared = dict(margins or {})
    if refusals := [refusal for name, margin in declared.items() if (refusal := run_margin_refusal(name, margin))]:
        raise ValidationFailedError(
            "; ".join(refusals),
            remedy=(
                f"declare a run margin only on {', '.join(sorted(RUN_MARGIN_MEASURES))}, as a number between 0 "
                "and 1 in its units"
            ),
        )
    return {name: float(margin) for name, margin in sorted(declared.items())}


def _refuse_raised_ceilings(
    settings: LaunchSettings, *, max_cost_usd: float | None, max_metered_calls: int | None
) -> None:
    """Refuse a launch whose per-run override would RAISE the host's ceiling, before anything is paid for.

    The rule is :func:`~threetears.evals.run.ceilings.refuse_raised_ceiling`, stated once for both
    currencies; this applies it to a launch's two overrides against its one settings snapshot and
    turns its refusal into the launch's own error. Every launch surface — a single launch and its quote —
    reaches it, so none can admit a raised ceiling another refuses.

    :param settings: The launch's settings snapshot.
    :ptype settings: LaunchSettings
    :param max_cost_usd: The per-run cost-cap override.
    :ptype max_cost_usd: float | None
    :param max_metered_calls: The per-run metered-call ceiling override.
    :ptype max_metered_calls: int | None
    :raises ValidationFailedError: Either override is above the host's configured ceiling.
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
        raise ValidationFailedError(refused.fault, remedy=refused.remedy) from refused


def _refuse_cell_timeout(settings: LaunchSettings, cell_timeout_s: float | None, *, ceiling: float | None) -> None:
    """Refuse a per-cell deadline no cell could run under, or one above the ceiling it is held to.

    Asked twice with one rule: before anything is read, against the host's declared ceiling (``None`` there checks
    only the value), and at the launch tail against the ceiling in force — the declared one, or the kind's own
    deadline for a host that declares none, which a launch may then only lower.

    :param settings: The launch's settings snapshot, which names the host's ceiling in a refusal.
    :ptype settings: LaunchSettings
    :param cell_timeout_s: The launch's per-cell deadline, or ``None`` for none.
    :ptype cell_timeout_s: float | None
    :param ceiling: The most it may be, or ``None`` for no ceiling to check here.
    :ptype ceiling: float | None
    :raises ValidationFailedError: The deadline is not a finite number above zero, or is above ``ceiling``.
    """
    if cell_timeout_s is None:
        return
    if not (math.isfinite(cell_timeout_s) and cell_timeout_s > 0):
        raise ValidationFailedError(
            f"cell_timeout_s must be a finite number of seconds above 0 (got {cell_timeout_s}); a cell with no time "
            "to run measures nothing",
            remedy="pass one, or omit it for the kind's own deadline",
        )
    if ceiling is not None and cell_timeout_s > ceiling:
        declared = settings.max_cell_timeout_s is not None
        raise ValidationFailedError(
            f"cell_timeout_s={cell_timeout_s:g} is above the {'host' if declared else 'kind'}'s ceiling of "
            f"{ceiling:g}s"
            + (
                f" ({settings.name_of('max_cell_timeout_s')})"
                if declared
                else f"; the host declares no {settings.name_of('max_cell_timeout_s')}, so a launch may only lower "
                "the kind's own deadline"
            ),
            remedy=f"pass a cell_timeout_s at or below {ceiling:g}",
        )


def _refuse_launch_arguments(
    host: LaunchHost,
    settings: LaunchSettings,
    *,
    k_runs: int,
    max_cost_usd: float | None,
    max_metered_calls: int | None,
    cassette_mode: str | None,
    cassette_corpus_id: str | None,
    cell_timeout_s: float | None = None,
) -> CassetteMode:
    """The refusals a launch makes of its arguments alone, before it reads anything — one place, for the launch and its
    quote.

    :param host: The host, named in a refusal.
    :ptype host: LaunchHost
    :param settings: The launch's settings snapshot.
    :ptype settings: LaunchSettings
    :param k_runs: Repeats per case.
    :ptype k_runs: int
    :param max_cost_usd: The per-run cost-cap override.
    :ptype max_cost_usd: float | None
    :param max_metered_calls: The per-run metered-call ceiling override.
    :ptype max_metered_calls: int | None
    :param cassette_mode: The cassette mode, unnormalised.
    :ptype cassette_mode: str | None
    :param cassette_corpus_id: The corpus a replay serves.
    :ptype cassette_corpus_id: str | None
    :param cell_timeout_s: The launch's per-cell deadline, or ``None``.
    :ptype cell_timeout_s: float | None
    :return: The cassette mode, normalised.
    :rtype: CassetteMode
    :raises ValidationFailedError: A ``cell_timeout_s`` that is not a finite number above zero, or above the
        host's declared ceiling; a non-positive ``max_cost_usd`` or ``max_metered_calls`` (or any
        ``max_metered_calls`` on a host declaring no metered tools), either one above the host's configured
        ceiling, a ``k_runs`` outside the run's bounds, a ``cassette_mode`` that is not a mode, or a corpus
        the mode cannot use.
    """
    # Per-run cost-cap override: a non-positive cap would stop the
    # run before the first result — reject it as a run-parameter error up
    # front, before any snapshot / generation spend.
    if max_cost_usd is not None and max_cost_usd <= 0:
        raise ValidationFailedError(
            f"max_cost_usd must be > 0 (got {max_cost_usd})", remedy="pass one above 0, or omit it for the host's"
        )
    # Same reasoning one currency over: a ceiling of zero or less would refuse the
    # candidate's first metered call and measure a candidate that could not search.
    if max_metered_calls is not None and max_metered_calls <= 0:
        raise ValidationFailedError(
            f"max_metered_calls must be > 0 (got {max_metered_calls})",
            remedy="pass one above 0, or omit it for the host's",
        )
    # A host that declares no metered tools has nothing for a ceiling to bound, and recording one would
    # claim a bound the run never had.
    if max_metered_calls is not None and settings.max_metered_calls is None:
        raise ValidationFailedError(
            f"max_metered_calls={max_metered_calls} names a metered-call ceiling, and host "
            f"{host.eval_host.profile.host_id!r} declares no metered tools, so it would bound nothing",
            remedy="launch without max_metered_calls",
        )
    _refuse_raised_ceilings(settings, max_cost_usd=max_cost_usd, max_metered_calls=max_metered_calls)
    _refuse_cell_timeout(settings, cell_timeout_s, ceiling=settings.max_cell_timeout_s)
    # The run model bounds k_runs, but it is built after the cases are resolved — so an out-of-range
    # value from a surface that does not bound it at the wire would be refused after generation had
    # been paid for. Checked against the model's own field, so the bound is stated once.
    k_field = EvalRun.model_fields["k_runs"]
    try:
        TypeAdapter(Annotated[int, *k_field.metadata]).validate_python(k_runs)
    except ValidationError as e:
        raise ValidationFailedError(
            f"invalid k_runs: {e.errors()[0]['msg']} (got {k_runs})",
            remedy="pass k_runs as a whole number of repeats within its bounds, or omit it for the default",
        ) from e
    mode = _normalized_cassette_mode(cassette_mode)
    _refuse_a_corpus_the_mode_cannot_use(mode, cassette_corpus_id)
    return mode


async def _admit_run(
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
    cell_timeout_s: float | None = None,
    case_set: CaseSetRef | None = None,
    margins: Mapping[str, float] | None = None,
    measure_latency: bool = False,
) -> AdmittedLaunch:
    """:func:`admit_launch` under a settings snapshot its caller read.

    Its parameters and refusals are :func:`start_run`'s; ``settings`` is the one read every refusal and every arm's
    tail is made under.

    :return: The admitted launch: every refusal made and every arm priced, no launcher run.
    :rtype: AdmittedLaunch
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
        cell_timeout_s=cell_timeout_s,
    )
    declared_margins = MappingProxyType(_refused_margins_or(margins))
    dispatch = partial(
        _dispatch,
        host,
        template_id,
        scope_id,
        subject_id=subject_id,
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
        case_set=case_set,
    )

    #: The writer a generating launch's calls were admitted on, built in the call and released once the
    #: launchers that make them have run.
    writers: list[BoundCompletionClient] = []

    async def priced(group: LaunchGroup, dispatched: _Dispatched) -> list[LaunchRequest]:
        """Every arm's request, planned and priced — no launcher has run, nothing is built and nothing spent.

        :param group: The launch the runs are prepared into.
        :ptype group: LaunchGroup
        :param dispatched: What the dispatch resolved.
        :ptype dispatched: _Dispatched
        :return: The priced requests, one per model in the order given.
        :rtype: list[LaunchRequest]
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
            cell_timeout_s=cell_timeout_s,
            measure_latency=measure_latency,
        )
        if declared_margins:
            requests = [replace(request, margins=declared_margins) for request in requests]
        # Every arm priced before the first launcher runs, whether it generates or not: a launcher is what
        # pays for the generation the arms share and what builds an arm's clients, so pricing an arm after
        # it would refuse a launch already billed. Priced exactly once, here.
        priced_requests = await price_arms(host, dispatched.launchable, requests)
        # And the generation's own calls, admitted here, in the call, against the out-of-run cap. Admitted rather
        # than only quoted, so the launcher makes exactly the calls priced here, on the writer they were priced on, and admits nothing in the job.
        return await _with_admitted_generation(
            host,
            dispatched.template,
            priced_requests,
            n_variations=n_variations,
            variation_model=variation_model,
            scope_id=scope_id,
            writers=writers,
        )

    async def launched(dispatched: _Dispatched, requests: list[LaunchRequest]) -> list[EvalRun]:
        """Hand each priced arm, one per model, to the kind's launcher — where a generating launch's calls are made."""
        async with contextlib.AsyncExitStack() as held:
            for writer in writers:
                await held.enter_async_context(writer)
            return [await dispatched.launchable.launch(request) for request in requests]

    async def prepare(_group: LaunchGroup, formed: tuple[_Dispatched, list[LaunchRequest]]) -> list[EvalRun]:
        return await launched(*formed)

    # Preparing arms into a caller's group: that caller admitted the launch, and it starts the
    # group once every arm is prepared or abandons it on a refusal, so this launch only prepares.
    if launch_group is not None:
        dispatched = await dispatch()
        return AdmittedLaunch.into(launch_group, (dispatched, await priced(launch_group, dispatched)), prepare)

    async def form() -> tuple[LaunchGroup, tuple[_Dispatched, list[LaunchRequest]]]:
        """Make the template's refusals, form the launch's group — one arm per model — and price every arm.

        :return: The group, and what the dispatch resolved with the priced requests.
        :rtype: tuple[LaunchGroup, tuple[_Dispatched, list[LaunchRequest]]]
        """
        dispatched = await dispatch()
        group = LaunchGroup(candidate_models=models)
        return group, (dispatched, await priced(group, dispatched))

    # This call is the whole launch. A caller holding a ticket for several launches
    # hands this one its share rather than letting it compete for room with launches that arrived
    # after the caller was admitted.
    return await admit_group(
        host,
        max(1, len(models)),
        settings=settings,
        form=form,
        prepare=prepare,
        admission=admission,
        event="eval.start_run",
    )


async def _with_admitted_generation(
    host: LaunchHost,
    template: EvalTemplate,
    requests: list[LaunchRequest],
    *,
    n_variations: int,
    variation_model: str | None,
    scope_id: str,
    writers: list[BoundCompletionClient],
) -> list[LaunchRequest]:
    """Admit a launch's case generation in its call, and carry the admission on every arm's request.

    The plan and the budget are the ones a quote reads (:func:`plan_variation_calls`,
    :func:`_generation_budget`), so a generation a quote passes is one this admits. The writer is built here
    and every call admitted on it, so the launcher makes exactly what was priced, on the client it was priced on;
    it is kept in ``writers`` for the launch to hold open around its launchers and release. A launch that
    generates nothing, or whose template no model writes, admits nothing and carries no generation.

    :param host: The host: its writer client and the store the existing cases are read from.
    :ptype host: LaunchHost
    :param template: The template whose ``llm`` axes the generation writes.
    :ptype template: EvalTemplate
    :param requests: The launch's priced requests, every one carrying the launch's generation budget.
    :ptype requests: list[LaunchRequest]
    :param n_variations: The cases the launch generates.
    :ptype n_variations: int
    :param variation_model: The writer model the launch named.
    :ptype variation_model: str | None
    :param scope_id: The scope the cases are generated into.
    :ptype scope_id: str
    :param writers: Where the built writer is kept for the launch to release.
    :ptype writers: list[BoundCompletionClient]
    :return: The requests, each carrying the admitted generation.
    :rtype: list[LaunchRequest]
    :raises ValidationFailedError: Under an enforced cap, a call the writer cannot price, or calls priced above it — the
        writer released, nothing called.
    :raises ValueError: The host supplies no completion clients.
    """
    budget = requests[0].generation_budget if requests else None
    if budget is None or n_variations <= 0 or variation_model is None or not _llm_axes(template):
        return requests
    existing = await run_blocking(
        host.eval_host.blocking_executor,
        partial(host.eval_host.storage.query_test_cases, scope_id, template_id=template.id),
    )
    planned = plan_variation_calls(template, n_variations, existing)
    writer = host.eval_host.completion_clients("a launch's generation")("variation", variation_model)
    try:
        await refresh_prices(writer)
        admitted = budget.admit(writer, "variation", list(planned.values()))
    except BaseException:
        async with writer:
            pass
        raise
    writers.append(writer)
    generation = AdmittedGeneration(writer=writer, budget=budget, calls=dict(zip(planned, admitted, strict=True)))
    return [replace(request, generation=generation) for request in requests]


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

    One construction for every reader, so nothing can pass a generation the launch would refuse.

    :param host: The host, whose store ledgers the calls.
    :ptype host: LaunchHost
    :param settings: The launch's settings snapshot; the cap is enforced exactly when its enforcement is.
    :ptype settings: LaunchSettings
    :param scope_id: The scope the generation is for.
    :ptype scope_id: str
    :param template_id: The template whose cases it writes.
    :ptype template_id: str
    :param subject_id: The launch's subject.
    :ptype subject_id: str
    :param launch_group_id: The launch's group, stamped on every ledger row.
    :ptype launch_group_id: str
    :return: The budget, nothing yet admitted.
    :rtype: OutOfRunBudget
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
    cell_timeout_s: float | None = None,
    measure_latency: bool = False,
) -> list[LaunchRequest]:
    """One request per arm — per model, or one on the kind's default when the launch names none.

    A launch that generates gets ONE out-of-run budget, shared by every arm's request: the arms share
    one generation (:meth:`LaunchGroup.resolve_once`), so its calls are admitted against one cap and
    ledgered under the launch's group.

    :param host: The host, whose store ledgers the generation.
    :ptype host: LaunchHost
    :param settings: The launch's settings snapshot, which caps the generation and travels on every request.
    :ptype settings: LaunchSettings
    :param dispatched: What the dispatch resolved.
    :ptype dispatched: _Dispatched
    :param group: The launch the arms are prepared into.
    :ptype group: LaunchGroup
    :param subject_id: The launch's subject.
    :ptype subject_id: str
    :param models: The launch's candidate models.
    :ptype models: Sequence[str]
    :param k_runs: Repeats per case.
    :ptype k_runs: int
    :param scope_id: The scope the runs live in.
    :ptype scope_id: str
    :param n_variations: New cases the launch generates.
    :ptype n_variations: int
    :param variation_model: The model that writes the ``llm`` axes' values.
    :ptype variation_model: str | None
    :param judge_model: The judge pin.
    :ptype judge_model: str | None
    :param judge_config_ids: The per-dim judge-configuration selection.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: The simulator pin.
    :ptype simulator_model: str | None
    :param cassette_mode: The cassette mode, normalised.
    :ptype cassette_mode: CassetteMode
    :param cassette_corpus_id: The corpus a replay serves.
    :ptype cassette_corpus_id: str | None
    :param max_cost_usd: The per-run cost-cap override.
    :ptype max_cost_usd: float | None
    :param max_metered_calls: The per-run metered-call ceiling override.
    :ptype max_metered_calls: int | None
    :param cell_timeout_s: The launch's per-cell deadline, or ``None`` for the kind's own.
    :ptype cell_timeout_s: float | None
    :param measure_latency: Whether the launch declared latency under test.
    :ptype measure_latency: bool
    :return: The requests, in arm order, with no arm plan yet.
    :rtype: list[LaunchRequest]
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
    # A template stored before a grammar rule can carry a check the rule refuses; every arm grades the rest.
    refused = refused_goal_checks(dispatched.template.goal_state_checks)
    graded = template_as_graded(dispatched.template, refused)
    return [
        LaunchRequest(
            template=graded,
            refused_goal_checks=MappingProxyType(dict(refused)),
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
            cell_timeout_s=cell_timeout_s,
            case_set=dispatched.case_set[0] if dispatched.case_set is not None else None,
            case_set_cases=dispatched.case_set[1] if dispatched.case_set is not None else (),
            measure_latency=measure_latency,
        )
        for arm_model in arm_models
    ]


async def _planned(host: LaunchHost, launchable: LaunchableKind, request: LaunchRequest) -> ArmPlan | None:
    """Ask the kind what one arm will run, and refuse a plan the arm contradicts.

    :param host: The host, whose blocking executor the plan is asked through — a plan may read the store.
    :ptype host: LaunchHost
    :param launchable: The kind's registry entry.
    :ptype launchable: LaunchableKind
    :param request: The arm.
    :ptype request: LaunchRequest
    :return: The plan, or ``None`` for a kind that plans nothing (:attr:`LaunchableKind.plan_arm`). For a kind that
        prices its calls (:attr:`LaunchableKind.price_largest_call`), it carries its largest call's ceiling.
    :rtype: ArmPlan | None
    :raises ValidationFailedError: The kind cannot plan the arm (its own refusal).
    :raises ValueError: The plan contradicts what the arm named — another model, another judge pin, a config
        selection it did not plan, or another simulator — a kind defect; or a client priced a call at an
        amount no reservation could hold.
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
    price_largest_call = launchable.price_largest_call
    if price_largest_call is None:
        return plan
    return replace(plan, call_ceiling=await price_largest_call(replace(request, arm_plan=plan)))


def _arm_described(request: LaunchRequest) -> str:
    """One arm, as a pricing refusal names it: its model, its cases by source, its repeats and its template.

    :param request: The arm, planned when its kind plans.
    :ptype request: LaunchRequest
    :return: The description.
    :rtype: str
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

    :param host: The host, whose pricer predicts.
    :ptype host: LaunchHost
    :param request: The arm, planned when its kind plans.
    :ptype request: LaunchRequest
    :return: The price, or a price with no prediction and the reason there is none.
    :rtype: ArmPrice
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

    :ivar described: The arm, as a refusal names it.
    :ivar candidate_model: The model it runs on — its plan's, or the one the launch named when its kind plans
        nothing (``None`` there for an arm on the kind's unnamed default).
    :ivar plan: What its kind planned it to run, or ``None`` for a kind that plans nothing.
    :ivar price: What the host's pricer predicted, or why nothing could.
    :ivar cap_usd: The cap its run will be held to; ``None`` when the host enforces no cap.
    :ivar cap_origin: ``chosen`` when the launch named the cap, ``inherited`` from the host's; ``None`` uncapped.
    :ivar outcome: What the rule made of it (:data:`ArmOutcome`).
    :ivar refusal: The refusal the launch makes of it, word for word; ``None`` unless ``outcome`` is ``refused``.
    :ivar remedy: What brings the arm under its cap, or names one; set exactly when ``refusal`` is.
    """

    described: str
    candidate_model: str | None
    plan: ArmPlan | None
    price: ArmPrice
    cap_usd: float | None
    cap_origin: Literal["chosen", "inherited"] | None
    outcome: ArmOutcome
    refusal: str | None
    remedy: str | None = None


def _cap_of(request: LaunchRequest) -> tuple[float | None, Literal["chosen", "inherited"] | None]:
    """The cap ``request``'s run will be held to, and where it came from — from the launch's one settings read.

    :param request: The arm.
    :ptype request: LaunchRequest
    :return: The cap and its origin, or ``(None, None)`` when the host enforces no cap.
    :rtype: tuple[float | None, Literal['chosen', 'inherited'] | None]
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


def _cents_up(usd: float) -> float:
    """``usd`` rounded up to the next whole cent, so a cap named at it is never below it."""
    return math.ceil(usd * 100 - 1e-9) / 100


def _largest_call_over_cap(
    request: LaunchRequest, ceiling: CallCeiling, cap: float, price: ArmPrice, *, what: str, refused: str
) -> tuple[str, str]:
    """The refusal of an arm whose largest call's ceiling is above its cap, and its remedy.

    :param request: The arm.
    :ptype request: LaunchRequest
    :param ceiling: Its largest call's ceiling.
    :ptype ceiling: CallCeiling
    :param cap: The cap its run would be held to.
    :ptype cap: float
    :param price: Its price, whose prediction (when there is one) sizes the room the rest of the run needs.
    :ptype price: ArmPrice
    :param what: The arm, as a refusal names it.
    :ptype what: str
    :param refused: When the refusal is made, in words.
    :ptype refused: str
    :return: The refusal and the remedy.
    :rtype: tuple[str, str]
    """
    needed = _cents_up(ceiling.usd)
    host_ceiling = request.settings.max_cost_usd
    if ceiling.lower_bound:
        # A floor: the real call carries its transcript on top, so the cap it needs is MORE than this, by an amount
        # nobody can know before the run — the remedy says so rather than name a cap that would admit the arm.
        refusal = (
            f"{what} cannot make {ceiling.call} under its ${cap:.2f} cap: even the smallest such call this arm could "
            f"make — its fixed parts alone, before any transcript or evidence is added — is reserved at "
            f"${ceiling.usd:.4f} before it is sent, more than the whole cap, so one such call alone would put the run "
            "past it. A cap is read between cells and refuses no call inside one, so the run would make the call "
            f"and stop budget_stopped before its next cell, after spending more than the cap on it. {refused}"
        )
        if needed <= host_ceiling:
            raise_it = (
                f"name max_cost_usd above ${needed:.2f} — a floor, not the cap the arm needs: the real call adds its "
                "transcript and evidence to that smallest one, so leave room for them and for the rest of the run"
            )
        else:
            raise_it = (
                f"even the smallest such call needs a cap above ${needed:.2f}, beyond the host's ${host_ceiling:.2f} "
                f"ceiling, which a launch cannot raise — only the host's operator can, through "
                f"{request.settings.name_of('max_cost_usd')}"
            )
        return refusal, f"{raise_it}; or lower the call's floor: {ceiling.lowered_by}"
    refusal = (
        f"{what} cannot make its largest call under its ${cap:.2f} cap: {ceiling.call} is reserved at up to "
        f"${ceiling.usd:.4f} before it is sent, more than the whole cap, so one such call alone would put the run "
        "past it. A cap is read between cells and refuses no call inside one, so the run would make the call and "
        f"stop budget_stopped before its next cell, after spending more than the cap on it. {refused}"
    )
    room = (
        f" (${_cents_up(ceiling.usd + price.predicted_usd):.2f} also leaves room for the "
        f"${price.predicted_usd:.2f} the arm is predicted to spend around it)"
        if price.predicted_usd is not None
        else ""
    )
    if needed <= host_ceiling:
        raise_it = f"name max_cost_usd of at least ${needed:.2f}, the cap one such call needs to be admitted{room}"
    else:
        raise_it = (
            f"one such call needs a cap of at least ${needed:.2f}, above the host's ${host_ceiling:.2f} ceiling, "
            f"which a launch cannot raise — only the host's operator can, through "
            f"{request.settings.name_of('max_cost_usd')}"
        )
    return refusal, f"{raise_it}; or lower the call's ceiling: {ceiling.lowered_by}"


def _judged(request: LaunchRequest, price: ArmPrice) -> ArmVerdict:
    """The one pricing rule, over one planned arm and its price.

    First, an arm is refused when its largest call's ceiling (:attr:`ArmPlan.call_ceiling`) is above its cap on
    its own: its run reserves that ceiling before making the call and refuses the call, so the run could never
    complete it, whatever the arm is predicted to cost in all. Then an arm is refused when its predicted cost is
    above the cap its run will be held to, or when nothing
    predicts it and that cap is one the run would merely inherit (the host's configured ceiling, which nobody
    chose for this run): unknown is not $0. An arm with a cap the launch named and no prediction goes ahead —
    the named cap is the most the operator chose to risk on a run nobody could price, and the run's own cost
    cap enforces it as the spend arrives.

    :param request: The arm, planned when its kind plans.
    :ptype request: LaunchRequest
    :param price: Its price.
    :ptype price: ArmPrice
    :return: The verdict.
    :rtype: ArmVerdict
    """
    cap, origin = _cap_of(request)
    what = _arm_described(request)
    plan = request.arm_plan
    model = plan.candidate_model if plan is not None else request.candidate_model

    def verdict(outcome: ArmOutcome, refusal: str | None = None, remedy: str | None = None) -> ArmVerdict:
        return ArmVerdict(
            described=what,
            candidate_model=model,
            plan=plan,
            price=price,
            cap_usd=cap,
            cap_origin=origin,
            outcome=outcome,
            refusal=refusal,
            remedy=remedy,
        )

    if cap is None:
        return verdict("uncapped")
    refused = (
        "Refused before the generation was paid for" if request.n_variations > 0 else "Refused before any launcher ran"
    )
    if plan is not None and plan.call_ceiling is not None and plan.call_ceiling.usd > cap:
        return verdict(
            "refused", *_largest_call_over_cap(request, plan.call_ceiling, cap, price, what=what, refused=refused)
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
            f"inherited from {request.settings.name_of('max_cost_usd')}, which nobody chose for it.{unplanned} "
            f"{refused}",
            f"launch it naming max_cost_usd (at most ${request.settings.max_cost_usd:.2f}; a launch may only lower "
            "the host's ceiling) — the most you will risk on a run nobody can price; what it costs then prices the "
            "next launch",
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
            f"{refused}",
            f"fewer cases, a smaller k_runs or a cheaper model brings it under; {raise_it}",
        )
    return verdict("admitted")


async def _planned_arms(
    host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]
) -> list[LaunchRequest]:
    """Every arm, carrying what its kind planned — all planned before any is priced.

    Planning first is what puts a kind's request-level refusals (:attr:`LaunchableKind.plan_arm`) ahead of
    every pricing refusal: an operator who forgot something the kind needs hears that, not "cannot be priced".

    :param host: The host, whose executor each plan is asked through.
    :ptype host: LaunchHost
    :param launchable: The kind's registry entry.
    :ptype launchable: LaunchableKind
    :param requests: The launch's arms, in order.
    :ptype requests: Sequence[LaunchRequest]
    :return: The requests with their plans.
    :rtype: list[LaunchRequest]
    :raises ValidationFailedError: The kind cannot plan an arm.
    :raises ValueError: A plan contradicts what its arm named.
    """
    return [replace(request, arm_plan=await _planned(host, launchable, request)) for request in requests]


async def price_arms(
    host: LaunchHost, launchable: LaunchableKind, requests: Sequence[LaunchRequest]
) -> list[LaunchRequest]:
    """Plan and price every arm of one launch by the engine's one rule, refusing before any launcher runs.

    The step every launch takes between building its arms' requests and calling their launchers —
    :func:`start_run` calls it, and a host composing its own launch through
    :func:`launch_as_group` must call it too: under an enforced cap the launch tail refuses an arm that did
    not come through here (:attr:`LaunchRequest.arm_price`). Every arm is planned before any is priced, and
    every arm is priced before the first launcher runs. With the host's enforcement off no cap is in force
    and nothing is priced; every arm is still planned, so the launch tail holds its launcher to its plan.

    :param host: The host: its pricer predicts.
    :ptype host: LaunchHost
    :param launchable: The kind's registry entry, whose ``plan_arm`` plans each arm.
    :ptype launchable: LaunchableKind
    :param requests: The launch's arms, in order; every one carries the same ``max_cost_usd`` and the same
        settings snapshot.
    :ptype requests: Sequence[LaunchRequest]
    :return: The requests, each carrying its plan (``None`` for a kind that plans nothing) and, under an enforced
        cap, its price.
    :rtype: list[LaunchRequest]
    :raises ValidationFailedError: The kind cannot plan an arm; an arm whose largest call's ceiling is above its cap,
        one predicted above its cap, or one unpriceable under an inherited cap.
    :raises ValueError: The kind planned an arm contradicting what it named, or ``requests`` is empty.
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
            assert judged.remedy is not None  # set with every refusal
            raise ValidationFailedError(judged.refusal, remedy=judged.remedy)
        priced.append(replace(request, arm_price=price))
    return priced


@dataclass(frozen=True, kw_only=True)
class LaunchQuote:
    """What a launch would make of its arms' prices, made by the launch's own steps and launching nothing.

    :ivar template_id: The template.
    :ivar subject_id: The subject.
    :ivar scope_id: The scope whose history a history pricer reads.
    :ivar cassette_mode: The cassette mode, normalised.
    :ivar k_runs: Repeats per case.
    :ivar n_variations: Cases the launch would generate first; ``0`` for stored cases.
    :ivar case_count: The case count the quote put in place of each plan's, for a hypothetical grid; ``None``
        when every arm is quoted at its plan, as the launch would price it.
    :ivar arms: Every arm's verdict, in arm order.
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
    cell_timeout_s: float | None = None,
    case_set: CaseSetRef | None = None,
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

    :param host: The host.
    :ptype host: LaunchHost
    :param template_id: As :func:`start_run` takes it.
    :ptype template_id: str
    :param subject_id: As :func:`start_run` takes it.
    :ptype subject_id: str
    :param models: As :func:`start_run` takes it.
    :ptype models: list[str]
    :param k_runs: As :func:`start_run` takes it.
    :ptype k_runs: int
    :param n_variations: As :func:`start_run` takes it.
    :ptype n_variations: int
    :param variation_model: As :func:`start_run` takes it.
    :ptype variation_model: str | None
    :param judge_model: As :func:`start_run` takes it.
    :ptype judge_model: str | None
    :param judge_config_ids: As :func:`start_run` takes it.
    :ptype judge_config_ids: dict[str, str] | None
    :param simulator_model: As :func:`start_run` takes it.
    :ptype simulator_model: str | None
    :param cassette_mode: As :func:`start_run` takes it.
    :ptype cassette_mode: str | None
    :param cassette_corpus_id: As :func:`start_run` takes it.
    :ptype cassette_corpus_id: str | None
    :param overlays: As :func:`start_run` takes it.
    :ptype overlays: Mapping[str, Any] | None
    :param apparatus_settings: As :func:`start_run` takes it.
    :ptype apparatus_settings: Mapping[str, Any] | None
    :param max_cost_usd: As :func:`start_run` takes it.
    :ptype max_cost_usd: float | None
    :param max_metered_calls: As :func:`start_run` takes it.
    :ptype max_metered_calls: int | None
    :param scope_id: As :func:`start_run` takes it.
    :ptype scope_id: str
    :param case_count: A case count to quote every planned arm at in place of its plan's, for a hypothetical
        grid; ``None`` quotes each at its plan, as the launch prices it.
    :ptype case_count: int | None
    :param cell_timeout_s: As :func:`start_run` takes it; refused as the launch refuses it, and prices nothing.
    :ptype cell_timeout_s: float | None
    :param case_set: As :func:`start_run` takes it; resolved and refused as the launch resolves and refuses it.
    :ptype case_set: CaseSetRef | None
    :return: The quote.
    :rtype: LaunchQuote
    :raises NotFoundError: The template is not found.
    :raises ValidationFailedError: Any refusal :func:`start_run` makes before it prices an arm, or a
        ``case_count`` below one.
    :raises ValueError: The kind planned an arm contradicting what it named.
    """
    if case_count is not None and case_count < 1:
        raise ValidationFailedError(
            f"case_count prices a grid of at least one case; got {case_count}",
            remedy="pass one of at least 1, or omit it to price each arm at its plan",
        )
    settings = host.settings()
    mode = _refuse_launch_arguments(
        host,
        settings,
        k_runs=k_runs,
        max_cost_usd=max_cost_usd,
        max_metered_calls=max_metered_calls,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
        cell_timeout_s=cell_timeout_s,
    )
    _refuse_oversized_launch(max(1, len(models)), settings.max_launch_arms, settings.name_of("max_launch_arms"))
    dispatched = await _dispatch(
        host,
        template_id,
        scope_id,
        subject_id=subject_id,
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
        case_set=case_set,
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


class AdmittedLaunch:
    """A launch admitted, refused where it would be and priced — every arm's launcher not yet run, nothing spent.

    The two halves of a launch, apart. :func:`admit_group` (and :func:`admit_launch` over it) makes everything a
    launch checks before it spends: the arm count, the admission ceiling (the room it reserves is held here), the
    template's refusals and every arm's price. :meth:`start` makes everything that spends: each arm's launcher —
    which pays for a generating launch's cases and builds the arm's clients — and the group's start. A transport
    admits in its call and starts in a job, so no call it serves waits on a model; a script does both in a row
    (:func:`start_run`).

    One that will not be started is :meth:`discard`-ed, which hands its reserved room back.
    """

    def __init__(
        self,
        host: LaunchHost | None,
        group: LaunchGroup,
        formed: Any,
        prepare: Callable[[LaunchGroup, Any], Awaitable[list[EvalRun]]],
        *,
        event: str,
        ticket: AdmissionTicket | None,
        owns_group: bool,
        attach: Callable[[list[EvalRun]], Awaitable[None]] | None = None,
        detach: Callable[[list[EvalRun]], Awaitable[None]] | None = None,
    ) -> None:
        """Hold an admitted launch; built by :func:`admit_group` and :meth:`into`, never directly.

        :param host: The host; its job manager starts the group. ``None`` only for arms prepared into a caller's group.
        :ptype host: LaunchHost | None
        :param group: The launch's group, formed.
        :ptype group: LaunchGroup
        :param formed: What the caller's ``form`` returned beside the group, for ``prepare``.
        :ptype formed: Any
        :param prepare: Prepares every arm into the group — the half that spends.
        :ptype prepare: Callable[[LaunchGroup, Any], Awaitable[list[EvalRun]]]
        :param event: How the group's log lines begin.
        :ptype event: str
        :param ticket: The admission reservation held for the launch; ``None`` when a caller holds it.
        :ptype ticket: AdmissionTicket | None
        :param owns_group: Whether :meth:`start` starts the group, or only prepares into a caller's.
        :ptype owns_group: bool
        :param attach: Records the prepared runs before they start.
        :ptype attach: Callable[[list[EvalRun]], Awaitable[None]] | None
        :param detach: Undoes ``attach`` when the group is abandoned after it landed.
        :ptype detach: Callable[[list[EvalRun]], Awaitable[None]] | None
        """
        self._host = host
        self._group = group
        self._formed = formed
        self._prepare = prepare
        self._event = event
        self._ticket = ticket
        self._owns_group = owns_group
        self._attach = attach
        self._detach = detach
        self._used = False

    @classmethod
    def into(
        cls, group: LaunchGroup, formed: Any, prepare: Callable[[LaunchGroup, Any], Awaitable[list[EvalRun]]]
    ) -> AdmittedLaunch:
        """Arms admitted into a caller's group: :meth:`start` prepares them, and the caller starts the group.

        :param group: The caller's group, which the caller admitted and starts.
        :ptype group: LaunchGroup
        :param formed: What the arms' admission resolved, for ``prepare``.
        :ptype formed: Any
        :param prepare: Prepares the arms into the group.
        :ptype prepare: Callable[[LaunchGroup, Any], Awaitable[list[EvalRun]]]
        :return: The admitted arms.
        :rtype: AdmittedLaunch
        """
        return cls(None, group, formed, prepare, event="", ticket=None, owns_group=False)

    @property
    def group(self) -> LaunchGroup:
        """The launch's group: its id is every run's ``launch_group_id``."""
        return self._group

    def _use(self) -> None:
        if self._used:
            raise RuntimeError(f"launch group {self._group.id} was already started or discarded")
        self._used = True

    async def start(self) -> list[EvalRun]:
        """Prepare every arm — the launchers run, a generating launch's calls are made — and start them together.

        A failure anywhere, a cancel included, abandons the group — every prepared member's clients are released —
        and undoes ``attach`` through ``detach`` if it had landed. The reservation is released however it ends:
        started runs are live tasks that count on their own, and a refused launch's share goes back.

        :return: The started runs (prepared only, into a caller's group), as ``prepare`` returned them.
        :rtype: list[EvalRun]
        :raises RuntimeError: Already started or discarded.
        :raises ValidationFailedError: Any refusal ``prepare`` makes.
        """
        self._use()
        if not self._owns_group or self._host is None:
            return await self._prepare(self._group, self._formed)
        group, job_manager = self._group, self._host.job_manager
        try:
            runs: list[EvalRun] = []
            attached = False
            try:
                runs = await self._prepare(group, self._formed)
                if self._attach is not None:
                    attaching = asyncio.ensure_future(self._attach(runs))
                    cancelled_while_attaching = await wait_through_cancellation(attaching)
                    attaching.result()
                    attached = True
                    if cancelled_while_attaching:
                        raise asyncio.CancelledError
                await job_manager.start_group(group.members, measure_latency=bool(group.measure_latency))
            except BaseException:
                log.warning("%s group=%s abandoned before starting", self._event, group.id)
                await group.abandon()
                if attached and self._detach is not None:
                    # The runs never started, so whatever recorded them must not list them.
                    await self._detach(runs)
                raise
            log.info("%s group=%s started runs=%s", self._event, group.id, [run.id for run in runs])
            return runs
        finally:
            if self._ticket is not None:
                self._ticket.release()

    async def discard(self) -> None:
        """Give the launch up unstarted: its reserved room handed back, whatever its arms built released. Idempotent."""
        if self._used:
            return
        self._used = True
        if self._ticket is not None:
            self._ticket.release()
        await self._group.abandon()


async def admit_group(
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
) -> AdmittedLaunch:
    """Admit one group of runs: refuse what cannot start, reserve its room and form it — preparing and spending nothing.

    The first half of :func:`launch_as_group`, in this order:

    1. Refuse a launch of more runs than one launch may start together. It reads only the run
       count, and it is the more specific refusal, so it comes before admission.
    2. Reserve room for the runs against the host's admission ceiling, or take their share of a
       reservation the caller already holds — before anything is read, so a launch this process
       has no room for costs nothing.
    3. ``form``: the caller's own reads, refusals and pricing, which form the group.

    The rest — ``prepare``, ``attach`` and the start — is :meth:`AdmittedLaunch.start`.

    :param host: The host; its job manager admits the launch.
    :ptype host: LaunchHost
    :param n_runs: How many runs the launch starts, one per arm.
    :ptype n_runs: int
    :param settings: The launch's settings snapshot, read once by the caller when the launch began; its arm
        and admission ceilings are applied here, and the caller's arms are prepared under the same read.
    :ptype settings: LaunchSettings
    :param form: Makes the caller's refusals under the reservation and returns the group, with
        whatever ``prepare`` needs of what it read.
    :ptype form: Callable[[], Awaitable[tuple[LaunchGroup, _Formed]]]
    :param prepare: Prepares every arm into the group and returns the prepared runs.
    :ptype prepare: Callable[[LaunchGroup, _Formed], Awaitable[list[EvalRun]]]
    :param event: How the group's log lines begin, e.g. ``eval.start_run``.
    :ptype event: str
    :param admission: A reservation the caller holds, of which this launch takes ``n_runs``.
    :ptype admission: AdmissionTicket | None
    :param attach: Records the prepared runs before they start.
    :ptype attach: Callable[[list[EvalRun]], Awaitable[None]] | None
    :param detach: Undoes ``attach`` when the group is abandoned after it landed.
    :ptype detach: Callable[[list[EvalRun]], Awaitable[None]] | None
    :return: The admitted launch, holding its reservation until it is started or discarded.
    :rtype: AdmittedLaunch
    :raises ValidationFailedError: ``n_runs`` is more than one launch may start together, or any refusal ``form`` makes.
    :raises AdmissionRefusedError: The runs would pass the host's admission ceiling.
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
    except BaseException:
        ticket.release()
        raise
    return AdmittedLaunch(
        host,
        group,
        formed,
        cast(Callable[[LaunchGroup, Any], Awaitable[list[EvalRun]]], prepare),
        event=event,
        ticket=ticket,
        owns_group=True,
        attach=attach,
        detach=detach,
    )


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

    :func:`admit_group`, then :meth:`AdmittedLaunch.start`: the one sequence every launch that owns its group goes
    through, in this order:

    1. Refuse a launch of more runs than one launch may start together.
    2. Reserve room for the runs against the host's admission ceiling, or take their share of a
       reservation the caller already holds — before anything is read, so a launch this process
       has no room for costs nothing.
    3. ``form``: the caller's own reads and refusals, which form the group.
    4. ``prepare``: every arm prepared into the group (every refusal made, every client built).
    5. ``attach``, when given: record the prepared runs somewhere before they start. Waited for
       through a cancel, because the write cannot be interrupted once a worker has it, and
       ``detach`` must know whether it landed.
    6. Start every member in one concurrency slot.

    **It prices nothing: a ``form`` or ``prepare`` composing its own arms prices them through :func:`price_arms`**
    before calling any launcher, as :func:`start_run`'s does. Under an enforced cap the launch tail refuses an arm
    that did not come through it, so this primitive cannot run an arm nobody priced.

    A failure anywhere in 4-6, a cancel included, abandons the group — every prepared member's
    clients are released — and undoes ``attach`` through ``detach`` if it had landed. The
    reservation is released however the launch ends.

    :param host: The host; its job manager admits the launch.
    :ptype host: LaunchHost
    :param n_runs: How many runs the launch starts, one per arm.
    :ptype n_runs: int
    :param settings: The launch's settings snapshot (see :func:`admit_group`).
    :ptype settings: LaunchSettings
    :param form: Makes the caller's refusals under the reservation and returns the group, with
        whatever ``prepare`` needs of what it read.
    :ptype form: Callable[[], Awaitable[tuple[LaunchGroup, _Formed]]]
    :param prepare: Prepares every arm into the group and returns the prepared runs.
    :ptype prepare: Callable[[LaunchGroup, _Formed], Awaitable[list[EvalRun]]]
    :param event: How the group's log lines begin, e.g. ``eval.start_run``.
    :ptype event: str
    :param admission: A reservation the caller holds, of which this launch takes ``n_runs``.
    :ptype admission: AdmissionTicket | None
    :param attach: Records the prepared runs before they start.
    :ptype attach: Callable[[list[EvalRun]], Awaitable[None]] | None
    :param detach: Undoes ``attach`` when the group is abandoned after it landed.
    :ptype detach: Callable[[list[EvalRun]], Awaitable[None]] | None
    :return: The started runs, as ``prepare`` returned them.
    :rtype: list[EvalRun]
    :raises ValidationFailedError: ``n_runs`` is more than one launch may start together, or any
        refusal ``form`` or ``prepare`` makes.
    :raises AdmissionRefusedError: The runs would pass the host's admission ceiling.
    """
    admitted = await admit_group(
        host,
        n_runs,
        settings=settings,
        form=form,
        prepare=prepare,
        event=event,
        admission=admission,
        attach=attach,
        detach=detach,
    )
    return await admitted.start()


def resolve_judge_pin(request: LaunchRequest, role_default: str, *, candidate_model: str) -> str:
    """The run-level judge pin an arm is scored under: the launch's, or the role default stepped off a candidate.

    A launch that names a judge gets that judge, whatever it is: a choice is recorded, not overridden. One
    that names none inherits the judge role's default — unless that default is one of the launch's candidate
    models (every arm's, :attr:`LaunchGroup.candidate_models`, and this arm's own, which is the kind's default
    for an arm that named none), in which case :attr:`LaunchSettings.judge_alternate_model` scores instead,
    provided it is set and is itself none of the candidates. Otherwise the default stands, and the run's
    surfaces disclose the overlap
    (:func:`~threetears.evals.schema.judge_attribution.judges_sharing_a_candidate_model`). A judged kind
    calls this in its ``plan_arm`` and in its launcher alike — the plan's judges and the launcher's must agree
    — and the launch tail refuses a launcher that kept a candidate's model where an alternate stood ready.

    :param request: The arm.
    :ptype request: LaunchRequest
    :param role_default: The judge role's default model, resolved as the host's clients resolve it.
    :ptype role_default: str
    :param candidate_model: The model the arm runs on — the one it named, or the kind's default.
    :ptype candidate_model: str
    :return: The judge pin, resolved.
    :rtype: str
    """
    if request.judge_model is not None:
        return request.judge_model
    candidates = _candidates_of(request, candidate_model)
    alternate = request.settings.judge_alternate_model
    if role_default in candidates and alternate is not None and alternate not in candidates:
        return alternate
    return role_default


def _judge_origin(request: LaunchRequest, judge_model: str) -> KnownRoleModelOrigin:
    """How an arm's judge was arrived at: named by the launch, its alternate stepped in, or the role default.

    ``alternate`` when the launch named none and the judge is the host's alternate — the pin
    :func:`resolve_judge_pin` steps to off a role default that is one of the launch's candidates (the
    tail has already refused a launcher that kept such a default where the alternate stood ready).

    :param request: The arm.
    :ptype request: LaunchRequest
    :param judge_model: The judge the launcher wired.
    :ptype judge_model: str
    :return: The origin recorded on the run's ``model_role_provenance``.
    :rtype: KnownRoleModelOrigin
    """
    if request.judge_model is not None:
        return "chosen"
    if judge_model == request.settings.judge_alternate_model:
        return "alternate"
    return "inherited"


def _candidates_of(request: LaunchRequest, candidate_model: str) -> frozenset[str]:
    """Every candidate model of ``request``'s launch: each arm's, and this arm's own.

    :param request: The arm.
    :ptype request: LaunchRequest
    :param candidate_model: The model it runs on.
    :ptype candidate_model: str
    :return: The candidates.
    :rtype: frozenset[str]
    """
    return frozenset({*request.launch_group.candidate_models, candidate_model})


def require_candidate_model(request: LaunchRequest, default: str | None) -> str:
    """The model ``request``'s arm runs on — the one it named, or the kind's role default — or the refusal.

    One refusal for the two places that ask: a kind's :attr:`LaunchableKind.plan_arm`, which makes it before
    the arm is priced, and the launch tail, which makes it for a kind that plans nothing. So an operator who
    named no model on a kind with no default hears exactly that, before anything is priced or built.

    :param request: The arm.
    :ptype request: LaunchRequest
    :param default: The kind's role default, or ``None`` for a kind with none.
    :ptype default: str | None
    :return: The model.
    :rtype: str
    :raises ValidationFailedError: The arm named no model and the kind has no default.
    """
    model = request.candidate_model or default
    if model is None:
        raise ValidationFailedError(f"kind {request.kind!r} has no default candidate model", remedy="name one")
    return model


def _refuse_wiring_the_request_contradicts(request: LaunchRequest, wiring: KindWiring) -> str:
    """Check what a kind's launcher resolved against what the launch asked for, and name the candidate model.

    The launch tail stamps the request onto the run; the wiring carries only what the kind resolved.
    Where the two speak to one fact they must agree, or the run would record one thing and run
    another — so each disagreement is refused, before anything is built.

    :param request: The arm as the dispatch resolved it.
    :ptype request: LaunchRequest
    :param wiring: What the kind's launcher resolved for it.
    :ptype wiring: KindWiring
    :return: The arm's candidate model: the one the request named, or the kind's default.
    :rtype: str
    :raises ValidationFailedError: The request named no model and the kind supplies no default.
    :raises ValueError: The wiring contradicts the request.
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
    if request.case_set is not None and (frozen := [case.id for case in wiring.test_cases]) != list(
        request.case_set.test_case_ids
    ):
        raise ValueError(
            f"the launch targets case set {request.case_set.ref.label!r} ({', '.join(request.case_set.test_case_ids)}) "
            f"and kind {kind!r}'s launcher froze {', '.join(frozen) or 'no cases'}; a launcher freezes the set's "
            "cases (LaunchRequest.cases_or)"
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
            f"recorded {counts.variation_model!r} as the model that wrote its cases; hand generate_variations the "
            "request's generation, whose writer the launch built from its variation_model, and whose counts name "
            "the model it called"
        )
    budget = request.generation_budget
    if counts is not None and counts.variation_model is not None and (budget is None or not budget.recorded):
        raise ValueError(
            f"kind {kind!r}'s launcher recorded cases written by {counts.variation_model!r}, and the launch's "
            "generation budget ledgered no call: hand generate_variations generation=request.generation, the calls "
            "the launch admitted against its out-of-run cap, so they are made as admitted and ledgered once they are"
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

    :param host: The host: its settings, its world placement, its job manager and what every run reads.
    :ptype host: LaunchHost
    :param request: The arm as :func:`start_run` handed it to the kind's launcher, unchanged. The run is
        stamped with the request's group's id, and its work function, budget and teardown are
        handed to that group, which then owns them and starts them with the run's siblings.
    :ptype request: LaunchRequest
    :param wiring: What the kind's launcher resolved for the arm.
    :ptype wiring: KindWiring
    :return: The run (status ``pending``), held by the request's group.
    :rtype: EvalRun
    :raises ValidationFailedError: The request named no candidate model and the kind has no default, or
        the assembled run does not validate.
    :raises ValueError: The wiring contradicts the request — a subject other than the one named, a case
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
        role_provenance: dict[str, KnownRoleModelOrigin] = {
            # Whether the launch named the candidate's model or ran it at the kind's own default: the one
            # record of whether its production-replicating cost was measured off the subject's model (#571).
            "candidate": "chosen" if request.candidate_model is not None else "inherited",
        }
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

        refused_checks = dict(request.refused_goal_checks)
        # The deadline every cell of this run runs under: the launch's, held to the host's ceiling — or, for a host
        # that declares none, to the kind's own deadline, which a launch may then only lower — else the kind's.
        kind_cell_timeout_s = wiring.cell_timeout_s if wiring.cell_timeout_s is not None else DEFAULT_CELL_TIMEOUT_S
        _refuse_cell_timeout(
            settings,
            request.cell_timeout_s,
            ceiling=settings.max_cell_timeout_s if settings.max_cell_timeout_s is not None else kind_cell_timeout_s,
        )
        cell_timeout_s = request.cell_timeout_s if request.cell_timeout_s is not None else kind_cell_timeout_s
        cell_timeout_origin: KnownCellTimeoutOrigin = (
            "launch"
            if request.cell_timeout_s is not None
            else "kind"
            if wiring.cell_timeout_s is not None
            else "default"
        )
        try:
            run = EvalRun(
                scope_id=request.scope_id,
                template_id=template.id,
                subject_snapshot=wiring.subject,
                host_payload=dict(wiring.payload),
                candidate_model=candidate_model,
                k_runs=request.k_runs,
                test_case_ids=[case.id for case in test_cases],
                case_set=request.case_set.ref if request.case_set is not None else None,
                variation_counts=wiring.variation_counts,
                candidate_kind=request.kind,
                # A launch is the act of fixing a rig and measuring against it, so every run it starts
                # was commissioned. A witnessed run never comes through here: its writer is the host
                # operation that captured traffic it did not control, and it says so itself.
                apparatus_provenance="commissioned",
                judge_model=judge.model if judge is not None else None,
                model_role_provenance=dict(role_provenance),
                effective_judges=judge.effective_judges if judge is not None else None,
                effective_judges_source="recorded" if judge is not None else None,
                judge_config_ids=judge.config_ids if judge is not None else None,
                judge_config_provenance=dict(judge.config_provenance) if judge is not None else None,
                rubric_scales={dim.name: dim.scale for dim in template.rubric},
                overlays=freeze(request.overlays),
                kind_spec=freeze(request.kind_spec),
                apparatus_settings=dict(request.apparatus_settings),
                declared_margins=dict(request.margins),
                # How the host declared each of its measures to be read, as this run launched under it: a later
                # reader's host may declare them otherwise, and a stored comparison is read on these.
                declared_measures={
                    name: declaration_of(descriptor)
                    for name in host.eval_host.profile.measures.names
                    if (descriptor := host.eval_host.profile.measures.get(name)) is not None
                },
                # The world and the tool bound the template states, frozen as this run launched them:
                # the template is editable, and the runner hands the candidate the template's seed.
                resolved_world_seed=dict(template.world_seed.namespaces),
                resolved_ambient_perturbation_turns=list(template.world_seed.ambient_perturbation_turns),
                resolved_tools_allowed=list(template.tools_allowed) if template.tools_allowed is not None else None,
                # Whether each goal check beats doing nothing, frozen as this run launched it: the controls are
                # editable, and every surface showing a check's pass rate marks one that is not proven.
                # Proven over the checks the cells will grade: a check the current grammar refuses (a template
                # stored before the rule) is graded on no cell and frozen beside the proofs with its reason, and
                # one such check no longer refutes every other check's proof by failing the evaluation.
                goal_check_proofs=goal_check_proofs(template, profile=host.eval_host.profile),
                goal_check_proof_rules=GOAL_CHECK_PROOF_RULES,
                refused_goal_checks=refused_checks,
                cassette_mode=request.cassette_mode,
                cassette_corpus_id=request.cassette_corpus_id,
                # Whether latency is under test, and the width the cells execute at, which follows from it: one
                # at a time when it is; when it is not, the narrowest of what the kind declares it is safe at, the
                # host's width (a ceiling) and the matrix.
                measure_latency=request.measure_latency,
                cell_concurrency=(
                    1
                    if request.measure_latency
                    else max(
                        1,
                        min(
                            settings.max_concurrent_cells,
                            wiring.max_concurrent_cells or settings.max_concurrent_cells,
                            len(test_cases) * request.k_runs,
                        ),
                    )
                ),
                simulator_model=wiring.simulator_model,
                turn_budget_s=wiring.turn_budget_s,
                cell_timeout_s=cell_timeout_s,
                cell_timeout_s_origin=cell_timeout_origin,
                # How each apparatus role was ASKED, stamped from the one value the host's client
                # builder applies to it, and only for a role this kind ran: a resolved model is
                # recorded exactly when the role ran, so it is the predicate here too.
                judge_asked_settings=JUDGE_REQUEST_SETTINGS if judge is not None else None,
                # What a dim with no JudgeConfig is requested at — the judge service's own default, so the
                # stamp and the requests cannot disagree. Part of the measurement context (#633).
                judge_temperature=DEFAULT_JUDGE_TEMPERATURE if judge is not None else None,
                simulator_asked_settings=SIMULATOR_REQUEST_SETTINGS if wiring.simulator_model is not None else None,
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
            raise ValidationFailedError(
                f"invalid run parameters: {e}", remedy="correct the run parameter the error names"
            ) from e

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
            # Each cell's own deadline, the one the run records: every cell opens its own timeout from it.
            cell_timeout_s=cell_timeout_s,
            # Measurement-condition probe (R4): the job manager is the only thing that can
            # see a SECOND job executing beside this one, which is contention that corrupts a
            # latency pool as the run's own concurrent cells do (``max_concurrent_cells``). The run is this job
            # manager's job, so the probe is the run's rather than the host's: a run executed
            # outside a job manager records no execution_mode rather than a guessed one.
            # Runs EXECUTING, never runs queued for a slot: a queued run calls no provider, and
            # counting it stamped a serial baseline `concurrent` whenever another run waited behind it.
            concurrent_eval_jobs_probe=lambda: job_manager.executing_count,
            # One tally rather than two: what the kind's collaborators count and what each
            # cell reports. A kind that calls no metered third party records zero refusals
            # under a ceiling that was in force, which is a different fact from nobody
            # having counted.
            metered_calls=metered_calls,
            # The width the run records, so what it executed at and what it says it did are one value.
            max_concurrent_cells=run.cell_concurrency,
            cell_executor=host.cell_executor,
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

        # Size the job's wall-clock backstop to the matrix. Each cell is capped at
        # options.cell_timeout_s, so a fixed 3600s cap guillotines a serial matrix past ~6
        # cells; scale it with N instead. Sized as if serial whatever the width: an upper
        # bound a concurrent run finishes inside, never a tighter one that assumes its width.
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


def build_judge_service(
    host: EvalHost,
    template: EvalTemplate,
    judge_model: str,
    selection: dict[str, str] | None = None,
    *,
    judged_artifact: JudgedArtifact,
) -> RunJudge:
    """Build the :class:`~threetears.evals.run.judge_service.JudgeService` for a run, with its judge attribution.

    Resolves one :class:`~threetears.evals.schema.models.JudgeConfig` per dim the kind's judge
    scores (:func:`~threetears.evals.schema.models.scored_dim_ids`: the two reserved dual-score
    axes for a conversation, then every declared ``template.rubric`` dim) — so per-result scoring is
    a dict lookup, and a document run's attribution names no conversation axis it is never asked. A dim named in
    ``selection`` resolves to exactly that config; every other dim resolves to the active
    one, which is what a launch that selected nothing gets for all of them. A rubric dim is held to the judge it
    declares (``RubricDim.judge``): one declared ``"configured"`` must resolve to a config, one declared
    ``"builtin"`` must resolve to none, and an undeclared one is refused, so no dim is scored by the built-in
    prompt in a config's place or the reverse.

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

    :param host: The host: where the judge configs are read, the judge clients it builds, and how a
        raised judge call reads.
    :ptype host: EvalHost
    :param template: The run's template, whose rubric names the scored dims.
    :ptype template: EvalTemplate
    :param judge_model: The run's run-level judge fallback, **already resolved** —
        the caller pins it so the model the run records is the model that
        scores.
    :ptype judge_model: str
    :param selection: Optional ``{dim_id: config_id}`` naming the configuration
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
    :ptype selection: dict[str, str] | None
    :param judged_artifact: What the kind's judge reads, which picks the dims it scores. Required:
        the kind's declaration is the only thing that knows, and a default would attribute a
        document run's judges for axes no judge is asked.
    :ptype judged_artifact: JudgedArtifact
    :return: The run's judge: the service, the pin, the selection, the model each scored dim is requested
        from, and the configs — all from ONE config load, so the attribution a run records cannot
        disagree with what scores it.
    :rtype: RunJudge
    :raises ValidationFailedError: ``selection`` names an unscored dim, a config
        that does not load in the template's scope, or a config authored for
        a different dim; or a rubric dim's declared judge is missing or one the
        scope contradicts (:func:`~threetears.evals.run.authoring.refuse_undeclared_judges`).
    :raises ValueError: The host supplies no completion clients, so nothing could score the run, or
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
    """The judges one arm of ``template`` will be scored by, resolved as :func:`build_judge_service` resolves them —
    building nothing.

    What a judged kind's :attr:`LaunchableKind.plan_arm` puts on its :class:`ArmPlan`, so the arm is priced by
    the judges it will be scored by and its launcher is held to them. The same config load and cascade the
    judge service is built from (one config per scored dim — the selection's where it names one, the active
    one elsewhere — and each dim's effective model), with no client built and nothing called: it pays for
    nothing. It makes the selection's refusals too, so a launch naming a config wrongly hears that before its
    arm is priced.

    :param host: The host, where the judge configs are read.
    :ptype host: EvalHost
    :param template: The arm's template, whose rubric names the scored dims.
    :ptype template: EvalTemplate
    :param judge_model: The run-level judge pin, **resolved** — the launch's, or the kind's role default.
    :ptype judge_model: str
    :param selection: The launch's ``judge_config_ids``, or ``None``.
    :ptype selection: Mapping[str, str] | None
    :param judged_artifact: What the kind's judge reads, which picks the dims it scores.
    :ptype judged_artifact: JudgedArtifact
    :return: The planned judge.
    :rtype: PlannedJudge
    :raises ValidationFailedError: ``selection`` names an unscored dim, a config that does not load in the
        template's scope, or one authored for another dim; or a rubric dim's declared judge is missing or one
        the scope contradicts (:func:`~threetears.evals.run.authoring.refuse_undeclared_judges`).
    :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
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

    :param storage: Where the judge configs are read.
    :ptype storage: DefinitionStore
    :param template: The template, whose rubric names the scored dims.
    :ptype template: EvalTemplate
    :param selection: ``{dim_id: config_id}`` as the launch named it, or ``None``.
    :ptype selection: Mapping[str, str] | None
    :param judged_artifact: What the kind's judge reads.
    :ptype judged_artifact: JudgedArtifact
    :return: The scored dim ids, and ``{dim_id: config}`` over the dims that have one.
    :rtype: tuple[list[str], dict[str, JudgeConfig]]
    :raises ValidationFailedError: See :func:`_resolve_selected_judge_configs`; or a rubric dim declares no judge,
        declares ``"configured"`` with no config, or ``"builtin"`` beside an active or selected one
        (:func:`~threetears.evals.run.authoring.refuse_undeclared_judges`).
    :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
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
    # The judge declaration's backstop, over the same load: a template stored before its dimensions declared their
    # judge, or a scope that moved since it was admitted (a configured dim's config archived, a config authored for a
    # dim declared built-in), is refused here, before any spend, rather than scored by a judge nobody declared.
    refuse_undeclared_judges(
        template, lambda dim: configs[dim].id if dim in configs else None, selected=dict(selection or {})
    )
    return dim_ids, configs


def _resolve_selected_judge_configs(
    storage: DefinitionStore, selection: dict[str, str] | None, dim_ids: list[str], scope_id: str
) -> dict[str, JudgeConfig]:
    """Load the judge configs a launch named, refusing every way the naming can be wrong.

    Three refusals, all before any spend, because each one otherwise produces a run
    that looks correct: an unscored dim silently selects nothing, an unloadable id
    silently falls back to the active config, and a config authored for a different
    dim scores that dim with someone else's prompt while the recorded set reads right.

    :param storage: Where the judge configs are read.
    :ptype storage: DefinitionStore
    :param selection: ``{dim_id: config_id}`` as passed to the launch, or ``None``.
    :ptype selection: dict[str, str] | None
    :param dim_ids: This template's scored dims, from
        :func:`~threetears.evals.schema.models.scored_dim_ids`.
    :ptype dim_ids: list[str]
    :param scope_id: The template's scope, where its judge configs live.
    :ptype scope_id: str
    :return: ``{dim_id: config}`` over exactly ``selection``'s keys — empty when nothing
        was named.
    :rtype: dict[str, JudgeConfig]
    :raises ValidationFailedError: A key is not a scored dim of this template, an id
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
                f"judge_config_ids names dim {dim_id!r}, which this template does not score",
                remedy=f"name a config only for a dim the template scores: {', '.join(dim_ids) or 'it scores none'}",
            )
        config = storage.load_judge_config(config_id, scope_id)
        if config is None:
            raise ValidationFailedError(
                f"judge_config_ids names judge config {config_id!r} for dim {dim_id!r}, which does not exist in "
                f"scope {scope_id!r}",
                remedy=f"name a config the scope holds ({FIND_JUDGE_CONFIG})",
            )
        if config.rubric_dim_id != dim_id:
            raise ValidationFailedError(
                f"judge config {config_id!r} was authored for dim {config.rubric_dim_id!r}, so it cannot "
                f"be named under {dim_id!r} — that dim would be scored by another dim's prompt",
                remedy=f"name it under {config.rubric_dim_id!r}, or name a config authored for {dim_id!r}",
            )
        resolved[dim_id] = config
    return resolved


__all__ = [
    "ArmOutcome",
    "ArmPlan",
    "ArmPrice",
    "ArmQuote",
    "ArmVerdict",
    "KindLauncher",
    "KindWiring",
    "LaunchArgument",
    "LaunchHost",
    "TEMPLATE_FIELDS",
    "TemplateField",
    "SubjectResolver",
    "UnknownSubject",
    "LaunchGroup",
    "LaunchPricer",
    "LaunchQuote",
    "LaunchRequest",
    "LaunchableKind",
    "LaunchSettings",
    "PlannedJudge",
    "RunJudge",
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
]
