"""One named surface a consuming product presents to the eval engine.

A :class:`HostProfile` is everything the engine needs from a host and the only thing it is
allowed to know about one. Two profiles registered in one process must be indistinguishable to
the engine except by their contents — that is the property the whole extraction rests on, and it
is what the toy-host fixture exists to prove on every run.

**Nothing here switches on host identity.** ``host_id`` is opaque and is never branched on; a
surface that would need to know *which* host it is holding is a surface that has host coupling it
has not admitted to. Anything that distinguishes one profile from another is self-describing
metadata on the registries, consumed generically.

**Evaluability is derived from the profile, not declared beside it** (R10). The sweepables
registry *is* the controllability map, the measures registry *is* the observability map, and the
world registry *is* the representability map — so none of the three can drift from the thing it
describes. **One declared exception, and it is the case derivation cannot reach:** each kind
contract's :attr:`~threetears.evals.kernel.host.kinds.KindContract.seats` names the seats in the rig
that kind's runs fill — pinned roles, or apparatus dimensions by name — and every apparatus dimension a
kind does not seat is one its runs do not HAVE, which no registry can express: the declaration it would
have to be derived from belongs to the core, and ``SweepableRegistry.extend`` only appends. It is
declared per KIND rather than per host because one host grades one kind with a judge and another with
code, and a host-wide answer is false for one of them. It is an allow-list — the seats a kind HAS, never
the dimensions it lacks — so a dimension added to the rig later is inapplicable to every kind that has
not claimed it, and cannot turn such a kind's runs ``undecided``. Drift is prevented by
:meth:`HostProfile.omits_apparatus` reading the runs' own values instead, so the claim is checked
against what happened rather than trusted. **Seats are then narrowed per RUN** where the run's own record
decides it (:meth:`HostProfile.seats`): a run naming no judge was not judged — the runner refuses to
execute a judged run that names none — so its rig had no judge seat whatever its kind declares, and a
cohort mixing judged and code-only runs of one kind reads the code-only runs' judge as
:data:`UNSEATED_LEVEL`, a level, rather than as ``undecided``. Recorded at ``boundary-patterns.md`` § Eval host-profile
registry seam. **They are reached to different depths, and which is which is recorded here** because a
reader of this paragraph would otherwise supply all three and have no way to tell what each one
does. Sweepables are read by the variant key and the coverage lens on every generation. The world
registry is reached end to end: it seeds every run, its run-time algebra is derived onto the run
record and hashed into the measurement context, this class refuses a knob registered in it and in
the sweepables registry without a declaration, and a TEMPLATE presuming a dimension no
registration supplies is refused where it is written — the authoring gate, which asks this class
:meth:`presumable` for a precondition and :meth:`addressable` for a goal check. The measures
registry has two reaches that ASK it a question. The first is the campaign declaration, which reads
a measure's descriptor before letting a campaign bar restate the better-direction the host already
declared for it. The second, and much the wider, is
:func:`~threetears.evals.kernel.metrics.describe_measure`: the engine's closed core is consulted first and
this registry second, so a measure whose vocabulary is a host's tool's — the outcome buckets of an
async delivery — is described by the host that has it and reads as unclassified to a host that does
not. That is what lets the core stop enumerating one product's tool measures without every read
surface silently losing them. (The registry is also read at every construction of this class, by
``bars.validate_against(self.measures)`` in ``__post_init__``; that is the registry checking itself,
not a caller consulting it.) Both callers read the
registry directly, through :meth:`~threetears.evals.kernel.host.measures.MeasureRegistry.get`, and this
class deliberately offers **no observability predicate** to ask before that lookup.

**Why observability is derived and unpredicated where controllability and representability have
methods**. Three facts, any one of which would be enough on its own:

* **A gate cannot use one.** Every gate that wants to know whether a measure is declared also
  wants its descriptor, and ``get`` returns both in one lookup. A membership predicate asked first
  cannot change a verdict, and no test can catch its removal.
* **A gate over the REGISTRY alone could not be right.** A measure name space is half closed and
  half open: the host declares a catalogue, and a template MINTS a measure every time it names a
  rubric dimension or writes a goal-state check — names ``threetears.evals.kernel.metrics`` deliberately does
  not enumerate and resolves by construction instead. A gate refusing every name the registry does
  not hold would refuse the open half wholesale. The campaign declaration gate
  (``refuse_an_undeclarable_design``) is right where such a gate would not be because it reads the
  TEMPLATE that mints the open half alongside the registry — and it adds no predicate here: it
  reads descriptors, which is the rule this list defends.
* **The reporting surface such a predicate would serve cannot say anything.** Returning a
  :class:`Coverage` — a state plus a REASON — suits a surface that RENDERS coverage rather than
  gating on it. Over a half-open name space that surface answers ``covered`` for every input it
  can name and ``uncovered`` for every input it cannot classify: it can never be wrong and never
  be informative. R10 asks for a gate rather than a page, and this is what a page here would be.

So the registry *is* the observability map, which is the property R10 asks for, and membership is
read where a caller needs the record. **Do not add a predicate ahead of the gate that would call
it** — that ordering is what produced one with no caller.

Fidelity is deliberately absent: whether the eval path constructs what production
constructs cannot be inferred from structure, and a map entry asserting it is exactly the claim
that was false in the incident that produced this rule. It is proven by a shared construction
path with a test that both callers reach it, and it gains no entry here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, get_args

from threetears.evals.kernel.host.bars import BarRegistry
from threetears.evals.kernel.host.kinds import KindContract
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.style import StyleProfile
from threetears.evals.kernel.host.sweepables import CANDIDATE_KIND_LEVER, CANDIDATE_MODEL_LEVER, SweepableRegistry
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.host.world import WorldRegistry
from threetears.observe import get_logger

log = get_logger(__name__)

#: The level an apparatus dimension reads at for a run whose rig has no such seat
#: (:meth:`HostProfile.apparatus_level`), where the dimension is kept because another run in the same
#: comparison does have it. A recorded level, never ``undecided``: the run's own record says the seat was
#: empty, and two such runs agree; against a run that filled the seat it reads as the difference it is.
UNSEATED_LEVEL = "(none — this run's rig has no such seat)"

#: The apparatus inputs a run's own record empties when it names no ``judge_model``: the run was not judged,
#: so it had no judge pin, no judge temperature, no judge request settings, no per-dim judge attribution and no
#: judge-config seat.
#: A BLANK in one of these on an unjudged run reads as no such seat (:data:`UNSEATED_LEVEL`), never
#: ``undecided``; a value the run did record is still its level, because :meth:`HostProfile.apparatus_level`
#: substitutes only for a blank. That matters for ``judge_config_ids``, which a code grader or a person can
#: fill on a run no model judged — the configs it recorded show, and only its empty reading is unseated.
#: Not the whole judge role: a host's grader nominated into the role is filled by whoever grades, so its
#: own value says whether the run had one.
_UNJUDGED_RUN_HAS_NO: frozenset[str] = frozenset(
    {"judge_model", "judge_temperature", "judge_request_settings", "judge_dim_divergence", "judge_config_ids"}
)

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalRun

#: Produce a run's level of each of the HOST'S OWN levers — its share of the variant key's pre-image.
#: A run is one arm, so every observation in it shares this map. Host code the engine calls and never
#: inspects.
#:
#: **What it does not return is the engine's.** The candidate model (:data:`CANDIDATE_MODEL_LEVER`),
#: the candidate kind (:data:`CANDIDATE_KIND_LEVER`) and every lever a kind contract on
#: :attr:`HostProfile.kinds` derives are resolved by the engine for every run, and a reader that
#: returns one of them is refused — two writers of one coordinate are two places to disagree.
#:
#: **Why this is not the sweepables registry's readers.** A registered reader projects one input
#: into whatever JSON-safe shape the bisection compares; this returns the typed levels a key is
#: digested from, in one map the host resolves as a whole. What keeps the two from drifting is not a
#: second list but a check: every key the composed map holds must name a ``lever`` in the same host's
#: registry, every fixed lever must be resolved, and
#: :func:`~threetears.evals.kernel.identity.derive_variant_identity` refuses a map that does not.
VariantLeverReader = Callable[["EvalRun"], "dict[str, SweepableValue]"]

#: ``(tool, action) -> that action's parameter JSON Schema``, or None for an action the host does not
#: describe. The schema is the one the subject is shown, so the host restates nothing: the goal-check
#: gate reads which parameters are closed values (``enum``/``const``/``pattern``) and treats every
#: other string as text the model wrote.
ActionParameterReader = Callable[[str, str], "Mapping[str, Any] | None"]

#: ``tool -> the names of its actions``, or None for a tool whose actions the host cannot list. The
#: goal-check gate reads it to refuse a check naming an action that does not exist; None is "cannot
#: say", never "has none", so a tool the host cannot describe is left unchecked rather than refused.
#: A tool the host does not have at all is answered with an EMPTY set, not None — the host can say
#: for certain that it offers no action — so a misspelled tool is refused by every gate reading this.
ToolActionReader = Callable[[str], "frozenset[str] | None"]


#: The answer to "does eval reach this precondition", per axis. Three states, not two: a
#: consumer with no simulated world has representability **inapplicable**, and rendering that as
#: "unsupported" would be the very error the coverage model warns against — *"this area is
#: unevaluable" is a conclusion the map is not entitled to draw*.
CoverageState = Literal["covered", "uncovered", "inapplicable"]


#: One segment of a dotted payload path: what a storage projection will interpolate.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ProfileRegistrationError(ValueError):
    """Two of a host's registries contradict each other, raised where both are in hand.

    Distinct from each registry's own error type because the defect belongs to neither
    registry alone: a name is sound in the sweepables registry, sound in the world registry,
    and unsound only in a profile that holds both.
    """


@dataclass(frozen=True)
class Coverage:
    """Whether eval reaches one precondition on one axis, and why when it does not."""

    state: CoverageState
    """``covered`` | ``uncovered`` | ``inapplicable``."""

    reason: str = ""
    """Required for anything but ``covered`` — the dimension that is missing, never the area."""


@dataclass(frozen=True)
class HostProfile:
    """Everything the engine knows about one consuming product.

    A frozen dataclass rather than a Pydantic model: this holds live reader callables and is a
    runtime registration, never a stored or wire type.
    """

    host_id: str
    """Opaque. The engine never interprets or branches on it — it is for logs and error text."""

    host_sweepables: SweepableRegistry
    """What this host itself declares it sweeps and what holds its measurements: the shared core
    (:data:`~threetears.evals.kernel.host.sweepables.SHARED_CORE`) extended with the host's own
    levers, apparatus and labels.

    Not what the engine reads: :attr:`sweepables` is this plus every kind contract's levers, which
    the profile derives from :attr:`kinds` so a kind is named once. A kind's lever registered here
    as well is refused.
    """

    measures: MeasureRegistry
    """What this host can see. The observability map."""

    bars: BarRegistry = field(default_factory=BarRegistry)
    """The incumbent standards per behavior. A host with none registers an empty registry."""

    style: StyleProfile = field(default_factory=StyleProfile)
    """The bounded presentation contract. No field of it is free text."""

    world: WorldRegistry | None = None
    """What a run may set before the subject starts, and what it may only witness.

    **Optional on purpose, and ``None`` is not an empty registry.** ``None`` says this host
    instantiates no world at all — the shape a consumer evaluating production traffic has, whose
    representability is ``inapplicable`` rather than ``uncovered``. An empty *registry* says this
    host has a world and seeds nothing in it, which is a claim a run record can expose. Collapsing
    the two would lose the only distinction that tells an unevaluable area from an unbuilt one.
    """

    caveat_kinds: frozenset[str] = frozenset()
    """Caveat kinds this host declares, BEYOND the four the engine owns.

    The engine's four (``apparatus``/``sampling``/``instrument``/``scope``) are always available;
    this is the host's extension of them, for the same reason R8 gives the lever vocabulary to the
    host. A closed engine-owned set on a host-facing field forces a domain with a legitimate fifth
    kind to jam it into ``scope``, and the field then stops meaning anything — which is the one
    thing a required classification exists to prevent. Empty is the normal case: the four cover
    what has been seen, and they were derived from one host's caveats, so they are offered rather
    than asserted to be complete.
    """

    variant_levers: VariantLeverReader | None = None
    """How this host resolves a run's level of each of its OWN fixed levers. See :data:`VariantLeverReader`.

    ``None`` for a host that declares no fixed lever of its own: the engine resolves the candidate
    model, the candidate kind and every kind contract's levers itself, so such a host's variant key
    needs nothing from it. A host that declares a lever of its own and wires no reader is refused at
    the first identity derived — a lever nobody resolves would drop out of the key and merge two
    variants.
    """

    observed_model_levers: Mapping[str, str] = field(default_factory=dict)
    """Levers whose INHERITED value is recoverable from what a run observably did.

    ``{lever name in the REGISTRY's vocabulary: the usage role whose ``model`` records its value}``.
    A key must be a lever a campaign could declare it swept — a fixed lever, or a member an open
    family recognises — and a value must be a usage role (``RoleUsage.role``); anything else is refused
    at registration: a misspelled key or role would resolve against nothing and leave the lever reading
    ``unknown`` with no error anywhere.
    A launch that did not name the lever as an overlay still ran at *some* value, and for a
    model-valued lever the role that spent the tokens is the record of which. The analysis
    bundle's effective-configuration lens reads this to distinguish ``inherited`` from
    ``unknown``; a lever absent from it, and from the run's overlays, simply does not apply to
    that run.

    Empty is the normal case. It has to be host-declared because the lever names are the host's
    tool vocabulary: the engine hardcoding one would attribute a second consumer's inner agent to
    the first consumer's tool, silently, in the module a paid generator reads. The engine's own
    entry — the candidate model, a declared coordinate of every observation — is not here and is
    not the host's to move; a declaration colliding with it is refused.
    """

    action_parameters: ActionParameterReader | None = None
    """How a goal check learns which of a call's recorded parameters are closed values.

    ``None`` for a host that describes no action parameters, and then every text comparison over a
    ``calls()`` parameter is refused at authoring: a call parameter is text the model wrote until its
    schema says otherwise (:func:`~threetears.evals.kernel.dsl.call_parameter_matches`). Presence and length
    tests need no schema and are always accepted.
    """

    tool_actions: ToolActionReader | None = None
    """How a goal check learns which actions a tool has, so a misspelled one is refused at authoring.

    ``None`` for a host that lists no tool's actions, and then no action name is checked: a check
    naming one that does not exist evaluates False on every trial
    (:func:`~threetears.evals.kernel.dsl.undefined_call_references`). A wired reader answers a tool
    the host does not have with an empty set, never None, so a misspelled TOOL is refused too.
    """

    kinds: tuple[KindContract, ...] = ()
    """What each candidate kind's runs carry beyond the engine's own fields: its overlays and its spec.

    One :class:`~threetears.evals.kernel.host.kinds.KindContract` per kind that declares any, keyed by
    its ``kind``. A kind with none is one a launch may turn nothing on. **Named here and nowhere else**:
    the profile adds every lever a contract derives to :attr:`sweepables`, and the engine resolves
    each run's level of them into its variant key, so a knob a launch can turn is one every analysis
    sees without the host registering it a second time. Two contracts whose lever prefixes overlap
    are refused — one would overwrite the other's levels.
    """

    listing_elisions: frozenset[str] = frozenset()
    """Paths inside ``EvalRun.host_payload`` that a read LISTING many runs leaves out.

    Dotted, relative to the payload (``"subject.history"``). A bulk read hydrates
    every run in a scope, so a heavy value the host freezes into each run's payload is paid
    once per run per listing; a host names here what no listing consumer of ITS payload reads,
    and the storage read drops it before it crosses the wire. Host-declared because the payload
    is the host's: the engine never names a key inside it.

    Every run a listing returns carries what was left out
    (:attr:`~threetears.evals.schema.models.EvalRun.elided_payload_paths`), so the host's own payload
    readers can refuse to rebuild from an incomplete payload rather than rebuild a wrong one. A
    read of ONE run (``get_run``, execution, comparison, rejudge) never elides anything.

    Empty is the normal case: a host with a small payload lists it whole.
    """

    release_label: str | None = None
    """The ``label`` sweepable whose value names the BUILD of the product that ran — an app version.

    What lets a campaign's time axis be the builds it spanned rather than the days it ran on: two runs
    carrying different values of it are two points in time, ordered by when each value first ran
    (:class:`~threetears.evals.kernel.surface.TimeAxis`). Host-declared because a label is the host's
    word and the engine cannot tell a build from any other identifier — a label naming a document batch
    is no more a time than a model name is.

    ``None`` for a host that labels no build, and then a time axis falls back to the days the runs started
    on. A name that is not a registered ``label`` is refused at registration.
    """

    analysis_writer_models: tuple[str, ...] = ()
    """The model ids this host allows to write a campaign's analysis. Empty: any model.

    An allow-list, not a deny-list: once a host opts in, a model nobody has measured writing the strict
    analysis contract is refused rather than admitted by default. ``analysis_generate`` (and
    :func:`~threetears.evals.analysis.generator.generate_analysis`) check the requested model against it
    before the first provider request, so a writer known unable to produce the contract costs nothing,
    not a billed call and its billed repair. A reporter run measuring writers is exempt: it exists to
    find out which models belong here. Ids are compared exactly, as the host's client is asked for them.
    """

    sweepables: SweepableRegistry = field(init=False, repr=False, compare=False)
    """Every input this host's runs carry: :attr:`host_sweepables` plus each kind contract's levers.

    The controllability map, and what every identity, gate and lens reads. Derived at construction
    from the two fields that name its parts, so it cannot disagree with either.
    """

    def __post_init__(self) -> None:
        """Name this host to its registries, then cross-check them against each other.

        Every registry here validates itself at construction. What none of them can see is a
        claim one makes *about another* — a bar naming a measure the host never declared, or
        restating that measure's better-direction as the opposite. This is the only place both
        are in hand.

        **It is also the only place a registry can learn whose it is.** A registry is built before
        any profile exists, so its own refusals — a bar looser than its incumbent, a measure this
        host does not declare, a handle no binding resolves — name the offending item and not the
        host that offered it. That costs nothing while one process serves one product and becomes
        unactionable the moment a second host runs, which is what the extraction is for. Binding
        happens first so the cross-checks below are named too.

        Raises:
            BarRegistrationError: A bar contradicts the measure registry.
            ProfileRegistrationError: A name is registered as both a sweepable and a world
                dimension, an ``observed_model_levers`` entry claims a lever name the engine
                reserves or names no lever this host declares, a lever's ``acts_on`` names no numeric
                measure the engine or this host declares, a kind contract's seats name
                neither a pinned role nor an apparatus dimension, or leave out a dimension whose
                blank is a real level, a kind is contracted twice or two contracts' lever prefixes overlap, a kind contract's lever is registered in
                :attr:`host_sweepables` as well, the registry lacks a lever the engine resolves
                for every run, or ``release_label`` names no registered ``label``.
        """
        self._refuse_overlapping_kind_contracts()
        self._refuse_a_kind_lever_registered_by_hand()
        object.__setattr__(
            self,
            "sweepables",
            self.host_sweepables.extend(declared for contract in self.kinds for declared in contract.sweepables),
        )
        self._refuse_a_registry_without_the_engines_levers()
        for registry in (self.sweepables, self.measures, self.bars, self.world):
            if registry is not None:
                registry.bind_host(self.host_id)
        self.bars.validate_against(self.measures)
        self._refuse_a_name_in_both_registries()
        self._refuse_an_engine_reserved_lever()
        self._refuse_an_undeclared_observed_lever()
        self._refuse_an_unknown_mechanism_measure()
        self._refuse_an_unsound_seat()
        self._refuse_a_malformed_listing_elision()
        self._refuse_a_release_label_that_is_not_a_label()

    def kind_contract(self, kind: str) -> KindContract:
        """The contract this host declares for ``kind`` — a contract declaring nothing when it declares none.

        A kind with no contract is one a launch may turn nothing on and a template may state nothing
        for, which is exactly what an empty contract refuses; so every caller asks one contract, and
        a refusal reads the same whether the host declared the kind's models or not.

        Args:
            kind: A kind's name, as a template's ``candidate_kind`` spells it.

        Returns:
            The declared contract, or an empty one.
        """
        return next((contract for contract in self.kinds if contract.kind == kind), None) or KindContract(kind)

    def analysis_writer_refusal(self, model: str) -> str | None:
        """Why ``model`` may not write an analysis for this host, or ``None`` when it may.

        Args:
            model: The writer model id, as requested or as the host's client resolved its default.

        Returns:
            The refusal, naming the allowed models; ``None`` when the host declares no list or lists ``model``.
        """
        if not self.analysis_writer_models or model in self.analysis_writer_models:
            return None
        return (
            f"model {model!r} is not an analysis writer host {self.host_id!r} allows; it allows "
            f"{', '.join(repr(allowed) for allowed in self.analysis_writer_models)} "
            "(HostProfile.analysis_writer_models). Refused before any provider request, so nothing was spent"
        )

    def engine_levels(self, run: EvalRun) -> dict[str, SweepableValue]:
        """The levels the engine resolves for ``run`` itself: its model, its kind and every kind contract's levers.

        The part of the variant key's pre-image no host writes. A host's own levers come from
        :attr:`variant_levers` and are composed beside these by
        :func:`~threetears.evals.kernel.identity.derive_variant_identity`.

        Args:
            run: The run, of any kind.

        Returns:
            Lever name -> level. A run of a kind other than a contract's sits at that contract's
            "not a run of this kind" level on each of its levers.
        """
        levels = {
            CANDIDATE_MODEL_LEVER: SweepableValue.of(run.candidate_model, display=run.candidate_model),
            CANDIDATE_KIND_LEVER: SweepableValue.of(run.candidate_kind, display=run.candidate_kind),
        }
        for contract in self.kinds:
            levels.update(contract.levels(run))
        return levels

    def _refuse_overlapping_kind_contracts(self) -> None:
        """Refuse two contracts for one kind, and two contracts whose levers could share a name.

        Lever names are ``<prefix>.<field>`` (and ``<prefix>.<field>.<key>`` for an open family's
        entries), so two prefixes that are equal, or one of which is a dotted prefix of the other,
        can name one lever twice — and the later contract's level, or its "not a run of this kind"
        level, would overwrite the earlier's in the variant map, collapsing arms that differ on it.

        Raises:
            ProfileRegistrationError: A kind is contracted twice, or two contracts' prefixes overlap.
        """
        names = [contract.kind for contract in self.kinds]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares more than one contract for kind(s) {', '.join(repeated)}"
            )
        overlapping = sorted(
            f"{first.kind!r} ({first.lever_prefix}) and {second.kind!r} ({second.lever_prefix})"
            for index, first in enumerate(self.kinds)
            for second in self.kinds[index + 1 :]
            if _prefixes_overlap(first.lever_prefix, second.lever_prefix)
        )
        if overlapping:
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares kind contracts whose lever prefixes overlap: {'; '.join(overlapping)} "
                "— their levers could share a name, and one kind's levels would overwrite the other's; give each "
                "kind a prefix of its own"
            )

    def _refuse_a_kind_lever_registered_by_hand(self) -> None:
        """Refuse a kind contract's lever that :attr:`host_sweepables` declares as well.

        The profile registers a contract's levers itself, so a second declaration of one is either
        the same lever named twice or a different input that happens to share its name; neither is
        expressible.

        Raises:
            ProfileRegistrationError: A contract's lever name is already declared by the host.
        """
        if twice := sorted(
            name for contract in self.kinds for name in contract.lever_names if self.host_sweepables.get(name)
        ):
            raise ProfileRegistrationError(
                f"host {self.host_id!r} registers {', '.join(twice)} in its own sweepables, but a kind contract on "
                "the profile derives them — name the kind once, on kinds, and drop its levers from the registry"
            )

    def _refuse_a_registry_without_the_engines_levers(self) -> None:
        """Refuse a registry that does not declare the levers the engine resolves for every run.

        The engine places the candidate model and the candidate kind in every run's variant key, so
        a registry that does not declare them as levers would refuse every key at the first run.
        Both come with :data:`~threetears.evals.kernel.host.sweepables.SHARED_CORE`.

        Raises:
            ProfileRegistrationError: Either lever is missing, or declared under another role.
        """
        missing = [
            name
            for name in (CANDIDATE_MODEL_LEVER, CANDIDATE_KIND_LEVER)
            if (declared := self.sweepables.get(name)) is None or declared.role != "lever"
        ]
        if missing:
            raise ProfileRegistrationError(
                f"host {self.host_id!r}'s sweepables do not declare {', '.join(missing)} as levers, which the engine "
                "resolves for every run — extend SHARED_CORE rather than building a registry without it"
            )

    def _refuse_a_malformed_listing_elision(self) -> None:
        """Refuse a listing elision that is not a dotted path of plain identifiers.

        The path reaches a storage projection, which refuses anything else at query time; refused
        here instead, so a host with a typo fails at startup rather than on its first listing.

        Raises:
            ProfileRegistrationError: A path with an empty or non-identifier segment.
        """
        if bad := sorted(
            p for p in self.listing_elisions if not all(_IDENTIFIER.fullmatch(seg) for seg in p.split("."))
        ):
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares listing_elisions that are not dotted identifier paths: "
                f"{', '.join(repr(p) for p in bad)}"
            )

    def _refuse_a_release_label_that_is_not_a_label(self) -> None:
        """Refuse a ``release_label`` naming anything but a registered ``label`` sweepable.

        A misspelled name would read nothing on every run, and the time axis would fall back to dates with
        no error anywhere; a lever or an apparatus input named here would place runs in time by what they
        swept or what measured them, which is a comparison drawn as a timeline.

        Raises:
            ProfileRegistrationError: The name is undeclared, or declared with another role.
        """
        if self.release_label is None:
            return
        declared = self.sweepables.get(self.release_label)
        if declared is None or declared.role != "label":
            what = "is not declared" if declared is None else f"is declared as a {declared.role}"
            raise ProfileRegistrationError(
                f"host {self.host_id!r} names release_label {self.release_label!r}, which {what} — a release "
                "label is a registered `label` sweepable naming the build that ran"
            )

    def _refuse_an_engine_reserved_lever(self) -> None:
        """Refuse a host recovery rule for a lever name the engine already owns.

        :data:`CANDIDATE_MODEL_LEVER` is a declared coordinate of every observation and the
        engine resolves it from the record itself. A host entry for the same name would not
        override that — the engine's own rule is applied alongside — so the two would silently
        disagree about which usage role establishes it, on the surface that reports a run's
        effective configuration. Refused where both name spaces are in hand.

        Raises:
            ProfileRegistrationError: The host declared a recovery rule for a reserved name.
        """
        if reserved := sorted(set(self.observed_model_levers) & {CANDIDATE_MODEL_LEVER}):
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares observed_model_levers for {', '.join(reserved)}, which the "
                "engine reserves and resolves from the observation itself — the two rules would disagree "
                "about one lever with nothing reporting it"
            )

    def _refuse_a_name_in_both_registries(self) -> None:
        """Refuse a knob registered as both a sweepable and a world dimension.

        A campaign that varies the world is running a different experiment from one that varies a
        lever, and two registries describing one knob are how the two answers come to disagree
        invisibly. There is no declaration that admits the overlap: no surface would record that a
        campaign swept stimulus, so a switch admitting it would only silence the refusal. A host
        whose knob is honestly both renames one of the two registrations.

        **Name-keyed, and it has to be.** The obvious-looking mechanism — resolve both registries
        through one binding table and watch for a collision — cannot work: a sweepable's reader
        answers a post-hoc question from the run record ("what did this run sweep") and a world
        handle answers a live one ("what is this dimension's value now"), so one knob registered
        in both places has two genuinely different callables and no resolution table could ever
        see them meet. It would report clean over the exact case it exists to catch, which is the
        worse of the two failures a detector can have.

        Names are the right key rather than a spelling-agreement trap, because both sides here are
        one host's own code over a name set that is already the compatibility surface — the same
        reason ``derive_variant_identity`` already refuses a variant map naming an unregistered
        lever.

        Raises:
            ProfileRegistrationError: A name appears in both registries.
        """
        if self.world is None:
            return
        if overlap := sorted(set(self.sweepables.names) & set(self.world.names)):
            raise ProfileRegistrationError(
                f"host profile {self.host_id!r} is unsound: registered as both a sweepable and a world "
                f"dimension: {', '.join(overlap)} — a campaign that varies the world is running a different "
                "experiment from one that varies a lever, and one knob in both registries reports an "
                "experimental k over evidence that varied the stimulus. Rename one of the two registrations"
            )

    def _refuse_an_undeclared_observed_lever(self) -> None:
        """Refuse an ``observed_model_levers`` key that names no lever this host declares.

        The key is read by the bundle's effective-configuration lens, which recovers a lever's
        inherited value from the usage role named beside it. A misspelled key recovers nothing and
        refuses nothing: the lever it meant reads ``unknown`` in every bundle, and the entry sits there
        looking like a live declaration. Admitted exactly when a campaign could declare the key as its
        axis (:meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.refuse_as_axis`) — a
        fixed lever, or a member an open family recognises.

        The value is held to the same rule: it names the usage role whose rows record the lever's model,
        and usage roles are a closed set (``RoleUsage.role``), so a misspelled role recovers nothing in
        exactly the same silent way.

        Raises:
            ProfileRegistrationError: A key names no declarable lever, with the remedy the registry prints;
                or a value names no usage role.
        """
        # Imported here: the models module imports from this package, so a module-level import would cycle.
        from threetears.evals.schema.models import UsageRole

        roles = get_args(UsageRole)
        if misnamed := {name: role for name, role in sorted(self.observed_model_levers.items()) if role not in roles}:
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares observed_model_levers recovering "
                + ", ".join(f"{name} from role {role!r}" for name, role in misnamed.items())
                + f", which is no usage role (roles: {', '.join(roles)}) — the lens would recover nothing and the "
                "lever would read unknown everywhere"
            )
        unknown = {
            name: reason
            for name in sorted(self.observed_model_levers)
            if name != CANDIDATE_MODEL_LEVER and (reason := self.sweepables.refuse_as_axis(name)) is not None
        }
        if unknown:
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares observed_model_levers for {', '.join(unknown)}, which name no lever "
                "this host declares — the lens would recover nothing and the lever would read unknown everywhere: "
                + "; ".join(unknown.values())
            )

    def _refuse_an_unknown_mechanism_measure(self) -> None:
        """Refuse a lever's ``acts_on`` unless it names a numeric measure each result carries as one value.

        Checked here because the name is a measure and the declaration is a sweepable, and this is the
        only place both registries are in hand. A misspelled mechanism would be observed on no result, and
        so would a name recorded per row or only per run. Every bundle would then report the lever
        ``unchecked`` with nothing to say the declaration was the cause — a declaration that reads as live
        and checks nothing, while the reason it gives blames the data.

        Raises:
            ProfileRegistrationError: An ``acts_on`` names no declared measure, a non-numeric one, or one
                no result carries as a single value.
        """
        # Imported here: both modules import the models module, which imports this package.
        from threetears.evals.kernel.covariates import REASONING_RATIO_KEY
        from threetears.evals.kernel.declaration import mechanism_measure_names
        from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS

        readable = mechanism_measure_names(self.measures)
        defects: list[str] = []
        # Fixed levers' acts_on and open families' named members (member_acts_on, #585) alike.
        for lever, measure in self.sweepables.mechanisms:
            descriptor = METRIC_DESCRIPTORS.get(measure) or self.measures.get(measure)
            if descriptor is None:
                defects.append(
                    f"{lever} acts on {measure!r}, which is no measure or covariate the engine's "
                    "catalogue or this host's measure registry declares"
                )
            elif descriptor.data_type != "numeric":
                defects.append(
                    f"{lever} acts on {measure!r}, a {descriptor.data_type} measure — the check "
                    "compares levels' values, which only a numeric measure has"
                )
            elif measure not in readable:
                defects.append(
                    f"{lever} acts on {measure!r}, which no result carries as one value — it is "
                    "recorded per row (per role, per delivery) or only over a whole run, so the check would read "
                    "nothing and blame the data. Declare a measure each result carries once instead: the "
                    f"engine's covariate {REASONING_RATIO_KEY!r} for how much a candidate reasoned, or a host "
                    "measure the kind reports per result (the calls one case used against a cap, say)"
                )
        if defects:
            raise ProfileRegistrationError(
                f"host {self.host_id!r} declares a mechanism that cannot be checked: " + "; ".join(defects)
            )

    def _refuse_an_unsound_seat(self) -> None:
        """Refuse a kind contract's seats that cannot mean what they say.

        Checked here because this is the only place the contracts and the registry are both in hand.
        Every defect is reported at once:

        * **An entry naming neither a pinned role nor an apparatus dimension.** Nothing would read it,
          and the kind would not fill the seat it meant, so that seat's dimensions would drop out of
          its runs' confound scans on a typo. A lever or a label is refused the same way: only
          apparatus is scanned for confounds, so seating one says nothing.
        * **An apparatus dimension left unseated that does not carry ``indeterminate_when_blank``.**
          :meth:`omits_apparatus` decides "did a run record this" through
          :meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.is_indeterminate`, which
          answers False for every value when the flag is off — a blank there is a real level. Leaving
          such a dimension unseated would never omit it AND would log a contradiction on every read
          about a value nobody set: a silent no-op wearing a warning. Seat it.

        Raises:
            ProfileRegistrationError: Any defect above, with every instance named.
        """
        roles = {role.name for role in self.sweepables.roles}
        defects: list[str] = []
        for contract in self.kinds:
            if contract.seats is None:
                continue
            for seat in sorted(contract.seats - roles):
                declared = self.sweepables.get(seat)
                if declared is None or declared.role != "apparatus":
                    what = "nothing this host declares" if declared is None else f"a {declared.role}"
                    defects.append(
                        f"kind {contract.kind!r} seats {seat!r}, which is {what} — a seat names a pinned role "
                        f"({', '.join(sorted(roles)) or 'none'}) or an apparatus dimension"
                    )
            seated = self._seated(contract.kind)
            defects.extend(
                f"kind {contract.kind!r} leaves {declared.name!r} unseated, but it does not carry "
                "indeterminate_when_blank — a blank there is a real recorded level, so it would never be omitted "
                "and every read would log a contradiction about a value nobody set; seat it"
                for declared in self.sweepables.declarations
                if declared.role == "apparatus"
                and seated is not None
                and declared.name not in seated
                and not declared.indeterminate_when_blank
            )
        if defects:
            raise ProfileRegistrationError(f"host profile {self.host_id!r} is unsound: " + "; ".join(defects))

    def _seated(self, kind: str) -> frozenset[str] | None:
        """The apparatus dimensions ``kind``'s runs have, each seated role expanded to its pins — None for all."""
        seats = self.kind_contract(kind).seats
        if seats is None:
            return None
        pins = {role.name: role.pins for role in self.sweepables.roles}
        return frozenset(name for seat in seats for name in pins.get(seat, (seat,)))

    def seats(self, run: EvalRun, dimension: str) -> bool:
        """Whether this run's rig had a seat for apparatus ``dimension``.

        The run's kind's seats (:attr:`~threetears.evals.kernel.host.kinds.KindContract.seats`, a pinned
        role seating its pins), narrowed by what the run's own record states: a run whose ``judge_model``
        is None was not judged — the runner refuses to execute a judged run that names none — so the judge
        inputs read off the run's judge — its pin, its request settings, its per-dim attribution, the judge
        configurations its results were scored with — are not seats of its rig, whatever its kind declares.
        A value the run recorded in one of them still reads as itself (:meth:`apparatus_level` substitutes
        only for a blank). Per run, because one kind runs
        judged and code-only templates alike, and a per-kind answer is false for one of them.

        Args:
            run: The run.
            dimension: The declared apparatus input name.

        Returns:
            Whether the run's rig had that seat.
        """
        if run.judge_model is None and dimension in _UNJUDGED_RUN_HAS_NO:
            return False
        return self._kind_seats(run.candidate_kind, dimension)

    def _kind_seats(self, kind: str, dimension: str) -> bool:
        """Whether ``kind``'s contract seats ``dimension`` — the declaration alone, before any run narrows it."""
        seated = self._seated(kind)
        return seated is None or dimension in seated

    def apparatus_level(self, run: EvalRun, dimension: str, value: Any) -> Any:
        """The level a run's value of an apparatus dimension reads at, once the dimension is kept for a comparison.

        A recorded value is its own level. A blank one is :data:`UNSEATED_LEVEL` when the run's rig had
        no such seat (:meth:`seats`) — the record says the seat was empty, which is a level and not an
        unknown — and stays the blank it is, ``undecided`` where its declaration says so, when the run did
        have the seat and recorded nothing in it.

        Args:
            run: The run.
            dimension: The declared apparatus input name.
            value: What the run's reader returned for it.

        Returns:
            ``value``, or :data:`UNSEATED_LEVEL`.
        """
        if not self.seats(run, dimension) and self.sweepables.is_indeterminate(dimension, value):
            return UNSEATED_LEVEL
        return value

    def omits_apparatus(self, dimension: str, observed: Iterable[tuple[EvalRun, Any]]) -> bool:
        """Whether a reporting surface should leave ``dimension`` out entirely, given what ran.

        The single authority for the omission. A dimension is omitted only when **no** run among them
        seats it (:meth:`seats`: its kind's seats, narrowed by its own record) **and** no run recorded a
        level for it. Two things decide it, and neither alone may:

        * **The runs present.** A cohort answers per run: one with a judged run keeps the judge axes —
          that run's unrecoverable judge still reads ``undecided``, and a code-only run beside it reads
          :data:`UNSEATED_LEVEL` (:meth:`apparatus_level`) — while a cohort of code-graded runs alone
          omits them, whether their kind seats a judge or not. A host-wide or kind-wide answer is false
          for one of the two.
        * **The values.** A kind's seats say "these runs have no such thing"; the runs are what say
          whether that is true. A non-blank value wins over the declaration: the dimension is
          reported, and — where the value contradicts the KIND's declaration — the contradiction is
          logged. Reported rather than raised because this runs inside assembly of an analysis an
          operator asked for, and a rig disagreement is exactly the thing they need to SEE. A level on
          a seat only the run's own record narrowed away (judge configs a code grader recorded on an
          unjudged run) contradicts no declaration, so it is reported without the warning.

        Args:
            dimension: The declared apparatus input name.
            observed: ``(run, value)`` for every run under consideration. Pass them all: one run that
                seats the dimension keeps it, and one run recording a level refutes the claim, so a
                caller that passes only the first would omit on the strength of the run that agreed.

        Returns:
            True when no kind present seats the dimension and nothing contradicts it, which is
            when a surface should behave as though the host never had it.

        Raises:
            ValueError: No runs were passed. Answering without them is the unvalidated omission this
                method replaces, and it is available to a caller that simply forgets the argument — so
                it is refused rather than defaulted.
        """
        observations = list(observed)
        if not observations:
            raise ValueError(
                f"omits_apparatus({dimension!r}) was called with no runs — the check IS the kinds and values that "
                "ran, and answering without them is the unvalidated omission this method exists to replace."
            )
        if any(self.seats(run, dimension) for run, _ in observations):
            return False
        recorded = [
            (run, value) for run, value in observations if not self.sweepables.is_indeterminate(dimension, value)
        ]
        if not recorded:
            return True
        # Only a level recorded where the KIND declares no seat contradicts a declaration. A run whose own
        # record narrowed the seat away (an unjudged run's judge configs) and that recorded a level anyway
        # is the narrowing's documented exception — a code grader's configs — so it is reported, unlogged.
        if contradicted := [
            (run, value) for run, value in recorded if not self._kind_seats(run.candidate_kind, dimension)
        ]:
            run, value = contradicted[0]
            log.warning(
                "host %r: run %s (kind %r) has no seat for apparatus dimension %r, but it recorded %r — "
                "reporting it rather than omitting it, because a recorded level is evidence the seat declaration "
                "is wrong and dropping it would hide a real apparatus difference",
                self.host_id,
                run.id,
                run.candidate_kind,
                dimension,
                value,
            )
        return False

    def controllable(self, dimension: str) -> Coverage:
        """Whether a run can DELIBERATELY vary ``dimension``, derived from the sweepables registry.

        **Registration is not enough, and the role is what settles it.** A ``lever`` is a knob a
        campaign sweeps; an ``apparatus`` input is the measuring rig, which is not supposed to
        move and is a confound when it does; a ``label`` identifies rather than determines. Only a
        lever is something an experiment varies on purpose, and the engine already reads it that
        way — ``derive_variant_identity`` builds the variant key from levers alone. So an axis
        declared on a non-lever can never become an arm: every run carrying it resolves to the
        same variant, and the analysis can only ever report a design gap no amount of data closes.

        **Lever is necessary, not sufficient, and the gap is deliberate.** A lever carrying
        :attr:`~threetears.evals.kernel.host.sweepables.Sweepable.no_own_coordinate` contributes no
        coordinate of its OWN — its identity flows through whatever it resolves to. Sweeping it
        does separate arms, just under the coordinate of that resolution, which is why this gate
        accepts it. The role check is the whole of what is enforced here; a finding about such an
        axis still has no coordinate of its own to cite.

        **An open family is decided the other way round, and the two are easily confused.** A
        family's own name is refused — the container identifies no knob, and the rule spells
        sweeping a family as sweeping its MEMBERS — while a member the family's own
        :attr:`~threetears.evals.kernel.host.sweepables.Sweepable.owns_member` recognises is admitted even
        though it carries no declaration. So the earlier argument for accepting a container,
        *"refusing it would refuse the sanctioned way to bake off tool parameters"*, no longer
        holds: that way is spelled as the members now, and they are what this admits.

        Reporting an unsweepable axis ``covered`` is :meth:`representable`'s founding incident in
        the other registry — a declaration accepted for state no run controls — and the remedies
        differ per refusal, which is why the ``uncovered`` reasons do: an unregistered dimension
        needs somebody to declare it, a non-lever one needs the campaign redesigned to hold it
        fixed instead of sweeping it, and a container needs the member named.

        **Caught here because here is where it is still free.** This gate runs at campaign
        authoring, before any run is launched; downstream every consumer filters by role correctly,
        so an accepted-but-unsweepable axis surfaces only after the runs are bought.

        **Decided by the registry, not here.** The verdict and its reason both come from
        :meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.refuse_as_axis`, so this method
        cannot come to a different answer from the message its caller prints. Resolving through
        ``sweepables.get`` here instead is what let the gate accept a container and refuse a
        member while advertising a list that held neither.

        Args:
            dimension: The input a campaign wants to sweep.

        Returns:
            ``covered`` when the host's registry admits the name as an axis; ``uncovered``
            carrying the registry's reason otherwise. The registry cannot drift from itself
            because it is the same set of declarations the engine reads to bisect and to build
            variant keys.
        """
        refusal = self.sweepables.refuse_as_axis(dimension)
        return Coverage("covered") if refusal is None else Coverage("uncovered", refusal)

    def representable(self, dimension: str) -> Coverage:
        """Whether a run can instantiate the precondition ``dimension``, derived from the registry.

        **Seedability is the question, not registration.** A dimension the host registered as
        ``witnessed`` — perceived by the subject, controlled by no run — is real, disclosed, and
        still not something a scenario may presume it has *set*. Reporting it ``covered`` would
        be the founding incident's own mistake with a declaration wrapped around it. The two
        ``uncovered`` reasons therefore differ, because the remedies differ: an unregistered
        dimension needs somebody to declare it, a witnessed one needs the scenario rewritten.

        Args:
            dimension: The state dimension a scenario presumes.

        Returns:
            ``inapplicable`` when this host instantiates no world at all — a different answer
            from ``uncovered``, and the one a consumer evaluating production traffic deserves;
            ``covered`` when a run can seed the dimension; ``uncovered`` naming the dimension
            otherwise. The reason names the **dimension**, never the area: "this area is
            unevaluable" is a conclusion this map is not entitled to draw, and an author told so
            would abandon a probe that another mechanism could have answered.
        """
        if self.world is None:
            return Coverage("inapplicable", "this host instantiates no simulated world")
        declared = self.world.get(dimension)
        if declared is None:
            return Coverage("uncovered", f"{dimension} is not a dimension this host's world declares")
        if not declared.seedable:
            return Coverage(
                "uncovered",
                f"{dimension} is declared but no run can seed it — the subject perceives it and the "
                "experiment does not control it, so a scenario cannot presume it was set",
            )
        return Coverage("covered")

    def addressable(self, path: str) -> Coverage:
        """Whether ``path`` names world state this host declares at all.

        The vocabulary question, and the weaker of the two an authored expression can be asked:
        it says a declared dimension covers the path, and says nothing about whether a run could
        set it. That is the whole question for a *postcondition* — a goal check reads whatever the
        world ended up holding, and reading back a dimension no run controls is an ordinary and
        correct thing for one to do.

        Asked at authoring because nothing else asks it. A goal check reading a path nothing
        declares resolves to ``Missing`` at evaluation, so the check is never established, and the
        subject is scored down for a typo that reports as a failed check. The same holds for a
        declared dimension with no ``read`` handle: the end state is read through ``read`` alone,
        so no cell's end state ever holds it and a check over it can never be established.

        Args:
            path: A dotted path into the world, with the language's own root already stripped.

        Returns:
            ``covered`` when a declared dimension covers the path and declares a ``read`` handle;
            ``uncovered`` naming the path otherwise — including on a host that instantiates no world
            at all, and for a dimension nothing reads back. The goal language
            roots ``state`` at declared dimensions only, so on such a host a state path names
            nothing a run could ever read, and a check over it is a typo or a check written for
            another host, not an inapplicable question.
        """
        if self.world is None:
            return Coverage(
                "uncovered",
                f"{path} reads world state, and this host instantiates no simulated world — a state path "
                "names a declared world dimension, and this host declares none",
            )
        dimension = self.world.resolve_path(path)
        if dimension is None:
            return Coverage("uncovered", f"{path} addresses no dimension this host's world declares")
        declared = self.world.get(dimension)
        if declared is not None and declared.read is None:
            return Coverage(
                "uncovered",
                f"{path} addresses {dimension}, which declares no read handle — no cell's end state ever holds "
                "it, so a check over it is never established",
            )
        return Coverage("covered")

    def presumable(self, path: str) -> Coverage:
        """Whether a scenario may presume the world state at ``path`` was SET before it started.

        The strong form of :meth:`addressable`, and the one a *precondition* is asked. Declared is
        not enough: a dimension the host registered as ``witnessed`` is real, disclosed, and still
        not something a scenario may presume it put in place. Presuming one is the founding
        incident itself — a probe written for a state nobody could seed, running against whatever
        the world happened to hold, and scoring the subject on a situation it was never placed in.

        Resolves the path to its dimension first, so ``jobs.queue.length`` is asked about
        ``jobs.queue``: a path addresses INSIDE a dimension's value, and only the dimension is
        registered.

        Args:
            path: A dotted path into the world, with the language's own root already stripped.

        Returns:
            ``inapplicable`` when this host instantiates no world at all — a scenario on such a
            host presumes nothing this map can speak for; ``covered`` when a run can seed the
            dimension the path addresses; ``uncovered`` naming the path and the remedy otherwise.
            The two ``uncovered`` reasons differ because the remedies do, exactly as
            :meth:`representable`'s do: an unknown path needs the dimension declared or the
            expression corrected, a witnessed one needs the scenario rewritten.
        """
        if self.world is None:
            return Coverage("inapplicable", "this host instantiates no simulated world")
        dimension = self.world.resolve_path(path)
        if dimension is None:
            return Coverage("uncovered", f"{path} addresses no dimension this host's world declares")
        supplied = self.representable(dimension)
        if supplied.state == "covered" or dimension == path:
            return supplied
        # The path addresses INSIDE the dimension's value, so the reason — written about the
        # dimension — has to say which dimension the path landed on, or an author reading it
        # goes looking for a registration under the name they wrote.
        return Coverage(supplied.state, f"{path} addresses a dimension that cannot be presumed: {supplied.reason}")


def _prefixes_overlap(first: str, second: str) -> bool:
    """Whether two lever prefixes could name one lever: equal, or one a dotted prefix of the other."""
    return first == second or first.startswith(f"{second}.") or second.startswith(f"{first}.")


__all__ = [
    "CANDIDATE_KIND_LEVER",
    "CANDIDATE_MODEL_LEVER",
    "Coverage",
    "CoverageState",
    "HostProfile",
    "ProfileRegistrationError",
    "UNSEATED_LEVEL",
]
