"""What a run may set before the subject starts — declared by the host, computed over by the engine.

Every agent-eval system in the field declares what the agent may *do* and what it may *see*.
Almost none declares the third space: **what a run may set before the agent starts**. Gymnasium
passes reset options as ``dict[str, Any]``; OpenEnv's ``reset()`` takes no arguments; MCP has no
notion of setting state at all. This module is that third declaration, in the shape R8 proved on
the input registry: the engine ships the record, the algebra and the refusals; the host ships the
dimensions.

**A dimension is serializable, and its operations are named rather than embedded.** A
:class:`WorldDimension` holds strings and a JSON Schema — never a callable — because a handle may
resolve to a local function or to an endpoint across a service boundary, and the engine must not
be able to tell which. :class:`WorldRegistry` holds the resolution table, and
:meth:`WorldRegistry.call` awaits uniformly, so a host whose world is a live device reached by
async RPC uses the same contract and the same conformance kit as one whose world is a dict.

**The engine never interprets a dimension name.** ``display.current_art`` and ``shelf_stock`` are
opaque to everything here; what the engine computes over is the *shape* of the registration —
whether a seed handle exists, whether any surface presents it. That is what
``tests/test_no_host_names_in_shared_contract.py`` holds this package to, and why the
capability algebra below branches on ``seed is None`` and never on a name. The same rule is why a
dimension declares its ``carrier`` rather than letting the engine read a dotted prefix off its
name: a host may spell its dimensions ``inventory.orders`` with tools as their carriers, but that
is one host's convention and reading it would be interpreting a name.

**Two algebras, asked at two moments.** :data:`WorldCapability` is what a host CAN do, computed
from a registration; :data:`WorldPlacement` is what one run DID, computed by
:meth:`WorldRegistry.place` from facts its caller reads off that run. The authoring gate reads the
first, because at authoring time no run exists; a run record carries the second, and nothing
anywhere declares it.

**The registry IS the seeding path.** A host's own runner seeds through these handles rather than
beside them. If ``seed`` named a second path written for the contract's benefit, the conformance
kit would prove a path no run takes — a declaration nobody reads, which is the defect class this
whole contract exists to catch. Two consequences fall out: declaring is not extra work, it is how
seeding is reached; and "this host registers no world" becomes a claim a run record can expose
rather than a shrug.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from threetears.evals.contracts.host.attribution import HostAttributed
from threetears.evals.contracts.host.world_schema import (
    UnsupportedSchemaError,
    json_equal,
    nested_schemas,
    schema_violations,
)

#: Is a dimension's ``read`` computed from the world, or an authored claim about it?
#:
#: A ``labeled`` read cannot prove the world holds the property — "this story contains a
#: contradiction" is a human's or a model's claim, and round-tripping it proves the plumbing and
#: nothing about the world. Conformance renders the two differently for exactly that reason, and
#: a coverage surface that collapsed them would launder a claim into a machine fact.
Evidence = Literal["machine", "labeled"]

#: What kind of condition brings a triggered dimension into being.
#:
#: ``human`` is not a defect: a dimension only a person can instantiate is a representability
#: answer — *not instantiable in an unattended run* — rather than a failed declaration.
TriggerKind = Literal["turn", "event", "human"]

#: What a run can do with a dimension, computed from its registration and nothing else.
#:
#: * ``representable`` — a run can set it and a subject can see it. The precondition a scenario
#:   presumes, instantiated.
#: * ``judge_only`` — a run can set it and no subject sees it. Legitimate (a goal check may read
#:   what the subject never saw) and worth warning about when a template *presumes* it.
#: * ``witnessed`` — a subject sees it and no run controls it. The quadrant no existing design
#:   names: what a television is showing in a room the host can observe and not set. Not an
#:   authoring error: a confound, with a disclosure obligation and a pooling rule.
#:
#: The fourth combination — neither seedable nor perceivable — is refused at registration, so it
#: is not a value here. That is a field, not a dimension.
WorldCapability = Literal["representable", "judge_only", "witnessed"]

#: What a RUN did with a dimension, computed from that run's own record and nothing else.
#:
#: The same two bits as :data:`WorldCapability`, asked one moment later. Capability asks whether a
#: seed handle exists and whether any surface could present the dimension; this asks whether THIS
#: run seeded it and whether THIS subject could perceive it — and both halves are derived from the
#: run, never declared on it. A declared per-run mode would be a claim that can disagree with the
#: run, and it cannot express the normal case: a run that seeds some dimensions and witnesses
#: others.
#:
#: The names are deliberately the capability's own where the meaning is the same, because they mean
#: the same thing one level down and inventing a second vocabulary for it would make two readers of
#: one distinction. Where they differ is the fourth value:
#:
#: * ``representable`` — this run seeded it and this subject perceived it. The precondition the
#:   scenario presumed, instantiated.
#: * ``judge_only`` — this run seeded it and this subject did not perceive it. A goal check reads
#:   what the subject never saw, which is legitimate and worth warning about when a template
#:   *presumes* it.
#: * ``witnessed`` — this subject perceived it and this run did not set it. A confound rather than
#:   an authoring error, carrying a disclosure obligation and a pooling rule.
#: * ``out_of_play`` — neither. **Refused at registration and ordinary here**, which is the whole
#:   reason this is a separate vocabulary: a dimension nothing can ever reach is a field, but a
#:   dimension THIS run neither seeded nor exposed is simply not in play as a precondition for it,
#:   and every run of a host whose world is larger than the subject it built has some.
WorldPlacement = Literal["representable", "judge_only", "witnessed", "out_of_play"]


#: How the engine calls each handle role. A host binds the callable and the engine calls it, so
#: the shape is contract — and a generic caller cannot guess a signature.
#:
#: Checked at registration rather than left to first use, for the reason an unresolvable handle is:
#: a host that binds ``def render(attached)`` instead of ``def render(*, surfaces)`` passes every
#: other refusal, because resolvability and callability say nothing about arity, and then fails as
#: a ``TypeError`` inside a run attributed to whatever the run was doing.
_HANDLE_CALL_SHAPES: Mapping[str, tuple[tuple[Any, ...], Mapping[str, Any]]] = {
    "seed": ((None,), {}),
    "perturb": ((None,), {}),
    "read": ((), {}),
    "fire": ((), {}),
    "subject_view": ((), {"surfaces": ()}),
    "perturb_ambient": ((), {}),
    "coherence": ((None,), {}),
    "settle": ((), {}),
}

#: Bound pairs that can contradict each other within one schema.
_SCHEMA_BOUND_PAIRS: tuple[tuple[str, str], ...] = (
    ("minimum", "maximum"),
    ("minLength", "maxLength"),
    ("minItems", "maxItems"),
)

#: What each JSON Schema type admits, for checking an ``enum`` against a ``type`` declared beside it.
#:
#: ``bool`` is excluded from the numeric types deliberately — it is a subclass of ``int`` in Python
#: and is not an integer in JSON Schema, so admitting it here would let ``{"type": "integer",
#: "enum": [true]}`` pass as coherent.
_SCHEMA_TYPE_MEMBERS: Mapping[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


class WorldRegistrationError(ValueError):
    """A world declaration contradicts what this module promises, raised where it is written."""


def _call_shape_text(role: str) -> str:
    """How a handle role is called, rendered for an error a host has to act on.

    Args:
        role: One of :data:`_HANDLE_CALL_SHAPES`.

    Returns:
        A parenthesised argument list, e.g. ``(value)`` or ``(*, surfaces=...)``.
    """
    args, kwargs = _HANDLE_CALL_SHAPES[role]
    rendered = ["value"] * len(args) + [f"{name}=..." for name in kwargs]
    return f"({', '.join(rendered)})"


def _admits(json_type: str, value: Any) -> bool:
    """Whether one JSON Schema type admits one value.

    Two places a bare ``isinstance`` gets JSON Schema wrong, in opposite directions:

    * ``bool`` is an ``int`` subclass in Python and is not an integer in JSON Schema, so
      ``{"type": "integer", "enum": [true]}`` would read as coherent;
    * a float with no fractional part IS an integer in JSON Schema, so ``{"type": "integer",
      "enum": [1.0]}`` would be refused though it is perfectly legal — a false refusal on a
      correct registration, which is worse than having no check at all.

    Args:
        json_type: A key of :data:`_SCHEMA_TYPE_MEMBERS`.
        value: The enumerated value.

    Returns:
        Whether the type admits it.
    """
    if isinstance(value, bool):
        return json_type == "boolean"
    if json_type == "integer":
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    return isinstance(value, _SCHEMA_TYPE_MEMBERS[json_type])


def _enum_type_defects(schema: Mapping[str, Any], *, path: str) -> list[str]:
    """Name any value the schema enumerates that its own ``type`` excludes.

    A schema may carry both, and a reader assumes they agree. Where they do not, one of the two is
    a typo and nothing downstream can tell which — a generator that reads the enum produces values
    the type forbids, and one that reads the type produces values the enum forbids. Refusing it
    here is the same move as the bound pairs above: incoherence the record shows on its face.

    Args:
        schema: The schema to check.
        path: Where this schema sits, for an error that can be acted on.

    Returns:
        One sentence naming the excluded values, or saying the enum admits nothing; empty when
        they agree, when there is neither ``enum`` nor ``const``, or when the declared type is
        one this table does not describe.
    """
    # `const` is normalised into a one-value enum rather than given a parallel check. The two are
    # the same declaration — here are the values, exhaustively — and every rule below applies to
    # both, so a second code path would be a second place for the next rule to be forgotten. That
    # is not hypothetical: this check closed enum-versus-type, then union types, then empty enums,
    # each as its own instance, and `const` survived all three by having its own branch.
    if "enum" in schema:
        keyword, enumerated = "enum", schema["enum"]
    elif "const" in schema:
        keyword, enumerated = "const", [schema["const"]]
    else:
        return []
    if not isinstance(enumerated, (list, tuple)):
        return []
    if not enumerated:
        return [f"{path} enumerates nothing, so it admits no value"]
    declared = schema.get("type")
    # A UNION type is checked against the union of what its members admit. Reading only a scalar
    # type left ``{"type": ["string", "null"], "enum": [1]}`` through both refusals — the kit
    # resolves an ``enum`` schema before it ever looks at ``type``, so its own union refusal never
    # fires for one either.
    candidates = declared if isinstance(declared, list) else [declared]
    members = [member for member in candidates if isinstance(member, str) and member in _SCHEMA_TYPE_MEMBERS]
    if len(members) != len(candidates):
        return []
    excluded = [value for value in enumerated if not any(_admits(member, value) for member in members)]
    if not excluded:
        return []
    return [
        (
            f"{path} declares type={declared!r} and its {keyword} names "
            f"{', '.join(repr(value) for value in excluded)}, which that type excludes"
        )
    ]


def _schema_defects(schema: Mapping[str, Any], *, path: str = "schema") -> list[str]:
    """Name every way a declared JSON Schema contradicts itself, nested schemas included.

    Self-contradiction is the only ground this refuses a dimension's schema on, wherever the schema is
    written: a bound pair that admits no value, an enum its own type excludes, an empty ``anyOf`` — the
    record wrong on its face, checkable with nothing but the schema in hand. What a conformance kit can
    synthesize from is NOT refused here: a registry that refused schemas on those grounds would be enforcing
    one consumer's limits as a property of the record, and the kit raises its own gap when it meets one.

    **The one place the honoured subset binds at registration is a base-world value**
    (:meth:`WorldRegistry._base_world_defects`): a value has to be checked against its schema there, and a
    schema outside the subset cannot be checked against, so a based dimension's schema must be inside it.
    An unbased dimension's schema is held to self-consistency alone.

    Args:
        schema: The schema to check.
        path: Where this schema sits, for an error that can be acted on.

    Returns:
        One sentence per contradiction; empty when the schema is coherent.
    """
    defects = [
        f"{path} declares {floor}={floor_value!r} above {ceiling}={ceiling_value!r}, which admits no value"
        for floor, ceiling in _SCHEMA_BOUND_PAIRS
        if (floor_value := schema.get(floor)) is not None
        and (ceiling_value := schema.get(ceiling)) is not None
        and floor_value > ceiling_value
    ]
    defects.extend(_enum_type_defects(schema, path=path))
    branches = schema.get("anyOf")
    if isinstance(branches, list) and not branches:
        defects.append(f"{path} declares an empty anyOf, which offers no shape and so admits no value")
    # The positions are :func:`nested_schemas`'s, the same the honoured-subset audit walks, so a contradiction is
    # found wherever a schema can be written. Whatever is not a mapping is skipped: that is a shape the readers
    # refuse, not a contradiction.
    for nested in nested_schemas(schema):
        if isinstance(nested.schema, Mapping):
            defects.extend(_schema_defects(nested.schema, path=nested.at(path)))
    return defects


@dataclass(frozen=True)
class Triggered:
    """State that arrives on a condition rather than at t=0.

    Seeding a triggered dimension *arms* it; the condition fires it. Conformance completes the
    round trip through an optional host fire binding where one exists, and records arming-only
    where none does — never a pass.
    """

    kind: TriggerKind
    """``turn`` | ``event`` | ``human`` — what sort of condition fires it."""

    condition: str
    """The host's word for the condition. Opaque to the engine, and required: a triggered
    dimension that does not say what triggers it cannot be armed by anything."""

    fire: str | None = None
    """OPTIONAL HANDLE naming the host operation that makes the condition happen. Called ``fire()``.

    A host capability rather than a contract obligation: with one, conformance completes the
    round trip (arm, fire, read back); without one it records arming-only, which is never a
    pass. The difference stays visible forever, which is the point — a host that can fire its
    own conditions proves more than one that cannot, and neither renders as the other.

    A ``human`` trigger may not carry one. Code that can fire the condition is proof the
    condition is not human-only, and the registry refuses the contradiction where it is written.
    """


#: When a dimension's value comes into being. ``initial`` is the common case — set before the
#: subject's first turn.
When = Literal["initial"] | Triggered


@dataclass(frozen=True)
class WorldDimension:
    """One state dimension a run may set, may read back, or may only witness.

    Serializable by construction: every operation is a *handle*, an opaque string the registry
    resolves through its binding table. Nothing here holds a callable, so a registration can
    cross a process boundary unchanged and a host whose world lives behind an RPC declares it
    the same way as one whose world is in memory.
    """

    name: str
    """The host's word for this dimension. The engine never interprets or branches on it."""

    schema: Mapping[str, Any]
    """JSON Schema for the value. Validates a seed, generates one, and feeds a generator."""

    matters: str
    """REQUIRED prose: why this is a precondition worth presuming.

    A bare dimension name is a label (R8). Unlike the input registry — where only ``apparatus``
    declarations carry prose, because only they are what a confound scan speaks for — every
    dimension here needs it, since a world dimension has no role axis to exempt anything.
    """

    carrier: str
    """REQUIRED: the host's name for whatever supplies this dimension. Opaque to the engine.

    Perception is per-SUBJECT, not per-host: a subject built without the thing that renders a
    dimension cannot perceive it however the registration reads, and one run of a host can build a
    subject the next one does not. So the run-time algebra needs to know which carrier each
    dimension hangs off, and it must learn it from the registration rather than from the shape of
    a name — a dotted prefix is a host's convention and the engine reading one would be
    interpreting a name, which is the one thing this package may never do.

    Required rather than optional because an absent carrier has no honest default. Treating it as
    "always attached" would report a dimension as perceived by a subject that never held the thing
    presenting it, which is the founding incident's own error; treating it as "never attached"
    would report every dimension out of play. A host with exactly one carrier names it once.
    """

    seed: str | None = None
    """HANDLE naming the host operation that instantiates this dimension. Called ``seed(value)``.

    Absent means **no run can set it**. That is a real and legitimate declaration — a television
    that decides for itself what it is displaying — and it puts the dimension in the ``witnessed``
    quadrant rather than failing it.

    This is the host runner's own seeding path, never a parallel one written for the contract.
    """

    read: str | None = None
    """HANDLE naming the host operation that reads the dimension back. Called ``read()``.

    Absent means nothing can verify a seed took, which is why ``seed`` without ``read`` is
    refused: an instantiation nobody can check is the defect this contract exists to catch,
    declared at birth.
    """

    perturb: str | None = None
    """OPTIONAL HANDLE forcing this dimension's value out of band. Called ``perturb(value)``.

    **Never a seeding path**, and refused outright on a dimension that has a ``seed`` handle: a
    second write path beside the run's own is exactly what this contract refuses, because
    conformance would then prove a path no run takes. This exists for the witnessed quadrant.
    No run can set what is on the television, but a host with a test rig can force it — and with
    that binding the kit proves the subject really does perceive the dimension instead of
    recording the claim unproved. A host without one proves less, visibly and permanently.

    It does not make the dimension seedable. The capability algebra reads ``seed`` alone, so a
    perturbable witnessed dimension stays witnessed — which is the truth about what runs can do.
    """

    evidence: Evidence = "machine"
    """Whether :attr:`read` computes from the world or reports an authored claim about it.

    Describes the read, so it says nothing on a dimension that has none.
    """

    perceived_by: tuple[str, ...] = ()
    """Every subject-facing surface that presents this dimension, named within the registry's
    ``subject_view``.

    Empty means no subject can see it. **Names**, not handles — the registry's one ``subject_view``
    handle does the rendering, and these say which surfaces inside it carry this dimension.
    **Every** surface, because one fact usually reaches a subject several ways — the full value,
    a summary or status line, a count — and an eval builds fresh components, so a surface the seed
    does not reach shows a fresh default beside one it does. Perception A/B holds each named
    surface to answering the dimension, which is how that contradiction is caught.
    """

    when: When = "initial"
    """``initial``, or a :class:`Triggered` condition that brings the value into being."""

    def __post_init__(self) -> None:
        """Reject a non-mapping schema, and detach the mapping from the caller's own object.

        The copy is shallow, so this is not deep immutability — it stops a caller's later
        ``schema["type"] = ...`` from rewriting a live registration, and nothing more. The
        isinstance check is the part that earns its place: a schema that is not a mapping fails
        here, naming the dimension, rather than as a confusing error inside a generator later.

        Raises:
            WorldRegistrationError: The schema is not a mapping, or ``perceived_by`` is not a tuple
                of distinct non-empty surface names.
        """
        if not isinstance(self.schema, Mapping):
            raise WorldRegistrationError(f"{self.name}'s schema is not a mapping — a JSON Schema object is required")
        object.__setattr__(self, "schema", dict(self.schema))
        # A bare string is refused rather than wrapped: iterating one names each of its characters
        # as a surface, which would register without complaint and render nothing.
        if not isinstance(self.perceived_by, tuple) or not all(isinstance(s, str) and s for s in self.perceived_by):
            raise WorldRegistrationError(
                f"{self.name}'s perceived_by must be a tuple of surface names, got {self.perceived_by!r}"
            )
        if len(set(self.perceived_by)) != len(self.perceived_by):
            raise WorldRegistrationError(f"{self.name} names a perceiving surface twice: {self.perceived_by!r}")

    @property
    def seedable(self) -> bool:
        """Whether any run can set this dimension."""
        return self.seed is not None

    @property
    def perceivable(self) -> bool:
        """Whether any subject can be shown this dimension."""
        return bool(self.perceived_by)

    @property
    def write_handle(self) -> str | None:
        """The handle that can put this dimension at a value, or None when nothing can.

        A dimension has at most one: the registry refuses a perturbation handle beside a seed
        handle, because a second write path beside the run's own is the parallel path this
        contract exists to refuse.

        **Not the capability question.** :attr:`seedable` asks whether a RUN can set this, which
        is what a scenario may presume and what the quadrants read. This asks whether ANY code
        here can move it — which a test rig forcing a witnessed dimension can, without that
        making the dimension seedable. Conflating them is how a perturbable television would
        start reporting as controllable.

        Returns:
            The seed handle, else the perturbation handle, else None.
        """
        return self.seed or self.perturb

    @property
    def settable(self) -> bool:
        """Whether anything here can put this dimension at a value. See :attr:`write_handle`."""
        return self.write_handle is not None

    @property
    def capability(self) -> WorldCapability:
        """What a run can do with this dimension — the registration-time quadrant.

        Returns:
            ``representable``, ``judge_only`` or ``witnessed``.

        Raises:
            WorldRegistrationError: The dimension is neither seedable nor perceivable — the
                fourth combination, which a registry refuses. This class is public and
                constructible standalone, so the property refuses to answer for a shape no
                registry would hold rather than returning the nearest quadrant, which would make
                a field silently indistinguishable from witnessed state.
        """
        if self.seedable:
            return "representable" if self.perceivable else "judge_only"
        if not self.perceivable:
            raise WorldRegistrationError(
                f"{self.name} is neither seedable nor perceivable, so it has no quadrant — a field, not a dimension"
            )
        return "witnessed"


def _key_is_the_name(carrier: str, key: str) -> str:
    """The default addressing: a seed key IS the dimension's name, and the namespace is its carrier.

    Args:
        carrier: The seed namespace (unused — the carrier is checked, not composed into the name).
        key: The seed key.

    Returns:
        ``key``.
    """
    del carrier
    return key


class WorldRegistry(HostAttributed):
    """One host's world declarations, its resolution table, and the algebra over them.

    Validated at construction, for the reason the input registry is: two hosts coexist in one
    process and the engine must tell them apart by nothing but their contents, which a
    module-level structure validated at import cannot express.

    Composition is extend-not-edit (:meth:`extend`) — a host adds to a copy of whatever core it
    started from, so one host's vocabulary can never reach another's.
    """

    def __init__(
        self,
        declarations: Iterable[WorldDimension] = (),
        *,
        bindings: Mapping[str, Callable[..., Any]] | None = None,
        subject_view: str | None = None,
        perturb_ambient: str | None = None,
        base_world: Mapping[str, Any] | None = None,
        coherence: str | None = None,
        address: Callable[[str, str], str] | None = None,
        settle: Mapping[str, str] | None = None,
    ) -> None:
        """Validate and store one host's world.

        Args:
            declarations: The dimensions this host declares, in reporting order.
            bindings: The resolution table — ``{handle: callable}``. A callable may be
                synchronous or return an awaitable; :meth:`call` awaits uniformly, so both kinds
                of consumer use one design with no migration between them.
            subject_view: HANDLE rendering what a subject perceives of this world, given the
                surfaces attached for that subject. Required before any dimension may declare
                ``perceived_by``, because perception the kit cannot vary against is an
                unverifiable claim.
            perturb_ambient: OPTIONAL HANDLE moving state this registry declares no dimension
                for. A host capability, not an obligation: with one, conformance can hold every
                declared dimension fixed, jostle the surroundings, and catch a subject
                perceiving state nothing declares — the founding defect, found mechanically.
                Without one that check records unproved, forever and visibly. It may return the
                names of what it moved, which the verdict then carries; an empty answer is a rig
                that moved nothing, and the check records that rather than passing over it.
            base_world: OPTIONAL. The world every conformance check composes over — dimension name
                to value — for a host whose dimensions are coupled the way its production world is:
                one that cannot hold some value of one dimension as stated unless another sits at a
                particular value. Each check starts from it, so a dimension is proved beside siblings
                the host can actually hold it with. Every name must be a declared dimension that
                something here can set at t=0, and every value must conform to its schema.
            coherence: OPTIONAL HANDLE answering whether this host holds a composed world as
                stated. Called ``coherence(world)`` with ``{dimension name: value}``; returns the
                reasons it does not — empty when it does. The kit draws only values the host holds
                beside what the world already holds, so a coupling production really has is not
                reported as two dimensions disturbing each other, and a value production would
                silently transform is never mistaken for one that did not land. It narrows which
                values a check draws and never decides whether a check runs: a handle leaving a
                dimension too few values over the base world fails that dimension's checks, so it
                cannot act as a waiver (``world_conformance._not_held``).
            address: OPTIONAL. How a key in a world seed names a dimension, given ``(carrier, key)``
                — the carrier being the seed namespace the key sits under. Omitted, the key IS the
                dimension's name. A host whose names compose the carrier in (``{"inbox": {"messages":
                ...}}`` seeding ``inbox.messages``) declares its composition here, because the engine
                never reads a dotted prefix off a name. Declared on the registry rather than passed
                per call, so the seed walk, a goal check's ``state.<dimension>`` and a control end
                state all read one answer. Pure and synchronous: the goal language calls it while
                evaluating.
            settle: OPTIONAL. Carrier name → HANDLE the engine awaits once a cell's seed has been
                written through that carrier and before the candidate's first turn. Called
                ``settle()``. For a carrier whose world does work of its own after a write — an
                index rebuilt, a queue drained, a scene recomputed — so the subject's first turn
                meets the world the seed describes rather than one still catching up to it. A
                capability, not an obligation: a carrier with nothing to settle declares none, and
                the engine awaits only the carriers a cell attached. Every key must be a carrier some
                declared dimension names.

        Raises:
            WorldRegistrationError: The declaration set is unsound. Every defect is reported at
                once rather than one per construction, so a host fixing them sees the whole list.
        """
        self._init_attribution()
        self._declarations: tuple[WorldDimension, ...] = tuple(declarations)
        self._bindings: dict[str, Callable[..., Any]] = dict(bindings or {})
        self._subject_view = subject_view
        self._perturb_ambient = perturb_ambient
        self._base_world: dict[str, Any] = dict(base_world or {})
        self._coherence = coherence
        self._address: Callable[[str, str], str] = address or _key_is_the_name
        self._settle: dict[str, str] = dict(settle or {})
        if defects := self._defects():
            raise WorldRegistrationError("world declaration is unsound: " + "; ".join(defects))
        self._by_name: dict[str, WorldDimension] = {declared.name: declared for declared in self._declarations}

    def _defects(self) -> list[str]:
        """Name every way the declaration set contradicts what this module promises.

        Returns:
            One sentence per defect; empty when the declaration set is sound.
        """
        defects: list[str] = []
        seen: set[str] = set()
        for declared in self._declarations:
            name = declared.name
            if not name.strip():
                defects.append(
                    "a dimension declares a blank name — the registry keys by name, so a nameless "
                    "dimension can be neither presumed by a template nor named in a report"
                )
            if name in seen:
                defects.append(f"{name} is declared twice — the registry keys by name, so one would shadow the other")
            seen.add(name)
            if not declared.matters.strip():
                defects.append(f"{name} states no reason it matters — a bare dimension name is a label")
            if not declared.carrier.strip():
                defects.append(
                    f"{name} names no carrier — the run-time algebra asks whether the thing supplying this "
                    "dimension was attached for the subject, and a blank answer would place the dimension "
                    "for every subject alike"
                )
            if not declared.seedable and not declared.perceivable:
                defects.append(
                    f"{name} declares neither a seed handle nor a perceiving surface — "
                    "no run can set it and no subject can see it, which is a field, not a dimension"
                )
            if declared.seedable and declared.read is None:
                defects.append(
                    f"{name} declares a seed handle but no read handle — "
                    "an instantiation nobody can verify took is the defect this contract exists to catch"
                )
            if declared.perceivable and self._subject_view is None:
                defects.append(
                    f"{name} declares it is perceived by {declared.perceived_by!r} but this registry has no "
                    "subject_view to render it — an unverifiable perception claim is worse than none"
                )
            if isinstance(declared.when, Triggered) and not declared.when.condition.strip():
                defects.append(f"{name} is triggered but names no condition — nothing could arm it")
            if declared.perturb is not None and declared.read is None:
                defects.append(
                    f"{name} declares a perturbation handle but no read handle — the seed-without-read "
                    "refusal by the other door: a write path nobody can verify took leaves conformance "
                    "blaming a renderer for a perturbation binding wired to nothing"
                )
            defects.extend(f"{name}'s {sentence}" for sentence in _schema_defects(declared.schema))
            if declared.perturb is not None and declared.seedable:
                defects.append(
                    f"{name} declares a perturbation handle beside its seed handle — a run already sets this "
                    "dimension, and a second write path is the parallel path this contract refuses: "
                    "conformance driven through it would prove a path no run takes"
                )
            if (
                isinstance(declared.when, Triggered)
                and declared.when.kind == "human"
                and declared.when.fire is not None
            ):
                defects.append(
                    f"{name} is triggered by a human and names fire handle {declared.when.fire!r} — code that "
                    "can fire the condition is proof the condition is not human-only"
                )
            if declared.evidence == "labeled" and declared.read is None:
                defects.append(
                    f"{name} declares labeled evidence but no read handle — evidence describes a read, "
                    "so with none it is a claim nothing will ever render"
                )
            defects.extend(self._binding_defects(declared))
        for role, handle in (
            ("subject_view", self._subject_view),
            ("perturb_ambient", self._perturb_ambient),
            ("coherence", self._coherence),
        ):
            if handle is None:
                continue
            if handle not in self._bindings:
                defects.append(
                    f"{role} names handle {handle!r}, which this registry's binding table does not resolve "
                    "— a typo'd handle is a registration bug, not a silent no-op"
                )
                continue
            defects.extend(self._signature_defects(role, handle))
        if self._perturb_ambient is not None:
            if self._subject_view is None:
                defects.append(
                    "perturb_ambient is declared but this registry has no subject_view — the only thing "
                    "ambient perturbation proves is that the subject's view did not move with it, so "
                    "without one it is a capability nothing can ever read"
                )
        defects.extend(self._base_world_defects())
        defects.extend(self._settle_defects())
        defects.extend(
            f"binding {handle!r} is not callable — refusing it here for the same reason an unresolvable "
            "handle is refused: left to first use it is a TypeError inside a run, attributed to whatever "
            "the run was doing"
            for handle, bound in self._bindings.items()
            if not callable(bound)
        )
        return defects

    def _base_world_defects(self) -> list[str]:
        """Refuse a base world no check could start from.

        Returns:
            One sentence per entry naming no declared dimension, one nothing here can set at t=0, or a
            value its dimension's schema refuses; empty when every entry is one a check can seed.
        """
        by_name = {declared.name: declared for declared in self._declarations}
        defects: list[str] = []
        for name, value in self._base_world.items():
            declared = by_name.get(name)
            if declared is None:
                defects.append(f"the base world names {name!r}, which this registry does not declare")
                continue
            if not declared.settable:
                defects.append(
                    f"the base world sets {name}, which nothing here can set — a check cannot start from a "
                    "value it has no way to put there"
                )
            if isinstance(declared.when, Triggered):
                defects.append(
                    f"the base world sets {name}, which is triggered — it arrives on its condition, so it is "
                    "not part of the world a check starts from"
                )
            try:
                violations = schema_violations(declared.schema, value, at=f"the base world's {name}")
            except UnsupportedSchemaError as gap:
                violations = [f"its value cannot be checked: {gap}"]
            defects.extend(violations)
        return defects

    def _settle_defects(self) -> list[str]:
        """Refuse a settle declaration nothing could ever await.

        Returns:
            One sentence per entry naming a carrier no declared dimension names, or a handle the binding
            table cannot resolve or the engine could not call as ``settle()``; empty when every entry is
            one a cell attaching that carrier would await.
        """
        carriers = {declared.carrier for declared in self._declarations}
        defects: list[str] = []
        for carrier, handle in self._settle.items():
            if carrier not in carriers:
                defects.append(
                    f"settle names carrier {carrier!r}, which no declared dimension names — no cell could attach "
                    "it, so its settle handle could never be awaited"
                )
            if handle not in self._bindings:
                defects.append(
                    f"carrier {carrier!r}'s settle handle {handle!r} is not in this registry's binding table"
                )
                continue
            defects.extend(
                f"carrier {carrier!r}'s {sentence}" for sentence in self._signature_defects("settle", handle)
            )
        return defects

    def _signature_defects(self, role: str, handle: str) -> list[str]:
        """Refuse a bound callable the engine could not call in this role's shape.

        Args:
            role: One of :data:`_HANDLE_CALL_SHAPES`.
            handle: The handle naming it, already known to resolve.

        Returns:
            One sentence when the signature cannot take the call, empty otherwise. Empty also when
            the callable has no introspectable signature — a builtin or a C extension is a real
            binding, and refusing what cannot be checked would be a canary crying wolf.
        """
        bound = self._bindings[handle]
        try:
            signature = inspect.signature(bound)
        except TypeError, ValueError:
            return []
        args, kwargs = _HANDLE_CALL_SHAPES[role]
        try:
            signature.bind(*args, **kwargs)
        except TypeError as mismatch:
            return [
                (
                    f"{role} handle {handle!r} is bound to {signature}, which the engine cannot call as "
                    f"{role}{_call_shape_text(role)} — {mismatch}"
                )
            ]
        return []

    def _binding_defects(self, declared: WorldDimension) -> list[str]:
        """Refuse a handle the binding table cannot resolve, at registration rather than first use.

        The registry knows its own resolution table, so an unresolvable handle is checkable the
        moment it is written. Left to first use it becomes a failure inside a run, attributed to
        whatever the run was doing.

        Args:
            declared: The dimension to check.

        Returns:
            One sentence per unresolvable handle.
        """
        handles: list[tuple[str, str | None]] = [
            ("seed", declared.seed),
            ("read", declared.read),
            ("perturb", declared.perturb),
        ]
        if isinstance(declared.when, Triggered):
            handles.append(("fire", declared.when.fire))
        defects: list[str] = []
        for role, handle in handles:
            if handle is None:
                continue
            if handle not in self._bindings:
                defects.append(f"{declared.name}'s {role} handle {handle!r} is not in this registry's binding table")
                continue
            defects.extend(f"{declared.name}'s {sentence}" for sentence in self._signature_defects(role, handle))
        return defects

    @property
    def declarations(self) -> tuple[WorldDimension, ...]:
        """Every dimension this host declared, in registration order."""
        return self._declarations

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared dimension name, in registration order."""
        return tuple(declared.name for declared in self._declarations)

    @property
    def subject_view(self) -> str | None:
        """The handle rendering what a subject perceives of this world, or None."""
        return self._subject_view

    @property
    def perturb_ambient(self) -> str | None:
        """The handle moving state no dimension here declares, or None when the host has none."""
        return self._perturb_ambient

    @property
    def base_world(self) -> Mapping[str, Any]:
        """The world every conformance check composes over, dimension name to value; empty when the host names none.

        A copy, so a caller cannot rewrite the registry's own.
        """
        return dict(self._base_world)

    @property
    def coherence(self) -> str | None:
        """The handle answering whether this host holds a composed world as stated, or None."""
        return self._coherence

    @property
    def settle(self) -> Mapping[str, str]:
        """Carrier name → the handle awaited once a cell's seed has gone through it; a copy.

        Empty for a host none of whose carriers has anything to settle.
        """
        return dict(self._settle)

    def get(self, name: str) -> WorldDimension | None:
        """The declaration for ``name``, or None when this host never declared it."""
        return self._by_name.get(name)

    def address(self, carrier: str, key: str) -> str:
        """The dimension name a world seed's ``key`` under ``carrier`` addresses, by this host's addressing.

        Args:
            carrier: The seed namespace the key sits under.
            key: The key.

        Returns:
            The dimension name. Whether this host declares it is :meth:`get`'s question.
        """
        return self._address(carrier, key)

    def held_at(self, name: str, namespaces: Mapping[str, Any]) -> str | None:
        """The key holding dimension ``name``'s value in a world laid out as a seed is, or None.

        A seed — and the end state a run grows from it — is keyed carrier → key → value, and only
        this host's :meth:`address` says which key names which dimension. Asked forwards over the
        keys actually held, so the engine never inverts a name it may not interpret.

        Args:
            name: A dimension name.
            namespaces: Carrier → (key → value).

        Returns:
            The key under the dimension's carrier that addresses it, or None when the dimension is
            undeclared or the world holds no value for it.
        """
        declared = self._by_name.get(name)
        if declared is None:
            return None
        held = namespaces.get(declared.carrier)
        if not isinstance(held, Mapping):
            return None
        return next((key for key in held if self._address(declared.carrier, key) == name), None)

    def named(self, namespaces: Mapping[str, Any]) -> dict[str, Any]:
        """A world laid out as a seed is (carrier → key → value), keyed by dimension name instead.

        The form the goal language reads ``state.<dimension>`` from: a run's end state is read back
        dimension by dimension, and an end state stated as a seed (a template's seed, which is the
        do-nothing control; a control end state laid over it) is named here through this host's
        :meth:`address`, so both reach a check in one shape.

        Args:
            namespaces: Carrier → (key → value).

        Returns:
            Dimension name → a deep copy of its value, in the order the keys appear.

        Raises:
            ValueError: A carrier's value is not a mapping, a key addresses no declared dimension,
                or a key sits under a carrier other than the one supplying its dimension. Each is a
                value no read handle could ever return, so naming it anyway would hand a check a
                world no run can leave.
        """
        named: dict[str, Any] = {}
        for carrier, held in namespaces.items():
            if not isinstance(held, Mapping):
                raise ValueError(f"{self._host}world state under {carrier!r} is not a mapping of key to value")
            for key, value in held.items():
                name = self._address(carrier, key)
                declared = self._by_name.get(name)
                if declared is None:
                    raise ValueError(f"{self._host}{carrier}.{key} addresses no dimension this world declares")
                if declared.carrier != carrier:
                    raise ValueError(
                        f"{self._host}{carrier}.{key} puts {name!r} under {carrier!r}, but {declared.carrier!r} "
                        "supplies it"
                    )
                named[name] = copy.deepcopy(value)
        return named

    def resolve_path(self, path: str) -> str | None:
        """The dimension a dotted path addresses into, or None when this host declares none.

        A path names a dimension and then addresses inside that dimension's *value* —
        ``inventory.orders.length`` and ``inventory.orders[0].sku`` are both the ``inventory.orders``
        dimension — so resolution takes the longest declared prefix rather than matching the
        whole string. Only the dimension's own schema could speak for what lies below it, and
        this answers the question the registry alone can: whether the vocabulary exists.

        Longest rather than shortest, because a host that declares both a dimension and
        something under it means the more specific one where a path reaches it. The registry
        already refuses two declarations of the same name, so the answer is unambiguous.

        Args:
            path: A dotted path with the language's own root already stripped.

        Returns:
            The declared dimension name, or None when no prefix of the path is declared. None is
            the refusal the authoring gate reads, and it is why an expression naming a dimension
            this host removed cannot resolve to nothing quietly.
        """
        segments = path.split(".")
        for depth in range(len(segments), 0, -1):
            candidate = ".".join(segments[:depth])
            if candidate in self._by_name:
                return candidate
        return None

    def capability(self, name: str) -> WorldCapability | None:
        """What a run can do with ``name``, or None when this host never declared it.

        Args:
            name: A dimension name.

        Returns:
            The registration-time quadrant, or None. None and ``witnessed`` are different
            answers — "we have no such concept" versus "your subject sees this and your
            experiment does not control it" — and collapsing them is the error this contract
            exists to prevent.
        """
        declared = self._by_name.get(name)
        return None if declared is None else declared.capability

    def of_capability(self, capability: WorldCapability) -> tuple[WorldDimension, ...]:
        """Every dimension whose REGISTRATION-time quadrant is ``capability``, in registration order.

        Registration-time, so this answers what a run *could* do rather than what one did — the
        witnessed set here is every dimension no run of this host can ever seed, which is the list a
        disclosure surface names once for the host rather than once per run.

        Args:
            capability: The quadrant to select.

        Returns:
            The matching declarations. Empty is an answer, not an absence: a host every one of
            whose dimensions a run can seed has no witnessed state to disclose.
        """
        return tuple(declared for declared in self._declarations if declared.capability == capability)

    def supplier(self, path: str) -> str | None:
        """The carrier supplying whatever dimension ``path`` addresses, or None when none does.

        The subject-level question, in one read. Host capability and subject availability are two
        different questions asked at two different moments: whether this host declares a dimension
        at all is answerable before any subject exists, and whether THIS subject can reach it is
        not. A subject built without the thing that supplies a dimension cannot be put in a state
        over it however the registration reads, so a caller assembling a run resolves each
        presumed path to its carrier and checks that carrier against what it actually attached.

        Composed from :meth:`resolve_path` and the declaration's ``carrier`` rather than left to
        the caller, because the two-step has a hole in the middle — a path resolving to no
        dimension — and a caller writing it by hand reads the carrier off a ``None``. Naming it
        once also keeps the carrier a thing the registration DECLARES: the alternative every
        caller reaches for is a dotted-prefix read off the name, which is the engine interpreting
        a name, and the one thing this package may never do.

        Args:
            path: A dotted path with the language's own root already stripped, as
                :meth:`resolve_path` takes it.

        Returns:
            The carrier name, or None when no declared dimension covers the path. None is "this
            host has no such vocabulary", which is the authoring gate's refusal rather than an
            assembly one — by the time a run is assembled it has already been made.
        """
        dimension = self.resolve_path(path)
        return None if dimension is None else self._by_name[dimension].carrier

    def place(self, *, seeded: Iterable[str], carriers: Iterable[str]) -> dict[str, WorldPlacement]:
        """What one run did with every declared dimension — the run-time algebra.

        Both inputs are FACTS the caller reads off the run and the subject it built, never modes
        anybody declared. That is the design's own rule: a declared per-run mode is a claim that
        can disagree with the run, and it cannot express the ordinary case of a run that seeds some
        dimensions and witnesses others.

        The two bits are asked independently, which is what lets one dimension land in different
        quadrants across two runs of one host. Seeding is per RUN; perception is per SUBJECT.

        **The carrier gates BOTH bits, not just perception.** A seed handle writes through the
        thing that supplies the dimension, so a subject that never attached that carrier is one
        no seed could have reached — and reporting such a dimension ``judge_only`` because a
        template named it would record an instantiation that did not happen, which is the founding
        incident's own shape. A dimension whose carrier is absent is therefore ``out_of_play``
        whatever the run asked for. The seeding path refuses that configuration where it is
        written; this reports what the run could actually have done.

        A name in ``seeded`` that this registry does not declare is IGNORED rather than refused.
        This is a report over what is declared, and the caller that decides whether an undeclared
        seed is an error is the seeding path itself, which already refuses one where it is written
        — refusing again here would put a second answer on a question that has one.

        **A dimension no run can seed is never placed as though one had.** The seeded branch reads
        :attr:`WorldDimension.seedable` as well as the caller's set, so a registration saying
        ``seed=None`` cannot be overridden by a name appearing in ``seeded``. Without that the two
        algebras could disagree — the registration calling a dimension ``witnessed`` while the run
        record called it ``representable`` — and the record would assert an instantiation that
        never happened, which is the founding incident's own shape. The caller's set is a report of
        what a run ASKED to set; the registration is what says whether asking could work.

        Args:
            seeded: Dimension names this run actually set, however it set them.
            carriers: Carrier names attached for the subject this run measured.

        Returns:
            Dimension name → its placement for this run, one entry per declaration. Every
            declaration appears, because ``out_of_play`` is a finding about this run rather than
            an absence of one.
        """
        was_seeded = set(seeded)
        attached = set(carriers)
        placements: dict[str, WorldPlacement] = {}
        for declared in self._declarations:
            reachable = declared.carrier in attached
            if reachable and declared.seedable and declared.name in was_seeded:
                placements[declared.name] = "representable" if declared.perceivable else "judge_only"
            elif reachable and declared.perceivable:
                placements[declared.name] = "witnessed"
            else:
                placements[declared.name] = "out_of_play"
        return placements

    def extend(
        self,
        declarations: Iterable[WorldDimension],
        *,
        bindings: Mapping[str, Callable[..., Any]] | None = None,
        subject_view: str | None = None,
        perturb_ambient: str | None = None,
        base_world: Mapping[str, Any] | None = None,
        coherence: str | None = None,
        settle: Mapping[str, str] | None = None,
    ) -> WorldRegistry:
        """Return a new registry carrying these declarations after this one's.

        This registry is never mutated — a host adds to a copy, so two hosts registered in one
        process cannot see each other's vocabulary.

        **Append-only on every axis, handles included.** The registry already refuses a duplicate
        *name* because one declaration would shadow the other; a handle rebound to a *different*
        callable is that same shadowing one level down and is far harder to see, because
        re-validation passes — the handle still resolves, just to something else. The damage
        lands in conformance: a round-trip seeds through the shadowing binding and reads through
        it too, so it reports PASSED for a dimension whose real seeding path nothing exercised. A
        false negative in a detector is the failure this contract exists to prevent, and it is why
        the shared binding table was rejected. The shape precedent — ``SweepableRegistry.extend``
        — only appends and has no override path at all.

        **Re-supplying the identical binding is not shadowing, and is permitted**, on the same
        reading that lets a re-supplied ``subject_view`` through: nothing is displaced, so
        nothing can go unexercised. A host composing several carriers over one shared table would
        otherwise have to remember which of them contributed a common handle first.

        Args:
            declarations: The host's own dimensions.
            bindings: Additional handle resolutions. A handle this registry already binds to a
                different callable is refused rather than replaced.
            subject_view: The host's subject-view handle. Refused when this registry already has
                a different one, for the same reason.
            perturb_ambient: The host's ambient-perturbation handle, under the same rule.
            base_world: Further base-world entries. One naming a dimension this registry already
                bases at a different value is refused, for the same reason: every check already
                proved was proved over the first.
            coherence: The host's coherence handle, under the subject view's rule.
            settle: Further carriers' settle handles. One naming a carrier this registry already settles
                through a different handle is refused, for the binding table's reason.

        Returns:
            A new validated registry.

        Raises:
            WorldRegistrationError: The combined set is unsound, or the addition would displace a
                binding, the subject view, the ambient-perturbation handle, the coherence handle, a
                carrier's settle handle or a base-world value.
        """
        added = dict(bindings or {})
        if rebound := sorted(handle for handle, bound in added.items() if self._bindings.get(handle, bound) != bound):
            raise WorldRegistrationError(
                "world declaration is unsound: extending would rebind "
                + ", ".join(repr(handle) for handle in rebound)
                + " — composition appends, and a silently replaced binding makes conformance pass over "
                "a path no run takes"
            )
        added_base = dict(base_world or {})
        if rebased := sorted(
            name
            for name, value in added_base.items()
            if name in self._base_world and not json_equal(self._base_world[name], value)
        ):
            raise WorldRegistrationError(
                "world declaration is unsound: extending would change the base world's "
                + ", ".join(repr(name) for name in rebased)
                + " — every check already proved composed over the first value"
            )
        added_settle = dict(settle or {})
        if resettled := sorted(
            carrier
            for carrier, handle in added_settle.items()
            if carrier in self._settle and self._settle[carrier] != handle
        ):
            raise WorldRegistrationError(
                "world declaration is unsound: extending would change the settle handle of "
                + ", ".join(repr(carrier) for carrier in resettled)
                + " — composition appends, and a replaced settle handle leaves the first carrier's own work unawaited"
            )
        for role, adding, held in (
            ("subject_view", subject_view, self._subject_view),
            ("perturb_ambient", perturb_ambient, self._perturb_ambient),
            ("coherence", coherence, self._coherence),
        ):
            if adding is not None and held is not None and adding != held:
                raise WorldRegistrationError(
                    f"world declaration is unsound: extending would replace {role} {held!r} with {adding!r} "
                    "— every claim inherited from the first was proven against it"
                )
        return WorldRegistry(
            (*self._declarations, *declarations),
            bindings={**self._bindings, **added},
            subject_view=self._subject_view if subject_view is None else subject_view,
            perturb_ambient=self._perturb_ambient if perturb_ambient is None else perturb_ambient,
            base_world={**self._base_world, **added_base},
            coherence=self._coherence if coherence is None else coherence,
            settle={**self._settle, **added_settle},
            # Inherited, never replaced: every inherited dimension is addressed by it already.
            address=self._address,
        )

    @property
    def bindings(self) -> Mapping[str, Callable[..., Any]]:
        """A copy of this registry's resolution table, for composing a derived registry.

        **Not a call path.** Everything that invokes a handle goes through :meth:`call`, because
        the whole design rests on a caller being unable to tell a local function from an async
        RPC — and handing back a raw callable is exactly what makes them distinguishable. A
        caller who invokes what this returns has opted out of that guarantee and owns the
        awaiting themselves.
        """
        return dict(self._bindings)

    def _resolve(self, handle: str) -> Callable[..., Any]:
        """The callable a handle names.

        Private, so :meth:`call` is the only way to reach a binding. A public resolver would be a
        second call path, and a caller writing ``registry.resolve(dim.read)()`` against a host
        whose read is asynchronous gets a coroutine object back — which compares unequal to every
        seeded value and reports as the host's world being wrong.

        Args:
            handle: A handle written on a registration or on ``subject_view``.

        Returns:
            The bound callable, which may be synchronous or return an awaitable.

        Raises:
            WorldRegistrationError: Nothing binds the handle. Registration refuses this for every
                handle it can see, so reaching it here means a caller supplied one from outside.
        """
        bound = self._bindings.get(handle)
        if bound is None:
            raise WorldRegistrationError(f"{self._host}no binding resolves handle {handle!r}")
        return bound

    async def call(self, handle: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke what a handle names, awaiting the result when the binding is asynchronous.

        The one call path, so a host whose world is a live device reached by RPC and one whose
        world is an in-memory dict are indistinguishable to every caller — including the
        conformance kit, which therefore needs no second set of checks for async hosts.

        A binding that raises propagates **with its own type intact** — an eval host raises
        ``ApparatusError`` from exactly these paths and a runner branches on it, so wrapping
        would erase the distinction between a broken rig and a world under test that failed
        legitimately. What the failure gains is a note naming the handle, because without one it
        surfaces attributed to whatever the run happened to be doing.

        Args:
            handle: A handle written on a registration or on ``subject_view``.
            *args: Passed through to the binding.
            **kwargs: Passed through to the binding.

        Returns:
            The binding's result, awaited if it produced an awaitable.

        Raises:
            WorldRegistrationError: Nothing binds the handle.
        """
        bound = self._resolve(handle)
        try:
            result = bound(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        except (
            Exception
        ) as error:  # prawduct:allow prawduct/broad-except -- annotate and re-raise; the type is load-bearing
            error.add_note(f"raised by world handle {handle!r}")
            raise


__all__ = [
    "Evidence",
    "TriggerKind",
    "Triggered",
    "When",
    "WorldCapability",
    "WorldDimension",
    "WorldPlacement",
    "WorldRegistrationError",
    "WorldRegistry",
]
