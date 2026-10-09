"""One cell's live handle on the host's world: seed it, settle it, move it, and read back what it became.

The host declares its world once, on its profile (:class:`~threetears.evals.contracts.host.world.WorldRegistry`).
A cell uses it through one of these, which the runner builds per cell and hands to the kind's
``prepare`` as ``world``. Everything a cell does to its world goes through here, so the runner can
record it whatever way the cell ends — the same reason a cell's spend goes through its sink:

1. :meth:`WorldSession.seed` — the case's seed, through the engine's seed walk
   (:func:`~threetears.evals.contracts.host.world_seed.check_seed`) and the dimensions' own ``seed``
   handles, for the carriers the kind attaches; then every attached carrier's ``settle`` handle,
   awaited before the candidate's first turn. Seeding a triggered dimension ARMS it, and its seed
   handle returns the host's identity of the event it armed, which the session keeps.
2. :meth:`WorldSession.at_turn` — before each candidate turn, the ambient perturbation the seed
   scheduled for that turn, if any. A kind announces every turn it takes, from 1 and without a gap,
   whenever the seed schedules perturbation: the runner refuses a cell that left its schedule unannounced
   (:meth:`WorldSession.require_schedule_announced`), since the run is keyed as perturbed and only the
   kind's announcement makes that true.
3. :meth:`WorldSession.fire` / :meth:`WorldSession.observe` — a triggered dimension's condition
   happening: made to happen by the rig through the host's ``fire`` handle, or seen happening in the
   world and recorded. Each is a :class:`~threetears.evals.contracts.world_events.WorldEvent` naming the
   event that fired, and it is ``armed`` exactly when that event is one the seed armed — so the world's
   own firing on a dimension the seed also armed is never recorded as the seed's.
4. :meth:`WorldSession.end_state` — every dimension of every attached carrier, read back through its
   ``read`` handle. Read ONCE: a kind that grades its goal checks reads it here, and the runner, which
   reads it after ``invoke`` returns for the cell's trace, gets the same reading. After it the world is
   closed and nothing more may move it — a firing recorded after the end state was read would describe
   a world the stored end state does not.

**Why the runner, and not the kind, reads the end state back.** A kind that grades against a world it
read at t=0 grades the seed, not what the candidate did — the founding defect of a world read. The
runner's read after ``invoke`` is what makes the stored end state the world the cell LEFT, for every
kind, including one that never thought to read it.

**Whose world a call lands in is fixed per session, before anything moves.** The profile's registry is
the DECLARATION every reader reads — preconditions, goal checks, the bundle, coverage, authoring — and
its binding table is the world the conformance kit proves. A host whose world is real per-cell state
(a fresh game world over its own store) cannot let two cells share that table, so the kind hands its
cell's own table to :meth:`WorldSession.bind` before seeding. The session then holds a session-local
registry carrying the profile's declarations over that table (built by
:meth:`~threetears.evals.contracts.host.world.WorldRegistry.with_bindings`, held to the profile's own
rules), and every call it makes goes through it. Two cells cannot cross-write because neither holds a
path to the other's table — there is no "current world" to look up, so there is nothing to look up
wrongly, whether the cells run concurrently, in two runs, or with ``prepare`` in a child task.

**An unbound session calls the profile's own table — unless the host declared that it must not.** A
host whose handles are stateless or open per-cell state themselves (the default,
``binds_per_cell=False``) needs no bind and changes nothing. A host that declares
``binds_per_cell=True`` is saying its table is the conformance kit's and nobody else's, so a session
seeding without a bind is refused: a forgotten bind is a loud :class:`WorldSessionError` on the first
cell rather than every cell writing into one shared world. The default stays permissive because the
opposite default would make every existing host bind its own table to itself — a ritual that proves
nothing — while the declaration puts the refusal exactly where the hazard is.

**What it refuses is the kind's code, not the cell's luck.** Every :class:`WorldSessionError` is a kind
asking for something its host's world cannot give or the moment does not allow, so every cell would do
the same: it is not caught at the dispatch site, and ends the run. A refused seed is the exception —
:class:`~threetears.evals.contracts.host.world_seed.SeedRefused` propagates for the kind to translate,
because what a refusal costs is the kind's to say. A handle that raises propagates with its type
intact, so a rig fault raised as :class:`~threetears.evals.contracts.host.ApparatusError` excludes the
one cell.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Mapping
from typing import Any

from pydantic import TypeAdapter, ValidationError

from threetears.evals.contracts.base import VerbatimJsonObject
from threetears.evals.contracts.host.world import Triggered, WorldDimension, WorldRegistry
from threetears.evals.contracts.host.world_seed import SeedWrite, check_seed
from threetears.evals.contracts.models import ApparatusProvenance, WorldSeed
from threetears.evals.contracts.world_events import Firings, WorldEvent

_END_STATE: TypeAdapter[dict[str, Any]] = TypeAdapter(VerbatimJsonObject)


class WorldSessionError(Exception):
    """A kind asked its cell's world session for something the host's world, or the moment, cannot give."""


class WorldSession:
    """One cell's handle on the host's world, and the record of what happened to it.

    Built by the runner, one per cell, over the host profile's world registry, and handed to the kind's
    ``prepare``. A kind whose host binds per cell calls :meth:`bind` with that cell's own table before
    :meth:`seed`. A kind that seeds no world never calls it, and the cell then records neither world
    events nor an end state — nobody opened the world, which is a different fact from an empty one.

    **Whose apparatus it records is the constructor's to say.** The runner opens one for each cell of the
    run it drives and passes ``provenance="commissioned"``: it is the rig, and the seed it applies is what
    arms the cell's events. A host grading a cell it WITNESSED through a session of its own — real people,
    no rig — constructs it with ``provenance="witnessed"``, so :attr:`fired` reads its firings as a
    witnessed cell's, the rule :func:`~threetears.evals.run.witnessed.record_witnessed_cell` and a re-check
    read the stored cell back by, and the host never overrides ``fired_armed`` itself.
    """

    def __init__(self, registry: WorldRegistry, *, provenance: ApparatusProvenance, read_at_seed: bool = False) -> None:
        """Bind the session to the host's world, under the apparatus of the run its cell belongs to.

        Args:
            registry: The host's world registry (``profile.world``).
            provenance: The apparatus provenance of the cell's run
                (:attr:`~threetears.evals.contracts.models.EvalRun.apparatus_provenance`): ``commissioned``
                when the runner drives the cell, ``witnessed`` when a host grades a session it observed.
                Required, with no default, for the reason the run's own field has none — a default would
                let a witnessed cell's firings read as a rig's.
            read_at_seed: Read the world back through every attached dimension's ``read`` handle the moment
                :meth:`seed` has settled it — the world at t=0, before the candidate's first turn — and keep
                it as :attr:`seeded_state`. The runner asks for it when the template declares preconditions,
                which it asserts against that reading; otherwise nothing is read until the end.
        """
        self._registry = registry
        self._provenance: ApparatusProvenance = provenance
        self._read_at_seed = read_at_seed
        self._seeded_state: dict[str, Any] | None = None
        self._bound = False
        self._attached: tuple[str, ...] | None = None
        self._seeded: tuple[str, ...] = ()
        self._armed_events: dict[str, str] = {}
        self._ambient_turns: frozenset[int] = frozenset()
        self._perturbed: set[int] = set()
        self._announced: set[int] = set()
        self._events: list[WorldEvent] = []
        self._end_state: dict[str, Any] | None = None

    @property
    def registry(self) -> WorldRegistry:
        """The world this session moves: the profile's registry, or after :meth:`bind` this cell's own.

        Either way it carries the profile's declarations, so reading a dimension off it reads the same
        answer. A kind that calls a handle itself calls it here, and so lands in this cell's world.
        """
        return self._registry

    @property
    def bound(self) -> bool:
        """Whether this cell bound its own world (:meth:`bind`)."""
        return self._bound

    def bind(self, bindings: Mapping[str, Callable[..., Any]]) -> WorldRegistry:
        """Make this cell's world the one ``bindings`` resolves to — every later call this session makes lands there.

        Call it once, from ``prepare``, before :meth:`seed`. Required when the host's world declares
        ``binds_per_cell``; allowed on any host. The table is held to the profile's own rules by
        :meth:`~threetears.evals.contracts.host.world.WorldRegistry.with_bindings`: exactly the handles
        the profile binds, each callable in its role's shape.

        Args:
            bindings: This cell's table — ``{handle: callable}``, closing over this cell's world.

        Returns:
            The session-local registry, also :attr:`registry` from here on.

        Raises:
            WorldSessionError: The session was already bound, or already seeded — a bind after the seed
                would move every later call to a world the seed never wrote.
            WorldRegistrationError: The table's handle set differs from the declaration's, or a callable
                cannot be called in its role's shape.
        """
        if self._bound:
            raise WorldSessionError("this cell's world was already bound; a world session binds once per cell")
        if self._attached is not None:
            raise WorldSessionError(
                "this cell's world was already seeded through the profile's bindings; bind before seeding, or the "
                "seed and everything after it land in two different worlds"
            )
        self._registry = self._registry.with_bindings(bindings)
        self._bound = True
        return self._registry

    @property
    def opened(self) -> bool:
        """Whether the kind seeded through this session — what opens the world for the cell."""
        return self._attached is not None

    @property
    def attached(self) -> tuple[str, ...]:
        """The carriers the kind attached when it seeded, sorted; empty before it has."""
        return self._attached or ()

    @property
    def seeded(self) -> tuple[str, ...]:
        """The dimensions the seed set — and so the triggered ones it armed — in the seed's order."""
        return self._seeded

    @property
    def armed_events(self) -> dict[str, str]:
        """The events the seed armed: triggered dimension → the identity its seed handle returned, a copy."""
        return dict(self._armed_events)

    @property
    def events(self) -> tuple[WorldEvent, ...]:
        """Everything that moved the world after it was seeded, in order."""
        return tuple(self._events)

    @property
    def provenance(self) -> ApparatusProvenance:
        """The apparatus provenance this session was constructed under, which :attr:`fired` reads by."""
        return self._provenance

    @property
    def fired(self) -> Firings:
        """What fired, whoever caused it, and which firings were the seed's armed events — what the goal language reads.

        Read under the session's :attr:`provenance` through
        :meth:`~threetears.evals.contracts.world_events.Firings.of`, the one rule a re-check reads the
        stored cell back by: a ``commissioned`` session's events carry the seed's arming, while a
        ``witnessed`` one's cannot, so there ``fired_armed()`` is not established.
        """
        return Firings.of(self._events, provenance=self._provenance)

    @property
    def seeded_state(self) -> dict[str, Any] | None:
        """The world as :meth:`seed` left it, read back before the first turn, a copy.

        None unless the session was built with ``read_at_seed`` and has been seeded.
        """
        return copy.deepcopy(self._seeded_state)

    @property
    def end_state_read(self) -> dict[str, Any] | None:
        """The end state once it has been read, a copy; None until then."""
        return copy.deepcopy(self._end_state)

    async def seed(self, world_seed: WorldSeed, *, attached: Collection[str]) -> tuple[SeedWrite, ...]:
        """Seed the cell's world for a subject attaching ``attached``, then settle each attached carrier.

        Every write is checked before any is made (the seed walk), so a refused seed leaves the world as it
        was. Each write goes through the dimension's own ``seed`` handle — the path the conformance kit
        proves. A triggered dimension's seed handle arms an event and returns the host's identity of it (a
        non-empty string), which is what later tells the seed's firing from one the world makes of its own
        on that dimension. Each attached carrier that declares a ``settle`` handle is awaited after all of them,
        in carrier order, so the candidate's first turn meets the world the seed describes. Call it once
        per cell, from ``prepare``, with the empty seed when the case sets nothing: it is also how the
        cell says which carriers its subject holds, which the end state is read over.

        Args:
            world_seed: The case's seed, as ``prepare`` received it.
            attached: The carriers the subject attaches.

        Returns:
            The writes made, in the seed's order.

        Raises:
            SeedRefused: The seed walk refused a value. Nothing has been written.
            WorldSessionError: The session was already seeded; the host's world binds per cell and this
                session was never bound; ``attached`` names a carrier no declared dimension names; the
                seed schedules ambient perturbation and the host's world has no ambient-perturbation handle;
                or a triggered dimension's seed handle returned no event identity.
        """
        if self._attached is not None:
            raise WorldSessionError("this cell's world was already seeded; a world session seeds once per cell")
        if self._registry.binds_per_cell and not self._bound:
            raise WorldSessionError(
                "this host's world binds per cell (binds_per_cell=True) and this cell's session was never bound; "
                "call world.bind(<this cell's bindings>) before seed, or every cell writes into the one world the "
                "profile's bindings reach"
            )
        carriers = {declared.carrier for declared in self._registry.declarations}
        if unknown := sorted(set(attached) - carriers):
            raise WorldSessionError(
                f"the kind attaches carrier(s) {', '.join(map(repr, unknown))}, which no declared dimension names "
                f"(carriers: {', '.join(sorted(carriers)) or 'none'})"
            )
        if world_seed.ambient_perturbation_turns and self._registry.perturb_ambient is None:
            raise WorldSessionError(
                "the seed schedules ambient perturbation before turn(s) "
                f"{world_seed.ambient_perturbation_turns!r}, and this host's world declares no perturb_ambient handle"
            )
        writes = check_seed(self._registry, world_seed.namespaces, attached=attached)
        armed: dict[str, str] = {}
        for write in writes:
            returned = await self._registry.call(write.handle, write.value)
            declared = self._registry.get(write.name)
            if declared is not None and isinstance(declared.when, Triggered):
                if not isinstance(returned, str) or not returned:
                    raise WorldSessionError(
                        f"{write.name} is triggered, so its seed handle arms an event and returns the host's identity "
                        f"of it — it returned {returned!r}. Without it the seed's firing cannot be told from one "
                        "the world makes of its own on this dimension"
                    )
                armed[write.name] = returned
        self._armed_events = armed
        self._attached = tuple(sorted(set(attached)))
        self._seeded = tuple(write.name for write in writes)
        self._ambient_turns = frozenset(world_seed.ambient_perturbation_turns)
        settle = self._registry.settle
        for carrier in self._attached:
            if (handle := settle.get(carrier)) is not None:
                await self._registry.call(handle)
        if self._read_at_seed:
            self._seeded_state = await self._read_attached()
        return writes

    async def at_turn(self, turn: int) -> WorldEvent | None:
        """Mark that the candidate is about to take turn ``turn``, applying any ambient perturbation due.

        A kind calls it before each candidate turn, counted from 1. The perturbation scheduled for a turn is
        applied once, however often that turn is announced.

        Args:
            turn: The turn about to be taken.

        Returns:
            The perturbation recorded, or None when none was due.

        Raises:
            WorldSessionError: The world is not open, or is already closed.
        """
        self._require_open("announce a turn")
        self._announced.add(turn)
        if turn not in self._ambient_turns or turn in self._perturbed:
            return None
        handle = self._registry.perturb_ambient
        # Refused at seed when absent, so a scheduled turn always has a handle here.
        assert handle is not None
        moved = await self._registry.call(handle)
        self._perturbed.add(turn)
        event = WorldEvent(
            kind="ambient",
            caused_by="rig",
            turn=turn,
            moved=None if moved is None else [str(name) for name in moved],
        )
        self._events.append(event)
        return event

    @property
    def announced_turns(self) -> tuple[int, ...]:
        """The turns the kind announced through :meth:`at_turn`, ascending."""
        return tuple(sorted(self._announced))

    def require_schedule_announced(self, *, ran_its_course: bool) -> None:
        """Refuse a cell whose kind never told the session which turns it took, when the seed scheduled perturbation.

        The run's identity keys its condition on the scheduled turns
        (``EvalRun.resolved_ambient_perturbation_turns``), and the only thing that applies one is the kind
        announcing that turn. A kind that never calls :meth:`at_turn` would produce cells under a condition
        keyed "perturbed" with nothing perturbed; one that announces turn 3 and not turn 2 would skip the
        perturbation due before turn 2 while having taken it. Either is the kind's code, so every cell would
        do the same. A scheduled turn past the last one announced is a turn the cell never reached, which is
        an honest outcome — the result's ``world_events`` carry each perturbation actually applied.

        **Announcing nothing is refused only for a cell that ran its course**: one that carries no error, was
        not stopped by the run's cost cap, and whose conversation, if it had one, stopped on its turn budget.
        Any other cell may have ended before turn 1 — a simulator or rig fault, a candidate failing first, the
        cap reached before the first answer, every simulated actor leaving before anything was delivered — and
        then announcing nothing is the truth, not a defect; such a cell is excluded, failed or stops the run on
        its own terms. A kind that never announces still fails on every cell that ran its course. The gap
        check holds for every cell: turns a kind did announce are its turns, however the cell ended.

        Called by the runner once ``invoke`` has returned, for a world the kind opened.

        Args:
            ran_its_course: Whether the cell ran to its own end — see above. The runner derives it from the
                kind's output and the cell's sink.

        Raises:
            WorldSessionError: The seed scheduled perturbation and a cell that ran its course announced no
                turn, or any cell announced turns with a gap (they must run 1, 2, … without one).
        """
        if not self._ambient_turns:
            return
        if not self._announced:
            if not ran_its_course:
                return
            raise WorldSessionError(
                "the seed schedules ambient perturbation before turn(s) "
                f"{sorted(self._ambient_turns)!r}, and the kind announced no turn through at_turn() in a cell that "
                "ran its course, so none was applied while the run is keyed as perturbed; a kind calls "
                "world.at_turn(n) before each turn it takes"
            )
        if gaps := sorted(set(range(1, max(self._announced) + 1)) - self._announced):
            raise WorldSessionError(
                f"the kind announced turns {sorted(self._announced)!r} and skipped {gaps!r}; a kind announces every "
                "turn it takes, from 1, or a perturbation scheduled for a skipped turn is never applied"
            )

    async def fire(self, dimension: str, *, turn: int | None = None) -> WorldEvent:
        """Make a triggered dimension's condition happen, through the host's ``fire`` handle, and record it.

        Args:
            dimension: The triggered dimension.
            turn: The candidate turn it fired before or during, counted from 1; None for a kind with no turns.

        Returns:
            The firing recorded.

        Raises:
            WorldSessionError: The world is not open or is closed; the dimension is undeclared, not
                triggered, or on a carrier this cell did not attach; it has no fire handle (a ``human``
                trigger never does — record a person's act with :meth:`observe`); or this cell's seed did
                not arm it.
        """
        declared, trigger = self._triggered(dimension, verb="fire")
        if trigger.fire is None:
            raise WorldSessionError(
                f"{dimension} has no fire handle, so the rig cannot make its condition happen"
                + (
                    " — only a person brings a human trigger about; record it with observe()"
                    if trigger.kind == "human"
                    else ""
                )
            )
        if dimension not in self._seeded:
            raise WorldSessionError(
                f"{dimension} was not armed by this cell's seed, so firing it would record a condition the run "
                "never set up"
            )
        await self._registry.call(trigger.fire)
        return self._record(declared, trigger, caused_by="rig", event=self._armed_events[dimension], turn=turn)

    def observe(self, dimension: str, *, event: str, turn: int | None = None) -> WorldEvent:
        """Record a triggered dimension's condition happening in the world, which the rig did not cause.

        The candidate's own action brought an event about, a person made a ruling, or the world's own
        clock or rules fired an event of its own: the world fired it, and the kind saw it happen. Nothing
        is called. The record is ``armed`` exactly when ``event`` is an identity a seed handle returned for
        this cell — the seed's own event, on this dimension or one it also moves — and not when the world
        fired an event of its own on a dimension the seed also armed.

        Args:
            dimension: The triggered dimension.
            event: The host's identity of the event that fired, as the world reports it.
            turn: The candidate turn it fired before or during, counted from 1; None for a kind with no turns.

        Returns:
            The firing recorded.

        Raises:
            WorldSessionError: The world is not open or is closed; the dimension is undeclared, not
                triggered, or on a carrier this cell did not attach; or ``event`` is not a non-empty string.
        """
        declared, trigger = self._triggered(dimension, verb="observe")
        if not isinstance(event, str) or not event:
            raise WorldSessionError(
                f"cannot observe {dimension}: a firing names the host's identity of the event that fired, and "
                f"{event!r} is not one — provenance is per event, and without it this firing's cannot be read"
            )
        return self._record(declared, trigger, caused_by="world", event=event, turn=turn)

    async def end_state(self) -> dict[str, Any]:
        """The world the cell left: every dimension of every attached carrier, read through its ``read`` handle.

        Read once. The first call reads and closes the world; every later call — the runner's after
        ``invoke``, when the kind read it first to grade — returns the same reading.

        Returns:
            Dimension name → value, in registration order, a copy.

        Raises:
            WorldSessionError: The world was never opened, or a read handle returned a value storage cannot
                hold as JSON.
        """
        if self._attached is None:
            raise WorldSessionError("this cell's world was never seeded, so it has no end state to read")
        if self._end_state is None:
            self._end_state = await self._read_attached()
        return copy.deepcopy(self._end_state)

    async def _read_attached(self) -> dict[str, Any]:
        """Every dimension of every attached carrier, read through its ``read`` handle, as storage holds it.

        Raises:
            WorldSessionError: A read handle returned a value storage cannot hold as JSON.
        """
        attached = set(self._attached or ())
        read = {
            declared.name: await self._registry.call(declared.read)
            for declared in self._registry.declarations
            if declared.carrier in attached and declared.read is not None
        }
        try:
            return _END_STATE.validate_python(read)
        except ValidationError as unstorable:
            raise WorldSessionError(
                f"a read handle returned a value the world state cannot store as JSON: {unstorable}"
            ) from unstorable

    def _require_open(self, action: str) -> None:
        """Refuse to move a world that is not open, or that is already closed.

        Args:
            action: What was asked, for the message.

        Raises:
            WorldSessionError: The world was never seeded, or its end state has been read.
        """
        if self._attached is None:
            raise WorldSessionError(f"cannot {action}: this cell's world was never seeded")
        if self._end_state is not None:
            raise WorldSessionError(
                f"cannot {action}: this cell's end state was already read, and anything recorded after it would "
                "describe a world the stored end state does not"
            )

    def _triggered(self, dimension: str, *, verb: str) -> tuple[WorldDimension, Triggered]:
        """The declaration and trigger of a dimension a firing names, or the refusal.

        Args:
            dimension: The dimension named.
            verb: What was asked, for the message.

        Returns:
            Its declaration and its trigger.

        Raises:
            WorldSessionError: See :meth:`fire`.
        """
        self._require_open(f"{verb} {dimension}")
        declared = self._registry.get(dimension)
        if declared is None:
            raise WorldSessionError(f"cannot {verb} {dimension}: this host's world declares no such dimension")
        if not isinstance(declared.when, Triggered):
            raise WorldSessionError(
                f"cannot {verb} {dimension}: it is set at t=0, and only a triggered dimension fires"
            )
        if declared.carrier not in self.attached:
            raise WorldSessionError(
                f"cannot {verb} {dimension}: its carrier {declared.carrier!r} is not attached for this cell's subject"
            )
        return declared, declared.when

    def _record(
        self, declared: WorldDimension, trigger: Triggered, *, caused_by: str, event: str, turn: int | None
    ) -> WorldEvent:
        """Append one firing to the record.

        Args:
            declared: The dimension that fired.
            trigger: Its trigger.
            caused_by: ``rig`` or ``world``.
            event: The host's identity of the event that fired.
            turn: The turn, or None.

        Returns:
            The record, ``armed`` when ``event`` is one the seed armed.
        """
        record = WorldEvent.model_validate(
            {
                "kind": trigger.kind,
                "dimension": declared.name,
                "condition": trigger.condition,
                "caused_by": caused_by,
                "event": event,
                "armed": event in self._armed_events.values(),
                "turn": turn,
            }
        )
        self._events.append(record)
        return record


__all__ = ["WorldSession", "WorldSessionError"]
