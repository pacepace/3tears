"""Proving a world declaration — obligations derived from shape, run by the host against its own code.

A registration is a claim, and R10's warning lands on this contract before anything else: *any
entry that is declared rather than derived is a claim under test*. Deriving the declaration from
code is not portable — static analysis of an unknown host in an unknown idiom is not a thing to
promise — so what the engine ships instead is a **conformance kit the host runs against itself**,
a TCK in the sense Java and MCP use the word.

**One rule, and it is the whole design.** A dimension's obligations are derived from its declared
shape; every derived obligation is mandatory; and ``unavailable`` names a host capability gap
rather than a skip. Do not restructure this into one universal mandate with waivers. A mandate the
hard hosts routinely waive becomes ceremony, and the hosts least like an in-memory dict — the ones
this contract exists for — are exactly the ones that would live in the waivers. What follows from
that: **an unproved declaration must never render identically to a proved one**, which is why
:attr:`ConformanceResult.proved` is a single property every renderer reads rather than a rule
written in prose and enforced nowhere.

Six checks live here.

* **Round trip** — synthesize a value, seed it, read it back. Catches a seeder wired to nothing.
* **Perception A/B** — render the subject view at two values and assert it moved. This is the
  check that survives a refactor: delete the renderer and the declaration fails the next day
  rather than going on saying "supported".
* **Perception stillness** — move a dimension and assert every surface its ``perceived_by`` does not
  name held still; for a judge-only dimension, every surface. The other half of A/B: a surface that
  shows hidden state *beside* what it is declared to carry still moves where A/B looks, so only this
  finds it — and a judge-only dimension's "no subject sees this" becomes a checked claim.
* **Ambient isolation** — hold every declared dimension fixed, move the surroundings, and assert
  the subject's view did not follow. Movement attributable to nothing declared is perception of
  undeclared state, which is the founding defect found mechanically rather than by an incident.
* **Independence** — seed one dimension, seed another, read the first back. A scenario presumes
  several preconditions at once, so if seeding B silently undoes A every composed precondition is
  a lie that surfaces only as a subject behaving oddly.
* **Vocabulary completeness** — parse the expressions a host's scenarios presume and check that
  every world path they read resolves to a declared dimension. Static: it needs no run, no seed
  and no subject, which is what lets an authoring gate refuse a typo'd path before it scores a
  subject down for a state nobody ever set.

**The kit moves the world it is given, and does not put it back.** It seeds, perturbs and fires
through the host's own handles, which is the point — a kit driving a parallel path would prove a
path no run takes. Run it against a rig, never against a world something else is reading.

**A host whose dimensions are coupled declares the coupling rather than having it found as a
defect.** Production state is rarely a set of independent dicts: a host may start the head of a
queue that lands on a silent output, or refuse two items under one id. Such a host names a
``base_world`` every check composes over and a ``coherence`` handle saying which composed worlds
it holds as stated, and the kit draws only values it holds beside what the world already holds.
Without either, every combination of schema-valid values is taken to be one the host can hold.

**A coupling narrows which values a check draws; it never decides whether a check runs.** That is
what keeps the coherence handle from being a waiver, which the one rule above forbids: a handle
answering "not held" for every value a check could use would otherwise turn that check into a gap
the host authored. So a declared dimension the host holds too few values of — over its own base
world, or beside a sibling at every value the dimension can take — is a ``failed`` verdict naming
the coupling, never an ``unavailable`` one. ``unavailable`` keeps meaning what the shape forces.

**Two conventions the kit fixes, because a generic caller needs them fixed.** A ``subject_view``
binding is called with one keyword, ``surfaces``, holding the surface names to render; and a
``seed``, ``perturb`` or ``fire`` binding is called with the value as its single positional
argument, ``fire`` with none. A triggered dimension's ``seed`` binding ARMS an event and returns the
host's identity of it — a non-empty string, which a cell's world session keeps so a firing of the
seed's own event is told from one the world makes of its own on that dimension; the round trip fails
a seed binding that returns none. **The identity must be unique to the event, an obligation no check
here can verify**: a firing is recorded armed exactly when the identity the kind observes is one a seed
returned, on any dimension that event moves (``WorldSession.observe``), so an identity that names the
DIMENSION rather than the event — or one a world's own events share — makes every firing on it read
armed, and ``fired_armed()`` passes for a candidate whose world fired an event of its own. Everything
else about a host stays the host's business.

The checks are derived from an obligations table, one row per registrable shape; its rows are
:data:`ObligationRow`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

from threetears.evals.contracts.dsl import DSLError, extract_paths
from threetears.evals.contracts.host.world import Triggered, WorldDimension, WorldRegistry
from threetears.evals.contracts.host.world_schema import UnsupportedSchemaError, honoured_kind, json_equal

#: One conformance check.
CheckName = Literal[
    "round_trip",
    "perception_ab",
    "perception_stillness",
    "ambient_isolation",
    "independence",
    "vocabulary_completeness",
]

#: What a check concluded.
#:
#: ``unavailable`` is non-fatal and is **not** a skip: it records that this host cannot supply what
#: the proof needed — a device asleep, a condition only a person can fire, an optional binding the
#: host never wrote. A kit that could not say so becomes a kit nobody runs, and one that said it
#: silently would let a gap render as a pass.
Outcome = Literal["passed", "failed", "unavailable"]

#: Why a result is narrower than a clean proof. Present on every outcome that is not one.
#:
#: * ``plumbing_only`` — the round trip carried the value, and the dimension's read is an authored
#:   claim rather than a measurement, so nothing here speaks for the world holding the property.
#: * ``arming_only`` — a triggered dimension was armed and the host binds no operation that fires
#:   its condition, so the trip stops short of the value ever arriving.
#: * ``not_instantiable_unattended`` — the condition is a person doing something. An answer about
#:   representability, not a failed declaration.
#: * ``no_perturbation_binding`` — the proof needed the host to move state no run controls, and
#:   this host cannot. samsung-frame-art-loader will carry this one forever: a brightness sensor, a
#:   heartbeat and a human with a remote all move its world.
#: * ``nothing_to_observe`` — there is no surface for the check to watch: no dimension here is
#:   perceived, or (for stillness) the dimension is perceived by every surface the registry names, or
#:   every other surface also perceives a sibling that moved with it. Recorded rather than passed,
#:   because a check with nothing to look at proves nothing.
#: * ``schema_admits_too_few_values`` — the dimension's own schema cannot supply the distinct
#:   values the check needed, so there is nothing to vary it between. A declared shape making a
#:   proof unreachable, which is what ``unavailable`` means — not an engine gap, and not a failure.
#: * ``nothing_to_resolve`` — nothing exercised this registry's names: either no expression was
#:   handed to the vocabulary check, or the ones handed to it read no world state. **One reason
#:   for both**, because they are one fact — a corpus of ledger predicates proves no more about
#:   the vocabulary than an empty one, and two verdicts for it would be the same vacuous pass by
#:   another door. Recorded rather than passed for the reason every other gap here is: "every
#:   path resolved" over no paths at all is vacuously true and renders identically to a corpus
#:   that was actually checked.
#: * ``seeding_did_not_take`` — the check's own setup did not land, so it never reached the
#:   question it exists to ask. Recorded rather than reported as a failure of THIS check, because
#:   round-trip already names that defect and a second verdict would blame the wrong thing: a
#:   perception check whose seed never landed reads as a deleted renderer, and an independence
#:   check whose seed never landed reads as an innocent sibling clobbering it.
Qualification = Literal[
    "plumbing_only",
    "arming_only",
    "not_instantiable_unattended",
    "no_perturbation_binding",
    "nothing_to_observe",
    "schema_admits_too_few_values",
    "nothing_to_resolve",
    "seeding_did_not_take",
]

#: A row of the obligations table. The kit derives obligations from these and nothing else,
#: so a shape that lands on no row would owe nothing — which
#: ``tests/test_world_conformance.py`` refuses, by asserting the fixture set covers every
#: row rather than by trusting this list to stay complete on its own.
ObligationRow = Literal[
    "seedable_machine_read",
    "seedable_labeled_read",
    "perceivable_not_seedable",
    "triggered_automatic",
    "triggered_human",
    "every_dimension",
]

#: Distinct values the kit will try to synthesize before giving up on a schema.
_VALUE_ATTEMPTS = 8

#: Letters the string synthesizer draws from, one per distinct value.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


#: One check that speaks for a single dimension. The registry-wide check has no such shape, which
#: is why it is appended rather than dispatched.
_DimensionCheck = Callable[["WorldRegistry", "WorldDimension"], Awaitable["ConformanceResult"]]


class _NotHeld(Exception):
    """The schema offers values the check could use, and the host holds none of them as stated here.

    Private, like :class:`_SchemaTooNarrow`, and for the same reason an exception: a check catches it
    and records a ``failed`` verdict (:func:`_not_held`) — never an ``unavailable`` one, because a
    coupling that starves a check is the host's declarations contradicting each other, not a gap the
    shape forces. The message names what the host's coherence handle said, because the coupling it
    describes is the host's to explain.
    """


class WorldConformanceError(RuntimeError):
    """An ENGINE gap: a check the kit cannot run, or a record its own vocabulary forbids.

    Deliberately not an ``unavailable`` result. ``unavailable`` means *this host cannot supply
    what the proof needed*, and reporting an engine limitation that way would launder a hole in
    the kit into a permanent property of the host — recorded, disclosed, and never fixed because
    nothing says it is fixable. Raised where somebody can extend the kit instead.

    **Only the engine's gaps come through here**, and keeping that true takes care, because the two
    look alike from inside a generator. A schema the synthesizer has no code for is this; a schema
    that simply cannot take two values is the HOST's declaration making a proof unreachable, and
    that is an ``unavailable`` verdict which costs no other dimension its own verdict.
    """


class _SchemaTooNarrow(Exception):
    """The declared schema cannot supply the distinct values a check needs.

    Private, and never seen outside this module: a check catches it and records
    ``schema_admits_too_few_values``. It is an exception rather than a sentinel return only because
    the value helpers are called from four places and a sentinel would have to be threaded through
    all of them — the shape it describes is a host declaration, not an error.
    """


@dataclass(frozen=True)
class ConformanceResult:
    """One check's verdict on one dimension, or on the registry when the check is registry-wide."""

    check: CheckName
    """Which check produced this."""

    outcome: Outcome
    """``passed`` | ``failed`` | ``unavailable``."""

    detail: str
    """REQUIRED prose. What was proved, or what stopped the proof, in terms a host can act on.

    Required on a pass as well as a failure, for the reason ``matters`` is required on a
    registration: a verdict with no sentence behind it is a label.
    """

    dimension: str | None = None
    """The dimension this speaks for, or None for a registry-wide check."""

    qualification: Qualification | None = None
    """Why this result is narrower than a clean proof, or None when it is one."""

    def __post_init__(self) -> None:
        """Refuse a verdict this module's own vocabulary does not admit.

        Every other record in this contract refuses incoherence where it is written — the registry
        validates its whole declaration set at construction, and ``WorldDimension.capability``
        refuses the unregistrable quadrant rather than returning the nearest one. This one did
        not, and the two shapes it let through are the two that matter most: an ``unavailable``
        with no reason is a disclosed gap in a module whose stated point is that the reasons ARE
        the disclosure surface, and a ``failed`` carrying a gap reason gives one verdict two
        answers.

        Only this module constructs these, and every path it takes is already right — which is
        precisely when a rule is cheap to make structural, and precisely when it otherwise stays
        prose until somebody adds a fifth check.

        Raises:
            WorldConformanceError: The outcome and qualification contradict each other.
        """
        if self.outcome == "unavailable" and self.qualification is None:
            raise WorldConformanceError(
                f"{self.check} recorded unavailable with no qualification — the reasons a proof was "
                "not reached are the disclosure surface, so a gap with none is unreadable"
            )
        if self.outcome == "failed" and self.qualification is not None:
            raise WorldConformanceError(
                f"{self.check} recorded failed and qualified it {self.qualification!r} — a failure names "
                "its defect in its detail, and a gap reason beside it is a second, different answer"
            )
        if self.outcome == "passed" and self.qualification not in (None, "plumbing_only"):
            raise WorldConformanceError(
                f"{self.check} recorded passed and qualified it {self.qualification!r} — only "
                "plumbing_only narrows a pass; every other reason replaces one"
            )

    @property
    def proved(self) -> bool:
        """Whether this result licenses the claim the declaration makes.

        The one place the rendering rule lives, so no surface can reimplement it wrongly: a
        qualified pass is not a proof, and neither is an ``unavailable``.
        """
        return self.outcome == "passed" and self.qualification is None


@dataclass(frozen=True)
class WorldConformanceReport:
    """Every verdict from one run of the kit over one registry."""

    results: tuple[ConformanceResult, ...]
    """Every check that ran, in check-major order."""

    @property
    def proved(self) -> tuple[ConformanceResult, ...]:
        """The results that license the claims their declarations make."""
        return tuple(result for result in self.results if result.proved)

    @property
    def qualified(self) -> tuple[ConformanceResult, ...]:
        """Passes narrower than the declaration they speak for — plumbing carried, world unproven."""
        return tuple(result for result in self.results if result.outcome == "passed" and not result.proved)

    @property
    def unavailable(self) -> tuple[ConformanceResult, ...]:
        """Proofs this host could not supply what was needed for. Non-fatal, and never a pass."""
        return tuple(result for result in self.results if result.outcome == "unavailable")

    @property
    def failures(self) -> tuple[ConformanceResult, ...]:
        """Declarations the host's own code contradicts."""
        return tuple(result for result in self.results if result.outcome == "failed")

    def for_dimension(self, name: str) -> tuple[ConformanceResult, ...]:
        """Every result speaking for one dimension.

        Args:
            name: A declared dimension name.

        Returns:
            Its results, in check-major order.
        """
        return tuple(result for result in self.results if result.dimension == name)


def obligation_rows(declared: WorldDimension) -> frozenset[ObligationRow]:
    """Which rows of the obligations table this dimension's declared shape sits on.

    Rows overlap on purpose — a triggered dimension is also a seedable one, and owes both rows'
    obligations. Exposed rather than kept private because the test suite generates its coverage
    matrix from this, so a shape that lands on no row is visible as a hole rather than as silence.

    Args:
        declared: The dimension.

    Returns:
        Every applicable row, always including ``every_dimension``.
    """
    rows: set[ObligationRow] = {"every_dimension"}
    if declared.seedable:
        rows.add("seedable_labeled_read" if declared.evidence == "labeled" else "seedable_machine_read")
    if declared.perceivable and not declared.seedable:
        rows.add("perceivable_not_seedable")
    if isinstance(declared.when, Triggered):
        rows.add("triggered_human" if declared.when.kind == "human" else "triggered_automatic")
    return frozenset(rows)


def obligations(declared: WorldDimension) -> tuple[CheckName, ...]:
    """The per-dimension checks this dimension's shape owes, derived and never declared.

    ``perception_stillness`` is owed by every dimension, perceived or not: a judge-only dimension
    claims no surface shows it, and a perceived one claims the surfaces it names are the only ones.
    Whether there is another surface to watch is a fact about the registry rather than the shape,
    so a dimension with none still owes the check and records ``nothing_to_observe``.

    ``ambient_isolation`` and ``vocabulary_completeness`` are absent here and are not thereby
    optional: both are the ``every_dimension`` row's obligations, and each asks one question of
    the whole registry rather than one per dimension, so the kit discharges each once. Reporting
    N copies of a single answer would be a lie told by repetition.

    Args:
        declared: The dimension.

    Returns:
        Its checks, in the order the kit runs them.
    """
    owed: list[CheckName] = []
    if declared.seedable:
        owed.append("round_trip")
    if declared.perceivable:
        owed.append("perception_ab")
    owed.append("perception_stillness")
    if declared.seedable:
        owed.append("independence")
    return tuple(owed)


async def check_world_conformance(
    registry: WorldRegistry,
    *,
    expressions: Sequence[str] = (),
) -> WorldConformanceReport:
    """Run every obligation this registry's declarations imply, against the host's own handles.

    Check-major rather than dimension-major, because independence deliberately disturbs siblings
    and a reader comparing one check across dimensions is the common way to read the report. The
    registry-wide verdicts land last, after the per-dimension ones they are not derived from.

    **This mutates the world the registry is bound to and does not restore it.**

    Args:
        registry: The host's world registry.
        expressions: The scenario expressions whose vocabulary this registry has to cover —
            preconditions and goal checks alike, since a path reads the same world whichever
            side of a run it is asked on. Bare text rather than a host's scenario type, because
            the kit resolves paths and knows nothing about how a host stores them; a caller that
            wants a failure to name a template does that naming itself. Supplying none is not a
            way to skip the check — it produces an ``unavailable`` verdict saying so.

    Returns:
        Every verdict, in check-major order.

    Raises:
        WorldConformanceError: A declared schema is outside what the kit can synthesize values
            from. An engine gap, raised rather than recorded, so it is fixed rather than disclosed.
    """
    per_dimension: tuple[tuple[CheckName, _DimensionCheck], ...] = (
        ("round_trip", _round_trip),
        ("perception_ab", _perception_ab),
        ("perception_stillness", _perception_stillness),
        ("independence", _independence),
    )
    results: list[ConformanceResult] = []
    for check, runner in per_dimension:
        for declared in registry.declarations:
            if check in obligations(declared):
                results.append(await runner(registry, declared))
    # The registry-wide checks last. Ambient isolation does not care what the others left behind:
    # it re-pins every declared dimension before it looks at anything. Vocabulary completeness
    # reads no world state at all, so its verdict is the same wherever it runs.
    results.append(await _ambient_isolation(registry))
    results.append(_vocabulary_completeness(registry, expressions))
    return WorldConformanceReport(tuple(results))


async def _round_trip(registry: WorldRegistry, declared: WorldDimension) -> ConformanceResult:
    """Seed a value the world does not already hold, read it back, and require it took.

    The value is chosen to *differ from what is there*, which is the difference between this
    check and a decorative one: synthesizing whatever the schema offers first would let a seeder
    that does nothing pass whenever the schema's first value happened to be the world's default.
    A triggered dimension's seed must also name the event it armed, since a run's world session
    refuses a seed that does not — so this is the check that finds it before a run does. That the
    name is unique to the event (never the dimension's name, never one the world's own events share)
    is the host's obligation, which no single round trip can observe; see the module docstring.

    Args:
        registry: The host's registry.
        declared: A seedable dimension.

    Returns:
        The verdict.
    """
    if unattended := _refuse_unattended("round_trip", declared):
        return unattended
    read = declared.read
    assert read is not None, "a seedable dimension has a read handle — the registry refuses one without"
    await _rebase(registry)
    before = await registry.call(read)
    try:
        value = await _a_value_other_than(registry, declared, before)
    except _SchemaTooNarrow as narrow:
        return _too_narrow("round_trip", declared, narrow)
    except _NotHeld as not_held:
        return _not_held("round_trip", declared, not_held)
    applied = await _apply(registry, declared, value)
    if (
        isinstance(declared.when, Triggered)
        and declared.write_handle == declared.seed
        and (not isinstance(applied.written, str) or not applied.written)
    ):
        return ConformanceResult(
            check="round_trip",
            outcome="failed",
            dimension=declared.name,
            detail=(
                f"{declared.name} is triggered, and its seed handle armed {value!r} and returned {applied.written!r} "
                "rather than the identity of the event it armed — a cell needs it to tell the seed's firing from one "
                "the world makes of its own on this dimension, and refuses the seed without it"
            ),
        )
    if qualification := applied.qualification:
        return ConformanceResult(
            check="round_trip",
            outcome="unavailable",
            dimension=declared.name,
            qualification=qualification,
            detail=(
                f"{declared.name} was armed with a value and this host binds nothing that fires "
                f"{_condition_of(declared)!r}, so nothing here says the value ever arrives"
            ),
        )
    observed = await registry.call(read)
    if not json_equal(observed, value):
        return ConformanceResult(
            check="round_trip",
            outcome="failed",
            dimension=declared.name,
            detail=(
                f"{declared.name} was seeded {value!r} and reads back {observed!r} — the seed handle and the "
                "read handle disagree, so a run that presumed this precondition would not have been given it"
            ),
        )
    if declared.evidence == "labeled":
        return ConformanceResult(
            check="round_trip",
            outcome="passed",
            dimension=declared.name,
            qualification="plumbing_only",
            detail=(
                f"{declared.name} carries a seeded value to its read handle. That read reports an authored "
                "claim rather than a measurement, so the world holding the property stays a claim"
            ),
        )
    return ConformanceResult(
        check="round_trip",
        outcome="passed",
        dimension=declared.name,
        detail=f"{declared.name} seeded {value!r} and read back the same, so the seeding path reaches the world",
    )


async def _perception_ab(registry: WorldRegistry, declared: WorldDimension) -> ConformanceResult:
    """Put the dimension at several values and require EVERY surface it names to move across them.

    Per surface rather than over the whole view, because a view that moved proves only that one of
    its surfaces did: a status line stuck on a fresh component's default sits beside a full
    rendering that moves, and the whole view still changes. The values are a generated distinct
    pair plus the schema's boundary values (:func:`_boundary_values`), because a summary surface —
    a status, a count, playing or stopped — typically moves only at a boundary, and two arbitrary
    non-empty values would hold it still while it was wired correctly.

    Args:
        registry: The host's registry.
        declared: A perceivable dimension.

    Returns:
        The verdict.
    """
    view = registry.subject_view
    assert view is not None, "a perceivable dimension implies a subject_view — the registry refuses one without"
    if unattended := _refuse_unattended("perception_ab", declared):
        return unattended
    if not declared.settable:
        return ConformanceResult(
            check="perception_ab",
            outcome="unavailable",
            dimension=declared.name,
            qualification="no_perturbation_binding",
            detail=(
                f"{declared.name} is perceived and no run controls it, and this host binds no way to move it "
                "out of band — so its perception claim stands unproved rather than proved"
            ),
        )
    surfaces = declared.perceived_by
    swept = await _sweep(registry, declared, "perception_ab", surfaces)
    if isinstance(swept, ConformanceResult):
        return swept
    renders, tried, not_held = swept.renders, swept.tried, swept.not_held
    skipped = f"; boundary value(s) {not_held!r} were not held by the world as seeded, so not tried" if not_held else ""
    if still := [surface for surface in surfaces if all(body == renders[surface][0] for body in renders[surface])]:
        return ConformanceResult(
            check="perception_ab",
            outcome="failed",
            dimension=declared.name,
            detail=(
                f"{declared.name} took each of {tried!r} and surface(s) {still!r} rendered identically every "
                "time — nothing on that surface carries this dimension, so a subject reading it sees something "
                f"other than the seeded world{skipped}"
            ),
        )
    return ConformanceResult(
        check="perception_ab",
        outcome="passed",
        dimension=declared.name,
        detail=(
            f"every surface {list(surfaces)!r} rendered differently somewhere across {tried!r}, so a subject "
            f"with them attached perceives this dimension on each{skipped}"
        ),
    )


async def _perception_stillness(registry: WorldRegistry, declared: WorldDimension) -> ConformanceResult:
    """Move the dimension and require every surface it does NOT name to hold still.

    Perception A/B proves the named surfaces move; nothing there proves the unnamed ones do not, and a
    judge-only dimension's "no subject sees this" is otherwise a declaration nobody checked. The leak
    this finds is the one A/B cannot: a player surface that shows hidden state *in addition to* what it
    is declared to carry passes A/B, because every distinction the declared surface carries is still
    there. So: put the dimension at the same values A/B uses (:func:`_sweep` — a distinct pair the host
    holds, plus the schema's boundaries), render the whole subject view at each, and require every
    surface outside ``perceived_by`` to render identically every time. For a judge-only dimension that
    is every surface the registry names.

    **Movement a sibling accounts for is not a leak.** A host can hold a dimension's value beside a
    sibling only by moving that sibling — a coupling it declares — and a surface perceiving the moved
    sibling then moves for a declared reason. Such surfaces are left out of this verdict and named in
    it; whether the sibling should have moved at all is independence's question, not this one.

    **The surfaces watched are the ones the registry names** — the union of every ``perceived_by``. A
    surface no dimension names is not a surface the kit can ask the host to render, so a leak onto one
    is outside this check; ambient isolation is no help there either, and the detail does not claim it.

    Args:
        registry: The host's registry.
        declared: Any dimension.

    Returns:
        The verdict.
    """
    if unattended := _refuse_unattended("perception_stillness", declared):
        return unattended
    everywhere = tuple(dict.fromkeys(surface for d in registry.declarations for surface in d.perceived_by))
    unnamed = tuple(surface for surface in everywhere if surface not in declared.perceived_by)
    if not unnamed:
        return ConformanceResult(
            check="perception_stillness",
            outcome="unavailable",
            dimension=declared.name,
            qualification="nothing_to_observe",
            detail=(
                f"no dimension here is perceived by any surface, so there is no subject view for {declared.name} "
                "to leak into and nothing for this check to watch"
                if not everywhere
                else f"{declared.name} is perceived by every surface this registry names ({list(everywhere)!r}), "
                "so there is no other surface for it to leak into and nothing for this check to watch"
            ),
        )
    if not declared.settable:
        return ConformanceResult(
            check="perception_stillness",
            outcome="unavailable",
            dimension=declared.name,
            qualification="no_perturbation_binding",
            detail=(
                f"no run controls {declared.name} and this host binds no way to move it out of band, so whether "
                f"surface(s) {list(unnamed)!r} stay still when it moves is unproved"
            ),
        )
    swept = await _sweep(registry, declared, "perception_stillness", everywhere)
    if isinstance(swept, ConformanceResult):
        return swept
    accounted = {
        surface
        for sibling in registry.declarations
        if sibling.name in swept.siblings_moved
        for surface in sibling.perceived_by
    }
    watched = [surface for surface in unnamed if surface not in accounted]
    excused = (
        f"; surface(s) {sorted(set(unnamed) - set(watched))!r} were not judged, because sibling(s) "
        f"{sorted(swept.siblings_moved)!r} they perceive moved with it"
        if len(watched) < len(unnamed)
        else ""
    )
    skipped = (
        f"; boundary value(s) {swept.not_held!r} were not held by the world as seeded, so not tried"
        if swept.not_held
        else ""
    )
    if not watched:
        return ConformanceResult(
            check="perception_stillness",
            outcome="unavailable",
            dimension=declared.name,
            qualification="nothing_to_observe",
            detail=(
                f"every surface outside {declared.name}'s perceived_by also perceives a sibling that moved with "
                f"it, so no surface could be attributed to {declared.name} alone{excused}"
            ),
        )
    claim = (
        f"{declared.name} is perceived by no surface"
        if not declared.perceivable
        else f"{declared.name} is perceived by {list(declared.perceived_by)!r} alone"
    )
    if moved := [
        surface for surface in watched if any(body != swept.renders[surface][0] for body in swept.renders[surface])
    ]:
        shown = "; ".join(f"{surface}: {swept.renders[surface]!r}" for surface in moved)
        return ConformanceResult(
            check="perception_stillness",
            outcome="failed",
            dimension=declared.name,
            detail=(
                f"{claim}, yet across {swept.tried!r} surface(s) {moved!r} rendered differently ({shown}) — that "
                "surface shows this dimension without declaring it, so a subject reading it perceives state the "
                f"declaration says it cannot{excused}{skipped}"
            ),
        )
    return ConformanceResult(
        check="perception_stillness",
        outcome="passed",
        dimension=declared.name,
        detail=(
            f"{claim}, and surface(s) {watched!r} rendered identically across {swept.tried!r}, so no other "
            f"surface this registry names shows it{excused}{skipped}"
        ),
    )


@dataclass(frozen=True)
class _Swept:
    """What one dimension's subject view rendered at each value :func:`_sweep` put it at."""

    renders: Mapping[str, Sequence[Any]]
    """Surface → its rendering at each value tried, in order."""

    tried: Sequence[Any]
    """The values the dimension was put at and held."""

    not_held: Sequence[Any]
    """Boundary values the host does not hold beside this world, so not tried."""

    siblings_moved: frozenset[str]
    """Other dimensions whose read differed across the values tried."""


async def _sweep(
    registry: WorldRegistry, declared: WorldDimension, check: CheckName, surfaces: tuple[str, ...]
) -> _Swept | ConformanceResult:
    """Put the dimension at a distinct pair plus its boundaries and render ``surfaces`` at each.

    The one value walk both perception checks take, so a value one of them tries is a value the other
    tries: the distinct pair (two values, so at least one differs from whatever the world held) plus the
    schema's boundaries (:func:`_boundary_values`), because a summary surface answers only at an edge.

    Args:
        registry: The host's registry.
        declared: A settable, attended dimension — the caller checks.
        check: The check asking, for the verdict a failed setup records.
        surfaces: The surfaces to attach when rendering.

    Returns:
        The renderings, or the verdict that ends the check: a setup that never landed, or a dimension
        whose values the schema or the host's coherence handle cannot supply.
    """
    view = registry.subject_view
    assert view is not None, "callers have a surface to render, which the registry refuses without a subject_view"
    await _rebase(registry)
    try:
        pair = await _distinct_held_values(registry, declared, 2)
    except _SchemaTooNarrow as narrow:
        return _too_narrow(check, declared, narrow)
    except _NotHeld as unheld:
        return _not_held(check, declared, unheld)
    renders: dict[str, list[Any]] = {surface: [] for surface in surfaces}
    sibling_reads: dict[str, list[Any]] = {
        other.name: [] for other in registry.declarations if other.name != declared.name and other.read is not None
    }
    tried: list[Any] = []
    not_held: list[Any] = []
    for value in (*pair, *_boundary_values_beside(declared.schema, pair)):
        if not any(json_equal(value, generated) for generated in pair) and await _incoherence(
            registry, declared, value
        ):
            # A boundary the host does not hold beside this world is not a state a subject can be in here.
            not_held.append(value)
            continue
        if qualification := await _put(registry, declared, value):
            if any(json_equal(value, generated) for generated in pair):
                return _setup_never_landed(check, declared, value, qualification)
            # A boundary the world does not hold as itself — a model that fills an empty value with
            # its defaults — is not a state a subject can be in, so it is not tried, and is named.
            not_held.append(value)
            continue
        tried.append(value)
        rendered = await registry.call(view, surfaces=surfaces)
        for surface in surfaces:
            renders[surface].append(rendered.get(surface))
        for other in registry.declarations:
            if other.name in sibling_reads:
                assert other.read is not None
                sibling_reads[other.name].append(await registry.call(other.read))
    moved = frozenset(
        name for name, reads in sibling_reads.items() if any(not json_equal(read, reads[0]) for read in reads)
    )
    return _Swept(renders=renders, tried=tried, not_held=not_held, siblings_moved=moved)


async def _ambient_isolation(registry: WorldRegistry) -> ConformanceResult:
    """Pin every declared dimension, move the surroundings, and require the view to stay put.

    Args:
        registry: The host's registry.

    Returns:
        The registry-wide verdict.
    """
    handle = registry.perturb_ambient
    if handle is None:
        return ConformanceResult(
            check="ambient_isolation",
            outcome="unavailable",
            qualification="no_perturbation_binding",
            detail=(
                "this host binds no way to move state its world declares no dimension for, so whether the "
                "subject perceives undeclared state is unproved here and will stay unproved"
            ),
        )
    surfaces = tuple(dict.fromkeys(surface for d in registry.declarations for surface in d.perceived_by))
    if not surfaces:
        return ConformanceResult(
            check="ambient_isolation",
            outcome="unavailable",
            qualification="nothing_to_observe",
            detail=(
                "no dimension here is perceived by any surface, so there is no subject view for undeclared "
                "state to leak into and nothing for this check to watch"
            ),
        )
    if unpinnable := [d.name for d in registry.declarations if d.perceivable and _cannot_be_pinned(d)]:
        return ConformanceResult(
            check="ambient_isolation",
            outcome="unavailable",
            qualification="no_perturbation_binding",
            detail=(
                f"{', '.join(sorted(unpinnable))} is perceived and cannot be held fixed by this host, so "
                "movement in the subject view could not be attributed to declared state or to the surroundings"
            ),
        )
    await _rebase(registry)
    base = registry.base_world
    for declared in registry.declarations:
        if declared.name in base:
            continue
        try:
            pin = (await _distinct_held_values(registry, declared, 1))[0]
        except _NotHeld as unheld:
            return ConformanceResult(
                check="ambient_isolation",
                outcome="failed",
                detail=(
                    f"{declared.name} could not be held at any value the host holds beside its base world: {unheld}. "
                    "A coupling narrows which values a check draws and cannot put a declared dimension beyond holding "
                    "— correct the coherence handle or the base world"
                ),
            )
        await _apply(registry, declared, pin)
    view = registry.subject_view
    assert view is not None, "a perceived surface implies a subject_view — the registry refuses one without"
    before = await registry.call(view, surfaces=surfaces)
    moved = await registry.call(handle)
    after = await registry.call(view, surfaces=surfaces)
    # A rig may say what it moved. Saying "nothing" is a rig that proved nothing this run, which is a
    # gap and never a pass; saying nothing at all is a rig that does not report, taken at its word.
    if moved is not None and not moved:
        return ConformanceResult(
            check="ambient_isolation",
            outcome="unavailable",
            qualification="no_perturbation_binding",
            detail=(
                "this host's ambient rig reported moving nothing, so whether the subject perceives state no "
                "dimension declares is unproved here"
            ),
        )
    named = f" The rig moved: {', '.join(str(part) for part in moved)}." if moved else ""
    if before != after:
        return ConformanceResult(
            check="ambient_isolation",
            outcome="failed",
            detail=(
                "every declared dimension was held fixed, the surrounding world moved, and the subject's view "
                f"moved with it — {before!r} became {after!r}. The subject perceives state no dimension here "
                f"declares, so every run varies on it silently.{named}"
            ),
        )
    return ConformanceResult(
        check="ambient_isolation",
        outcome="passed",
        detail=(
            "the subject's view held still while state no dimension declares moved beneath it, so what the "
            f"subject perceives is accounted for by the declarations.{named}"
        ),
    )


async def _independence(registry: WorldRegistry, declared: WorldDimension) -> ConformanceResult:
    """Seed this dimension, then every other settable one, and require this one to survive.

    Exhaustive over siblings rather than sampled. The design allows sampling because full
    coverage is quadratic, and no host is near the size where that costs anything — exhaustive is
    strictly the stronger proof, so it is what runs until a registry exists that cannot afford it.

    **A sibling the host holds at nothing new beside this dimension's value is tried beside another.**
    A coupling can pin a sibling at one value of this dimension and free it at the next — a silent output
    holds no queue, a playing one holds any — and stopping at the first value would let a coherence handle
    excuse every pair it liked. A sibling the host holds at nothing new beside ANY value of this dimension
    it holds is the pair declared unable to move together, which fails: two dimensions that can never move
    together are one dimension with a structured value.

    Args:
        registry: The host's registry.
        declared: A seedable dimension.

    Returns:
        The verdict.
    """
    if unattended := _refuse_unattended("independence", declared):
        return unattended
    read = declared.read
    assert read is not None, "a seedable dimension has a read handle — the registry refuses one without"
    await _rebase(registry)
    current = await registry.call(read)
    try:
        mine = await _a_value_other_than(registry, declared, current)
    except _SchemaTooNarrow as narrow:
        return _too_narrow("independence", declared, narrow)
    except _NotHeld as not_held:
        return _not_held("independence", declared, not_held)
    if qualification := await _put(registry, declared, mine):
        return _setup_never_landed("independence", declared, mine, qualification)
    siblings = [
        other for other in registry.declarations if other.name != declared.name and not _cannot_be_pinned(other)
    ]
    # Each sibling is moved off whatever it currently holds, so a shared write path has something
    # to reveal. A sibling set to the value it already had would exonerate itself.
    #
    # A WITNESSED sibling counts, through its perturbation handle. No run moves one, but the world
    # does — that is what witnessed means — so a witnessed dimension sharing state with a seeded
    # one silently rewrites a precondition mid-campaign, which is worth more than the seedable
    # pairs a narrower reading would cover.

    moved: list[str] = []
    unmoved: list[str] = []
    coupled: list[WorldDimension] = []
    for sibling in siblings:
        held = await registry.call(sibling.read) if sibling.read is not None else None
        try:
            # Drawn beside ``mine``: a sibling value the host does not hold with it is a coupling the
            # host declared, not a clobber this check exists to find.
            target = await _a_value_other_than(registry, sibling, held)
        except _SchemaTooNarrow:
            unmoved.append(sibling.name)
            continue
        except _NotHeld:
            coupled.append(sibling)
            continue
        if clobbered := await _move_sibling(registry, declared, mine, sibling, target, moved, unmoved):
            return clobbered
    beside: dict[str, Any] = {}
    for sibling in coupled:
        outcome = await _compose_beside_another(registry, declared, (current, mine), sibling, moved, unmoved)
        if isinstance(outcome, ConformanceResult):
            return outcome
        if sibling.name in moved:
            beside[sibling.name] = outcome
    return ConformanceResult(
        check="independence",
        outcome="passed",
        dimension=declared.name,
        detail=_independence_prose(declared.name, moved, unmoved, beside),
    )


async def _move_sibling(
    registry: WorldRegistry,
    declared: WorldDimension,
    mine: Any,
    sibling: WorldDimension,
    target: Any,
    moved: list[str],
    unmoved: list[str],
) -> ConformanceResult | None:
    """Move one sibling to ``target`` and require ``declared`` to still hold ``mine``.

    Args:
        registry: The host's registry.
        declared: The dimension under test, already holding ``mine``.
        mine: What it holds.
        sibling: The sibling to move.
        target: A value the host holds for the sibling beside ``mine``.
        moved: Appended with the sibling's name when it took ``target``.
        unmoved: Appended with the sibling's name when it did not.

    Returns:
        A ``failed`` verdict naming the sibling when moving it moved ``declared``; None otherwise.
    """
    read = declared.read
    assert read is not None and sibling.read is not None, "a settable dimension has a read handle"
    await _apply(registry, sibling, target)
    # The clobber check runs for EVERY sibling, including one whose own value did not change:
    # a seeder that writes the wrong dimension moves nothing of its own, and that is precisely
    # the shared-state defect this check exists to find.
    if not json_equal(observed := await registry.call(read), mine):
        return ConformanceResult(
            check="independence",
            outcome="failed",
            dimension=declared.name,
            detail=(
                f"{declared.name} held {mine!r}, setting {sibling.name} left it {observed!r} — the two share "
                "underlying state, so a scenario presuming both preconditions is presuming a lie"
            ),
        )
    (moved if json_equal(await registry.call(sibling.read), target) else unmoved).append(sibling.name)
    return None


async def _compose_beside_another(
    registry: WorldRegistry,
    declared: WorldDimension,
    tried: tuple[Any, Any],
    sibling: WorldDimension,
    moved: list[str],
    unmoved: list[str],
) -> Any:
    """Find another value of ``declared`` the host holds a moved ``sibling`` beside, and compose the two there.

    Args:
        registry: The host's registry.
        declared: The dimension under test.
        tried: Its base value and the value the check first put it at, both already spent.
        sibling: A sibling the host held at nothing new beside the first value.
        moved: Appended with the sibling's name when it took a new value.
        unmoved: Appended with the sibling's name when its write did not land.

    Returns:
        The value of ``declared`` the sibling was composed beside, or a verdict that ends the check: ``failed``
        for a clobber or for a pair the host holds at no value of ``declared`` together, and the setup answer
        when a value of ``declared`` did not land.
    """
    read = declared.read
    assert read is not None and sibling.read is not None, "a settable dimension has a read handle"
    refusals: list[str] = []
    for candidate in _synthesize(declared.schema, _VALUE_ATTEMPTS, named=declared.name):
        if any(json_equal(candidate, spent) for spent in tried):
            continue
        await _rebase(registry)
        if await _incoherence(registry, declared, candidate):
            continue
        if qualification := await _put(registry, declared, candidate):
            return _setup_never_landed("independence", declared, candidate, qualification)
        try:
            target = await _a_value_other_than(registry, sibling, await registry.call(sibling.read))
        except (_SchemaTooNarrow, _NotHeld) as starved:
            refusals.append(str(starved))
            continue
        if clobbered := await _move_sibling(registry, declared, candidate, sibling, target, moved, unmoved):
            return clobbered
        return candidate
    return ConformanceResult(
        check="independence",
        outcome="failed",
        dimension=declared.name,
        detail=(
            f"at every value of {declared.name} the host holds over its base world, its coherence handle holds "
            f"{sibling.name} at nothing but the value it already held, so the two can never move together and "
            f"composing them proves nothing{': ' + '; '.join(dict.fromkeys(refusals)) if refusals else ''}. Two "
            "dimensions that cannot move independently are one dimension with a structured value — declare them as "
            "one, or correct the coherence handle"
        ),
    )


def _independence_prose(name: str, moved: list[str], unmoved: list[str], beside: Mapping[str, Any]) -> str:
    """What an independence pass actually established, naming only what was actually moved.

    A pass reading "survived setting A, B, C" over siblings that never moved off the value they
    already held is a claim nothing tested — the seeding proved nothing about composition, and the
    verdict said it did.

    Args:
        name: The dimension the verdict speaks for.
        moved: Siblings that took a new value.
        unmoved: Siblings that could not be moved off what they held.
        beside: Siblings composed beside a later value of ``name`` than the first, with that value — the
            host held them at nothing new beside the first.

    Returns:
        The verdict's prose.
    """
    if not moved and not unmoved:
        return f"{name} is the only settable dimension here, so nothing can compose with it"
    if not moved:
        return (
            f"no sibling of {name} could be moved off the value it already held ({', '.join(unmoved)}), so nothing "
            "here speaks for composing preconditions"
        )
    survived = f"{name} survived setting {', '.join(moved)}, so a scenario may presume it alongside them"
    if beside:
        survived += " (" + "; ".join(f"{sibling} beside {name}={value!r}" for sibling, value in beside.items()) + ")"
    if not unmoved:
        return survived
    return f"{survived}. {', '.join(unmoved)} could not be moved off the value already held, so nothing here speaks for composing with those"


def _vocabulary_completeness(registry: WorldRegistry, expressions: Sequence[str]) -> ConformanceResult:
    """Does every world path a host's scenarios read resolve to something this registry declares?

    The only check here that needs no run, no seed and no subject — it reads text and names, which
    is what lets the same resolution refuse a path at authoring time, before a subject is scored
    down for a state nobody ever set. The founding incident has a postcondition half for
    exactly that reason: a goal check reading a path nothing writes resolves to *missing*,
    comparisons against missing are false, and the run reports a failure that is really a typo.

    **A path resolving to a dimension no subject perceives passes this check**, and that is not an
    oversight. Judge-only state is legitimate — a goal check may read what the subject never saw —
    so the vocabulary is complete. Whether a *precondition* should presume such a dimension is a
    different question, asked where the two kinds of expression are still told apart.

    An expression that will not parse fails this check rather than being passed over. The defect
    is in the expression rather than in the registry, and the verdict says so; ignoring it would
    let a scenario nothing can read sit in a corpus reported conformant.

    Args:
        registry: The host's world registry.
        expressions: Scenario expressions, preconditions and goal checks alike.

    Returns:
        ``passed`` when every world path resolved; ``failed`` naming each that did not and each
        expression that would not parse; ``unavailable`` when nothing exercised the vocabulary at
        all — no expressions, or expressions that read no world state. That last is recorded
        rather than passed because "every path resolved" over no paths is vacuously true and would
        render identically to a corpus somebody actually checked.
    """
    unreadable: list[str] = []
    unresolved: list[str] = []
    resolved = 0
    for expression in expressions:
        try:
            reads = extract_paths(expression)
        except DSLError as malformed:
            unreadable.append(f"{expression!r} does not parse: {malformed}")
            continue
        for path in reads.world:
            if registry.resolve_path(path) is None:
                unresolved.append(f"{path!r} in {expression!r}")
            else:
                resolved += 1
    if unreadable or unresolved:
        return ConformanceResult(
            check="vocabulary_completeness",
            outcome="failed",
            detail=_vocabulary_prose(unresolved, unreadable),
        )
    if resolved == 0:
        return ConformanceResult(
            check="vocabulary_completeness",
            outcome="unavailable",
            detail=(
                "no expression was supplied, so nothing exercised this registry's vocabulary"
                if not expressions
                else f"the {len(expressions)} expressions supplied read no world state, so nothing "
                "exercised this registry's vocabulary"
            ),
            qualification="nothing_to_resolve",
        )
    return ConformanceResult(
        check="vocabulary_completeness",
        outcome="passed",
        detail=(
            f"every one of the {resolved} world paths across {len(expressions)} expressions "
            "resolves to a dimension this registry declares"
        ),
    )


def _vocabulary_prose(unresolved: list[str], unreadable: list[str]) -> str:
    """Say which paths reached nothing and which expressions could not be read at all.

    Both halves in one sentence rather than one verdict each, because the check has one question
    and a reader fixing a corpus wants the whole list. Naming every offender rather than the first
    is the difference between one pass over the corpus and one pass per typo.

    Args:
        unresolved: Rendered ``path in expression`` pairs that resolved to no dimension.
        unreadable: Rendered expressions the language could not parse, with the reason.

    Returns:
        One sentence a host can act on.
    """
    parts: list[str] = []
    if unresolved:
        parts.append(
            "these paths resolve to no dimension this registry declares, so a scenario reading one "
            f"presumes state nothing can set or see: {'; '.join(unresolved)}"
        )
    if unreadable:
        parts.append(f"these expressions cannot be read at all: {'; '.join(unreadable)}")
    # Joined rather than concatenated with a leading "and", so a verdict carrying only the second
    # half still reads as a sentence — an error a host acts on is read far more often in its
    # one-defect form than in the form the author had both halves of in mind.
    return "; and ".join(parts)


def _too_narrow(check: CheckName, declared: WorldDimension, narrow: _SchemaTooNarrow) -> ConformanceResult:
    """The answer for a check whose dimension cannot take the values it needed.

    An ``unavailable``, not a raise: the schema is the host's declaration, and a declared shape
    that makes a proof unreachable is exactly what this outcome is for. Raising instead would cost
    every OTHER dimension in the registry its verdict, and point whoever read the traceback at the
    synthesizer for a registration decision.

    Args:
        check: The check that could not proceed.
        declared: The dimension.
        narrow: The signal, whose message says what the schema admits.

    Returns:
        The recorded answer.
    """
    return ConformanceResult(
        check=check,
        outcome="unavailable",
        dimension=declared.name,
        qualification="schema_admits_too_few_values",
        detail=str(narrow),
    )


def _not_held(check: CheckName, declared: WorldDimension, not_held: _NotHeld) -> ConformanceResult:
    """The answer for a check whose dimension has too few values the host holds over its base world.

    ``failed``, unlike :func:`_too_narrow`'s ``unavailable``, and the difference is the whole bound on the
    coherence handle. A schema too narrow for a check is the declared shape making the proof unreachable,
    which is what ``unavailable`` means. A schema offering the values while the host's coherence handle
    holds too few of them over the base world it named is two of the host's declarations contradicting each
    other: the dimension is declared settable, and the host says it cannot hold it at the values a check
    needs in the world every check starts from. Recording that as ``unavailable`` would let a coherence
    handle switch off any check it liked, a waiver authored by the host — which the kit's one rule forbids.

    Args:
        check: The check that could not proceed.
        declared: The dimension.
        not_held: The signal, whose message carries the host's reasons.

    Returns:
        The recorded answer.
    """
    return ConformanceResult(
        check=check,
        outcome="failed",
        dimension=declared.name,
        detail=(
            f"{not_held}. That is the host's coherence handle refusing the values {check} needs over its own base "
            "world, so the declarations contradict each other: a coupling narrows which values a check draws and "
            "cannot put a declared dimension beyond proof — correct the coherence handle, or name a base world in "
            "which the dimension can move"
        ),
    )


def _setup_never_landed(
    check: CheckName,
    declared: WorldDimension,
    value: Any,
    qualification: Qualification,
) -> ConformanceResult:
    """The answer for a check whose own setup did not reach the world.

    Never ``failed``: this check never got to ask its question, and the defect that stopped it
    belongs to whichever check owns it — round-trip, for a seeding path that does not reach the
    world. Two verdicts for one defect is how a report starts blaming innocent code.

    Args:
        check: The check that could not proceed.
        declared: The dimension it was setting up.
        value: What it tried to put there.
        qualification: Why the value is not in the world.

    Returns:
        The recorded answer.
    """
    reasons = {
        "arming_only": (
            f"this host binds nothing that fires {_condition_of(declared)!r}, so the value stayed armed and "
            "never entered the world"
        ),
        "seeding_did_not_take": (
            "the seeding path did not reach the world, which the round-trip verdict for this dimension names; "
            "until it does there is nothing here to observe"
        ),
    }
    return ConformanceResult(
        check=check,
        outcome="unavailable",
        dimension=declared.name,
        qualification=qualification,
        detail=f"{declared.name} could not be put at {value!r}: {reasons[qualification]}",
    )


def _refuse_unattended(check: CheckName, declared: WorldDimension) -> ConformanceResult | None:
    """The answer for a dimension only a person can bring into being, or None.

    Nothing is attempted, deliberately: the question this row asks is whether an unattended run
    can instantiate the dimension, and the answer is no regardless of what the arming path does.

    Args:
        check: The check asking.
        declared: The dimension.

    Returns:
        The recorded answer, or None when the dimension is not human-triggered.
    """
    when = declared.when
    if not isinstance(when, Triggered) or when.kind != "human":
        return None
    return ConformanceResult(
        check=check,
        outcome="unavailable",
        dimension=declared.name,
        qualification="not_instantiable_unattended",
        detail=(
            f"{declared.name} arrives only when a person does something ({when.condition!r}), so no unattended "
            "run can put a subject in this state. An answer about what is representable, not a failed claim"
        ),
    )


def _cannot_be_pinned(declared: WorldDimension) -> bool:
    """Whether the kit can put this dimension at a value of its choosing and leave it there.

    Args:
        declared: The dimension.

    Returns:
        True when nothing can set it, or when setting it only arms a condition nothing fires.
    """
    if not declared.settable:
        return True
    when = declared.when
    return isinstance(when, Triggered) and when.fire is None


def _condition_of(declared: WorldDimension) -> str:
    """The host's word for what fires this dimension, for a message about an unfired one.

    Args:
        declared: A triggered dimension.

    Returns:
        The condition.
    """
    when = declared.when
    return when.condition if isinstance(when, Triggered) else ""


async def _rebase(registry: WorldRegistry) -> None:
    """Put every dimension the host's base world names at its base value, so a check starts from that world.

    Believed rather than verified: a base value that does not land is the round trip's to report for that
    dimension, and it does.

    Args:
        registry: The host's registry.
    """
    for name, value in registry.base_world.items():
        declared = registry.get(name)
        assert declared is not None, "the registry refuses a base world naming an undeclared dimension"
        await _apply(registry, declared, value)


async def _incoherence(registry: WorldRegistry, declared: WorldDimension, value: Any) -> list[str]:
    """Why the host would not hold ``value`` for this dimension beside what every other one holds now.

    The world composed is what each sibling's READ answers, so it is what the host holds rather than what
    the kit last asked for.

    Args:
        registry: The host's registry.
        declared: The dimension the value is for.
        value: The candidate.

    Returns:
        The host's reasons; empty when it holds that world, or when it declares no coherence handle.
    """
    handle = registry.coherence
    if handle is None:
        return []
    world = {
        other.name: await registry.call(other.read)
        for other in registry.declarations
        if other.name != declared.name and other.read is not None
    }
    world[declared.name] = value
    return [str(reason) for reason in await registry.call(handle, world)]


class _Applied(NamedTuple):
    """What putting a dimension at a value came to."""

    #: ``arming_only`` when the value was armed and this host fires nothing; None when the host's handles
    #: claim the dimension now holds it.
    qualification: Qualification | None
    #: What the write handle returned — for a triggered dimension's seed, the identity of the event it armed.
    written: Any


async def _apply(registry: WorldRegistry, declared: WorldDimension, value: Any) -> _Applied:
    """Put the dimension at ``value`` through the host's own handles, believing the result.

    Args:
        registry: The host's registry.
        declared: The dimension. Must be settable — a caller checks first.
        value: A schema-valid value.

    Returns:
        Whether it holds the value or was only armed, and what the write handle returned.
    """
    handle = declared.write_handle
    assert handle is not None, "callers check settability before instantiating"
    written = await registry.call(handle, value)
    when = declared.when
    if not isinstance(when, Triggered):
        return _Applied(None, written)
    if when.fire is None:
        return _Applied("arming_only", written)
    await registry.call(when.fire)
    return _Applied(None, written)


async def _put(registry: WorldRegistry, declared: WorldDimension, value: Any) -> Qualification | None:
    """Put the dimension at ``value`` and confirm it landed, so a later verdict is about the check.

    The confirmation is what stops one defect producing three findings that blame three different
    things. Round-trip deliberately does NOT use this — verifying the seed is the question it is
    asking, and it must report ``failed`` where this reports a setup that never got started.

    Args:
        registry: The host's registry.
        declared: The dimension. Must be settable — a caller checks first.
        value: A schema-valid value.

    Returns:
        ``arming_only`` or ``seeding_did_not_take`` when the value is not in the world; None when
        it is.
    """
    if qualification := (await _apply(registry, declared, value)).qualification:
        return qualification
    read = declared.read
    assert read is not None, "a settable dimension has a read handle — the registry refuses seed and perturb without"
    return None if json_equal(await registry.call(read), value) else "seeding_did_not_take"


async def _distinct_held_values(registry: WorldRegistry, declared: WorldDimension, count: int) -> tuple[Any, ...]:
    """Exactly ``count`` mutually distinct schema-valid values the host holds beside the world as it stands.

    The first ``count`` the schema offers, in the schema's own order, when the host holds them all — which
    is every value on a host declaring no coherence. Values it does not hold are passed over for the next
    the schema offers, up to :data:`_VALUE_ATTEMPTS`.

    Args:
        registry: The host's registry.
        declared: The dimension.
        count: How many are needed.

    Returns:
        The values.

    Raises:
        _SchemaTooNarrow: The schema does not admit that many. A host declaration, so a check
            records it rather than letting it escape.
        _NotHeld: The schema admits that many and the host holds fewer of them here.
        WorldConformanceError: The kit cannot read the schema. An engine gap.
    """
    offered = _synthesize(declared.schema, count, named=declared.name)
    if len(offered) < count:
        raise _SchemaTooNarrow(
            f"{declared.name}'s schema ({declared.schema!r}) admits {len(offered)} distinct value(s) and this check needs {count}"
        )
    for further in _synthesize(declared.schema, _VALUE_ATTEMPTS, named=declared.name):
        if not any(json_equal(further, held) for held in offered):
            offered.append(further)
    values: list[Any] = []
    refusals: list[str] = []
    for candidate in offered:
        if reasons := await _incoherence(registry, declared, candidate):
            refusals.extend(reasons)
            continue
        values.append(candidate)
        if len(values) == count:
            return tuple(values)
    raise _NotHeld(
        f"{declared.name} needs {count} distinct value(s) and the host holds {len(values)} of the {len(offered)} its "
        f"schema offered beside the world as it stands: {'; '.join(dict.fromkeys(refusals))}"
    )


def _boundary_values_beside(schema: Mapping[str, Any], pair: Sequence[Any]) -> list[Any]:
    """The schema's boundary values that are not already in ``pair``, in order.

    Args:
        schema: The dimension's schema.
        pair: The generated values already being tried.

    Returns:
        The boundary values to try as well.
    """
    beside: list[Any] = []
    for boundary in _boundary_values(schema):
        if not any(json_equal(boundary, held) for held in (*pair, *beside)):
            beside.append(boundary)
    return beside


def _boundary_values(schema: Mapping[str, Any]) -> list[Any]:
    """The values at the edges of what a schema admits — the empty value, the minimum, the maximum.

    Boundary-value analysis, applied because a surface that summarises a dimension rather than
    rendering it — a status, a count, a present/absent line — typically answers only at an edge:
    empty versus some, below versus above a bound. Only edges the schema itself admits are
    returned, so every value here is one a run could seed.

    Args:
        schema: The dimension's schema.

    Returns:
        The boundary values, possibly none (an ``enum`` or ``const`` is already exhaustive, and a
        boolean already yields both values). An ``anyOf`` returns every branch's, in branch order.
    """
    # Never raises here: the pair beside these values was synthesized from the same schema first.
    kind = _generator_for(schema, named="the dimension's schema")
    if kind == "anyOf":
        # Each shape's own edges: a summary surface that answers only at an edge answers at the edge of
        # whichever shape the value took, so every branch's edges are values a run could seed.
        edges: list[Any] = []
        for branch in schema["anyOf"]:
            edges.extend(edge for edge in _boundary_values(branch) if not any(json_equal(edge, held) for held in edges))
        return edges
    if kind == "object":
        return [] if schema.get("required") else [{}]
    if kind == "array":
        min_items = int(schema.get("minItems", 0))
        return [[]] if min_items == 0 else []
    if kind == "string":
        return [""] if int(schema.get("minLength", 0)) == 0 else []
    if kind in {"integer", "number"}:
        edges = [schema[bound] for bound in ("minimum", "maximum") if bound in schema]
        low, high = schema.get("minimum"), schema.get("maximum")
        if (low is None or low <= 0) and (high is None or high >= 0):
            edges.append(0)
        return edges
    return []


async def _a_value_other_than(registry: WorldRegistry, declared: WorldDimension, current: Any) -> Any:
    """A schema-valid value the dimension does not already hold, and the host holds beside the world as it stands.

    The reason this exists rather than "the first value the schema offers": a seeder wired to
    nothing passes a round trip whenever that first value happens to be what the world already
    held, and an empty list or a zero is exactly the value a schema offers first.

    Args:
        registry: The host's registry, whose coherence handle (if any) says which values it holds here.
        declared: The dimension.
        current: What it holds now.

    Returns:
        A value that compares unequal to ``current``.

    Raises:
        _SchemaTooNarrow: Every value the schema admits is the one already there, so seeding
            could not be told apart from doing nothing.
        _NotHeld: The schema offers other values and the host holds none of them beside the world.
        WorldConformanceError: The kit cannot read the schema. An engine gap.
    """
    refusals: list[str] = []
    for candidate in _synthesize(declared.schema, _VALUE_ATTEMPTS, named=declared.name):
        if json_equal(candidate, current):
            continue
        if reasons := await _incoherence(registry, declared, candidate):
            refusals.extend(reasons)
            continue
        return candidate
    if refusals:
        raise _NotHeld(
            f"{declared.name} holds {current!r}, and the host holds none of the other values its schema offered beside "
            f"the world as it stands: {'; '.join(dict.fromkeys(refusals))}"
        )
    raise _SchemaTooNarrow(
        f"{declared.name} already holds every value its schema ({declared.schema!r}) admits, so seeding it "
        "could not be distinguished from doing nothing"
    )


def _synthesize(schema: Mapping[str, Any], wanted: int, *, named: str) -> list[Any]:
    """Up to ``wanted`` distinct values drawn from a JSON Schema.

    Args:
        schema: The schema, which may be a dimension's own or an ``items`` schema nested in one.
        wanted: How many to try for.
        named: The dimension this schema belongs to, for error messages — a host reading one needs
            to know which registration to open, and a nested schema cannot say on its own.

    Returns:
        The distinct values found, in a deterministic order, possibly fewer than ``wanted``.

    Raises:
        WorldConformanceError: The schema uses a construct the kit cannot synthesize from. An
            ENGINE gap — extend the generators — never a property of the host.
    """
    kind = _generator_for(schema, named=named)
    if kind == "anyOf":
        candidates: Sequence[Any] = _interleaved(schema["anyOf"], wanted, named=named)
    elif kind == "enum":
        candidates = list(schema["enum"])
    elif kind == "const":
        candidates = [schema["const"]]
    else:
        candidates = _by_type(schema, kind, wanted, named=named)
    distinct: list[Any] = []
    for candidate in candidates:
        if not any(json_equal(candidate, held) for held in distinct):
            distinct.append(candidate)
        if len(distinct) == wanted:
            break
    return distinct


def _interleaved(branches: Sequence[Mapping[str, Any]], wanted: int, *, named: str) -> list[Any]:
    """Each ``anyOf`` branch's values taken in turn — the first of every branch, then the second of every branch.

    In turn rather than branch after branch, because a caller wanting two values from a schema offering two
    shapes must get one of each: drawn in sequence, a first branch admitting many values would supply them
    all and the second shape would never be tried, so a host that renders only one of them would pass. The
    order is the branches' own, so the result is deterministic. Distinctness is left to :func:`_synthesize`,
    which deduplicates under the seed check's equality.

    Args:
        branches: The ``anyOf`` list, already audited by :func:`_generator_for`.
        wanted: How many values each branch is asked for.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        Candidates, not yet deduplicated.
    """
    per_branch = [_synthesize(branch, wanted, named=named) for branch in branches]
    return [values[index] for index in range(wanted) for values in per_branch if index < len(values)]


def _generator_for(schema: Mapping[str, Any], *, named: str) -> str:
    """Which generator will serve this schema, from the one resolution the seed check also reads.

    :func:`~threetears.evals.contracts.host.world_schema.honoured_kind` decides it — ``enum`` and ``const``
    winning over ``type``, a union or an absent type refused, every keyword audited against the set the
    resolved generator honours, nested schemas included. It lives there rather than here so that a schema
    this kit synthesizes from is exactly a schema a seeded value can be checked against: the two readers
    once each kept their own copy and disagreed on a typeless schema and on a union.

    Args:
        schema: The schema.
        named: The dimension it belongs to, for the error.

    Returns:
        A key of :data:`~threetears.evals.contracts.host.world_schema.HONOURED_KEYWORDS`.

    Raises:
        WorldConformanceError: The schema is outside the honoured subset. Ignoring any part of it would
            seed a value the host's own schema forbids and pass a round trip over it, so it is loud — and
            it is an ENGINE gap, never a property of the host.
    """
    try:
        return honoured_kind(schema, at=named)
    except UnsupportedSchemaError as gap:
        raise WorldConformanceError(
            f"{gap}. That is a gap in the kit, not a property of the host: extend the readers in world_schema "
            "rather than recording the dimension unproved, or declare the schema in terms they honour"
        ) from gap


def _by_type(schema: Mapping[str, Any], kind: str, wanted: int, *, named: str) -> list[Any]:
    """Candidate values for a schema that declares a type rather than enumerating values.

    Args:
        schema: The schema.
        kind: The type generator to use, already resolved and audited by :func:`_synthesize`.
        wanted: How many candidates to produce.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        Candidates, not yet deduplicated.

    Raises:
        WorldConformanceError: The schema's bounds contradict each other.
    """
    if kind == "boolean":
        return [False, True]
    if kind in {"integer", "number"}:
        return _numbers(schema, kind, wanted, named=named)
    if kind == "string":
        return _strings(schema, wanted, named=named)
    if kind == "object":
        return _objects(schema, wanted, named=named)
    return _arrays(schema, wanted, named=named)


def _numbers(schema: Mapping[str, Any], kind: str, wanted: int, *, named: str) -> list[Any]:
    """Ascending numbers inside a numeric schema's bounds.

    Neither bound is assumed. A schema with only a ``maximum`` of -1 admits every integer below it,
    and defaulting the floor to 0 made that read as bounds admitting no value — the kit's own
    limit stated as a fact about the host's registration. A bounded ``number`` divides its range
    rather than stepping by 1, since a range narrower than 1 is a real declaration and not an
    empty one.

    Args:
        schema: The schema.
        kind: ``integer`` or ``number``.
        wanted: How many to produce.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        The numbers, possibly fewer than ``wanted`` where the bounds allow fewer.

    Raises:
        WorldConformanceError: The bounds contradict each other. Registration refuses this for
            every schema it can see, so reaching it means one arrived from outside a registry.
    """
    low = schema.get("minimum")
    high = schema.get("maximum")
    if low is not None and high is not None and high < low:
        raise WorldConformanceError(f"{named}'s schema bounds admit no value: {schema!r}")
    if kind == "integer":
        int_floor = int(low) if low is not None else (0 if high is None or high >= 0 else int(high) - wanted + 1)
        return [value for value in (int_floor + step for step in range(wanted)) if high is None or value <= high]
    floor = float(low) if low is not None else (0.0 if high is None or high >= 0 else float(high) - wanted)
    if high is None:
        return [floor + step for step in range(wanted)]
    step_size = (float(high) - floor) / wanted
    return [floor] if step_size <= 0 else [floor + step_size * index for index in range(wanted)]


def _strings(schema: Mapping[str, Any], wanted: int, *, named: str) -> list[str]:
    """Distinct strings inside a string schema's length bounds.

    One repeated letter per value, so length is whatever the schema's floor demands and every value
    stays distinct from every other.

    Args:
        schema: The schema.
        wanted: How many to produce.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        The strings.

    Raises:
        WorldConformanceError: The length bounds contradict each other (refused at registration for
            every schema a registry can see), or more values are wanted than the alphabet supplies.
    """
    length = max(1, int(schema.get("minLength", 1)))
    longest = schema.get("maxLength")
    if longest is not None and longest < length:
        raise WorldConformanceError(f"{named}'s schema length bounds admit no value: {schema!r}")
    if wanted > len(_ALPHABET):
        raise WorldConformanceError(
            f"{named} needs {wanted} distinct strings and this kit synthesizes at most {len(_ALPHABET)}"
        )
    return [letter * length for letter in _ALPHABET[:wanted]]


def _objects(schema: Mapping[str, Any], wanted: int, *, named: str) -> list[dict[str, Any]]:
    """Objects that differ in exactly one property, the rest held at a fixed value.

    **Every declared property is populated, not only the required ones.** A partial object would
    exercise less of whatever consumes it, and the consumer is what a round trip is trying to
    reach — an element carrying one field cannot tell a renderer that reads five from one that
    reads one. The declared set is also what a host schema means: an optional property it took
    the trouble to describe is a property something reads.

    **One property varies and the rest do not**, because distinctness is the only thing the
    callers need and varying everything at once would make a failure say nothing about which
    field carried it. The varying property is the first declared one that can supply more than a
    single value, so a schema whose first field is a ``const`` still produces distinct objects
    through whichever field can.

    ``additionalProperties`` is honoured by construction rather than by inspection: this
    generator emits declared properties and nothing else, so it satisfies the keyword at either
    setting. It is listed as honoured for that reason, not ignored.

    Args:
        schema: The object schema.
        wanted: How many to produce.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        The objects, possibly fewer than ``wanted`` where no property offers that many values.

    Raises:
        WorldConformanceError: A property schema admits no value. A property that is not a schema,
            or a ``required`` name no property describes, was already refused by
            :func:`_generator_for`, which audits the whole tree before any generator runs.
    """
    properties = schema.get("properties")
    if not properties:
        # An object that describes no properties admits exactly one value this kit can build, and
        # that is a real answer rather than a gap: {} satisfies the schema. Callers wanting two
        # distinct values get one and report the schema as too narrow, which is true of it.
        return [{}]
    base: dict[str, Any] = {}
    for name, sub in properties.items():
        first = _synthesize(sub, 1, named=named)
        if not first:
            raise WorldConformanceError(f"{named}'s property {name!r} declares a schema admitting no value: {sub!r}")
        base[name] = first[0]
    for name, sub in properties.items():
        values = _synthesize(sub, wanted, named=named)
        if len(values) > 1:
            return [{**base, name: value} for value in values]
    return [base]


def _arrays(schema: Mapping[str, Any], wanted: int, *, named: str) -> list[list[Any]]:
    """Arrays of ascending length, each element drawn from the schema's ``items``.

    Length is what varies, because it is the one axis every array schema has — an ``items`` schema
    may admit a single value and the arrays must still differ from one another.

    **The elements of one array differ from each other** wherever the ``items`` schema offers that
    many values, and repeat in turn only past that. One element repeated exercises one element, and a
    host whose elements are keyed — a queue of items under ids — cannot hold the same element twice,
    so a repeated element would be a world the host refuses rather than one it was asked to hold.

    **Every non-empty array holds one element of every shape its ``items`` admits**, where the
    ``items`` schema is an ``anyOf``: the first non-empty length is raised to the number of shapes.
    A list of shapes is perceived through each shape's own surface — a queue of jobs and
    notices renders the notices in a section of their own — and an array of one element would only ever
    carry the first shape, so a surface reading a later one would render its empty state on every
    value tried and could never be declared. ``maxItems`` still bounds the length, so a schema
    admitting fewer elements than shapes covers as many as it admits. The elements come from
    :func:`_array_elements`, which keeps their keys distinct across shapes.

    Args:
        schema: The schema.
        wanted: How many to produce.
        named: The dimension this schema belongs to, for error messages.

    Returns:
        The arrays.

    Raises:
        WorldConformanceError: The bounds contradict each other, or the schema needs elements and
            declares no ``items`` to draw them from.
    """
    shortest = int(schema.get("minItems", 0))
    longest = schema.get("maxItems")
    if longest is not None and longest < shortest:
        raise WorldConformanceError(f"{named}'s schema length bounds admit no array: {schema!r}")
    items = schema.get("items")
    shapes = _item_shapes(items, named=named) if isinstance(items, Mapping) else 1
    if longest is not None:
        shapes = min(shapes, longest)
    # The first length past the floor covers every shape; each later one is one longer, so the arrays stay distinct.
    lengths = [shortest, *(max(shortest + step, shapes + step - 1) for step in range(1, wanted))]
    if longest is not None:
        lengths = [length for length in lengths if length <= longest]
    if max(lengths) == 0:
        return [[]]
    if not isinstance(items, Mapping):
        raise WorldConformanceError(
            f"{named} declares an array that must hold elements and no 'items' schema to draw them from: "
            f"{schema!r}. Extend the synthesizer or declare the element shape"
        )
    elements = _array_elements(items, max(lengths), named=named)
    if not elements:
        raise WorldConformanceError(f"{named}'s 'items' schema admits no value: {items!r}")
    return [[elements[index % len(elements)] for index in range(length)] for length in lengths]


def _item_shapes(items: Mapping[str, Any], *, named: str) -> int:
    """How many element shapes an ``items`` schema admits: one per ``anyOf`` branch, else one.

    Args:
        items: The array's ``items`` schema.
        named: The dimension it belongs to, for error messages.

    Returns:
        The number of shapes an array must hold one of each to exercise them all.
    """
    return len(items["anyOf"]) if _generator_for(items, named=named) == "anyOf" else 1


def _array_elements(items: Mapping[str, Any], wanted: int, *, named: str) -> list[Any]:
    """Up to ``wanted`` elements for one array, cycling through the item shapes with distinct values.

    For a single shape, the shape's own distinct values. For an ``anyOf``, element ``i`` is shape
    ``i mod shapes`` — so the first ``shapes`` elements are one of each — and takes that shape's ``i``-th
    value rather than its first. The difference matters to a keyed host: each shape synthesizes its
    values from the same generators, so every shape's FIRST object carries the same key, and an array
    of first values would hold one key under every shape — a world such a host refuses. Taking the
    ``i``-th keeps the varied field distinct along the array. A shape that offers fewer values repeats
    its last.

    Args:
        items: The array's ``items`` schema.
        wanted: How many elements the longest array needs.
        named: The dimension it belongs to, for error messages.

    Returns:
        The elements, not deduplicated beyond what each shape supplies.
    """
    if _generator_for(items, named=named) != "anyOf":
        return _synthesize(items, wanted, named=named)
    per_shape = [_synthesize(branch, wanted, named=named) for branch in items["anyOf"]]
    elements: list[Any] = []
    for index in range(wanted):
        values = per_shape[index % len(per_shape)]
        if values:
            elements.append(values[min(index, len(values) - 1)])
    return elements


__all__ = [
    "CheckName",
    "ConformanceResult",
    "ObligationRow",
    "Outcome",
    "Qualification",
    "WorldConformanceError",
    "WorldConformanceReport",
    "check_world_conformance",
    "obligation_rows",
    "obligations",
]
