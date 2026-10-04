"""What a run swept, what held it, and what merely names it — declared by the host.

Two surfaces need the same list and must never drift apart. A **bisection** diffs two runs to
explain why a score moved; a **confound scan** asks which of these varied *inside* a cohort
being compared on one lever, because anything that did is a rival explanation for the movement.
Both questions are "which of the score-determining inputs differ across these runs", asked over
two and over N.

They were once one list authored twice, and the second copy was missing entries. **Each input
carries its own reader**, so a name with no reader — or a reader for a name nobody declared —
is not expressible, and the drift cannot be reintroduced by adding one and forgetting the other.

**One module, one lifetime.** What is declared here is a **runtime registration**: what an input
is, who owns it, and how to read it. One *level* of an input — the thing that crosses storage, and
whose every field is a persisted-format commitment — is :class:`~threetears.evals.contracts.host.values.
SweepableValue`, in :mod:`threetears.evals.contracts.host.values`. Both names are re-exported here so a host
reads its whole vocabulary from one import.

**The list is the host's; the algebra is the engine's.** This module ships the *classification*
(``lever`` / ``apparatus`` / ``label``), the same/differs/unknown algebra, and the registration
rules. It does not ship the inputs. :data:`SHARED_CORE` holds only what every LLM product has —
models, a judge, a simulator, a cost ceiling — and everything else arrives from a host adapter.
The test is not "is this product-specific" but *would a second product have this concept under a
different name, or not at all?*

**Roles say who owns each input.** A ``lever`` is a knob a campaign deliberately sweeps. An
``apparatus`` input is the measuring rig: it is not supposed to move, and when it does, the
comparison silently stopped being about the lever. A ``label`` is neither — it identifies rather
than determines, so a change in it is reportable in a diff and is never a confound. Only
``apparatus`` inputs carry :attr:`~Sweepable.confounds` prose, because they are the only ones a
confound scan speaks for.

**The prose is the point, not decoration.** A bare dimension name is a label; the reason is what
lets a reader judge whether it matters for the measure in front of them. Two runs on different
templates are two different rubrics over two different case sets, so one composite is not the
other composite — while a purely mechanical measure like latency means the same thing on both
sides and stays worth comparing. That judgement belongs to the reader, and it needs the reason
to make it. An eval system whose inputs are opaque names degrades to "component 3 moved" and
cannot write a readable analysis, which is why :attr:`~Sweepable.reader_prose` is required on
every declaration and :attr:`~Sweepable.confounds` on every apparatus one.

**Bounded claim: this is the vocabulary those two surfaces share, not every input that
determines a score.** :func:`threetears.evals.contracts.identity.compute_context_components` enumerates the
measurement context for a different purpose — per-component hashes, so a comparability mismatch
can badge *which* condition moved — and it disagrees with this list in both directions, most
notably by covering the frozen case set and the resolved world seed, which this one cannot
see. Reconciling them is its own piece of work; what must not happen is a reader taking the
sentence above as "these are all of them".
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.contracts.host.attribution import HostAttributed
from threetears.evals.contracts.host.values import IntervalScale, NominalScale, OrdinalScale, Scale, SweepableValue

if TYPE_CHECKING:
    from threetears.evals.contracts.models import EvalResult, EvalRun

SweepableRole = Literal["lever", "apparatus", "label"]

#: The three outcomes of comparing one input across a set of runs. Three, not two: an input
#: nobody recorded is neither agreement nor difference, and forcing it into either manufactures
#: an observation.
Comparability = Literal["same", "differs", "unknown"]

#: Reader signature. A reader is host code the engine calls and never inspects.
SweepableReader = Callable[["EvalRun", "Sequence[EvalResult]"], Any]

#: Whether one bare name is a member of an open family, asked with no run in hand. Host code the
#: engine calls and never inspects, for the same reason :data:`SweepableReader` is: the member
#: keyspace is the host's and the engine has no way to tell one of its names from a typo.
FamilyMemberTest = Callable[[str], bool]

#: The surface an open family resolves into, with some of its members taken back out. Called with
#: the run, its results, and the member names to remove; returns JSON-safe content two runs can be
#: compared on, or ``None`` when the run does not carry enough of the surface to say. Host code
#: the engine calls and never inspects, because only the host knows how a member name maps onto
#: the surface it was written into.
ResidualReader = Callable[["EvalRun", "Sequence[EvalResult]", frozenset[str]], Any]

#: The one name the candidate model answers to, everywhere — a DECLARED COORDINATE of every
#: observation (``ScoreRecord.model``) as well as the core lever below, which is why it is the
#: only lever a reporting lens resolves off the observation rather than off the run.
#:
#: It lives here, beside the declaration that carries it, so the literal has ONE owner: it was
#: previously spelled in ``host/profile.py`` while the core declared the same knob as ``models``,
#: and the two names for one lever are what made a coverage row and a variant coordinate
#: unmatchable. :mod:`threetears.evals.contracts.host.profile` re-exports it, since
#: that is where the engine's reserved name and a host's :attr:`HostProfile.observed_model_levers`
#: meet and a host claiming it is refused.
CANDIDATE_MODEL_LEVER = "model"

#: The name the candidate KIND answers to: the other half of what a run ran, beside its model. A
#: core lever the engine resolves for every run, so two runs of different kinds are two variants
#: however alike their models and overlays — a router and a game master on one model are not one
#: arm, and nothing a host writes can make them pool as one.
CANDIDATE_KIND_LEVER = "candidate_kind"


@dataclass(frozen=True)
class Sweepable:
    """One score-determining input: what it is called, who owns it, how to read it.

    A plain frozen dataclass rather than a Pydantic model on purpose: this carries a live
    :attr:`read` callable and is a **runtime** registration, never a stored or wire type. Its
    counterpart :class:`~threetears.evals.contracts.host.values.SweepableValue` — one *level* of this input, and
    the thing that crosses storage — is a Pydantic model for the same reason inverted, and lives in
    its own module because the two have opposite lifetimes and only one of them needs format
    review when it changes. It is re-exported here, since a host reads both from one place.
    """

    name: str
    """Stable dimension name — the key both readers report under."""

    role: SweepableRole
    """``lever`` (a swept knob) | ``apparatus`` (the rig) | ``label`` (identifies only)."""

    read: SweepableReader
    """Extract this input's value, JSON-safe and order-independent (sets come out sorted).

    **What one call answers is the RUN-level projection of this input** — what the run carried,
    which for an input that varies WITHIN a run is the set of levels it carried. That is the
    bisection's question ("did these two runs carry the same judges"), and it is not a second
    source of the lever: :data:`~threetears.evals.contracts.host.profile.VariantLeverReader`
    resolves the same name into the variant key's pre-image, and the two must agree about the NAME.

    On an :attr:`open_family` declaration this returns ``{member name: level}`` instead, and the
    members are what every lens sees.
    """

    reader_prose: str
    """What this input is, in words a reader of the analysis can act on. Required on every role.

    Not the same sentence as :attr:`confounds`, which says why a *change* clouds a comparison.
    This one says what the dimension is at all, and it is what keeps a memo from reading
    "component 3 moved".
    """

    confounds: str | None = None
    """Why a change in this input clouds a comparison. Required for ``apparatus``, absent otherwise."""

    indeterminate_when_blank: bool = False
    """True when a blank value means 'never recorded' rather than a value — see :func:`SweepableRegistry.is_indeterminate`."""

    result_level: bool = False
    """True when the value is observed ACROSS a run's results rather than declared on the run.

    Descriptive, not a rule — it tells a reader that omitting ``results`` makes the value
    meaningless. Undecidability is decided by :attr:`indeterminate_when_blank` alone.
    """

    no_own_coordinate: str | None = None
    """Why this ``lever`` contributes no coordinate of its own to the variant key.

    Every registered lever is expected to appear in the map a host resolves per observation, and
    :func:`~threetears.evals.contracts.identity.derive_variant_identity` checks both directions against this
    field. A lever may legitimately be absent — a request-shaped lever whose effect is wholly
    carried by the resolved configuration it produces determines identity through that, and putting the request in
    beside the resolution would split two runs that resolved alike — but the exception has to be
    **data**, not a comment. While it lived in a comment, the next lever whose reader was simply
    forgotten was indistinguishable from the intended case, and the result is a wrong merge: two
    genuinely different variants sharing a key, which nothing downstream can undo.

    Required on ``lever`` only; a value here on any other role is a contradiction, since apparatus
    and label inputs never carry a coordinate to begin with.
    """

    open_family: str | None = None
    """Why this ``lever``'s members cannot be enumerated at registration.

    A declaration carrying this is an **open family**: its :attr:`read` returns
    ``{member name: level}`` rather than one level, and the MEMBERS are the levers. The family's
    own name is a container — it is excluded from :attr:`SweepableRegistry.lever_names`, carries
    no variant coordinate, and exists only so the expansion has somewhere to be declared and
    validated. The rule's other half — it earns no coverage row — is the same fact rather than
    a second one: no resolution emits the container, so the only door left open was a campaign
    DECLARING it, and a declared axis earns a row whether or not anything resolved it.

    **That door is closed by :meth:`SweepableRegistry.refuse_as_axis`**, which is what
    :meth:`~threetears.evals.contracts.host.profile.HostProfile.controllable` decides with: a campaign naming
    the CONTAINER as its axis is refused and pointed at the members, and a member the family's own
    :attr:`owns_member` recognises is admitted. Both halves were wrong in opposite directions
    while the gate resolved through ``get`` and saw an ordinary ``role == "lever"`` declaration.
    **A campaign that persisted a container as its axis before the gate closed still carries one**
    — the gate runs at create and update, and rewriting stored declarations is deliberately not
    part of this.

    This is how the generic-engine norm serves a host whose lever
    set is open by construction: the host declares the family, and a member reaches every lens
    under the name the host gave it. A dotted member name like ``search.model`` is
    therefore a lever NAME, not a carrier path that leaked into reporting — which is what keeps
    "zero per-key UI work" a mechanism rather than a lens reading one host's fields by hand.

    Prose rather than a bool for the reason :attr:`no_own_coordinate` is: a family is how a host
    stops the engine from checking its member names, so saying why has to cost a sentence. A
    declaration whose members ARE enumerable declares them one by one instead, and gets the
    registration checks that come with a name.
    """

    owns_member: FamilyMemberTest | None = None
    """Whether one bare name is a member of this family, asked at authoring with no run in hand.

    The authoring gate has a name and no run, and :attr:`read` needs a run — so without this the
    engine has nothing to ask, and a first-class member is indistinguishable from a misspelling.
    Declaring it is what lets :meth:`SweepableRegistry.refuse_as_axis` ADMIT a member, which is
    the half of the open-family rule a gate can act on.

    **Optional, and what its absence means is a refusal rather than a waiver.** A family that
    declares none recognises nothing, so every ad-hoc member of it is refused as an unregistered
    dimension — the engine declining to guess rather than opening the gate. That is the same
    choice :attr:`confounds` makes on apparatus: the host states the thing, or the thing is not
    stated. :attr:`SweepableRegistry.axis_remedy` says so in the message, so the refusal points at
    the host's registration rather than reading as "no such knob".

    Valid on an :attr:`open_family` declaration only. On anything else it is a contradiction —
    a fixed declaration's members are its own name — and is refused at registration.
    """

    resolves_into: str | None = None
    """The fixed lever this family's members are WRITTEN INTO, when the host declares one as well.

    A host can register one change twice without meaning to: once as the member a launch named
    (``search.max_rounds``) and once as the resolved surface that member was merged into (the
    subject's whole tool configuration, levelled by content hash). Both are honest levers — the
    member is what a campaign swept, the surface is what would ship — and both move whenever the
    member does, so an engine that reads them as two levers reports every single-key arm as having
    moved two things and every such comparison as confounded by itself.

    Naming the surface here is what lets the engine tell that apart from a genuine second change:
    it asks :attr:`read_residual` for the surface with the swept members taken out, and only where
    those residuals AGREE across a cohort is the surface's movement the members' movement seen
    again. Where they disagree, something besides the swept knobs changed and the surface stays a
    lever of its own. Nothing is folded by inference: an unreadable residual folds nothing.

    Must name a ``lever`` this registry declares that is not itself an open family, and may be
    named by at most one family — a surface two families write into would need both families'
    members removed at once, which is a shape no host has and the engine does not guess.
    """

    read_residual: ResidualReader | None = None
    """Read :attr:`resolves_into`'s surface off a run with the given member names removed.

    Required exactly when :attr:`resolves_into` is set. The engine compares what this returns
    across a cohort and never inspects it, so any JSON-safe content will do provided two runs
    whose surfaces differ only in the removed members return equal values. ``None`` means the run
    does not carry the surface at all — a run older than the host's capture of it — and is read as
    "cannot say", never as agreement.
    """


@dataclass(frozen=True)
class RolePins:
    """One pinned ROLE — who filled a seat in the rig, and the inputs that identify them.

    The engine owns two roles because it declares the core dimensions pinned into them
    (:data:`CORE_ROLES`); it does not own the **keyspace**. A host whose rig has a role the core
    never heard of — an adjudication pool, a retrieval index, a fixture generator — declares it
    here and reaches every role disclosure under its own name, with no per-role edit anywhere in
    the engine. Ratified on the ground that most consuming products are dynamic and
    not enumerable in advance; it is the same rule :attr:`Sweepable.open_family` makes about
    levers, one axis over.

    **The engine owns the frame; the host owns the phrase.** A role disclosure is rendered in two
    fixed sentences — *"These runs were NOT {subject_phrase}"* and *"Cannot say whether these
    runs were {subject_phrase}"* — so the engine never composes English out of a role NAME, which
    is the thing it has no right to do with a host's vocabulary. What it needs is one phrase that
    fits that frame, and the host is its source, exactly as it is for
    :attr:`Sweepable.reader_prose` and :attr:`Sweepable.confounds`. A blank one is refused where
    it is written.
    """

    name: str
    """Stable role name — the key a disclosure groups under. Never rendered to a reader."""

    pins: tuple[str, ...] = ()
    """The declared apparatus inputs that say who filled this role.

    Each must be declared by the same registry and carry ``role == "apparatus"``; selecting by
    name is the one way the name-and-reader pairing can still be broken. A pin belongs to exactly
    one role — two roles claiming it would print one difference under two headings.
    """

    subject_phrase: str | None = None
    """The host's words for what the runs shared, in the frame above — e.g. ``"judged the same
    way"``.

    Required when this entry INTRODUCES a role, and omitted when it adds pins to one the registry
    already carries: composition is extend-not-edit, so a second entry may append pins to
    ``judge`` but may not re-word what ``judge`` means. Re-supplying the identical phrase is
    permitted, on the same terms a world handle may be re-supplied but not displaced.
    """


#: What a run reports when its results WERE read and none of them carried a versioned judge
#: config. An explicit value rather than an empty list because that is an OBSERVATION, and an
#: empty collection is falsy — :meth:`SweepableRegistry.is_indeterminate` would read it as
#: "nobody recorded this" and refuse to compare two runs that in fact agree. The sentinel says
#: "no result was scored by one" rather than "every scored dim used the built-in prompt", because
#: a run whose cells all died before scoring reaches this state too and the narrower claim is
#: true in both cases.
NO_JUDGE_CONFIGS = "(none — no result was scored by a versioned judge config)"

#: What a run reports when it recorded attribution and every dim was scored by the pin. Explicit
#: for the same reason :data:`NO_JUDGE_CONFIGS` is: "no dim departed from the pin" is an
#: observation, and an empty collection would read as an absence.
NO_JUDGE_DIM_DIVERGENCE = "(none — every scored dim used the run's judge pin)"


class RegistrationError(ValueError):
    """A declaration contradicts what this module promises, raised where it is written."""


@dataclass(frozen=True)
class ResolvedLevers:
    """What one run's levers are called and what it ran them at.

    A value rather than a bare dict because ``overlaid`` has no home on one: the level and the
    question "did the launch NAME this, or did the run resolve it from the subject's own
    configuration" are two facts about one lever, and a lens that had to re-derive the second
    from a host's carrier is the coupling :attr:`Sweepable.open_family` exists to remove.
    """

    values: dict[str, Any]
    """Lever name → the level this run carried, families expanded into their members."""

    overlaid: frozenset[str]
    """The levers an open family resolved for this run — the ones the launch NAMED.

    Read as provenance, not as membership: every name here is also a key of :attr:`values`, and a
    lever absent from it was resolved from the run record or recovered from what the run did.
    """

    members_by_family: Mapping[str, frozenset[str]] = field(default_factory=dict)
    """Open family name → the members it resolved for this run. :attr:`overlaid` is their union.

    Kept per family because a family that declares :attr:`Sweepable.resolves_into` has its
    surface's residual read with ITS members removed, and the union cannot say which family a
    member came from.
    """


class SweepableRegistry(HostAttributed):
    """The declared inputs for one host, validated at construction.

    Validation happens here rather than at module import because **two hosts coexist**: the
    engine must be able to hold one product's registry and a second product's at the same time and
    tell them apart by nothing but their contents. A module-level tuple validated at import can
    only ever express one host.
    """

    def __init__(self, declarations: Iterable[Sweepable], *, roles: Sequence[RolePins] = ()) -> None:
        """Validate and store one host's declarations.

        Args:
            declarations: The inputs, in reporting order. Names are stable — they appear in the
                bisection payload and in the analysis bundle's confound lists.
            roles: The pinned roles — who was held fixed so the candidate is the only thing
                varying. Entries naming the same role are merged in order, which is how a host
                nominates its own dimension into a role the core declared; the flat union is
                :attr:`role_pins`, the set the ``roles_differ`` badge speaks for. Selecting pins
                by name is the one way the name-and-reader pairing can still be broken, so a pin
                naming an undeclared or non-apparatus input is refused here.

        Raises:
            RegistrationError: The declaration set is unsound. Every defect is reported at once
                rather than one per construction, so a host fixing them sees the whole list.
        """
        self._init_attribution()
        self._declarations: tuple[Sweepable, ...] = tuple(declarations)
        self._role_entries: tuple[RolePins, ...] = tuple(roles)
        self._roles: tuple[RolePins, ...] = self._merge_roles()
        self._role_pins: tuple[str, ...] = tuple(name for role in self._roles for name in role.pins)
        if defects := self._defects():
            raise RegistrationError("sweepable declaration is unsound: " + "; ".join(defects))
        self._by_name: dict[str, Sweepable] = {declared.name: declared for declared in self._declarations}

    def _merge_roles(self) -> tuple[RolePins, ...]:
        """Fold the entries into one record per role, in first-declared order.

        Tolerant by design: a contradiction between two entries for one role is a DEFECT, and
        :meth:`_defects` reports it with every other one rather than this raising the first it
        meets. Duplicate pins collapse — a host re-stating a pin the core already declared is
        restating, not adding a second heading.

        Returns:
            One :class:`RolePins` per role name, pins in declaration order, carrying the phrase
            of the entry that introduced it.
        """
        pins_by_role: dict[str, list[str]] = {}
        phrases: dict[str, str] = {}
        for entry in self._role_entries:
            pins = pins_by_role.setdefault(entry.name, [])
            pins.extend(pin for pin in entry.pins if pin not in pins)
            if entry.subject_phrase and entry.name not in phrases:
                phrases[entry.name] = entry.subject_phrase
        return tuple(
            RolePins(name=name, pins=tuple(pins), subject_phrase=phrases.get(name))
            for name, pins in pins_by_role.items()
        )

    def _defects(self) -> list[str]:
        """Name every way the declaration set contradicts what this module promises.

        Note:
            **One promise this cannot check, deliberately.** Every declaration carrying
            ``indeterminate_when_blank`` states, in a comment, why that blank is an absence
            rather than a level — and ``judge_config_ids`` reaching production with the flag and
            no reason at all is what let a mis-declaration fabricate an apparatus confound in
            every generated analysis. This method runs over the declarations, where comments do
            not exist. Promoting the reason to a field so it could be checked here would add
            prose no surface reads, so the promise is a review obligation on each declaration;
            no test in this repository scans for the comment.

        Returns:
            One sentence per defect; empty when the declaration set is sound.
        """
        defects: list[str] = []
        seen: set[str] = set()
        for declared in self._declarations:
            name = declared.name
            if name in seen:
                defects.append(f"{name} is declared twice — the readers key by name, so one would shadow the other")
            seen.add(name)
            if not declared.reader_prose.strip():
                defects.append(f"{name} states no reader prose — a bare dimension name cannot be rendered to a reader")
            if declared.role == "apparatus" and not declared.confounds:
                defects.append(f"{name} is apparatus but states no reason it confounds — a bare name is a label")
            if declared.role != "apparatus" and declared.confounds:
                defects.append(f"{name} states a confound reason but is not apparatus — no scan would ever read it")
            defects.extend(self._coordinate_defects(declared))
            defects.extend(self._family_defects(declared))
        for name in self._role_pins:
            pinned = self._by_declared_name(name)
            if pinned is None:
                defects.append(
                    f"{name} is named as a role pin but is not declared — the role disclosure would read a shorter set"
                )
            elif pinned.role != "apparatus":
                defects.append(
                    f"{name} is named as a role pin but is not apparatus — a role pin is the rig, and only apparatus states why it confounds"
                )
        defects.extend(self._role_defects())
        defects.extend(self._resolution_defects())
        return defects

    def _resolution_defects(self) -> list[str]:
        """Check every :attr:`Sweepable.resolves_into` names a surface a lens could fold.

        Across declarations rather than per declaration, because the surface a family names is
        another entry in the same set — and a host extending a core adds both halves in one
        :meth:`extend`, so the union is the only place the pair can be checked.

        Returns:
            One sentence per defect; empty when every named surface is sound.
        """
        defects: list[str] = []
        claimed: dict[str, str] = {}
        for family in self._declarations:
            surface = family.resolves_into
            if family.open_family is None or surface is None:
                continue
            target = self._by_declared_name(surface)
            if target is None:
                defects.append(
                    f"{family.name!r} resolves into {surface!r}, which is not declared — the engine would fold a "
                    "lever no run ever resolves, so the defect it exists to remove could never be removed"
                )
            elif target.role != "lever" or target.open_family is not None:
                defects.append(
                    f"{family.name!r} resolves into {surface!r}, which is not a fixed lever — only a fixed lever "
                    "reaches a contrast as a level of its own, so there is nothing a residual could fold"
                )
            owner = claimed.setdefault(surface, family.name)
            if owner != family.name:
                defects.append(
                    f"{surface!r} is named as the resolved surface of both {owner!r} and {family.name!r} — its residual "
                    "would need both families' members removed at once, and which ones is not something the engine "
                    "can decide"
                )
        return defects

    def _role_defects(self) -> list[str]:
        """Name every way the pinned-role entries contradict what :class:`RolePins` promises.

        Returns:
            One sentence per defect; empty when the role set is sound.
        """
        defects: list[str] = []
        phrases: dict[str, str] = {}
        introduced: set[str] = set()
        pin_owner: dict[str, str] = {}
        for entry in self._role_entries:
            if not entry.name.strip():
                defects.append("a pinned role is declared with a blank name — a disclosure would group under nothing")
                continue
            phrase = entry.subject_phrase
            if phrase is not None and not phrase.strip():
                defects.append(
                    f"role {entry.name} declares a blank subject phrase — the disclosure frame would read "
                    f'"these runs were NOT " and stop'
                )
                phrase = None
            if entry.name in introduced:
                if phrase is not None and phrase != phrases.get(entry.name):
                    defects.append(
                        f"role {entry.name} is re-declared with a different subject phrase — composition is "
                        f"extend-not-edit, so an entry may add pins to a role but never re-word what it means"
                    )
            else:
                introduced.add(entry.name)
                if phrase is None:
                    defects.append(
                        f"role {entry.name} is introduced with no subject phrase — the engine renders the frame and "
                        f"the host supplies the words, so a role with none cannot be disclosed at all"
                    )
                else:
                    phrases[entry.name] = phrase
            for pin in entry.pins:
                owner = pin_owner.setdefault(pin, entry.name)
                if owner != entry.name:
                    defects.append(
                        f"{pin} is pinned into both {owner} and {entry.name} — one difference would print under two "
                        f"headings, and a reader cannot tell that it is one"
                    )
        for role in self._roles:
            if not role.pins:
                defects.append(
                    f"role {role.name} is declared with no pins — it would take zero axes, compute zero "
                    f"verdicts and disclose nothing, forever, with no error for its author to read"
                )
        return defects

    @staticmethod
    def _coordinate_defects(declared: Sweepable) -> list[str]:
        """Check the ``no_own_coordinate`` waiver is on a role that could have one.

        Args:
            declared: The declaration to check.

        Returns:
            One message per defect, empty when the declaration is well-formed.
        """
        if declared.no_own_coordinate is None:
            return []
        if declared.role != "lever":
            return [
                (
                    f"{declared.name!r} is a {declared.role} and declares no_own_coordinate — only a lever carries a "
                    "coordinate in the variant key, so waiving one it never had says nothing"
                )
            ]
        return []

    @staticmethod
    def _family_defects(declared: Sweepable) -> list[str]:
        """Check an open-family declaration is on a role that can have members, and says why.

        Args:
            declared: The declaration to check.

        Returns:
            One message per defect, empty when the declaration is well-formed.
        """
        defects: list[str] = []
        if declared.open_family is None:
            if declared.owns_member is not None:
                defects.append(
                    f"{declared.name!r} declares owns_member and is not an open family — a fixed declaration's "
                    "only member is its own name, so nothing would ever ask the test"
                )
            if declared.resolves_into is not None or declared.read_residual is not None:
                defects.append(
                    f"{declared.name!r} declares a resolved surface and is not an open family — only a family has "
                    "members to take back out of one, so the residual would be the surface itself"
                )
            return defects
        if (declared.resolves_into is None) != (declared.read_residual is None):
            defects.append(
                f"{declared.name!r} declares one of resolves_into / read_residual without the other — a surface with "
                "no residual reader can never be shown to have moved only by its members, and a reader with no "
                "surface names nothing to compare it against"
            )
        if declared.role != "lever":
            defects.append(
                f"{declared.name!r} is a {declared.role} and declares open_family — only a lever expands into "
                "levers, and a family of apparatus or labels is a shape no lens reads"
            )
        if declared.indeterminate_when_blank:
            defects.append(
                f"{declared.name!r} is an open family and declares indeterminate_when_blank — a family that "
                "resolves no members is a run that carried none, which is a level rather than an absence, and the "
                "flag would make every such run undecidable against every other"
            )
        return defects

    def _by_declared_name(self, name: str) -> Sweepable | None:
        """Look a declaration up before ``_by_name`` exists — validation runs first."""
        return next((declared for declared in self._declarations if declared.name == name), None)

    @property
    def declarations(self) -> tuple[Sweepable, ...]:
        """Every declaration, in reporting order."""
        return self._declarations

    @property
    def names(self) -> tuple[str, ...]:
        """Every declared name, in reporting order."""
        return tuple(declared.name for declared in self._declarations)

    @property
    def lever_names(self) -> tuple[str, ...]:
        """Every ``lever`` name, in reporting order — the inputs a campaign may declare it sweeps.

        Narrower than :attr:`names` on purpose, and the difference is the whole point. ``names``
        answers "is this input known to the host"; this answers "is this input a knob a campaign
        may say it swept". An ``apparatus`` input is the measuring rig and a ``label`` identifies
        rather than determines, so neither is something an experiment varies deliberately — and
        the engine already agrees, since ``derive_variant_identity`` builds the variant key from
        levers alone. An axis outside this set can be declared and can never become an arm.
        Membership is necessary rather than sufficient: a lever carrying
        :attr:`Sweepable.no_own_coordinate` is listed here and still contributes no coordinate
        of its own.

        **The ENUMERABLE half of the authoring vocabulary, and not the whole of it.** The gate's
        decision is :meth:`refuse_as_axis` and the remedy it prints is :attr:`axis_remedy`; both
        are built from this list, so the three cannot disagree about a fixed lever. What this one
        cannot carry is the other half — an open family's members, which are names rather than a
        set — so a caller that renders the vocabulary to an operator wants ``axis_remedy``, and a
        caller that decides wants ``refuse_as_axis``. Reading this list to DECIDE is what refused
        every ad-hoc member; reading it as the whole remedy is what advertised a vocabulary the
        gate did not implement.

        **An open family's own name is NOT here.** A family is a container whose members are the
        levers, and its name identifies no knob — declaring a campaign swept an overlay bag names
        the request rather than the thing that moved. The members a given run carried come from
        :meth:`resolve_levers`, which needs the run this set does not have; that is the whole
        reason a family is declared instead of enumerated.
        """
        return tuple(
            declared.name
            for declared in self._declarations
            if declared.role == "lever" and declared.open_family is None
        )

    @property
    def open_families(self) -> tuple[Sweepable, ...]:
        """Every open-family declaration, in reporting order."""
        return tuple(declared for declared in self._declarations if declared.open_family is not None)

    @property
    def resolution_surfaces(self) -> dict[str, Sweepable]:
        """Surface lever name → the open family whose members are written into it.

        The set a lens consults before counting a surface as a lever that moved on its own — see
        :attr:`Sweepable.resolves_into`. Empty for a host that declares none, which is every host
        whose families do not also register what they resolve into.
        """
        return {
            family.resolves_into: family
            for family in self.open_families
            if family.resolves_into is not None and family.read_residual is not None
        }

    def read_residual(self, surface: str, run: EvalRun, results: Sequence[EvalResult], removed: frozenset[str]) -> Any:
        """Read ``surface`` off one run with the ``removed`` member names taken out.

        Args:
            surface: A name in :attr:`resolution_surfaces`.
            run: The run to read.
            results: That run's results.
            removed: Member names of the family that resolves into ``surface``.

        Returns:
            Whatever the family's :attr:`Sweepable.read_residual` returns — comparable content, or
            ``None`` when this run does not carry the surface.

        Raises:
            KeyError: No family resolves into ``surface``.
        """
        family = self.resolution_surfaces.get(surface)
        if family is None or family.read_residual is None:
            raise KeyError(f"{self._host}no open family resolves into {surface!r}")
        return family.read_residual(run, results, removed)

    def family_owning(self, name: str) -> Sweepable | None:
        """The open family whose own membership test claims ``name``, or None when none does.

        Asked with no run in hand, which is why it goes through
        :attr:`Sweepable.owns_member` rather than :attr:`Sweepable.read`: an authoring gate has a
        name and no launch, and the run-shaped reader cannot answer for one. A family declaring no
        test claims nothing — the engine does not guess a host's keyspace.

        Args:
            name: A bare dimension name.

        Returns:
            The first family that claims it, in reporting order, or None. First rather than an
            ambiguity refusal: two families claiming one name is a host defect this registry
            cannot see at registration — the tests are opaque callables — and the run-time
            resolution in :meth:`_resolve` is where an overlap becomes visible and is refused.
        """
        return next(
            (family for family in self.open_families if family.owns_member is not None and family.owns_member(name)),
            None,
        )

    def refuse_as_axis(self, name: str) -> str | None:
        """Why a campaign may not declare it swept ``name``, or None when it may.

        **The single source the authoring gate decides from.** It replaced a ``get`` lookup that
        saw a family container as an ordinary ``role == "lever"`` declaration, which accepted the
        one name that identifies no knob and refused every ad-hoc member — the two halves of the
        open-family rule, wrong in opposite directions, with
        :attr:`lever_names` printed as the remedy for both and able to help with neither.

        The four answers, in the order they are decided:

        * **A fixed lever** — admitted. Membership is necessary rather than sufficient: a lever
          carrying :attr:`Sweepable.no_own_coordinate` is admitted here and still contributes no
          coordinate of its own, because sweeping it does separate arms under the coordinate of
          what it resolved to.
        * **An open family's own name** — refused, pointed at its members. The container carries
          no coordinate and no resolution emits it, so a campaign declaring it has declared the
          request rather than the thing that moved — and would earn a coverage row no lens can
          ever resolve a level for.
        * **A declaration that is not a lever** — refused, naming the role. The remedy differs
          from the unregistered one and must read differently: an apparatus input is already
          declared, and sending an author to declare it again is a loop.
        * **Anything else** — admitted when some family's :attr:`Sweepable.owns_member` claims it,
          refused as unregistered otherwise.

        Checked in that order because a declaration wins over a family's claim: a name a family
        recognises AND an apparatus declaration holds is the rig wearing a lever's name, which
        :meth:`_resolve` already refuses once a run makes it visible. A family restating a fixed
        LEVER is the one permitted overlap and is admitted by the first branch either way.

        Args:
            name: The axis a campaign wants to declare.

        Returns:
            None when the name may be declared; otherwise the reason, ending in the remedy
            :attr:`axis_remedy` renders — so a caller that prints this prints one vocabulary.
        """
        declared = self._by_name.get(name)
        if declared is not None:
            if declared.open_family is not None:
                return (
                    f"{name} is an open family rather than a knob — its MEMBERS are the levers, and the container "
                    f"names the request rather than the thing that moved ({declared.open_family}). Declare the "
                    f"member this campaign swept. {self.axis_remedy}"
                )
            if declared.role != "lever":
                return (
                    f"{name} is registered as {declared.role}, not as a lever a campaign sweeps — "
                    "a run can carry it, but every run carrying it resolves to the same variant, so it can "
                    f"never become an arm. Hold it fixed and sweep a lever instead. {self.axis_remedy}"
                )
            return None
        if self.family_owning(name) is not None:
            return None
        return f"{name} is not registered as sweepable, so no run can vary it. {self.axis_remedy}"

    @property
    def axis_remedy(self) -> str:
        """The vocabulary an author picks a replacement axis from, as one sentence.

        The remedy half of :meth:`refuse_as_axis`, and derived from the same declarations, so the
        list an operator reads off a refusal cannot offer a name the same gate refuses on the next
        call. Naming every declaration here is what makes a message a trap; naming only
        :attr:`lever_names` is what left a host with an open family advertising a vocabulary
        missing the half it actually sweeps.

        Returns:
            The fixed levers, then one clause per open family — its name, whether it recognises
            members at all, and the reason it is open. A family declaring no
            :attr:`Sweepable.owns_member` is reported as recognising none, because the alternative
            is an author reading "not registered as sweepable" about a member the host does in
            fact sweep and going looking for a declaration that will never exist.
        """
        levers = ", ".join(sorted(self.lever_names)) or "(none)"
        clauses = [f"Declarable axes: {levers}"]
        for family in self.open_families:
            if family.owns_member is None:
                clauses.append(
                    f"{family.name} is an open family and declares no membership test, so this host recognises "
                    f"none of its members as axes ({family.open_family})"
                )
            else:
                clauses.append(f"plus any member of {family.name} this host recognises ({family.open_family})")
        return "; ".join(clauses)

    @property
    def role_pins(self) -> tuple[str, ...]:
        """Every pinned-role input across every role — the union ``roles_differ`` speaks for.

        Flat on purpose: the badge asks one question over the whole rig. A surface that renders
        a disclosure reads :attr:`roles` instead, because a headline covering two roles would be
        false about whichever one did not move.
        """
        return self._role_pins

    @property
    def roles(self) -> tuple[RolePins, ...]:
        """Every pinned role, in first-declared order, with its pins and the host's phrase.

        The engine's two come first because the shared core declares them; a host's own follow in
        the order it extended. A disclosure surface iterates this rather than naming the roles it
        knows about, which is what lets a host-declared role reach the operator with no per-role
        edit in the engine.
        """
        return self._roles

    def extend(self, declarations: Iterable[Sweepable], *, roles: Sequence[RolePins] = ()) -> SweepableRegistry:
        """Return a new registry carrying these declarations after this one's.

        The shared core is never mutated — a host adds to a copy, so two hosts registered in one
        process cannot see each other's vocabulary.

        Args:
            declarations: The host's own inputs.
            roles: Pinned-role entries appended to this registry's. An entry naming a role this
                registry already carries NOMINATES its pins into that role and omits the subject
                phrase; an entry naming a new one INTRODUCES it and must supply the phrase. Both
                are appends — neither can re-word or displace what is already declared.

        Returns:
            A new validated registry.

        Raises:
            RegistrationError: The combined set is unsound.
        """
        return SweepableRegistry(
            (*self._declarations, *declarations),
            roles=(*self._role_entries, *roles),
        )

    def get(self, name: str) -> Sweepable | None:
        """The declaration for ``name``, or None when this host never declared it."""
        return self._by_name.get(name)

    def read_all(self, run: EvalRun, results: Sequence[EvalResult] = ()) -> dict[str, Any]:
        """Read every declared input off one run.

        Args:
            run: The run to read.
            results: That run's results, for the result-level inputs. Omitting them reads those
                as empty, which is correct for a caller that has no results loaded and wrong to
                interpret as "the run used none".

        Returns:
            ``{name: value}`` for every declared input, JSON-safe. Set-valued inputs come out
            sorted so element order is never read as a difference. A host that declared nothing
            beyond an empty core gets ``{}`` — a well-formed empty answer, not an exception.

            An :attr:`Sweepable.open_family` declaration contributes its MEMBERS, never itself:
            the family name identifies no knob, so a caller comparing two runs on it would be
            comparing the request rather than the thing that moved.

        Raises:
            RegistrationError: A family resolved a member name an ``apparatus`` or ``label``
                declaration claims — see :meth:`resolve_levers` for why a fixed LEVER is the one
                overlap that is permitted. It is a declaration defect like any other here, and
                the only one that cannot be seen until a run is in hand, since a family's
                members are whatever that run carried.
        """
        return self._resolve(run, results)[0]

    def resolve_levers(self, run: EvalRun, results: Sequence[EvalResult] = ()) -> ResolvedLevers:
        """Read the LEVERS one run carried — fixed declarations plus this run's family members.

        The lever half of :meth:`read_all`, and the one call every reporting lens makes to learn
        what a run's swept axes are called. It exists so the coverage map, the variant lens and
        the declared-design check cannot come to different answers about which levers a run has:
        two hand-written ``role == "lever"`` walks are free to disagree, and one of them would
        also have to remember to expand families.

        **A family MAY restate a fixed lever, and that is the one permitted overlap.** The level
        is then read from the family, and the restatement is what :attr:`ResolvedLevers.overlaid`
        reports — the generic signal a lens needs to say the launch NAMED this value rather than
        the run having resolved it from the subject's own configuration. A background tool's
        model is the worked case: it needs a fixed declaration so every run resolves a variant
        coordinate for it, and it needs to report ``overridden`` on the arm that chose it. A
        collision with an ``apparatus`` or ``label`` declaration stays refused, because no
        reading makes a silent shadow of the measuring rig true.

        Args:
            run: The run to read.
            results: That run's results, for the result-level inputs.

        Returns:
            The resolution. Two runs of one host can carry DIFFERENT lever sets — that is what
            "open" means — so a caller comparing them unions the names rather than indexing one
            by the other's.

        Raises:
            RegistrationError: A family resolved a member name an apparatus or label claims.
        """
        return self._resolve(run, results)[1]

    def _resolve(self, run: EvalRun, results: Sequence[EvalResult] = ()) -> tuple[dict[str, Any], ResolvedLevers]:
        """Read every declaration ONCE, expanding families, and separate the levers out.

        One pass rather than two, because :meth:`read_all` and :meth:`resolve_levers` calling the
        host's readers separately would let a reader that is not perfectly pure hand the two
        surfaces different answers about one run — the drift this module exists to make
        inexpressible, re-created one layer up.

        Args:
            run: The run to read.
            results: That run's results.

        Returns:
            ``(every value by name, the lever resolution)``.

        Raises:
            RegistrationError: A family resolved a member name an apparatus or label claims.
        """
        values: dict[str, Any] = {}
        lever_names: set[str] = set()
        families: list[tuple[Sweepable, dict[str, Any]]] = []
        for declared in self._declarations:
            read = declared.read(run, results)
            if declared.open_family is not None:
                families.append((declared, dict(read or {})))
                continue
            values[declared.name] = read
            if declared.role == "lever":
                lever_names.add(declared.name)
        overlaid: set[str] = set()
        members_by_family: dict[str, frozenset[str]] = {}
        for declared, members in families:
            if shadowed := sorted(set(members) & (set(values) - lever_names)):
                raise RegistrationError(
                    f"{self._host}open family {declared.name!r} resolved member name(s) claimed by an apparatus or label "
                    f"declaration: {', '.join(shadowed)} — the rig would be reported as a swept knob"
                )
            values.update(members)
            lever_names.update(members)
            overlaid.update(members)
            members_by_family[declared.name] = frozenset(members)
        return values, ResolvedLevers(
            values={name: values[name] for name in lever_names},
            overlaid=frozenset(overlaid),
            members_by_family=members_by_family,
        )

    def read_role_pins(self, run: EvalRun, results: Sequence[EvalResult] = ()) -> dict[str, Any]:
        """Read the pinned-role inputs off one run, through the declarations above.

        A comparison surface that wants to say "these runs were judged differently" reads the
        pins from here rather than reaching for the run fields itself, so it cannot come to a
        different answer from the bisection about one pair of runs: both end up calling the SAME
        :attr:`Sweepable.read` callables.

        Args:
            run: The run to read.
            results: That run's results. ``judge_model`` and ``judge_config_ids`` are observed
                ACROSS results, so omitting them reads each as ``[]`` — which means "nobody
                supplied results", never "no model judged" or "the run used no judge config".

        Returns:
            ``{name: value}`` for every pinned role, JSON-safe; ``{}`` when this host pins none.
        """
        return {name: self._by_name[name].read(run, results) for name in self._role_pins}

    def comparability(self, name: str, values: Sequence[Any]) -> Comparability:
        """Decide whether a set of runs agree on one input, disagree, or cannot be asked.

        The N-run form of the three-way rule a bisection applies over two: undecidability is
        tested FIRST, because an input nobody recorded compares equal to another nobody recorded
        and would otherwise report agreement that was never observed.

        Args:
            name: A declared input name — it selects the ``indeterminate_when_blank`` rule.
            values: One observed value per run.

        Returns:
            ``"unknown"`` when at least one arm's value is a blank on an input whose blank means
            "never recorded"; otherwise ``"same"`` when every arm agrees, ``"differs"`` when they
            do not. Fewer than two values is ``"same"`` — there is nothing to disagree with.
        """
        items = list(values)
        if len(items) < 2:
            return "same"
        if self.is_indeterminate(name, *items):
            return "unknown"
        return "same" if all(item == items[0] for item in items[1:]) else "differs"

    def confound_reason(self, name: str) -> str:
        """The declared reason a change in ``name`` clouds a comparison.

        Disclosure surfaces render this rather than authoring their own sentence, so the caveat
        an operator reads above a compare table and the one they read on a bisection are the same
        claim about the same input.

        Args:
            name: A declared ``apparatus`` input name.

        Returns:
            The declaration's :attr:`Sweepable.confounds` prose.

        Raises:
            KeyError: ``name`` is not declared by this host, or is declared with no confound
                reason — which validation already makes impossible for an ``apparatus`` input.
        """
        declared = self._by_name.get(name)
        if declared is None:
            raise KeyError(f"{self._host}{name} is not declared by this host")
        if declared.confounds is None:
            raise KeyError(f"{self._host}{name} declares no confound reason — only apparatus inputs speak for one")
        return declared.confounds

    def reader_prose(self, name: str) -> str:
        """What ``name`` is, in words a reader can act on.

        Args:
            name: A declared input name.

        Returns:
            The declaration's :attr:`Sweepable.reader_prose`.

        Raises:
            KeyError: ``name`` is not declared by this host.
        """
        declared = self._by_name.get(name)
        if declared is None:
            raise KeyError(f"{self._host}{name} is not declared by this host")
        return declared.reader_prose

    def is_indeterminate(self, name: str, *values: Any) -> bool:
        """True when ``name``'s comparison cannot be decided rather than coming out equal.

        One rule, declared per input: a blank value on an input where blank means "nobody
        recorded this" rather than "this is the recorded value". Comparing two such blanks for
        equality manufactures an observation, so the honest answer is that the comparison cannot
        be decided — the same ``missing != zero`` discipline the rest of the eval surface applies.

        Which inputs those are is a per-input judgement made at the declaration, never here: a
        blank is a real recorded level on some of them and an absence on others, and no rule
        expressed at this call site could tell them apart. A reader can also make the value
        non-blank when it knows the state is real, which is what :data:`NO_JUDGE_CONFIGS` and its
        siblings are for — the judgement is still the declaration's, but a sentinel keeps a
        recorded level from arriving here disguised as an absence.

        Args:
            name: A declared input name. A name this host never declared is never indeterminate.
            *values: The observed values to judge.

        Returns:
            True when the comparison cannot be decided.
        """
        declared = self._by_name.get(name)
        return bool(declared and declared.indeterminate_when_blank and not all(values))


def _judge_config_ids(_run: EvalRun, results: Sequence[EvalResult]) -> list[str] | str:
    """Every judge-config id OBSERVED across a run's results, sorted.

    The run also carries a declared ``judge_config_ids`` — the set it pinned at launch — and
    reading that instead is a tempting simplification that would change what this input means.
    Observed and declared answer different questions: the declared set says what the run
    committed to, while this says what the results were actually scored with — a dim that was
    never judged leaves no key here while the run still declares its config. The ``_run``
    parameter is unused deliberately, not by oversight.

    Args:
        _run: Unused — see above.
        results: The run's results. An EMPTY sequence is the one absence this reader has: the
            caller either holds no results or was handed none, and neither says anything about
            what judged the run.

    Returns:
        Sorted config ids when any result carried one; :data:`NO_JUDGE_CONFIGS` when results were
        read and none did; ``[]`` when no results were supplied at all.
    """
    if not results:
        return []
    observed = sorted({config_id for result in results for config_id in result.judge_config_ids.values()})
    return observed or NO_JUDGE_CONFIGS


def served_models_by_score(result: EvalResult) -> list[str | None]:
    """Who scored each of one result's stored scores, as the provider named it — ``None`` where it did not.

    Every stored score counts: the rubric dims and both dual-score axes. The comparison reader
    below and a result's read surface both ask this, so they cannot disagree about which scores a
    result has.

    Args:
        result: The result to read.

    Returns:
        One entry per stored score, in rubric-then-axes order; empty when nothing was scored.
    """
    axes = (score for score in (result.transcript_score, result.outcome_score) if score is not None)
    return [score.served_model for score in (*result.rubric_scores, *axes)]


def _judge_served_models(_run: EvalRun, results: Sequence[EvalResult]) -> list[str] | None:
    """The models that actually scored a run's results, as the provider named them, sorted.

    Read off each stored score's ``served_model`` — what the provider's response said answered —
    and never off the run's ``judge_model``, which is the model the launch ASKED for. The two
    differ whenever the pin is a provider-side floating alias: ``~anthropic/claude-haiku-latest``
    may be answered by one concrete model this month and by another
    model the next, so comparing pins made two runs judged by different models read as one
    apparatus. The ``_run`` parameter is unused deliberately.

    Every scored dimension counts, including one whose ``JudgeConfig`` pinned its own model: that
    is still a model scoring this run's numbers, and a config-pinned alias drifts exactly as the run
    pin does. A dim scored away from the pin therefore also moves ``judge_dim_divergence``, which
    reports the REQUESTED side of the same fact; the two are read separately because a served
    model can move with nothing requested moving, which is this defect.

    Args:
        _run: Unused — see above.
        results: The run's results. An empty sequence is an absence: the caller holds none.

    Returns:
        The sorted distinct served models when every stored score names one. ``None`` when ANY
        score names none — the response did not say which model answered — because a
        run whose scorer is partly unobserved cannot be said to match another on the observed part.
        ``[]`` when no results were supplied or none carries a score: nothing scored, so there is
        no scorer to compare. Both blanks read as undecidable, never as agreement.

        A partial record is not reported as the part that is known, although that part can
        prove a DIFFERENCE: the comparison takes one value per run, and a partial set there
        equals a full one whenever the observed parts agree, which is the false agreement this
        reader exists to refuse. The price is that such a pair reads ``unknown`` rather than
        ``differs`` — an understatement, never a contradiction. A single result's read surface
        does name the known part, since it describes rather than matches.
    """
    served: set[str] = set()
    for result in results:
        for model in served_models_by_score(result):
            if model is None:
                return None
            served.add(model)
    return sorted(served)


def _judge_dim_divergence(run: EvalRun, _results: Sequence[EvalResult]) -> list[str] | str | None:
    """Which dims were scored by something other than the run's judge pin.

    Isolates the axis ``judge_model`` cannot express. Reading the whole ``effective_judges`` map
    here would report one physical difference twice, since attribution is derived from the pin
    and moves whenever the pin does.

    Args:
        run: The run to read.
        _results: Unused — divergence is declared on the run.

    Returns:
        Sorted ``"dim=model"`` entries for each dim that departed from the pin;
        :data:`NO_JUDGE_DIM_DIVERGENCE` when attribution was recorded and none did; ``None`` when
        the run recorded no attribution, or carries only a reconstruction — an inference must not
        evidence a confound.
    """
    judges = run.hashable_effective_judges
    if judges is None:
        return None
    diverged = sorted(f"{dim_id}={model}" for dim_id, model in judges.items() if model != run.judge_model)
    return diverged or NO_JUDGE_DIM_DIVERGENCE


def _request_settings(settings: Any) -> dict[str, Any] | None:
    """One role's recorded request settings as a JSON-safe level, or ``None`` when unrecorded.

    Args:
        settings: The run's :class:`~threetears.evals.contracts.models.ClientRequestSettings` for a role, or
            ``None``.

    Returns:
        The settings as a plain mapping — ``reasoning_max_tokens: None`` inside it is a recorded
        level (no reasoning parameter was sent), so the mapping is never blank — or ``None`` when
        the run carries no stamp.
    """
    return None if settings is None else settings.model_dump(mode="json")


#: The four concepts every LLM product has, and nothing else. A host adds to this; it never
#: edits it. Eight declarations cover the four concepts because "the judge" is four separable
#: axes — which model, how it was asked, which configuration, and whether any dimension departed
#: from the pin — and "the simulator" two, and collapsing them reports one physical difference
#: under another's name.
#:
#: ``indeterminate_when_blank`` marks an input whose empty value means "never recorded", not
#: "recorded as empty". The two result-level inputs are blank-indeterminate from the other
#: direction: they are observed ACROSS a run's results, so an empty reading can mean the caller
#: supplied no results at all. Compared as a value, that empty reports the apparatus as CHANGED
#: against any run that did record one, which is a difference the runs never had.
#:
#: What that reasoning does NOT license is reading "the results were present and recorded no
#: config" as the same absence. It is a recorded level, and calling it an absence manufactured a
#: confound in every generated analysis of a campaign that pinned no judge configs. The fix
#: belongs in the READER, which knows which of the two cases it is in, not in the flag, which
#: cannot: :func:`_judge_config_ids` returns an explicit sentinel for the observed state and
#: reserves the falsy empty for the genuine absence.
CORE_SWEEPABLES: tuple[Sweepable, ...] = (
    # A run is one arm, so its candidate model is one level. It was once declared as `models` while
    # every reporting lens called the same knob `model`, so a campaign declaring it as its axis got a
    # coverage row it could not be matched to; one name serves both now.
    Sweepable(
        name=CANDIDATE_MODEL_LEVER,
        role="lever",
        read=lambda run, _results: run.candidate_model,
        reader_prose="the models the subject itself ran on",
    ),
    # The kind is what the candidate IS — its code, its seam, what it was asked to produce — so it
    # is a coordinate of every variant. Without it, two kinds at one model with no overlays derived
    # one key and pooled into one arm.
    Sweepable(
        name=CANDIDATE_KIND_LEVER,
        role="lever",
        read=lambda run, _results: run.candidate_kind,
        reader_prose="what kind of candidate the subject ran as",
    ),
    # The models that SCORED, observed across the results — not the run's ``judge_model`` pin,
    # which this once read. The pin is what the launch asked for, and a
    # provider-side floating alias (``~vendor/model-latest``) resolves to a different concrete model
    # from one month to the next, so two runs stamping one alias compared equal here while judged by
    # different models. Only the provider's response names the model that answered, and each score
    # records it (``RubricScore.served_model``). Indeterminate when blank: ``None`` is a score nobody
    # observed the scorer of (a response that named none), and
    # ``[]`` is a run with no scores — neither may read as a shared judge. The name stays
    # ``judge_model`` because every consumer — the role pins, host applicability declarations, the
    # bisection's vocabulary — means "which model judged", which is now what it answers.
    Sweepable(
        name="judge_model",
        role="apparatus",
        read=_judge_served_models,
        reader_prose="the model that scored the work, as the provider named it",
        confounds="a different model scored the work, and two judges do not grade the same answer the same way",
        indeterminate_when_blank=True,
        result_level=True,
    ),
    # How the judge was ASKED, beside which judge it was: the host's client builder applies one
    # output cap and one reasoning budget to every judge client, process-wide, and moving them
    # regrades every later run without touching `judge_model`. Indeterminate when blank because a
    # run with no stamp recorded none, and reading two of those as equal — or one as equal to
    # today's values — asserts a shared instrument across the change the stamp exists to expose.
    Sweepable(
        name="judge_request_settings",
        role="apparatus",
        read=lambda run, _results: _request_settings(run.judge_request_settings),
        reader_prose="the output cap and private-reasoning budget every judge request was sent with",
        confounds=(
            "the judge was asked with a different output cap or reasoning budget, so the same judge model may have "
            "reasoned differently — or not at all — before scoring"
        ),
        indeterminate_when_blank=True,
    ),
    # Built from the HASHABLE map: a `derived` reconstruction is an inference about what scored a
    # run, and letting it report `differs` here would evidence a confound from a guess. Blank
    # means absent OR derived — in both cases nobody recorded it, which is what
    # indeterminate_when_blank is for. A run that recorded attribution and had NO divergence is a
    # real observed state, not a blank, so it reads as an explicit sentinel rather than an empty
    # collection — an empty list is falsy and would be misread as undecidable.
    Sweepable(
        name="judge_dim_divergence",
        role="apparatus",
        read=_judge_dim_divergence,
        reader_prose="the dimensions scored by something other than the run's judge pin",
        confounds=(
            "a dimension was scored by a model other than the run's judge pin, so the runs were graded by different "
            "apparatus even where their pins agree"
        ),
        indeterminate_when_blank=True,
    ),
    # Indeterminate when blank ONLY for the genuine absence. Results that were read and carried
    # no config are a recorded level, and the reader says so with NO_JUDGE_CONFIGS rather than an
    # empty list. Two runs that both pinned none genuinely agree. Reading them as undecidable is
    # what fabricated an apparatus confound in every analysis of a campaign that used no judge
    # configs — the defect this whole flag's discipline exists to prevent.
    Sweepable(
        name="judge_config_ids",
        role="apparatus",
        read=_judge_config_ids,
        reader_prose="the versioned judge configurations the results were actually scored with",
        confounds="a different judge configuration scored the results — other dimensions, or the same ones weighted differently",
        indeterminate_when_blank=True,
        result_level=True,
    ),
    # Indeterminate when blank: this one is the RESOLVED model, stored precisely so a None cannot
    # mean "role default". None means no simulated user was pinned — or that the run's writer
    # recorded none, which the blank cannot tell apart — and comparing two such runs equal would
    # assert a shared simulator nobody recorded. That hazard is the stated reason the field stores
    # the resolved value at all.
    Sweepable(
        name="simulator_model",
        role="apparatus",
        read=lambda run, _results: run.simulator_model,
        reader_prose="the model that played the other side of the conversation",
        confounds="a different model played the user, so the candidate was answering a different conversation",
        indeterminate_when_blank=True,
    ),
    # How the simulated user was asked, for the reason `judge_request_settings` is recorded: the
    # same simulator model at a different cap can cut a turn off, and the candidate then answers
    # a different conversation. Indeterminate when blank on the same terms.
    Sweepable(
        name="simulator_request_settings",
        role="apparatus",
        read=lambda run, _results: _request_settings(run.simulator_request_settings),
        reader_prose="the output cap and private-reasoning budget every simulated-user request was sent with",
        confounds=(
            "the simulated user was asked with a different output cap or reasoning budget, so the same simulator "
            "model may have played the conversation differently"
        ),
        indeterminate_when_blank=True,
    ),
    # Indeterminate when blank: None has TWO causes on this field — "the run was uncapped" and
    # "the run's writer recorded no ceiling" — and they are not the same fact. One empty standing for two
    # states cannot be compared for equality without choosing one of them silently.
    Sweepable(
        name="max_cost_usd",
        role="apparatus",
        read=lambda run, _results: run.max_cost_usd,
        reader_prose="the spend ceiling in force for the run",
        confounds="a different spend ceiling was in force, which can stop a run before it finishes its cases",
        indeterminate_when_blank=True,
    ),
)

#: Who GRADED the work. Kept separate from :data:`SIMULATOR_INPUTS` rather than folded in with
#: them, because the two confound a quality delta by different mechanisms and a reader has to act
#: on them differently: a judge change regrades the same work, while a simulator change hands the
#: candidate a different conversation to do. A single caveat covering both would be false about
#: whichever one did not move, and the disclosure surfaces render them under separate headings for
#: exactly that reason.
JUDGE_INPUTS: tuple[str, ...] = ("judge_model", "judge_request_settings", "judge_dim_divergence", "judge_config_ids")

#: Who PLAYED THE USER. See :data:`JUDGE_INPUTS` for why this is its own tuple.
SIMULATOR_INPUTS: tuple[str, ...] = ("simulator_model", "simulator_request_settings")

#: The two roles the engine itself declares, with the words each disclosure is rendered in. The
#: core owns these because it owns the dimensions pinned into them; it does not own the keyspace,
#: and a host adds its own through :meth:`SweepableRegistry.extend` (see :class:`RolePins`).
CORE_ROLES: tuple[RolePins, ...] = (
    RolePins(name="judge", pins=JUDGE_INPUTS, subject_phrase="judged the same way"),
    RolePins(name="simulator", pins=SIMULATOR_INPUTS, subject_phrase="run against the same simulator"),
)

#: The registry a host extends. Nothing here names a product.
SHARED_CORE = SweepableRegistry(CORE_SWEEPABLES, roles=CORE_ROLES)


__all__ = [
    "CANDIDATE_KIND_LEVER",
    "CANDIDATE_MODEL_LEVER",
    "CORE_ROLES",
    "CORE_SWEEPABLES",
    "JUDGE_INPUTS",
    "NO_JUDGE_CONFIGS",
    "NO_JUDGE_DIM_DIVERGENCE",
    "SHARED_CORE",
    "SIMULATOR_INPUTS",
    "Comparability",
    "FamilyMemberTest",
    "IntervalScale",
    "NominalScale",
    "OrdinalScale",
    "RegistrationError",
    "ResidualReader",
    "ResolvedLevers",
    "RolePins",
    "Scale",
    "Sweepable",
    "SweepableReader",
    "SweepableRegistry",
    "SweepableRole",
    "SweepableValue",
    "served_models_by_score",
]
