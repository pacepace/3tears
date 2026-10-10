"""What one candidate kind's runs carry beyond the engine's own fields, declared as plain Pydantic models.

A kind owns two things the engine carries without reading: its **launch overlays** — the knobs a
launch turns for its runs — and its **spec** — what a template of that kind declares beyond the
engine's own fields (a router's label set, a game master's table). The engine owns everything done
with them. Overlays are validated against the kind's model when a launch is made (refused, by field,
before any run exists), frozen onto every run, read as levers through the host's sweepables registry
and content-addressed into the variant key. A spec is validated when a template is authored (refused,
by field), validated again and frozen onto every run its template launches, and hashed into the
run's measurement context. A product writes the two models and names them once, on its profile's
``kinds``; nothing else is per-field work and nothing else registers them.

Example — a game master whose launch may turn six knobs, and whose template states its table::

    class GMOverlays(BaseModel):
        gm_model: str = Field("gm-large", description="the model narrating the session")
        difficulty: Annotated[Literal["easy", "medium", "hard", "deadly"], Ordinal()] = Field(
            "medium", description="the encounter difficulty the GM builds to"
        )
        narration_words: Annotated[int, Interval(unit="words")] = Field(120, description="how long a narration runs")
        max_tool_rounds: int = Field(4, ge=1, description="how many tool rounds one GM turn may take")
        style_prompt: str = Field("", description="the GM style prompt, by content")
        house_rules: dict[str, str] = Field(default_factory=dict, description="the house rules the table plays by")

    class GMSpec(BaseModel):
        players: int = Field(ge=1, le=6)
        starting_level: int = Field(1, ge=1, le=20)
        hidden_facts: list[str] = []  # what only the GM and the judge may know

    GM = KindContract("gm", overlays=GMOverlays, spec=GMSpec)
    profile = HostProfile(host_id="dow", host_sweepables=SHARED_CORE, kinds=(GM,), ...)

Each field becomes the lever ``<kind>.<field>`` (the prefix is the kind's name unless one is given).
How a level sits on an axis follows the field: a number is an interval (``Interval`` names its
unit), a ``Literal`` or ``Enum`` marked ``Ordinal()`` ranks in declaration order, and anything else
is a nominal level addressed by its content — a long prompt joins across runs by what it says, not
by what it is called. A ``dict[str, ...]`` field is an **open family**: every key a launch sets is
its own lever (``gm.house_rules.flanking``), and the whole map is one more lever
(``gm.house_rules``) that carries the variant coordinate, so two runs with different rules are two
variants however many rules each named.

Two more markers say what a knob does rather than how its levels sit: ``ActsOn`` names the measure it
is supposed to move (``MemberActsOn`` names it per entry of a map, whose entries are distinct knobs),
and ``ResolvesInto`` names the host lever it is written into — the resolved surface the host records
as well — so one turn of the knob reads as one lever rather than two that always move together.

**What is frozen is the resolved model, defaults included** (:func:`freeze`). A launch naming
``difficulty="medium"`` and one naming nothing ran the same knob at the same level, so they record
the same overlays and share a variant — identity is computed over what ran, never over the request.
The cost is the other half of the same rule: which knobs a launch NAMED is not recorded, because it
is not a measurement condition. A spec is frozen the same way, so a template whose spec leaves a
field at its default and one that states the default are one condition. Two shapes would make the
record disagree with what ran, and both are settled at declaration: a field serialization leaves out
(``exclude``) is refused, since the record would merge runs the launcher ran differently, and a set
is recorded sorted, since its iteration order is the process's and not the configuration's.

**One mention.** The contract is named on the profile's ``kinds`` and nowhere else: the profile adds
its levers to the registry every lens reads (:attr:`~threetears.evals.kernel.host.profile.HostProfile.sweepables`),
and the engine resolves each run's level of them into the variant key beside the candidate model and
kind, which it also resolves itself. A host's own variant-lever reader returns only the levers the
host declares beyond these, and one that returns a kind's lever is refused, so a knob a launch can
turn but no analysis can see is not expressible — and neither is a second writer of one coordinate.
"""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, TypeAdapter, ValidationError

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.kernel.host.sweepables import (
    FamilyMemberTest,
    ProductionDepartureReader,
    ResidualReader,
    Sweepable,
    SweepableReader,
)
from threetears.evals.schema.values import IntervalScale, NominalScale, OrdinalScale, Scale, SweepableValue

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo

    from threetears.evals.schema.models import EvalResult, EvalRun

#: The longest display a nominal level is shown at before it is cut. A level is joined on its content
#: hash, so the cut costs a reader nothing but the tail of a long value.
_DISPLAY_LIMIT = 60

#: The suffix an open family's container lever carries, so the container and the map it expands are
#: two names: ``gm.house_rules`` is the whole rule set as one level, ``gm.house_rules.*`` the family.
_FAMILY_SUFFIX = ".*"

#: Renders any validated Python value in its JSON form, as ``model_dump(mode="json")`` would.
_JSON_FORM: TypeAdapter[Any] = TypeAdapter(Any)


@dataclass(frozen=True)
class Ordinal:
    """Mark a ``Literal`` or ``Enum`` field as ordered: its levels rank in the order they are declared.

    Unmarked, such a field is nominal — the honest default for categories nobody said were ordered.
    """


@dataclass(frozen=True)
class Interval:
    """Mark a numeric field's unit, rendered beside each level (``120words``).

    A number is an interval without this; the marker only names what it counts.
    """

    unit: str | None = None


@dataclass(frozen=True)
class ActsOn:
    """Name the measure or covariate an overlay knob is supposed to move — its lever's ``acts_on``.

    The overlay twin of :attr:`~threetears.evals.kernel.host.sweepables.Sweepable.acts_on`, written
    as a marker beside ``Ordinal`` and ``Interval``: ``Annotated[Literal["low", "high"], ActsOn("reasoning_ratio")]``.
    The lever it marks gets the same mechanism check, the same refusals (an unknown, non-numeric or
    per-row name is refused when the profile is built) and the same identity guarantee — it is not a
    level, so it moves no variant key. Refused on a map field, whose entries are distinct knobs: name
    each entry's mechanism with :class:`MemberActsOn` instead.
    """

    measure: str


@dataclass(frozen=True, init=False)
class MemberActsOn:
    """Name the measure each listed ENTRY of an overlay map is supposed to move — :class:`ActsOn` per member.

    A map field is an open family, and its entries are distinct knobs one mechanism cannot speak for, so
    :class:`ActsOn` is refused there. This marker names them one key at a time:
    ``Annotated[dict[str, int], MemberActsOn({"max_search_calls": "search_calls"})]``. The member lever
    ``<kind>.<field>.max_search_calls`` then gets the same mechanism check, and the same refusals at the profile
    (an unknown, non-numeric or per-row measure), as a knob marked :class:`ActsOn`; every entry the marker does
    not list still reads ``unchecked`` / ``not_declared``. Declaration metadata: it moves no variant key.
    Refused on a field that is not a map, given twice, or with a blank key or measure.
    """

    members: tuple[tuple[str, str], ...]

    def __init__(self, members: Mapping[str, str]) -> None:
        """Record the map's key -> measure pairs.

        Args:
            members: Each map key a launch may set that has a known mechanism, mapped to the measure's name.
        """
        object.__setattr__(self, "members", tuple(members.items()))


@dataclass(frozen=True)
class ResolvesInto:
    """Name the fixed host lever this overlay knob is WRITTEN INTO — its lever's ``resolves_into``.

    A host that both lets a launch turn a knob and records what the knob resolved into registers one
    change twice: ``reasoning_effort`` as the overlay a launch named, and the resolved model parameters
    it was merged into (``llm_parameters``, levelled by content hash) as a lever of the host's own. Both
    are honest levers — the knob is what a campaign swept, the surface is what would ship — and both
    move whenever the knob does, so read as two levers every effort sweep reports two changes and each
    as confounded by the other, and no sweep of the knob can ever be read on its own.

    The marker names the surface, and the engine folds it into this knob wherever the runs show the
    surface moved ONLY where the knob did: across the runs being compared, every level of the knob
    carries one level of the surface. Where the surface differs between runs that held the knob at one
    level, something else wrote into it, and it stays a lever and a confound of its own. A surface a run
    did not record (its reader returned ``None``) folds nothing — that run cannot show agreement. The
    surface is in the variant key, so only two ARMS at one level of the knob can show it moving on its
    own; where no level has two, the fold still applies and every comparison it applies to is marked
    ``unverified_fold``, because a check that could not run is not a pass.

    Written beside the other markers, and combinable with them:
    ``Annotated[Literal["low", "high"], Ordinal(), ResolvesInto("llm_parameters")]``. The name
    must be a ``lever`` the host itself declares, not an open family, and not one another knob or
    family already claims — refused when the profile is built, because a surface two knobs write into
    cannot say which of them moved it. Refused twice on one field: a knob is written into one surface.

    **On a map field** the marker rides on the map's own lever (``gm.tool_configs``), whose level is
    the whole map — exactly the members a launch set and the values it set them to — so the surface
    folds into the map wherever the members alone account for its movement, and the map in turn folds
    into its members by the family's own residual
    (:attr:`~threetears.evals.kernel.host.sweepables.Sweepable.resolves_into`). Like :class:`ActsOn`,
    it is declaration metadata, not a level: it moves no variant key.
    """

    lever: str


class KindContractError(TypeError):
    """A kind's model cannot be read the way the engine promises to read it — raised where it is declared."""


@dataclass(frozen=True)
class _Knob:
    """One overlay field, read once at declaration: its lever name, its axis, and whether it is a family."""

    field_name: str
    lever: str
    prose: str
    family: bool
    ordinal_levels: tuple[Any, ...] | None
    interval_unit: str | None
    numeric: bool
    acts_on: str | None
    resolves_into: str | None
    member_acts_on: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class KindContract:
    """What one candidate kind's runs carry, declared as Pydantic models — see the module docstring.

    A runtime registration, like :class:`~threetears.evals.kernel.host.sweepables.Sweepable`: it
    holds model classes and is never stored. What is stored is the values those models validated:
    :attr:`~threetears.evals.schema.models.EvalTemplate.kind_spec` on a template, and
    :attr:`~threetears.evals.schema.models.EvalRun.overlays` and
    :attr:`~threetears.evals.schema.models.EvalRun.kind_spec` on a run.

    Attributes:
        kind: The kind's name, as a template's ``candidate_kind`` spells it. A run of any other kind
            carries none of these overlays.
        overlays: The kind's overlay model, or ``None`` for a kind a launch may not turn anything on —
            whose launches are then refused any overlay at all.
        spec: The kind's template spec model, or ``None`` for a kind whose templates state nothing
            beyond the engine's fields — whose templates are then refused any ``kind_spec`` at all.
        prefix: The namespace of this kind's levers, ``<prefix>.<field>``; the kind's name when
            omitted. Dotted lever names are what a pivot reads as open coordinates.
        seats: The seats in the rig this kind's runs fill, as an ALLOW-list: each entry names a pinned
            role of the host's registry (:class:`~threetears.evals.kernel.host.sweepables.RolePins` —
            the engine's ``judge`` and ``simulator``, or one the host adds), which seats every pin of that
            role, or names one apparatus dimension directly. Every apparatus dimension it does not seat is
            inapplicable to this kind's runs: a blank there is no confound, and a dimension added to the
            rig later — by the engine or the host — is inapplicable to this kind until it claims one,
            so it cannot make the kind's runs ``undecided``. A recorded level still wins over the
            declaration (:meth:`~threetears.evals.kernel.host.profile.HostProfile.omits_apparatus`).
            ``None``, the default, declares nothing and holds the kind to every dimension — the
            conservative reading for a kind that has not said.
    """

    kind: str
    overlays: type[BaseModel] | None = None
    spec: type[BaseModel] | None = None
    prefix: str | None = None
    seats: frozenset[str] | None = None
    _knobs: tuple[_Knob, ...] = field(init=False, repr=False, compare=False)
    _default_overlays: dict[str, Any] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Read the overlay model once, refusing every field the engine could not read as a lever.

        Raises:
            KindContractError: The kind or prefix is blank, an overlay field has no description, an
                alias, a marker its type cannot carry, or a map keyed by anything but strings, a spec
                field has an alias, or a field of either model (at any depth) is excluded from
                serialization, so the frozen record would not see it. Every defect is named at once.
        """
        if not self.kind.strip():
            raise KindContractError("a kind contract needs the kind's name")
        prefix = self.lever_prefix
        if not prefix.strip() or prefix.endswith(_FAMILY_SUFFIX):
            raise KindContractError(f"kind {self.kind!r} names a lever prefix {prefix!r} no lever could be read under")
        knobs: list[_Knob] = []
        defects: list[str] = []
        for name, info in (self.overlays.model_fields if self.overlays is not None else {}).items():
            knob, problems = _read_knob(prefix, name, info)
            defects.extend(f"{self.kind}.{name}: {problem}" for problem in problems)
            if knob is not None:
                knobs.append(knob)
        for name, info in (self.spec.model_fields if self.spec is not None else {}).items():
            if info.alias is not None or info.validation_alias is not None:
                defects.append(f"{self.kind}.{name}: declares an alias, and a template names its spec by field")
        for model in (self.overlays, self.spec):
            if model is not None:
                defects.extend(
                    f"{self.kind}.{path}: is left out of the frozen record (exclude), so runs at different values "
                    "would record one value and share a key while the launcher acted on each"
                    for path in _excluded_paths(model)
                )
        if defects:
            raise KindContractError(f"kind {self.kind!r}'s models cannot be read by the engine: " + "; ".join(defects))
        object.__setattr__(self, "_knobs", tuple(knobs))
        object.__setattr__(self, "_default_overlays", self._read_default_overlays())

    def _read_default_overlays(self) -> dict[str, Any] | None:
        """What a launch naming no overlay records: the subject's own setting of every knob.

        Returns:
            The frozen record of the overlay model's defaults; ``None`` when the model has a required
            field, so no launch runs at a default the engine could compare against.
        """
        if self.overlays is None:
            return {}
        try:
            return freeze(self.overlays.model_validate({}))
        except ValidationError:
            # NOSILENT: None IS the answer -- a required overlay field means no launch runs at a default
            return None

    @property
    def lever_prefix(self) -> str:
        """The namespace this kind's levers are named under."""
        return self.prefix if self.prefix is not None else self.kind

    # --- validation --------------------------------------------------------------------------

    def validate_overlays(self, overlays: Mapping[str, Any] | None) -> BaseModel | None:
        """Validate a launch's overlays against the kind's model, refusing by field.

        Args:
            overlays: What the launch asked for, by field name. ``None`` and ``{}`` both ask for the
                model's defaults.

        Returns:
            The validated model — every field resolved, defaults included — or ``None`` for a kind
            with no overlay model and a launch that turned nothing.

        Raises:
            ValidationFailedError: A name the model does not have, a value it refuses, or any overlay
                at all for a kind with no overlay model. The message names each field.
        """
        return self._validated(self.overlays, overlays, "overlay", "a launch cannot turn")

    def validate_spec(self, kind_spec: Mapping[str, Any] | None) -> BaseModel | None:
        """Validate a template's ``kind_spec`` against the kind's spec model, refusing by field.

        Args:
            kind_spec: What the template states, by field name. ``None`` and ``{}`` both ask for
                the model's defaults — and are refused by a model with a required field.

        Returns:
            The validated model, every field resolved, or ``None`` for a kind with no spec model
            and a template that states nothing.

        Raises:
            ValidationFailedError: A name the model does not have, a value it refuses, a required
                field left out, or any spec at all for a kind with no spec model. The message names
                each field.
        """
        return self._validated(self.spec, kind_spec, "kind_spec field", "a template cannot state")

    def _validated(
        self, model: type[BaseModel] | None, values: Mapping[str, Any] | None, noun: str, refused_as: str
    ) -> BaseModel | None:
        """Validate ``values`` against ``model``, naming every field refused — one rule for both models.

        Args:
            model: The kind's model, or ``None`` when it declares none.
            values: What was supplied, by field name.
            noun: What a field of this model is called in a refusal.
            refused_as: How a refusal says supplying a field to a kind with no model reads.

        Returns:
            The validated model, or ``None`` for no model and nothing supplied.

        Raises:
            ValidationFailedError: As :meth:`validate_overlays` and :meth:`validate_spec` describe.
        """
        supplied = dict(values or {})
        if model is None:
            if supplied:
                raise ValidationFailedError(
                    f"kind {self.kind!r} declares no {noun}s, so {refused_as} {', '.join(sorted(supplied))}"
                )
            return None
        known = set(model.model_fields)
        if unknown := sorted(set(supplied) - known):
            raise ValidationFailedError(
                f"kind {self.kind!r} has no {noun} {', '.join(unknown)}; its {noun}s are {', '.join(sorted(known))}"
            )
        try:
            return model.model_validate(supplied)
        except ValidationError as refused:
            raise ValidationFailedError(
                f"kind {self.kind!r} refused its {noun}s: {_field_errors(refused)}",
                details={"errors": refused.errors(include_url=False, include_context=False)},
            ) from refused

    # --- levers --------------------------------------------------------------------------------

    @property
    def sweepables(self) -> tuple[Sweepable, ...]:
        """One lever per overlay field, for the host's registry; an open family adds its container.

        Returns:
            The declarations, in field order. Empty for a kind with no overlay model.
        """
        declared: list[Sweepable] = []
        for knob in self._knobs:
            declared.append(
                Sweepable(
                    name=knob.lever,
                    role="lever",
                    read=self._reader(knob.field_name),
                    reader_prose=knob.prose,
                    acts_on=knob.acts_on,
                    resolves_into=knob.resolves_into,
                    departs_production=self._departs(knob.field_name),
                )
            )
            if knob.family:
                member_prefix = f"{knob.lever}."
                declared.append(
                    Sweepable(
                        name=f"{knob.lever}{_FAMILY_SUFFIX}",
                        role="lever",
                        read=self._member_reader(knob),
                        reader_prose=f"each entry of {knob.lever} a launch set, one lever per entry",
                        open_family=f"a launch may set any entry of {knob.field_name}, so its entries are whatever it named",
                        owns_member=_member_test(member_prefix),
                        resolves_into=knob.lever,
                        read_residual=self._residual_reader(knob),
                        member_acts_on=(
                            {f"{member_prefix}{key}": measure for key, measure in knob.member_acts_on}
                            if knob.member_acts_on
                            else None
                        ),
                    )
                )
        return tuple(declared)

    @property
    def lever_names(self) -> tuple[str, ...]:
        """Every lever name :attr:`sweepables` declares, families' containers included."""
        return tuple(declared.name for declared in self.sweepables)

    def levels(self, run: EvalRun) -> dict[str, SweepableValue]:
        """This run's level of every fixed overlay lever — what the engine composes into its variant key.

        Args:
            run: The run, of this kind or another.

        Returns:
            Lever name -> level. A run of another kind sits at a level of its own that every such
            run shares, so the lever splits nothing among them — and that level hashes apart from
            every value a field can hold, ``None`` included, so a pivot over the lever never joins
            a run of another kind with a run of this one whose field is ``None``.
        """
        overlays = self._overlays_of(run)
        if overlays is None:
            absent = SweepableValue.not_this_kind(self.kind)
            return {knob.lever: absent for knob in self._knobs}
        return {knob.lever: _level(knob, overlays.get(knob.field_name)) for knob in self._knobs}

    def _overlays_of(self, run: EvalRun) -> Mapping[str, Any] | None:
        """The run's frozen overlays when it is a run of this kind, else ``None``."""
        return run.overlays if run.candidate_kind == self.kind else None

    def _reader(self, field_name: str) -> SweepableReader:
        """A reader of one field off a run's frozen overlays."""

        def read(run: EvalRun, _results: Sequence[EvalResult]) -> Any:
            overlays = self._overlays_of(run)
            return None if overlays is None else overlays.get(field_name)

        return read

    def _departs(self, field_name: str) -> ProductionDepartureReader:
        """Whether a run set one overlay field away from its default — the subject's own setting (#571).

        The record holds every field, defaults included, so a value is no sign the launch set it; the
        default is what a launch naming nothing runs at, and a value other than it is a departure. A run
        of another kind does not carry the field and holds. A model with a required field has no default
        to compare against, so its runs read unchecked.
        """

        def departs(run: EvalRun, _results: Sequence[EvalResult]) -> bool | None:
            overlays = self._overlays_of(run)
            if overlays is None:
                return False
            if self._default_overlays is None or field_name not in overlays:
                return None
            return canonical_json(overlays[field_name]) != canonical_json(self._default_overlays.get(field_name))

        return departs

    def _member_reader(self, knob: _Knob) -> SweepableReader:
        """A reader of an open family's members: one per entry the run's map holds."""

        def read(run: EvalRun, _results: Sequence[EvalResult]) -> dict[str, Any]:
            overlays = self._overlays_of(run)
            entries = (overlays or {}).get(knob.field_name) or {}
            return {f"{knob.lever}.{key}": value for key, value in entries.items()}

        return read

    def _residual_reader(self, knob: _Knob) -> ResidualReader:
        """A reader of an open family's map with the named members taken back out."""

        def read(run: EvalRun, _results: Sequence[EvalResult], removed: frozenset[str]) -> dict[str, Any] | None:
            overlays = self._overlays_of(run)
            if overlays is None:
                return None
            entries = overlays.get(knob.field_name) or {}
            return {key: value for key, value in entries.items() if f"{knob.lever}.{key}" not in removed}

        return read


def freeze(validated: BaseModel | None) -> dict[str, Any]:
    """What is recorded of values a kind's model validated: the model's JSON form, or nothing.

    One form for every record of a kind's values — a template's ``kind_spec``, and a run's
    ``overlays`` and ``kind_spec`` — so what a run recorded and what its template stated are
    compared, and hashed, alike. The launch stamps it onto every run it creates; a host driving
    the runner without a launch stamps the same, from :meth:`KindContract.validate_overlays` and
    :meth:`KindContract.validate_spec`.

    **A set is recorded sorted.** A set has no order, but its JSON form is a list in whatever order
    the set iterates — which for strings is the process's hash seed — so one configuration would
    record a different list, and hash to a different key, in each process. Every consumer reads the
    record or the set itself, never the iteration order, so sorting changes nothing any of them sees.
    Elements sort by their canonical JSON, so a set of anything JSON can hold has one order.

    Args:
        validated: The validated model, or ``None`` for a kind that declares none.

    Returns:
        Every field of the model at the level it validated to; ``{}`` when there is no model.
    """
    if validated is None:
        return {}
    frozen: dict[str, Any] = _JSON_FORM.dump_python(_sorted_sets(validated.model_dump(mode="python")), mode="json")
    return frozen


def _sorted_sets(value: Any) -> Any:
    """``value`` with every set, at any depth, replaced by a list in one canonical order."""
    if isinstance(value, Mapping):
        return {key: _sorted_sets(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        members = [_sorted_sets(item) for item in value]
        return sorted(members, key=lambda member: canonical_json(_JSON_FORM.dump_python(member, mode="json")))
    if isinstance(value, (list, tuple)):
        return [_sorted_sets(item) for item in value]
    return value


def _excluded_paths(model: type[BaseModel], prefix: str = "", seen: frozenset[type] = frozenset()) -> list[str]:
    """Every field of ``model``, at any depth, that serialization leaves out — by dotted path.

    A field excluded from ``model_dump`` (``exclude=True`` or ``exclude_if``) is held by the validated
    model the launcher acts on and dropped from :func:`freeze`'s record, so two values of it would
    record alike. Nested models are walked through every container their annotation names.

    Args:
        model: The model to walk.
        prefix: The dotted path of ``model`` within the kind's model.
        seen: Models already on the path, so a recursive model is walked once.

    Returns:
        The excluded fields' paths, in declaration order.
    """
    paths: list[str] = []
    for name, info in model.model_fields.items():
        path = f"{prefix}{name}"
        if info.exclude or info.exclude_if is not None:
            paths.append(path)
        for nested in _models_in(info.annotation):
            if nested not in seen | {model}:
                paths.extend(_excluded_paths(nested, f"{path}.", seen | {model}))
    return paths


def _models_in(annotation: Any) -> list[type[BaseModel]]:
    """The Pydantic models an annotation holds, through any container or union."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    return [model for arg in typing.get_args(annotation) for model in _models_in(arg)]


def _member_test(member_prefix: str) -> FamilyMemberTest:
    """Whether a bare name is an entry of the family whose members start with ``member_prefix``."""

    def owns(name: str) -> bool:
        return name.startswith(member_prefix) and name != f"{member_prefix[:-1]}{_FAMILY_SUFFIX}"

    return owns


def _read_knob(prefix: str, name: str, info: FieldInfo) -> tuple[_Knob | None, list[str]]:
    """Read one overlay field into a knob, or say why it cannot be one.

    Args:
        prefix: The kind's lever prefix.
        name: The field's name.
        info: The field's Pydantic info.

    Returns:
        The knob (``None`` when there are problems) and the problems found.
    """
    problems: list[str] = []
    if not (info.description or "").strip():
        problems.append("has no description, and a lever with no prose reads as 'component 3 moved'")
    if info.alias is not None or info.validation_alias is not None:
        problems.append("declares an alias, and a launch names an overlay by its field")
    annotation = _unwrap_optional(info.annotation)
    ordinal = any(isinstance(marker, Ordinal) for marker in info.metadata)
    interval = next((marker for marker in info.metadata if isinstance(marker, Interval)), None)
    acts_on = [marker.measure for marker in info.metadata if isinstance(marker, ActsOn)]
    resolves_into = [marker.lever for marker in info.metadata if isinstance(marker, ResolvesInto)]
    member_acts_on = [marker.members for marker in info.metadata if isinstance(marker, MemberActsOn)]
    numeric = annotation in (int, float)
    family = typing.get_origin(annotation) in (dict, Mapping)
    levels: tuple[Any, ...] | None = None
    if ordinal:
        levels = _ordered_levels(annotation)
        if levels is None:
            problems.append("is marked Ordinal but is not a Literal or an Enum, so it has no declared order")
    if interval is not None and not numeric:
        problems.append("is marked Interval but is not an int or a float")
    if family and typing.get_args(annotation)[:1] != (str,):
        problems.append("is a map keyed by something other than str, and each key names a lever")
    if family and acts_on:
        problems.append(
            "is a map marked ActsOn, and its entries are distinct knobs one mechanism cannot speak for — "
            "name each entry's with MemberActsOn"
        )
    if len(acts_on) > 1:
        problems.append("is marked ActsOn more than once, and a lever names one mechanism")
    if member_acts_on and not family:
        problems.append("is marked MemberActsOn but is not a map, so it has no entries to name; mark it ActsOn")
    if len(member_acts_on) > 1:
        problems.append("is marked MemberActsOn more than once; name every entry's mechanism in one marker")
    problems.extend(
        f"names a blank entry or mechanism in MemberActsOn ({key!r}: {measure!r}), which could never be checked"
        for members in member_acts_on
        for key, measure in members
        if not key.strip() or not measure.strip()
    )
    if len(resolves_into) > 1:
        problems.append(
            "is marked ResolvesInto more than once, and a knob written into two surfaces could not be folded "
            "into either — each would also carry whatever the other did"
        )
    if problems:
        return None, problems
    return (
        _Knob(
            field_name=name,
            lever=f"{prefix}.{name}",
            prose=str(info.description),
            family=family,
            ordinal_levels=levels,
            interval_unit=interval.unit if interval is not None else None,
            numeric=numeric,
            acts_on=acts_on[0] if acts_on else None,
            resolves_into=resolves_into[0] if resolves_into else None,
            member_acts_on=member_acts_on[0] if member_acts_on else (),
        ),
        [],
    )


def _unwrap_optional(annotation: Any) -> Any:
    """``X | None`` read as ``X``; a ``None`` level is rendered on its own."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def _ordered_levels(annotation: Any) -> tuple[Any, ...] | None:
    """The declared levels of a ``Literal`` or ``Enum``, in order, as their JSON values."""
    if typing.get_origin(annotation) is Literal:
        return typing.get_args(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return tuple(member.value for member in annotation)
    return None


def _level(knob: _Knob, value: Any) -> SweepableValue:
    """One field's frozen value as a level on its axis."""
    if value is None:
        return SweepableValue.of(None, display="(none)")
    scale: Scale
    if knob.numeric and not isinstance(value, bool):
        scale = IntervalScale(value=float(value), unit=knob.interval_unit)
        return SweepableValue.of(value, scale=scale)
    if knob.ordinal_levels is not None and value in knob.ordinal_levels:
        scale = OrdinalScale(rank=knob.ordinal_levels.index(value))
    else:
        scale = NominalScale()
    return SweepableValue.of(value, display=_display(value), scale=scale)


def _display(value: Any) -> str:
    """A short rendering of a nominal level; the content hash, not this, is what joins it."""
    text = value if isinstance(value, str) else canonical_json(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    return text if len(text) <= _DISPLAY_LIMIT else f"{text[: _DISPLAY_LIMIT - 1]}…"


def _field_errors(refused: ValidationError) -> str:
    """Each refused field, named by its path, with what the model said of it."""
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or '(the overlays)'}: {error['msg']}"
        for error in refused.errors(include_url=False)
    )


__all__ = ["Interval", "KindContract", "KindContractError", "Ordinal", "freeze"]
