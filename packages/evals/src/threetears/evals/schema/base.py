"""The eval engine's own Pydantic bases: one stance, and the document model built from it.

Every eval model inherits :class:`EvalBaseModel` — mutable, strict, whitespace-stripping — or
:class:`EvalDocumentModel`, the same stance plus what a model serialized as a document needs.
The engine is installed in hosts that have no model layer of their own to lend it, so the stance
is spelled here, once, and ``tests/test_base.py`` pins its values literally: a change to it is
diff-visible on its own terms.

**Reads are strict, and so is construction.** ``extra="forbid"`` refuses an unknown key on every
path — a kwarg typed in code, a key in a stored document, a field in a payload a host supplied.
There is no filter on the way in: a filter would turn "this document is from another schema" into a
quietly different document, which is the one outcome worse than refusing it. ``schema_version`` on
the stored models says which schema wrote a document, and the two kinds of stored document read it
differently (:mod:`~threetears.evals.schema.versioning`):

- **A core document is kept** (:class:`CoreDocumentModel`). Its stored read upgrades an older core
  major step by step through registered upgraders before validating, and refuses a version it
  cannot read: a newer major, one from before the first release, or one that is not a number.
- **A regenerable document is refused at any other version**, and regenerated from the core.

**One tolerance, scoped to the case a contract makes safe: a core document of this build's major at
a newer minor.** A minor is additive only, so what a newer minor added is a field this build may
ignore. Its stored read keeps every field this build does not declare — at the top level and inside
every shape the document embeds (:class:`CoreShapeModel`) — on the instance it was found on, out of
every attribute, dump and digest, and its ``to_dict`` writes them back where they were. Only the
stored read is tolerant: construction and payload validation at a newer minor are refused, and a
core document at this build's minor or older is read exactly as strictly as any other. When two
builds share one store, the older refuses only what a newer MAJOR wrote, by name; that is the
deliberate cost of never guessing what a changed field means.

**The JSON pair is a transport API.** Every eval store path goes through :meth:`to_dict`, so
:meth:`to_json` / :meth:`from_json` serve a consumer that moves these models over a wire.

**The lone-surrogate fallback in :meth:`to_json` is the part not to lose.** Externally ingested
data reaches these models — scraped strings, provider responses — and a lone UTF-16 surrogate in
one of them fails serialization outright. :meth:`to_dict` deliberately does *not* sanitize,
because it never encodes and so never fails; a caller that hands its output to an encoder of its
own is the caller that meets the error. A declared ``str`` field cannot carry a lone surrogate at
all — pydantic refuses it during validation — so the fallback's live population is the free-form
``Any`` values (a host's opaque payload, a provider payload), and a test that reaches for a ``str``
field to exercise it proves nothing. A :data:`VerbatimJsonObject` refuses one too: what it promises
is storage exactly as written, and no UTF-8 store can hold a lone surrogate as written.

**Frozen has no sibling, and that absence is deliberate.** A model needing ``frozen=True``, or a
narrower option set, declares its own ``ConfigDict``. The stance that genuinely constrains a
*field* is ``str_strip_whitespace``: a model whose strings carry meaningful leading or trailing
space cannot use these bases at all, and says so where it declares its own config. **A field that
carries an opaque payload is the one exception made inside a model**: its strings, and its keys, are
the owner's bytes rather than text the engine reads, so it is typed :data:`VerbatimJsonObject` or
:data:`VerbatimObject`, which validate without stripping. Trimming there is not tidying: two keys
differing only by a space collapse into one and a value is silently lost.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Annotated, Any, ClassVar, Self, get_args

from pydantic import (
    AfterValidator,
    BaseModel,
    ValidationError,
    ConfigDict,
    JsonValue,
    PlainValidator,
    PrivateAttr,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_validator,
)

from threetears.evals.schema.versioning import (
    CORE_SCHEMA_MINOR,
    CORE_VERSION_SPACE,
    DURABLE_SCHEMA_MINOR,
    DURABLE_VERSION_SPACE,
    CoreVersionRefused,
    VersionSpace,
    read_core_minor,
    upgrade_document,
)
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: Lone UTF-16 surrogates, which are unencodable in UTF-8 and reach these models
#: from externally-ingested data. Replaced with U+FFFD rather than dropped, so the
#: damage stays visible in the stored document.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

#: Eval's model configuration, spelled out rather than imported. Every value here
#: is pinned by a test, so changing one is a deliberate, diff-visible act.
_EVAL_BASE_CONFIG_OPTIONS = ConfigDict(
    # Validation behavior
    extra="forbid",  # Reject unknown fields (catches typos)
    validate_assignment=True,  # Validate on attribute assignment
    str_strip_whitespace=True,  # Strip whitespace from strings
    # Serialization behavior
    ser_json_inf_nan="constants",  # Handle infinity/NaN in JSON
    # Schema generation
    populate_by_name=True,  # Allow both alias and field name
)


#: Validates without the stance's whitespace stripping: an opaque payload's strings are its owner's.
_VERBATIM = ConfigDict(str_strip_whitespace=False)
_VERBATIM_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue], config=_VERBATIM)
_VERBATIM_OBJECT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any], config=_VERBATIM)


#: A JSON object stored exactly as its owner wrote it — every key and every string, whitespace
#: included — and read by nothing in the engine. JSON values only, refused at construction, so a
#: value storage could not hold as written (an object, a lone surrogate) is refused where it is built
#: rather than at the write. The type of a kind's ``kind_payload``.
def _verbatim_json_object(value: Any) -> dict[str, JsonValue]:
    """Validate an opaque JSON object without trimming it, refusing what no store can hold as written.

    Args:
        value: The payload as its owner built it.

    Returns:
        The payload, every key and string as written.

    Raises:
        ValueError: A value that is not JSON, or a lone surrogate, which no UTF-8 store can encode.
    """
    validated = _VERBATIM_JSON_OBJECT.validate_python(value)
    try:
        json.dumps(validated, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError as unencodable:
        raise ValueError(
            "carries a lone UTF-16 surrogate, which no UTF-8 store can hold as written; "
            "a verbatim payload is stored exactly or not at all"
        ) from unencodable
    return validated


VerbatimJsonObject = Annotated[dict[str, JsonValue], PlainValidator(_verbatim_json_object)]

#: A host's opaque payload: its top-level keys kept exactly as written, its values unread and
#: unvalidated. The type of ``host_payload`` on a run and a test case, which the engine never reads.
VerbatimObject = Annotated[dict[str, Any], PlainValidator(_VERBATIM_OBJECT.validate_python)]

#: A string kept exactly as written, leading and trailing whitespace included: a serialized body (a CSV
#: export, a report) whose bytes are its producer's. The stance strips every other string, and a stripped
#: body differs from the one its serializer wrote — a CSV loses its final row terminator, a report its
#: final newline — so two surfaces handing back one export would not be byte-identical.
VerbatimText = Annotated[str, StringConstraints(strip_whitespace=False)]


class EvalBaseModel(BaseModel):
    """Base model for the eval engine's own Pydantic models.

    Features:
        - Strict validation (rejects unknown fields)
        - Transport-safe JSON serialization, including a lone-surrogate fallback
        - Convenience JSON and dict serialization methods

    Example:
        class MyModel(EvalBaseModel):
            id: str
            name: str
    """

    model_config = ConfigDict(**_EVAL_BASE_CONFIG_OPTIONS)

    # =========================================================================
    # JSON Serialization
    # =========================================================================

    def to_json(self, *, indent: int | None = None, exclude_none: bool = False) -> str:
        r"""Serialize to JSON string.

        Strips lone UTF-16 surrogates (\uD800-\uDFFF) that can appear in data
        ingested from external sources and would otherwise cause
        ``UnicodeEncodeError`` during JSON serialization.

        Args:
            indent: Indentation level for pretty printing (None for compact)
            exclude_none: If True, exclude fields with None values

        Returns:
            JSON string representation
        """
        try:
            return self.model_dump_json(indent=indent, exclude_none=exclude_none)
        except (UnicodeEncodeError, ValueError) as e:
            # Fallback: sanitize surrogates and retry via dict path
            log.debug("model_dump_json failed for %s, falling back to surrogate-safe path: %s", type(self).__name__, e)
            data = self.model_dump(mode="json", exclude_none=exclude_none)
            raw = json.dumps(data, indent=indent, ensure_ascii=False, default=str)
            return _LONE_SURROGATE_RE.sub("\ufffd", raw)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            Model instance

        Raises:
            ValidationError: If JSON is invalid or fails validation
        """
        return cls.model_validate_json(json_str)

    # =========================================================================
    # Dict Serialization
    # =========================================================================

    def to_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """Convert to dictionary.

        Serializes enums to their values for JSON compatibility.

        Args:
            exclude_none: If True, exclude fields with None values
            exclude_unset: If True, exclude fields that were not explicitly set
            by_alias: If True, use field aliases as keys

        Returns:
            Dictionary representation with JSON-compatible types
        """
        return self.model_dump(
            mode="json",  # Serialize enums to values, datetimes to strings, etc.
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            by_alias=by_alias,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create instance from dictionary.

        Args:
            data: Dictionary with field values

        Returns:
            Model instance

        This is the stored read: every load path in the engine goes through it, and it is as strict as
        construction.

        Raises:
            ValidationError: If data fails validation
        """
        return cls.model_validate(data)


class EvalDocumentModel(EvalBaseModel):
    """The base of every eval model serialized as a document — stored, or served to a reader.

    The stance is :class:`EvalBaseModel`'s, spread from the same option table rather than retyped,
    so it has exactly one spelling. Two additions, both about the document a model becomes:

    - ``json_schema_serialization_defaults_required=True``. Serialization always emits defaulted
      fields, so the OUTPUT JSON schema marks them required, and a client coding against it
      never guards ``id`` / ``status`` / ``created_at`` as optional when they cannot be absent.
    - :meth:`_discard_computed_field_echo`. ``to_dict`` emits every ``computed_field``, so every
      stored or served document carries this model's own derived values; they are re-derived on
      validation, so the echo is discarded rather than believed or refused. That is what keeps
      ``Model.from_dict(model.to_dict())`` a round trip. It is not a tolerance: any other unknown
      key is refused, as on the base.
    """

    model_config = ConfigDict(**_EVAL_BASE_CONFIG_OPTIONS, json_schema_serialization_defaults_required=True)

    @model_validator(mode="before")
    @classmethod
    def _discard_computed_field_echo(cls, data: Any) -> Any:
        """Remove this model's computed-field keys from the input, silently, on every path.

        Args:
            data: The input under validation; only a mapping is touched.

        Returns:
            The input without computed-field keys.
        """
        return without_computed_field_echo(cls, data)


def without_computed_field_echo(model: type[BaseModel], data: Any) -> Any:
    """``data`` without ``model``'s computed-field keys — the echo a dump of it carries, which a read re-derives.

    :class:`EvalDocumentModel`'s rule, as a function, for a served model that keeps a stance of its own (a frozen
    summary) and so cannot inherit it: the derived values are discarded rather than believed or refused, and any
    other unknown key is still refused by the model's own ``extra="forbid"``.

    Args:
        model: The model under validation.
        data: The input under validation; only a mapping is touched.

    Returns:
        The input without computed-field keys.
    """
    derived = model.model_computed_fields
    if not derived or not isinstance(data, Mapping) or not any(key in derived for key in data):
        return data
    return {key: value for key, value in data.items() if key not in derived}


#: The validation-context key :meth:`CoreDocumentModel.from_dict` sets, and nothing else does: this validation is
#: the stored read of a document of this build's major at a newer minor, whose value is that minor.
_NEWER_MINOR_READ = "threetears.evals.newer_minor_read"


def _newer_minor_read(info: ValidationInfo) -> int | None:
    """The newer minor a stored read is reading at, or ``None`` when this validation is not one."""
    context = info.context
    if not isinstance(context, Mapping):
        return None
    minor = context.get(_NEWER_MINOR_READ)
    return minor if type(minor) is int else None


@dataclass(frozen=True)
class _NewerMinorFields:
    """What a newer minor's stored read left on one shape: the minor it was read at, and the fields it kept.

    Set on every shape a newer minor's read validates, whether or not it held a field this build does not declare,
    so a shape carrying only a value of an open vocabulary this build does not know still carries its minor into
    any document it is moved to. A plain ``dict``, copied on the read and again on every write, so the marker
    deep-copies and pickles with its instance; nothing mutates it.

    Attributes:
        minor: The minor the document it was read from was written at.
        fields: Each undeclared key and its stored value, exactly as read; empty when there were none.
    """

    minor: int
    fields: dict[str, Any]


@cache
def _fields_holding_shapes(model: type[BaseModel]) -> tuple[str, ...]:
    """The fields of ``model`` whose declared type can hold a model, which a write walks for kept fields.

    Args:
        model: A model class.

    Returns:
        The names, in declaration order.
    """

    def holds_a_model(annotation: Any) -> bool:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return True
        return any(holds_a_model(argument) for argument in get_args(annotation))

    return tuple(name for name, field in model.model_fields.items() if holds_a_model(field.annotation))


class CoreShapeModel(BaseModel):
    """A model the evidence core stores: a core document, or a shape one embeds, at any depth.

    It declares no field and no configuration; it carries the one tolerance the core has
    (the module's rule). On the stored read of a document of this build's major at a newer minor —
    and only there, signalled by a validation context only :meth:`CoreDocumentModel.from_dict` sets —
    the keys a stored mapping holds that this class does not declare are split off before the class's
    own strict validation runs, and kept on the instance that validation returns. The split is each
    class's own field declarations (its fields, their aliases, its computed fields), and pydantic
    carries the context into every nested validation — a list item, a dict value, an optional, a
    discriminated union's member — so every level keeps its own, with no type walked here. Whatever
    the class declares is validated exactly as it is anywhere else, ``extra="forbid"`` included.

    The kept fields are not data: they are no attribute, and appear in no ``model_dump``,
    ``model_dump_json``, ``repr`` or JSON schema, so no digest a pre-image is taken over a dump of can
    reach them. :meth:`CoreDocumentModel.to_dict` alone writes them back. They travel with the
    instance through ``model_copy`` (with or without ``update``), a validated assignment, a deep copy
    and a pickle; an instance built afresh from a read one is a new instance and carries none, which is
    why construction at a newer minor is refused (:data:`CoreSchemaMinor`).

    Every model a core document embeds carries this base; ``tests/test_core_minor.py`` walks the core's
    declarations and names one that does not.
    """

    _newer_minor_fields: _NewerMinorFields | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _keep_newer_minor_fields(cls, data: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo) -> Any:
        """On a newer minor's stored read, validate what this class declares and keep the rest on the instance.

        Args:
            data: The input under validation; only a mapping, on a newer minor's stored read, is split.
            handler: The class's own validation.
            info: The validation's context, which says whether this is that read.

        Returns:
            The validated instance, marked with the minor it was read at and carrying the fields it does not
            declare (none, often).
        """
        minor = _newer_minor_read(info)
        if minor is None or not isinstance(data, Mapping):
            return handler(data)
        declared = cls._declared_keys()
        undeclared = {key: value for key, value in data.items() if key not in declared}
        shape: CoreShapeModel = handler({key: value for key, value in data.items() if key in declared})
        shape._newer_minor_fields = _NewerMinorFields(minor=minor, fields=copy.deepcopy(undeclared))
        return shape

    def __eq__(self, other: object) -> bool:
        """Equal when of one type, with equal declared fields and equal kept fields; the minor marker is not compared.

        Pydantic's own equality compares every private attribute, and every shape a newer minor's read validates
        carries the minor it was read at. So under it a shape read from a document a newer build wrote — even one
        that kept no field at all, because the newer minor changed some other type — would be unequal to the same
        shape built here, and every check that compares a stored shape with one this build holds (a run's request
        settings against today's, a control's lever map echoed back) would refuse every document a newer minor
        wrote. Two shapes are equal when they hold the same thing: the same declared field values, compared as
        pydantic compares them (a nested shape by this same rule), the same fields a newer build wrote and this
        build kept, and the same value for every other private attribute a subclass declares. Where the marker
        came from is not what a shape holds.

        Defining equality here sets this class's ``__hash__`` to ``None``, which is what pydantic's own mutable
        models have; a frozen subclass is still given pydantic's field-only hash at class creation (its metaclass
        generates one when the inherited ``__hash__`` is ``None``), and that hash agrees with this equality: two
        equal shapes have equal declared fields, so equal hashes.

        Args:
            other: What this shape is compared with.

        Returns:
            Whether the two hold the same thing; ``NotImplemented`` for an object that is not a model.
        """
        if not isinstance(other, BaseModel):
            return NotImplemented
        if not isinstance(other, CoreShapeModel) or type(self) is not type(other):
            return False
        return self._same_contents(other)

    def _same_contents(self, other: CoreShapeModel) -> bool:
        """:meth:`__eq__` for two shapes of one type: their declared fields, kept fields and other private state."""
        names = type(self).model_fields.keys()
        if any(self.__dict__.get(name) != other.__dict__.get(name) for name in names):
            return False
        return self._compared_private(self) == self._compared_private(other)

    def _compared_private(self, shape: CoreShapeModel) -> dict[str, Any]:
        """``shape``'s private state as equality reads it: its kept fields in place of the marker that carries them."""
        private = dict(shape.__pydantic_private__ or {})
        kept = private.pop("_newer_minor_fields", None)
        private["_newer_minor_fields"] = {} if kept is None else kept.fields
        return private

    @classmethod
    def _declared_keys(cls) -> frozenset[str]:
        """Every key a mapping of this class may hold: its fields, their aliases, and its computed fields."""
        keys = set(cls.model_fields) | set(cls.model_computed_fields)
        for field in cls.model_fields.values():
            keys.update(alias for alias in (field.alias, field.validation_alias) if isinstance(alias, str))
        return frozenset(keys)

    def _write_kept_fields(
        self, shape: CoreShapeModel, dumped: dict[str, Any], minors: list[int], *, by_alias: bool
    ) -> None:
        """Put back into ``shape``'s dump what a newer minor's read kept on it and on every shape below it.

        Args:
            shape: A shape, and ``dumped`` its dump.
            dumped: The mapping ``to_dict`` dumped ``shape`` to, written into in place.
            minors: Collects the minor of every shape a newer minor's read marked, so the document is stamped no
                older than anything it carries — a kept field, or a value of an open vocabulary this build does not
                know.
            by_alias: Whether the dump is keyed by alias.
        """
        kept = shape._newer_minor_fields
        if kept is not None:
            dumped.update(copy.deepcopy(kept.fields))
            minors.append(kept.minor)
        model_fields = type(shape).model_fields
        for name in _fields_holding_shapes(type(shape)):
            field = model_fields[name]
            key = (field.serialization_alias or field.alias or name) if by_alias else name
            if key in dumped:
                self._write_kept_value(getattr(shape, name), dumped[key], minors, by_alias=by_alias)

    def _write_kept_value(self, value: Any, dumped: Any, minors: list[int], *, by_alias: bool) -> None:
        """:meth:`_write_kept_fields` for one field's value: a shape, or a list, tuple or mapping holding some.

        Raises:
            ValueError: ``value`` holds kept fields where its dump does not mirror it, so they cannot be put back
                where they were read; refused rather than dropped.
        """
        if isinstance(value, CoreShapeModel) and isinstance(dumped, dict):
            self._write_kept_fields(value, dumped, minors, by_alias=by_alias)
        elif isinstance(value, (list, tuple)) and isinstance(dumped, list) and len(dumped) == len(value):
            for item, dumped_item in zip(value, dumped, strict=True):
                self._write_kept_value(item, dumped_item, minors, by_alias=by_alias)
        elif isinstance(value, Mapping) and isinstance(dumped, dict) and all(isinstance(key, str) for key in value):
            for key, item in value.items():
                if key in dumped:
                    self._write_kept_value(item, dumped[key], minors, by_alias=by_alias)
        elif self._holds_kept_fields(value):
            raise ValueError(
                f"{type(self).__name__} holds fields a newer minor wrote in a value its dump does not mirror "
                f"({type(value).__name__}), so they cannot be written back where they were read"
            )

    def _holds_kept_fields(self, value: Any) -> bool:
        """Whether ``value`` holds, at any depth, a shape carrying fields a newer minor's read kept."""
        if isinstance(value, CoreShapeModel):
            return self._shape_holds_kept_fields(value)
        if isinstance(value, Mapping):
            return any(self._holds_kept_fields(item) for item in value.values())
        if isinstance(value, Collection) and not isinstance(value, (str, bytes)):
            return any(self._holds_kept_fields(item) for item in value)
        return False

    def _shape_holds_kept_fields(self, shape: CoreShapeModel) -> bool:
        """:meth:`_holds_kept_fields` for one shape: its own kept fields, or any below it."""
        if shape._newer_minor_fields is not None and shape._newer_minor_fields.fields:
            return True
        return any(self._holds_kept_fields(getattr(shape, name)) for name in _fields_holding_shapes(type(shape)))


def _built_at_this_minor(space: VersionSpace) -> AfterValidator:
    """The validator of a ``schema_minor`` in ``space``: this build's minor, or the newer one its own stored read is
    reading.

    A stored read at this build's minor or older reaches it at this build's minor (its read stamps it, and the
    document now holds this build's defaults); a newer minor's read reaches it at that minor, carrying what it
    kept. Anything else is a construction or a payload naming a minor this build did not write.

    Args:
        space: The version space whose minor the field holds.

    Returns:
        An ``AfterValidator`` for ``Annotated[StrictInt, ...]``.
    """

    def at_this_minor(minor: int, info: ValidationInfo) -> int:
        if minor == space.minor or minor == _newer_minor_read(info):
            return minor
        raise ValueError(
            f"a {space.name} document is built at {space.name} minor {space.minor}, not {minor}; a newer minor is read "
            "only by its stored read (from_dict), which keeps the fields this build does not know"
        )

    return AfterValidator(at_this_minor)


#: The ``schema_minor`` field type of every core document: this build's minor, or the newer one its stored read is
#: reading at.
CoreSchemaMinor = Annotated[StrictInt, _built_at_this_minor(CORE_VERSION_SPACE)]

#: The ``schema_minor`` field type of every durable document: :data:`CoreSchemaMinor` in the durable space.
DurableSchemaMinor = Annotated[StrictInt, _built_at_this_minor(DURABLE_VERSION_SPACE)]


def known_unless_newer_minor(vocabulary: str, known: tuple[str, ...]) -> AfterValidator:
    """The validator an open vocabulary's stored field carries: a value this build knows, or a newer minor's.

    An open vocabulary grows by a minor: a new value is additive, because every reader that branches on the
    vocabulary has a branch for a value it does not know that names it. So a value outside ``known`` is read on
    a newer minor's stored read, and refused everywhere else — a writer at this build's minor cannot have written
    it, which makes it drift or a typo.

    Args:
        vocabulary: What a value names, for the refusal (``"usage role"``).
        known: The values this build knows, in the order its readers list them.

    Returns:
        An ``AfterValidator`` for ``Annotated[str, ...]``.
    """

    def known_or_newer(value: str, info: ValidationInfo) -> str:
        if value in known or _newer_minor_read(info) is not None:
            return value
        raise ValueError(
            f"{value!r} is not a {vocabulary} this build knows ({', '.join(known)}); a {vocabulary} this build does "
            "not know is read only from a document a newer minor wrote"
        )

    return AfterValidator(known_or_newer)


class CoreDocumentModel(EvalDocumentModel, CoreShapeModel):
    """A stored document in the evidence core: kept across releases, upgraded when read at an older major, and
    kept whole when read at a newer minor.

    **Which version space it is read in is the class's** (:attr:`version_space`): the core's here, and the durable
    space's on :class:`DurableDocumentModel`, which is this class under another space's numbers. Every rule below
    reads "core" as "the class's space".

    Its stored read, :meth:`from_dict`, upgrades the document through its space's chain (the durable space's
    :data:`~threetears.evals.schema.versioning.DURABLE_UPGRADERS`; the core has none and reads only its own major)
    before validating it, so every validator — the model's own and its bases' — sees today's shape. Upgrading
    before validation rather than in a ``before`` validator is deliberate: pydantic runs a concrete
    model's own ``before`` validators ahead of one it inherits, so an inherited upgrade could not
    promise to run first. A refusal (a newer major, a pre-release version, no version number at all,
    or a minor that is not a whole number) is raised as a ``ValidationError`` on ``schema_version`` or
    ``schema_minor``, so every caller that isolates an unreadable stored document already isolates it.

    A document of this build's major at a newer minor is read with the one tolerance the core has
    (:class:`CoreShapeModel`), and :meth:`to_dict` writes what it kept back. Every other path stays
    strict: constructing a core document, or validating a payload, at any major but the current one,
    or at any minor but this build's, is refused by the ``schema_version`` and ``schema_minor`` fields
    themselves.

    **Only stored bytes reach the tolerant read.** :meth:`from_dict` is the stored read; every caller
    in the engine hands it what a store returned, or a ``to_dict`` of a document a store returned. A
    payload a caller supplied — an action's arguments, a request body, a host's dict, a document on a
    wire (:meth:`from_json`) — is validated with ``model_validate``, which never tolerates.
    """

    schema_minor: CoreSchemaMinor = CORE_SCHEMA_MINOR

    #: The version space a stored read of this class reads in: its numbers, its baseline and its upgrade chain.
    version_space: ClassVar[VersionSpace] = CORE_VERSION_SPACE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Read a stored core document: upgraded to the current major first, and kept whole at a newer minor.

        Args:
            data: The stored document, as the store returned it.

        Returns:
            Model instance, at the current core major. Its ``schema_minor`` is this build's, or the newer minor
            the document was written at, whose fields this build does not know it keeps.

        Raises:
            ValidationError: The document's version or minor is not one this build reads, or the upgraded
                document fails validation.
        """
        if not isinstance(data, Mapping):
            return cls.model_validate(data)
        space = cls.version_space
        try:
            upgraded = upgrade_document(
                {"doc_type": cls.model_fields["doc_type"].default, **data},
                steps=space.upgraders,
                current=space.version,
                baseline=space.baseline,
                space=space.name,
            )
        except CoreVersionRefused as refused:
            raise cls._refused("schema_version", data.get("schema_version"), refused) from refused
        try:
            stored_minor = read_core_minor(data, space=space.name)
        except CoreVersionRefused as refused:
            raise cls._refused("schema_minor", data.get("schema_minor"), refused) from refused
        newer = data.get("schema_version") == space.version and stored_minor > space.minor
        upgraded["schema_minor"] = stored_minor if newer else space.minor
        if newer:
            return cls.model_validate(upgraded, context={_NEWER_MINOR_READ: stored_minor})
        return cls.model_validate(upgraded)

    @classmethod
    def _refused(cls, field: str, value: Any, refused: CoreVersionRefused) -> ValidationError:
        """A version refusal as the ``ValidationError`` on ``field`` every caller already isolates."""
        return ValidationError.from_exception_data(
            cls.__name__,
            [{"type": "value_error", "loc": (field,), "input": value, "ctx": {"error": refused}}],
        )

    def to_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_unset: bool = False,
        by_alias: bool = False,
    ) -> dict[str, Any]:
        """The document as it is written: every declared field, and every field a newer minor's read kept.

        The kept fields go back where they were read, at whatever depth, exactly as stored — ``exclude_none``
        and ``exclude_unset`` apply to the fields this build declares, never to another build's bytes — and
        ``schema_minor`` is stamped with the newest minor any shape it carries was read at, or this document's own
        when that is newer. So a rewrite of a document a newer build wrote keeps every field that build wrote on
        each shape the rewrite carries over, and a shape carried into another document carries its minor with it.
        A shape the rewrite REPLACES with one built afresh (a renewed lease, a recomputed completeness record)
        takes its kept fields with it: they described the shape that was replaced.

        Args:
            exclude_none: If True, exclude declared fields with None values
            exclude_unset: If True, exclude declared fields that were not explicitly set
            by_alias: If True, use field aliases as keys

        Returns:
            Dictionary representation with JSON-compatible types

        Raises:
            ValueError: A kept field sits in a value whose dump does not mirror it, so it cannot be written
                back where it was read.
        """
        document = super().to_dict(exclude_none=exclude_none, exclude_unset=exclude_unset, by_alias=by_alias)
        minors: list[int] = []
        self._write_kept_fields(self, document, minors, by_alias=by_alias)
        if "schema_minor" in document or minors:
            document["schema_minor"] = max([self.schema_minor, *minors])
        return document

    def to_json(self, *, indent: int | None = None, exclude_none: bool = False) -> str:
        r"""Serialize to a JSON string, for a wire: refused when the document holds fields a newer minor wrote.

        The JSON pair is a transport of what this build declares, and :meth:`from_json` reads it as strictly as
        any payload. A document read at a newer minor that kept fields this build does not know cannot travel
        that way whole, and dropping them silently is the one outcome worse than refusing, so it is refused by
        name: write it through :meth:`to_dict`, which every store path does. Lone UTF-16 surrogates
        (\uD800-\uDFFF) are replaced with U+FFFD, as the base does.

        Args:
            indent: Indentation level for pretty printing (None for compact)
            exclude_none: If True, exclude fields with None values

        Returns:
            JSON string representation

        Raises:
            ValueError: The document, or a shape it carries, holds fields a newer minor's stored read kept.
        """
        if self._shape_holds_kept_fields(self):
            raise ValueError(
                f"{type(self).__name__} '{getattr(self, 'id', '?')}' holds fields a newer minor wrote, which this build "
                "does not know; to_json is a transport of what this build declares and would drop them, so it is "
                "refused — write the document through to_dict, which keeps them"
            )
        return super().to_json(indent=indent, exclude_none=exclude_none)

    @classmethod
    def from_json(cls, json_str: str) -> Self:
        """Read a core document from a wire, as strictly as any payload: never the stored read's tolerance.

        A document a transport carries is caller input, so it is validated as construction is: a newer minor
        is refused, an unknown field is refused, and an older major is refused rather than upgraded. A stored
        document is read with :meth:`from_dict`.

        Args:
            json_str: JSON string to parse

        Returns:
            Model instance

        Raises:
            ValidationError: If the JSON is invalid or fails validation
        """
        return cls.model_validate_json(json_str)


class DurableDocumentModel(CoreDocumentModel):
    """A stored document people author: :class:`CoreDocumentModel` read in the durable version space.

    Every rule of the core's stored read and write holds, under the durable space's numbers
    (:data:`~threetears.evals.schema.versioning.DURABLE_VERSION_SPACE`): its baseline never moves, and its majors
    always ship an upgrader (:mod:`~threetears.evals.schema.versioning`). Internal: a host never names it, since a
    durable model is one of the engine's eight, each exported under its own name.
    """

    schema_minor: DurableSchemaMinor = DURABLE_SCHEMA_MINOR

    version_space: ClassVar[VersionSpace] = DURABLE_VERSION_SPACE


__all__ = [
    "CoreDocumentModel",
    "CoreSchemaMinor",
    "CoreShapeModel",
    "DurableDocumentModel",
    "DurableSchemaMinor",
    "EvalBaseModel",
    "EvalDocumentModel",
    "VerbatimJsonObject",
    "VerbatimObject",
    "VerbatimText",
    "known_unless_newer_minor",
    "without_computed_field_echo",
]
