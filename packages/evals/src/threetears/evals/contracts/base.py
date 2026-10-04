"""The eval engine's own Pydantic bases: one stance, and the document model built from it.

Every eval model inherits :class:`EvalBaseModel` — mutable, strict, whitespace-stripping — or
:class:`EvalDocumentModel`, the same stance plus what a model serialized as a document needs.
The engine is installed in hosts that have no model layer of their own to lend it, so the stance
is spelled here, once, and ``tests/test_base.py`` pins its values literally: a change to it is
diff-visible on its own terms.

**Reads are strict, and so is construction.** ``extra="forbid"`` refuses an unknown key on every
path — a kwarg typed in code, a key in a stored document, a field in a payload a host supplied.
There is no tolerant read: stored eval documents are disposable, so a document written under a
shape this build does not declare is dropped and regenerated, never migrated or filtered on the
way in. A filter would turn "this document is from another schema" into a quietly different
document, which is the one outcome worse than refusing it. ``schema_version`` on the stored
models says which schema wrote a document, and a read refuses any other
(:data:`~threetears.evals.contracts.models.EVAL_SCHEMA_VERSION`).

That stance has a known cost, stated so it is a decision rather than an oversight: when two
builds share one store, the older refuses what the newer wrote. While one build writes and reads
every eval document that is a typo caught early; the second concurrent writer is when it has to
be re-decided.

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

import json
import re
from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, JsonValue, PlainValidator, TypeAdapter, model_validator

from threetears.observe import get_logger

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
        derived = cls.model_computed_fields
        if not derived or not isinstance(data, Mapping) or not any(key in derived for key in data):
            return data
        return {key: value for key, value in data.items() if key not in derived}


__all__ = [
    "EvalBaseModel",
    "EvalDocumentModel",
    "VerbatimJsonObject",
    "VerbatimObject",
]
