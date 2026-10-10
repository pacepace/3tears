"""The JSON Schema subset a world dimension may declare, and the check a seeded value must pass against it.

A dimension's ``schema`` says what its value looks like. Two readers use it: the conformance kit, which
synthesizes values from it to prove the host's handles, and the seed walk, which refuses a seeded value that
does not conform. Both ask :func:`honoured_kind` which keyword set serves a schema, and that one function
refuses whatever it cannot serve — so a schema the kit can generate from is exactly a schema a seed can be
checked against, and a keyword or shape one of them would ignore is refused by both.

A value that may take one of several shapes is an ``anyOf``, each branch a schema in its own right. It exists
for the dimension whose empty value is itself a statement and whose stated value must be complete — an idle
worker's ``{}`` beside a running job carrying every field — which no other honoured keyword can say: a
``required`` refuses the empty value, and leaving it out admits a half-stated one.

The check is about presence and type, never plausibility. A host that requires a timestamp gets one; whether
it is a sensible timestamp is the author's, and a run seeded with implausible values is suspect in the way any
eval with an unrealistic fixture is — not refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from threetears.evals.schema.prose import PROSE_SCHEMA_KEY
from threetears.evals.schema.schema_nesting import NestedSchema, nested_schemas

__all__ = [
    "ANNOTATION_KEYWORDS",
    "HONOURED_KEYWORDS",
    "NestedSchema",
    "UnsupportedSchemaError",
    "honoured_kind",
    "json_equal",
    "nested_schemas",
    "schema_violations",
]

#: Keywords that describe a schema without constraining its values, so ignoring one cannot admit a value the
#: schema forbids. The prose marker is one: it says who writes the string (a model), which the authoring gate
#: reads, and admits every string it did before.
ANNOTATION_KEYWORDS = frozenset(
    {"title", "description", "$comment", "default", "examples", "deprecated", "readOnly", "writeOnly", PROSE_SCHEMA_KEY}
)

#: The constraint keywords honoured for each way a schema names its values, keyed by what
#: :func:`honoured_kind` resolves the schema to. ``enum`` and ``const`` are keys too, because they enumerate
#: the values outright and win over ``type``; they honour only themselves, since neither reader checks that
#: the enumerated values satisfy a bound declared beside them, so neither can promise anything about one.
#: ``anyOf`` honours only itself for the same reason, and refuses a ``type`` beside it as well: each branch
#: states its own type, and a constraint written beside the branches would be one neither reader applies.
#:
#: **A keyword outside its set is refused, exactly as an unknown ``type`` is.** The two are one rule and
#: were twice not: first an unsupported constraint was silently ignored while an unsupported type was loud
#: (so ``exclusiveMinimum: 0`` was synthesized as 0), then the audit lived in the type branch that ``enum``
#: and ``const`` never reach (so ``{"enum": [1, 2, 3], "minimum": 2}`` still yielded 1). Ignoring a keyword
#: lets a value through that the host's own schema forbids, and nothing downstream can tell that happened.
HONOURED_KEYWORDS: Mapping[str, frozenset[str]] = {
    "enum": frozenset({"enum"}),
    "const": frozenset({"const"}),
    "boolean": frozenset(),
    "integer": frozenset({"minimum", "maximum"}),
    "number": frozenset({"minimum", "maximum"}),
    "string": frozenset({"minLength", "maxLength"}),
    "array": frozenset({"minItems", "maxItems", "items"}),
    "object": frozenset({"properties", "required", "additionalProperties"}),
    "anyOf": frozenset({"anyOf"}),
}

#: The resolutions that do not name a JSON type: the two that enumerate values, and the one that offers shapes.
#: Checked in this order, so a schema carrying two of them resolves to the first and the second is refused as a
#: keyword outside its set.
_NOT_TYPES = ("enum", "const", "anyOf")

#: The resolutions that name a JSON type.
_TYPES = tuple(kind for kind in HONOURED_KEYWORDS if kind not in _NOT_TYPES)


class UnsupportedSchemaError(ValueError):
    """A schema uses a construct outside :data:`HONOURED_KEYWORDS` — a gap to extend, never a value to pass."""


def honoured_kind(schema: Any, *, at: str) -> str:
    """Which keyword set serves ``schema`` — the one resolution both readers of world schemas use.

    ``enum`` and ``const`` win over ``type``, because they enumerate the values outright, and ``anyOf`` resolves
    to itself, its branches each a schema. Otherwise the schema must name exactly one type this module knows.
    Every keyword beside the resolved one must be honoured for it (or be an annotation), and every schema
    nested in it (:func:`nested_schemas` names the positions) must resolve too — the whole tree is audited
    here rather than wherever a reader happens to descend, because the kit and the seed check descend into
    different parts of it.

    Refused, and why each is a refusal rather than a pass:

    * **No ``type``, ``enum`` or ``const``.** Nothing says what the value looks like, so a ``required`` or a
      bound beside it would go unchecked by one reader while the other could synthesize nothing.
    * **A union ``type``.** Taking one member would speak for only part of the declaration —
      ``{"type": ["string", "null"]}`` would generate strings and never say ``null`` was dropped — and
      accepting any member would check a value the kit can never have proved a handle for.
    * **A keyword outside the resolved set.** See :data:`HONOURED_KEYWORDS`.
    * **A ``required`` property the schema never describes.** Nothing can build a valid instance, and a
      seed carrying the key would be checked against nothing.
    * **An ``anyOf`` that is not a non-empty list of schemas, or carries a ``type`` beside it.** An empty one
      admits nothing, and a ``type`` beside the branches would be a second statement of the shape that
      neither reader consults.

    Args:
        schema: The schema, or one nested in it.
        at: Where the schema sits — a dimension name or a seed path — for the message.

    Returns:
        A key of :data:`HONOURED_KEYWORDS`.

    Raises:
        UnsupportedSchemaError: The schema, or one nested in it, is outside the honoured subset. The message
            names where and what, and is phrased for either reader.
    """
    kind = _resolve(schema, at=at)
    if kind == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise UnsupportedSchemaError(f"{at} declares properties {properties!r}, which is not a mapping of schemas")
        if undescribed := sorted(set(schema.get("required") or ()) - set(properties)):
            raise UnsupportedSchemaError(
                f"{at} declares {', '.join(repr(word) for word in undescribed)} required and describes no schema for "
                f"{'them' if len(undescribed) > 1 else 'it'}: {schema!r}. Nothing can build a valid instance, so "
                "describe the property or drop it from 'required'"
            )
    # Every nesting position, unconditionally: ``_resolve`` has already refused a nesting keyword outside the
    # resolved kind's set, so only the positions this kind honours can be present here.
    for nested in nested_schemas(schema):
        honoured_kind(nested.schema, at=nested.at(at))
    return kind


def _resolve(schema: Any, *, at: str) -> str:
    """The keyword set serving ``schema`` itself, its own keywords audited — :func:`honoured_kind` minus the nesting."""
    if not isinstance(schema, Mapping):
        raise UnsupportedSchemaError(f"{at} declares schema {schema!r}, which is not a JSON Schema object")
    kind = next((keyword for keyword in _NOT_TYPES if keyword in schema), None)
    if kind == "anyOf":
        branches = schema["anyOf"]
        if not isinstance(branches, list) or not branches:
            raise UnsupportedSchemaError(
                f"{at} declares anyOf {branches!r}, which is not a non-empty list of schemas — it offers no shape, "
                "so it admits no value"
            )
    elif kind is None:
        declared = schema.get("type")
        if declared is None:
            raise UnsupportedSchemaError(
                f"{at} declares schema {schema!r}, which names no type and enumerates no values — nothing says "
                "what its value looks like, so any constraint beside it would go unread"
            )
        if isinstance(declared, list) and len(declared) != 1:
            raise UnsupportedSchemaError(
                f"{at} declares schema {schema!r}, whose type is a union of "
                f"{', '.join(repr(member) for member in declared)}. A world schema names one type: taking one of "
                "several would speak for only part of the declaration — declare the type the dimension actually takes"
            )
        kind = declared[0] if isinstance(declared, list) else declared
        if not isinstance(kind, str) or kind not in _TYPES:
            raise UnsupportedSchemaError(
                f"{at} declares schema {schema!r}, whose type {kind!r} is not one of {', '.join(_TYPES)}"
            )
    # ``type`` beside ``enum`` or ``const`` is the registry's to reconcile (it refuses one that excludes an
    # enumerated value); beside ``anyOf`` it is a second statement of the shape that nothing reads.
    beside = {"type"} if kind != "anyOf" else set()
    if unhandled := sorted(set(schema) - HONOURED_KEYWORDS[kind] - ANNOTATION_KEYWORDS - beside):
        raise UnsupportedSchemaError(
            f"{at} declares schema {schema!r}, whose {', '.join(repr(word) for word in unhandled)} the world-schema "
            f"contract does not honour when generating from {kind} or checking a value against it — ignoring it "
            "would admit a value the schema forbids, so extend the readers or drop the keyword"
        )
    return kind


def schema_violations(schema: Mapping[str, Any], value: Any, *, at: str) -> list[str]:
    """Every way ``value`` fails ``schema``, each naming the path it fails at.

    Args:
        schema: A dimension's schema, or one nested in it.
        value: The seeded value.
        at: The path ``value`` sits at, for the messages (``jobs.queue[2].duration_ms``).

    Returns:
        One sentence per violation; empty when the value conforms.

    Raises:
        UnsupportedSchemaError: The schema is outside the honoured subset (:func:`honoured_kind`).
    """
    kind = honoured_kind(schema, at=at)
    if kind == "anyOf":
        return _any_branch(schema["anyOf"], value, at=at)
    if kind in ("enum", "const"):
        allowed = schema["enum"] if kind == "enum" else [schema["const"]]
        return (
            []
            if any(json_equal(value, option) for option in allowed)
            else [f"{at} is {value!r}, not one of {allowed!r}"]
        )
    if not _is_type(value, kind):
        return [f"{at} is {type(value).__name__} {value!r}, not {kind}"]
    return _within(schema, kind, value, at=at)


def _any_branch(branches: list[Mapping[str, Any]], value: Any, *, at: str) -> list[str]:
    """Nothing when any branch accepts ``value``; otherwise one sentence naming what each branch refused.

    One sentence rather than every branch's violations flattened, because the reader needs to see them
    grouped: "missing 'title'" means something different under a branch that wanted every field than under
    one that wanted none. A branch is named by its ``title`` annotation when it has one, so the host's own word
    for the shape ("a stopped room") reaches the author.

    Args:
        branches: The ``anyOf`` list, already audited by :func:`honoured_kind`.
        value: The seeded value.
        at: The path ``value`` sits at.

    Returns:
        Empty, or one sentence.
    """
    refusals: list[str] = []
    for index, branch in enumerate(branches):
        found = schema_violations(branch, value, at=at)
        if not found:
            return []
        name = branch.get("title") or f"shape {index + 1} of {len(branches)}"
        refusals.append(f"as {name}, {'; '.join(found)}")
    return [f"{at} fits none of the {len(branches)} shapes its schema accepts — {' | '.join(refusals)}"]


def json_equal(left: Any, right: Any) -> bool:
    """JSON equality: ``==``, except that a boolean equals only a boolean, at any depth.

    Python's ``True == 1`` would let ``{"enum": [0, 1]}`` admit a seeded ``true``; JSON Schema — and the
    registry's own enum-versus-type check — treat boolean and integer as distinct. ``1 == 1.0`` stays equal,
    as it is in JSON.
    """
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right, strict=True))
    return bool(left == right)


def _is_type(value: Any, kind: str) -> bool:
    """Whether ``value`` is of JSON type ``kind`` — a key of :data:`HONOURED_KEYWORDS` naming a type."""
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if kind == "string":
        return isinstance(value, str)
    if kind == "array":
        return isinstance(value, list | tuple)
    return isinstance(value, Mapping)


def _within(schema: Mapping[str, Any], kind: str, value: Any, *, at: str) -> list[str]:
    """The keyword constraints of ``kind`` applied to a value already of that type."""
    found: list[str] = []
    if kind in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            found.append(f"{at} is {value!r}, below the minimum {schema['minimum']!r}")
        if "maximum" in schema and value > schema["maximum"]:
            found.append(f"{at} is {value!r}, above the maximum {schema['maximum']!r}")
    elif kind == "string":
        if "minLength" in schema and len(value) < schema["minLength"]:
            found.append(f"{at} is shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            found.append(f"{at} is longer than {schema['maxLength']} characters")
    elif kind == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            found.append(f"{at} has {len(value)} items, fewer than {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            found.append(f"{at} has {len(value)} items, more than {schema['maxItems']}")
        if "items" in schema:
            for index, element in enumerate(value):
                found.extend(schema_violations(schema["items"], element, at=f"{at}[{index}]"))
    elif kind == "object":
        properties: Mapping[str, Any] = schema.get("properties") or {}
        found.extend(f"{at} is missing {key!r}" for key in schema.get("required") or () if key not in value)
        extra = schema.get("additionalProperties", True)
        for key, element in value.items():
            if key in properties:
                found.extend(schema_violations(properties[key], element, at=f"{at}.{key}"))
            elif extra is False:
                found.append(f"{at} carries {key!r}, which the schema does not declare")
            elif isinstance(extra, Mapping):
                found.extend(schema_violations(extra, element, at=f"{at}.{key}"))
    return found
