"""input-value coercion helpers for TearsTool subclasses.

LLMs routinely send loose values for object/array tool parameters:
empty strings where a dict is expected, JSON-encoded strings where
a dict/list is expected, arguments omitted entirely. this module
provides pure functions that rewrite those loose values toward the
declared JSON schema type before dispatch, so subclasses receive
the native Python container they expect.

the coercion engages only for fields whose declared type is
``object`` or ``array``. every other field passes through
untouched. explicit ``None`` is preserved as a "not provided"
sentinel so required-field checks in subclasses still fire.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "coerce_value",
    "normalize_kwargs",
]


def coerce_value(value: Any, declared_type: str | None) -> Any:
    """coerce single MCP input value toward its declared JSON schema type.

    handles two LLM mistakes: empty string supplied for complex type
    (becomes empty container), and JSON-encoded string supplied where
    native dict/list is expected (gets decoded). values already of
    correct type, or values that cannot be safely coerced, pass
    through unchanged.

    :param value: raw value supplied by caller
    :ptype value: Any
    :param declared_type: JSON schema ``type`` for this parameter or None
    :ptype declared_type: str | None
    :return: value coerced to declared type when possible, else original
    :rtype: Any
    """
    result = value
    # only empty *strings* are treated as "give me the empty container" --
    # literal 0, False, or empty tuple must pass through so type-strict
    # subclasses still see the original value (and can reject it).
    if declared_type == "object" and not isinstance(value, dict):
        if value == "":
            result = {}
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except TypeError, ValueError:
                parsed = None
            if isinstance(parsed, dict):
                result = parsed
    elif declared_type == "array" and not isinstance(value, list):
        if value == "":
            result = []
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except TypeError, ValueError:
                parsed = None
            if isinstance(parsed, list):
                result = parsed
    return result


#: where a local definition lives, for each spelling of it.
_DEFINITION_PREFIXES = ("#/$defs/", "#/definitions/")


def _declared_type(prop: Any, input_schema: dict[str, Any], seen: frozenset[str] = frozenset()) -> str | None:
    """the one JSON schema type ``prop`` declares, read through the shapes pydantic writes.

    a property's own ``type`` answers when it is a string, or a list
    naming one type besides ``null``. an optional field has no
    ``type`` -- pydantic writes ``anyOf: [X, {"type": "null"}]`` --
    and answers with ``X``'s. a nested model is a ``$ref`` into the
    schema's own ``$defs`` and answers with the definition's. a union
    of two or more real types answers ``None``: the value may be any
    of them, and decoding a string would choose for the caller. so
    does a reference that names nothing, or one already being followed.

    :param prop: one property's schema
    :ptype prop: Any
    :param input_schema: the whole input schema, whose ``$defs`` a reference names
    :ptype input_schema: dict[str, Any]
    :param seen: references already followed on this path
    :ptype seen: frozenset[str]
    :return: the declared type, or ``None`` when there is no single one
    :rtype: str | None
    """
    result: str | None = None
    if not isinstance(prop, dict):
        return result
    declared = prop.get("type")
    ref = prop.get("$ref")
    members = prop.get("anyOf") or prop.get("oneOf")
    if isinstance(declared, str):
        result = declared
    elif isinstance(declared, list):
        real_types = [t for t in declared if t != "null"]
        result = real_types[0] if len(real_types) == 1 and isinstance(real_types[0], str) else None
    elif isinstance(ref, str) and ref not in seen:
        prefix = next((p for p in _DEFINITION_PREFIXES if ref.startswith(p)), None)
        definitions = {**(input_schema.get("definitions") or {}), **(input_schema.get("$defs") or {})}
        if prefix is not None:
            result = _declared_type(definitions.get(ref[len(prefix) :]), input_schema, seen | {ref})
    elif isinstance(members, list):
        real = [m for m in members if not (isinstance(m, dict) and m.get("type") == "null")]
        result = _declared_type(real[0], input_schema, seen) if len(real) == 1 else None
    return result


def normalize_kwargs(
    kwargs: dict[str, Any],
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """coerce loose LLM-supplied kwargs to match declared schema types.

    inspects ``input_schema['properties']`` and rewrites entries in
    kwargs whose declared type is ``object`` or ``array`` when the
    supplied value is a wrong-shape loose container (empty string
    or JSON-encoded string). the declared type is read through an
    optional union and a ``$ref`` into the schema's ``$defs`` (see
    :func:`_declared_type`). values matching their declared type
    are passed through untouched. keys not present in the schema
    are passed through untouched. explicit ``None`` is preserved
    so per-action required-field checks still function.

    :param kwargs: raw tool input parameters as supplied by caller
    :ptype kwargs: dict[str, Any]
    :param input_schema: tool's JSON schema for input parameters
    :ptype input_schema: dict[str, Any]
    :return: parameters with wrong-type values coerced where possible
    :rtype: dict[str, Any]
    """
    properties = input_schema.get("properties", {})
    if not properties:
        return dict(kwargs)
    normalized: dict[str, Any] = {}
    for key, value in kwargs.items():
        prop = properties.get(key)
        if prop is None or value is None:
            normalized[key] = value
            continue
        normalized[key] = coerce_value(value, _declared_type(prop, input_schema))
    return normalized
