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
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                result = parsed
    elif declared_type == "array" and not isinstance(value, list):
        if value == "":
            result = []
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, list):
                result = parsed
    return result


def normalize_kwargs(
    kwargs: dict[str, Any],
    input_schema: dict[str, Any],
) -> dict[str, Any]:
    """coerce loose LLM-supplied kwargs to match declared schema types.

    inspects ``input_schema['properties']`` and rewrites entries in
    kwargs whose declared type is ``object`` or ``array`` when the
    supplied value is a wrong-shape loose container (empty string
    or JSON-encoded string). values matching their declared type
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
        declared_type = prop.get("type")
        normalized[key] = coerce_value(value, declared_type)
    return normalized
