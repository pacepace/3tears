"""a tool's JSON Schema made self-contained, and the one type a property declares.

A tool advertises its arguments as a JSON Schema -- pydantic's ``model_json_schema()``, an MCP
server's ``inputSchema``, a LangChain tool's ``args_schema`` -- and pydantic writes every nested
model as a ``$ref`` into the schema's ``$defs`` and every optional field as
``anyOf: [X, {"type": "null"}]``. A consumer that reads only a property's own ``type`` then sees a
nested model as nothing and an optional list as nothing: a model is shown a string field where the
tool wants a list of objects, and an input normaliser leaves a JSON-encoded list as a string.

:func:`self_contained_input_schema` resolves all of that into one schema with no references, for a
model or validator that is handed a single schema per tool. :func:`declared_type` answers the
narrower question a normaliser asks of one property: which single JSON type does it declare.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

__all__ = [
    "declared_type",
    "self_contained_input_schema",
]

#: JSON Schema keywords whose value is one subschema.
_SUBSCHEMA_KEYWORDS = frozenset(
    {"items", "additionalProperties", "not", "contains", "if", "then", "else", "propertyNames", "unevaluatedItems"}
)

#: JSON Schema keywords whose value is a list of subschemas.
_SUBSCHEMA_LIST_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})

#: JSON Schema keywords whose value maps names to subschemas. The names are data, not keywords.
_SUBSCHEMA_MAP_KEYWORDS = frozenset({"properties", "patternProperties", "dependentSchemas", "$defs", "definitions"})

#: Where a local definition lives, for each spelling of it: ``$defs`` (JSON Schema 2019-09 on,
#: pydantic 2) and ``definitions`` (draft 7, pydantic 1).
_DEFINITION_PREFIXES = ("#/$defs/", "#/definitions/")


def _definitions(schema: Mapping[str, Any]) -> dict[str, Any]:
    """the schema's local definitions, both spellings merged.

    :param schema: the root schema
    :ptype schema: Mapping[str, Any]
    :return: definition name to definition
    :rtype: dict[str, Any]
    """
    return {**(schema.get("definitions") or {}), **(schema.get("$defs") or {})}


def _local_definition_name(ref: str) -> str | None:
    """the name of the local definition ``ref`` points at, or ``None`` when it points elsewhere.

    :param ref: a ``$ref`` value
    :ptype ref: str
    :return: the definition's name, JSON-pointer escapes decoded, or ``None``
    :rtype: str | None
    """
    prefix = next((p for p in _DEFINITION_PREFIXES if ref.startswith(p)), None)
    name: str | None = None
    if prefix is not None and "/" not in ref[len(prefix) :]:
        name = ref[len(prefix) :].replace("~1", "/").replace("~0", "~")
    return name


def _walk_subschemas(node: Mapping[str, Any], transform: Callable[[Any], Any]) -> dict[str, Any]:
    """``node`` with ``transform`` applied to every subschema it holds, and everything else copied.

    Only schema-bearing keywords are walked: a ``default``, ``enum``, ``const`` or ``examples`` value
    is data, and a dict inside one must not be read as a schema.

    :param node: one schema object
    :ptype node: Mapping[str, Any]
    :param transform: applied to each immediate subschema
    :ptype transform: Callable[[Any], Any]
    :return: a new schema object; ``node`` is not mutated
    :rtype: dict[str, Any]
    """
    walked: dict[str, Any] = {}
    for key, value in node.items():
        if key in _SUBSCHEMA_KEYWORDS and isinstance(value, Mapping):
            walked[key] = transform(value)
        elif key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, list):
            walked[key] = [transform(item) for item in value]
        elif key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, Mapping):
            walked[key] = {name: transform(item) for name, item in value.items()}
        else:
            walked[key] = value
    return walked


def _collapse_optional(node: Any) -> Any:
    """``node`` with every ``X | None`` union collapsed to ``X``, at every depth.

    Pydantic renders an optional field as ``anyOf: [X, {"type": "null"}]`` with no top-level
    ``type``. A union with exactly one non-null member becomes that member, carrying the field's
    own keywords over it -- the field's description is the more specific one -- minus the
    ``default: null`` that would contradict the member's type. A union of two or more real members
    is left whole: choosing one would drop the others.

    :param node: a schema, or any value inside one
    :ptype node: Any
    :return: the schema with optional unions collapsed; ``node`` is not mutated
    :rtype: Any
    """
    if not isinstance(node, Mapping):
        return node
    result = _walk_subschemas(node, _collapse_optional)
    union_key = "anyOf" if "anyOf" in result else "oneOf" if "oneOf" in result else None
    if union_key is not None and "type" not in result:
        members = result[union_key]
        real = [m for m in members if not (isinstance(m, Mapping) and m.get("type") == "null")]
        if len(real) == 1 and len(real) < len(members) and isinstance(real[0], Mapping):
            field = {k: v for k, v in result.items() if k != union_key and not (k == "default" and v is None)}
            result = {**real[0], **field}
    return result


def _inline_refs(
    node: Any,
    definitions: dict[str, Any],
    tool_name: str,
    expanding: tuple[str, ...] = (),
) -> Any:
    """``node`` with every ``$ref`` replaced by the definition it names, and titles removed.

    A reference's sibling keywords override the definition's -- pydantic writes a field's own
    description beside the ``$ref``, and it is the more specific one. Titles go because they are
    noise to a model, and LangChain's own conversion for the provider APIs removes them too.

    A recursive model cannot be inlined completely: its schema is infinite. It is expanded until a
    definition recurs inside its own expansion, and at that point the schema says in words what
    the value is -- the same shape as the enclosing one -- keeping the definition's ``type``. A
    bounded expansion rather than a refusal, so a tool with a tree-shaped argument stays usable;
    described rather than cut to ``{}``, so a model is not shown "anything" where the tool requires
    a particular shape.

    :param node: a schema, or any value inside one
    :ptype node: Any
    :param definitions: the root schema's ``$defs`` and ``definitions``, merged
    :ptype definitions: dict[str, Any]
    :param tool_name: the tool whose schema this is, for errors
    :ptype tool_name: str
    :param expanding: the definitions being expanded on the path to ``node``, outermost first
    :ptype expanding: tuple[str, ...]
    :return: the inlined schema; ``node`` is not mutated
    :rtype: Any
    :raises ValueError: when a ``$ref`` names no local definition
    """
    if not isinstance(node, Mapping):
        return node
    siblings = _walk_subschemas(
        {k: v for k, v in node.items() if k not in ("$ref", "title", "$defs", "definitions")},
        lambda child: _inline_refs(child, definitions, tool_name, expanding),
    )
    ref = node.get("$ref")
    result: Any = siblings
    if isinstance(ref, str):
        name = _local_definition_name(ref)
        if name is None:
            raise ValueError(
                f"tool {tool_name!r}: cannot make its input schema self-contained: $ref {ref!r} does "
                "not name a definition in the schema's own $defs, so it cannot be inlined"
            )
        if name not in definitions:
            raise ValueError(
                f"tool {tool_name!r}: cannot make its input schema self-contained: $ref {ref!r} names "
                "a definition the schema does not carry"
            )
        definition = definitions[name]
        if name in expanding:
            note = f"A {name}: the same shape as the {name} that contains it."
            field_description = siblings.pop("description", None)
            recursion: dict[str, Any] = {"description": f"{field_description} {note}" if field_description else note}
            if isinstance(definition, Mapping) and "type" in definition:
                recursion = {"type": definition["type"], **recursion}
            result = {**recursion, **siblings}
        else:
            expanded = _inline_refs(definition, definitions, tool_name, (*expanding, name))
            result = {**expanded, **siblings} if isinstance(expanded, Mapping) else expanded
    return result


def self_contained_input_schema(schema: Mapping[str, Any], *, tool_name: str) -> dict[str, Any]:
    """a tool's input schema with every reference inlined, as one ``type: object`` schema.

    For a consumer handed one schema per tool and no shared definitions -- a model's tool listing,
    an MCP client, a validator given the properties alone. Every ``$ref`` into the schema's own
    ``$defs`` / ``definitions`` is inlined, through ``items``, unions and nested properties, with
    each level's ``required`` list and descriptions kept; a field's description wins over its
    model's. Every optional union (``anyOf: [X, null]``) collapses to ``X`` at every depth; a union
    of two or more real members is kept whole; an untyped field stays untyped. A recursive model is
    expanded until it recurs, and the point of recursion reads
    ``"A Node: the same shape as the Node that contains it."`` with the definition's type. Titles
    and the root description are dropped -- the tool's own description travels beside its schema.

    The result always has ``type: "object"``, ``properties`` and ``required``, so a builder that
    would otherwise mark every property required uses it as it stands.

    :param schema: the tool's input JSON Schema -- pydantic's ``model_json_schema()``, an MCP
        ``inputSchema``, a LangChain ``args_schema`` or ``tool_call_schema``; not mutated
    :ptype schema: Mapping[str, Any]
    :param tool_name: the tool's name, for errors
    :ptype tool_name: str
    :return: the self-contained schema
    :rtype: dict[str, Any]
    :raises ValueError: when a ``$ref`` points anywhere but the schema's own definitions, or names
        one it does not carry -- a reference left in place would point at nothing the reader has
    """
    collapsed = _collapse_optional(schema)
    root = _inline_refs(collapsed, _definitions(collapsed), tool_name)
    root.pop("description", None)
    return {
        **root,
        "type": "object",
        "properties": root.get("properties") or {},
        "required": list(root.get("required") or []),
    }


def declared_type(prop: Any, schema: Mapping[str, Any]) -> str | None:
    """the one JSON Schema type ``prop`` declares, read through the shapes pydantic writes.

    A property's own ``type`` answers when it is a string, or a list naming one type besides
    ``null``. An optional field has no ``type`` -- pydantic writes ``anyOf: [X, {"type": "null"}]``
    -- and answers with ``X``'s. A nested model is a ``$ref`` into the schema's own definitions and
    answers with the definition's. A union of two or more real types answers ``None``: the value
    may be any of them, and treating it as one would choose for the caller. So does a reference
    that points elsewhere or names nothing, and one already being followed.

    Never raises: it is for code that must not fail on a schema it half understands, such as a
    normaliser that coerces a JSON-encoded string into the list a property declares.

    :param prop: one property's schema
    :ptype prop: Any
    :param schema: the whole schema, whose definitions a reference names
    :ptype schema: Mapping[str, Any]
    :return: the declared type, or ``None`` when there is no single one
    :rtype: str | None
    """
    return _declared_type(prop, schema, frozenset())


def _declared_type(prop: Any, schema: Mapping[str, Any], following: frozenset[str]) -> str | None:
    """:func:`declared_type`, carrying the references already followed so a cycle ends.

    :param prop: one property's schema
    :ptype prop: Any
    :param schema: the whole schema
    :ptype schema: Mapping[str, Any]
    :param following: references already followed on this path
    :ptype following: frozenset[str]
    :return: the declared type, or ``None``
    :rtype: str | None
    """
    result: str | None = None
    if not isinstance(prop, Mapping):
        return result
    declared = prop.get("type")
    ref = prop.get("$ref")
    members = prop.get("anyOf") or prop.get("oneOf")
    if isinstance(declared, str):
        result = declared
    elif isinstance(declared, list):
        real_types = [t for t in declared if t != "null"]
        result = real_types[0] if len(real_types) == 1 and isinstance(real_types[0], str) else None
    elif isinstance(ref, str) and ref not in following:
        name = _local_definition_name(ref)
        if name is not None:
            result = _declared_type(_definitions(schema).get(name), schema, following | {ref})
    elif isinstance(members, list):
        real = [m for m in members if not (isinstance(m, Mapping) and m.get("type") == "null")]
        result = _declared_type(real[0], schema, following) if len(real) == 1 else None
    return result
