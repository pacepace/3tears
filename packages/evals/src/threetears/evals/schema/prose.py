"""Model prose as a TYPE — the marker that says a string holds text a model wrote.

Code checks the STRUCTURE of a model's output, never its prose:
whether a sentence is accurate, relevant or well-put is an eval of
the model that wrote it, and a deterministic check over it — a regex, a substring test, an
equality against a literal — is an eval dimension wearing a validator's clothes. To hold that
line mechanically the population has to be KNOWN, and a hand-kept list of "the prose fields"
fails by omission without ever going red. So prose is declared where the field is declared:

* a Pydantic field is annotated :data:`ModelProse` (or carries :data:`MODEL_PROSE_MARKER` in
  its ``Annotated`` metadata, directly or inside a container such as ``list[ModelProse]``);
* a JSON Schema — a world dimension's value schema — marks a string property with
  :data:`PROSE_SCHEMA_KEY`.

The marker is Python-only metadata. It adds nothing to a model's JSON Schema, so marking a field
changes no wire contract (``openapi.json`` is unaffected) and no strict output schema sent to a
provider.

Two consumers read it:

* ``tests/test_no_matching_over_model_prose.py`` derives the prose field names from
  the loaded models and refuses deterministic string matching over those attributes anywhere in
  ``threetears/evals/``;
* :func:`threetears.evals.kernel.dsl.world_prose_matches` tells a template-authoring gate which of a goal check's text
  predicates read prose, so ``contains()`` keeps its membership meaning over structured values
  while substring matching over model-written text is refused where the template is written.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin

from threetears.evals.schema.schema_nesting import nested_schemas

if TYPE_CHECKING:
    from pydantic.fields import FieldInfo


class ModelProseMarker:
    """``Annotated`` metadata declaring a string to be text a model wrote.

    A class rather than a bare sentinel so it reads in a ``repr`` of the annotation, and a single
    instance (:data:`MODEL_PROSE_MARKER`) so detection is an identity test nobody can spoof by
    constructing a look-alike.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Name the marker in an annotation's repr."""
        return "MODEL_PROSE"


#: The one marker instance; detection compares by identity.
MODEL_PROSE_MARKER = ModelProseMarker()

#: A string a model wrote. Use it as a field's type (``body: ModelProse``), inside a container
#: (``list[ModelProse]``), or nest it under further metadata
#: (``Annotated[ModelProse, BeforeValidator(...)]``) — every form is detected.
ModelProse = Annotated[str, MODEL_PROSE_MARKER]

#: The JSON Schema keyword marking a string property as model prose, for vocabularies declared as
#: schema rather than as Pydantic models (a world dimension's value schema). A vendor extension
#: keyword, so a validator ignores it.
PROSE_SCHEMA_KEY = "x-model-prose"


def annotation_carries_prose(annotation: Any) -> bool:
    """Whether ``annotation`` is, or contains, :data:`ModelProse`.

    Walks ``Annotated`` metadata and every type argument, so ``ModelProse | None``,
    ``list[ModelProse]`` and ``Annotated[ModelProse, BeforeValidator(...)]`` all count.

    Args:
        annotation: A type annotation, as a field declares it.

    Returns:
        True when the marker appears anywhere in it.
    """
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        return any(item is MODEL_PROSE_MARKER for item in metadata) or annotation_carries_prose(base)
    return any(annotation_carries_prose(argument) for argument in get_args(annotation))


def field_is_prose(field: FieldInfo) -> bool:
    """Whether a Pydantic field is declared model prose.

    Pydantic lifts a field's top-level ``Annotated`` metadata into ``FieldInfo.metadata`` and
    strips it from ``FieldInfo.annotation``, so both are read.

    Args:
        field: One entry of a model's ``model_fields``.

    Returns:
        True when the field, or a type inside it, carries the marker.
    """
    return any(item is MODEL_PROSE_MARKER for item in field.metadata) or annotation_carries_prose(field.annotation)


def schema_is_prose(schema: Mapping[str, Any] | None) -> bool:
    """Whether a JSON Schema node is marked model prose, directly or as an array's items.

    An array of prose strings counts: matching a literal against its members is equality over
    model-written text, which is the same check as a substring test in a different shape.

    The node and its items are read through :func:`schema_nodes_at`, so every shape either can take
    counts — an array whose items are an ``anyOf`` with one prose branch is prose, as the path addressing
    finds a prose field under a branch.

    Args:
        schema: A schema node, or None for a position no schema describes.

    Returns:
        True when the node, or the node's ``items``, carries :data:`PROSE_SCHEMA_KEY` in any of its shapes.
    """
    if not isinstance(schema, Mapping):
        return False
    if schema.get(PROSE_SCHEMA_KEY) is True:
        return True
    shapes = (*schema_nodes_at(schema, ()), *schema_nodes_at(schema, ("[]",)))
    return any(shape.get(PROSE_SCHEMA_KEY) is True for shape in shapes)


def schema_nodes_at(schema: Mapping[str, Any], segments: tuple[str, ...]) -> tuple[Mapping[str, Any], ...]:
    """Every schema node a sequence of DSL segments can address inside ``schema``.

    ``[]`` steps into an array's ``items``; a name steps into the property of that name, or into the
    ``additionalProperties`` schema where the object describes no such property and holds one — a map's values
    are addressed by key exactly as declared properties are. A value under an ``anyOf`` takes one of its
    shapes, and which one is a property of the value rather than the schema, so a position is described by
    every branch that describes it.

    The positions are :func:`~threetears.evals.schema.schema_nesting.nested_schemas`'s, the ones every schema
    walker in the engine reads, so this cannot stop short of a position the honoured-subset audit or the
    registry's self-contradiction check descends into.

    Args:
        schema: The root schema — a world dimension's value schema, or an action's parameter schema.
        segments: What the expression addresses below the root, as
            :attr:`threetears.evals.schema.goal_grammar.TextMatch.operand` spells it.

    Returns:
        The addressed nodes, each a schema the position can take; empty where no shape describes the position
        (a ``.length`` pseudo-attribute, an undeclared property). Each caller decides what an undescribed
        position means: the world prose gate refuses nothing on one, the call-parameter gate refuses it.
    """
    nodes: list[Any] = [schema]
    for segment in segments:
        stepped: list[Any] = []
        for node in _shapes(nodes):
            stepped.extend(_step(node, segment))
        nodes = stepped
    return tuple(_shapes(nodes))


def _step(node: Mapping[str, Any], segment: str) -> list[Any]:
    """What one DSL segment addresses directly inside one schema node (which is not an ``anyOf``)."""
    nested = list(nested_schemas(node))
    if segment == "[]":
        return [position.schema for position in nested if position.keyword == "items"]
    named = [position.schema for position in nested if position.keyword == "properties" and position.key == segment]
    return named or [position.schema for position in nested if position.keyword == "additionalProperties"]


def _shapes(nodes: list[Any]) -> list[Mapping[str, Any]]:
    """The schema nodes in ``nodes``, each ``anyOf`` replaced by its branches, recursively; non-schemas dropped."""
    found: list[Mapping[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if "anyOf" in node:
            found.extend(_shapes([position.schema for position in nested_schemas(node) if position.keyword == "anyOf"]))
        else:
            found.append(node)
    return found


__all__ = [
    "MODEL_PROSE_MARKER",
    "PROSE_SCHEMA_KEY",
    "ModelProse",
    "ModelProseMarker",
    "annotation_carries_prose",
    "field_is_prose",
    "schema_is_prose",
    "schema_nodes_at",
]
