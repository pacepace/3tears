"""The strict structured-output contract every package call whose reply is read into a typed model sends, and the
shape a retired memo's recommendation had.

**The strict projection.** :func:`strict_schema` projects a pydantic JSON schema onto the portable strict subset
(:data:`PORTABLE_STRICT_KEYWORDS`) both major providers' strict structured-output modes accept, and
:func:`strict_response_format` wraps it as the ``json_schema`` directive the judge, the simulator, the proposer and the
out-of-run calls send.

**The retired memo's recommendation.** The memo writer is retired, and its memo is stored as JSON
(``EvalAnalysis.document``), never read. :class:`Recommendation` and the models it nests (:class:`FloorOverride`,
:class:`BehindOn`, :class:`BarMissed`) stay for this minor because ``CampaignReport.recommendation`` still types its
field with them; no page fills it, and the next minor removes the field and them. :data:`Confidence` is the four
confidence words the report writer's document and the stored analyses share.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.schema.prose import ModelProse

#: How sure the author is of a finding or a decision.
Confidence = Literal["very_high", "high", "medium", "low"]
#: What a piece of evidence reads: a ``measure``, or a ``judged`` dimension.
Reading = Literal["measure", "judged"]


class _Authored(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


class FloorOverride(_Authored):
    """A broken floor the recommended arm is recommended over, named by its key, and why breaking it is acceptable."""

    floor: str = Field(description="The broken floor's `floor` key, spelled as the bundle spells it.")
    why: ModelProse = Field(description="Why recommending the arm over this broken floor is acceptable here.")


class BehindOn(_Authored):
    """An aspect on which code showed another arm ahead of the one a choice recommends, and why it is picked anyway."""

    aspect: str = Field(description="The aspect's label, spelled as the bundle spells it.")
    ahead_arm: str = Field(description="The arm code showed ahead there, by its label.")
    why: ModelProse = Field(description="Why the recommended arm is picked over it all the same.")


class BarMissed(_Authored):
    """An absolute bar the recommended arm missed on one aspect, and why it is recommended all the same."""

    aspect: str = Field(description="The aspect's label, spelled as the bundle spells it.")
    measure: str = Field(description="The bar's measure, spelled as the bundle spells it.")
    why: ModelProse = Field(description="Why recommending the arm below this bar is acceptable here.")


class Recommendation(_Authored):
    """The one call a retired memo recommended for a campaign: which arm, why, how sure, and the findings it rests on.

    Kept as the type of ``CampaignReport.recommendation``, which no page fills now that the memo writer is retired;
    the next minor removes the field and this model. ``floors_overridden``, ``behind_on`` and ``bars_missed`` were the
    memo's answers to what code warned of on the recommended arm; ``rests_on`` named the findings it rested on.
    """

    arm: str = Field(
        description=(
            "The arm recommended, by its label as the campaign bundle names it; the control's label to keep it. Code "
            "stores the arm's variant key beside the memo (``EvalAnalysis.recommended_variant_key``)."
        )
    )
    reasons: list[ModelProse] = Field(description="Why, one reason each, strongest first.")
    confidence: Confidence
    rests_on: list[int] = Field(
        description="Positions in `findings` this rests on; at least one, except when keeping the control."
    )
    floors_overridden: list[FloorOverride] = Field(
        description="Every floor broken on the recommended arm, each once, with why; empty when it broke none."
    )
    behind_on: list[BehindOn] = Field(
        description="Every aspect a choice's pair readings show another arm ahead of this one, with why; else empty."
    )
    bars_missed: list[BarMissed] = Field(
        description="Every absolute bar the recommended arm missed, each once, with why; empty when it missed none."
    )


class MapFieldInSchema(TypeError):
    """An object admits arbitrary keys, which no strict structured-output mode accepts."""


class UnportableSchema(TypeError):
    """A schema keyword no strict structured-output mode is known to accept, and none that a reply's validation
    stands in for: a structural keyword (``allOf``, ``not``, ``if``), which dropping would change the shape sent; or an
    ``enum`` or ``const`` holding a value that is not a string, which not every strict mode enforces."""


#: The JSON Schema keywords a strict ``json_schema`` sent to a provider carries: the portable strict subset, which
#: both major providers' strict structured-output modes accept.
#:
#: Read from the providers' own SDKs in this workspace. Anthropic's (``anthropic`` 0.116, ``lib/_parse/_transform.py``,
#: ``transform_schema``) keeps ``type``, ``properties``, ``required``, ``additionalProperties`` (false), ``items``,
#: ``enum``, ``anyOf`` (``oneOf`` read as it), ``allOf``, ``$ref``/``$defs``, ``description``, ``title``, a string
#: ``format`` from a short list and ``minItems`` of 0 or 1, and moves every other keyword — ``minLength``,
#: ``maxLength``, ``minimum``, ``maximum``, ``pattern``, ``const`` — into the description as a hint the model may
#: ignore. OpenAI's (``openai`` 2.45, ``lib/_pydantic.py``, ``_ensure_strict_json_schema``) closes objects, requires
#: every property and inlines a ``$ref`` with siblings, and encodes no keyword list; its documented strict mode adds
#: number bounds, ``pattern``, ``format`` and item counts to the structural set, and not string lengths or ``allOf``.
#: The intersection is the structural set below. Everything else a model's schema states is enforced where it always
#: was: the reply is validated against the model on return.
#:
#: **An ``enum`` enumerates strings only.** Gemini's structured output supports string enums alone, and an integer
#: enum sent to ``google/gemini-3.5-flash-lite`` through OpenRouter, ``require_parameters`` on, was accepted and not
#: enforced: the reply left the field out. So a value that is not a string is refused in an ``enum`` or a ``const``
#: (:func:`strict_schema`): a reply the engine reads asks for strings and reads them back
#: (:data:`threetears.evals.run.judge.ORDINAL_ANSWERS`), and a host's schema says so in its own types.
PORTABLE_STRICT_KEYWORDS: frozenset[str] = frozenset(
    {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf", "$ref", "$defs", "description"}
)

#: Keywords a strict schema leaves out because the reply's validation on return enforces them (a bound, a length, a
#: pattern, a format, an item count) or because they say nothing about the shape (a title, a default, an example).
_VALIDATED_ON_RETURN: frozenset[str] = frozenset(
    {
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "default",
        "title",
        "examples",
        "discriminator",
        "readOnly",
        "writeOnly",
        "deprecated",
    }
)


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Project a pydantic JSON schema onto the portable strict subset (:data:`PORTABLE_STRICT_KEYWORDS`).

    Every object closed and fully required; ``oneOf`` as ``anyOf``; a ``const`` as the one-value ``enum`` it is; a
    ``$ref`` with nothing beside it (OpenAI refuses a sibling keyword); and every keyword a reply's validation on
    return enforces (a bound, a length, a pattern) or that says nothing about the shape (a title, a default) left
    out — sent, a provider's strict mode refuses the schema or quietly demotes the keyword to a hint. A keyword that
    is neither is refused rather than dropped: dropping a structural one would send another shape.

    **An ``enum`` or ``const`` holding a value that is not a string is refused, not rewritten.** A string enum is the
    one enum form every provider's strict mode enforces; an integer one was sent and not enforced
    (:data:`PORTABLE_STRICT_KEYWORDS`). Projecting ``[1, 2, 3]`` to ``["1", "2", "3"]`` here would change the type of
    the reply its reader parses, and this projection does not know the reader: the engine's own judge asks for
    strings and reads them back (:func:`threetears.evals.run.judge.judge_reply_format`), and a host projecting its own
    model declares the values as strings in it (a ``Literal["1", "2"]``, a ``str`` enum) and converts on read.

    :param schema: A pydantic-generated JSON schema.
    :ptype schema: dict[str, Any]
    :return: The projected schema.
    :rtype: dict[str, Any]
    :raises MapFieldInSchema: An object carries an open ``additionalProperties``.
    :raises UnportableSchema: A node carries a keyword outside the portable subset that validation does not stand in
        for, or an ``enum`` or ``const`` holding a value that is not a string.
    """

    def string_values(key: str, values: list[Any], path: str) -> list[str]:
        """An ``enum``'s or ``const``'s values, refused by name when one is not a string."""
        others = [value for value in values if not isinstance(value, str)]
        if others:
            raise UnportableSchema(
                f"{path or '/'} carries {key!r} with {others!r}, values that are not strings: a string enum is the one "
                "enum form every provider's strict structured-output mode enforces (Gemini's takes no other), so a "
                "strict schema enumerates strings only; declare the values as strings (a Literal of strings, a str "
                "enum) and read them back to the type your code wants"
            )
        return list(values)

    def walk(node: Any, path: str) -> Any:
        if isinstance(node, list):
            return [walk(item, f"{path}[{index}]") for index, item in enumerate(node)]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return {"$ref": node["$ref"]}
        extra = node.get("additionalProperties")
        if isinstance(extra, dict) or extra is True:
            raise MapFieldInSchema(f"{path or '/'} admits arbitrary keys")
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _VALIDATED_ON_RETURN:
                continue
            if key == "const":
                out["enum"] = string_values(key, [value], f"{path}/const")
                continue
            if key == "enum":
                out["enum"] = string_values(key, list(value), f"{path}/enum")
                continue
            target = "anyOf" if key == "oneOf" else key
            if target not in PORTABLE_STRICT_KEYWORDS:
                raise UnportableSchema(
                    f"{path or '/'} carries {key!r}, which no strict structured-output mode is known to accept "
                    "and which no validation on return stands in for; state the shape with the portable keywords"
                )
            if key in ("properties", "$defs"):
                out[target] = {name: walk(child, f"{path}/{key}/{name}") for name, child in value.items()}
            else:
                out[target] = walk(value, f"{path}/{key}")
        if node.get("type") == "object" or "properties" in node:
            out["additionalProperties"] = False
            out["required"] = list(node.get("properties", {}))
        return out

    projected: dict[str, Any] = walk(schema, "")
    return projected


def strict_response_format(name: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """The strict ``json_schema`` directive that asks a provider for exactly ``schema``'s shape.

    Every package call whose reply is read into a typed model sends this, derived from that model, so a provider
    that enforces it cannot write a reply the model refuses — whatever the prompt, stored or not, says. Projected
    through :func:`strict_schema`; the schema's own top-level description is dropped, because it is written for
    the next developer and the prompt is what tells the model what the fields mean. A provider that cannot
    enforce a strict schema is the host client's to handle — it sends the nearest mode its provider has (JSON
    mode, a tool's input schema) — and the reply is validated against the model on return either way.

    :param name: The schema's name, as the directive carries it.
    :ptype name: str
    :param schema: A pydantic-generated JSON schema (``Model.model_json_schema()``), or one shaped like it.
    :ptype schema: Mapping[str, Any]
    :return: The OpenAI-shaped directive.
    :rtype: dict[str, Any]
    :raises MapFieldInSchema: An object in it admits arbitrary keys, which no strict mode accepts.
    :raises UnportableSchema: A node in it carries a keyword outside the portable subset that validation does not stand
        in for, or an ``enum`` or ``const`` holding a value that is not a string (:func:`strict_schema`).
    """
    projected = {key: value for key, value in strict_schema(dict(schema)).items() if key != "description"}
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": projected}}


__all__ = [
    "BarMissed",
    "BehindOn",
    "Confidence",
    "FloorOverride",
    "MapFieldInSchema",
    "PORTABLE_STRICT_KEYWORDS",
    "Reading",
    "Recommendation",
    "UnportableSchema",
    "strict_response_format",
    "strict_schema",
]
