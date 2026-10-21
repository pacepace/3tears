"""What the analysis generator authors: a small document shape carrying freeform analysis.

**The boundary.** This contract types what the document is MADE OF and the
hooks code acts on, never what it says or how much it says:

- **Typed here:** the unit kinds (the summary, finding, the one recommendation, an aspect's summary, next step); the
  order of every list, which is the reading order; links between units by POSITION, so a recommendation that rests on
  findings ``[0, 2]`` cites the first and third; the readings code turns into numbers (a cell and a
  measure or judged dimension); and a closed vocabulary code branches or renders on.
- **Not the model's to say:** what a verdict stands on. The evidence tier is read off the readings a
  finding names (:func:`threetears.evals.kernel.campaign.evidence_tier_of`), so this contract offers no
  tier field, and a payload that supplies one is refused as an unknown key.
- **The model's judgment:** what a finding is about and how many there are; every title, body,
  proposal, answer and rationale, in prose with markdown allowed.
- **Not the model's at all:** the headline, a chart, an arm's standing, and any count, rate, cost or n.
  The report's warnings, readings and scoreboard are written by code from the evidence bundle, so the writer authors
  one short summary across every aspect, findings, one recommendation, a short summary per aspect and next steps
  only, the recommendation pinned to the findings it rests on and each finding to the trials it is drawn from
  (``trials``, by the campaign bundle's trial ids). Code reports, the writer recommends, a person decides.

**Sized to what every writer accepts.** A strict schema derived from the stored analysis models was
refused by every Anthropic writer as a grammar too large, and the limit counts free-text and list
slots across the whole schema.
So nothing here is nullable and nothing is a union: absence is an empty string or an empty list, which
cannot be read as zero because this side of the contract carries no numbers. :data:`SLOT_BUDGET` is
checked by a test, so growth fails CI rather than a paid call.

**One source.** The models below are what is sent (:func:`response_format`, their strict projection)
and what is checked on return (:func:`validate_authored`), so the two cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from threetears.evals.schema.prose import ModelProse

#: The name the schema is sent under.
SCHEMA_NAME = "analysis"

#: The most free-text and list slots the sent schema may hold (:func:`count_slots`). Measured
#: against one strict-schema provider by padding this contract with string fields until the
#: provider refused it, and the budget sits just under the largest size measured to pass. Moving it
#: up is a decision to re-probe every eligible writer first, not an edit.
SLOT_BUDGET = 40

#: How sure the author is of a finding or a decision.
Confidence = Literal["very_high", "high", "medium", "low"]
#: What a piece of evidence reads: a ``measure``, or a ``judged`` dimension.
Reading = Literal["measure", "judged"]


class _Authored(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


class EvidenceRef(_Authored):
    """One reading at one cell. Code fills the number, its sample size and its spread."""

    cell: str = Field(description="The cell this reading is taken at.")
    measure_id: str = Field(description="The measure or judged dimension, spelled as the bundle spells it.")
    reading: Reading = Field(description="`measure` for a measure, `judged` for a judged dimension.")


class Caveat(_Authored):
    """What qualifies a finding, and which class of qualification it is."""

    kind: str
    text: ModelProse


class Finding(_Authored):
    """One claim the evidence supports, in the author's words."""

    title: ModelProse = Field(description="The finding's conclusion, in plain words a non-specialist understands.")
    body: ModelProse = Field(
        description="Conclusion first, then the evidence; paragraphs of at most three short sentences, lists for lists."
    )
    confidence: Confidence
    axes: list[str] = Field(description="Levers it concerns; empty when it is about the subject or the rig.")
    evidence: list[EvidenceRef] = Field(description="The readings it rests on.")
    trials: list[str] = Field(
        description="The ids of the trials (`trials[].trial_id`) it is drawn from — the failure mode it names; empty when none."
    )
    caveats: list[Caveat]
    invalidates: list[int] = Field(description="Positions of findings this one invalidates.")
    durable: ModelProse = Field(description="A durable claim about the subject for later analyses; empty when none.")


class NextStep(_Authored):
    """An experiment worth running next."""

    title: ModelProse = Field(description="The experiment, stated as an action.")
    why: ModelProse = Field(description="What it would change and how to run it.")
    leverage: Literal["high", "medium", "low"]
    lever: str = Field(description="The lever it would measure; empty when none.")


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
    """The one call the writer recommends for a campaign: which arm, why, how sure, and the findings it rests on.

    One per campaign, over every aspect: a person decides, and this is what they decide over. Which kind of campaign
    it is decides what it may say, and is derived from its arms: a **measurement** (one arm) recommends nothing; a
    **choice** (several arms, no control) picks one, or recommends nothing when no arm stood out (no clear winner);
    a **change** (a control and its candidates) recommends a candidate or keeps the control, named by the control's
    label. What code warned of on the recommended arm is never blocked, and the writer must address each:
    ``floors_overridden`` every floor it broke, ``behind_on`` every aspect another arm was shown ahead of it on (a
    choice), and ``bars_missed`` every absolute bar it missed. The generator refuses a recommendation that leaves one
    out. ``rests_on`` names the findings it rests on; keeping a change's control may rest on none, since keeping what is
    there needs no evidence of its own.
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


class AspectSummary(_Authored):
    """One aspect, as the writer reads it: what changed there, how sure the evidence is, and what it means here."""

    aspect: str = Field(description="The aspect's label, spelled as the bundle spells it; one entry per aspect.")
    what_changed: ModelProse = Field(description="What the candidate changed on this aspect, in one plain sentence.")
    how_sure: ModelProse = Field(
        description="How sure the evidence is, in plain words; the reading code gave, never a new number."
    )
    means: ModelProse = Field(description="What it means for this campaign's question: the writer's judgment.")


class AuthoredCampaignAnalysis(_Authored):
    """A memo over a campaign of aspects: a summary across them, findings, at most one recommendation, a summary per
    aspect.

    The recommendation is a list of at most one, because the contract holds no nullable slot: empty is a measurement's
    memo, which recommends nothing, and a choice's that found no clear winner. :meth:`recommended` reads it.

    ``summary`` is the one part every kind writes: the writer's read across every aspect, which the campaign's page
    opens on after code's warnings, before the recommendation, so a memo that recommends nothing still says what the
    campaign found. It is written after the aspect summaries, the order a writer reaches a read across them in; the
    page's order is the page's. It is prose like a finding's body, held to the same rules: a number in it is a figure
    reference code renders, and the page holds each sentence to the style pass.
    """

    findings: list[Finding] = Field(description="The findings, across every aspect, in reading order.")
    recommendation: list[Recommendation] = Field(
        max_length=1,
        description=(
            "The one recommendation, or none: empty for a measurement, and for a choice with no clear winner."
        ),
    )
    aspect_summaries: list[AspectSummary] = Field(description="One per aspect, in the campaign's order.")
    summary: ModelProse = Field(
        description=(
            "Your read across every aspect, in one sentence: what the campaign found and what it means for its "
            "question, whatever it recommends."
        ),
    )
    next: list[NextStep] = Field(description="Experiments to run next, highest leverage first.")

    @field_validator("summary")
    @classmethod
    def _summary_is_written(cls, summary: str) -> str:
        """Refuse a summary of nothing but whitespace: every kind's memo says what the campaign found.

        :raises ValueError: The summary holds no word.
        """
        if not summary.strip():
            raise ValueError("`summary` is empty: write the memo's read across every aspect, whatever it recommends")
        return summary

    def recommended(self) -> Recommendation | None:
        """The memo's one recommendation, or None where it makes none.

        :return: The recommendation.
        :rtype: Recommendation | None
        """
        return self.recommendation[0] if self.recommendation else None


class MapFieldInSchema(TypeError):
    """An object admits arbitrary keys, which no strict structured-output mode accepts."""


class UnportableSchema(TypeError):
    """A schema keyword no strict structured-output mode is known to accept, and none that a reply's validation
    stands in for: a structural keyword (``allOf``, ``not``, ``if``), which dropping would change the shape sent; or an
    ``enum`` or ``const`` holding a value that is not a string, which not every strict mode enforces."""


class OffVocabulary(ValueError):
    """A caveat kind the host did not offer."""


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


def count_slots(schema: dict[str, Any]) -> int:
    """The free-text and list slots a schema holds, counting each definition once.

    Strings and arrays are what a provider's grammar pays for (enums and integers barely register), so
    they are what :data:`SLOT_BUDGET` bounds.

    :param schema: A projected schema.
    :ptype schema: dict[str, Any]
    :return: The slot count.
    :rtype: int
    """

    def count(node: Any) -> int:
        if isinstance(node, list):
            return sum(count(item) for item in node)
        if not isinstance(node, dict):
            return 0
        is_free_string = node.get("type") == "string" and "enum" not in node and "const" not in node
        own = 1 if is_free_string or node.get("type") == "array" else 0
        return own + sum(count(value) for key, value in node.items() if key not in ("enum", "const"))

    return count(schema)


def authored_schema(caveat_kinds: Iterable[str]) -> dict[str, Any]:
    """The strict JSON schema of the authored payload, with the host's caveat kinds as an enum.

    :param caveat_kinds: The caveat kinds on offer — the engine's and any the host registered.
    :ptype caveat_kinds: Iterable[str]
    :return: The schema, as sent.
    :rtype: dict[str, Any]
    """
    schema = strict_schema(AuthoredCampaignAnalysis.model_json_schema())
    schema["$defs"]["Caveat"]["properties"]["kind"] = {"type": "string", "enum": sorted(set(caveat_kinds))}
    return schema


def response_format(caveat_kinds: Iterable[str]) -> dict[str, Any]:
    """The ``response_format`` directive that sends :func:`authored_schema` in strict mode.

    :param caveat_kinds: The caveat kinds on offer.
    :ptype caveat_kinds: Iterable[str]
    :return: The OpenAI-shaped ``json_schema`` directive.
    :rtype: dict[str, Any]
    """
    schema = authored_schema(caveat_kinds)
    return {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": schema}}


def validate_authored(payload: Any, caveat_kinds: Iterable[str]) -> AuthoredCampaignAnalysis:
    """Check a returned payload against the schema it was sent: the same models and vocabulary.

    :param payload: The parsed JSON object the generator returned.
    :ptype payload: Any
    :param caveat_kinds: The caveat kinds that were on offer.
    :ptype caveat_kinds: Iterable[str]
    :return: The validated payload.
    :rtype: AuthoredCampaignAnalysis
    :raises pydantic.ValidationError: The payload does not match the models.
    :raises OffVocabulary: A caveat kind that was not on offer.
    """
    document = AuthoredCampaignAnalysis.model_validate(payload)
    kinds = set(caveat_kinds)
    for position, finding in enumerate(document.findings):
        for index, caveat in enumerate(finding.caveats):
            if caveat.kind not in kinds:
                raise OffVocabulary(
                    f"findings[{position}].caveats[{index}].kind {caveat.kind!r} is not one of {sorted(kinds)}"
                )
    return document


__all__ = [
    "SCHEMA_NAME",
    "SLOT_BUDGET",
    "AspectSummary",
    "AuthoredCampaignAnalysis",
    "BarMissed",
    "BehindOn",
    "Caveat",
    "Confidence",
    "EvidenceRef",
    "Finding",
    "FloorOverride",
    "MapFieldInSchema",
    "NextStep",
    "OffVocabulary",
    "Reading",
    "Recommendation",
    "authored_schema",
    "count_slots",
    "response_format",
    "strict_schema",
    "validate_authored",
]
