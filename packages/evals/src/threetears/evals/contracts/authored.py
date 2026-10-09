"""What the analysis generator authors: a small document shape carrying freeform analysis.

**The boundary.** This contract types what the document is MADE OF and the
hooks code acts on, never what it says or how much it says:

- **Typed here:** the unit kinds (finding, decision, question answer, next step); the order of every
  list, which is the reading order; links between units by POSITION, so a decision that rests on
  findings ``[0, 2]`` cites the first and third; the readings code turns into numbers (a cell and a
  measure or judged dimension); and a closed vocabulary code branches or renders on.
- **Not the model's to say:** what a verdict stands on. The evidence tier is read off the readings a
  finding names (:func:`threetears.evals.contracts.campaign.evidence_tier_of`), so this contract offers no
  tier field, and a payload that supplies one is refused as an unknown key.
- **The model's judgment:** what a finding is about and how many there are; every title, body,
  proposal, answer and rationale, in prose with markdown allowed; and whatever else is worth saying,
  which goes in the summary.

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

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.contracts.prose import ModelProse

#: The name the schema is sent under.
SCHEMA_NAME = "analysis"

#: The most free-text and list slots the sent schema may hold (:func:`count_slots`). Measured
#: against one strict-schema provider by padding this contract with string fields until the
#: provider refused it, and the budget sits just under the largest size measured to pass. Moving it
#: up is a decision to re-probe every eligible writer first, not an edit.
SLOT_BUDGET = 40

#: The chart type meaning "this finding has no chart" — a value, not an absence, so no union is needed.
NO_CHART = "none"

#: How sure the author is of a finding or a decision.
Confidence = Literal["very_high", "high", "medium", "low"]
#: What a piece of evidence reads: a ``measure``, or a ``judged`` dimension.
Reading = Literal["measure", "judged"]


class _Authored(BaseModel):
    """A closed object: the schema sent is strict, the check on return is strict, and so is a stored read."""

    model_config = ConfigDict(extra="forbid")


class MeasureRef(_Authored):
    """A measure or judged dimension, named in its namespace — which one is stated, never inferred."""

    measure_id: str = Field(description="The measure or judged dimension, spelled as the bundle spells it.")
    reading: Reading = Field(description="`measure` for a measure, `judged` for a judged dimension.")


class EvidenceRef(_Authored):
    """One reading at one cell. Code fills the number, its sample size and its spread."""

    cell: str = Field(description="The cell this reading is taken at.")
    measure_id: str = Field(description="The measure or judged dimension, spelled as the bundle spells it.")
    reading: Reading = Field(description="`measure` for a measure, `judged` for a judged dimension.")


class Chart(_Authored):
    """A chart code draws from the named cells and measures; each type reads its lists by position."""

    type: str = Field(description=f"The chart type, or `{NO_CHART}` when the finding has no chart.")
    cells: list[str] = Field(description="The cells drawn, in order; empty draws every cell where the type allows.")
    measures: list[MeasureRef] = Field(description="The readings drawn, in the order the type reads them.")
    axis: str = Field(description="The lever a sweep or attribution is drawn along; empty otherwise.")
    note: ModelProse = Field(description="A null result's mechanism: why the lever cannot act. Empty otherwise.")
    caption: ModelProse = Field(
        description="Your prose beside the chart: what it supports that it does not draw; empty when nothing."
    )


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
    chart: Chart
    caveats: list[Caveat]
    invalidates: list[int] = Field(description="Positions of findings this one invalidates.")
    durable: ModelProse = Field(description="A durable claim about the subject for later analyses; empty when none.")


class Decision(_Authored):
    """A proposal and the verdict on it."""

    proposal: ModelProse = Field(description="One thing to do, stated positively: what to configure, or change.")
    disposition: Literal["adopted", "rejected", "deferred"] = Field(
        description="`adopted` means act on it now; an option that still needs a confirming run is `deferred`."
    )
    cells: list[str] = Field(description="The cells whose arm this adopts or rejects; empty when it names no arm.")
    confidence: Confidence
    rests_on: list[int] = Field(description="Positions in `findings` this rests on.")
    revisit_when: ModelProse = Field(description="What specifically would settle it; empty unless deferred.")


class QuestionAnswer(_Authored):
    """Where one declared question stands on this evidence."""

    question_id: str = Field(description="A live declared question's id; one entry per live question.")
    resolution: Literal["answered", "partial", "unanswerable"]
    answer: ModelProse = Field(
        description="Opens with the answer in one plain sentence; or why it cannot be given and what would give it."
    )
    rests_on: list[int] = Field(description="Positions in `findings` this rests on.")


class NextStep(_Authored):
    """An experiment worth running next."""

    title: ModelProse = Field(description="The experiment, stated as an action.")
    why: ModelProse = Field(description="What it would change and how to run it.")
    leverage: Literal["high", "medium", "low"]
    lever: str = Field(description="The lever it would measure; empty when none.")


class AuthoredAnalysis(_Authored):
    """A decision memo over one campaign's evidence."""

    headline: ModelProse = Field(
        description=(
            "At most twelve plain words a non-specialist understands: what to do and why. No numbers, setting names, or jargon."
        )
    )
    summary: ModelProse = Field(
        description=(
            "Three to five markdown bullets for a reader who reads nothing else: what was learned, what to do, what is "
            "still open. Plain words a non-specialist understands, short sentences, no jargon; evidence and caveats "
            "belong in the findings."
        )
    )
    findings: list[Finding] = Field(description="The findings, in reading order.")
    decisions: list[Decision]
    questions: list[QuestionAnswer]
    next: list[NextStep] = Field(description="Experiments to run next, highest leverage first.")


class MapFieldInSchema(TypeError):
    """An object admits arbitrary keys, which no strict structured-output mode accepts."""


class OffVocabulary(ValueError):
    """A caveat kind or chart type the host did not offer."""


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Project a pydantic JSON schema onto the strict subset every eligible provider accepts.

    Every object closed and fully required; ``oneOf`` as ``anyOf``; a ``$ref`` with nothing beside it
    (OpenAI refuses a sibling keyword); and no ``default``, ``discriminator`` or ``title``.

    Args:
        schema: A pydantic-generated JSON schema.

    Returns:
        The projected schema.

    Raises:
        MapFieldInSchema: An object carries an open ``additionalProperties``.
    """

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
            if key in ("default", "discriminator", "title"):
                continue
            target = "anyOf" if key == "oneOf" else key
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


def count_slots(schema: dict[str, Any]) -> int:
    """The free-text and list slots a schema holds, counting each definition once.

    Strings and arrays are what a provider's grammar pays for (enums and integers barely register), so
    they are what :data:`SLOT_BUDGET` bounds.

    Args:
        schema: A projected schema.

    Returns:
        The slot count.
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


def authored_schema(caveat_kinds: Iterable[str], chart_types: Iterable[str] | Mapping[str, str]) -> dict[str, Any]:
    """The strict JSON schema of the authored payload, with the host's vocabularies as enums.

    Args:
        caveat_kinds: The caveat kinds on offer — the engine's and any the host registered.
        chart_types: The chart types on offer; :data:`NO_CHART` is always among them. Given as a
            mapping, each type's value is how it reads the chart's positional lists, and the model is
            told so in the chart type's description — the one place that contract reaches it.

    Returns:
        The schema, as sent.
    """
    schema = strict_schema(AuthoredAnalysis.model_json_schema())
    defs = schema["$defs"]
    defs["Caveat"]["properties"]["kind"] = {"type": "string", "enum": sorted(set(caveat_kinds))}
    chart_type: dict[str, Any] = {"type": "string", "enum": sorted({NO_CHART, *chart_types})}
    if isinstance(chart_types, Mapping):
        readings = "; ".join(f"{name}: {reads}" for name, reads in sorted(chart_types.items()))
        chart_type["description"] = f"`{NO_CHART}` when the finding has no chart. Each type reads: {readings}."
    defs["Chart"]["properties"]["type"] = chart_type
    return schema


def response_format(caveat_kinds: Iterable[str], chart_types: Iterable[str] | Mapping[str, str]) -> dict[str, Any]:
    """The ``response_format`` directive that sends :func:`authored_schema` in strict mode.

    Args:
        caveat_kinds: The caveat kinds on offer.
        chart_types: The chart types on offer.

    Returns:
        The OpenAI-shaped ``json_schema`` directive.
    """
    schema = authored_schema(caveat_kinds, chart_types)
    return {"type": "json_schema", "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": schema}}


def validate_authored(payload: Any, caveat_kinds: Iterable[str], chart_types: Iterable[str]) -> AuthoredAnalysis:
    """Check a returned payload against the schema it was sent: the same models and vocabularies.

    Args:
        payload: The parsed JSON object the generator returned.
        caveat_kinds: The caveat kinds that were on offer.
        chart_types: The chart types that were on offer.

    Returns:
        The validated payload.

    Raises:
        pydantic.ValidationError: The payload does not match the models.
        OffVocabulary: A caveat kind or chart type that was not on offer.
    """
    document = AuthoredAnalysis.model_validate(payload)
    kinds, types = set(caveat_kinds), {NO_CHART, *chart_types}
    for position, finding in enumerate(document.findings):
        if finding.chart.type not in types:
            raise OffVocabulary(f"findings[{position}].chart.type {finding.chart.type!r} is not one of {sorted(types)}")
        for index, caveat in enumerate(finding.caveats):
            if caveat.kind not in kinds:
                raise OffVocabulary(
                    f"findings[{position}].caveats[{index}].kind {caveat.kind!r} is not one of {sorted(kinds)}"
                )
    return document


__all__ = [
    "NO_CHART",
    "SCHEMA_NAME",
    "SLOT_BUDGET",
    "AuthoredAnalysis",
    "Caveat",
    "Chart",
    "Confidence",
    "Decision",
    "EvidenceRef",
    "Finding",
    "MapFieldInSchema",
    "MeasureRef",
    "NextStep",
    "OffVocabulary",
    "QuestionAnswer",
    "Reading",
    "authored_schema",
    "count_slots",
    "response_format",
    "strict_schema",
    "validate_authored",
]
