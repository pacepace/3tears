"""Listing the entities a piece of prose names: the model step that turns an answer into a set to grade.

A structured output names its entities itself, and a host builds :class:`~threetears.evals.kernel.set_grade.NamedEntity`
values from it in code. Prose does not, so :class:`EntityExtractor` asks a model to list them, and code then grades
the list: a :class:`~threetears.evals.kernel.set_grade.SetGrader` bound to the extractor does both
(``grade_text``). The extractor never sees the case's key, so the list is not primed by what the key holds.

**It records what it spent, and how the list was made.** Every completion, parsed or not, is folded into the cell's
``judge`` ledger, which the kind puts on ``CandidateTelemetry.usage``. Its :attr:`EntityExtractor.identity` names the
model and a digest of everything that shapes the list (the prompts, the entity kind, the fields, the reply schema),
and enters the formula of the measures a grader bound to it declares
(:meth:`~threetears.evals.kernel.set_grade.SetGrader.descriptors`), so sets listed two ways never pool.

**It fails rather than guesses.** A reply cut short raises at once, since the same request would meet the same cap; a
reply that will not parse is asked again twice, then raises. Either is :class:`EntityExtractionFailed`, an
:class:`~threetears.evals.kernel.host.apparatus.ApparatusError`: a fault in the measuring rig, never the candidate's.
Raised out of a kind's ``invoke``, where a kind grades its output, the runner excludes the cell as
``apparatus_failed`` and never scores it, so a host that does not catch it is safe by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from pydantic import Field

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.set_grade import NamedEntity
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.completion import CompletionClient
from threetears.evals.schema.hashing import canonical_digest

#: How many times one text is asked about: the first call and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: The role every extraction call is billed under: it reads the candidate's output, as a judge does.
_ROLE: Final = "judge"

_SYSTEM_PROMPT: Final = (
    "You list the named entities a text mentions. List every entity of the kind you are asked for that the text "
    "names, once each, in the order the text first names them, with its name as the text writes it. List only what "
    "the text itself names: never add an entity it does not mention, and never correct or complete a name. For each "
    "field you are asked for, give the value the text states for that entity, or null when it states none. Answer "
    "with JSON only, in the shape the response format states."
)

_USER_TEMPLATE: Final = "Kind of entity: {entity_kind}\nFields to give for each: {fields}\n\n# Text\n\n{text}"


class EntityExtractionFailed(ApparatusError):
    """The extractor could not list a text's entities: its reply was cut short, or would not parse after retries.

    A fault in the measuring rig, never the candidate's. Out of a kind's ``invoke`` the runner records the cell
    excluded as ``apparatus_failed`` and goes on, so it is never scored as a set naming nothing.
    """


class _ListedEntity(EvalBaseModel):
    """One entity as the extractor's reply lists it."""

    name: str = Field(min_length=1)
    fields: dict[str, str | None] = Field(default_factory=dict)


class _Listing(EvalBaseModel):
    """The extractor's whole reply."""

    entities: list[_ListedEntity]


@dataclass(frozen=True)
class EntityExtractor:
    """A model step that lists the entities of one kind a text names, with the fields asked for each.

    :ivar client: The completion client the step calls. It must name its model (``model_name``), since an unnamed
        extractor would let two models' lists pool.
    :ivar entity_kind: What kind of entity to list, in the words the prompt shows the model ("cities", "sources").
    :ivar fields: The other values to ask for each entity, by field name; each lands on ``NamedEntity.fields`` when
        the text states it.
    """

    client: CompletionClient
    entity_kind: str
    fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a client that names no model, a blank entity kind, and a blank or repeated field.

        :raises ValueError: One of those.
        """
        model = getattr(self.client, "model_name", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"an entity extractor's client names its model (model_name), and {type(self.client).__name__} names "
                "none: the measures record which model listed the set, so an unnamed one would pool with any other"
            )
        if not self.entity_kind.strip():
            raise ValueError("an entity extractor lists entities of one kind, and its entity_kind is blank")
        if any(not field.strip() for field in self.fields) or len(set(self.fields)) != len(self.fields):
            raise ValueError(f"an entity extractor's fields are distinct, non-blank names, and {self.fields!r} are not")

    @property
    def model_name(self) -> str:
        """The model the client calls."""
        # Checked present and a non-blank string at construction.
        return str(getattr(self.client, "model_name", ""))

    @property
    def response_format(self) -> dict[str, Any]:
        """The strict ``json_schema`` directive every call sends: ``{"entities": [{"name", "fields": {...}}]}``, with
        ``fields`` holding exactly the declared fields, each a string or null, and left out when none is declared."""
        entity: dict[str, Any] = {"type": "object", "properties": {"name": {"type": "string"}}}
        if self.fields:
            entity["properties"]["fields"] = {
                "type": "object",
                "properties": {field: {"anyOf": [{"type": "string"}, {"type": "null"}]} for field in self.fields},
            }
        schema = {"type": "object", "properties": {"entities": {"type": "array", "items": entity}}}
        return strict_response_format("entity_listing", schema)

    @property
    def identity(self) -> str:
        """``<model>|extract:<sha12>``: the model, and a digest of the prompts, the entity kind, the fields and the
        reply schema, so any change to how the list is made is a changed identity."""
        digest = canonical_digest(
            {
                "system": _SYSTEM_PROMPT,
                "user": _USER_TEMPLATE,
                "entity_kind": self.entity_kind,
                "fields": list(self.fields),
                "schema": self.response_format,
            }
        )
        return f"{self.model_name}|extract:{digest[:12]}"

    async def extract(self, text: str, *, ledger: RoleUsageLedger) -> tuple[NamedEntity, ...]:
        """List the entities ``text`` names, in the order the reply gives them.

        Blank text names nothing and makes no call. Every completion is added to ``ledger``, a cut-short or
        unreadable one included, since each was billed. What the client itself raises (a provider failure, a
        recording that holds no answer) is not caught: it is the caller's to classify.

        :param text: The prose to read.
        :param ledger: The cell's ``judge`` ledger.
        :return: The entities, each with the declared fields the text stated.
        :raises ValueError: ``ledger`` is not the ``judge`` role's.
        :raises EntityExtractionFailed: A reply was cut short, or none of the attempts' replies parsed.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"an entity extractor's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        if not text.strip():
            return ()
        user = _USER_TEMPLATE.format(
            entity_kind=self.entity_kind, fields=", ".join(self.fields) if self.fields else "none", text=text
        )
        response_format = self.response_format
        unreadable: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            completion = await self.client.generate(system=_SYSTEM_PROMPT, user=user, response_format=response_format)
            ledger.add_llm_result(completion)
            cut_short = describe_incomplete_completion(completion)
            if cut_short is not None:
                raise EntityExtractionFailed(f"the entity extractor's reply was cut short: {cut_short}")
            try:
                return self._read(completion.content)
            except ValueError as refused:
                unreadable = refused
        raise EntityExtractionFailed(
            f"the entity extractor's reply did not parse in {_ATTEMPTS} attempts"
        ) from unreadable

    def _read(self, content: str) -> tuple[NamedEntity, ...]:
        """The entities one finished reply lists.

        :param content: The reply.
        :return: The entities, with each declared field the reply gave a value for.
        :raises ValueError: The reply is not the listing's shape, or an entity's fields are not exactly the declared
            ones.
        """
        listing = _Listing.model_validate(extract_json(content))
        declared = set(self.fields)
        named: list[NamedEntity] = []
        for entity in listing.entities:
            if set(entity.fields) != declared:
                raise ValueError(
                    f"a listed entity gave the fields {sorted(entity.fields)}, and the extractor asked for "
                    f"{sorted(declared)}"
                )
            stated = {field: value for field, value in entity.fields.items() if value is not None and value.strip()}
            named.append(NamedEntity(name=entity.name, fields=stated))
        return tuple(named)


__all__ = ["EntityExtractionFailed", "EntityExtractor"]
