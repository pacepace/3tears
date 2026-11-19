"""A case's reference key: the entities an answer is graded against, which a host fills."""
from __future__ import annotations
import unicodedata
from typing import Literal, Self
from pydantic import Field, field_validator, model_validator
from threetears.evals.schema.base import CoreShapeModel, EvalDocumentModel

def fold_name(text: str) -> str:
    """The name folded for matching: Unicode NFKC, case folded, whitespace runs as one space, ends stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

class ReferenceFact(EvalDocumentModel, CoreShapeModel):
    """One checkable fact about an entity, with where it is stated."""
    claim: str = Field(min_length=1)
    source_url: str = Field(min_length=1)

class ReferenceEntity(EvalDocumentModel, CoreShapeModel):
    """One entity of a key: its name, the other names it goes by, and facts about it."""
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    facts: tuple[ReferenceFact, ...] = ()

    @field_validator("aliases", mode="after")
    @classmethod
    def _aliases_in_one_order(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        """Aliases are a set: kept in one order, so writing them in another is the same key."""
        return tuple(sorted(aliases, key=lambda alias: (fold_name(alias), alias)))

    @field_validator("facts", mode="after")
    @classmethod
    def _facts_in_one_order(cls, facts: tuple[ReferenceFact, ...]) -> tuple[ReferenceFact, ...]:
        """Facts are a set: kept in one order, so writing them in another is the same key."""
        return tuple(sorted(facts, key=lambda fact: (fact.claim, fact.source_url)))

class ReferenceKey(EvalDocumentModel, CoreShapeModel):
    """The entities a case's answer is graded against: a required core and the others that also count."""
    type: Literal["entity_set"] = "entity_set"
    core: tuple[ReferenceEntity, ...] = ()
    acceptable: tuple[ReferenceEntity, ...] = ()

    @field_validator("core", "acceptable", mode="after")
    @classmethod
    def _entities_in_one_order(cls, entities: tuple[ReferenceEntity, ...]) -> tuple[ReferenceEntity, ...]:
        """Each part is a set: kept in one order, so writing it in another is the same key and the same case."""
        return tuple(sorted(entities, key=lambda entity: (fold_name(entity.name), entity.name)))

    @property
    def entities(self) -> tuple[ReferenceEntity, ...]:
        """Every entity of the key, core first."""
        return (*self.core, *self.acceptable)

    @model_validator(mode="after")
    def _one_entity_per_name(self) -> Self:
        """Refuse a key with no entity, or a name or alias that folds alike for two entities."""
        if not self.entities:
            raise ValueError("a reference key names at least one entity, in core or acceptable")
        owner: dict[str, str] = {}
        for entity in self.entities:
            for said in {fold_name(n) for n in (entity.name, *entity.aliases)}:
                if said in owner:
                    raise ValueError(f"{said!r} names both {owner[said]!r} and {entity.name!r}")
                owner[said] = entity.name
        return self
