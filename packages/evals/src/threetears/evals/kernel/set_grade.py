"""Grading a named set against a case's reference key: how much of the key an output found, and how much of what it
named the key holds.

An output names entities: a list a host tool returned, or the names a model step lists from prose
(:class:`~threetears.evals.kernel.entity_extraction.EntityExtractor`). :func:`grade_set` matches each against the
case's :class:`~threetears.evals.schema.reference.ReferenceKey` and counts both directions at once: recall over the
full key and over its core, and the share of what was named that the key holds. A named entity the key lacks is
**unverified, never wrong**: a key is allowed to be non-exhaustive, so an entity outside it may be a good answer the
key's author did not list. No reading here calls one an error, and :attr:`SetGrade.unverified_names` lists them for a
reader to check.

**Matching is the host's to replace.** :class:`FoldingMatcher` is the default: a name or alias equal after
:func:`~threetears.evals.schema.reference.fold_name` (Unicode NFKC, case folded, whitespace runs as one space). It
uses no word list and strips no accent, since both are language-specific. A host with its own normaliser plugs it in
as an :class:`EntityMatcher`, which sees the whole key, and names it by its ``identity``.

**The measures state their population by absence.** :meth:`SetGrade.host_measures` lands each reading on a kind's
``CandidateOutput.host_measures`` under ``<prefix>_<measure>``. A reading with no value on a cell (core recall where the
key has no core, the verified share where nothing was named) is left off it, so that cell is outside that measure's
n, and an empty output reads differently from a perfect one. :func:`set_grade_descriptors` declares the five
measures, and its ``formula`` names the matcher and the extractor, so a changed matcher or extractor is a changed
measure definition and runs graded two ways never pool.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

from pydantic import ConfigDict, Field, model_validator

from threetears.evals.kernel.metrics import MeasureFamily, MetricDescriptor
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.reference import ReferenceKey, fold_name

if TYPE_CHECKING:
    from threetears.evals.kernel.entity_extraction import EntityExtractor

#: The family a set grade's measures carry when a model step listed the entities: the numbers are code's, over names
#: a judge read off prose, so they are a judge's in the sense that decides ranking (``graded_by``). A host that grades
#: an extracted set declares it on its measure registry; a structured set's measures are the engine's ``mechanical``.
EXTRACTED_SET_FAMILY = MeasureFamily(
    name="extracted_set",
    graded_by="judge",
    description="A named set graded by code against a case's reference key, over the names a judge listed from prose.",
)

#: What the default matcher is called in a measure's formula. Raise it with any change to how it matches.
_FOLDING_IDENTITY = "fold-v1"

#: The revision of the grading rule itself, as a measure's formula names it.
_GRADE_REVISION = "set grade v1"

#: What a measure's formula names in place of an extractor when the host supplied the named set itself.
_STRUCTURED = "structured"

#: A measure-name prefix: the shape a measure key is written in.
_PREFIX_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

_RECALL = "recall"
_CORE_RECALL = "core_recall"
_VERIFIED_SHARE = "verified_share"
_NAMED = "named"
_UNVERIFIED = "unverified"


def _measure_name(prefix: str, measure: str) -> str:
    """The key one reading is stored under: ``<prefix>_<measure>``.

    :param prefix: The host's prefix for this set.
    :param measure: The reading.
    :return: The measure key.
    :raises ValueError: The prefix is not a measure key's shape (lower case, digits and ``_``, starting with a letter).
    """
    if _PREFIX_PATTERN.fullmatch(prefix) is None:
        raise ValueError(
            f"a set grade's prefix is the start of a measure key: lower case letters, digits and '_', starting with "
            f"a letter, and {prefix!r} is not"
        )
    return f"{prefix}_{measure}"


class NamedEntity(EvalBaseModel):
    """One entity an output named: what it was called, and any fields that came with it (a host tool's pick)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="What the output called the entity.")
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Other values the output gave for the entity, by field name. A matcher may read them; the default "
        "reads none.",
    )


class EntityMatcher(Protocol):
    """How a named entity is found in a key: the host's normaliser, or :class:`FoldingMatcher`."""

    @property
    def identity(self) -> str:
        """What this matching is called. A host changes it whenever its matching changes, since it enters the
        measures' definition and a changed definition never pools with the old one."""
        ...

    def match(self, named: NamedEntity, key: ReferenceKey) -> str | None:
        """The name of the key entity ``named`` is, or None when the key holds no entity it is.

        :param named: One entity the output named.
        :param key: The case's whole key, so a matcher may read every entity, alias and fact of it.
        :return: A key entity's ``name``, exactly as the key spells it, or None.
        """
        ...


class FoldingMatcher:
    """The default matcher: a named entity is a key entity when its folded name equals that entity's folded name or
    one of its folded aliases (:func:`~threetears.evals.schema.reference.fold_name`).

    It folds case and Unicode compatibility forms only: ``É`` matches ``é`` and ``ﬁ`` matches ``fi``, while ``é`` does
    not match ``e``, since which accents a language may drop is that language's rule, and no word list is read.
    """

    @property
    def identity(self) -> str:
        """``fold-v1``."""
        return _FOLDING_IDENTITY

    def match(self, named: NamedEntity, key: ReferenceKey) -> str | None:
        """The key entity whose folded name or alias equals ``named``'s folded name, or None.

        :param named: One entity the output named.
        :param key: The case's key. Its validation guarantees no folded name or alias belongs to two entities.
        :return: That entity's name, or None.
        """
        said = fold_name(named.name)
        for entity in key.entities:
            if said in {fold_name(name) for name in (entity.name, *entity.aliases)}:
                return entity.name
        return None


class SetGrade(EvalBaseModel):
    """One output's named set graded against one key: the counts, the names behind them, and the readings.

    ``named`` counts distinct picks: every spelling the matcher finds as one key entity is one pick, and unmatched
    names whose folded names are equal are one pick. ``verified`` counts the picks the matcher found in the key, one
    per key entity found, and the rest are unverified: outside a key that may be non-exhaustive, so never counted
    wrong.
    """

    model_config = ConfigDict(frozen=True)

    named: int = Field(
        ge=0, description="Distinct picks the output made: one per key entity found, one per unmatched name."
    )
    verified: int = Field(ge=0, description="How many of them the matcher found in the key: one per key entity found.")
    core_size: int = Field(ge=0, description="Entities in the key's core.")
    key_size: int = Field(ge=1, description="Entities in the whole key, core and acceptable.")
    found: tuple[str, ...] = Field(description="The key entities the output named, by key name, core first.")
    core_missed: tuple[str, ...] = Field(description="The core entities the output did not name, in key order.")
    unverified_names: tuple[str, ...] = Field(
        description="The named entities the key does not hold, as the output first spelled them, in its order."
    )

    @model_validator(mode="after")
    def _counts_agree(self) -> SetGrade:
        """Refuse counts that contradict each other or the names listed beside them."""
        if self.verified > self.named:
            raise ValueError(f"{self.verified} verified of {self.named} named: a verified entity is a named one")
        if self.verified != len(self.found):
            raise ValueError(f"{self.verified} verified for {len(self.found)} key entities found: one pick each")
        if len(self.unverified_names) != self.named - self.verified:
            raise ValueError(
                f"{len(self.unverified_names)} unverified names listed for {self.named - self.verified} unverified "
                "entities"
            )
        if self.core_size > self.key_size or len(self.found) > self.key_size:
            raise ValueError(
                f"a key of {self.key_size} entities holds a core of {self.core_size} and {len(self.found)} found"
            )
        if len(self.core_missed) > self.core_size:
            raise ValueError(f"{len(self.core_missed)} core entities missed of a core of {self.core_size}")
        return self

    @property
    def unverified(self) -> int:
        """Named entities the key does not hold: unverified, never wrong."""
        return self.named - self.verified

    @property
    def recall(self) -> float:
        """The share of the whole key the output named; 0 when it named nothing."""
        return len(self.found) / self.key_size

    @property
    def core_recall(self) -> float | None:
        """The share of the key's core the output named, or None when the key has no core."""
        if self.core_size == 0:
            return None
        return (self.core_size - len(self.core_missed)) / self.core_size

    @property
    def verified_share(self) -> float | None:
        """The share of what the output named that the key holds, or None when it named nothing."""
        if self.named == 0:
            return None
        return self.verified / self.named

    def host_measures(self, prefix: str) -> dict[str, float]:
        """The readings as a kind lands them on ``CandidateOutput.host_measures``, each under ``<prefix>_<measure>``.

        A reading with no value is left out, so the cell is outside that measure's n: core recall where the key has
        no core, the verified share where nothing was named.

        :param prefix: The prefix :func:`set_grade_descriptors` declared the measures under.
        :return: Measure key to value.
        :raises ValueError: The prefix is not a measure key's shape.
        """
        readings: dict[str, float | None] = {
            _RECALL: self.recall,
            _CORE_RECALL: self.core_recall,
            _VERIFIED_SHARE: self.verified_share,
            _NAMED: float(self.named),
            _UNVERIFIED: float(self.unverified),
        }
        return {_measure_name(prefix, measure): value for measure, value in readings.items() if value is not None}


def grade_set(named: Sequence[NamedEntity], key: ReferenceKey, *, matcher: EntityMatcher | None = None) -> SetGrade:
    """Grade the entities an output named against a case's key, in both directions.

    Every named entity is matched, so a host matcher sees each spelling and its fields. One the matcher finds is
    verified and the key entity it names is found, and every spelling found as the same key entity is one pick: a
    name repeated or written two ways does not count twice. One it does not find is unverified, and unmatched names
    whose folded names are equal are one pick, the first spelling kept.

    :param named: What the output named, in its order: a host's structured picks, or an extractor's list.
    :param key: The case's reference key.
    :param matcher: How a named entity is found in the key; None is :class:`FoldingMatcher`.
    :return: The grade.
    :raises ValueError: The matcher named an entity the key does not hold, which is the matcher's defect.
    """
    matching = matcher if matcher is not None else FoldingMatcher()
    names = {entity.name for entity in key.entities}
    found: set[str] = set()
    unmatched: set[str] = set()
    unverified: list[str] = []
    for entity in named:
        hit = matching.match(entity, key)
        if hit is None:
            said = fold_name(entity.name)
            if said not in unmatched:
                unmatched.add(said)
                unverified.append(entity.name)
            continue
        if hit not in names:
            raise ValueError(
                f"the matcher {matching.identity!r} matched {entity.name!r} to {hit!r}, which the key does not hold; "
                "a matcher answers a key entity's name or None"
            )
        found.add(hit)
    return SetGrade(
        named=len(found) + len(unverified),
        verified=len(found),
        core_size=len(key.core),
        key_size=len(key.entities),
        found=tuple(entity.name for entity in key.entities if entity.name in found),
        core_missed=tuple(entity.name for entity in key.core if entity.name not in found),
        unverified_names=tuple(unverified),
    )


def set_grade_descriptors(
    prefix: str,
    *,
    reader_label: str,
    matcher: EntityMatcher | None = None,
    extractor: EntityExtractor | None = None,
) -> tuple[MetricDescriptor, ...]:
    """The five measures :meth:`SetGrade.host_measures` lands, for a host's measure registry.

    Each description states its population. The ``formula`` names the grading rule, the matcher's identity and the
    extractor's (``structured`` when the host supplied the set), so a change to either is a changed definition and
    a run graded one way never pools with one graded another. Without an extractor the family is ``mechanical``;
    with one it is :data:`EXTRACTED_SET_FAMILY`, which the host then declares on its registry.

    :param prefix: The start of each measure key, as :meth:`SetGrade.host_measures` is passed it.
    :param reader_label: What a reader calls the set ("Picks"), which starts each measure's reader name.
    :param matcher: The matcher the kind grades with; None is :class:`FoldingMatcher`.
    :param extractor: The extractor that lists the set from prose, or None for a set the host supplies.
    :return: Recall, core recall, verified share, named and unverified, in that order.
    :raises ValueError: The prefix is not a measure key's shape, or the reader label is blank.
    """
    label = reader_label.strip()
    if not label:
        raise ValueError("a set grade's reader label is what a reader calls the set, and it is blank")
    matching = matcher if matcher is not None else FoldingMatcher()
    source = extractor.identity if extractor is not None else _STRUCTURED
    common: dict[str, object] = {
        "data_type": "numeric",
        "family": EXTRACTED_SET_FAMILY.name if extractor is not None else "mechanical",
        "transferability_class": "judge_mediated" if extractor is not None else "mechanical",
        "attribution_scope": "end_to_end",
        "formula": f"{_GRADE_REVISION} | matcher {matching.identity} | extractor {source}",
        "population": "scored",
    }
    share: dict[str, object] = {
        **common,
        "higher_is_better": True,
        "value_range": (0.0, 1.0),
        "merit_axis": "quality",
        "scale": "ratio",
    }
    count: dict[str, object] = {**common, "diagnostic": True, "nonnegative": True, "unit": "entities"}
    return tuple(
        MetricDescriptor.model_validate(fields)
        for fields in (
            {
                **share,
                "name": _measure_name(prefix, _RECALL),
                "reader_name": f"{label} found, of the key",
                "description": "The share of the case's whole key the output named, on every graded cell.",
            },
            {
                **share,
                "name": _measure_name(prefix, _CORE_RECALL),
                "reader_name": f"{label} found, of the core",
                "description": "The share of the key's core the output named, on the cells whose key has a core.",
            },
            {
                **share,
                "name": _measure_name(prefix, _VERIFIED_SHARE),
                "reader_name": f"{label} named, found in the key",
                "description": "The share of what the output named that the key holds, on the cells that named "
                "at least one; the rest are unverified, not wrong.",
            },
            {
                **count,
                "name": _measure_name(prefix, _NAMED),
                "reader_name": f"{label} named",
                "description": "How many distinct picks the output made, one per key entity found and one per "
                "unmatched name, on every graded cell.",
            },
            {
                **count,
                "name": _measure_name(prefix, _UNVERIFIED),
                "reader_name": f"{label} named, not in the key",
                "description": "How many named entities the key does not hold, on every graded cell; a key may be "
                "non-exhaustive, so these are unverified, never wrong.",
            },
        )
    )


__all__ = [
    "EXTRACTED_SET_FAMILY",
    "EntityMatcher",
    "FoldingMatcher",
    "NamedEntity",
    "SetGrade",
    "grade_set",
    "set_grade_descriptors",
]
