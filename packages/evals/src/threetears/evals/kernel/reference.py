"""A case's reference key as a judge reads it: the case material a kind appends for every judge grading against it.

The engine injects nothing. A kind whose judges grade an answer against the case's key appends
:func:`reference_material` to the ``JudgeEvidence.case_material`` it renders for each of them, so the key enters the
label fingerprint as all judged evidence does; it never hands the key to the candidate. Code measures read
``EvalTestCase.reference`` directly.
"""

from __future__ import annotations

from threetears.evals.schema.reference import ReferenceEntity, ReferenceFact, ReferenceKey

#: The line the material opens on. A key may be non-exhaustive, so a judge reading an answer that names an entity
#: outside it must not count that entity wrong.
_NON_EXHAUSTIVE_LINE = (
    "Reference key. It may not list every entity that counts: an entity the answer names outside it is unverified, "
    "not wrong."
)


def _fact_line(fact: ReferenceFact) -> str:
    return f"    - {fact.claim} (source: {fact.source_url})"


def _entity_lines(entity: ReferenceEntity) -> list[str]:
    head = f"  - {entity.name}"
    if entity.aliases:
        head += f" (also called: {'; '.join(entity.aliases)})"
    return [head, *(_fact_line(fact) for fact in entity.facts)]


def _part_lines(heading: str, entities: tuple[ReferenceEntity, ...]) -> list[str]:
    if not entities:
        return [f"{heading} none."]
    return [heading, *(line for entity in entities for line in _entity_lines(entity))]


def reference_material(key: ReferenceKey) -> str:
    """The key as plain text for a judge: the non-exhaustive line, then the core, then the acceptable others.

    Each entity is listed with the other names it goes by and its facts, each with where it is stated. The text is
    deterministic: the key keeps its parts in one order, so two keys that are the same key render the same text, and
    a judge's label fingerprint over it moves only when the key does.

    :param key: The case's reference key.
    :return: The material, one line per entity, alias list and fact, with no trailing newline.
    """
    return "\n".join(
        [
            _NON_EXHAUSTIVE_LINE,
            *_part_lines("Required (core):", key.core),
            *_part_lines("Also acceptable:", key.acceptable),
        ]
    )


__all__ = ["reference_material"]
