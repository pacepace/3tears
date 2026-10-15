"""What a seeded definition's content is, as one digest: the record a seed writes and a drift read compares.

A seeded definition is compared three ways (:mod:`threetears.evals.run.definition_drift`): what the store holds, what
the seed corpus carries now, and what was seeded. All three are digests of **content only**, under the discipline
:func:`~threetears.evals.kernel.prompts.drift.seed_digest` keeps for prompts: never an id, a revision number, a scope
or a timestamp, since a reset writes the same content under a new id and revision, and the record of what was seeded
is not itself content. A field at its default is left out of the pre-image, so a field a later minor adds (default
None) moves no digest of a record written before it.

- A template, a rubric dim and a judge config digest their declared fields less the record's own (:data:`RECORD_FIELDS`).
- A judge case digests as the frozen case it is (``JudgeCase.content_digest``), which a stored case already carries
  as its ``content_hash``.
- A template's own case digests what its corpus file states of it: its ``variation_params``, its ``host_payload`` and
  its ``stratum``, read alike off a corpus case and a stored one (so a stratum relabelled in place moves it), and
  never its name or its template's id, so the same case on a newer revision digests alike.
- A case set digests its name, its template's name, ``tracked`` and the content digests of its cases in order, so
  the same cases frozen into a newer revision of the template digest alike, and a case whose words changed moves it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import CatalogRubricDim, EvalTemplate, JudgeConfig, current_revision

#: The fields that say which record a definition is and how it was written, never what it says: left out of every
#: definition digest.
RECORD_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "doc_type",
        "schema_version",
        "schema_minor",
        "scope_id",
        "created_at",
        "updated_at",
        "revision",
        "seed_digest",
    }
)


def _content(document: BaseModel) -> dict[str, object]:
    """A definition's declared content fields, each left out at its default."""
    declared = set(type(document).model_fields) - RECORD_FIELDS
    return document.model_dump(mode="json", include=declared, exclude_defaults=True)


def template_digest(template: EvalTemplate) -> str:
    """The digest of a template's content, as stored (its ``kind_spec`` as the kind's spec model resolved it)."""
    return canonical_digest({"eval_template": _content(template)})


def rubric_dim_digest(dim: CatalogRubricDim) -> str:
    """The digest of a catalogue rubric dim's content."""
    return canonical_digest({"rubric_dim": _content(dim)})


def judge_config_digest(config: JudgeConfig) -> str:
    """The digest of a judge config's content."""
    return canonical_digest({"judge_config": _content(config)})


def case_set_digest(*, name: str, template: str, tracked: bool, cases: Sequence[str]) -> str:
    """The digest of a case set's content.

    Args:
        name: The set's name.
        template: Its template's name: every revision of the template shares it.
        tracked: Whether it is followed over time.
        cases: The content digest of each case it lists, in order (a stored case's ``content_hash``).
    """
    return canonical_digest(
        {"case_set": {"name": name, "template": template, "tracked": tracked, "cases": list(cases)}}
    )


def template_case_digest(
    *, variation_params: Mapping[str, str], host_payload: Mapping[str, Any], stratum: str | None
) -> str:
    """The digest of a template's own case: what varies, the host's stimulus and the stratum, each left out empty.

    Args:
        variation_params: The case's axis values.
        host_payload: Its stimulus in the host's vocabulary, digested as written and never read.
        stratum: The kind of case it is, or ``None``.
    """
    content: dict[str, object] = {}
    if variation_params:
        content["variation_params"] = dict(variation_params)
    if host_payload:
        content["host_payload"] = dict(host_payload)
    if stratum is not None:
        content["stratum"] = stratum
    return canonical_digest({"template_case": content})


def current_revisions(templates: Iterable[EvalTemplate]) -> dict[str, EvalTemplate]:
    """Each template name's current revision (:func:`~threetears.evals.schema.models.current_revision`), by name."""
    by_name: dict[str, list[EvalTemplate]] = {}
    for template in templates:
        by_name.setdefault(template.name, []).append(template)
    resolved = {name: current_revision(records) for name, records in by_name.items()}
    return {name: template for name, template in resolved.items() if template is not None}


__all__ = [
    "RECORD_FIELDS",
    "case_set_digest",
    "current_revisions",
    "judge_config_digest",
    "rubric_dim_digest",
    "template_case_digest",
    "template_digest",
]
