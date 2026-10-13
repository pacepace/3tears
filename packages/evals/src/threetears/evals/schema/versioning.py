"""Which stored documents are kept across releases, which are regenerated, and how a kept one is read when it is old.

**The evidence core is kept.** A document the engine cannot regenerate, and that an audit needs, is in the
core (:data:`CORE_DOC_TYPES`, the ``kept`` entries of :data:`~threetears.evals.schema.doc_types.DOC_TYPE_FACTS`):
the evidence a run produced (its test cases, the run, its results and traces, human calibration ratings, the
out-of-run spend ledger), the launch inputs those documents name by id (the template, the judge config, the
rubric dim, their tombstones, and the case set), the campaign a person declared and gathered runs into, and
the records of what a sweep and each analysis generation came to, which attempt holds a campaign's
generation slot, and the records of every other job — a launch, a second judge's pass, a judge temperature
comparison, a host action. A core document is read
forever from :data:`CORE_BASELINE_VERSION` on: a read of an older core document upgrades it in memory, one
version at a time, through :data:`CORE_UPGRADERS`, and the result is validated as strictly as a document
written today. The stored bytes are not touched; the next write of that document (a rescore, an archive
toggle's read-modify-write) writes the current version, so migration is lazy and a scope may hold documents
at several core versions at once.

**Everything else is regenerable.** Analyses, insights and judge profiles are derived from the core, and a cassette
is re-captured, so they keep the strict rule they always had: a
document written under any :data:`REGENERABLE_SCHEMA_VERSION` but this build's is refused on read, and the
operator regenerates it. The bundle (``schema_version``), the report (``REPORT_VERSION``), the cell model and
the identity keys (``IDENTITY_VERSION``) are separate version spaces with their own pins; an upgrader never
recomputes a key or a digest.

**Reads stay strict.** A core document whose ``schema_version`` is not an integer, is above
:data:`CORE_SCHEMA_VERSION` (written by a newer build — two builds sharing one store), or is below
:data:`CORE_BASELINE_VERSION` (written before the first public release) is refused, by name, never guessed.

**How a core change ships.** Any change to a core type's shape is a bump, an added optional field included:
copy the frozen fixtures under ``tests/fixtures/core_documents/`` to the next version's directory before
changing a shape, bump :data:`CORE_SCHEMA_VERSION` with a ledger line, append one :class:`CoreUpgrader` whose
``from_version`` is the old number (an identity step for a purely additive change), and re-pin
``tests/test_core_schema_pins.py``. A step obeys four rules: it is pure (no store, host, clock or randomness);
it never touches :data:`CORE_ADDRESSING_FIELDS`, ``host_payload``, ``kind_payload`` or a stored key or digest
(``variant_key``, ``context_key``, ``context_components``, ``content_hash``, ``identity_version``), because the
store filters, sorts and projects on the stored bytes before any upgrade runs; a value it cannot derive
becomes None, read as "not recorded"; and its output must validate strictly.

The document field stays ``schema_version`` for both spaces; ``doc_type`` says which number it is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from threetears.evals.schema.doc_types import DOC_TYPE_FACTS

#: The first core version a public release wrote, and the oldest this and every later build reads. Below it a
#: document is pre-release, and is refused by name.
CORE_BASELINE_VERSION: Final[int] = 14

#: The core version this build writes. Every bump appends one line here and one :class:`CoreUpgrader`.
#:
#: - **v14** — the baseline: the first public release's evidence core. Every field a writer always sets is
#:   required; an optional field's ``None`` has one present-day meaning, stated on the field.
CORE_SCHEMA_VERSION: int = 14
"""The core version this build writes, and the newest a read of a core document accepts."""

#: The version every regenerable document is written under, and the only one a read of one accepts.
#:
#: - **v9** — the baseline: the first public release's regenerable shapes.
REGENERABLE_SCHEMA_VERSION: int = 9
"""The version every regenerable document is written under, and the only one a read of one accepts.

A bump is a drop: regenerable documents written under the old version are refused on read
(:data:`~threetears.evals.schema.models.SchemaVersion`) and regenerated from the core, never migrated.
"""

#: The kept documents, by ``doc_type``: the ``kept`` entries of
#: :data:`~threetears.evals.schema.doc_types.DOC_TYPE_FACTS`, derived so the registry is the one place a type's
#: lifecycle is decided. Embedded shapes (a result's rubric scores, usage rows and goal-state outcomes, a run's
#: context components and subject snapshot, a ledger's calls, a campaign's declared design, a sweep's arms) are
#: core through their parent.
CORE_DOC_TYPES: Final[frozenset[str]] = frozenset(
    doc_type for doc_type, facts in DOC_TYPE_FACTS.items() if facts.lifecycle == "kept"
)

#: The fields every core document is addressed by in the store.
_ADDRESSED_EVERYWHERE = frozenset({"id", "doc_type", "schema_version", "scope_id"})

#: Per core type, every field the store reads on the stored bytes, before any upgrade: what a query filters or
#: orders on, what a projection keeps, and what a partial write merges. An upgrader may not touch one, since
#: an old document's value is what the store matches; renaming one needs a decision and a rewrite of the store.
CORE_ADDRESSING_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "eval_test_case": _ADDRESSED_EVERYWHERE | {"template_id", "stratum"},
        "eval_run": _ADDRESSED_EVERYWHERE
        | {"status", "archived", "created_at", "cancel_requested_at", "cancel_request_reason"},
        "eval_result": _ADDRESSED_EVERYWHERE | {"eval_run_id", "test_case_id", "model"},
        "eval_trace": _ADDRESSED_EVERYWHERE,
        "calibration_rating": _ADDRESSED_EVERYWHERE
        | {"run_id", "result_id", "rated_at", "output_fingerprint", "criterion_fingerprint"},
        "eval_out_of_run_spend": _ADDRESSED_EVERYWHERE | {"purpose", "launch_group_id", "template_id", "created_at"},
        "eval_template": _ADDRESSED_EVERYWHERE | {"name", "archived", "universal"},
        "judge_config": _ADDRESSED_EVERYWHERE | {"rubric_dim_id", "archived", "created_at"},
        "rubric_dim": _ADDRESSED_EVERYWHERE | {"key", "archived", "created_at"},
        "rubric_dim_tombstone": _ADDRESSED_EVERYWHERE | {"deleted_at"},
        "judge_config_tombstone": _ADDRESSED_EVERYWHERE | {"deleted_at"},
        "case_set": _ADDRESSED_EVERYWHERE | {"name", "version"},
        "eval_campaign": _ADDRESSED_EVERYWHERE | {"subject_id", "behavior", "archived", "created_at"},
        "eval_sweep": _ADDRESSED_EVERYWHERE | {"cancel_requested_at", "cancel_request_reason"},
        "eval_analysis_attempt": _ADDRESSED_EVERYWHERE
        | {"campaign_id", "created_at", "cancel_requested_at", "cancel_request_reason"},
        # The store reads a slot by id alone, and the id is generation_slot_id(campaign_id); the claim is
        # attempt_id. A step that moved either would move or reassign a live claim, so neither is a step's to touch.
        "eval_generation_slot": _ADDRESSED_EVERYWHERE | {"campaign_id", "attempt_id"},
        # The four job records are read by id alone; a cancel request is merged into them in place.
        "eval_launch": _ADDRESSED_EVERYWHERE | {"cancel_requested_at", "cancel_request_reason"},
        "eval_second_judge_pass": _ADDRESSED_EVERYWHERE | {"cancel_requested_at", "cancel_request_reason"},
        "eval_judge_temperature_check": _ADDRESSED_EVERYWHERE | {"cancel_requested_at", "cancel_request_reason"},
        "eval_action_job": _ADDRESSED_EVERYWHERE | {"cancel_requested_at", "cancel_request_reason"},
    }
)


@dataclass(frozen=True)
class CoreUpgrader:
    """One step of the core's upgrade chain: a document at ``from_version`` rewritten to ``from_version + 1``.

    Attributes:
        from_version: The core version the step reads.
        doc_types: The core types it rewrites. Every other core type passes through the step unchanged but
            for its number.
        reason: What changed at the bump, in one line.
        upgrade: The rewrite: pure, and never touching an addressing field, a host or kind payload, or a
            stored key or digest (the module's step rules).
    """

    from_version: int
    doc_types: frozenset[str]
    reason: str
    upgrade: Callable[[dict[str, Any]], dict[str, Any]]


#: The registered steps, oldest first: ``[s.from_version for s in CORE_UPGRADERS]`` is
#: ``range(CORE_BASELINE_VERSION, CORE_SCHEMA_VERSION)``. A constant, so no import registers a step.
CORE_UPGRADERS: tuple[CoreUpgrader, ...] = ()


class CoreVersionRefused(ValueError):
    """A stored core document's version is one this build does not read: unreadable, newer, or pre-release."""


def upgrade_document(
    document: Mapping[str, Any], *, steps: Sequence[CoreUpgrader], current: int, baseline: int
) -> dict[str, Any]:
    """Read a stored core document at its version and return it at ``current``, through ``steps``.

    Each step from the stored version up to ``current`` is applied when its ``doc_types`` holds the document's
    ``doc_type``, and the document's ``schema_version`` is then set to the step's ``from_version + 1`` here, so a
    step never writes the number itself. A document already at ``current`` is returned as a copy, unchanged.

    Args:
        document: The stored document, as the store returned it.
        steps: The upgrade chain (:data:`CORE_UPGRADERS` in production).
        current: The version to read to (:data:`CORE_SCHEMA_VERSION`).
        baseline: The oldest version read (:data:`CORE_BASELINE_VERSION`).

    Returns:
        A new document at ``current``; the input is not modified.

    Raises:
        CoreVersionRefused: ``schema_version`` is missing or not an integer, above ``current``, below
            ``baseline``, or a step the chain needs is not registered.
    """
    version = document.get("schema_version")
    doc_type = document.get("doc_type")
    if type(version) is not int:
        raise CoreVersionRefused(
            f"this {doc_type or 'core'} document's schema_version is {version!r}, not a core version number, so "
            "this build cannot say which shape it was written in"
        )
    if version > current:
        raise CoreVersionRefused(
            f"this {doc_type} document was written under core v{version} by a newer build; this build reads core "
            f"v{baseline} to v{current}, so it refuses rather than guess what the newer fields mean"
        )
    if version < baseline:
        raise CoreVersionRefused(
            f"this {doc_type} document was written under core v{version}, before the first public release "
            f"(core v{baseline}); pre-release documents are not read"
        )
    upgraded = dict(document)
    by_version = {step.from_version: step for step in steps}
    for step_from in range(version, current):
        step = by_version.get(step_from)
        if step is None:
            raise CoreVersionRefused(f"no core upgrader is registered from v{step_from} to v{step_from + 1}")
        if doc_type in step.doc_types:
            upgraded = step.upgrade(dict(upgraded))
        upgraded["schema_version"] = step_from + 1
    return upgraded


def upgrade_core_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`upgrade_document` with this build's chain, version and baseline.

    Args:
        document: A stored core document.

    Returns:
        The document at :data:`CORE_SCHEMA_VERSION`.

    Raises:
        CoreVersionRefused: See :func:`upgrade_document`.
    """
    return upgrade_document(document, steps=CORE_UPGRADERS, current=CORE_SCHEMA_VERSION, baseline=CORE_BASELINE_VERSION)


__all__ = [
    "CORE_ADDRESSING_FIELDS",
    "CORE_BASELINE_VERSION",
    "CORE_DOC_TYPES",
    "CORE_SCHEMA_VERSION",
    "CORE_UPGRADERS",
    "REGENERABLE_SCHEMA_VERSION",
    "CoreUpgrader",
    "CoreVersionRefused",
    "upgrade_core_document",
    "upgrade_document",
]
