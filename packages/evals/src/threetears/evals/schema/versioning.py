"""Which stored documents are kept across releases, which are regenerated, and how a kept one is read, old or new.

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

**Everything else is regenerable.** Analyses, insights and judge profiles are derived from the core, so they keep
the strict rule they always had: a document written under any :data:`REGENERABLE_SCHEMA_VERSION` but this build's is
refused on read, and the operator regenerates it.

**A cassette has a version space of its own.** It is a regenerable record (what a tool answered while a capture
ran), refused at any :data:`CASSETTE_SCHEMA_VERSION` but this build's and replaced by a fresh capture under a new
corpus. Its number moves only when the cassette's own format changes, never with the analysis shapes: an analysis is
cheap to regenerate from the kept runs, but a corpus is re-captured by calling the third-party tools it exists to
spare, so a new chart type on a finding must not refuse every corpus a host recorded.

The bundle (``schema_version``), the report (``REPORT_VERSION``), the cell model and the identity keys
(``IDENTITY_VERSION``) are separate version spaces with their own pins too; an upgrader never recomputes a key or a
digest. A kept key is stored beside its pre-image and the identity version that computed
it — a run's ``variant_levers`` beside the keys its results carry, a campaign control's ``control_levers`` and
``control_identity_version`` beside ``control`` — so no upgrader recomputes it. A read under a later
``IDENTITY_VERSION`` names the control's arm by its stored key, which the cells stamped under its own predicate keep,
and by the key its map re-keys to (``resolve_declared_control``). That follows a change to how a map is DIGESTED
only: a host renaming a lever, a kind or an overlay field, or the engine adding a lever to every map, gives new runs
a different map, which no recomputation of the stored one reaches, so across such a change the control names only
the cells stamped before it and the map keeps it described.

**A judge's identity is a version space that is a fingerprint, not a counter.** A result records, per dim, what
asked it (``EvalResult.judge_config_ids``): ``builtin:<sha12>``, a digest of the engine's own judge prompt as sent
(:func:`~threetears.evals.schema.models.builtin_judge_config_id`), or ``<config id>@<sha12>``, a configured judge's
config and a digest of the reply-format suffix the engine appends to its prompt
(:func:`~threetears.evals.schema.models.configured_judge_config_id`). Nothing is bumped and nothing stored is
refused when either wording changes; the fingerprint moves, and every result judged before the change is another
judge to everything that keys a judge by it — agreement, judge profiles, drift and evidence tiers — so a scope's
``calibrated`` tier reads ``undetermined`` until the new judge is calibrated. Two wordings are two instruments, so
that is the design; the policy is that it never ships unannounced. A release that rewords a built-in prompt or the
suffix is breaking, and its ``MIGRATIONS.toml`` notes carry
:data:`~threetears.evals.schema.JUDGE_WORDING_CHANGE_SENTENCE` verbatim, naming the wording that moved;
``tests/test_judge_identity_pins.py`` pins every fingerprint by value, as the tree computes it and as the newest
final release shipped it, and fails while one moved since that release and the notes do not carry the sentence.
**The run is not re-keyed by it.** ``EvalRun.judge_config_ids``, the config set a run committed to and a component
of its context key, holds the bare config id: the launch writes ``judge.config_ids`` (``run/launch.py``), and
``RunJudge.config_ids`` is ``{dim_id: config.id for dim_id, config in self.configs.items()}``, with no key for a dim
the built-in prompt asks. So runs either side of a wording change share their context, and only their results'
recorded judges tell the two judges apart.

**A core version is a major and a minor.** ``schema_version`` is the major (:data:`CORE_SCHEMA_VERSION`) and
``schema_minor`` the minor within it (:data:`CORE_SCHEMA_MINOR`); the store addresses documents by the major
alone, so the minor is not one of :data:`CORE_ADDRESSING_FIELDS`. A **minor** is additive and nothing else: a new
optional field with a default, a new optional sub-object, a new member of an open vocabulary (below), a new doc
type. It needs no upgrader. A **major** is everything else — a rename, a removal, a type change, a new required
field, a new member of a closed vocabulary, any change to what a stored value means — and ships an upgrader.

**How a core document is read.** By its stored read, :meth:`~threetears.evals.schema.base.CoreDocumentModel.from_dict`:

- *Its major is this build's and its minor this build's or older:* read strictly. A field this build does not
  declare, at any level, is refused, and so is a value outside a vocabulary this build knows: a writer at this
  minor cannot have written either, so either is drift, and drift is refused by name.
- *Its major is this build's and its minor newer:* read. Every field this build does not declare — at the top
  level and inside every shape the document embeds — is kept on the instance it was found on, opaquely: never
  validated, never an attribute, never in ``model_dump`` or in any digest, and written back exactly as read. A
  value of an open vocabulary this build does not know is read, and every reader that branches on that
  vocabulary has a branch for it that names it rather than folding it into a value it knows.
  The tolerance has a floor, the build that wrote the v15 baseline (``v0.67.0-evaladopt.25``): every build
  before it writes and reads core v14 or older, so it refuses every v15 document as written by a newer build,
  and that build refuses theirs as pre-release. A fleet rolls back no further than ``.25``.
- *Its major is older:* upgraded through :data:`CORE_UPGRADERS` to this build's shape at this build's minor; a
  field the chain left behind is refused.
- *Its major is newer, below* :data:`CORE_BASELINE_VERSION`*, or not an integer, or its minor is not a whole
  number:* refused, by name, never guessed.

Only the stored read is tolerant. Constructing a core document, or validating a payload, at a minor newer
than this build's is refused, so a document cannot be rebuilt from a read one and silently lose what a newer
build wrote: it is copied (``model_copy``), or read again from its ``to_dict`` (the read-modify-write every
writer here does).

**How a core document is written.** ``to_dict`` writes every field this build declares, puts back every field a
newer minor's read kept, where it was found, and stamps ``schema_minor`` with the newest minor any shape it
carries was read at (a shape read at a newer minor carries that minor whether or not it held a field this build
does not know, so a value of an open vocabulary moves with its minor too), or this build's when that is newer. A
write therefore keeps every field a newer build wrote on each shape it carries over, and never claims a minor older
than anything it carries. A shape the write REPLACES with one built afresh — a renewed lease, a recomputed
completeness record, a re-scored rubric list — takes its kept fields with it, which is right: they described the
shape that was replaced, and the additivity sentence below is what makes the fields beside it stay true. Two
rules keep that true on the paths that write less than a whole document, and on the digests:

- **A partial write carries baseline fields only.** ``merge_fields`` sets top-level fields on the stored bytes
  and cannot take the newest of a minor it never read, so it names only fields of its major's first minor
  (today, addressing fields). A field a minor introduces is written by a read-modify-write.
- **A field a minor adds stays out of every digest pre-image while it holds its default.** A pre-image taken
  over a whole shape's ``model_dump`` (a judged output's fingerprint, a judge case's digest) would otherwise
  split every key between the two builds, even for documents that never use the field.

**A vocabulary on a core shape is open or closed, and which is decided here.** An *open* vocabulary is a ``str``
with the values this build knows registered beside it; a new value is a minor, because every reader that branches
on it has a branch for a value it does not know, which names that value (a role's spend on a line of its own, an
origin stated by name, a departure read as "cannot say") rather than folding it into one it knows. A value outside
the registry is read only from a newer minor's document, and refused anywhere else. Only a STORED field is open:
each vocabulary also has a ``Literal`` of the values this build knows (``KnownUsageRole``, …), which every
caller-facing parameter (an action's filter, a query's) and every writer here takes, so a caller's misspelt value
is refused as before, and the registry is that ``Literal``'s values. The open vocabularies:
``UsageRole`` (``USAGE_ROLES``), ``SimulatorPurpose`` (``SIMULATOR_PURPOSES``), ``CostCapOrigin``
(``COST_CAP_ORIGINS``), ``MeteredCallOrigin`` (``METERED_CALL_ORIGINS``), ``CellTimeoutOrigin``
(``CELL_TIMEOUT_ORIGINS``), ``ModelRoleOrigin`` (``MODEL_ROLE_ORIGINS``), ``RoleModelOrigin``
(``ROLE_MODEL_ORIGINS``), ``ReasoningEffort`` (``REASONING_EFFORTS``), ``OutOfRunPurpose``
(``OUT_OF_RUN_PURPOSES``) and ``RefusalKind`` (``REFUSAL_KINDS``).

Every other vocabulary on a core shape is *closed by design*: a new value changes how an older build must read
the document, so extending one is a major. Each, and why:

- ``EvalRunStatus``, ``JobOutcome`` (and ``SweepOutcome``), ``AttemptOutcome`` — lease settling, the boot reclaim and
  every terminal test branch on them; a state misread strands a job or resurrects one.
- ``CellTermination`` — decides whether a cell is excluded from every figure.
- ``ConversationStopCause`` — its members are split into a conversation ending on its own terms and a rig fault,
  which excludes the cell.
- ``GoalCheckProof``, ``GoalCheckIntent`` — what a goal check proves, and how it is checked.
- ``ApparatusProvenance`` — witnessed or commissioned decides which levers read as moved; cells never pool across it.
- ``RubricScale``, ``RubricAxis``, the scale ``kind`` tags (``nominal``, ``ordinal``, ``interval``) and the
  ``model_default`` temperature — how a score or a level is read; a new scale tag is a new model.
- ``JudgedArtifact`` — which axes a cell is judged on.
- ``WorldPlacement``, ``WorldEventKind``, ``WorldEventCause`` — what a transcript or a world means to the judge.
- ``AsyncDeliveryStatus`` — what a delivered count counts.
- ``CassetteMode`` — replay semantics, and identity.
- ``TemperatureSelection``, ``TemperatureSetting``, ``CompletenessSource``, ``JudgeAttributionSource`` — facts of
  method that a reader trusts or re-derives by.
- ``DeclaredMeritAxis``, ``GuardrailDirection``, ``CampaignDesign.crossing``, ``ControlDeclaration.stimulus`` — the
  semantics of an analysis design.
- ``VariationAxis.generator``, ``ConversationSpec.turn_scheduler`` — instructions an older build must carry out.
- ``RaterKind`` — calibration reads a person as ground truth.
- ``OutOfRunOutcome`` — binary and complete.
- ``SecondJudgeAskedBy`` — an unknown asker's pass would be pooled into the wrong agreement.
- ``StopReason`` — the normalized provider contract, which already has a catch-all for output cut short by a
  cause it does not name (``truncated``), kept apart from a call that failed (``error``).
- Every ``doc_type`` — a new type is a new model, never a new value.

**A minor's ledger entry asserts its additivity in words**, and review holds it there, because a schema diff
cannot see a field that is additive in shape and breaking in meaning (a ``faults`` field meaning "exclude this
cell"). Every minor's entry ends with :data:`MINOR_ADDITIVITY_SENTENCE`, verbatim; when that sentence is false,
the change is a major. A newly computed field is safe under it: a newer build discards the stored echo and
re-derives it.

**A new doc type is a minor, with two gaps an older build cannot close.** An older build never reads the new
type (every read selects by ``doc_type``), but its operator wipe sweeps only the types it knows, so it leaves the
new type's documents in the store; and its deletes (``run_delete`` and the cascades it runs) cannot cascade to a
type it does not know, so a delete on an older build can leave the new type's documents orphaned for the
upgraded build to find. Both are disclosed rather than prevented; an operator wiping or deleting during a roll
does it from the newest build.

**How a core change ships.** Every change to a core shape re-pins ``tests/test_core_schema_pins.py``. A minor
bumps :data:`CORE_SCHEMA_MINOR` with a ledger line carrying the sentence above; no fixture is copied and no
step is written. A major copies the frozen fixtures under ``tests/fixtures/core_documents/`` to the next major's
directory before changing a shape, bumps :data:`CORE_SCHEMA_VERSION` with a ledger line, resets
:data:`CORE_SCHEMA_MINOR` to 0, and appends one :class:`CoreUpgrader` whose ``from_version`` is the old major. A
step reads every minor of its ``from_version`` up to the newest one released, and obeys four rules: it is pure
(no store, host, clock or randomness); it never touches :data:`CORE_ADDRESSING_FIELDS`, ``host_payload``,
``kind_payload`` or a stored key or digest (``variant_key``, ``context_key``, ``context_components``,
``content_hash``, ``identity_version``), because the store filters, sorts and projects on the stored bytes
before any upgrade runs; a value it cannot derive becomes None, read as "not recorded"; and its output must
validate strictly.

**Hook points this release leaves for later ones**, none of which an older build needs: a mechanical gate that a
minor's schema diff from the pinned shape adds only non-required properties (in ``tests/test_core_schema_pins.py``,
beside the digests it would diff); a write-version pin for rolling a major, which writes the older major through a
downgrader at :meth:`~threetears.evals.schema.base.CoreDocumentModel.to_dict`, where the minor is stamped; and
per-type minors, which turn :data:`CORE_SCHEMA_MINOR` into a mapping by ``doc_type`` so a change to one embedded
shape does not restamp every type.

The document field stays ``schema_version`` in all three spaces; ``doc_type`` says which number it is.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from threetears.evals.schema.doc_types import DOC_TYPE_FACTS

#: The first core version a public release wrote, and the oldest this and every later build reads. Below it a
#: document is pre-release, and is refused by name.
CORE_BASELINE_VERSION: Final[int] = 15

#: The core version this build writes. Every bump appends one line here and one :class:`CoreUpgrader`.
#:
#: - **v15** — the baseline: the first public release's evidence core. Every field a writer always sets is
#:   required; an optional field's ``None`` has one present-day meaning, stated on the field. What a request was
#:   asked for and what it was sent with are two facts under two names: a run's ``judge_asked_settings`` and
#:   ``simulator_asked_settings``, and each call's ``RubricScore.judge_sent_settings``, ``RoleUsage.sent_settings``,
#:   the first score's half of a repeated or second-judge pair and ``OutOfRunSpend.sent_settings``. It replaced the
#:   pre-release v14 baseline in ``v0.67.0-evaladopt.25`` with no upgrader: a v14 document is refused by name as
#:   pre-release, and the rollback floor is ``.25``.
CORE_SCHEMA_VERSION: int = 15
"""The core major this build writes, and the newest a read of a core document accepts."""

#: The core minor this build writes within :data:`CORE_SCHEMA_VERSION`. A minor is additive only (the module's
#: rule), so a read accepts any minor of this build's major. Every minor appends one line here, ending with
#: :data:`MINOR_ADDITIVITY_SENTENCE`; a major resets it to 0.
#:
#: - **v15.0** — the baseline minor.
CORE_SCHEMA_MINOR: int = 0
"""The core minor this build writes; a read of this build's major at a newer minor keeps what it does not know."""

#: The sentence every minor's ledger entry ends with, verbatim: what makes trusting a newer minor legitimate. When
#: it is false for a change, the change is a major.
MINOR_ADDITIVITY_SENTENCE: Final[str] = (
    "An older build that ignores this field reads the document correctly, and its writes leave it true."
)

#: The version every regenerable document is written under, the cassette apart (:data:`CASSETTE_SCHEMA_VERSION`),
#: and the only one a read of one accepts.
#:
#: - **v9** — the pre-release baseline (``.21`` to ``.24``); no release reads it.
#: - **v10** — the first public release's regenerable shapes (from ``v0.67.0-evaladopt.25``): a judge's identity
#:   includes the request settings its calls were sent with, so a stored judge profile carries
#:   ``judge_sent_settings`` and its id derives from it, and an analysis's evidence tiers, agreements and bundle carry
#:   it on every judge they name. Profiles and analyses regenerate from the core; nothing is upgraded.
REGENERABLE_SCHEMA_VERSION: int = 10
"""The version every regenerable document but the cassette is written under, and the only one a read of one accepts.

A bump is a drop: regenerable documents written under the old version are refused on read
(:data:`~threetears.evals.schema.models.SchemaVersion`) and regenerated from the core, never migrated. It governs
the analysis, the insight and the judge profile, never the cassette (:data:`CASSETTE_SCHEMA_VERSION`).
"""

#: The version every cassette is written under, and the only one a read of one accepts. It moves only when the
#: cassette's own format changes (the fields of :class:`~threetears.evals.schema.models.EvalCassette`), never with
#: :data:`REGENERABLE_SCHEMA_VERSION`; a recorded payload that no longer rebuilds as its tool's type is the replay
#: seam's refusal, not a version.
#:
#: - **v1** — the baseline: the first public release's cassette format.
CASSETTE_SCHEMA_VERSION: int = 1
"""The version every cassette is written under, and the only one a read of one accepts.

A bump is a drop: a cassette written under the old version is refused on read
(:data:`~threetears.evals.schema.models.CassetteSchemaVersion`), and the corpus is re-captured (a new capture run,
so a new corpus), never migrated. Bump it only for a change to the cassette's own format, since every bump spends
the third-party quota of every corpus again.
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


def read_core_minor(document: Mapping[str, Any]) -> int:
    """The minor a stored core document was written at: ``0`` when it carries none, refused when it is not a number.

    A document stored before a minor existed lacks the key, and reads as its major's first minor, as any optional
    field's absence reads as its default.

    Args:
        document: The stored document, as the store returned it.

    Returns:
        Its ``schema_minor``.

    Raises:
        CoreVersionRefused: ``schema_minor`` is present and is not a whole number of zero or more.
    """
    if "schema_minor" not in document:
        return 0
    minor = document["schema_minor"]
    if type(minor) is not int or minor < 0:
        raise CoreVersionRefused(
            f"this {document.get('doc_type') or 'core'} document's schema_minor is {minor!r}, not a core minor "
            "number, so this build cannot say whether it knows every field the document holds"
        )
    return minor


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
    "CASSETTE_SCHEMA_VERSION",
    "CORE_ADDRESSING_FIELDS",
    "CORE_BASELINE_VERSION",
    "CORE_DOC_TYPES",
    "CORE_SCHEMA_MINOR",
    "CORE_SCHEMA_VERSION",
    "CORE_UPGRADERS",
    "MINOR_ADDITIVITY_SENTENCE",
    "REGENERABLE_SCHEMA_VERSION",
    "CoreUpgrader",
    "CoreVersionRefused",
    "read_core_minor",
    "upgrade_core_document",
    "upgrade_document",
]
