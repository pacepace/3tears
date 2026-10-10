"""Which stored documents are kept across releases, which are regenerated, and how a kept one is read when it is old.

**The evidence core is kept.** A document the engine cannot regenerate, and that an audit needs, is in the
core (:data:`CORE_DOC_TYPES`): the evidence a run produced (its test cases, the run, its results and traces,
human calibration ratings, the out-of-run spend ledger) and the launch inputs those documents name by id (the
template, the judge config, the rubric dim, their tombstones, and the case set). A core document is read
forever from :data:`CORE_BASELINE_VERSION` on: a read of an older core document upgrades it in memory, one
version at a time, through :data:`CORE_UPGRADERS`, and the result is validated as strictly as a document
written today. The stored bytes are not touched; the next write of that document (a rescore, an archive
toggle's read-modify-write) writes the current version, so migration is lazy and a scope may hold documents
at several core versions at once.

**Everything else is regenerable.** Campaigns, analyses and their attempts, insights, sweeps and cassettes are
derived from the core (or, for a cassette, re-captured), so they keep the strict rule they always had: a
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

#: The first core version a public release wrote, and the oldest this and every later build reads. Below
#: it a document is pre-release (``v0.66.0-evaladopt.*`` wrote v8, so nothing a released build wrote is).
CORE_BASELINE_VERSION: Final[int] = 8

#: The core version this build writes. Every bump appends one line here and one :class:`CoreUpgrader`.
#:
#: - **v8** — the baseline, the first public release's evidence core: every core document as eval schema v8
#:   stored it, including each optional field v8 gained after it was first written (the notes below list
#:   them). A field a v8 document predates reads as the note says, as it always did.
#: - **v9** — a judge score and a calibration rating carry the label key of what was read (#628):
#:   ``RubricScore.output_fingerprint`` / ``criterion_fingerprint`` (stamped by the judge) and the same pair on
#:   ``CalibrationRating`` (copied from the rated score), both OPTIONAL. Additive: the v8 step is the identity,
#:   and a v8 document reads with neither — a score judged before the stamp, and a rating read by its result alone.
CORE_SCHEMA_VERSION: int = 9
"""The core version this build writes, and the newest a read of a core document accepts.

How each value a core document stored at v8 predates reads — a field v8 gained after the document was
written reads as the note says, never as today's default:

**Within v8**: ``RubricScore.axis`` joined as an OPTIONAL field — the rubric axis the judge
stamped from the dimension's definition, so a boundary (guardrail) score stays out of the composite and
pass^k. A score judged before it carries None, and is read as capability, which is how it was read then:
its result's composite does not move, and the bundle names the dimensions read that way
(``GuardrailReadings.unstamped_dimensions``) rather than presenting them as known capability.

**Within v8**: ``EvalResult.turns_delivered`` joined as an OPTIONAL field — how many turns the
candidate delivered, which decides whether a model failure's time and spend are a turn's. A result written
before it carries none and still means what it says; it reads as None, "nothing counted", and every reader
falls back to the failure's cause alone (``delivered_a_turn``). ``EvalRun.goal_check_proofs`` joined the same
way: whether each goal check was shown, at launch, to beat doing nothing; None on a run launched before it, read
as unproven.

**Within v8**: ``RubricDimTombstone`` joined as a new stored type — the record a rubric dim delete
leaves so the definition seed does not write the key back. A store written before it holds none, which reads as
"no key was deleted since": a dim deleted before then is still written back at the next seed, as it was then.

**Within v8**: ``RoleUsage.served_model`` joined as an OPTIONAL field — the model the provider's
response named as having answered the row's calls, which for a candidate launched on a floating alias is the
only record of which model produced its numbers. A row stored before it carries None and reads as "not
recorded", never as the alias in ``model``: the analysis names such an arm's served model unknown rather than
the one requested.

**Within v8**: the judge's temperature joined as OPTIONAL fields (#633) — ``RubricScore.judge_temperature``
(what the call was sent at), ``EvalRun.judge_temperature`` (what a dimension with no config was requested at) and
``RepeatedScore.first_judge_temperature``. A document stored before them carries None and reads as not recorded:
its unconfigured dimensions were requested at the provider's default, which is not today's 0, so such a run's
roles component is not composable, its scores' judge reads unknown, and nothing pools it with a run judged at 0.

**Within v8**: ``EvalRun.measure_latency`` and ``EvalRun.cell_concurrency`` joined as OPTIONAL fields
(#701) — whether the launch declared latency under test, and how many of the run's cells executed at once. A run
stored before them carries None in both: its cells executed one at a time (the runner of that build had no other
way), so its ``cell_concurrency`` reads as 1, and whether it declared latency reads as not recorded — never as
declared. Whether another RUN executed beside it is what its results' ``execution_mode`` says, as it always was.

``JudgeConfigTombstone`` joined the same way, for a judge config's slot; a config deleted before it is written
back at the next seed. ``EvalRun`` gained ``goal_check_proof_rules`` (None on a run stored before it, read as rules
1, so its ``proven`` checks read unproven) and ``refused_goal_checks`` (None, not recorded), and ``EvalResult``
gained ``judge_cannot_tell_boundary`` (empty, its can't-tells read as capability) — all optional within v8.

**Within v8**: ``ClientRequestSettings.strict_output`` joined as a defaulted field (#686) — whether a
role's requests were to be routed only to providers honouring every parameter sent. A stamp stored before it
carries none and reads False, "no such requirement was stated", which is what the engine sent then. Its apparatus
level (``judge_request_settings`` / ``simulator_request_settings``) leaves the flag out while it is False, so a
stored run's level is unchanged; a judge stamp carrying True reads as a different level from one stored before.

**Within v8**: ``EvalResult.judge_seconds`` joined as an OPTIONAL field (#646, #597) — a second judge's
scores of the result's stored evidence (:class:`SecondJudging`), each beside the first score it pairs with. A result
stored before it carries none and reads as "no second judge was asked", which is what it means: no agreement and no
drift is read from it, never a zero.

**Within v8**: ``EvalRun.declared_margins`` joined as an OPTIONAL field (#698) — the margins a launch
declared on core rate measures (accuracy). A run stored before it carries none and reads as declaring none, so no
comparison over it reads a margin it never declared.

**Within v8**: ``EvalRun.declared_measures`` joined as an OPTIONAL field — how the launching host
declared each of its own measures to be read (direction, merit axis, guardrail, margin, range). A run stored before
it carries none, and a campaign of such runs is read on the reading host's declarations, as every campaign was.

**Within v8**: ``CaseSet`` joined as a new stored type and ``EvalRun.case_set`` as an OPTIONAL field
(#676). A store written before them holds no sets, and a run stored before carries None: it was launched over its
template's cases, which is what None says, and history epochs it by its frozen ids as before.

**Within v8**: ``ConversationSpec.world_rounds`` joined as an OPTIONAL field, and ``actors`` may be empty
when world rounds supply every round (#578). A template stored before it carries none, and reads as it did: every
round is the actors'.

**Within v8**: ``EvalRun.cell_timeout_s`` and ``cell_timeout_s_origin`` joined as OPTIONAL fields (#649)
— the per-cell deadline the run's cells ran under and whether the launch, the kind or the engine's default set it.
A run stored before them carries None for both and reads as "deadline not recorded", never as today's default:
the kind's wiring may have set another.
"""

#: The version every regenerable document is written under, and the only one a read of one accepts.
#:
#: - **v6** — one opaque ``scope_id`` on every stored document; kind-owned overlays, specs, judge evidence and
#:   payloads; every field its writers set required; one ``candidate_model`` per run.
#: - **v7** — a calibration rating records the kind of rater that wrote it.
#: - **v8** — judged readings carry a code-decided evidence tier; ``directional`` left ``EvidenceTier``; a
#:   repeated judge score records the config that asked for the score it repeats.
#:
#: Within v8, regenerable shapes also gained optional fields and retired others (``__retired_fields__``); the
#: notes below say how each stored value reads.
REGENERABLE_SCHEMA_VERSION: int = 8
"""The version every regenerable document is written under, and the only one a read of one accepts.

A bump is a drop: regenerable documents written under the old version are refused on read
(:data:`~threetears.evals.schema.models.SchemaVersion`) and regenerated from the core, never migrated.
Bump it when a regenerable shape changes so that a document written before the change would not mean what
it says after it — a field renamed, retyped, removed or made required. Until v8 this was the one version
every stored document carried; the core's history before its baseline is here.

**v6** is the scope-and-kind contract: one opaque ``scope_id`` on every stored document, kind-owned launch
overlays and template ``kind_spec``, kind-rendered judge evidence, kind-owned result payloads
(``async_deliveries``, ``kind_payload``, ``candidate_instance_id``), and every field its writers
set required — no field is read as "absent because older" — and a run naming one
``candidate_model`` (with its one ``variant_levers`` map) where it carried a list. Nothing written
before it loads. (The required fields and the one-model run joined v6 before it was first released,
so they share its number; the one stored value the required fields change is
``GenerationProvenance.cell_model_version``, which the generator had never written. So did the
cassette corpus — ``EvalCassette`` keyed by corpus and occurrence, ``EvalRun.cassette_corpus_id`` in
place of ``cassette_version`` on the run and the result — and the background-work spend
``AsyncDelivery`` carries; and ``WorldEvent.event``, the identity of the event a firing names, required
on every firing so a firing's ``armed`` is the event's provenance rather than the dimension's.)

**v7**: a calibration rating records who KIND of rater wrote it (``CalibrationRating.rater_kind``, a person or
an agent), required, so an agent's rating is never read as a person's. A rating written before it says
nothing about which it was, so nothing written under v6 loads.

**v8**: judged readings carry a code-decided evidence tier (PD-13). A stored analysis's judged evidence rows
(``EvidenceRow.judged_tier``) and its decision surface's judged readings (``JudgedReading.evidence_tier``)
carry a tier, required; the finding tier ``directional`` is gone from ``EvidenceTier``; a repeated judge score
records the config that asked for the score it repeats (``RepeatedScore.first_judge_config_id``, required),
so a repeat under one judge prompt never measures another. A v7 analysis holding a judged reading cannot say
what tier it stood on, so nothing written under v7 loads.

**Within v8, not a bump**: the decision surface's ``CellFacts.n_candidate_failed`` and ``n_no_turn`` (and their
``StratumFacts`` twins) joined as OPTIONAL fields, None on an analysis frozen before them: requiring them would drop
every stored document to learn counts the old ones never had, and their honest reading is "unknown", which None
states. ``EvalAnalysis.judged_tier_rule`` joined the same way: the rule its judged tiers were decided by, None on an
analysis stored before tiers were decided on the agreement's interval — whose tiers were the point estimate against
the bar, and are rendered as that, never as the interval rule's claim. And ``GenerationProvenance.bundle_schema_version``
and ``.host_declarations_digest``: the bundle shape and the host's declarations a generation ran over, None on an
analysis stored before them, read as "cannot say" — never as the current version.

**Within v8, not a bump: fields retired** (``__retired_fields__``, read only by a stored read — see
:mod:`threetears.evals.schema.base`). ``LeverCoverage.confidence`` is removed: it was a fixed lookup on the
lever's ``status``, so a stored analysis loses nothing when the key is discarded on read. ``EvalCampaign.status``
(open / closed) is removed: nothing could change it after creation and nothing enforced it, so a stored
campaign's ``closed`` froze nothing and discarding it changes no membership and no analysis; the one thing it
fed, ``list_campaigns``'s ``status`` filter, is gone with it. ``CampaignDesign.controls`` is renamed ``held_fixed``
(one letter from ``control``, it named a different thing), and the bundle's ``controls_reading`` with it
(``held_fixed_reading``): a stored campaign, an analysis's ``design_snapshot`` and a reporter case's frozen bundle
read the old key under the new name, value unchanged.

**Within v8, not a bump**: ``EvidenceRow.n_cases`` joined as an OPTIONAL field — the distinct test cases behind an
evidence row's value, which a report's evidence table shows (``n`` counts observations, a case judged k times
counting k). A row stored before it carries None and reads as not recorded: its table cell is empty, never its
observation count under the cases heading.

**Within v8, not a bump**: ``CampaignDesign.measure_latency`` joined the same way, defaulting to False: a campaign (or an analysis's design
snapshot) stored before it reads as not declaring latency under test, which is what it declared — a stored
design asking about latency still loads, and is refused only when it is declared again.

**Within v8, not a bump**: ``CampaignDesign.guardrail_margins`` joined as an OPTIONAL field (#697) — the margin
each judged guardrail (a boundary rubric dimension) is held to. A campaign, or an analysis's design snapshot,
stored before it carries none and reads as declaring none: its judged guardrails are held at zero change, exactly
as they were decided then, so no stored decision moves.

**Within v8, not a bump**: ``CampaignDesign.crossing`` and ``CampaignDesign.skipped_cells`` joined as OPTIONAL
fields (#654) — which combinations of the declared levels a design meant to run. A campaign, or an analysis's
design snapshot, stored before them carries None and an empty list, and reads as declaring nothing about
combinations: no cell is read as skipped by design or as missing, exactly as before.

**Within v8, not a bump**: ``DecisionSurface.frontier_disqualified`` joined as an OPTIONAL field (#613) — the arms the
frontier disqualified on its boundary pillar, each with the guardrail dimensions it breached. A surface frozen before
it carries None and reads as "not recorded": its frontier chart marks no arm disqualified, as it did when frozen,
because the frontier then disqualified none.

**Within v8, not a bump**: ``MeasureSummary.case_means`` and ``JudgedReading.case_means`` joined as OPTIONAL fields
(#677) — each case's mean, recorded only below 5 cases, where a chart draws the cases as points instead of an
interval band. An analysis stored before them carries None and reads as not recorded: its small cells draw no band
and no points, and the chart is refused with that reason rather than drawn from the interval.

**Within v8, not a bump**: ``JudgedReading.prediction_powered`` joined as an OPTIONAL field (#598) — the judged mean
re-estimated with people's calibration ratings (prediction-powered inference), beside the judge's own. An analysis
stored before it carries None and reads as not recorded: its judged readings state the judge's mean alone, as they
did when frozen.

**Within v8, not a bump**: ``EvalSweep`` joined as a new stored type (#632) — the record of a multi-arm launch
run arm after arm. A store written before it holds none, which reads as "no sweep was started".
"""

#: The kept documents, by ``doc_type``. Embedded shapes (a result's rubric scores, usage rows and goal-state
#: outcomes, a run's context components and subject snapshot, a ledger's calls) are core through their parent.
CORE_DOC_TYPES: Final[frozenset[str]] = frozenset(
    {
        "eval_test_case",
        "eval_run",
        "eval_result",
        "eval_trace",
        "calibration_rating",
        "eval_out_of_run_spend",
        "eval_template",
        "judge_config",
        "rubric_dim",
        "rubric_dim_tombstone",
        "judge_config_tombstone",
        "case_set",
    }
)

#: The fields every core document is addressed by in the store.
_ADDRESSED_EVERYWHERE = frozenset({"id", "doc_type", "schema_version", "scope_id"})

#: Per core type, every field the store reads on the stored bytes, before any upgrade: what a query filters or
#: orders on, what a projection keeps, and what a partial write merges. An upgrader may not touch one, since
#: an old document's value is what the store matches; renaming one needs a decision and a rewrite of the store.
CORE_ADDRESSING_FIELDS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "eval_test_case": _ADDRESSED_EVERYWHERE | {"template_id", "stratum"},
        "eval_run": _ADDRESSED_EVERYWHERE | {"status", "archived", "created_at"},
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
CORE_UPGRADERS: tuple[CoreUpgrader, ...] = (
    CoreUpgrader(
        from_version=8,
        doc_types=frozenset({"eval_result", "calibration_rating"}),
        reason=(
            "v9 adds the optional label key (output_fingerprint, criterion_fingerprint) to a judge score and a "
            "calibration rating; a v8 document has neither, which v9 reads as not stamped"
        ),
        upgrade=lambda document: document,
    ),
)


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
