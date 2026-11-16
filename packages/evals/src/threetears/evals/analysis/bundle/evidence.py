"""What a bundle was assembled from, as a manifest of digests: the pre-image of its fingerprint (version 3, #1062).

A bundle's fingerprint answers one question, "is this the same evidence?", for a stored analysis, a reporter case and
the inspection's ``mismatch_cause``. Versions 1 and 2 hashed the bundle itself, so they hashed every statistic the
package computes from the evidence: a more accurate formula, another platform's maths library or a reordered sum read
as the evidence moving, and what was stored against the old fingerprint was regenerated at a model's price. Version 3
hashes **what the assembly read, and nothing it computed**.

**The rule: anything that can change a verdict or a printed number is evidence.** Leaving evidence out is the
dangerous direction: a stored report then reads as current after what it rests on changed. So a stored field is left
out only with a stated reason, and every field of every stored record read here is decided one way or the other
(:func:`_census`, checked when this module is imported: a field added to a record later fails the import until it
is named as evidence or as not, so none is taken or dropped by default).

The components:

- ``declaration``: the campaign and aspect asked about, and what the aspect declares: its question, design (each
  level by its content hash; its bars, guardrail margins and merit order), control, margin, floors, and when each
  was declared.
- ``criteria``: what a verdict is decided against that the host, not the campaign, declares: the pass threshold for
  the behavior, each bar the host registers for it (measure, threshold, direction), and each measure the bundle
  catalogues as it is declared (direction, range, materiality threshold, merit axis, guardrail, unit, population and
  the rest of :data:`_MEASURE`). The campaign's own criteria (its margin, floors and declared bars) are in
  ``declaration``, and each run's launch declarations in ``runs``.
- ``runs``: each member run the assembly read, as stored (:data:`_RUN`): its content-addressed keys and what they
  digest, its configuration, its judges, its caps and budgets, how it ended and what it delivered. Beside them, the
  member runs archived and the ones that did not resolve, by id.
- ``results``: each result the assembly read, as the readers read it
  (:class:`~threetears.evals.kernel.result_reading.ResultReading`, :data:`_RESULT`), in id order.
- ``cases``: each case the results name that the store holds, with the stratum it is filed under.
- ``ratings``: each person's or model's rating of a judged output the assembly read (:data:`_RATING`).
- ``judge_profiles``: each stored judge profile a judge's tier consulted: its id, when it was measured and recorded,
  the cases and the runs that measured it. Which profiles are consulted is the tier rule's choice, so a change to
  that rule can move this component with no store write (accepted: the rule decides a printed tier).
- ``insights``: every prior insight of the subject the assembly read, as of its cutoff (:data:`_INSIGHT`), before
  the cap on how many a bundle carries, and each one retracted, by id.
- ``host_declarations``: :func:`~threetears.evals.analysis.bundle.schema.host_declarations_digest`, as the bundle
  carries it.

**Out, each for a stated reason.** Every computed statistic (a mean, an interval, a p-value, an effect size, a
verdict), so changing how one is computed moves no fingerprint. Every name and label (a campaign's, an aspect's, an
arm's, a lever's, a level's, a measure's, a test case's, a subject's): names are presentation, never evidence (owner
ruling, 2026-11-11). Pricing (owner rulings, 2026-10-28 and 2026-11-14, #1011): where a cost came from
(``price_source``) and who served it (``served_provider``); the cost a result recorded is evidence. Prose a package
or a host writes about a value (a description, a rationale, a rater's reason). The store's bookkeeping, the bundle's
``schema_version`` and the build (``package_version``).

**Records are built from named fields, never taken whole** (:func:`_take`). A record's shape names each field that
is evidence, at every depth, so a field added to a stored record later is in no fingerprint until it is named, and
adding one, whatever its default, moves no stored fingerprint. A named field that is unset (null, or an empty list
or map) is left out, so naming a new field whose default is one of those moves nothing either while it is unset: the
engine reads a stored null and a field never written as one value (both hydrate to ``None``), so they are one here.
A field whose default is anything else (``False``, ``0``, a word) is named only together with the stored-shape step
that says what earlier documents hold. Inside a free-form map a host or a kind writes (an overlay, a kind's spec, a
world seed, a host measure) every key is taken as stored, a null kept: there whether a key was written can carry
meaning, and no package change adds a key to it. The declaration is the one record taken as its
stored document: it is rendered without any field that holds its declared default, which is the same guarantee by
another route (:func:`declaration_evidence`).

**How it is rendered.** Through :func:`~threetears.evals.schema.hashing.input_json`: keys sorted at every depth, so
a store that reorders them (``jsonb``) moves nothing; numbers as that function states, which reads no maths library
and rounds nothing, since a stored value parses to the same double on every platform. Lists whose order a store does
not promise (the runs, the results, the cases, the ratings, the judge profiles, the insights) are sorted by id here.

**What a fingerprint proves.** It identifies the evidence. It says nothing about the bundle's body: a frozen bundle
whose statistics were altered, or computed by another build, carries the same fingerprint as long as its manifest
does. The manifest is a few hundred bytes, one digest per component and a count where one helps a reader, and rides
on the bundle (:attr:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.evidence`); comparing two
manifests names which component moved.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import BaseModel, Field

from threetears.evals.kernel.campaign import EvalInsight
from threetears.evals.kernel.declaration import (
    ARM_PRESENTATION_FIELDS,
    AXIS_PRESENTATION_FIELDS,
    LEVEL_PRESENTATION_FIELDS,
)
from threetears.evals.kernel.host.bars import Bar
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.kernel.result_reading import (
    GoalReading,
    LatencyReading,
    PreconditionReading,
    RepeatedScoreReading,
    RepeatReading,
    RescoreReading,
    ResultReading,
    ResultView,
    ScoreReading,
    SecondJudgeReading,
    SecondJudgingReading,
    SecondScoreReading,
    UsageReading,
)
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import input_digest, input_json
from threetears.evals.schema.job_lease import JobLoss
from threetears.evals.schema.models import (
    AsyncDelivery,
    AsyncExternalSpend,
    BudgetStop,
    CalibrationRating,
    CaseSetRef,
    ClientRequestSettings,
    ContextComponents,
    EvalCaseStratum,
    EvalRun,
    JudgeCutShort,
    MeasureAttestation,
    ServedModelAttestation,
    MeasureDeclaration,
    MeasureRemeasure,
    RunCompleteness,
    RunDisclosure,
    VariationCounts,
)
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.schema.values import SweepableValue

if TYPE_CHECKING:
    from threetears.evals.analysis.aspects import AspectView
    from threetears.evals.kernel.host.profile import HostProfile
    from threetears.evals.schema.models import EvalResult

__all__ = [
    "EVIDENCE_FINGERPRINT_PREFIX",
    "EvidenceManifest",
    "ResultsDigest",
    "campaign_fingerprint",
    "evidence_manifest",
    "manifest_fingerprint",
    "unnamed_axes",
    "unnamed_levels",
    "unpresented",
]

#: What a bundle fingerprint over inputs starts with: version 3. Version 2 (``v2:``) hashed the bundle's computed
#: document with every float rounded to twelve figures, and is still computed for what was stored under it
#: (:func:`~threetears.evals.analysis.bundle.schema.evidence_digest`'s ``like``).
EVIDENCE_FINGERPRINT_PREFIX: Final[str] = "v3:"

#: What a campaign bundle's fingerprint over inputs starts with: version 4 of its own numbering, whose version 3
#: hashed the campaign's computed readings (:func:`campaign_fingerprint`).
_CAMPAIGN_EVIDENCE_PREFIX: Final[str] = "v4:"


# ----------------------------------------------------------------------------- shapes: which fields are evidence


@dataclass(frozen=True)
class _Each:
    """A list of records, each of shape ``of``."""

    of: Mapping[str, Any]


@dataclass(frozen=True)
class _Keyed:
    """A map whose values are records of shape ``of``; its keys are kept."""

    of: Mapping[str, Any]


#: A field held as stored: a scalar, or a list or map of scalars (a free-form map is one). Never a record: a record
#: under this shape is refused (:func:`_stored_value`), so no nested record is taken whole.
_AS_STORED: Final = None

#: A request's settings as sent or asked.
_SETTINGS: Final[Mapping[str, Any]] = dict.fromkeys(
    ("max_tokens", "reasoning_max_tokens", "reasoning_effort", "strict_output", "unsent_reasoning_max_tokens"),
    _AS_STORED,
)

#: One judged score.
_SCORE: Final[Mapping[str, Any]] = {
    "dim": _AS_STORED,
    "scale": _AS_STORED,
    "axis": _AS_STORED,
    "score": _AS_STORED,
    "served_model": _AS_STORED,
    "judge_temperature": _AS_STORED,
    "judge_sent_settings": _SETTINGS,
    "output_fingerprint": _AS_STORED,
    "criterion_fingerprint": _AS_STORED,
}

_GOAL: Final[Mapping[str, Any]] = dict.fromkeys(("expression", "passed", "detail"), _AS_STORED)
_PRECONDITION: Final[Mapping[str, Any]] = dict.fromkeys(("expression", "presumes", "detail"), _AS_STORED)
_LATENCY: Final[Mapping[str, Any]] = dict.fromkeys(
    ("total_ms", "llm_ms", "tool_ms", "async_wait_ms", "judge_ms"), _AS_STORED
)
_RESCORE: Final[Mapping[str, Any]] = {"cost_usd": _AS_STORED}
#: When a re-judge was made dates a look (`reading_scope.looks`), which is counted at read time and kept out of the
#: fingerprint (`_ADDED_SINCE_V2`): no verdict, interval or figure reads it.
_RESCORE_NOT: Final[Mapping[str, str]] = {
    "rejudged_at": "dates a look only: the looks are counted at read time and are outside the fingerprint"
}

#: The fields of a usage row, or a background delivery, that are pricing and so no evidence (owner rulings,
#: 2026-10-28 and 2026-11-14): where its cost came from and who served it. The cost, the tokens and the calls stay.
_PRICING: Final[Mapping[str, str]] = {
    "price_source": "pricing: where the cost came from (owner rulings, 2026-10-28 and 2026-11-14)",
    "served_provider": "pricing: who served it (owner rulings, 2026-10-28 and 2026-11-14)",
}

_USAGE: Final[Mapping[str, Any]] = {
    **dict.fromkeys(
        (
            "role",
            "model",
            "served_model",
            "prompt_tokens",
            "completion_tokens",
            "reasoning_tokens",
            "cost_usd",
            "call_count",
            "provider",
            "provider_unit",
            "provider_units",
            "actor_id",
            "purpose",
            "unanswered_retried",
            "unanswered_lost",
            "unanswered_reasons",
            "maybe_billed_sends",
        ),
        _AS_STORED,
    ),
    "sent_settings": _Each(_SETTINGS),
}

_REPEATED_SCORE: Final[Mapping[str, Any]] = {
    "dim": _AS_STORED,
    "scale": _AS_STORED,
    "first_score": _AS_STORED,
    "first_served_model": _AS_STORED,
    "first_judge_config_id": _AS_STORED,
    "first_judge_temperature": _AS_STORED,
    "first_judge_sent_settings": _SETTINGS,
    "repeat": _SCORE,
    "cannot_tell": _AS_STORED,
}
_REPEAT: Final[Mapping[str, Any]] = {
    "judge_config_ids": _AS_STORED,
    "scores": _Each(_REPEATED_SCORE),
    # Whether the repeat was a random draw decides whether an arm's re-score noise is read from it.
    "sample": _AS_STORED,
}
_SECOND_SCORE: Final[Mapping[str, Any]] = {
    **{name: shape for name, shape in _REPEATED_SCORE.items() if name != "repeat"},
    "second": _SCORE,
}
_SECOND_JUDGE: Final[Mapping[str, Any]] = dict.fromkeys(("model", "config_ids", "temperature"), _AS_STORED)
_SECOND_JUDGING: Final[Mapping[str, Any]] = {
    "pass_id": _AS_STORED,
    "judge": _SECOND_JUDGE,
    "judge_config_ids": _AS_STORED,
    "scores": _Each(_SECOND_SCORE),
}

_EXTERNAL_SPEND: Final[Mapping[str, Any]] = dict.fromkeys(
    ("provider", "calls", "provider_units", "provider_unit", "money"), _AS_STORED
)
_DELIVERY: Final[Mapping[str, Any]] = {
    **dict.fromkeys(
        (
            "tool",
            "requested_by",
            "status",
            "model",
            "served_model",
            "acknowledged_turn",
            "delivered_turn",
            "elapsed_ms",
            "delivered_items",
            "error",
            "substituted",
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "llm_calls",
            "cost_usd",
        ),
        _AS_STORED,
    ),
    "external_spend": _Each(_EXTERNAL_SPEND),
}
_CUT_SHORT: Final[Mapping[str, Any]] = {
    "stop_reason": _AS_STORED,
    "sent_settings": _SETTINGS,
    "output_tokens": _AS_STORED,
    "reasoning_tokens": _AS_STORED,
}

#: A result's evidence: what its trial recorded, as the readers read it. Each is a stored value, copied from the
#: result and never recomputed (:meth:`~threetears.evals.kernel.result_reading.ResultReading.of`).
_RESULT: Final[Mapping[str, Any]] = {
    **dict.fromkeys(
        (
            "id",
            "eval_run_id",
            "test_case_id",
            "model",
            "k_iteration",
            "variant_key",
            "identity_version",
            "candidate_kind",
            "candidate_instance_id",
            "subject_id",
            "termination",
            "stop_cause",
            "turns_delivered",
            "runner_error",
            "judge_error",
            "candidate_error",
            "infra_error",
            "model_unserved",
            "judging_set_aside",
            "judge_model",
            "judge_config_ids",
            "judge_cannot_tell",
            "judge_cannot_tell_boundary",
            "cost_usd",
            "cost_roles",
            "superseded_spend_usd",
            "metered_calls",
            "metered_calls_refused",
            "covariates",
            "phase_timings",
            "host_measures",
            "scored_at",
        ),
        _AS_STORED,
    ),
    "precondition_outcomes": _Each(_PRECONDITION),
    "goal_state_outcomes": _Each(_GOAL),
    "rubric_scores": _Each(_SCORE),
    "transcript_score": _SCORE,
    "outcome_score": _SCORE,
    "judge_rescores": _Each(_RESCORE),
    "judge_repeats": _Each(_REPEAT),
    "judge_seconds": _Each(_SECOND_JUDGING),
    "judge_cut_short": _Keyed(_CUT_SHORT),
    "usage": _Each(_USAGE),
    "latency": _LATENCY,
    "async_deliveries": _Each(_DELIVERY),
}

#: The attributes of a result's reading that are not evidence, each with why.
_RESULT_NOT: Final[Mapping[str, str]] = {
    "doc_type": "the store's name for the document",
    "schema_version": "the shape the store wrote it in, which a migration moves and the trial did not record",
    "schema_minor": "the shape the store wrote it in, which a migration moves and the trial did not record",
    "scope_id": "stated once, in the declaration",
    "lease_epoch": "which job lease wrote it: the engine's bookkeeping",
    "has_trace": "whether a trace is still kept beside it, which retention decides and the trial did not record",
}

#: A level: held by its content hash alone, which digests its value. How a page shows it is presentation.
_LEVEL: Final[Mapping[str, Any]] = {"content_hash": _AS_STORED}
_LEVEL_NOT: Final[Mapping[str, str]] = {
    "display": "presentation: how a page shows the level (LEVEL_PRESENTATION_FIELDS)",
    "scale": "digested by content_hash",
    "raw": "digested by content_hash",
}

_SUBJECT: Final[Mapping[str, Any]] = {"subject_id": _AS_STORED, "components": _Keyed(_LEVEL), "state": _Keyed(_LEVEL)}
_SUBJECT_NOT: Final[Mapping[str, str]] = {
    "subject_label": "a name: names are presentation, never evidence (owner ruling, 2026-11-11)",
    "labels": "the host's tags for the subject: curation, as a run's labels are (#648)",
    "captured_at": "when the snapshot was taken; what it captured is the evidence",
}

_COMPLETENESS: Final[Mapping[str, Any]] = dict.fromkeys(
    ("expected_cells", "produced_cells", "persisted_cells", "infra_excluded_cells", "counted_from"), _AS_STORED
)
_BUDGET_STOP: Final[Mapping[str, Any]] = dict.fromkeys(
    (
        "currency",
        "ceiling",
        "landed_at_stop",
        "landed_final",
        "cells_in_flight",
        "cells_not_started",
        "cells_not_judged",
        "unpriced_results",
    ),
    _AS_STORED,
)

#: A host's statement of what a run's measures counted, for a run that recorded none.
_ATTESTATION: Final[Mapping[str, Any]] = dict.fromkeys(("attested_at", "attested_by", "definitions"), _AS_STORED)
_ATTESTATION_NOT: Final[Mapping[str, str]] = {
    "note": "the host's prose for why it knows; the definitions are the evidence"
}

#: A host's statement of the model that answered a run's candidate calls whose responses named none (#1187). The rows
#: stay as the responses left them; the statement, and the results it covers, are what the cells read them through.
_SERVED_ATTESTATION: Final[Mapping[str, Any]] = dict.fromkeys(
    ("attested_at", "attested_by", "served_model", "result_ids"), _AS_STORED
)
_SERVED_ATTESTATION_NOT: Final[Mapping[str, str]] = {
    "note": "the host's prose for why it knows; the stated model and the results it covers are the evidence"
}

#: One re-measure of a run: the definitions it held before and after, and what was read again.
_REMEASURE: Final[Mapping[str, Any]] = dict.fromkeys(
    ("remeasured_at", "completed_at", "was", "now", "reread", "results_reread"), _AS_STORED
)

#: A member run's evidence, as stored: everything the run was launched with and how it ended.
_RUN: Final[Mapping[str, Any]] = {
    **dict.fromkeys(
        (
            "id",
            "created_at",
            "completed_at",
            "template_id",
            "template_digest",
            "context_key",
            "identity_version",
            "launch_group_id",
            "candidate_model",
            "candidate_kind",
            "k_runs",
            "test_case_ids",
            "apparatus_provenance",
            "judge_model",
            "model_role_provenance",
            "effective_judges",
            "effective_judges_source",
            "rubric_scales",
            "judge_config_ids",
            "judge_config_provenance",
            "judge_temperature",
            "overlays",
            "apparatus_settings",
            "declared_margins",
            "resolved_world_seed",
            "resolved_ambient_perturbation_turns",
            "kind_spec",
            "world_placements",
            "goal_check_proofs",
            "goal_check_proof_rules",
            "refused_goal_checks",
            "resolved_tools_allowed",
            "cassette_mode",
            "cassette_corpus_id",
            "measure_latency",
            "cell_concurrency",
            "arm_schedule",
            "simulator_model",
            "simulator_prompt_digest",
            "max_cost_usd",
            "max_cost_usd_origin",
            "cap_checked_on_estimates",
            "max_metered_calls",
            "max_metered_calls_origin",
            "metered_calls_refused",
            "cell_timeout_s",
            "cell_timeout_s_origin",
            "turn_budget_s",
            "status",
            "cancellation_reason",
            "budget_stop_reason",
            "failed_because",
            "exhausted_because",
            "error_details",
            "faults",
        ),
        _AS_STORED,
    ),
    "subject_snapshot": _SUBJECT,
    "variation_counts": dict.fromkeys(("requested", "kept", "reused", "variation_model"), _AS_STORED),
    "declared_measures": _Keyed(
        dict.fromkeys(
            ("higher_is_better", "merit_axis", "guardrail", "materiality_threshold", "value_range"), _AS_STORED
        )
    ),
    "judge_asked_settings": _SETTINGS,
    "simulator_asked_settings": _SETTINGS,
    "run_disclosure": dict.fromkeys(("values", "statements"), _AS_STORED),
    "case_set": dict.fromkeys(("name", "version"), _AS_STORED),
    "context_components": dict.fromkeys(
        (
            "subject_state",
            "seeded_world",
            "case_basis",
            "roles",
            "cassette",
            "tool_permissions",
            "world",
            "apparatus_settings",
            "scope",
            "measure_definitions",
        ),
        _AS_STORED,
    ),
    "variant_levers": _Keyed(_LEVEL),
    "completeness": _COMPLETENESS,
    "budget_stops": _Each(_BUDGET_STOP),
    # Why a lost run ended is what a page opens on for it; when and under which lease is the engine's bookkeeping.
    "lost": {"cause": _AS_STORED},
    # What each of the host's measures counted (#1160): as recorded at launch, and as a host's attestation or a
    # re-measure moved it afterwards, which decides which runs pool. None of the three on a run that recorded none.
    "measure_definitions": _AS_STORED,
    "measure_attestation": _ATTESTATION,
    "served_model_attestation": _SERVED_ATTESTATION,
    "measure_remeasures": _Each(_REMEASURE),
}

#: The stored fields of a run that are not evidence, each with why.
_RUN_NOT: Final[Mapping[str, str]] = {
    "doc_type": "the store's name for the document",
    "schema_version": "the shape the store wrote it in, which a migration moves",
    "schema_minor": "the shape the store wrote it in, which a migration moves",
    "scope_id": "stated once, in the declaration",
    "host_payload": "the host's own record, which a listing may elide and the assembly reads elided",
    "package_version": "the build that launched it: a build is not evidence",
    "package_rev": "the build that launched it: a build is not evidence",
    "progress": "a counter written while the run is in flight; what it delivered is completeness",
    "archived": "stated by membership: an archived member is listed by id and none of its fields is read",
    "labels": "curation: a label moves no key and no fingerprint (#648)",
    "lease": "which job holds the run now: the engine's bookkeeping",
    "cancel_requested_at": "the request to stop; cancellation_reason is how it ended",
    "cancel_request_reason": "the request to stop; cancellation_reason is how it ended",
    "lifetimes": "the run's earlier lifetimes before a resume, which no reader of a campaign reads",
}

_LOSS_NOT: Final[Mapping[str, str]] = {
    "at": "when the lease was found lapsed: the engine's bookkeeping",
    "epoch": "which lease lapsed: the engine's bookkeeping",
}

#: A rating's evidence: which output it rates, who rated it, and what they gave.
_RATING: Final[Mapping[str, Any]] = dict.fromkeys(
    (
        "id",
        "run_id",
        "result_id",
        "rubric_dim",
        "rater",
        "rater_kind",
        "scale",
        "score",
        "rated",
        "rated_withheld",
        "output_fingerprint",
        "criterion_fingerprint",
    ),
    _AS_STORED,
)
_RATING_NOT: Final[Mapping[str, str]] = {
    "doc_type": "the store's name for the document",
    "schema_version": "the shape the store wrote it in",
    "schema_minor": "the shape the store wrote it in",
    "scope_id": "stated once, in the declaration",
    "reason": "the rater's prose about the score; the score is the evidence",
    "source": "where the person gave it: provenance of the relay, not what they gave",
    "collected_by": "who relayed it: provenance of the relay, not what they gave",
    "rated_at": "when it was given; a rating given again replaces the one it restates",
}

#: A prior insight's evidence: the claim the bundle hands its reader, what it rests on and which analysis minted it.
_INSIGHT: Final[Mapping[str, Any]] = dict.fromkeys(
    (
        "id",
        "subject_id",
        "subject_kind",
        "statement",
        "scope",
        "confidence",
        "evidence_run_ids",
        "evidence_result_ids",
        "observed_at",
        "model_versions",
        "invalidation_trigger",
        "source_campaign_id",
        "source_analysis_id",
    ),
    _AS_STORED,
)
_INSIGHT_NOT: Final[Mapping[str, str]] = {
    "doc_type": "the store's name for the document",
    "schema_version": "the shape the store wrote it in",
    "scope_id": "stated once, in the declaration",
}

#: A registered bar's evidence: the standard itself.
_BAR: Final[Mapping[str, Any]] = dict.fromkeys(
    ("behavior", "measure", "threshold", "higher_is_better", "judged", "vacuous_seed"), _AS_STORED
)
_BAR_NOT: Final[Mapping[str, str]] = {"rationale": "the host's prose for why the bar stands; the bar is the evidence"}

#: A catalogued measure's evidence: everything it declares that a reading, a comparison or a verdict is decided by.
_MEASURE: Final[Mapping[str, Any]] = dict.fromkeys(
    (
        "name",
        "data_type",
        "unit",
        "scale",
        "categories",
        "family",
        "higher_is_better",
        "merit_axis",
        "guardrail",
        "diagnostic",
        "materiality_threshold",
        "value_range",
        "nonnegative",
        "pass_fail",
        "population",
        "contained_by",
        "attribution_scope",
        "subject_spend",
        "transferability_class",
        "definition_revision",
        "shapes_run",
    ),
    _AS_STORED,
)
_MEASURE_NOT: Final[Mapping[str, str]] = {
    "description": "prose about the measure",
    "formula": "prose: how a derived measure is computed, shown at point of use",
    "reader_name": "presentation: what a reader calls it (MEASURE_PRESENTATION_FIELDS)",
    "reader_prose": "presentation: what a reader is told of it (MEASURE_PRESENTATION_FIELDS)",
}

#: What a tier's entry holds of a stored judge profile it consulted, each a value the profile stores.
_PROFILE_EVIDENCE: Final[tuple[str, ...]] = (
    "profile_id",
    "criterion_digest",
    "measured_at",
    "recorded_at",
    "case_set_fingerprint",
    "run_ids",
)

#: The fields of an aspect's declaration that are evidence. Left out: ``unread_without_control``, a sentence code
#: writes about the declaration.
_DECLARATION_EVIDENCE: Final[tuple[str, ...]] = (
    "question",
    "asked_at",
    "margin_declared_at",
    "design",
    "control",
    "margin",
    "floors",
    "declared_at",
    "declared_by",
)


def _fields_of(record: type) -> frozenset[str]:
    """The fields a stored record's type declares: a model's, or a slim reading's."""
    if issubclass(record, BaseModel):
        return frozenset(record.model_fields)
    if not dataclasses.is_dataclass(record):
        raise TypeError(f"{record.__name__} is neither a model nor a dataclass, so it declares no fields to decide")
    return frozenset(field.name for field in dataclasses.fields(record))


def _census(record: type, shape: Mapping[str, Any], left_out: Mapping[str, str] | None = None) -> None:
    """Hold that every field of ``record`` is decided: named in its ``shape`` as evidence, or in ``left_out`` with
    why it is not, and never both.

    :raises RuntimeError: A field is undecided, decided twice, or named and not a field of the record.
    """
    declared, taken, dropped = _fields_of(record), frozenset(shape), frozenset(left_out or {})
    undecided, twice, unknown = declared - taken - dropped, taken & dropped, (taken | dropped) - declared
    if undecided or twice or unknown:
        raise RuntimeError(
            f"{record.__name__}'s fields are not all decided as evidence or not "
            f"(threetears.evals.analysis.bundle.evidence): undecided {sorted(undecided)}, decided twice "
            f"{sorted(twice)}, named and not fields {sorted(unknown)}. Anything that can change a verdict or a "
            "printed number is evidence; name a field in the record's shape, or in its left-out map with the reason."
        )


_census(ResultReading, _RESULT, _RESULT_NOT)
_census(ScoreReading, _SCORE)
_census(GoalReading, _GOAL)
_census(PreconditionReading, _PRECONDITION)
_census(LatencyReading, _LATENCY)
_census(RescoreReading, _RESCORE, _RESCORE_NOT)
_census(UsageReading, _USAGE, _PRICING)
_census(RepeatedScoreReading, _REPEATED_SCORE)
_census(RepeatReading, _REPEAT)
_census(SecondScoreReading, _SECOND_SCORE)
_census(SecondJudgeReading, _SECOND_JUDGE)
_census(SecondJudgingReading, _SECOND_JUDGING)
_census(ClientRequestSettings, _SETTINGS)
_census(AsyncDelivery, _DELIVERY, {**_PRICING, "summary": "prose the delivery returned; what it delivered is counted"})
_census(AsyncExternalSpend, _EXTERNAL_SPEND)
_census(JudgeCutShort, _CUT_SHORT)
_census(EvalRun, _RUN, _RUN_NOT)
_census(MeasureAttestation, _ATTESTATION, _ATTESTATION_NOT)
_census(ServedModelAttestation, _SERVED_ATTESTATION, _SERVED_ATTESTATION_NOT)
_census(MeasureRemeasure, _REMEASURE)
_census(SubjectSnapshot, _SUBJECT, _SUBJECT_NOT)
_census(SweepableValue, _LEVEL, _LEVEL_NOT)
_census(VariationCounts, _RUN["variation_counts"])
_census(MeasureDeclaration, _RUN["declared_measures"].of)
_census(RunDisclosure, _RUN["run_disclosure"])
_census(CaseSetRef, _RUN["case_set"])
_census(ContextComponents, _RUN["context_components"])
_census(RunCompleteness, _COMPLETENESS)
_census(BudgetStop, _BUDGET_STOP)
_census(JobLoss, _RUN["lost"], _LOSS_NOT)
_census(CalibrationRating, _RATING, _RATING_NOT)
_census(EvalInsight, _INSIGHT, _INSIGHT_NOT)
_census(EvalCaseStratum, {"id": _AS_STORED, "stratum": _AS_STORED}, {"name": "a name: presentation"})
_census(Bar, _BAR, _BAR_NOT)
_census(MetricDescriptor, _MEASURE, _MEASURE_NOT)


def _stored_value(value: Any) -> Any:
    """A value held as stored (:data:`_AS_STORED`): a scalar, or a list or map of them, as plain JSON.

    :raises TypeError: The value is, or holds, a record: a record is built from its named fields, never taken whole.
    """
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _stored_value(held) for key, held in value.items()}
    if isinstance(value, list | tuple):
        return [_stored_value(held) for held in value]
    if isinstance(value, set | frozenset):
        return sorted(_stored_value(held) for held in value)
    raise TypeError(
        f"a {type(value).__name__} is held where evidence takes a stored value: a record is built from its named "
        f"fields, never taken whole ({value!r})"
    )


def _take(value: Any, shape: Any) -> Any:
    """``value`` as its ``shape`` reads it: a record as its named fields that are set (not null, and not an empty list
    or map), a list or map of records each so, anything else as stored. A record is read by attribute or, where it is a stored document, by key."""
    if value is None:
        return None
    if shape is _AS_STORED:
        return _stored_value(value)
    if isinstance(shape, _Each):
        return [_take(item, shape.of) for item in value]
    if isinstance(shape, _Keyed):
        return {str(key): _take(held, shape.of) for key, held in value.items()}
    read = value.get if isinstance(value, Mapping) else lambda name: getattr(value, name)
    taken = {name: _take(read(name), nested) for name, nested in shape.items()}
    # Unset is null, or an empty list or map: a record written before a field existed and one that holds it empty
    # are one, so naming a field later moves nothing for a record that never set it.
    return {name: held for name, held in taken.items() if held is not None and held != [] and held != {}}


# ----------------------------------------------------------------------------- the manifest


class EvidenceManifest(EvalDocumentModel):
    """What a bundle was assembled from: one digest per kind of input, and a count where one helps (see the module).

    Its :func:`manifest_fingerprint` is the bundle's fingerprint. Two manifests that differ name, by the component
    whose digest differs, what moved.
    """

    declaration: str = Field(description="Digest of the campaign and aspect asked about and what the aspect declares.")
    criteria: str = Field(
        description=(
            "Digest of what a verdict is decided against that the host declares: the pass threshold, the "
            "registered bars and each catalogued measure's declaration."
        )
    )
    host_declarations: str = Field(description="The bundle's `host_declarations_digest`.")
    runs: str = Field(description="Digest of the member runs read, and of the archived and unresolved members' ids.")
    n_runs: int = Field(ge=0, description="Member runs read.")
    results: str = Field(description="Digest of every result read, as the readers read it, in id order.")
    n_results: int = Field(ge=0, description="Results read.")
    cases: str = Field(description="Digest of each case the results name and the stratum it is filed under.")
    n_cases: int = Field(ge=0, description="Cases read.")
    ratings: str = Field(description="Digest of each rating of a judged output read.")
    n_ratings: int = Field(ge=0, description="Ratings read.")
    judge_profiles: str = Field(description="Digest of each stored judge profile a judge's tier consulted.")
    n_judge_profiles: int = Field(ge=0, description="Stored judge profiles consulted.")
    insights: str = Field(description="Digest of each prior insight read as of the cutoff and each one retracted.")
    n_insights: int = Field(ge=0, description="Prior insights read.")


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """A manifest's fingerprint: :data:`EVIDENCE_FINGERPRINT_PREFIX` and the digest of its stored document.

    :param manifest: An :class:`EvidenceManifest` as its stored dict.
    :ptype manifest: Mapping[str, Any]
    :return: ``v3:`` and a 64-character lowercase hex digest.
    :rtype: str
    """
    return EVIDENCE_FINGERPRINT_PREFIX + input_digest(dict(manifest))


def campaign_fingerprint(declaration: Mapping[str, Any], aspects: Sequence[tuple[str, str]]) -> str:
    """A campaign bundle's fingerprint over its evidence: :data:`_CAMPAIGN_EVIDENCE_PREFIX` and the digest of what the
    campaign declares and each aspect's bundle fingerprint, which is over that aspect's evidence.

    Nothing the campaign computes is read: no reading across aspects, no verdict, no reason text. It means for the
    campaign what a ``v3:`` fingerprint means for one aspect.

    :param declaration: The campaign's own evidence: its ids and what it declares, names left out.
    :ptype declaration: Mapping[str, Any]
    :param aspects: Each aspect's label and ``v3:`` fingerprint, in the campaign's order.
    :ptype aspects: Sequence[tuple[str, str]]
    :return: ``v4:`` and a 64-character lowercase hex digest.
    :rtype: str
    """
    return _CAMPAIGN_EVIDENCE_PREFIX + input_digest(
        {"campaign": dict(declaration), "aspects": [list(aspect) for aspect in aspects]}
    )


# ----------------------------------------------------------------------------- names are presentation


def unpresented(items: Sequence[Any], fields: frozenset[str]) -> list[Any]:
    """``items`` with each mapping's ``fields`` left out; anything else as it is.

    :param items: Stored dicts.
    :ptype items: Sequence[Any]
    :param fields: The presentation fields to leave out.
    :ptype fields: frozenset[str]
    :return: The items without them.
    :rtype: list[Any]
    """
    return [
        {key: value for key, value in item.items() if key not in fields} if isinstance(item, Mapping) else item
        for item in items
    ]


def unnamed_levels(levels: Mapping[str, Any]) -> dict[str, Any]:
    """A stored lever map (lever id to level) as a fingerprint reads it: each level without how a page shows it
    (:data:`~threetears.evals.kernel.declaration.LEVEL_PRESENTATION_FIELDS`), by its content hash alone, so naming or
    renaming a level moves no bundle's or campaign bundle's fingerprint.

    :param levels: Lever id to the level, as its stored dict.
    :ptype levels: Mapping[str, Any]
    :return: The map, each level unnamed.
    :rtype: dict[str, Any]
    """
    return {
        lever: {key: value for key, value in level.items() if key not in LEVEL_PRESENTATION_FIELDS}
        if isinstance(level, Mapping)
        else level
        for lever, level in levels.items()
    }


def unnamed_axes(design: Mapping[str, Any]) -> dict[str, Any]:
    """A stored design as a fingerprint reads it: each declared axis without how a page names its lever
    (:data:`~threetears.evals.kernel.declaration.AXIS_PRESENTATION_FIELDS`) and each of its levels without how a page
    shows it (:data:`~threetears.evals.kernel.declaration.LEVEL_PRESENTATION_FIELDS`), so naming or renaming a lever
    or a level moves no bundle's or campaign bundle's fingerprint. A design whose axes carry no name is returned as
    it is.

    :param design: An aspect's design, as its stored dict.
    :ptype design: Mapping[str, Any]
    :return: The design, its axes and their levels unnamed.
    :rtype: dict[str, Any]
    """
    axes = design.get("axes")
    if not isinstance(axes, list):
        return dict(design)
    unnamed = []
    for axis in axes:
        if not isinstance(axis, Mapping):
            unnamed.append(axis)
            continue
        held = {key: value for key, value in axis.items() if key not in AXIS_PRESENTATION_FIELDS}
        if isinstance(values := held.get("values"), list):
            held["values"] = unpresented(values, LEVEL_PRESENTATION_FIELDS)
        unnamed.append(held)
    return {**design, "axes": unnamed}


# ----------------------------------------------------------------------------- each component


def _result_evidence(result: ResultView) -> dict[str, Any]:
    """One result's evidence: the fields :data:`_RESULT` names, of its reading.

    A store hands the assembly each result as its reading. A view that is the stored result itself is read as its
    dump, by the same names, so either gives the same evidence (``tests/test_evidence_fingerprint.py``).

    :param result: A result as the store returned it, before any lens reads it.
    :ptype result: ResultView
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    # A stored result is a model; its reading is read by attribute.
    held = cast("EvalResult", result).model_dump(mode="json") if isinstance(result, BaseModel) else result
    evidence: dict[str, Any] = _take(held, _RESULT)
    # The dimensions the judge could not tell, without its notes: held as a mapping's keys, whose order a store does
    # not keep.
    evidence["judge_cannot_tell"] = sorted(evidence.get("judge_cannot_tell") or [])
    return evidence


class ResultsDigest:
    """The digest of the results an assembly reads, taken as each is read, whatever order the store hands them in.

    Each result's evidence is rendered and hashed on its own (:func:`_result_evidence`), and :meth:`digest` hashes
    those digests in id order, so the read's order is not read and no result's evidence is held after it is hashed.
    """

    __slots__ = ("_digests",)

    def __init__(self) -> None:
        self._digests: dict[str, str] = {}

    def add(self, result: ResultView) -> None:
        """Take one result as the store returned it.

        :param result: The result.
        :ptype result: ResultView
        """
        rendered = input_json(_result_evidence(result)).encode("utf-8")
        self._digests[result.id] = hashlib.sha256(rendered).hexdigest()

    def __len__(self) -> int:
        return len(self._digests)

    def digest(self) -> str:
        """The digest of every result taken: each id and its evidence's digest, in id order.

        :return: A 64-character lowercase hex digest.
        :rtype: str
        """
        return input_digest(sorted(self._digests.items()))


def _stored(document: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    """The named ``fields`` of a stored document, each as stored; one it does not carry is left out."""
    return {name: document[name] for name in fields if name in document}


def _run_evidence(run: EvalRun) -> dict[str, Any]:
    """One member run's evidence: the fields :data:`_RUN` names, as stored, each level by its content hash.

    :param run: The stored run.
    :ptype run: EvalRun
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    evidence: dict[str, Any] = _take(run, _RUN)
    return evidence


def declaration_evidence(view: AspectView) -> dict[str, Any]:
    """What was asked: the campaign and aspect, and the aspect's declaration without how a page prints it.

    The declaration is taken as its stored document, rendered **without any field that holds its declared default**
    (at every depth), so a field a later build adds to a design, a floor or a clause, whatever its default, is not in
    the pre-image until a campaign sets it, and moves no stored fingerprint.

    :param view: The aspect as a per-aspect reader reads it.
    :ptype view: AspectView
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    declared: dict[str, Any] | None = None
    if view.declared is not None:
        declared = _stored(view.declared.model_dump(mode="json", exclude_defaults=True), _DECLARATION_EVIDENCE)
        if isinstance(control := declared.get("control"), Mapping):
            declared["control"] = {key: value for key, value in control.items() if key not in ARM_PRESENTATION_FIELDS}
        if isinstance(design := declared.get("design"), Mapping):
            declared["design"] = unnamed_axes(design)
    return {
        "campaign_id": view.campaign_id,
        "scope_id": view.scope_id,
        "aspect": view.aspect,
        "subject_id": view.subject_id,
        "subject_kind": view.subject_kind,
        "behavior": view.behavior,
        "template_id": view.template_id,
        "declared": declared,
    }


def criteria_evidence(behavior: str, catalog: Mapping[str, MetricDescriptor], profile: HostProfile) -> dict[str, Any]:
    """What a verdict is decided against that the host declares: the pass threshold for ``behavior``, each bar the
    host registers for it, and each catalogued measure as it is declared.

    :param behavior: The campaign's behavior, which a pass threshold and a registered bar are keyed under.
    :ptype behavior: str
    :param catalog: The bundle's measure catalog: each measure it reads, as launched.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :param profile: The host profile the bundle was assembled under.
    :ptype profile: HostProfile
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    registered = sorted(
        (bar for bar in profile.bars.bars if bar.behavior == behavior), key=lambda bar: (bar.measure, bar.judged)
    )
    return {
        "pass_threshold": profile.bars.pass_threshold(behavior),
        "bars": [_take(bar, _BAR) for bar in registered],
        "measures": {name: _take(catalog[name], _MEASURE) for name in sorted(catalog)},
    }


def _consulted_profiles(tiers: Any) -> list[dict[str, Any]]:
    """Every stored judge profile the tiers name, once each, by id: wherever a tier's entry holds a ``profile_id``."""
    found: dict[str, dict[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if isinstance(profile_id := value.get("profile_id"), str):
                found[profile_id] = _stored(value, _PROFILE_EVIDENCE)
            for held in value.values():
                walk(held)
        elif isinstance(value, list):
            for held in value:
                walk(held)

    walk(tiers)
    return [found[profile_id] for profile_id in sorted(found)]


def evidence_manifest(
    view: AspectView,
    *,
    profile: HostProfile,
    host_declarations: str,
    catalog: Mapping[str, MetricDescriptor],
    runs: Sequence[EvalRun],
    archived_run_ids: Sequence[str],
    unresolved_run_ids: Sequence[str],
    results: ResultsDigest,
    cases: Sequence[EvalCaseStratum],
    ratings: Sequence[CalibrationRating],
    tiers: Sequence[BaseModel],
    insights: Sequence[EvalInsight],
    retracted_insights: Mapping[str, str],
) -> EvidenceManifest:
    """The manifest of one assembly: what it read, each kind digested (see the module).

    :param view: The aspect assembled.
    :ptype view: AspectView
    :param profile: The host profile assembled under, as the runs were launched to be read.
    :ptype profile: HostProfile
    :param host_declarations: The bundle's ``host_declarations_digest``.
    :ptype host_declarations: str
    :param catalog: The bundle's measure catalog.
    :ptype catalog: Mapping[str, MetricDescriptor]
    :param runs: The member runs read: resolved and not archived.
    :ptype runs: Sequence[EvalRun]
    :param archived_run_ids: The members left out as archived.
    :ptype archived_run_ids: Sequence[str]
    :param unresolved_run_ids: The members the store did not return.
    :ptype unresolved_run_ids: Sequence[str]
    :param results: Every result read, taken as the store returned it.
    :ptype results: ResultsDigest
    :param cases: Each case the results name, as read.
    :ptype cases: Sequence[EvalCaseStratum]
    :param ratings: Each rating read.
    :ptype ratings: Sequence[CalibrationRating]
    :param tiers: The judges' tiers, which name each stored profile consulted.
    :ptype tiers: Sequence[BaseModel]
    :param insights: Every prior insight read as of the cutoff, retracted ones included, before the carried cap.
    :ptype insights: Sequence[EvalInsight]
    :param retracted_insights: Each retracted insight's id, to the archived analysis that minted it.
    :ptype retracted_insights: Mapping[str, str]
    :return: The manifest.
    :rtype: EvidenceManifest
    """
    profiles = _consulted_profiles([tier.model_dump(mode="json") for tier in tiers])
    return EvidenceManifest(
        declaration=input_digest(declaration_evidence(view)),
        criteria=input_digest(criteria_evidence(view.behavior, catalog, profile)),
        host_declarations=host_declarations,
        runs=input_digest(
            {
                "read": sorted((_run_evidence(run) for run in runs), key=lambda run: run["id"]),
                "archived": sorted(archived_run_ids),
                "unresolved": sorted(unresolved_run_ids),
            }
        ),
        n_runs=len(runs),
        results=results.digest(),
        n_results=len(results),
        cases=input_digest(sorted(([case.id, case.stratum] for case in cases), key=lambda case: str(case[0]))),
        n_cases=len(cases),
        ratings=input_digest(sorted((_take(rating, _RATING) for rating in ratings), key=lambda rating: rating["id"])),
        n_ratings=len(ratings),
        judge_profiles=input_digest(profiles),
        n_judge_profiles=len(profiles),
        insights=input_digest(
            {
                "read": sorted((_take(insight, _INSIGHT) for insight in insights), key=lambda insight: insight["id"]),
                "retracted": sorted(retracted_insights.items()),
            }
        ),
        n_insights=len(insights),
    )
