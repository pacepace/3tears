"""What a bundle was assembled from, as a manifest of digests: the pre-image of its fingerprint (version 3, #1062).

A bundle's fingerprint answers one question, "is this the same evidence?", for a stored analysis, a reporter case and
the inspection's ``mismatch_cause``. Versions 1 and 2 hashed the bundle itself, so they hashed every statistic the
package computes from the evidence: a more accurate formula, another platform's maths library or a reordered sum read
as the evidence moving, and what was stored against the old fingerprint was regenerated at a model's price. Version 3
hashes **what the assembly read from the store, and nothing it computed**:

- ``declaration``: the campaign and aspect asked about, and what the aspect declares: its design (each level by its
  content hash), control, margin, floors, and when each was declared.
- ``runs``: each member run the assembly read: its id, its content-addressed keys (context key, identity version,
  template digest, each varied lever's content hash), how it ended and what it delivered (status, completeness), and
  the launch declarations no key digests (:data:`RUN_EVIDENCE`). Beside them, the member runs archived and the ones
  that did not resolve, by id.
- ``results``: each result the assembly read, as the readers read it
  (:class:`~threetears.evals.kernel.result_reading.ResultReading`): its outcome, scores, goal states, usage, latency,
  exclusion cause and every other recorded value (:data:`RESULT_EVIDENCE`), in id order.
- ``cases``: each case the results name that the store holds, with the stratum it is filed under.
- ``ratings``: each person's or model's rating of a judged output the assembly read (:data:`RATING_EVIDENCE`).
- ``judge_profiles``: each stored judge profile a judge's tier consulted: its id, when it was measured and recorded,
  the cases and the runs that measured it.
- ``insights``: each prior insight the bundle carries (:data:`INSIGHT_EVIDENCE`), and each one retracted, by id.
- ``host_declarations``: :func:`~threetears.evals.analysis.bundle.schema.host_declarations_digest`, as the bundle
  carries it.

**Out, by construction.** Every computed statistic (a mean, an interval, a p-value, an effect size, a verdict), so
changing how one is computed moves no fingerprint. Every name and label (a campaign's, an aspect's, an arm's, a
lever's, a level's, a measure's, a test case's): names are presentation, never evidence (owner ruling, 2026-11-11).
Pricing (owner rulings, 2026-10-28 and 2026-11-14, #1011): where a usage row's cost came from (``price_source``) and
who served it (``served_provider``) are no part of a result's evidence; the cost a result recorded is. Reason text,
the bundle's ``schema_version`` and the build that assembled it (``package_version``).

**Constructed, not subtracted.** Each list below names the stored fields that are evidence. A field added to a
stored shape later enters a fingerprint only when it is added to its list, and a list grows only by a field that is
unset (null or absent) on every document stored before it, which :func:`~threetears.evals.schema.hashing.input_json`
renders exactly as it rendered the document without the field. So an additive change to a stored shape, or to the
bundle, moves no stored fingerprint.

**How it is rendered.** Through :func:`~threetears.evals.schema.hashing.input_json`: keys sorted at every depth, so
a store that reorders them (``jsonb``) moves nothing; null and absent alike; numbers as that function states, which
reads no maths library and rounds nothing, since a stored value parses to the same double on every platform. Lists
whose order a store does not promise (the runs, the results, the cases, the ratings, the judge profiles) are sorted
by id here.

The manifest is a few hundred bytes: one digest and one count per component. It rides on the bundle
(:attr:`~threetears.evals.analysis.bundle.schema.AnalysisContextBundle.evidence`), so a frozen bundle still proves
its own fingerprint, and comparing two manifests names which component moved.
"""

from __future__ import annotations

import dataclasses
import hashlib
from enum import Enum
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from pydantic import BaseModel, Field

from threetears.evals.kernel.declaration import (
    ARM_PRESENTATION_FIELDS,
    AXIS_PRESENTATION_FIELDS,
    LEVEL_PRESENTATION_FIELDS,
)
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

if TYPE_CHECKING:
    from threetears.evals.analysis.aspects import AspectView
    from threetears.evals.kernel.campaign import EvalInsight
    from threetears.evals.schema.models import CalibrationRating, EvalCaseStratum, EvalResult, EvalRun

__all__ = [
    "EVIDENCE_FINGERPRINT_PREFIX",
    "INSIGHT_EVIDENCE",
    "RATING_EVIDENCE",
    "RESULT_EVIDENCE",
    "RESULT_NOT_EVIDENCE",
    "RUN_EVIDENCE",
    "USAGE_NOT_EVIDENCE",
    "EvidenceManifest",
    "ResultsDigest",
    "evidence_manifest",
    "manifest_fingerprint",
    "result_evidence",
    "unnamed_axes",
    "unnamed_levels",
    "unpresented",
]

#: What a fingerprint over inputs starts with: version 3. Version 2 (``v2:``) hashed the bundle's computed document
#: with every float rounded to twelve figures, and is still computed for what was stored under it
#: (:func:`~threetears.evals.analysis.bundle.schema.evidence_digest`'s ``like``).
EVIDENCE_FINGERPRINT_PREFIX: Final[str] = "v3:"

#: The stored fields of a member run that are evidence, each rendered as stored. Its content-addressed keys first
#: (they digest the apparatus, the levers and the template revision), then how it ended and what it delivered, then
#: the launch declarations the assembly reads that no key digests. A varied lever is held by its content hash alone
#: (:func:`unnamed_levels`). Left out: what a host keeps for itself (``host_payload``, ``labels``), the store's own
#: bookkeeping (``schema_version``, ``lease``, ``progress``), the build that launched it, and every reason text.
RUN_EVIDENCE: Final[tuple[str, ...]] = (
    "id",
    "context_key",
    "identity_version",
    "template_id",
    "template_digest",
    "variant_levers",
    "status",
    "completeness",
    "created_at",
    "candidate_kind",
    "candidate_model",
    "k_runs",
    "test_case_ids",
    "case_set",
    "declared_margins",
    "declared_measures",
    "goal_check_proofs",
    "goal_check_proof_rules",
    "rubric_scales",
    "judge_model",
    "effective_judges",
    "judge_config_ids",
    "run_disclosure",
)

#: The attributes of a result's reading that are evidence: what its trial recorded. Each is a stored value, copied
#: from the result and never recomputed (:meth:`~threetears.evals.kernel.result_reading.ResultReading.of`).
RESULT_EVIDENCE: Final[tuple[str, ...]] = (
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
    "precondition_outcomes",
    "goal_state_outcomes",
    "rubric_scores",
    "transcript_score",
    "outcome_score",
    "judge_model",
    "judge_config_ids",
    "judge_rescores",
    "judge_repeats",
    "judge_seconds",
    "judge_cannot_tell",
    "judge_cannot_tell_boundary",
    "judge_cut_short",
    "cost_usd",
    "cost_roles",
    "superseded_spend_usd",
    "metered_calls",
    "metered_calls_refused",
    "usage",
    "latency",
    "async_deliveries",
    "covariates",
    "phase_timings",
    "host_measures",
    "scored_at",
)

#: The attributes of a result's reading that are not evidence, each with why: the store's own bookkeeping, and what
#: the manifest states once. Every attribute of a reading is in :data:`RESULT_EVIDENCE` or here, so a new one is
#: decided on, never taken or dropped by default (``tests/test_evidence_fingerprint.py``).
RESULT_NOT_EVIDENCE: Final[Mapping[str, str]] = {
    "doc_type": "the store's name for the document",
    "schema_version": "the shape the store wrote it in, which a migration moves and the trial did not record",
    "schema_minor": "the shape the store wrote it in, which a migration moves and the trial did not record",
    "scope_id": "stated once, in the declaration",
    "lease_epoch": "which job lease wrote it: the engine's bookkeeping",
    "has_trace": "whether a trace is still kept beside it, which retention decides and the trial did not record",
}

#: The fields of a result's usage row that are not evidence: where its cost came from and who served it, which is
#: pricing (owner rulings, 2026-10-28 and 2026-11-14). The census of them is
#: :class:`~threetears.evals.analysis.bundle.schema.ArmPricing`, which version 2 left out too. The cost itself, the
#: tokens and the calls stay.
USAGE_NOT_EVIDENCE: Final[frozenset[str]] = frozenset({"price_source", "served_provider"})

#: The stored fields of a rating that are evidence: which output it rates, who rated it, and what they gave. Left out:
#: the rater's prose (``reason``), who relayed it and where it was given (``collected_by``, ``source``), and when.
RATING_EVIDENCE: Final[tuple[str, ...]] = (
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
)

#: The stored fields of a prior insight that are evidence: the claim the bundle hands its reader, what it rests on
#: and which analysis minted it.
INSIGHT_EVIDENCE: Final[tuple[str, ...]] = (
    "id",
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
)

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


class EvidenceManifest(EvalDocumentModel):
    """What a bundle was assembled from: one digest and one count per kind of input (see the module).

    Its :func:`manifest_fingerprint` is the bundle's fingerprint. Two manifests that differ name, by the component
    whose digest differs, what moved.
    """

    declaration: str = Field(description="Digest of the campaign and aspect asked about and what the aspect declares.")
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
    insights: str = Field(description="Digest of each prior insight carried and each one retracted.")
    n_insights: int = Field(ge=0, description="Prior insights carried.")


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """A manifest's fingerprint: :data:`EVIDENCE_FINGERPRINT_PREFIX` and the digest of its stored document.

    :param manifest: An :class:`EvidenceManifest` as its stored dict.
    :ptype manifest: Mapping[str, Any]
    :return: ``v3:`` and a 64-character lowercase hex digest.
    :rtype: str
    """
    return EVIDENCE_FINGERPRINT_PREFIX + input_digest(dict(manifest))


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


def _plain(value: Any) -> Any:
    """A held value as the JSON it was stored as: a reading's slim record as a dict of its attributes, a stored model
    as its dump, a tuple as a list."""
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {name: _plain(getattr(value, name)) for name in _attributes(type(value))}
    if isinstance(value, Mapping):
        return {str(key): _plain(held) for key, held in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(held) for held in value]
    if isinstance(value, set | frozenset):
        return sorted(_plain(held) for held in value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"a reading holds a {type(value).__name__}, which no stored result holds: {value!r}")


_ATTRIBUTES: dict[type, tuple[str, ...]] = {}


def _attributes(record: type) -> tuple[str, ...]:
    """A slim record's attribute names, read once per type."""
    names = _ATTRIBUTES.get(record)
    if names is None:
        names = _ATTRIBUTES[record] = tuple(field.name for field in dataclasses.fields(record))
    return names


#: Which slim record each attribute of a reading, and of a slim record, holds: the shape a stored result is read in.
#: An attribute not named here holds stored values whole.
_RECORD_OF: Final[Mapping[tuple[type | None, str], type]] = {
    (None, "precondition_outcomes"): PreconditionReading,
    (None, "goal_state_outcomes"): GoalReading,
    (None, "rubric_scores"): ScoreReading,
    (None, "transcript_score"): ScoreReading,
    (None, "outcome_score"): ScoreReading,
    (None, "judge_rescores"): RescoreReading,
    (None, "judge_repeats"): RepeatReading,
    (None, "judge_seconds"): SecondJudgingReading,
    (None, "usage"): UsageReading,
    (None, "latency"): LatencyReading,
    (RepeatReading, "scores"): RepeatedScoreReading,
    (RepeatedScoreReading, "repeat"): ScoreReading,
    (SecondJudgingReading, "judge"): SecondJudgeReading,
    (SecondJudgingReading, "scores"): SecondScoreReading,
    (SecondScoreReading, "second"): ScoreReading,
}


def _as_read(stored: Any, record: type) -> Any:
    """A stored value (a result's dumped record, or a list of them) with only what its slim ``record`` carries: what
    :func:`_plain` gives for the reading of the same result."""
    if isinstance(stored, list):
        return [_as_read(item, record) for item in stored]
    if not isinstance(stored, Mapping):
        return stored
    return {
        name: _as_read(stored.get(name), nested) if (nested := _RECORD_OF.get((record, name))) else stored.get(name)
        for name in _attributes(record)
    }


def result_evidence(result: ResultView) -> dict[str, Any]:
    """One result's evidence: the :data:`RESULT_EVIDENCE` attributes of its reading, each usage row without
    :data:`USAGE_NOT_EVIDENCE`.

    A store hands the assembly each result as its reading. A view that is the stored result itself is read as its
    dump, cut to what its reading carries, so either gives the same evidence
    (``tests/test_evidence_fingerprint.py``).

    :param result: A result as the store returned it, before any lens reads it.
    :ptype result: ResultView
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    if isinstance(result, ResultReading):
        evidence = {name: _plain(getattr(result, name)) for name in RESULT_EVIDENCE}
    else:
        stored = cast("EvalResult", result).model_dump(mode="json")
        evidence = {
            name: _as_read(stored.get(name), record) if (record := _RECORD_OF.get((None, name))) else stored.get(name)
            for name in RESULT_EVIDENCE
        }
    evidence["usage"] = unpresented(evidence["usage"] or [], USAGE_NOT_EVIDENCE)
    # The dimensions the judge could not tell, without its notes: held as a mapping's keys, whose order a store does
    # not keep.
    evidence["judge_cannot_tell"] = sorted(evidence["judge_cannot_tell"] or [])
    return evidence


class ResultsDigest:
    """The digest of the results an assembly reads, taken as each is read, whatever order the store hands them in.

    Each result's evidence is rendered and hashed on its own (:func:`result_evidence`), and :meth:`digest` hashes
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
        rendered = input_json(result_evidence(result)).encode("utf-8")
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


def run_evidence(run: EvalRun) -> dict[str, Any]:
    """One member run's evidence: its :data:`RUN_EVIDENCE` fields as stored, each varied lever by its content hash.

    :param run: The stored run.
    :ptype run: EvalRun
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    evidence = _stored(run.to_dict(), RUN_EVIDENCE)
    if isinstance(levers := evidence.get("variant_levers"), Mapping):
        evidence["variant_levers"] = unnamed_levels(levers)
    return evidence


def declaration_evidence(view: AspectView) -> dict[str, Any]:
    """What was asked: the campaign and aspect, and the aspect's declaration without how a page prints it.

    :param view: The aspect as a per-aspect reader reads it.
    :ptype view: AspectView
    :return: The evidence, as plain JSON.
    :rtype: dict[str, Any]
    """
    declared: dict[str, Any] | None = None
    if view.declared is not None:
        declared = _stored(view.declared.to_dict(), _DECLARATION_EVIDENCE)
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
    host_declarations: str,
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
    """The manifest of one assembly: what it read from the store, each kind digested (see the module).

    :param view: The aspect assembled.
    :ptype view: AspectView
    :param host_declarations: The bundle's ``host_declarations_digest``.
    :ptype host_declarations: str
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
    :param insights: The prior insights the bundle carries.
    :ptype insights: Sequence[EvalInsight]
    :param retracted_insights: Each retracted insight's id, to the archived analysis that minted it.
    :ptype retracted_insights: Mapping[str, str]
    :return: The manifest.
    :rtype: EvidenceManifest
    """
    profiles = _consulted_profiles([tier.model_dump(mode="json") for tier in tiers])
    return EvidenceManifest(
        declaration=input_digest(declaration_evidence(view)),
        host_declarations=host_declarations,
        runs=input_digest(
            {
                "read": sorted((run_evidence(run) for run in runs), key=lambda run: run["id"]),
                "archived": sorted(archived_run_ids),
                "unresolved": sorted(unresolved_run_ids),
            }
        ),
        n_runs=len(runs),
        results=results.digest(),
        n_results=len(results),
        cases=input_digest(sorted(([case.id, case.stratum] for case in cases), key=lambda case: case[0])),
        n_cases=len(cases),
        ratings=input_digest(
            sorted((_stored(rating.to_dict(), RATING_EVIDENCE) for rating in ratings), key=lambda rating: rating["id"])
        ),
        n_ratings=len(ratings),
        judge_profiles=input_digest(profiles),
        n_judge_profiles=len(profiles),
        insights=input_digest(
            {
                "carried": sorted(
                    (_stored(insight.to_dict(), INSIGHT_EVIDENCE) for insight in insights),
                    key=lambda insight: insight["id"],
                ),
                "retracted": sorted(retracted_insights.items()),
            }
        ),
        n_insights=len(insights),
    )
