"""Stored judge profiles: recorded from a judge campaign's readout, read by every campaign that judge scores (#628).

Two halves, both pure:

- :func:`judge_profiles_of` — the profiles a judge campaign's readout (:func:`judge_kind_readings`) records: one
  :class:`~threetears.evals.kernel.EvalJudgeProfile` per (judge, criterion), carrying its two agreements as the
  evidence tiers read them, parse validity, the cases and runs it was measured on, and when. A reading whose judge's
  served model or temperature was not recorded is skipped with why: an unobserved judge cannot be shown to be the
  judge another campaign used.
- :func:`tiers_with_judge_profiles` — a campaign's evidence tiers, each one its own evidence left ``undetermined``
  replaced by the stored profile of the very same judge and criterion when that profile decides a tier. Which
  criterion a campaign's judge was asked is read off the template its runs were judged against
  (:func:`judged_criteria`).

**When a campaign reads a profile — "thinner", precisely.** A campaign's own evidence about the judge that scored
it is the first evidence: it was measured on the campaign's own outputs. A stored profile is read in its place for a
judge only when BOTH hold:

1. the campaign's own evidence decides no tier for that judge (its tier is ``undetermined`` — each criterion short
   of its floor, or its bounds across the bar), and
2. the profile's evidence decides one (``calibrated``, ``separation`` or ``incidental``), by the same rule, bars and
   floors as the campaign's own (distinct frozen cases counting as the campaign's distinct results).

A campaign whose own ratings or repeats decided a tier keeps it, even against a profile measured on more cases; a
profile that decides nothing either is not read. The tier read is flagged on the entry
(:class:`~threetears.evals.kernel.JudgeTierFromProfile`): which profile, when and on which cases it was measured,
and the campaign's own two criteria it was read in place of — never presented as the campaign's own.

**A profile of another judge is never read.** The profile is looked up by its id, which derives from the whole judge
(dim, scale, served model, config, temperature) and the criterion's wording, so a judge whose model, prompt (a new
versioned config) or temperature changed, or a criterion reworded since, finds no profile: the old one is stale. A
campaign whose criterion cannot be known — a run with no template, a template edited after the run was judged, a
rubric whose scale moved, or runs of one judge asked two wordings — reads no profile for that judge.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from threetears.evals.analysis.agreement import DimensionAgreement, JudgeKey, SelfAgreementDimension, judge_key
from threetears.evals.analysis.judge_kind_readings import JudgeKindReading, JudgeKindReadings
from threetears.evals.kernel.evidence_tiers import (
    JudgeEvidenceTier,
    JudgeTierFromProfile,
    agreement_statistic,
    tier_of,
)
from threetears.evals.kernel.judge_cases import judge_criterion_digest
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile, JudgeProfileAgreement, judge_profile_id
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID, utc_now_iso

if TYPE_CHECKING:
    from threetears.evals.schema.models import EvalResult, EvalRun, EvalTemplate

__all__ = [
    "JudgeProfileDrafts",
    "SkippedJudgeProfile",
    "judge_profiles_of",
    "judged_criteria",
    "tiers_with_judge_profiles",
]


class SkippedJudgeProfile(EvalDocumentModel):
    """A judge campaign reading no profile was recorded for, and why."""

    rubric_dim: str = Field(min_length=1)
    judge_model: str | None
    judge_config_id: str | None
    criterion_digest: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class JudgeProfileDrafts(EvalDocumentModel):
    """The profiles a judge campaign's readout records, and the readings it could not record."""

    profiles: list[EvalJudgeProfile] = Field(default_factory=list, description="One per (judge, criterion).")
    skipped: list[SkippedJudgeProfile] = Field(default_factory=list)


def _agreement(row: DimensionAgreement | SelfAgreementDimension | None) -> JudgeProfileAgreement | None:
    """One agreement row as a profile holds it: the figure the tiers are held to, its bounds and its counts."""
    if row is None:
        return None
    interval = row.agreement_interval
    return JudgeProfileAgreement(
        n=row.n,
        results=row.results,
        exact_agreement=row.exact_agreement,
        agreement=agreement_statistic(row.scale, row.kappa, row.weighted_kappa),
        lower=None if interval is None else interval[0],
        upper=None if interval is None else interval[1],
    )


def _profile_of(reading: JudgeKindReading, *, scope_id: str, recorded_at: str) -> EvalJudgeProfile:
    """One reading as its stored profile; the caller has checked the judge's model and temperature were recorded."""
    assert reading.judge_model is not None and reading.judge_temperature is not None
    validity = reading.parse_validity
    return EvalJudgeProfile(
        scope_id=scope_id,
        rubric_dim=reading.rubric_dim,
        scale=reading.scale,
        judge_model=reading.judge_model,
        judge_config_id=reading.judge_config_id,
        judge_temperature=reading.judge_temperature,
        criterion_digest=reading.criterion_digest,
        cases=reading.cases,
        trials=reading.trials,
        case_set_fingerprint=reading.case_set_fingerprint,
        run_ids=list(reading.run_ids),
        label_agreement=_agreement(reading.label_agreement),
        self_agreement=_agreement(reading.self_agreement),
        parse_replies=validity.replies,
        parse_valid=validity.scored + validity.cannot_tell,
        parse_validity=validity.rate,
        measured_at=reading.measured_at,
        recorded_at=recorded_at,
    )


def judge_profiles_of(
    readings: JudgeKindReadings, *, scope_id: str, recorded_at: str | None = None
) -> JudgeProfileDrafts:
    """The profiles a judge campaign's readout records: one per judge and criterion whose judge can be named.

    Args:
        readings: The judge campaign's readout (:func:`~threetears.evals.analysis.judge_kind_readings`).
        scope_id: The scope the campaign's runs live in, which the profiles are stored in.
        recorded_at: When the profiles are recorded; now when omitted.

    Returns:
        The profiles, and every reading skipped with why: a judge whose responses named no served model, or whose
        calls' temperature was not recorded, is a judge nothing could match to the judge another campaign used.
    """
    stamp = recorded_at or utc_now_iso()
    drafts = JudgeProfileDrafts()
    for reading in readings.readings:
        missing = [
            what
            for what, value in (
                ("its responses named no served model", reading.judge_model),
                ("its calls' temperature was not recorded", reading.judge_temperature),
            )
            if value is None
        ]
        if missing:
            drafts.skipped.append(
                SkippedJudgeProfile(
                    rubric_dim=reading.rubric_dim,
                    judge_model=reading.judge_model,
                    judge_config_id=reading.judge_config_id,
                    criterion_digest=reading.criterion_digest,
                    reason=(
                        " and ".join(missing)
                        + ", so no other campaign's judge could be shown to be this one; no profile is recorded"
                    ),
                )
            )
            continue
        drafts.profiles.append(_profile_of(reading, scope_id=scope_id, recorded_at=stamp))
    return drafts


def _criterion_of(template: EvalTemplate, key: JudgeKey) -> str | None:
    """The digest of the criterion ``key``'s judge was asked under ``template``, or None when it cannot be told."""
    if key.rubric_dim in (TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID):
        # A reserved axis's criterion is the engine's own, asked on 1-5 (as a judge case freezes it).
        return judge_criterion_digest(key.rubric_dim, key.scale, None) if key.scale == "ordinal" else None
    criteria = [dim for dim in template.rubric if dim.name == key.rubric_dim]
    if len(criteria) != 1 or criteria[0].scale != key.scale:
        return None  # not in the rubric, named twice, or moved to another scale since the score was given
    return judge_criterion_digest(key.rubric_dim, key.scale, criteria[0])


def judged_criteria(
    runs: Sequence[EvalRun],
    results_by_run: Mapping[str, Sequence[EvalResult]],
    templates: Mapping[str, EvalTemplate],
) -> dict[JudgeKey, str | None]:
    """The criterion each judge behind a campaign's judged scores was asked, as the digest a profile is keyed by.

    Read off the template each run was judged against, under the rule a re-judge reads it by: a run with no
    template, or whose template was edited after the run was created, read a wording nothing records, so its
    judges' criterion is unknown. A judge whose scores were asked two wordings across the runs is unknown too.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results.
        templates: The runs' templates by id, as loaded; a template absent here did not load.

    Returns:
        Every judge behind a judged score, mapped to its criterion's digest, or None when it cannot be told.
    """
    criteria: dict[JudgeKey, str | None] = {}
    for run in runs:
        template = templates.get(run.template_id) if run.template_id else None
        readable = template is not None and template.updated_at <= run.created_at
        for result in results_by_run.get(run.id, ()):
            for score in (*result.rubric_scores, result.transcript_score, result.outcome_score):
                if score is None or (key := judge_key(result, score.dim)) is None:
                    continue
                digest = _criterion_of(template, key) if readable and template is not None else None
                if key in criteria and criteria[key] != digest:
                    criteria[key] = None
                else:
                    criteria.setdefault(key, digest)
    return criteria


def tiers_with_judge_profiles(
    tiers: Iterable[JudgeEvidenceTier],
    profiles: Iterable[EvalJudgeProfile],
    criteria: Mapping[JudgeKey, str | None],
) -> list[JudgeEvidenceTier]:
    """A campaign's evidence tiers, each its own evidence left undetermined read from its judge's stored profile.

    See the module docstring for when a profile is read. Every entry keeps its place and its judge; an entry read
    from a profile carries the profile's two criteria and tier, and names the profile in ``from_profile``.

    Args:
        tiers: The campaign's own tiers (:func:`~threetears.evals.analysis.judge_evidence_tiers`).
        profiles: The stored profiles of the campaign's scope.
        criteria: The criterion each judge was asked (:func:`judged_criteria`).

    Returns:
        The tiers, in the order given.
    """
    by_id = {profile.id: profile for profile in profiles}
    read: list[JudgeEvidenceTier] = []
    for tier in tiers:
        key = JudgeKey(tier.rubric_dim, tier.scale, tier.judge_model, tier.judge_config_id, tier.judge_temperature)
        digest = criteria.get(key)
        if tier.tier != "undetermined" or digest is None or key.judge_model is None or key.judge_temperature is None:
            read.append(tier)
            continue
        profile = by_id.get(
            judge_profile_id(
                key.rubric_dim, key.scale, key.judge_model, key.judge_config_id, key.judge_temperature, digest
            )
        )
        if profile is None:
            read.append(tier)
            continue
        calibration, separation = profile.criteria()
        decided = tier_of(calibration, separation)
        if decided == "undetermined":
            read.append(tier)
            continue
        # Built, never copied with an update: the entry's own validator holds the tier to the criteria it carries.
        read.append(
            JudgeEvidenceTier(
                rubric_dim=tier.rubric_dim,
                scale=tier.scale,
                judge_model=tier.judge_model,
                judge_config_id=tier.judge_config_id,
                judge_temperature=tier.judge_temperature,
                tier=decided,
                calibration=calibration,
                separation=separation,
                from_profile=JudgeTierFromProfile(
                    profile_id=profile.id,
                    criterion_digest=profile.criterion_digest,
                    measured_at=profile.measured_at,
                    recorded_at=profile.recorded_at,
                    cases=profile.cases,
                    case_set_fingerprint=profile.case_set_fingerprint,
                    run_ids=list(profile.run_ids),
                    own_calibration=tier.calibration,
                    own_separation=tier.separation,
                ),
            )
        )
    return read
