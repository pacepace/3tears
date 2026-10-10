"""Recording and listing stored judge profiles: how a judge campaign's measurement reaches other campaigns (#628).

A judge campaign's readout (:func:`~threetears.evals.analysis.judge_kind_readings`) lives in that campaign alone.
:func:`judge_profiles_record` stores it — one :class:`~threetears.evals.kernel.EvalJudgeProfile` per judge and
criterion — from the runs the operator names, so every campaign the same judge scores reads its evidence tier from
it where its own evidence decides none (:func:`~threetears.evals.analysis.tiers_with_judge_profiles`).

**An explicit step, never a write on completion.** Which runs make up one measurement is the operator's call: a
campaign's arms are several runs, a re-run of one arm may be meant to replace it, and a run abandoned halfway should
not overwrite a full measurement. A recording replaces the stored profile of the same judge and criterion, and its
receipt says when the replaced one had been measured.
"""

from __future__ import annotations

from pydantic import Field

from threetears.evals.analysis.judge_kind_readings import UnreadJudgeTrial, judge_kind_readings
from threetears.evals.analysis.judge_profiles import SkippedJudgeProfile, judge_profiles_of
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.evidence_tiers import JudgedEvidenceTier
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.judge_cases import JUDGE_KIND
from threetears.evals.kernel.judge_profiles import EvalJudgeProfile
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import NON_TERMINAL_RUN_STATUSES, EvalResult

__all__ = [
    "JudgeProfileEntry",
    "JudgeProfileListing",
    "JudgeProfileRecording",
    "JudgeProfilesRecord",
    "judge_profiles_list",
    "judge_profiles_record",
]


class JudgeProfilesRecord(EvalBaseModel):
    """What recording judge profiles names: the judge campaign's runs whose trials make up the measurement.

    The one declaration of a recording's arguments; the ``judge_profiles_record`` action's parameters derive from it.
    """

    judge_run_ids: list[str] = Field(
        min_length=1,
        description=(
            "The finished judge-kind runs whose trials make up the measurement — every arm of the judge campaign, as "
            "runs_list names them. Each judge and criterion they measured is recorded from all of them together."
        ),
    )


class JudgeProfileEntry(EvalBaseModel):
    """One stored profile, with the tier its own measures decide."""

    profile: EvalJudgeProfile
    tier: JudgedEvidenceTier = Field(
        description="The tier the profile's measures decide; only a decided one is ever read by another campaign."
    )
    replaced_measured_at: str | None = Field(
        default=None,
        description="On a recording: when the profile this one replaced had been measured; None when none was stored.",
    )


class JudgeProfileRecording(EvalBaseModel):
    """What a recording stored, and what it could not."""

    profiles: list[JudgeProfileEntry] = Field(description="Each profile written, one per judge and criterion.")
    skipped: list[SkippedJudgeProfile] = Field(
        default_factory=list, description="Judges and criteria measured but not recorded, with why."
    )
    unread: list[UnreadJudgeTrial] = Field(
        default_factory=list, description="Results of the runs the readout left out, with why."
    )


class JudgeProfileListing(EvalBaseModel):
    """The stored judge profiles of a scope."""

    profiles: list[JudgeProfileEntry]


def judge_profiles_record(host: EvalHost, record: JudgeProfilesRecord, scope_id: str) -> JudgeProfileRecording:
    """Record the profile of every judge and criterion a judge campaign's runs measured.

    Args:
        host: The host whose store holds the runs and receives the profiles.
        record: The runs.
        scope_id: The scope the runs live in, and the profiles are stored in.

    Returns:
        The receipt: each profile written (with the tier it decides and when any profile it replaced had been
        measured), every judge measured but not recorded, and every result the readout left out.

    Raises:
        NotFoundError: A run is not in the scope.
        ValidationFailedError: A run is not of the judge kind, or is still running; or no profile could be recorded.
        StorageError: A profile could not be persisted.
    """
    results: list[EvalResult] = []
    for run_id in dict.fromkeys(record.judge_run_ids):
        run = host.storage.load_eval_run(run_id, scope_id)
        if run is None:
            raise NotFoundError("run", run_id)
        if run.candidate_kind != JUDGE_KIND:
            raise ValidationFailedError(
                f"run {run_id!r} is of kind {run.candidate_kind!r}, not {JUDGE_KIND!r}; a judge profile is recorded "
                "from a judge campaign's runs, whose trials measured the judge as their subject"
            )
        if run.status in NON_TERMINAL_RUN_STATUSES:
            raise ValidationFailedError(f"run {run_id!r} is {run.status} — its trials are still being written")
        results.extend(host.storage.query_eval_results_by_run(run.id, scope_id))
    readings = judge_kind_readings(results)
    drafts = judge_profiles_of(readings, scope_id=scope_id)
    if not drafts.profiles:
        why = "; ".join(
            [f"{skip.rubric_dim}: {skip.reason}" for skip in drafts.skipped]
            + [f"{unread.result_id}: {unread.reason}" for unread in readings.unread]
        )
        raise ValidationFailedError(f"no judge profile could be recorded — {why or 'the runs have no results'}")
    entries = []
    for profile in drafts.profiles:
        replaced = host.storage.load_judge_profile(profile.id, scope_id)
        host.storage.save_judge_profile(profile)
        entries.append(
            JudgeProfileEntry(
                profile=profile,
                tier=profile.tier,
                replaced_measured_at=None if replaced is None else replaced.measured_at,
            )
        )
    return JudgeProfileRecording(profiles=entries, skipped=drafts.skipped, unread=readings.unread)


def judge_profiles_list(host: EvalHost, scope_id: str, *, rubric_dim: str | None = None) -> JudgeProfileListing:
    """The stored judge profiles of a scope, each with the tier it decides.

    Args:
        host: The host whose store is read.
        scope_id: The scope.
        rubric_dim: Only the profiles of this criterion's dim.

    Returns:
        The listing, oldest recording first.
    """
    return JudgeProfileListing(
        profiles=[
            JudgeProfileEntry(profile=profile, tier=profile.tier)
            for profile in host.storage.query_judge_profiles(scope_id, rubric_dim=rubric_dim)
        ]
    )
