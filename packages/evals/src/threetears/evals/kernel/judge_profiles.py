"""A judge's stored profile: what a judge campaign measured one judge to be on one criterion, for others to read (#628).

A judge campaign (:mod:`threetears.evals.kernel.judge_cases`) measures a judge configuration as its subject, and its
readout (:func:`~threetears.evals.analysis.judge_kind_readings`) says, per judge and criterion, how far the judge
agreed with people's labels, with its own repeats, and how often its replies kept the protocol. That readout lives
in the judge campaign alone. :class:`EvalJudgeProfile` is the record that outlives it: one stored document per
(judge, criterion) — the judge as :class:`~threetears.evals.analysis.JudgeKey` keys one (dim, scale, served model,
config, temperature), the criterion as the cases worded it — holding the measures, the cases they were measured on
and when, so a campaign scored by that same judge reads its evidence tier from it where its own evidence decides
none (:func:`~threetears.evals.analysis.tiers_with_judge_profiles`).

**Recorded on purpose, never on completion.** A profile is written by an explicit step over the judge campaign's
runs (:func:`~threetears.evals.analysis.judge_profiles_of` and the ``judge_profiles_record`` operation), because
which runs make up one measurement is the operator's call: a campaign's arms are several runs, a re-run of one arm
may be meant to replace it, and a run abandoned halfway should not overwrite a full measurement.

**One per judge and criterion, replaced by the next recording.** The id derives from the judge and the criterion,
so recording again over a newer campaign replaces the profile, and the record always says what it was measured on.

**A profile of another judge is never read.** Every field of the judge is part of its key, so a changed model,
config (a new prompt is a new versioned config) or temperature, or a reworded criterion, is a different judge whose
profile does not exist — the old one is stale, and nothing reads it for the new judge. A judge whose served model or
temperature was not recorded cannot be shown to be the judge in use anywhere else, so no profile is recorded for it.

**Regenerable.** A profile is derived from stored runs and results, so it follows the regenerable documents' rule
(:mod:`threetears.evals.schema.versioning`): written under :data:`REGENERABLE_SCHEMA_VERSION`, refused at another,
and recorded again from its runs.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import Field, field_validator, model_validator

from threetears.evals.kernel.evidence_tiers import (
    JudgedEvidenceTier,
    TierCriterion,
    calibration_criterion,
    separation_criterion,
    tier_of,
)
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    DimName,
    JudgeTemperature,
    RubricScale,
    SchemaVersion,
    utc_now_iso,
)
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

__all__ = [
    "EvalJudgeProfile",
    "JudgeProfileAgreement",
    "JudgeProfileStore",
    "judge_profile_id",
]


def judge_profile_id(
    rubric_dim: str,
    scale: str,
    judge_model: str,
    judge_config_id: str | None,
    judge_temperature: JudgeTemperature,
    criterion_digest: str,
) -> str:
    """The id of the one profile of a judge on a criterion: derived from both, so recording again replaces it.

    Args:
        rubric_dim: The criterion's dim.
        scale: Its scale.
        judge_model: The model that served the judge's replies.
        judge_config_id: The versioned config that asked; None for the built-in prompt.
        judge_temperature: What the judge's calls were sent at.
        criterion_digest: The criterion's wording (:func:`~threetears.evals.kernel.judge_criterion_digest`).

    Returns:
        ``judge-profile:`` and the digest of the six.
    """
    return "judge-profile:" + canonical_digest(
        [rubric_dim, scale, judge_model, judge_config_id, judge_temperature, criterion_digest]
    )


class JudgeProfileAgreement(EvalDocumentModel):
    """One of a profile's two agreements, as the evidence tiers read it: the figure, its bounds, and what it counts."""

    n: int = Field(ge=0, description="The pairs the agreement was read over.")
    results: int = Field(ge=0, description="The distinct cases among the raters whose kappa entered the figure.")
    exact_agreement: float = Field(ge=0.0, le=1.0, description="The share of pairs that gave the same score.")
    agreement: float | None = Field(
        description="Weighted kappa on 1-5, kappa on pass/fail, pooled by case; None when undefined."
    )
    lower: float | None = Field(default=None, description="The one-sided 95% lower bound on `agreement`.")
    upper: float | None = Field(default=None, description="The one-sided 97.5% upper bound on `agreement`.")

    @property
    def interval(self) -> tuple[float, float] | None:
        """``(lower, upper)``, or None when the agreement carries no bounds."""
        if self.lower is None or self.upper is None:
            return None
        return (self.lower, self.upper)


class EvalJudgeProfile(EvalDocumentModel):
    """What one judge was measured to be on one criterion, over frozen cases, by a judge campaign — stored for reuse.

    One per (``rubric_dim``, ``scale``, ``judge_model``, ``judge_config_id``, ``judge_temperature``,
    ``criterion_digest``): the ``id`` derives from them, and a stored id that disagrees with its own fields is refused.
    """

    doc_type: Literal["eval_judge_profile"] = "eval_judge_profile"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1, description="The scope the judge campaign's runs live in.")

    rubric_dim: DimName = Field(description="The criterion's dim.")
    scale: RubricScale = Field(description="The scale it was asked on.")
    judge_model: str = Field(min_length=1, description="The model the judge's replies named as having answered.")
    judge_config_id: str | None = Field(description="The versioned JudgeConfig that asked; None = the built-in prompt.")
    judge_temperature: JudgeTemperature = Field(
        description="What the judge's calls were sent at ('model_default' = sent none, the model refusing one)."
    )
    criterion_digest: str = Field(min_length=1, description="The criterion as the cases worded it (its digest).")
    id: str = Field(
        default_factory=lambda data: _derived_profile_id(data),
        description="Derived from the judge and the criterion: one profile per judge per criterion.",
    )

    cases: int = Field(ge=0, description="Distinct frozen cases the judge replied to on this criterion.")
    trials: int = Field(ge=0, description="Trials read: every reply, a case's repeats included.")
    case_set_fingerprint: str = Field(
        min_length=1, description="A digest of the cases measured on — each case's id and content digest, sorted."
    )
    run_ids: list[str] = Field(min_length=1, description="The judge campaign's runs the trials came from, sorted.")
    label_agreement: JudgeProfileAgreement | None = Field(
        description="Agreement with the cases' person labels; None when no labelled case was scored."
    )
    self_agreement: JudgeProfileAgreement | None = Field(
        description="Agreement of each case's later answers with its first; None when no case was answered twice."
    )
    parse_replies: int = Field(ge=0, description="Replies read: scored, can't tell and invalid.")
    parse_valid: int = Field(ge=0, description="Replies that kept the protocol: a score on the scale, or can't tell.")
    parse_validity: float | None = Field(
        ge=0.0, le=1.0, description="parse_valid / parse_replies; None when there was no reply to read."
    )
    measured_at: str = Field(description="When the measurement was taken: the latest scored_at among its trials.")
    recorded_at: str = Field(default_factory=utc_now_iso, description="When this profile was recorded.")

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_judge_profile":
            raise ValueError(f"doc_type must be 'eval_judge_profile', got '{v}'")
        return v

    @model_validator(mode="after")
    def _id_derived(self) -> EvalJudgeProfile:
        """Refuse an id that is not the one the judge and criterion derive, and more valid replies than replies."""
        derived = judge_profile_id(
            self.rubric_dim,
            self.scale,
            self.judge_model,
            self.judge_config_id,
            self.judge_temperature,
            self.criterion_digest,
        )
        if self.id != derived:
            raise ValueError(
                f"id {self.id!r} is not the one the judge and criterion derive ({derived!r}): a profile under another "
                "id would stand beside the judge's profile instead of replacing it"
            )
        if self.parse_valid > self.parse_replies:
            raise ValueError(f"{self.parse_valid} valid replies out of {self.parse_replies} replies")
        return self

    def criteria(self) -> tuple[TierCriterion, TierCriterion]:
        """The two criteria the profile's agreements read as, held to the ruled bars like a campaign's own.

        Returns:
            ``(calibration, separation)``: agreement with the labels, and with the judge's own repeats.
        """
        people, itself = self.label_agreement, self.self_agreement
        calibration = calibration_criterion(
            people.n if people else 0,
            people.results if people else 0,
            people.agreement if people else None,
            people.interval if people else None,
        )
        separation = separation_criterion(
            itself.n if itself else 0,
            itself.results if itself else 0,
            itself.agreement if itself else None,
            itself.interval if itself else None,
        )
        return calibration, separation

    @property
    def tier(self) -> JudgedEvidenceTier:
        """The tier the profile's own measures decide, by the rule every judged tier is decided by."""
        return tier_of(*self.criteria())


def _derived_profile_id(data: dict[str, object]) -> str:
    """The default ``id`` of a profile, from the fields validated before it; blank when one of them failed."""
    try:
        return judge_profile_id(
            str(data["rubric_dim"]),
            str(data["scale"]),
            str(data["judge_model"]),
            data["judge_config_id"],  # type: ignore[arg-type]
            data["judge_temperature"],  # type: ignore[arg-type]
            str(data["criterion_digest"]),
        )
    except KeyError:
        return ""


class JudgeProfileStore(Protocol):
    """The writes and reads of stored judge profiles.

    Structural, so :class:`~threetears.evals.kernel.storage.EvalStorage` satisfies it by having the methods.
    """

    def save_judge_profile(self, profile: EvalJudgeProfile, /) -> None:
        """Write one profile, replacing the stored profile of the same judge and criterion."""
        ...

    def load_judge_profile(self, profile_id: str, scope_id: str, /) -> EvalJudgeProfile | None:
        """The profile with this id in a scope, or None."""
        ...

    def query_judge_profiles(self, scope_id: str, /, *, rubric_dim: str | None = None) -> list[EvalJudgeProfile]:
        """Every profile in a scope, optionally of one dim, oldest recording first."""
        ...
