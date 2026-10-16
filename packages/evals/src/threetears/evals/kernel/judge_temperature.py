"""What a judge temperature comparison asked and was answered (#633): the record the run layer hands the reading.

The comparison is two steps, on either side of the package matrix. The work
:func:`~threetears.evals.run.judge_temperature_work` admits spends — it re-judges a finished run's borderline cases at
the pinned temperature and at the provider's default — and records every answer as a :class:`JudgeTemperatureAnswers`
on the comparison's record; :func:`~threetears.evals.analysis.read_judge_temperatures`
reads those answers into per-dimension variance and self-agreement, side by side. This module is the contract
between the two: the settings compared, what was selected, and each answer as it came back.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Self

from pydantic import Field, model_validator

from threetears.evals.schema.base import CoreShapeModel, EvalBaseModel
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    MODEL_DEFAULT_TEMPERATURE,
    JudgeTemperature,
    RubricScale,
    RubricScore,
)

#: The two samplings compared. ``pinned``: every call requested at ``DEFAULT_JUDGE_TEMPERATURE`` (the policy).
#: ``provider_default``: every call sent no temperature, so the provider's own default applies.
TemperatureSetting = Literal["pinned", "provider_default"]

#: Which scored dims are re-judged: ``borderline`` (stored score inside its scale, or a recorded repeat or second
#: judge disagreed on it — :func:`~threetears.evals.run.borderline_dims`) or ``all``.
TemperatureSelection = Literal["borderline", "all"]

#: How many times each dim is judged at each setting when the caller names no number.
DEFAULT_TEMPERATURE_REPEATS = 5

#: The fewest repeats per setting: one answer has no variance and nothing to agree with.
MIN_TEMPERATURE_REPEATS = 2


class TemperatureSettingSpec(NamedTuple):
    """One setting compared: its name, what each call is requested at, and what its answers must record."""

    setting: TemperatureSetting
    #: The temperature each call is requested at; ``None`` = sent none.
    requested: float | None
    #: What a score answered under this setting records when the client honoured it.
    recorded: JudgeTemperature


#: The settings, in the order a call round makes them.
TEMPERATURE_SETTINGS: tuple[TemperatureSettingSpec, ...] = (
    TemperatureSettingSpec("pinned", DEFAULT_JUDGE_TEMPERATURE, DEFAULT_JUDGE_TEMPERATURE),
    TemperatureSettingSpec("provider_default", None, MODEL_DEFAULT_TEMPERATURE),
)


class TemperatureSkip(EvalBaseModel, CoreShapeModel):
    """A result of the run the comparison did not re-judge, and why."""

    result_id: str
    reason: str


class TemperatureAnswer(EvalBaseModel, CoreShapeModel):
    """One judge call's answer: a score, a "can't tell", or a failure."""

    score: RubricScore | None = Field(default=None, description="The score, as the judge gave it and recorded it.")
    config_id: str | None = Field(
        default=None,
        description=(
            "The versioned JudgeConfig that asked, or a ``builtin:`` id = the built-in prompt, by its wording; None "
            "only for a call that raised before the judge could name either. A score always names it."
        ),
    )
    cannot_tell: str | None = Field(default=None, description="The judge's reason, when it could not tell.")
    error: str | None = Field(default=None, description="Why the call failed, when it did.")

    @model_validator(mode="after")
    def _a_score_names_its_judge(self) -> Self:
        """Refuse a score with no judge identity: it could not be told from another judge's."""
        if self.score is not None and self.config_id is None:
            raise ValueError("a scored temperature answer names the judge config that asked")
        return self


class TemperatureCaseAnswers(EvalBaseModel, CoreShapeModel):
    """Every answer one result's dimension got at one setting, in call order."""

    result_id: str
    rubric_dim: str
    scale: RubricScale
    stored_score: int = Field(description="The score the run's judge stored, which made the case borderline or not.")
    setting: TemperatureSetting
    answers: list[TemperatureAnswer]


class JudgeTemperatureAnswers(EvalBaseModel, CoreShapeModel):
    """What a judge temperature comparison asked, was answered and spent — before anything is read off it.

    :ivar run_id: The run.
    :ivar judge_model: The run's judge pin.
    :ivar selection: Which scored dims were re-judged.
    :ivar repeats: Calls per dim per setting.
    :ivar results: The results re-judged.
    :ivar cases: The (result, dim) pairs re-judged.
    :ivar answers: Per case and setting, every answer in call order.
    :ivar skipped: The run's results not re-judged, each with why — decided before anything was spent.
    :ivar stopped: Why it stopped before its last result, when it did.
    :ivar calls_made: Judge calls made, ledgered under purpose ``judge``, and one whose ledger row could not be written
        counted all the same.
    :ivar cost_usd: What they cost together; ``None`` when any went unpriced.
    :ivar cap_usd: The out-of-run cap they were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    judge_model: str
    selection: TemperatureSelection
    repeats: int
    results: int
    cases: int
    answers: list[TemperatureCaseAnswers]
    skipped: list[TemperatureSkip]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None


__all__ = [
    "DEFAULT_TEMPERATURE_REPEATS",
    "MIN_TEMPERATURE_REPEATS",
    "TEMPERATURE_SETTINGS",
    "JudgeTemperatureAnswers",
    "TemperatureAnswer",
    "TemperatureCaseAnswers",
    "TemperatureSelection",
    "TemperatureSetting",
    "TemperatureSettingSpec",
    "TemperatureSkip",
]
