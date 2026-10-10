"""A judge job's record: a second judge's pass, a drift check or a temperature comparison run as a job, and how it ended.

The three re-judge a finished run's stored evidence, and each makes many judge calls: a result's every scored dim
for a second judge, every selected dim ``repeats`` times at each of two temperatures for a comparison. A host that
budgets a call's wall time by the model requests it makes while its caller waits cannot hold them inside one call,
so each is started as a job (:mod:`threetears.evals.ops.jobs`): every refusal is decided in the call, before
anything is spent, and only the judge calls run in the job.

:class:`EvalJudgeJob` is the record that job writes and its ``judge:<id>`` job id names. It is written when the job
starts, after each result is asked (its progress), and when it ends, however it ends — with what the work handed
back: the second judge's :class:`~threetears.evals.kernel.SecondJudgeReport` (whose second scores are recorded on
the results themselves, the pass's id naming them), or the comparison's
:class:`~threetears.evals.kernel.JudgeTemperatureAnswers`, which are stored nowhere else. Its job reads the
agreement, drift or comparison off that record by the same code the work's own readers use, so a job polled after
it ended answers what the call would have returned. One writer, the job's own task, so a write never races another.
A record reading ``running`` while no task runs it was ended by a restart, which its job reports as lost.

**Regenerable.** It follows the regenerable documents' rule (:mod:`threetears.evals.schema.versioning`): written
under :data:`REGENERABLE_SCHEMA_VERSION`, refused at another. The second judge's scores outlive it on the results;
a comparison's answers are measured again, so a caller keeps the comparison it read.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import Field, field_validator, model_validator

from threetears.evals.kernel.judge_second import SecondJudgeReport
from threetears.evals.kernel.judge_temperature import JudgeTemperatureAnswers
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import SchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import REGENERABLE_SCHEMA_VERSION

__all__ = [
    "EvalJudgeJob",
    "JudgeJobAction",
    "JudgeJobOutcome",
]

#: What a judge job does: a second judge over a seeded share (``judge_second``), a second judge over every result
#: under a changed judge (``judge_drift_check``), or the temperature comparison (``judge_temperature``).
JudgeJobAction = Literal["judge_second", "judge_drift_check", "judge_temperature"]

#: Where a judge job stands: ``running``; ``completed`` — every admitted call was made, or the account behind the
#: judge refused one and the work stopped as it does in a call; ``failed`` — the work raised (its in-job admission
#: refused, say, because the run changed after the call admitted it); ``cancelled`` — a cancel stopped it, its
#: partial report kept.
JudgeJobOutcome = Literal["running", "completed", "failed", "cancelled"]


class EvalJudgeJob(EvalDocumentModel):
    """A judge job's own record: what it re-judges, how far it got, how it ended, and what it handed back."""

    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    doc_type: Literal["eval_judge_job"] = "eval_judge_job"
    schema_version: SchemaVersion = REGENERABLE_SCHEMA_VERSION
    scope_id: str = Field(min_length=1)

    action: JudgeJobAction = Field(description="The action that started it.")
    run_id: str = Field(min_length=1, description="The finished run whose stored evidence is re-judged.")
    results_total: int = Field(ge=1, description="The results it was admitted to ask about.")
    results_done: int = Field(default=0, ge=0, description="The results asked about so far.")
    max_calls: int = Field(ge=1, description="The most judge calls it was admitted to make, parse retries included.")
    calls_made: int = Field(default=0, ge=0, description="The judge calls made so far, as the ledger recorded them.")
    outcome: JudgeJobOutcome = "running"
    detail: str | None = Field(default=None, description="Why it ended other than completed.")
    second_judge: SecondJudgeReport | None = Field(
        default=None, description="A second judge's report so far, then as it ended; None for a comparison."
    )
    temperature_answers: JudgeTemperatureAnswers | None = Field(
        default=None, description="A comparison's answers so far, then as it ended; None for a second judge."
    )
    created_at: str = Field(default_factory=utc_now_iso)
    ended_at: str | None = None

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_judge_job":
            raise ValueError(f"doc_type must be 'eval_judge_job', got '{v}'")
        return v

    @model_validator(mode="after")
    def check_report_matches_action(self) -> EvalJudgeJob:
        """Refuse a record carrying the other action's report.

        Returns:
            This record, unchanged.

        Raises:
            ValueError: A comparison carrying a second judge's report, or a second judge carrying answers.
        """
        if self.action == "judge_temperature" and self.second_judge is not None:
            raise ValueError("a temperature comparison's record carries its answers, never a second judge's report")
        if self.action != "judge_temperature" and self.temperature_answers is not None:
            raise ValueError(f"a {self.action} record carries a second judge's report, never temperature answers")
        return self
