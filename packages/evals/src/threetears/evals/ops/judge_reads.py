"""What a second judge's pass says, read off the stored results: the shape a judge job hands back when it ends.

Apart from :mod:`threetears.evals.ops.judge_jobs`, which starts the job, because the job contract
(:mod:`threetears.evals.ops.jobs`) reads a finished judge job back into this shape, and the start needs the job
contract: the read sits below both.
"""

from __future__ import annotations

from pydantic import Field

from threetears.evals.analysis.agreement import InterJudgeAgreement, inter_judge_agreement
from threetears.evals.analysis.judge_drift import JudgeDrift, judge_drift
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.judge_second import SecondJudgeReport
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.schema.base import EvalBaseModel


class SecondJudgeRead(EvalBaseModel):
    """A second judge's pass over a run, and what its pairs say: agreement between the judges, and drift.

    Both readings are over this pass's pairs only, read off the stored records by the code every other surface reads
    them with (:func:`~threetears.evals.analysis.inter_judge_agreement`, :func:`~threetears.evals.analysis.judge_drift`).
    """

    report: SecondJudgeReport = Field(description="What was asked, what was left out, and what it cost.")
    agreement: InterJudgeAgreement = Field(
        description="Per-dimension agreement between the run's judge and the second."
    )
    drift: JudgeDrift = Field(description="Per-dimension movement from the run's judge to the second, with intervals.")


async def read_second_judge(host: EvalHost, report: SecondJudgeReport, scope_id: str) -> SecondJudgeRead:
    """Read a second judge's pass off its run's stored results: the agreement and drift of its pairs alone.

    Args:
        host: The host whose store holds the run's results.
        report: The pass, as its work reported it.
        scope_id: The scope the run lives in.

    Returns:
        The pass, with the agreement and drift its recorded pairs read.
    """
    results = await run_blocking(
        host.blocking_executor, host.storage.query_eval_results_by_run, report.run_id, scope_id
    )
    return SecondJudgeRead(
        report=report,
        agreement=inter_judge_agreement(results, pass_id=report.pass_id),
        drift=judge_drift(results, pass_id=report.pass_id),
    )


__all__ = ["SecondJudgeRead", "read_second_judge"]
