"""What a second judge's pass asked, recorded and spent (#646, #597): the report the run layer hands its readers.

:func:`~threetears.evals.run.ask_second_judge` spends — it asks a second judge about a finished run's stored
evidence and records each answer on its result — and returns a :class:`SecondJudgeReport`. What the pairs say
(agreement, drift) is read off the stored results by :mod:`threetears.evals.analysis`. The report sits here, below
both, because a judge job's record (:class:`~threetears.evals.kernel.EvalJudgeJob`) stores it: a pass started as a
job is read back from that record once its job has ended.
"""

from __future__ import annotations

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import SecondJudge


class SecondJudgeSkip(EvalBaseModel):
    """A result of the run that the second judge is not asked about, and why."""

    result_id: str
    reason: str


class SecondJudgeReport(EvalBaseModel):
    """What a second judge's pass did, what it spent apart from the candidate's cost, and what it read.

    Attributes:
        run_id: The run.
        pass_id: The pass, as every entry it recorded names it.
        judge: The second judge.
        sample_fraction: The share of the eligible results it drew.
        sample_seed: The seed it drew them with.
        eligible: The run's results it could have asked about.
        sampled: The results the sample drew, sorted by id.
        judged: The results a second score was recorded on, in the order they were asked.
        scores_paired: Dims the second judge scored.
        scores_unanswered: Dims whose call failed or answered it could not tell.
        skipped: The run's results not asked about, each with why — decided before anything was spent.
        unwritten: Results asked and paid for whose record did not land. Their spend is in the ledger.
        stopped: Why the pass stopped before its last result, when it did — the account behind the judge refused a
            call, so every later one would be refused too, or the pass was cancelled.
        calls_made: Calls made, as the ledger recorded them under purpose ``second_judge``.
        cost_usd: What they cost together — measurement cost, never the candidate's; ``None`` when any went unpriced.
        cap_usd: The out-of-run cap the calls were admitted under; ``None`` when the host enforces none.
    """

    run_id: str
    pass_id: str
    judge: SecondJudge
    sample_fraction: float
    sample_seed: int
    eligible: int
    sampled: list[str]
    judged: list[str]
    scores_paired: int
    scores_unanswered: int
    skipped: list[SecondJudgeSkip]
    unwritten: list[str]
    stopped: str | None = None
    calls_made: int
    cost_usd: float | None
    cap_usd: float | None


__all__ = [
    "SecondJudgeReport",
    "SecondJudgeSkip",
]
