"""``Answer``: a candidate's answer, with what producing it spent.

A :func:`~threetears.evals.quick.run_eval` candidate is an opaque async callable, so the engine sees what it
returns and nothing of what it spent: a candidate that calls a paid model reports a ``cost_usd`` of 0 unless
it says otherwise. Returning an :class:`Answer` is how it says so. Its ``value`` is what the scorers, the
classifier's ``expected=`` and the judge grade, exactly as a plain return value would be; its spend becomes
the cell's ``candidate`` usage row, which the engine derives the result's ``cost_usd`` from as it does for
any kind (:attr:`~threetears.evals.contracts.CandidateTelemetry.usage`). A candidate that returns anything
else reports no spend, as before.

The spend fields carry the names a completion client's reply carries, so the usage ledger reads an
``Answer`` as it reads one: **missing is not zero** — a token count or a price left ``None`` is unreported,
and an unpriced answer makes its result's cost unknown rather than quietly smaller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from threetears.evals.contracts import CandidateTelemetry, RoleUsageLedger


@dataclass(frozen=True)
class Answer:
    """What a candidate returns to report its own spend beside its answer.

    Attributes:
        value: The answer itself: what is graded, and what the run stores as the cell's output.
        model: The model the call went to, which the usage row is attributed to; ``None`` when there is none.
        input_tokens: Prompt tokens; ``None`` when unreported.
        output_tokens: Completion tokens, any thinking included; ``None`` when unreported.
        cost_usd: What the call cost in US dollars, as you priced it; ``None`` when unpriced, which the
            engine records as unknown and never as $0.
    """

    value: Any
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        """Refuse a spend no usage row can hold, so the candidate that built it fails rather than the run.

        Raises:
            ValueError: A negative or non-integer token count, or a negative or non-finite cost.
        """
        for name in ("input_tokens", "output_tokens"):
            count = getattr(self, name)
            if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 0):
                raise ValueError(f"Answer.{name} is a token count, a non-negative int or None, not {count!r}")
        cost = self.cost_usd
        if cost is not None and (
            isinstance(cost, bool) or not isinstance(cost, int | float) or not math.isfinite(cost) or cost < 0
        ):
            raise ValueError(f"Answer.cost_usd is dollars, a finite non-negative number or None, not {cost!r}")


def unwrap_answer(returned: Any) -> tuple[Any, CandidateTelemetry]:
    """What a candidate returned, as the answer to grade and the telemetry its spend lands on.

    Args:
        returned: The candidate's return value: an :class:`Answer`, or a plain answer.

    Returns:
        The answer's value and a telemetry carrying its ``candidate`` usage row; a plain answer as given,
        with telemetry that reports no usage.
    """
    if not isinstance(returned, Answer):
        return returned, CandidateTelemetry()
    ledger = RoleUsageLedger(role="candidate")
    ledger.add_llm_result(returned)
    return returned.value, CandidateTelemetry(usage=ledger.rows())


__all__ = ["Answer", "unwrap_answer"]
