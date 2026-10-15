"""The package's inner ceilings as one value: what a host derives its outer budgets from.

A host bounds the engine's work from outside (a job's timeout, a client's output cap, a retry budget around a
scheduling call). Each such outer budget must sit at or above the inner ceiling the engine holds the same work to,
or the host cuts off work the engine would still finish inside its own bound. :class:`Ceilings` carries those inner
ceilings in one place, so a host computes its budgets from it rather than copying numbers that a release may move.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from threetears.evals.kernel import inner_ceilings
from threetears.evals.run.simulator import (
    SCHEDULER_CALL_ATTEMPTS,
    SIMULATOR_ANSWER_BUDGET_TOKENS,
    SIMULATOR_MAX_TOKENS,
)


@dataclass(frozen=True)
class Ceilings:
    """The package's inner ceilings a host derives its outer budgets from, so a budget cannot drift below one.

    ``Ceilings()`` is this build's: every field is the value or function of the same name the package exports
    (``run.SCHEDULER_CALL_ATTEMPTS``, ``run.SIMULATOR_MAX_TOKENS``, ``run.SIMULATOR_ANSWER_BUDGET_TOKENS``,
    ``gen.PROPOSER_MAX_TOKENS``, ``analysis.generation_ceiling_s``, ``analysis.judge_phase_ceiling_s``), and none is
    set by the caller. A host computes each outer budget from a field (a scheduling call's retry budget from
    ``scheduler_call_attempts``, the simulator's and proposer's client caps from the token fields, a generation's or
    a judge phase's deadline from the two ceiling functions over its own per-request ceiling) instead of restating
    the number, so a release that raises an inner ceiling raises the budget over it.
    """

    #: Calls one ``llm_decided`` scheduling decision may make: the first and one repair.
    scheduler_call_attempts: int = field(default=SCHEDULER_CALL_ATTEMPTS, init=False)
    #: The simulated user's output cap, in tokens: its reasoning allowance plus its answer budget.
    simulator_max_tokens: int = field(default=SIMULATOR_MAX_TOKENS, init=False)
    #: The output room kept for the simulated user's visible reply, in tokens.
    simulator_answer_budget_tokens: int = field(default=SIMULATOR_ANSWER_BUDGET_TOKENS, init=False)
    #: The capability and boundary proposer's output cap, in tokens.
    proposer_max_tokens: int = field(default=inner_ceilings.PROPOSER_MAX_TOKENS, init=False)
    #: The wall-clock ceiling of one generation, in seconds, from the host's per-request ceiling
    #: (``generation_ceiling_s(request_s=..., generator_max_tokens=...)``).
    generation_ceiling_s: Callable[..., float] = field(
        default_factory=lambda: inner_ceilings.generation_ceiling_s, init=False
    )
    #: The wall-clock ceiling of one cell's judge phase, in seconds, from the host's per-request ceiling
    #: (``judge_phase_ceiling_s(judge_dims=..., judge_concurrency=..., judge_call_attempts=..., judge_max_tokens=...,
    #: request_s=...)``).
    judge_phase_ceiling_s: Callable[..., float] = field(
        default_factory=lambda: inner_ceilings.judge_phase_ceiling_s, init=False
    )


__all__ = ["Ceilings"]
