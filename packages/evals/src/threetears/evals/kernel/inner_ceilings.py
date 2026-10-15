"""The package's inner ceilings a host derives its outer budgets from, defined once below every package using them.

Each is a bound the engine itself holds a call or a phase to: the proposer's output cap, the requests one generation
may send and the wall-clock ceilings of a generation and of a cell's judge phase. They live in ``kernel`` so that
:class:`threetears.evals.run.Ceilings` can carry them beside the simulator's caps without ``run`` importing ``gen``
or ``analysis``; ``gen`` and ``analysis`` export them from their roots as before, from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from threetears.evals.schema.completion import RequestCeiling

#: Output-token ceiling for the capability/boundary proposer. It drafts a full
#: rubric (template + several dims + variation axes + new-dim suggestions) in one
#: shot — far larger than the judge's per-dim score — so a 4096 default truncates
#: it mid-JSON (``finish_reason=length`` → unparseable object). This generous
#: ceiling lets the draft complete; cost scales with actual output, not the
#: ceiling. The host applies it when it builds the client it hands the proposer.
#: The judge's cap is its own, derived in ``threetears.evals.run.judge`` from its
#: reasoning budget plus its answer budget.
PROPOSER_MAX_TOKENS = 16384

#: Provider requests one generation can send: the generation, and the one repair round-trip a
#: refused output buys. :func:`generate_analysis` bounds it structurally; a test pins the two.
MAX_GENERATION_CALLS = 2


def generation_ceiling_s(*, request_s: RequestCeiling, generator_max_tokens: int) -> float:
    """The wall-clock ceiling of one :func:`generate_analysis`, derived from the ceilings it wraps.

    Up to :data:`MAX_GENERATION_CALLS` requests run back to back, each capped at ``generator_max_tokens``
    and each bounded by the host's own answer for one such request (``request_s``), which counts every
    provider call and retry wait the host's client spends on it. A budget over a generation must sit
    ABOVE this: one below it cancels a repair round-trip that is still writing inside its cap, after
    both calls were billed and with nothing stored.

    Not covered: whatever the caller does around the generation (assembling the bundle, storing the
    result).

    Args:
        request_s: The host's ceiling for one request on its client, by output cap.
        generator_max_tokens: The generator's output cap, as its clients are built with.

    Returns:
        The ceiling in seconds: ``MAX_GENERATION_CALLS * request_s(generator_max_tokens)``.
    """
    return MAX_GENERATION_CALLS * request_s(generator_max_tokens)


def judge_phase_ceiling_s(
    *,
    judge_dims: int,
    judge_concurrency: int,
    judge_call_attempts: int,
    judge_max_tokens: int,
    request_s: RequestCeiling,
) -> float:
    """The wall-clock ceiling of one cell's judge phase, derived from the ceilings of its requests.

    The judge scores ``judge_dims`` dimensions in waves of ``judge_concurrency``, each dimension up to
    ``judge_call_attempts`` requests capped at ``judge_max_tokens``, each bounded by the host's own answer
    for one such request (``request_s``), which counts every provider call and retry wait the host's
    client spends on it. Every kind whose cell ends in a judge phase adds this to its ceiling, since the
    cell's deadline bounds the judging too, and a ceiling without it cuts a slow cell off while it is scored.

    Args:
        judge_dims: Dimensions the judge scores on each cell.
        judge_concurrency: Judge requests in flight at once.
        judge_call_attempts: Requests one judge dimension can make, its parse retries included.
        judge_max_tokens: The judge's output cap, as its requests are built with.
        request_s: The host's ceiling for one request on its client, by output cap.

    Returns:
        The judge phase's ceiling in seconds: ``waves * judge_call_attempts * request_s(judge_max_tokens)``,
        with ``waves`` the dimensions over the concurrency, rounded up.
    """
    judge_waves = -(-max(judge_dims, 0) // max(judge_concurrency, 1))
    return judge_waves * judge_call_attempts * request_s(judge_max_tokens)


__all__ = [
    "MAX_GENERATION_CALLS",
    "PROPOSER_MAX_TOKENS",
    "generation_ceiling_s",
    "judge_phase_ceiling_s",
]
