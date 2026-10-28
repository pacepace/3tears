"""The package's inner ceilings a host derives its outer budgets from, defined once below every package using them.

Each is a bound the engine itself holds a call or a phase to: the proposer's output cap, the requests one generation
may send and the wall-clock ceilings of a generation, of a report the writer writes and of a cell's judge phase. They live in ``kernel`` so that
:class:`threetears.evals.run.Ceilings` can carry them beside the simulator's caps without ``run`` importing ``gen``
or ``analysis``; ``gen`` and ``analysis`` export them from their roots as before, from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from threetears.evals.ops.report_writer.loop import WriterLimits
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

#: Provider requests one analysis generation can send. A generation is written by the report planner
#: (:func:`~threetears.evals.analysis.planner.generate_report_plan`) in two calls, a structure and then its prose, so
#: neither contract grows with the page; a refused reply of either buys that call one repair round-trip. So: the
#: structure, its repair, the prose, its repair. The planner bounds it structurally; a test pins the four.
MAX_GENERATION_CALLS = 4


def generation_ceiling_s(*, request_s: RequestCeiling, generator_max_tokens: int) -> float:
    """The wall-clock ceiling of one analysis generation, derived from the ceilings it wraps.

    Up to :data:`MAX_GENERATION_CALLS` requests (the planner's) run back to back, each capped at ``generator_max_tokens``
    and each bounded by the host's own answer for one such request (``request_s``), which counts every
    provider call and retry wait the host's client spends on it. A budget over a generation must sit
    ABOVE this: one below it cancels a repair round-trip that is still writing inside its cap, after
    both calls were billed and with nothing stored.

    Not covered: whatever the caller does around the generation (assembling the bundle, storing the
    result).

    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :param generator_max_tokens: The generator's output cap, as its clients are built with.
    :ptype generator_max_tokens: int
    :return: The ceiling in seconds: ``MAX_GENERATION_CALLS * request_s(generator_max_tokens)``.
    :rtype: float
    """
    return MAX_GENERATION_CALLS * request_s(generator_max_tokens)


#: The editor's answers one review reads: its first, and the one it is asked for again when the first cannot be read
#: (``ops/report_writer/editor.py``). The review round reads it from here, so the writer's ceiling counts what it sends.
EDITOR_ANSWERS = 2


def writer_turns(limits: WriterLimits) -> int:
    """The most turns one report can send under ``limits``, counted from how the writing loop sends them.

    A round's tool calls are its call budget: each turn that calls the tool spends at least one, and a turn after the
    last is answered as not run or ends the round, so a round sends at most its calls in tool turns. Every send of a
    message (the brief or a revision request, each bounce, the editor's request and its one ask again) ends on one more
    turn: the reply that calls no tool, one cut at its output cap, or one stopped short. The warnings and the "submit
    now" note ride on tool results and send no turn of their own. So:

    - the draft: ``analyst.draft_calls`` tool turns and ``1 + analyst.bounces`` closing turns;
    - each of ``revisions`` revisions: ``analyst.revision_calls`` tool turns and ``1 + analyst.bounces`` closing turns;
    - each review, one per analyst round (``1 + revisions``): ``editor.calls`` tool turns and :data:`EDITOR_ANSWERS`
      closing turns.

    :param limits: What the writing loop is held to.
    :ptype limits: WriterLimits
    :return: The worst-case turn count.
    :rtype: int
    """
    return _analyst_turns(limits) + _editor_turns(limits)


def _analyst_turns(limits: WriterLimits) -> int:
    analyst, closing = limits.analyst, 1 + limits.analyst.bounces
    return analyst.draft_calls + closing + limits.revisions * (analyst.revision_calls + closing)


def _editor_turns(limits: WriterLimits) -> int:
    return (1 + limits.revisions) * (limits.editor.calls + EDITOR_ANSWERS)


def writer_ceiling_s(limits: WriterLimits, *, request_s: RequestCeiling) -> float:
    """The wall-clock ceiling of one report the writer writes (``analysis_generate``'s job), from the ceilings it wraps.

    Every turn the loop can send (:func:`writer_turns`) runs back to back, each bounded by the host's own answer for
    one request at its session's output cap (``request_s(limits.analyst.session.max_tokens)`` for the analyst's turns,
    ``request_s(limits.editor.session.max_tokens)`` for the editor's), which counts every provider call and retry
    wait the host's client spends on it. A job budget over a report (``AnalysisGeneration.budget_s``) must sit ABOVE
    this: one below it cancels a turn still writing inside its cap, after it was billed.

    Not covered: the tool calls the turns make (reads of the frozen snapshot, in the host's process), the price refresh
    each turn makes before it is priced (``refresh_prices()``, a catalogue read a host's ``request_s`` may not count),
    freezing the snapshot and storing the report. A host adds its headroom for them.

    :param limits: What the writing loop is held to: the same object the host sets as ``AnalysisGeneration.limits``,
        which ``analysis_generate`` runs unchanged.
    :ptype limits: WriterLimits
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :return: The ceiling in seconds.
    :rtype: float
    """
    analyst = _analyst_turns(limits) * request_s(limits.analyst.session.max_tokens)
    return analyst + _editor_turns(limits) * request_s(limits.editor.session.max_tokens)


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

    :param judge_dims: Dimensions the judge scores on each cell.
    :ptype judge_dims: int
    :param judge_concurrency: Judge requests in flight at once.
    :ptype judge_concurrency: int
    :param judge_call_attempts: Requests one judge dimension can make, its parse retries included.
    :ptype judge_call_attempts: int
    :param judge_max_tokens: The judge's output cap, as its requests are built with.
    :ptype judge_max_tokens: int
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :return: The judge phase's ceiling in seconds: ``waves * judge_call_attempts * request_s(judge_max_tokens)``,
        with ``waves`` the dimensions over the concurrency, rounded up.
    :rtype: float
    """
    judge_waves = -(-max(judge_dims, 0) // max(judge_concurrency, 1))
    return judge_waves * judge_call_attempts * request_s(judge_max_tokens)


__all__ = [
    "EDITOR_ANSWERS",
    "MAX_GENERATION_CALLS",
    "PROPOSER_MAX_TOKENS",
    "generation_ceiling_s",
    "judge_phase_ceiling_s",
    "writer_ceiling_s",
    "writer_turns",
]
