"""Latency read under concurrency is never compared as if it were clean (#701).

A run whose launch did not declare latency under test executes its cells side by side, and a run beside
another run contends with it for one provider and one box. Either way the wall-clock its results record is
real but not clean: it measures the candidate AND the contention, in a mix nothing recorded. Every such
result says so — its ``execution_mode`` covariate reads ``concurrent`` — and this module is the one place
the analysis honours the mark.

**Withheld, not pooled.** :func:`withhold_contended_latency` hands every analysis surface each marked result
with its elapsed-time readings removed: its latency record, its phase timings, the elapsed time of its
background work and every host measure of elapsed time (:func:`~threetears.evals.contracts.metrics.is_latency_measure`).
What a surface then reads is only latency taken serially, so a contrast, the Holm family, a bar, the frontier's
latency axis, a history series, the mechanism and scope lenses, a chart and the report never compare a
contended reading with anything — and a cell pooling runs of both kinds keeps the serial ones' latency and
leaves the rest out, which is keeping them apart. Every other measure of the result is untouched: concurrency
moves the clock, not what the candidate answered or what it spent.

**Disclosed, every time it removed something.** :func:`contended_latency_sentence` is the one line a surface
prints, counting what it left out. A result whose latency was never measured (a candidate nothing timed)
loses nothing here and is not counted, so a run with no latency says nothing about withholding it.

**A stored ``concurrent`` from 3tears-evals 0.66.0 and earlier is an upper bound** — that build counted runs
queued for a slot as running — and it is withheld like any other: a reading that may have been contended is
never presented as one that was not.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from threetears.evals.contracts.metrics import describe_measure, is_latency_measure

if TYPE_CHECKING:
    from threetears.evals.contracts.host.measures import MeasureRegistry
    from threetears.evals.contracts.models import EvalResult

#: The covariate that marks a result's latency as read under concurrency, and the value that marks it.
EXECUTION_MODE = "execution_mode"
CONCURRENT = "concurrent"


def latency_contended(result: EvalResult) -> bool:
    """Whether ``result`` was recorded while other eval work executed beside it — its latency is not clean.

    Args:
        result: The result.

    Returns:
        True when its ``execution_mode`` reads ``concurrent``. A result that recorded no mode (nothing sampled
        it) is not marked: nothing says it was contended, and its run's cells executed one at a time.
    """
    return result.covariates.get(EXECUTION_MODE) == CONCURRENT


def _carries_latency(result: EvalResult, measures: MeasureRegistry | None) -> bool:
    """Whether ``result`` holds any reading of elapsed time for :func:`withhold_contended_latency` to remove."""
    latency = result.latency
    if latency is not None and any(
        value is not None
        for value in (latency.total_ms, latency.llm_ms, latency.tool_ms, latency.async_wait_ms, latency.judge_ms)
    ):
        return True
    if result.phase_timings:
        return True
    if any(delivery.elapsed_ms is not None for delivery in result.async_deliveries or ()):
        return True
    return measures is not None and any(_latency_host_measure(name, measures) for name in result.host_measures)


def _latency_host_measure(name: str, measures: MeasureRegistry) -> bool:
    """Whether the host measure ``name`` reads elapsed time."""
    return is_latency_measure(describe_measure(name, measures))


def withheld_latency(results: Iterable[EvalResult], measures: MeasureRegistry | None) -> list[EvalResult]:
    """The results whose latency :func:`withhold_contended_latency` removes: marked, and holding some.

    Args:
        results: The results.
        measures: The host's measure registry, which says which of its own measures read elapsed time;
            ``None`` reads the engine's elapsed-time fields alone.

    Returns:
        The marked results that carry a reading of elapsed time, in the order given.
    """
    return [result for result in results if latency_contended(result) and _carries_latency(result, measures)]


def withhold_contended_latency(results: Sequence[EvalResult], measures: MeasureRegistry | None) -> list[EvalResult]:
    """``results`` as every analysis surface reads them: each marked result with its elapsed time removed.

    Idempotent, and a copy: the stored results are untouched, and a result with nothing to withhold is
    handed back as it is.

    Args:
        results: The results, as loaded.
        measures: The host's measure registry, which says which of its own measures read elapsed time;
            ``None`` reads the engine's elapsed-time fields alone, for a surface that reads no host measure.

    Returns:
        The results in the order given, each marked one without its latency, phase timings, background-work
        elapsed time or host measures of elapsed time.
    """
    withheld = {result.id for result in withheld_latency(results, measures)}
    if not withheld:
        return list(results)
    return [_without_latency(result, measures) if result.id in withheld else result for result in results]


def _without_latency(result: EvalResult, measures: MeasureRegistry | None) -> EvalResult:
    """One result with every reading of elapsed time removed."""
    deliveries = (
        [delivery.model_copy(update={"elapsed_ms": None}) for delivery in result.async_deliveries]
        if result.async_deliveries is not None
        else None
    )
    host_measures = (
        {name: value for name, value in result.host_measures.items() if not _latency_host_measure(name, measures)}
        if measures is not None
        else dict(result.host_measures)
    )
    return result.model_copy(
        update={
            "latency": None,
            "phase_timings": {},
            "async_deliveries": deliveries,
            "host_measures": host_measures,
        }
    )


def contended_latency_sentence(withheld: int, of: int, *, where: str = "", declared: bool = False) -> str | None:
    """The one line a surface prints when it left latency read under concurrency out, or ``None``.

    Args:
        withheld: How many results' latency was left out.
        of: How many results the surface read.
        where: Where, when only part of the surface was affected (``" in 2 of 5 cells"``); blank for all of it.
        declared: The campaign declares latency under test, so the runs read under concurrency are ones it cannot
            read its question from — the remedy says so.

    Returns:
        The sentence, or ``None`` when nothing was withheld.
    """
    if withheld <= 0:
        return None
    every = withheld == of
    count = f"every result's latency{where}" if every else f"the latency of {withheld} of {of} results{where}"
    return (
        f"Latency is not compared here: {count} was read while other cells or runs executed beside it "
        f"(execution_mode `{CONCURRENT}`), so it is left out of every comparison, bar and ranking"
        + ("." if every else ", and the latency shown is read only from results measured serially.")
        + (
            " This campaign declares latency under test, and those runs were not measured that way: relaunch them "
            "with measure_latency=True."
            if declared
            else " Launch with measure_latency=True to read latency clean."
        )
    )


def marked_latency_sentence(marked: int, of: int) -> str | None:
    """The line a ONE-run surface prints beside latency it shows that was read under concurrency, or ``None``.

    A run's own summary describes that run and compares nothing, so it shows the latency it recorded — marked,
    so the figure is never set against another run's as if it were clean.

    Args:
        marked: How many of the run's results recorded latency under concurrency.
        of: How many results the run holds.

    Returns:
        The sentence, or ``None`` when none was.
    """
    if marked <= 0:
        return None
    count = "Every result's latency" if marked == of else f"The latency of {marked} of {of} results"
    return (
        f"{count} here was read while other cells or runs executed beside it (execution_mode `{CONCURRENT}`): it "
        "describes this run as it ran, and is never compared with another run's. Launch with measure_latency=True "
        "to read latency clean."
    )


__all__ = [
    "CONCURRENT",
    "EXECUTION_MODE",
    "contended_latency_sentence",
    "marked_latency_sentence",
    "latency_contended",
    "withheld_latency",
    "withhold_contended_latency",
]
