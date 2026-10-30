"""Where a read's wall time went, by phase, as one ``eval.read_tier`` log line per call.

A slow read is answered by a measurement, not a guess: the line says how much of a call's time went to the store,
to rebuilding models from what it returned, to the folds over them and to rendering the answer, and how much it
took in. :func:`phase_logged` opens a call's line and logs it when the call ends; :func:`charged` charges a block's
time to one phase; :func:`took_in` adds the runs and results a call read. A host reads the lines in its log store,
so the line's field names are a contract (the CHANGELOG lists them):

``eval.read_tier lens=<lens> scope_id=<scope> campaign_id=<campaign> runs_in= results_in= store_ms= hydrate_ms=
fold_ms= render_ms= elapsed_ms= path= store_calls=``, plus `` raised=<exception class>`` when the call raised.

- ``store_ms``, ``hydrate_ms``, ``fold_ms``, ``render_ms``: milliseconds charged to each phase. A phase is charged
  its block's time less the time of any phase charged inside it, so the phases never count one millisecond twice.
  The call's ``residual`` phase (``fold`` unless the call says otherwise) is charged whatever no other phase was:
  ``elapsed_ms`` less the rest. A phase nothing charged is left **empty** (``store_ms=``), which means "not
  measured here", never zero time.
- ``runs_in``, ``results_in``: what :func:`took_in` reported during the call, summed; empty when nothing did.
- ``elapsed_ms``: the call's whole wall time.
- ``path``, ``store_calls``: which read path served the call and how many store calls it made. Both are empty
  until the store reads report them.

**Nesting.** A call logged inside another (a bundle assembled inside a report read) logs its own line, and its
charges and its intake reach the enclosing call's line too: the outer line's phases still add up to its own
elapsed time, and its ``runs_in`` counts what its nested calls read. A phase charged around a nested call keeps
the nested call's residual time.

**Threads.** The open calls and charges live in a :class:`~contextvars.ContextVar`, so each thread and each task
sees its own, and nothing outlives the call that opened it. A worker that inherits a copy of the caller's context
(:func:`~threetears.evals.kernel.offload.run_blocking`) charges the caller's open line, under that line's lock.

**Timing a read.** A store read keeps its timing only when it runs where the open call can see it: in the call's own
thread, or on a worker reached through :func:`~threetears.evals.kernel.offload.run_blocking`, which copies the
context; a bare ``executor.submit`` starts from an empty context and charges nothing. Never open :func:`charged`
around a ``yield`` in a lazy generator: the block would time the consumer's work between items, and could close in
another context than the one it opened in.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

from threetears.evals.schema.logs import get_logger

__all__ = ["charged", "phase_logged", "took_in"]

log = get_logger(__name__)

#: A phase a read's time is charged to.
type Phase = Literal["store", "hydrate", "fold", "render"]

#: Every phase, in the order the line prints them.
_PHASES: tuple[Phase, ...] = ("store", "hydrate", "fold", "render")


class _CallLine:
    """One open call's line: its time by phase and its intake, added to under a lock."""

    def __init__(self, lens: str, scope_id: str, campaign_id: str, residual: Phase) -> None:
        self.lens = lens
        self.scope_id = scope_id
        self.campaign_id = campaign_id
        self.residual = residual
        self.charged_ms: dict[Phase, float] = {}
        self.runs_in: int | None = None
        self.results_in: int | None = None
        self.path = ""
        self.store_calls: int | None = None
        self.lock = threading.Lock()

    def charge(self, phase: Phase, ms: float) -> None:
        with self.lock:
            self.charged_ms[phase] = self.charged_ms.get(phase, 0.0) + ms

    def take_in(self, runs: int, results: int) -> None:
        with self.lock:
            self.runs_in = (self.runs_in or 0) + runs
            self.results_in = (self.results_in or 0) + results

    def message(self, elapsed_ms: float, raised: BaseException | None) -> str:
        with self.lock:
            phases = dict(self.charged_ms)
            phases[self.residual] = phases.get(self.residual, 0.0) + max(0.0, elapsed_ms - sum(phases.values()))
            fields = [
                f"lens={self.lens}",
                f"scope_id={self.scope_id}",
                f"campaign_id={self.campaign_id}",
                f"runs_in={_blank(self.runs_in)}",
                f"results_in={_blank(self.results_in)}",
                *(f"{phase}_ms={_ms(phases.get(phase))}" for phase in _PHASES),
                f"elapsed_ms={elapsed_ms:.1f}",
                f"path={self.path}",
                f"store_calls={_blank(self.store_calls)}",
            ]
        suffix = "" if raised is None else f" raised={type(raised).__name__}"
        return "eval.read_tier " + " ".join(fields) + suffix


@dataclass
class _Charge:
    """One open :func:`charged` block: the lines open when it began, and the time of the charges nested in it."""

    phase: Phase
    lines: tuple[_CallLine, ...]
    started: float
    nested_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_nested(self, ms: float) -> None:
        with self.lock:
            self.nested_ms += ms


@dataclass(frozen=True)
class _Open:
    """What is open in this context: its call lines and its charges, innermost last."""

    lines: tuple[_CallLine, ...] = ()
    charges: tuple[_Charge, ...] = ()


_OPEN: ContextVar[_Open] = ContextVar("threetears_evals_phase_log", default=_Open())


def _blank(value: int | None) -> str:
    return "" if value is None else str(value)


def _ms(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


@contextmanager
def phase_logged(lens: str, *, scope_id: str, campaign_id: str, residual: Phase = "fold") -> Iterator[None]:
    """Log one ``eval.read_tier`` line for the block, at INFO, when it ends, whether or not it raised.

    :param lens: The call, as the line names it (``assemble_context_bundle``, ``report_read``).
    :ptype lens: str
    :param scope_id: The scope read.
    :ptype scope_id: str
    :param campaign_id: The campaign read.
    :ptype campaign_id: str
    :param residual: The phase charged the time no :func:`charged` block took.
    :ptype residual: Phase
    :return: A context manager; the line is logged on its exit.
    :rtype: Iterator[None]
    """
    line = _CallLine(lens, scope_id, campaign_id, residual)
    outer = _OPEN.get()
    token = _OPEN.set(_Open(lines=(*outer.lines, line), charges=outer.charges))
    started = time.perf_counter()
    raised: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        # Recorded on the line, then raised on unchanged: the line is an observation, never a handler.
        raised = exc
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _OPEN.reset(token)
        log.log(logging.INFO, "%s", line.message(elapsed_ms, raised))


@contextmanager
def charged(phase: Phase) -> Iterator[None]:
    """Charge the block's time, less any phase charged inside it, to ``phase`` on every open call's line.

    Outside any :func:`phase_logged` call it measures nothing and costs a clock read.

    :param phase: The phase the block's time goes to.
    :ptype phase: Phase
    :return: A context manager timing the block.
    :rtype: Iterator[None]
    """
    outer = _OPEN.get()
    charge = _Charge(phase=phase, lines=outer.lines, started=time.perf_counter())
    token = _OPEN.set(_Open(lines=outer.lines, charges=(*outer.charges, charge)))
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - charge.started) * 1000.0
        _OPEN.reset(token)
        if outer.charges:
            outer.charges[-1].add_nested(elapsed_ms)
        with charge.lock:
            own_ms = max(0.0, elapsed_ms - charge.nested_ms)
        for line in charge.lines:
            line.charge(phase, own_ms)


def took_in(*, runs: int, results: int) -> None:
    """Add ``runs`` and ``results`` to what every open call's line reports it read.

    :param runs: Runs read.
    :ptype runs: int
    :param results: Results read.
    :ptype results: int
    """
    for line in _OPEN.get().lines:
        line.take_in(runs, results)
