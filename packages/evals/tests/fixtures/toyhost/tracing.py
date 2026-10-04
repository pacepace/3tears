"""The toy host's own tracing, implementing the engine's trace-sink port without OpenTelemetry.

:mod:`threetears.evals.contracts.host.traces` states two obligations and takes an implementation of
them — *"a host implements this against its own tracing"*. This one is deliberately **not** OTel: the
toy host's tracing is a list of records appended by whatever work is running inside the collection
scope, and that is enough to fill :class:`~threetears.evals.contracts.host.traces.CellTrace`
completely — the spans as storable JSON, and the three buckets. A host whose tracing is a log file,
an APM vendor's client or a ContextVar of dicts is the ordinary case, and this is what the port owes
it.

**Two things every sink should agree on.**

*The bucket rule.* ``total_ms`` is summed across the spans that mark one unit of the candidate's
processing — the turn roots — and ``llm_ms`` across the model calls, which is what the engine's own
latency record documents its fields to mean. Buckets are compared across hosts on one
cost-vs-latency axis, and a host that summed something else would be reporting a different quantity
under the same name.

*The wire attribute.* :data:`OPERATION_ATTR` and its two values are **wire** names: they appear in
every persisted ``EvalTrace.otel_trace``, so they are spelled here as the literals a stored trace
carries.

**Narrowing is a question this sink does not have.** A process-wide exporter has to take a cell's
own spans back out of a shared buffer. The toy host collects into a scope installed for the
duration of one cell, so a span either landed in this cell's recorder or was never offered to one.
``traces.py`` names that host explicitly — *"a host collecting per cell has nothing to put there"*
— and this is that host.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.contracts.host import CellIdentity, CellTrace

#: The span attribute naming what kind of operation a span covers. A wire name — see the module
#: docstring for why it is a literal here and not an import.
OPERATION_ATTR = "gen_ai.operation.name"

#: One unit of the candidate's own processing. ``total_ms`` is summed across these.
OP_TURN_ROOT = "agent.invoke"

#: One model call inside a turn. ``llm_ms`` is summed across these.
OP_MODEL_CALL = "llm.call"

#: The host's own grading pass — a toy-host operation the engine has no bucket for, and the
#: reason it is here rather than left unspanned: ``summarize`` has to have a span that
#: contributes to NO bucket, or its "spans of other operations contribute to nothing" branch is
#: unexercised. The grader runs outside the collection window, so a ``grade`` span reaching a
#: cell's collected record is how the two windows would be caught having the same extent.
OP_HOST_GRADE = "host.grade"


@dataclass
class ToySpan:
    """One span the toy host's instrumentation recorded.

    A plain record rather than a tracing library's object: what the port asks for is data, and
    the toy host has no library to hand one back from.
    """

    name: str
    attributes: dict[str, Any]
    start_ns: int
    end_ns: int | None = None

    @property
    def duration_ms(self) -> float | None:
        """How long the span took, or ``None`` while it is still open.

        Returns:
            Milliseconds, or ``None`` for a span whose scope has not closed — which a finished
            recorder should not hold, and which is skipped rather than counted as zero.
        """
        if self.end_ns is None:
            return None
        return (self.end_ns - self.start_ns) / 1_000_000.0

    def as_json(self) -> dict[str, Any]:
        """The span as the engine stores it — opaque JSON in the host's own shape.

        Returns:
            A JSON-safe dict.
        """
        return {
            "name": self.name,
            "attributes": dict(self.attributes),
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ms": self.duration_ms,
        }


@dataclass
class _Recorder:
    """Where spans land while one collection scope is open."""

    spans: list[ToySpan] = field(default_factory=list)


#: The scope spans are offered to. ``None`` means nothing is collecting, which is the state every
#: toy-host run that wires no sink is in — and the state :func:`toy_span` must cost nothing in.
_RECORDER: ContextVar[_Recorder | None] = ContextVar("toyhost_span_recorder", default=None)

#: The cell whose work is being done, for the identity half of the port. Held so a host surface
#: could attribute a log line to a cell; the sink itself needs nothing from it, which is exactly
#: what makes the two obligations separate extents rather than one scope.
_CELL: ContextVar[CellIdentity | None] = ContextVar("toyhost_span_cell", default=None)


def active_cell() -> CellIdentity | None:
    """Which cell's work is running, as the toy host's own code can see it.

    The identity scope's whole observable effect. It exists so a test can assert that the
    identity window really is wider than the collection window — the difference between the two
    extents is a measurement decision the port's docstring rests on, and a fixture that could
    not see it would be taking that on trust.

    Returns:
        The cell, or ``None`` outside every identity scope.
    """
    return _CELL.get()


@contextmanager
def toy_span(name: str, *, operation: str, **attributes: Any) -> Iterator[ToySpan]:
    """Record one span of the toy host's work, if anything is collecting.

    Args:
        name: What the span covers, in the host's own words.
        operation: The operation kind — :data:`OP_TURN_ROOT` or :data:`OP_MODEL_CALL` for the two
            the buckets read, or anything else for a span that contributes to no bucket.
        **attributes: Further attributes, carried verbatim into the stored JSON.

    Yields:
        The span record. A caller needs nothing from it; it is yielded so a test can read what
        was recorded without reaching into the recorder.
    """
    span = ToySpan(
        name=name,
        attributes={OPERATION_ATTR: operation, **attributes},
        start_ns=time.perf_counter_ns(),
    )
    recorder = _RECORDER.get()
    try:
        yield span
    finally:
        span.end_ns = time.perf_counter_ns()
        # Appended on the way OUT, so a span that raised is still recorded with its real extent
        # rather than dropped — the work happened and cost what it cost.
        if recorder is not None:
            recorder.spans.append(span)


def summarize(spans: list[ToySpan]) -> dict[str, float | None]:
    """Sum the toy host's spans into the engine's three buckets.

    The bucket rule the module docstring states, applied to the toy host's records.

    **A bucket no span contributed to is ``None``, never ``0.0``**, which is the one thing
    :class:`~threetears.evals.contracts.host.traces.CellTrace` says a sink may not get wrong: on an axis where
    lower is better, a zero standing in for "unmeasured" wins every comparison it should have
    been excluded from. ``tool_ms`` is always ``None`` here, and legitimately — the extractor
    calls no tool, so there is nothing to have measured.

    Args:
        spans: Every span one cell's collection scope recorded.

    Returns:
        ``{"total_ms", "llm_ms", "tool_ms"}``, each a sum or ``None``.
    """
    totals: dict[str, float | None] = {OP_TURN_ROOT: None, OP_MODEL_CALL: None}
    for span in spans:
        operation = span.attributes.get(OPERATION_ATTR)
        duration = span.duration_ms
        if operation not in totals or duration is None:
            continue
        # ``or 0.0`` seeds on the first contribution: the None means "nothing has contributed
        # yet", so it must not survive a span that did.
        totals[operation] = (totals[operation] or 0.0) + duration
    return {"total_ms": totals[OP_TURN_ROOT], "llm_ms": totals[OP_MODEL_CALL], "tool_ms": None}


class ToyTraceSink:
    """The toy host's :class:`~threetears.evals.contracts.host.traces.TraceSink`.

    Two scopes and a handful of assignments, which is what the port claims implementing it costs.
    Nothing here catches on the engine's behalf and nothing here degrades: a cell that emitted no
    span leaves the empty record, which reads as "measured nothing" rather than "measured zero".
    """

    def __init__(self) -> None:
        """Start with no cells seen."""
        #: Every cell whose collection scope opened AND closed, in order. Held for the fixture's
        #: own assertions — the engine reads only what a scope yields.
        self.collected: list[tuple[CellIdentity, CellTrace]] = []
        #: Every cell whose identity scope opened, in order.
        self.identified: list[CellIdentity] = []

    @contextmanager
    def cell_identity(self, cell: CellIdentity) -> Iterator[None]:
        """Mark everything inside as this cell's work, apparatus included.

        Args:
            cell: The execution the scope belongs to.

        Yields:
            Nothing. The identity travels on a ContextVar, so work the scope spawns carries it.
        """
        self.identified.append(cell)
        token = _CELL.set(cell)
        try:
            yield
        finally:
            _CELL.reset(token)

    @contextmanager
    def cell_spans(self, cell: CellIdentity) -> Iterator[CellTrace]:
        """Collect the spans this cell's own work emits.

        Args:
            cell: The execution being collected.

        Yields:
            The record, filled as the scope closes. A body that raises leaves it holding the
            empty answer and the engine's read unreached, which is the same answer as a cell
            that never ran.
        """
        collected = CellTrace()
        recorder = _Recorder()
        token = _RECORDER.set(recorder)
        try:
            yield collected
        finally:
            _RECORDER.reset(token)
        collected.spans = [span.as_json() for span in recorder.spans]
        buckets = summarize(recorder.spans)
        collected.total_ms = buckets["total_ms"]
        collected.llm_ms = buckets["llm_ms"]
        collected.tool_ms = buckets["tool_ms"]
        self.collected.append((cell, collected))


__all__ = [
    "OPERATION_ATTR",
    "OP_HOST_GRADE",
    "OP_MODEL_CALL",
    "OP_TURN_ROOT",
    "ToySpan",
    "ToyTraceSink",
    "active_cell",
    "summarize",
    "toy_span",
]
