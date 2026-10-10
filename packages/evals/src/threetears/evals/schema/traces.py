"""What the engine needs from a host's tracing, and nothing about how the host traces.

One eval cell is a unit of work whose wall-clock the engine reports and whose spans it
stores. Neither is something the engine can produce for itself: it drives a candidate it
did not instrument, through a host's own runtime, and only that host knows what a span is
there. So the engine states two obligations and takes an implementation of them.

**The two obligations are separate extents, and that is not an accident of shape.**
Identity has to be established *before* the rig is set up and released *after* it is torn
down, because a cell's apparatus work is still that cell's; collection covers only the
turns, because apparatus setup is not work the candidate did and must not land in a
latency bucket that reads as though it were. Folding them into one scope would silently
move slot startup and teardown into the measured window — a change to what the numbers
mean, arriving as a refactor.

**A sink that fails is allowed to fail loudly, and is not allowed to be quiet.** The two
states a reader cannot tell apart from a stored result are "this cell emitted no span" and
"the thing that reports spans was broken", and only one of them is a measurement. So
nothing here catches on the implementation's behalf: an implementation that raises brings
the work down where the failure is attributable, rather than degrading to an empty trace
and an absent latency that look exactly like a quiet cell. **Size that before reaching for
a raise: the engine contains a cell timeout and nothing else, so an exception out of either
scope unwinds the whole run and the cells after it never execute.** Raising is the right
signal for a rig that is broken; it is not a cheap way to abandon one cell.

The one failure an implementation *can* see and the engine cannot — spans arriving that
belong to no cell it knows — is the implementation's to report, loudly, at the moment it
observes it. That is why nothing below carries a count of foreign spans back to the engine:
the fact is real only for an implementation buffering many cells at once, a host collecting
per cell has nothing to put there, and a field no consumer reads is a field that exists to
go stale.

Wiring no sink at all is a third state and a legitimate one: see
``EvalHost.trace_sink``, where ``None`` says nobody was watching. It differs from a
broken sink in exactly the way that matters — it was a caller's decision, recorded at the
call site, rather than a mechanism that stopped working where nothing looks.

**What crosses the boundary is data, not behaviour.** :class:`CellTrace` is a plain record
the implementation fills, so implementing this is two scopes and a handful of assignments;
the engine owns the reading. The alternative — handing back an object with reader methods,
or an exporter — puts a second contract on every host and, in the exporter's case, is not a
port at all.

This module imports no host module at all, and — like
:mod:`~threetears.evals.kernel.host.apparatus` — that is load-bearing rather than incidental: a host
implements this against its own tracing, acquiring nothing of the engine's internals, and
the engine reaches its tracing through this and nothing else. Two gates hold it, and only
together: the tree walk over ``kernel/host/`` and this module catches an import of any package
outside ``threetears.evals`` that the host layer may not reach, and a per-leaf assertion beside it catches the one that
walk permits — a ``threetears.evals.*`` import, which is how the buckets below would have been
"tightened" by reaching for the engine's own latency model.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["CellIdentity", "CellTrace", "TraceSink"]


@dataclass(frozen=True, slots=True)
class CellIdentity:
    """Which cell execution a stretch of work belongs to.

    The engine mints this; a sink stamps it on what it collects and filters by it. Frozen
    because a sink may hold it across the whole cell and read it afterwards — a mutated
    identity would re-attribute work already collected under the old one.
    """

    run_id: str
    test_case_id: str
    #: Descriptive, never part of the scope — a run's subject is shared by every cell in it.
    subject_id: str
    #: **Unique per cell EXECUTION**, and that is the contract rather than a convention.
    #: A run loops k x model x test case, so ``(run_id, test_case_id)`` recurs across a
    #: cell's own siblings; a sink scoping on the coordinates would pool a cell with the
    #: earlier siblings that share them and report a total covering all of it.
    cell_id: str


@dataclass(slots=True)
class CellTrace:
    """What one cell's tracing yielded — filled by the sink, read by the engine.

    Mutable and pre-populated with the empty answer, which is what makes the failure modes
    line up: a sink that collected nothing, a scope that unwound on an exception before it
    filled anything, and a cell that genuinely emitted no span all leave the same record,
    and that record is "measured nothing" rather than "measured zero".

    Every field describes the same population, so a sink must not narrow one and not the
    rest: a stored trace showing work the total does not count is a second disagreement
    rather than a cross-check.

    **The three buckets are named fields rather than a mapping, and that is a correctness
    decision.** The engine reads them by name into its latency record, so a sink filling
    ``"total"`` or ``"llm_latency_ms"`` in a free-form mapping would produce a cell whose spans
    are stored and whose every bucket reads unmeasured, which is precisely the failure
    everything here is organised against. Named fields make that spelling mistake impossible
    to write.

    A bucket no span contributed to is ``None``, and **never** ``0.0``: on an axis where
    lower is better, a zero standing in for "unmeasured" wins every comparison it should
    have been excluded from. All three being ``None`` does not on its own leave the cell
    without a latency record — the engine builds one when ANY component was measured, and
    the drain wait and the judge phase are timed off a clock rather than collected here.
    """

    #: The spans as storable JSON, in the order they finished. The engine stores this
    #: verbatim on the cell's trace document and interprets no field of it — the shape is
    #: the host's own, and the one surface that displays it renders it as opaque JSON.
    spans: list[dict[str, Any]] = field(default_factory=list)
    #: Wall-clock the cell spent processing turns, summed across its turn-root spans.
    total_ms: float | None = None
    #: Of that, the model-attributable part — summed across the cell's model calls.
    llm_ms: float | None = None
    #: Of that, the tool-execution part.
    tool_ms: float | None = None


class TraceSink(Protocol):
    """The engine's whole reach into a host's tracing.

    A host supplies one of these on its ``EvalHost``; the engine calls nothing else and
    imports nothing else. An implementation that traces nothing is not what this is for —
    that is ``trace_sink=None``, which says so at the call site.

    The two scopes take the identity separately rather than one nesting inside the other.
    Nesting would make a mismatched pair unrepresentable, which is the better property in
    the abstract; it costs a second protocol for the inner handle, and the pairing it
    protects is made by the engine, which constructs one :class:`CellIdentity` per cell
    and hands the same object to both. The burden lands where there is one implementation
    rather than on every host.

    **Cells of one run can execute at once** — a run whose launch did not declare latency under
    test runs several, each in its own task — as cells of two runs always could. So a sink carries
    the identity on the context (a ``ContextVar``), never in a "current cell" field of its own: two
    scopes open at once must each see only their own work.
    """

    def cell_identity(self, cell: CellIdentity) -> AbstractContextManager[None]:
        """Mark work done inside the scope as this cell's.

        Entered before the cell's rig is set up and exited after it is torn down, so
        apparatus work carries the cell's identity even though it is deliberately outside
        the collection window.

        Args:
            cell: The execution the scope belongs to.

        Returns:
            A context manager yielding nothing. Nesting is not required of it: the engine
            opens exactly one per cell.
        """
        ...

    def cell_spans(self, cell: CellIdentity) -> AbstractContextManager[CellTrace]:
        """Collect the spans this cell's turns emit.

        Opened after the rig is up and closed before it comes down, so what it collects is
        the candidate's work and not the harness's.

        The record is filled **as the scope closes** and read after it has: an
        implementation buffering into a store shared with other cells has to take its own
        spans out of that store on the way out, and there is nothing left to narrow once it
        has. Filling it earlier is allowed and pointless; the engine does not look until the
        scope is done. Yielding anything other than the record — ``None`` included — is a
        programming error rather than a way to say "nothing": the empty record already says
        that, and the engine reads through what it is given rather than checking.

        Args:
            cell: The execution being collected.

        Returns:
            A context manager yielding the record to fill. A body that raises leaves the
            engine's read unreached and the cell recording no trace — the same answer as a
            cell that never ran its turns.
        """
        ...
