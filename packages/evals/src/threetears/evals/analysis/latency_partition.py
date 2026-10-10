"""The latency partition: one result's ``total_ms`` split into its measured parts and a named remainder.

:func:`decompose_total_ms` splits the turn-root wall-clock into its LLM and tool parts and what is left over,
and withholds the split, with the reason, when a part was not measured or the parts sum to more than the whole.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import model_validator

from threetears.evals.analysis.numbers import format_number
from threetears.evals.schema.base import EvalBaseModel

if TYPE_CHECKING:
    from threetears.evals.schema.models import LatencyMetrics


# How far the parts may overrun the whole before the split is refused rather than
# reported.
#
# The parts nest inside the whole structurally — ``llm.call`` and ``tool.execute``
# spans are children of the ``agent.invoke`` turn root, and both the rounds loop and
# the action loop are serial — so a correct measurement can only leave the remainder
# at or above zero. The tolerance is therefore NOT a modelling allowance for
# overlapping work; it absorbs one thing only: the three numbers are summed in
# different groupings, so binary floating-point addition can put the difference a few
# units in the last place below zero on an exact partition.
#
# **Sized to that noise and not to a round number**, because a threshold generous
# enough to swallow a real overlap would silently convert a capture fault into a
# plausible-looking split. Durations are exact integer nanoseconds divided by 1e6, so
# the only error is the summation itself: bounded by roughly n·ε relative, and a cell
# emitting ~100 bucketed spans across a ~17-minute total puts that near 1e-8 ms. One
# nanosecond sits about two orders of magnitude above the noise and six below the
# shortest overlap any real span pair could produce. An overrun larger than this means
# a part was measured outside its whole — a span reaching a bucket from outside the
# turn roots — and that is a capture fault to report, not a negative component to
# publish.
PARTITION_TOLERANCE_MS = 1e-6

# The distinguishing clause of each reason the partition is withheld. Split out for the
# same reason the divergence lens splits its own: a test pins the CLAUSE, so the
# surrounding sentence can be rewritten for a reader without breaking the pin.
WITHHELD_UNMEASURED_COMPONENT = "not measured"
WITHHELD_PARTS_EXCEED_WHOLE = "sum to more than"


class LatencyPartition(EvalBaseModel):
    """One result's ``total_ms`` split into its named parts and a named remainder.

    ``total_ms`` is the wall-clock of the candidate's turn-root spans, and the two
    measured parts nest inside it. What is left over — whatever the candidate does
    between its model and tool calls: building its input, parsing its output, updating
    and persisting state — had no name, so a whole-run latency movement that lived there could not be placed. It
    could only be reported as a total that moved while both named parts stayed flat,
    which reads exactly like a measurement error. Naming the remainder is what turns
    that into an attribution.

    **Derived, never stored.** ``orchestration_ms`` is ``total_ms`` minus the two
    parts, recomputed per query exactly as :func:`project_score_records` and
    :func:`~threetears.evals.kernel.scoring.compute_pass_hat_k` are. A persisted copy would be a
    second answer to a question the three captured components already settle.

    **What is deliberately NOT in here.** The drain wait, the judge phase and every
    background phase timing are milliseconds measured on the same clock that fall
    *outside* the turn roots — background work on a detached trace root, and scoring
    that happens after the turns end. They are disjoint from ``total_ms``, not
    components of it, which the registry records by leaving their ``contained_by``
    unset. Folding any of them into the remainder is the arithmetic that once produced
    a ~95-second unattributed swing describing no stretch of wall-clock at all, and
    the reason this class takes a :class:`~threetears.evals.schema.models.LatencyMetrics` and
    reads three fields of it rather than summing whatever it holds.

    Exactly one of the split and ``withheld`` is present, enforced below: a caller
    that gets no numbers gets a sentence saying why, and never a silent ``None`` it
    has to invent an explanation for.
    """

    #: The whole being partitioned — wall-clock across the result's turn-root spans.
    total_ms: float | None = None
    #: The share inside model calls.
    llm_ms: float | None = None
    #: The share inside tool executions.
    tool_ms: float | None = None
    #: The named remainder: turn wall-clock inside neither a model call nor a tool
    #: execution. Clamped to exactly 0.0 when the parts overrun the whole by less than
    #: the float-noise tolerance, since a negative share of a whole describes nothing.
    orchestration_ms: float | None = None
    #: Why no split is published, as a sentence for a reader — set exactly when the
    #: split is absent. A partition needs all three components measured, and needs the
    #: parts to fit inside the whole.
    withheld: str | None = None

    @model_validator(mode="after")
    def _split_or_reason(self) -> LatencyPartition:
        """Refuse a record that is neither a whole split nor a whole refusal.

        Checks all four numbers rather than the remainder alone: a record carrying a
        total beside a withheld sentence would be a partition that half-published, and a
        reader has no way to tell which half to believe.
        """
        numbers = (self.total_ms, self.llm_ms, self.tool_ms, self.orchestration_ms)
        present = [value is not None for value in numbers]
        if any(present) != all(present) or all(present) == (self.withheld is not None):
            raise ValueError(
                "exactly one of the complete split / withheld must be set — got "
                f"total_ms={self.total_ms!r}, llm_ms={self.llm_ms!r}, tool_ms={self.tool_ms!r}, "
                f"orchestration_ms={self.orchestration_ms!r}, withheld={self.withheld!r}"
            )
        return self


def decompose_total_ms(latency: LatencyMetrics | None) -> LatencyPartition:
    """Split one result's ``total_ms`` into its parts and the named remainder.

    Args:
        latency: The result's captured latency, or ``None`` for a cell that timed
            nothing whatever.

    Returns:
        A :class:`LatencyPartition` carrying either the four numbers or a sentence
        naming why they were withheld.
    """
    if latency is None:
        return LatencyPartition(withheld="this cell timed nothing, so there is no total to partition.")

    components = {"total_ms": latency.total_ms, "llm_ms": latency.llm_ms, "tool_ms": latency.tool_ms}
    unmeasured = [name for name, value in components.items() if value is None]
    if unmeasured:
        # Not zero-filled. A cell can measure an `llm.call` while producing no turn
        # root — the candidate failing outside the turn wrapper does exactly that —
        # and treating the missing one as zero would publish a remainder computed
        # from a number nobody observed, in the same units as ones somebody did.
        verb = "was" if len(unmeasured) == 1 else "were"
        return LatencyPartition(
            withheld=(
                f"{', '.join(unmeasured)} {verb} {WITHHELD_UNMEASURED_COMPONENT}, so the parts cannot account for the whole."
            )
        )

    # Read back through the same dict the absence check walked, so the two can't
    # diverge into checking one set of fields and partitioning another.
    measured = {name: value for name, value in components.items() if value is not None}
    total, llm, tool = (float(measured[name]) for name in ("total_ms", "llm_ms", "tool_ms"))
    remainder = total - llm - tool
    if remainder < -PARTITION_TOLERANCE_MS:
        return LatencyPartition(
            withheld=(
                f"llm_ms and tool_ms {WITHHELD_PARTS_EXCEED_WHOLE} total_ms by "
                # The one number rule, and both ends of the range are why. A fixed
                # `.1f` renders every overrun below 50 microseconds as "by 0.0ms" — a
                # refusal whose own sentence says nothing happened — and the threshold
                # is a nanosecond, so those are reachable. A capped `%g` breaks the
                # other end: it goes scientific once the exponent reaches the precision,
                # so `.3g` prints a 464ms overrun, the size actually measured here, as
                # "by 4.64e+02ms". `format_number` keeps a small value non-zero and a
                # large one whole. This sentence is read by a person.
                f"{format_number(-remainder)}ms, so something reached a component bucket from outside the turn roots."
            )
        )
    return LatencyPartition(
        total_ms=total,
        llm_ms=llm,
        tool_ms=tool,
        # Clamped rather than published negative: within the tolerance the overrun is
        # float-addition noise on an exact partition, and a share of a whole that reads
        # as less than none of it would be a worse answer than the zero it really is.
        orchestration_ms=max(remainder, 0.0),
    )


__all__ = [
    "decompose_total_ms",
    "LatencyPartition",
    "PARTITION_TOLERANCE_MS",
    "WITHHELD_PARTS_EXCEED_WHOLE",
    "WITHHELD_UNMEASURED_COMPONENT",
]
