"""The one failure claims grading raises: recorded per criterion on the cell, never a judge error or an apparatus fault.

Engine-internal: the claims steps raise it, and the judge phase records it.
"""

from __future__ import annotations


class ClaimGradingFailed(Exception):
    """Claims grading could not grade a cell's text: a reply was cut short, a reply would not parse after its retries,
    or an item still failed its checks after the repair rounds.

    It is no :class:`~threetears.evals.kernel.host.apparatus.ApparatusError` and no judge error: it costs the cell
    only that criterion's readings, so the trial's pass, its rubric readings and every other criterion keep the cell.
    """


__all__: list[str] = []
