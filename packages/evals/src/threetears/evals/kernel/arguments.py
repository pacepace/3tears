"""How a surface reads an optional argument it was handed blank: as unsupplied, identically everywhere.

One rule with two packages behind it that may not import each other: the analysis read lenses read a
blank metric, weighting, export format or cassette mode through it, and the run package's launch reads
a blank cassette mode through it. Contracts is the package both rows of the dependency matrix admit,
which is why the rule lives here rather than beside either reader.
"""

from __future__ import annotations


def normalize_blank(value: str | None, default: str) -> str:
    """Treat a blank argument as "not supplied", identically on every surface.

    Lives in one place for the reason
    :func:`~threetears.evals.kernel.status_filter.normalize_status_filter` does,
    and after the same failure: the two surfaces disagreed about ``?metric=``. MCP wrote
    ``metric or METRIC_COMPOSITE``, so an empty string silently became the
    default; REST passed it through and the seam refused it as an unknown
    metric. Same request, one answer and one 422 — the divergence class the
    parity gate exists to catch, reintroduced by a one-word idiom in an adapter.

    Blank means *unsupplied* rather than *invalid*, because a query string is
    where a value goes missing (``?metric=`` is what an unset template variable
    renders as), and every other optional parameter on these routes already
    reads it that way.

    Args:
        value: Raw argument from the caller.
        default: What an unsupplied argument means.

    Returns:
        The stripped value, or ``default`` when it is empty or whitespace.
    """
    return (value or "").strip() or default


__all__ = [
    "normalize_blank",
]
