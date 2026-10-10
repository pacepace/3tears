"""How many entries each of the context bundle's growing lists may carry, and the one cap they share.

Every list here grows with the campaign, the ledger or the measure count rather than with what a reader
can act on, and each rides whole into a paid prompt. :func:`_capped` keeps at most the cap and counts the
rest, so a capped list is never read as a complete one. Part of the bundle assembler
(:mod:`threetears.evals.analysis.bundle.assemble`).
"""

from __future__ import annotations

from collections.abc import Callable


# How many divergences may be reported. Every (lever × level-pair × cross-scope measure
# pair) is a candidate, so the space is quadratic in a campaign's measure count and a
# thorough campaign can produce more of these than a reader will ever act on — which is
# the failure the gate exists to prevent, arriving by volume instead of by noise. The
# count dropped is reported rather than silently truncated.
_MAX_DIVERGENCES = 8

# The same cap, for the three other lists that grow with the campaign or the ledger rather than with what a
# reader can act on, each with what it dropped counted beside it (see `_capped`). `refused_merges` is
# quadratic in the cells a variant spans, `next_experiments` grows with variants × unrecorded dimensions, and
# `prior_insights` grows with every generation over the subject — and all three ride whole into a paid prompt.
# Starting values, not measured ones: tuning them is separate work.
_MAX_REFUSED_MERGES = 8
_MAX_NEXT_EXPERIMENTS = 8
_MAX_PRIOR_INSIGHTS = 12
# Pivots over co-varying factor pairs, which grow with the square of the factors in the worst case.
_MAX_FACTOR_PAIR_PIVOTS = 8
# The cells of a declared crossing listed, which grow with the product of the declared levels.
_MAX_DECLARED_CELLS = 64
_CELL_STATES = ("ran", "not_run", "skipped_by_design", "undetermined")


def _capped[T](items: list[T], cap: int, *, weight: Callable[[T], float] | None) -> tuple[list[T], int]:
    """Keep at most ``cap`` of ``items`` and count the rest, so a capped list is never read as a complete one.

    The one cap the bundle's growing lists share. Deterministic, so the fingerprint stays stable: the kept
    entries are the ``cap`` heaviest by ``weight`` (earlier entries winning ties), or the first ``cap`` when
    there is no weight, and they keep their order in ``items``.

    Args:
        items: The full list, in the order it is reported.
        cap: How many may be reported.
        weight: What ranks an entry for keeping, heaviest first; None keeps the leading entries.

    Returns:
        ``(kept, omitted)``.
    """
    if len(items) <= cap:
        return items, 0
    if weight is None:
        return items[:cap], len(items) - cap
    ranked = sorted(range(len(items)), key=lambda index: (-weight(items[index]), index))
    kept = sorted(ranked[:cap])
    return [items[index] for index in kept], len(items) - cap


__all__: list[str] = []
