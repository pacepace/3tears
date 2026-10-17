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

#: How many trials the bundle's `trials` may carry. Each carries a capped excerpt, so this bounds the bundle near a
#: megabyte; the lost and failed trials are kept first, and every trial is still counted by the scoreboard.
MAX_TRIALS = 500
#: How much of what the judge was handed a trial keeps: its tail, where a conversation ends and a cut-off shows.
MAX_EXCERPT_CHARS = 2000
#: How much of what a case asked a trial keeps.
MAX_INPUT_CHARS = 500
#: How much of what a case asked its label carries.
MAX_LABEL_INPUT_CHARS = 80
#: How much of a judge's reasoning a trial keeps.
MAX_REASON_CHARS = 1000
#: How much of a recorded error a trial keeps.
MAX_ERROR_CHARS = 500
#: The marker a capped text ends with.
ELLIPSIS = "…"


def _capped[T](items: list[T], cap: int, *, weight: Callable[[T], float] | None) -> tuple[list[T], int]:
    """Keep at most ``cap`` of ``items`` and count the rest, so a capped list is never read as a complete one.

    The one cap the bundle's growing lists share. Deterministic, so the fingerprint stays stable: the kept
    entries are the ``cap`` heaviest by ``weight`` (earlier entries winning ties), or the first ``cap`` when
    there is no weight, and they keep their order in ``items``.

    :param items: The full list, in the order it is reported.
    :ptype items: list[T]
    :param cap: How many may be reported.
    :ptype cap: int
    :param weight: What ranks an entry for keeping, heaviest first; None keeps the leading entries.
    :ptype weight: Callable[[T], float] | None
    :return: ``(kept, omitted)``.
    :rtype: tuple[list[T], int]
    """
    if len(items) <= cap:
        return items, 0
    if weight is None:
        return items[:cap], len(items) - cap
    ranked = sorted(range(len(items)), key=lambda index: (-weight(items[index]), index))
    kept = sorted(ranked[:cap])
    return [items[index] for index in kept], len(items) - cap


def capped_text(text: str, cap: int) -> str:
    """``text`` cut to at most ``cap`` characters, ending in :data:`ELLIPSIS` when it was cut.

    :param text: The text.
    :ptype text: str
    :param cap: The most characters it may keep, the marker included.
    :ptype cap: int
    :return: The text, whole when it fits.
    :rtype: str
    """
    if len(text) <= cap:
        return text
    return text[: cap - len(ELLIPSIS)].rstrip() + ELLIPSIS


def tail_text(text: str, cap: int) -> str:
    """The last ``cap`` characters of ``text``, led by how many were dropped when any were.

    The tail rather than the head because what a judge was handed ends where the conversation did, and a cut-off reply
    shows there.

    :param text: The text.
    :ptype text: str
    :param cap: The most characters of it to keep.
    :ptype cap: int
    :return: The text whole when it fits; else ``[… N characters earlier]`` and its last ``cap`` characters.
    :rtype: str
    """
    if len(text) <= cap:
        return text
    return f"[{ELLIPSIS} {len(text) - cap} characters earlier]\n{text[-cap:]}"


__all__ = [
    "ELLIPSIS",
    "MAX_ERROR_CHARS",
    "MAX_EXCERPT_CHARS",
    "MAX_INPUT_CHARS",
    "MAX_LABEL_INPUT_CHARS",
    "MAX_REASON_CHARS",
    "MAX_TRIALS",
    "capped_text",
    "tail_text",
]
