"""The two states of an attribution remainder, named once.

A whole-vs-part remainder is either STATED (a number: ``unattributed_delta``) or WITHHELD (a sentence
saying why it cannot be one: ``unattributed_withheld``), and exactly one of the two is set. Two types
carry the pair: :class:`~threetears.evals.analysis.viz.payloads.AttributionPayload`, the chart's payload,
and :class:`~threetears.evals.analysis.bundle.schema.ScopeDivergence`, the bundle's divergence. Both
enforce the invariant with :func:`remainder_state` and expose the result as a ``remainder`` property,
and every consumer reads that property rather than testing either field itself.

**A blank sentence explains nothing.** The reader is owed the reason no number is stated; an empty or
whitespace-only string carries none, so it counts as no sentence at all. The two types once disagreed
here — one tested the sentence for truth, the other for ``None`` — so the same empty string was
"explained" in the bundle and "unexplained" in the chart. Read here, the answer is one.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "RemainderState",
    "remainder_state",
]

#: ``"stated"``: the remainder is a number. ``"withheld"``: a sentence says why it cannot be one.
RemainderState = Literal["stated", "withheld"]


def remainder_state(delta: float | None, withheld: str | None) -> RemainderState:
    """Name which of its two states a remainder is in, refusing a pair that is in neither or both.

    Args:
        delta: The quantified remainder, or None.
        withheld: The sentence withholding it, or None. A blank sentence counts as None.

    Returns:
        ``"stated"`` when only ``delta`` is set, ``"withheld"`` when only a non-blank ``withheld`` is.

    Raises:
        ValueError: If both are set, or neither is.
    """
    stated = delta is not None
    explained = withheld is not None and bool(withheld.strip())
    if stated and explained:
        raise ValueError(
            "unattributed_delta and unattributed_withheld are both set — a remainder is either quotable or it "
            f"is not, and stating a number beside the reason it cannot be stated is neither (got delta={delta!r}, "
            f"withheld={withheld!r})"
        )
    if not stated and not explained:
        raise ValueError(
            "neither unattributed_delta nor unattributed_withheld is set (a blank sentence counts as none) — two "
            "movements published with no word about their difference invite the subtraction the pair exists to "
            "refuse "
            f"(got delta={delta!r}, withheld={withheld!r})"
        )
    return "stated" if stated else "withheld"
