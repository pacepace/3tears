"""``Guardrail``: something every arm of a :func:`~threetears.evals.quick.compare` must not get worse on.

A scorer or a judged dimension is ordinarily capability: something an arm should do well, tested against the
control in the contrasts table and traded against the other readings there. Some readings are not for trading
— the answer never leaks a customer's email, never promises a refund — and a gain elsewhere must not pay for a
loss on one. Named in ``compare(guardrails=...)``, a reading is a guardrail instead: kept out of every contrast
and composite, and decided on its own for each arm against the control, ``held``, ``breached`` or
``undecided``, in the report's guardrails table and :meth:`~threetears.evals.quick.Comparison.guardrails`.

**A guardrail declares its margin and its direction, and neither is assumed.** The margin is how much worse
than the control an arm may be and still hold, in the reading's own units (0.02 is two points on a pass rate);
the direction is which way is better. ``held`` needs the arm shown no worse than the margin, ``breached``
needs it shown worse by more, and anything in between is ``undecided``, which is never read as safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

#: Which way is better on a guardrail, in the words a campaign's bar declares its own direction in.
GuardrailDirection = Literal["higher_is_better", "lower_is_better"]

_DIRECTIONS: tuple[GuardrailDirection, ...] = ("higher_is_better", "lower_is_better")


def _a_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


@dataclass(frozen=True, kw_only=True)
class Guardrail:
    """A reading every arm must not get worse on than the control by more than ``margin``.

    Attributes:
        margin: How much worse than the control an arm may be and still hold, in the reading's own units: a
            positive number. On a pass/fail scorer it is a share of cases (0.02: two cases in a hundred); on a
            judged 1-5 dimension, points of the scale. Required: no margin is ever assumed, and with none no
            arm could ever be shown to hold.
        direction: Which way is better on the reading: ``"higher_is_better"`` (``no_leak`` passes) or
            ``"lower_is_better"`` (``leaked`` passes). Required, since "worse" needs one. A judged dimension
            is scored with higher better, so it takes ``"higher_is_better"``.
        value_range: The least and most a scorer that is not a pass/fail can return, when it is bounded (a
            score from 0 to 1, say). With a range, two arms that score every case alike — the state of a
            guardrail at its ceiling — read the interval the range allows, rather than ``undecided`` for want
            of one, and a value outside it excludes its case as a fault of the scorer. A pass/fail scorer (one
            annotated ``-> bool``) is on 0 to 1 already, and a judged dimension is on its scale, so neither
            takes one.
    """

    margin: float
    direction: GuardrailDirection
    value_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Refuse a guardrail no arm could be decided against.

        Raises:
            ValueError: A margin that is not a positive finite number, a direction that is neither, or a range
                that is not two finite numbers, least first, wider than the margin.
        """
        if not _a_number(self.margin) or self.margin <= 0:
            raise ValueError(
                f"a guardrail's margin is how much worse than the control an arm may be and still hold, a positive "
                f"number in the reading's own units (0.02 is two points on a pass rate), not {self.margin!r}; no "
                "margin is assumed, and with none no arm could be shown to hold"
            )
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"a guardrail's direction is which way is better on it, {' or '.join(map(repr, _DIRECTIONS))}, "
                f"not {self.direction!r}"
            )
        if self.value_range is not None:
            bounds = tuple(self.value_range) if isinstance(self.value_range, tuple | list) else ()
            if len(bounds) != 2 or not all(_a_number(bound) for bound in bounds) or not bounds[0] < bounds[1]:
                raise ValueError(
                    f"a guardrail's value_range is the least and the most its scorer can return, two finite "
                    f"numbers, least first (0.0, 1.0), not {self.value_range!r}"
                )
            if self.margin >= bounds[1] - bounds[0]:
                raise ValueError(
                    f"a margin of {self.margin!r} on a range of {bounds[0]!r} to {bounds[1]!r} is as wide as every "
                    "value the scorer can return, so every arm would hold whatever it did; declare a smaller one"
                )

    @property
    def higher_is_better(self) -> bool:
        """Whether higher is better on the reading, as a measure declares it."""
        return self.direction == "higher_is_better"


__all__ = ["Guardrail", "GuardrailDirection"]
