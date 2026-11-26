"""How a key pool picks its active key, and the key that serves one call: the policy seam (KP-D6).

The pool works out which keys are eligible, from the shared state, the breakers and the room it
knows of, and hands them to a :class:`SelectionPolicy` in preference order. The policy holds no state
and reads nothing; it only chooses. :class:`StickySelection` is the one v1 policy; a second, such as a
room-weighted spread, implements the same two methods.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Protocol

__all__ = ["KeyCandidate", "SelectionPolicy", "StickySelection"]


@dataclass(frozen=True, slots=True)
class KeyCandidate:
    """One key, as a policy sees it.

    :ivar label: the key's label
    :ivar priority: its configured priority; lower is preferred, ``None`` ranks after every set one
    :ivar eligible: whether it can serve now: its state is ``ok`` (for the model, where the call
        names one), its breaker admits calls, and it has room as far as the pool knows
    """

    label: str
    priority: int | None
    eligible: bool


class SelectionPolicy(Protocol):
    """Chooses a pool's active key, and the key that serves one call.

    Both methods get every key in preference order: by ``(priority, configuration order)``, a key with
    no priority after every key with one.
    """

    def active_key(self, candidates: Sequence[KeyCandidate], current: str | None) -> str | None:
        """The key the pool should hold active.

        :param candidates: every key, in preference order
        :ptype candidates: Sequence[KeyCandidate]
        :param current: the active key now, ``None`` when there is none (a new pool, or a period
            rollover, which starts the pool over)
        :ptype current: str | None
        :return: the key to hold active: ``current`` to stay, another label to move; ``None`` when no
            key is eligible, which leaves the active key where it is
        :rtype: str | None
        """
        ...

    def serving_key(self, candidates: Sequence[KeyCandidate], active: str | None, excluded: Set[str]) -> str | None:
        """The key one call should try next.

        :param candidates: every key, in preference order
        :ptype candidates: Sequence[KeyCandidate]
        :param active: the active key
        :ptype active: str | None
        :param excluded: keys this call has tried, or skipped for room; none of them is returned
        :ptype excluded: Set[str]
        :return: the key to try, or ``None`` when no eligible key is left
        :rtype: str | None
        """
        ...


class StickySelection:
    """Sticky rotation (KP-D2 to D5): every call starts on the active key, which moves only when it
    can't serve.

    - The active key stays while it is eligible, unless a key with a strictly lower priority is
      eligible again; then it returns to the most-preferred such key (KP-D5). Equal priorities, the
      default, never pull back.
    - When the active key can't serve, the next eligible key after it in preference order takes over,
      wrapping around (KP-D4).
    - With no active key, the most-preferred eligible key is chosen.
    - A call skips the keys it has tried or found too small, starting from the active key and going on
      in order, without moving the active key (KP-D3).
    """

    def active_key(self, candidates: Sequence[KeyCandidate], current: str | None) -> str | None:
        """See :meth:`SelectionPolicy.active_key`."""
        position = _position(candidates, current)
        if position is None:
            return next((key.label for key in candidates if key.eligible), None)
        here = candidates[position]
        if here.eligible:
            preferred = next(
                (key.label for key in candidates if key.eligible and _rank(key.priority) < _rank(here.priority)), None
            )
            return preferred if preferred is not None else here.label
        return _next_after(candidates, position, frozenset())

    def serving_key(self, candidates: Sequence[KeyCandidate], active: str | None, excluded: Set[str]) -> str | None:
        """See :meth:`SelectionPolicy.serving_key`."""
        position = _position(candidates, active)
        if position is None:
            return next((key.label for key in candidates if key.eligible and key.label not in excluded), None)
        here = candidates[position]
        if here.eligible and here.label not in excluded:
            return here.label
        return _next_after(candidates, position, excluded)


def _rank(priority: int | None) -> float:
    """A priority as a sortable number: ``None`` after every set priority."""
    return float("inf") if priority is None else float(priority)


def _position(candidates: Sequence[KeyCandidate], label: str | None) -> int | None:
    """Where ``label`` is among ``candidates``, ``None`` when it isn't one."""
    return next((index for index, key in enumerate(candidates) if key.label == label), None)


def _next_after(candidates: Sequence[KeyCandidate], position: int, excluded: Set[str]) -> str | None:
    """The first eligible, unexcluded key after ``position``, wrapping around; never the key at it."""
    count = len(candidates)
    for step in range(1, count):
        key = candidates[(position + step) % count]
        if key.eligible and key.label not in excluded:
            return key.label
    return None
