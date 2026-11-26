"""copies' positions for a test of a fold that deletes (``retention="delete"``, ``docs/design-pod-delta-files.md`` decision
54): none at all, or beats a test sets and counts the reads of."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from threetears.core.collections.delta_copy_beats import CopyBeat, CopyPositions, CopyPositionsReader, Verdict

__all__ = ["NO_COPIES", "SetCopies"]


class SetCopies(CopyPositionsReader):
    """copies' beats a test sets (:meth:`set`), or a refusal (:attr:`refused`); every read counted (:attr:`reads`).

    :param beats: each copy's epochs, by copy; a copy given None is starting
    :ptype beats: Mapping[str, Mapping[str, str] | None]
    """

    def __init__(self, beats: Mapping[str, Mapping[str, str] | None] | None = None) -> None:
        self._positions = CopyPositions()
        self.reads = 0
        self.set(beats or {})

    def set(self, beats: Mapping[str, Mapping[str, str] | None]) -> None:
        """replace every copy's beat.

        :param beats: each copy's epochs, by copy; None for a copy still starting
        :ptype beats: Mapping[str, Mapping[str, str] | None]
        :return: nothing
        :rtype: None
        """
        self._positions = CopyPositions(
            beats=tuple(
                CopyBeat(copy=copy, epochs=MappingProxyType(dict(epochs or {})), starting=epochs is None)
                for copy, epochs in sorted(beats.items())
            )
        )

    def refuse(self, why: Verdict) -> None:
        """make every read refuse: an unreadable beat (``unreadable``) or a failed read (``read_failed``).

        :param why: the refusal
        :ptype why: str
        :return: nothing
        :rtype: None
        """
        self._positions = CopyPositions(refused=why)

    async def positions(self) -> CopyPositions:
        """the beats as set.

        :return: them
        :rtype: CopyPositions
        """
        self.reads += 1
        return self._positions


#: no copy at all: nothing holds a covered file back (each test of a deleting fold that is not about copies)
NO_COPIES: Final = SetCopies()
