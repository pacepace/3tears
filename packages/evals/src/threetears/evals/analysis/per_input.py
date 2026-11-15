"""What one report input has been read as, kept with the input it was read off.

A report input is frozen and never changed in place: an input with a moved line is a copy
(:meth:`~pydantic.BaseModel.model_copy`), another object with values kept of its own. So anything that is pure in the
input (a key's figure, a page's forms, its arms' names) can be worked out once per input and read back for every later
caller, which a cited figure's text, a page sized by halving and a listing ordered by its figures all are.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from threetears.evals.analysis.report_input import ReportInput

__all__ = ["kept_for"]


class _Kept:
    """The values worked out of one report input, each built the first time it is asked for.

    Unlocked: two threads asking first for one value each build it, and one of the two equal values is kept. A build
    may ask for another value of the same input (a page's forms read the input's figures), which a lock held across the
    build would deadlock on.
    """

    def __init__(self) -> None:
        self._values: dict[Hashable, object] = {}

    def get[V](self, key: Hashable, build: Callable[[], V]) -> V:
        """The value kept under ``key``, built by ``build`` the first time.

        :param key: What the value is, unique among everything kept of one input: the function or class it is read
            through, or a tuple led by that function and naming what the value is read for.
        :ptype key: Hashable
        :param build: Works the value out of the input; it must be pure in the input.
        :ptype build: Callable[[], V]
        :return: The value.
        :rtype: V
        """
        if key in self._values:
            return cast("V", self._values[key])
        value = build()
        self._values[key] = value
        return value


@dataclass(frozen=True, eq=False)
class _ByIdentity:
    """A report input by its identity, held weakly: equal only to a handle on the same live input, so what is kept of
    an input never outlives its identity (an input made later at the same address is another input) and never keeps
    it alive.
    """

    ident: int
    ref: weakref.ref[ReportInput]

    def __hash__(self) -> int:
        return hash(self.ident)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _ByIdentity) or other.ident != self.ident:
            return False
        held = self.ref()
        return held is not None and held is other.ref()


#: The most report inputs whose values are kept at once, the least recently read let go first: a sweep reads a copy
#: of its input per value, each read for a few keys, while the input it sweeps is read throughout.
_CACHED_INPUTS = 256


@functools.lru_cache(maxsize=_CACHED_INPUTS)
def _kept_of(held: _ByIdentity) -> _Kept:
    """Nothing kept yet, for one input (:func:`kept_for`)."""
    return _Kept()


def kept_for(report_input: ReportInput) -> _Kept:
    """What is kept of one report input, empty on its first read.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :return: Its kept values.
    :rtype: _Kept
    """
    return _kept_of(_ByIdentity(id(report_input), weakref.ref(report_input)))
