"""What one report input has been read as, kept with the input it was read off.

A report input is frozen and never changed in place: an input with a moved line is a copy
(:meth:`~pydantic.BaseModel.model_copy`), another object with a memo of its own. So anything that is pure in the input
(a key's figure, a page's forms, the stats it holds of a measure) can be worked out once per input and read back for
every later caller, which a cited figure's text, a page sized by halving and a listing ordered by its figures all are.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from threetears.evals.analysis.report_input import ReportInput


class InputMemo:
    """The values worked out of one report input, each built the first time it is asked for."""

    def __init__(self) -> None:
        self._values: dict[Hashable, object] = {}

    def get[V](self, key: Hashable, build: Callable[[], V]) -> V:
        """The value held under ``key``, built by ``build`` the first time.

        :param key: What the value is, unique among everything kept on one input (a class, or a tuple led by one).
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
    """A report input by its identity, held weakly: equal only to a handle on the same live input, so a memo never
    outlives its input's identity (an input made later at the same address is another input) and never keeps it alive.
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


#: The most report inputs whose memos are kept at once, the least recently read let go first: a sweep reads a copy of
#: its input per value, each read for a few keys, while the input it sweeps is read throughout.
_CACHED_INPUTS = 256


@functools.lru_cache(maxsize=_CACHED_INPUTS)
def _memo_of(held: _ByIdentity) -> InputMemo:
    """A fresh memo for one input (:func:`input_memo`)."""
    return InputMemo()


def input_memo(report_input: ReportInput) -> InputMemo:
    """The memo of one report input, empty on its first read.

    :param report_input: The frozen report input.
    :ptype report_input: ReportInput
    :return: Its memo.
    :rtype: InputMemo
    """
    return _memo_of(_ByIdentity(id(report_input), weakref.ref(report_input)))
