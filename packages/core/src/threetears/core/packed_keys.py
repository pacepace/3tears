"""keys in flight as packed batches: one string a batch of keys, never one object a key.

Core imports no Arrow (``tests/enforcement/test_core_imports_without_arrow.py``: the SDK's agents carry none), so the
keys a process's in-flight bookkeeping holds -- the L1 fence's tickets and writes in flight and the changes it
records (:class:`~threetears.core.collections.base.BaseCollection`), the write buffer's keys
(:class:`~threetears.core.collections.flush.WriteBuffer`) -- are columnar the way Arrow's string column is: every
key's text in one buffer, ``NUL text NUL text ... NUL``. A batch is immutable and its owner replaces it whole; whether it
names a key is one substring search (C, the batch's length), and a sorted batch with its offsets is searched by
bisection.

A key's text (:func:`key_text`) is exact -- two keys share a text only when they are equal -- and holds no NUL, so a
search never matches across two texts.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Final

__all__ = [
    "NO_KEYS",
    "SEP",
    "find_sorted",
    "key_text",
    "names",
    "offsets_of",
    "packed",
    "text_at",
    "texts",
    "with_text",
    "without_text",
]

#: the separator around every text of a batch; no text holds one
SEP: Final = "\x00"

#: the batch of no keys
NO_KEYS: Final = SEP

#: between a key's parts in its plain text
_PART: Final = "\x1f"

#: leads the text of a key a part of which holds a control character the plain text uses; that text is the key's
#: ``repr``, which writes every control character as an escape
_ESCAPED: Final = "\x1e"


def key_text(key: Sequence[str]) -> str:
    """``key``'s text in a batch: its parts joined by ``\\x1f``, or, when a part holds ``\\x00``, ``\\x1e`` or
    ``\\x1f`` (or the key has no part), ``\\x1e`` and its ``repr``.

    Exact: a plain text holds no ``\\x1e`` and splits back into its parts, an escaped one starts with ``\\x1e``, and
    neither holds a NUL. The plain form costs a join and three scans; a ``repr`` of every key made a 33,786-key
    eviction's batch 5x slower to build (2026-11-23).

    :param key: the key's parts, as text
    :ptype key: Sequence[str]
    :return: the text
    :rtype: str
    """
    text = _PART.join(key)
    if SEP in text or _ESCAPED in text or text.count(_PART) != len(key) - 1:
        return _ESCAPED + repr(tuple(key))
    return text


def packed(texts: Sequence[str]) -> str:
    """``texts`` as one batch.

    :param texts: each key's :func:`key_text`
    :ptype texts: Sequence[str]
    :return: the batch
    :rtype: str
    """
    if not texts:
        return NO_KEYS
    return f"{SEP}{SEP.join(texts)}{SEP}"


def names(batch: str, text: str) -> bool:
    """whether ``batch`` names the key whose text is ``text``.

    Batches joined end to end (``a + b``) are a batch too, naming the keys of both.

    :param batch: a batch (:func:`packed`)
    :ptype batch: str
    :param text: the key's :func:`key_text`
    :ptype text: str
    :return: ``True`` when it does
    :rtype: bool
    """
    return f"{SEP}{text}{SEP}" in batch


def with_text(batch: str, text: str) -> str:
    """``batch`` with the key whose text is ``text`` added (once more, when it already names it: a multiset).

    :param batch: a batch
    :ptype batch: str
    :param text: the key's :func:`key_text`
    :ptype text: str
    :return: the new batch
    :rtype: str
    """
    return f"{batch}{text}{SEP}"


def without_text(batch: str, text: str) -> str:
    """``batch`` with one of its entries of the key whose text is ``text`` taken out.

    :param batch: a batch naming the key
    :ptype batch: str
    :param text: the key's :func:`key_text`
    :ptype text: str
    :return: the new batch
    :rtype: str
    :raises ValueError: when ``batch`` does not name the key
    """
    at = batch.index(f"{SEP}{text}{SEP}")
    return batch[: at + 1] + batch[at + len(text) + 2 :]


def texts(batch: str) -> list[str]:
    """``batch``'s texts, in order (a list for the caller's own use, made by one split).

    :param batch: a batch
    :ptype batch: str
    :return: the texts
    :rtype: list[str]
    """
    return batch[1:-1].split(SEP) if len(batch) > 1 else []


def offsets_of(texts: Sequence[str]) -> array[int]:
    """where each text's leading NUL is in :func:`packed` ``(texts)``, and last where its closing NUL is.

    :param texts: the texts
    :ptype texts: Sequence[str]
    :return: the offsets, one more than the texts
    :rtype: array[int]
    """
    offsets = array("q", [0])
    at = 0
    for text in texts:
        at += len(text) + 1
        offsets.append(at)
    return offsets


def text_at(batch: str, offsets: array[int], row: int) -> str:
    """text ``row`` of ``batch``, by its :func:`offsets_of`.

    :param batch: a batch
    :ptype batch: str
    :param offsets: its offsets
    :ptype offsets: array[int]
    :param row: the text's place
    :ptype row: int
    :return: the text
    :rtype: str
    """
    return batch[offsets[row] + 1 : offsets[row + 1]]


def find_sorted(batch: str, offsets: array[int], text: str) -> int:
    """the place of ``text`` in a batch of sorted, distinct texts, by bisection.

    :param batch: a batch whose texts are sorted and distinct
    :ptype batch: str
    :param offsets: its :func:`offsets_of`
    :ptype offsets: array[int]
    :param text: the text sought
    :ptype text: str
    :return: its place, or ``-1`` when the batch does not hold it
    :rtype: int
    """
    low, high = 0, len(offsets) - 1
    count = high
    while low < high:
        middle = (low + high) // 2
        if batch[offsets[middle] + 1 : offsets[middle + 1]] < text:
            low = middle + 1
        else:
            high = middle
    return low if low < count and batch[offsets[low] + 1 : offsets[low + 1]] == text else -1
