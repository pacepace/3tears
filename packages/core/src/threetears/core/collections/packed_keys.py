"""keys in flight as packed batches: one string a batch of keys, never one object a key.

Core imports no Arrow (``tests/enforcement/test_core_imports_without_arrow.py``: the SDK's agents carry none), so the
keys a process's in-flight bookkeeping holds -- the L1 fence's writes in flight and the changes it records
(:class:`~threetears.core.collections.base.BaseCollection`), the write buffer's keys
(:class:`~threetears.core.collections.flush.WriteBuffer`) -- are columnar the way Arrow's string column is: every
key's text in one buffer, ``NUL text NUL text ... NUL``. A batch is immutable and its owner replaces it whole; whether it
names a key is one substring search (C, the batch's length), and a sorted batch with its offsets is searched by
bisection.

A key's text (:func:`key_text`) is exact -- two keys share a text only when they are equal -- and holds no NUL, so a
search never matches across two texts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

__all__ = [
    "NO_KEYS",
    "SEP",
    "key_text",
    "names",
    "packed",
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
