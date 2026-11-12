"""what a changed-rows read holds, as :func:`~threetears.core.keyed.changed.read_changed` compares it: a mapping of key
-> stored hash, or an Arrow table of the key columns and :data:`ROW_HASH_COLUMN`.

Both answer the same questions (each held group's hash, the held keys outside the answered groups, which answered keys
are held with the same hash and which held keys went unanswered) and give the same answers. The Arrow form never
builds a Python row of what it holds: each key's canonical text (:func:`~threetears.core.keyed.changed.key_text`) is
computed column by column in ``pyarrow.compute`` (a column of a type no kernel reads as the kind does is read one
distinct value at a time), a hash's 16 bytes are decoded from either spelling in numpy, and held keys are matched to
the answered ones by ``pc.index_in`` on one text a key. Only its answers (the gone keys) become Python values.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, Protocol

from threetears.interfaces.keyed.changed import ROW_HASH_COLUMN, GroupHash, HashColumn, HashKind

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    _ARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; a held Arrow table needs it, a held mapping does not
    _ARROW = False

__all__ = [
    "Held",
    "HeldArrow",
    "HeldMapping",
    "KeyText",
    "arrow_key_texts",
    "encoded_key",
    "group_hash",
    "hash_bytes",
    "same_hash",
]

#: the separator between a key's parts, and the marks of a NULL part and a present one, in one key's text
_SEPARATOR: Final = "\x1e"
_NULL_PART: Final = "\x00"
_PRESENT: Final = "\x01"

#: the canonical texts of a boolean, by what it may be held as
_TRUE_TEXTS: Final = ("t", "true", "1", "y", "yes")
_FALSE_TEXTS: Final = ("f", "false", "0", "n", "no")

#: the hex digits' and base64's values by byte (-1: no digit)
_HEX_DIGITS: Final = {
    **{digit: at for at, digit in enumerate("0123456789abcdef")},
    **{d: a for a, d in enumerate("0123456789ABCDEF")},
}
_B64_DIGITS: Final = {
    digit: at for at, digit in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
}


def _lookup(digits: Mapping[str, int]) -> Any:
    table = np.full(256, -1, dtype=np.int16)
    for digit, value in digits.items():
        table[ord(digit)] = value
    return table


_HEX: Final[Any] = _lookup(_HEX_DIGITS) if _ARROW else None
_B64: Final[Any] = _lookup(_B64_DIGITS) if _ARROW else None

KeyTexts = tuple[str | None, ...]


def encoded_key(texts: KeyTexts) -> str:
    """one key's canonical texts as one string (a NULL part kept apart from every text): what the Arrow form matches
    held keys to answered ones by."""
    return _SEPARATOR.join(_NULL_PART if text is None else _PRESENT + text for text in texts)


def hash_bytes(hashed: str) -> bytes:
    """the 16 bytes a hash's text spells: 32 hex digits or 22 characters of unpadded base64.

    :param hashed: the text
    :ptype hashed: str
    :return: its bytes
    :rtype: bytes
    :raises ValueError: for a text that is neither
    """
    try:
        if len(hashed) == 32:
            return bytes.fromhex(hashed)
        if len(hashed) == 22:
            spelled = base64.b64decode(hashed + "==", validate=True)
            if len(spelled) == 16 and base64.b64encode(spelled).decode().rstrip("=") == hashed:
                return spelled
    except (ValueError, binascii.Error) as refused:
        raise ValueError(f"a hash is 32 hex digits or 22 of unpadded base64, got {hashed!r}") from refused
    raise ValueError(f"a hash is 32 hex digits or 22 of unpadded base64, got {hashed!r}")


def same_hash(held: str | None, answered: Any) -> bool:
    """True when a held hash spells the same bytes as an answered one; False for none, or a held text no hash."""
    if held is None or answered is None:
        return False
    if held == answered:
        return True
    try:
        return hash_bytes(held) == hash_bytes(str(answered))
    except ValueError:  # NOSILENT: a held text that is no hash is a row to read again (False), as documented
        return False


def group_hash(hashes: Sequence[str | None] | None, lanes: Sequence[int], modulus: int) -> GroupHash | None:
    """the group hash of a held group's stored hashes: each lane the sum of one 4-byte slice of every hash; None (the
    group always moved) for a group not held, or holding a row with no hash or one that is no row hash."""
    if hashes is None or any(hashed is None for hashed in hashes):
        return None
    sums = [0] * len(lanes)
    try:
        for hashed in hashes:
            assert hashed is not None
            spelled = hash_bytes(hashed)
            for lane, start in enumerate(lanes):
                at = (start - 1) // 2
                sums[lane] += int.from_bytes(spelled[at : at + 4], "big")
    except ValueError:  # NOSILENT: a stored hash that is no row hash makes its group moved (None), as documented
        return None
    return GroupHash(rows=len(hashes), lanes=tuple(total % modulus for total in sums))  # type: ignore[arg-type]


class Held(Protocol):
    """what a changed-rows read holds, as it compares it."""

    def group_hashes(self, at: int) -> dict[str | None, GroupHash | None]:
        """each held group's text (the key's ``at``-th part) -> the group hash of its stored hashes (None: moved)."""
        ...

    def outside(self, at: int, answered: set[str | None]) -> tuple[list[tuple[Any, ...]], dict[str | None, Any]]:
        """the held keys whose group is none of ``answered``, in held order, and each such group's value as held."""
        ...

    def within(self, at: int, groups: set[str | None]) -> Held:
        """the held keys of ``groups`` alone."""
        ...

    def compare(self, answered: Sequence[KeyTexts], hashes: Sequence[Any]) -> tuple[list[bool], list[tuple[Any, ...]]]:
        """for each answered key (its texts) and hash, whether it is held with the same hash; and the held keys no
        answered key is, in held order."""
        ...


class HeldMapping:
    """held keys as a mapping: each key's canonical texts -> the key as held and its stored hash."""

    def __init__(self, by_text: dict[KeyTexts, tuple[tuple[Any, ...], str | None]], lanes: Sequence[int], modulus: int):
        self._by_text = by_text
        self._lanes = lanes
        self._modulus = modulus

    @classmethod
    def of(
        cls,
        held: Mapping[tuple[Any, ...], str | None],
        canonical: Callable[[Sequence[Any]], KeyTexts],
        lanes: Sequence[int],
        modulus: int,
    ) -> HeldMapping:
        return cls({canonical(key): (key, hashed) for key, hashed in held.items()}, lanes, modulus)

    def group_hashes(self, at: int) -> dict[str | None, GroupHash | None]:
        grouped: dict[str | None, list[str | None]] = {}
        for text, (_, hashed) in self._by_text.items():
            grouped.setdefault(text[at], []).append(hashed)
        return {text: group_hash(hashes, self._lanes, self._modulus) for text, hashes in grouped.items()}

    def outside(self, at: int, answered: set[str | None]) -> tuple[list[tuple[Any, ...]], dict[str | None, Any]]:
        vanished = [key for text, (key, _) in self._by_text.items() if text[at] not in answered]
        groups: dict[str | None, Any] = {}
        for text, (key, _) in self._by_text.items():
            if text[at] not in answered:
                groups.setdefault(text[at], key[at])
        return vanished, groups

    def within(self, at: int, groups: set[str | None]) -> HeldMapping:
        kept = {text: holding for text, holding in self._by_text.items() if text[at] in groups}
        return HeldMapping(kept, self._lanes, self._modulus)

    def compare(self, answered: Sequence[KeyTexts], hashes: Sequence[Any]) -> tuple[list[bool], list[tuple[Any, ...]]]:
        same = []
        for text, hashed in zip(answered, hashes, strict=True):
            holding = self._by_text.get(text)
            same.append(holding is not None and same_hash(holding[1], hashed))
        seen = set(answered)
        return same, [key for text, (key, _) in self._by_text.items() if text not in seen]


#: one value's canonical text (:func:`~threetears.core.keyed.changed.key_text`)
KeyText = Callable[[Any, HashKind, int | None], str | None]


def _texts_by_value(column: pa.Array, kind: HashKind, scale: int | None, key_text: KeyText) -> pa.Array:
    """a column's canonical texts, one distinct value at a time (a type no kernel reads as ``kind`` does)."""
    encoded = pc.dictionary_encode(column)
    texts = pa.array([key_text(value, kind, scale) for value in encoded.dictionary.to_pylist()], pa.string())
    return texts.take(encoded.indices)


def arrow_key_texts(column: pa.Array, kind: HashKind, scale: int | None, key_text: KeyText) -> pa.Array:
    """a held key column's canonical texts (NULL for NULL), as :func:`~threetears.core.keyed.changed.key_text`
    gives each value.

    :param column: the column
    :ptype column: pa.Array
    :param kind: its kind
    :ptype kind: HashKind
    :param scale: a NUMERIC's fractional digits
    :ptype scale: int | None
    :param key_text: one value's canonical text, for a type no kernel reads as ``kind`` does
    :ptype key_text: KeyText
    :return: the texts
    :rtype: pa.Array
    :raises ValueError: for a value that is no value of its kind
    """
    typed = column.type
    textual = pa.types.is_string(typed) or pa.types.is_large_string(typed)
    try:
        match kind:
            case HashKind.TEXT:
                if textual or pa.types.is_integer(typed):
                    return column.cast(pa.string())
            case HashKind.INTEGER:
                if pa.types.is_integer(typed):
                    return column.cast(pa.string())
                if pa.types.is_floating(typed):
                    if pc.any(pc.not_equal(column, pc.floor(column))).as_py():
                        raise ValueError(f"an integer key value is whole, got {column.type} values with fractions")
                    raise ValueError("an integer key value is whole, got a float")
            case HashKind.NUMERIC:
                if pa.types.is_decimal(typed) and (scale is None or typed.scale == scale):
                    return column.cast(pa.string())
            case HashKind.BOOLEAN:
                if pa.types.is_boolean(typed):
                    return pc.if_else(column, "t", "f")
                if textual:
                    lowered = pc.utf8_lower(pc.utf8_trim_whitespace(column))
                    true, false = pc.is_in(lowered, pa.array(_TRUE_TEXTS)), pc.is_in(lowered, pa.array(_FALSE_TEXTS))
                    if pc.any(pc.and_(pc.is_valid(column), pc.invert(pc.or_(true, false)))).as_py():
                        raise ValueError("a boolean key value is true or false")
                    return pc.if_else(true, "t", pc.if_else(false, "f", pa.scalar(None, pa.string())))
            case HashKind.DATE:
                if pa.types.is_timestamp(typed):
                    raise ValueError("a date key value is a date, got a time")
                if pa.types.is_date(typed):
                    return pc.strftime(column.cast(pa.timestamp("s")), format="%Y-%m-%d")
            case HashKind.TIMESTAMP:
                if pa.types.is_timestamp(typed) or pa.types.is_date(typed):
                    # a zoned time is held as UTC: dropping its zone keeps the UTC wall time; a naive one is UTC
                    naive = column.cast(pa.timestamp("us", tz=typed.tz)) if pa.types.is_timestamp(typed) else column
                    return pc.strftime(naive.cast(pa.timestamp("us")), format="%Y-%m-%d %H:%M:%S")
    except pa.ArrowInvalid as refused:
        raise ValueError(f"a {kind} key value is no value of its kind: {refused}") from refused
    return _texts_by_value(column, kind, scale, key_text)


def _hash_matrix(hashes: pa.Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """each hash's 16 bytes (a row of the matrix), whether it is a hash in either spelling, and whether it is held
    at all (not NULL)."""
    count = len(hashes)
    present = np.asarray(pc.is_valid(hashes).to_numpy(zero_copy_only=False), dtype=bool)
    text = pc.fill_null(hashes.cast(pa.string()), "")
    lengths = np.asarray(pc.binary_length(text).to_numpy(zero_copy_only=False), dtype=np.int64)
    fixed = pc.utf8_slice_codeunits(pc.utf8_rpad(text, width=32, padding=" "), 0, 32)
    # ASCII only is a hash: any other text's 32 code units are more bytes, and none is one
    ascii_only = np.asarray(pc.equal(pc.binary_length(fixed), 32).to_numpy(zero_copy_only=False), dtype=bool)
    fixed = pc.if_else(pa.array(ascii_only), fixed, pa.scalar(" " * 32))
    chars = np.frombuffer(fixed.cast(pa.binary(32)).buffers()[1], dtype=np.uint8)[: count * 32].reshape(count, 32)
    nibbles = _HEX[chars]
    is_hex = (lengths == 32) & (nibbles >= 0).all(axis=1)
    as_hex = ((nibbles[:, 0::2] << 4) | nibbles[:, 1::2]).astype(np.uint8)
    sextets = _B64[chars[:, :22]]
    padded = np.concatenate([np.clip(sextets, 0, 63), np.zeros((count, 2), dtype=np.int16)], axis=1)
    quads = padded.reshape(count, 6, 4).astype(np.uint32)
    words = (quads[:, :, 0] << 18) | (quads[:, :, 1] << 12) | (quads[:, :, 2] << 6) | quads[:, :, 3]
    spelled = np.stack([(words >> 16) & 0xFF, (words >> 8) & 0xFF, words & 0xFF], axis=2).reshape(count, 18)
    # canonical base64 of 16 bytes leaves the last character's low 4 bits zero
    is_b64 = (lengths == 22) & (sextets >= 0).all(axis=1) & ((np.clip(sextets[:, 21], 0, 63) & 0x0F) == 0)
    matrix = np.where(is_hex[:, None], as_hex, spelled[:, :16].astype(np.uint8))
    return matrix, (is_hex | is_b64) & ascii_only & present, present


class HeldArrow:
    """held keys as an Arrow table: the key columns (in the key's order) and :data:`ROW_HASH_COLUMN`."""

    def __init__(
        self,
        table: pa.Table,
        key_columns: Sequence[str],
        kinds: Sequence[HashColumn],
        lanes: Sequence[int],
        modulus: int,
        key_text: KeyText,
        *,
        texts: list[pa.Array] | None = None,
    ) -> None:
        missing = [name for name in (*key_columns, ROW_HASH_COLUMN) if name not in table.column_names]
        if missing:
            raise ValueError(f"a held table holds the key columns and {ROW_HASH_COLUMN}, missing {missing}")
        self._table = table
        self._key_columns = tuple(key_columns)
        self._kinds = tuple(kinds)
        self._lanes = lanes
        self._modulus = modulus
        self._key_text = key_text
        self._texts = (
            texts
            if texts is not None
            else [
                arrow_key_texts(table.column(name).combine_chunks(), kind.kind, kind.scale, key_text)
                for name, kind in zip(key_columns, kinds, strict=True)
            ]
        )
        parts = [
            pc.fill_null(pc.binary_join_element_wise(pa.scalar(_PRESENT), text, ""), _NULL_PART) for text in self._texts
        ]
        self._encoded = parts[0] if len(parts) == 1 else pc.binary_join_element_wise(*parts, _SEPARATOR)
        self._hashes = table.column(ROW_HASH_COLUMN).combine_chunks()

    def _keys(self, mask: np.ndarray) -> list[tuple[Any, ...]]:
        """the held keys of the rows ``mask`` keeps, as held: the answer's values, the only Python values built."""
        kept = self._table.select(list(self._key_columns)).filter(pa.array(mask))
        return list(zip(*(kept.column(name).to_pylist() for name in self._key_columns), strict=True))

    def _in(self, at: int, texts: set[str | None]) -> np.ndarray:
        named = pa.array(sorted(text for text in texts if text is not None), pa.string())
        inside = np.asarray(pc.is_in(self._texts[at], value_set=named).to_numpy(zero_copy_only=False), dtype=bool)
        if None in texts:
            inside |= np.asarray(pc.is_null(self._texts[at]).to_numpy(zero_copy_only=False), dtype=bool)
        return inside

    def group_hashes(self, at: int) -> dict[str | None, GroupHash | None]:
        if not len(self._encoded):
            return {}
        encoded = pc.dictionary_encode(self._texts[at], null_encoding="encode")
        groups = encoded.dictionary.to_pylist()
        indices = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
        matrix, valid, _ = _hash_matrix(self._hashes)
        rows = np.bincount(indices, minlength=len(groups))
        broken = np.bincount(indices, weights=(~valid).astype(np.int64), minlength=len(groups))
        sums = np.zeros((len(groups), len(self._lanes)), dtype=np.int64)
        for lane, start in enumerate(self._lanes):
            at_byte = (start - 1) // 2
            word = matrix[:, at_byte : at_byte + 4].astype(np.int64)
            values = (word[:, 0] << 24) | (word[:, 1] << 16) | (word[:, 2] << 8) | word[:, 3]
            np.add.at(sums[:, lane], indices, np.where(valid, values, 0) % self._modulus)
        answer: dict[str | None, GroupHash | None] = {}
        for place, group in enumerate(groups):
            answer[group] = (
                None
                if broken[place]
                else GroupHash(
                    rows=int(rows[place]),
                    lanes=tuple(int(total) % self._modulus for total in sums[place]),  # type: ignore[arg-type]
                )
            )
        return answer

    def outside(self, at: int, answered: set[str | None]) -> tuple[list[tuple[Any, ...]], dict[str | None, Any]]:
        mask = ~self._in(at, answered)
        if not mask.any():
            return [], {}
        vanished = self._keys(mask)
        texts = self._texts[at].filter(pa.array(mask)).to_pylist()
        groups: dict[str | None, Any] = {}
        for text, key in zip(texts, vanished, strict=True):
            groups.setdefault(text, key[at])
        return vanished, groups

    def within(self, at: int, groups: set[str | None]) -> HeldArrow:
        mask = pa.array(self._in(at, groups))
        return HeldArrow(
            self._table.filter(mask),
            self._key_columns,
            self._kinds,
            self._lanes,
            self._modulus,
            self._key_text,
            texts=[text.filter(mask) for text in self._texts],
        )

    def compare(self, answered: Sequence[KeyTexts], hashes: Sequence[Any]) -> tuple[list[bool], list[tuple[Any, ...]]]:
        answered_keys = pa.array([encoded_key(texts) for texts in answered], pa.string())
        place = pc.index_in(self._encoded, value_set=answered_keys)
        held_at = np.asarray(pc.fill_null(place, -1).to_numpy(zero_copy_only=False), dtype=np.int64)
        matched = held_at >= 0
        same = [False] * len(answered)
        if matched.any():
            held_matrix, held_valid, _ = _hash_matrix(self._hashes)
            answered_text = pa.array([None if hashed is None else str(hashed) for hashed in hashes], pa.string())
            answered_matrix, answered_valid, _ = _hash_matrix(answered_text)
            rows = np.nonzero(matched)[0]
            targets = held_at[rows]
            held_text = self._hashes.cast(pa.string()).take(pa.array(rows))
            equal_text = np.asarray(
                pc.fill_null(pc.equal(held_text, answered_text.take(pa.array(targets))), False).to_numpy(
                    zero_copy_only=False
                ),
                dtype=bool,
            )
            equal_bytes = (
                held_valid[rows] & answered_valid[targets] & (held_matrix[rows] == answered_matrix[targets]).all(axis=1)
            )
            for target in targets[equal_text | equal_bytes]:
                same[int(target)] = True
        return same, self._keys(~matched) if (~matched).any() else []
