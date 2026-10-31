"""answers computed once per version of their data, shared by every replica, and retired when the version moves.

A read whose answer is a pure function of (the data's version, the request) -- a report's rows at a
version of its copy, say -- need be computed once per version, by one replica, and read by the rest.
:class:`VersionedAnswers` is that cache: a :class:`~threetears.core.collections.derived.DerivedCollection`
keyed by ``(version, request digest)`` whose value is the answer gzip-compressed, held in L2 and in
each replica's L1.

- **Shared in L2, read from L1.** The shared ``{ns}-collections`` bucket, under the owner's key scope,
  is what every replica reads; there is no L3 (the answer is derived, and rebuilt on any miss). Each
  replica keeps the answers it read, and the index, in its L1 as well (when its registry offers one):
  an answer is a pure function of its key, so a copy is never stale, and :meth:`VersionedAnswers.answer`
  answers a key its L1 holds with no L2 round trip and no index write (whoever stored it recorded it).
- **Coherent by the epoch system, no lifetimes.** Both tables change only by compare-and-set, whose
  broadcast evicts every peer's copy, and keep their write generation in L2
  (:data:`~threetears.core.collections.generation.L2_WRITE_GENERATION`): a replica that missed a
  broadcast drops its copies at the next catch-up pass (``CollectionRegistry.catch_up_l2_generations``,
  run by ``threetears.epoch.generation_catchup_tick``), which its host schedules.
- **Computed once.** A miss computes under the in-process gate and the cross-pod build lock, so one
  replica derives a key and the others wait for its value. A tool pod passes a
  :class:`~threetears.core.collections.derived.LeaseBuildLock`: it may not declare a bucket.
- **Never expires; retired by version, oldest first.** An entry is the answer at one version, so it
  is never stale and carries no lifetime (owner ruling, 2026-10-08: caches are invalidated by the
  epoch system, never by a TTL). Each version comes with an ``order`` that grows with the data (the
  caller's: for a copy of epochs, their sum), because versions are digests and say nothing of which
  is newer. :meth:`current_version` retires every version of a LOWER order, never a higher one, so a
  replica still on an older copy cannot delete the answers the rest have moved on to. Each scope
  (``scope=``, for data whose versions are ordered on scales of their own: one state's copy and
  another's) is its own line of versions with its own floor, so one scope's move retires nothing of
  another's; a version belongs to one scope.
- **Retirement reads an index, never a listing.** A pod's grant on the shared bucket is key-addressed
  (get, put, compare-and-set, delete under its own scope; no consumer), so the entries cannot be
  listed. The owner keeps an index as rows of a small L2-only collection beside the answers, changed
  only by :meth:`~BaseCollection.l2_cas_mutate`: ``versions`` (each version's order, and a ``floor``
  below which no version is recorded any more) and, per version, sixteen shards of request digests
  (``{version}.{first hex digit}``), so no one value is rewritten by every request.
- **Race-free by one ordering.** A computing replica records its version (refused at or below the
  floor) and its digest before it computes, and writes its answer while it holds the key's build lock.
  Retirement raises the floor first, then, for each digest a shard names, deletes the answer -- or,
  when there is none yet, drops the digest only if it can pause the key's derivation (no writer holds
  the lock); a digest whose writer is still at work stays, and with it the version, for the next
  retirement. So an answer that is ever written is named by its digest until a retirement deletes it,
  whether its writer finishes, is cancelled or dies (its lease lapses and the next retirement takes
  the digest). A writer that finishes after its version was retired also takes its answer back at
  once. A version is forgotten only once every shard is empty.
- **A failure is never cached, and bookkeeping never fails a read.** ``compute`` raising (a conflict,
  a refusal, data not ready) reaches the caller and leaves nothing behind. The index failing (L2
  unreachable, contention past the retry budget) answers the read uncached and writes no entry, so
  nothing unindexed is ever stored.

- **Nothing left behind.** A KV delete writes a marker message, which at history 1 is the key's one
  message from then on, so a retirement that only deleted would leave one marker per retired answer
  forever. Once a version is retired (its answers and shards deleted by compare-and-set, the version
  forgotten), every key it touched is PURGED by the owner's ``purger``: a stream purge filtered to the
  key's exact subject, which leaves nothing. A pod holds no purge verb, so its purger asks the hub
  (:func:`threetears.nats.collection_key_requests.purge_pod_collection_keys`). With no purger, or one
  that cannot be reached (a hub older than the request), the markers stay and the miss is logged once:
  space, never a wrong answer. Purging happens only after the compare-and-set delete, on keys of a
  version below the floor, which no writer may record again.
- **Says how each answer was given.** :meth:`VersionedAnswers.answer_with_outcome` returns the
  :class:`AnswerOutcome` beside the answer: read, computed under the build lock, computed after a peer
  held it, or not stored (and why), so a host counts it without knowing how the lock is keyed.
- **An announced version only rises.** A host whose replicas each name a version in an index a CDN
  caches keeps one record per scope beside the index (:meth:`VersionedAnswers.announce`): the newest
  version any replica announced, by compare-and-set so it only rises, and the version it replaced.
  Every replica names the record's version once it is newer than its own, so the index never steps
  back across replicas, and retirement keeps the version replaced: a page that read the index a
  moment ago is still answered.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import gzip
import hashlib
import itertools
import json
import math
import re
from collections.abc import Awaitable, Buffer, Callable, Coroutine, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any, ClassVar, Final, Literal, SupportsIndex, cast

from pydantic import ValidationError
from sqlalchemy import MetaData
from threetears.core.collections.base import NATS_CLIENT_FROM_REGISTRY, BaseCollection
from threetears.core.collections.derived import BuildLock, BuildLockHeld, DerivedCollection
from threetears.core.collections.generation import L2_WRITE_GENERATION, WriteGenerationDeclaration
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
)
from threetears.core.collections.tier_reads import ReadTier, record_tier_read
from threetears.core.config import CoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import ConcurrentModificationError, GenerationUnavailableError
from threetears.nats.collection_key_requests import MAX_PURGED_KEYS, CollectionKeysRequestError
from threetears.nats.errors import KvError
from threetears.observe import contain, get_logger

__all__ = [
    "Announced",
    "AnnouncedVersion",
    "AnswerNotComputable",
    "AnswerOutcome",
    "GivenAnswer",
    "VersionedAnswer",
    "VersionedAnswers",
    "answer_json",
]

log = get_logger(__name__)

#: a version as a key segment: KV-grammar characters without ``_`` (l2_key joins a composite key's
#: parts with it) or ``.`` (a shard's name is ``{version}.{digit}``)
_VERSION: Final = re.compile(r"^[-=a-zA-Z0-9]+$")

#: the name of the index row holding each version's order and the floor, for the default scope; another
#: scope's row is ``versions_{scope}`` (a version holds no ``_``, so no shard and no other row is named so)
_VERSIONS: Final = "versions"


def _versions_row(scope: str) -> str:
    """the index row holding ``scope``'s versions and floor.

    :param scope: the scope, ``""`` for the default
    :ptype scope: str
    :return: the row's name
    :rtype: str
    """
    return _VERSIONS if not scope else f"{_VERSIONS}_{scope}"


#: what a write to the index can raise: transport, a grant refusal, the retry budget spent, or a write
#: that landed and could not advance its table's write generation
_INDEX_ERRORS: Final = (KvError, ConcurrentModificationError, GenerationUnavailableError)

#: the index row holding a scope's announced version is ``announced_{scope}``: a version holds no ``_``,
#: so it is neither ``versions`` nor a shard
_ANNOUNCED: Final = "announced_"

_Action = tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]

#: purges keys (relative to the owner's scope) so they leave no marker; returns how many it purged.
#: It raises :class:`CollectionKeysRequestError` when the purge did not happen (no hub answered, the hub
#: refused or failed) and pydantic's ``ValidationError`` when a key breaks the request's bounds, as
#: ``purge_pod_collection_keys`` does; the markers then stay. Anything else it raises is a bug, and
#: reaches whoever runs the retirement.
KeyPurger = Callable[[list[str]], Awaitable[int]]

#: what a :data:`KeyPurger` raises when the markers merely stay
_PURGE_ERRORS: Final = (CollectionKeysRequestError, ValidationError)


#: the shards of each version's digest index
_SHARDS: Final = "0123456789abcdef"


class AnswerNotComputable(LookupError):
    """a read reached ``compute`` without :meth:`VersionedAnswers.answer`: only it knows how to compute."""


class AnswerOutcome(StrEnum):
    """how one answer was given.

    :cvar READ: read from the cache (this replica's L1, or L2), computed by nobody here
    :cvar COMPUTED: computed here under the key's build lock, and stored
    :cvar PEER_FALLBACK: computed here after a peer held the build lock and its value never landed
        (it finished as this replica retook the lock, or it died and its lease lapsed), and stored
    :cvar RETIRED: computed here and not stored: its version was retired, before the compute or while it ran
    :cvar INDEX_UNWRITABLE: computed here and not stored: the index could not record it
    """

    READ = "read"
    COMPUTED = "computed"
    PEER_FALLBACK = "peer_fallback"
    RETIRED = "retired"
    INDEX_UNWRITABLE = "index_unwritable"

    @property
    def stored(self) -> bool:
        """whether the answer is in the cache for the next read.

        :return: ``False`` for :attr:`RETIRED` and :attr:`INDEX_UNWRITABLE`
        :rtype: bool
        """
        return self not in (AnswerOutcome.RETIRED, AnswerOutcome.INDEX_UNWRITABLE)


@dataclass(frozen=True)
class GivenAnswer:
    """an answer, and how it was given.

    :ivar body: the answer, gzip bytes
    :ivar outcome: how it was given
    """

    body: bytes
    outcome: AnswerOutcome


@dataclass(frozen=True)
class AnnouncedVersion:
    """a version a replica named in an index, with whatever its host keeps beside it.

    :ivar version: the version
    :ivar order: where it stands among the data's versions: higher is newer
    :ivar detail: the host's own JSON beside it (what a replica following it needs to answer at it)
    """

    version: str
    order: int
    detail: Mapping[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        """the version as the record stores it.

        :return: ``version``, ``order`` and ``detail``
        :rtype: dict[str, Any]
        """
        return {"version": self.version, "order": self.order, "detail": dict(self.detail)}


@dataclass(frozen=True)
class Announced:
    """a scope's announced-version record: the newest version any replica announced, and the one it replaced.

    :ivar latest: the newest version announced
    :ivar previous: the version it replaced; ``None`` for a scope's first
    """

    latest: AnnouncedVersion
    previous: AnnouncedVersion | None = None

    @property
    def versions(self) -> tuple[AnnouncedVersion, ...]:
        """the versions a page may be reading at because an index named them: the newest, then the one before.

        :return: the latest, then the previous when there is one
        :rtype: tuple[AnnouncedVersion, ...]
        """
        return (self.latest,) if self.previous is None else (self.latest, self.previous)

    def as_json(self) -> dict[str, Any]:
        """the record as the index stores it.

        :return: ``latest`` and ``previous``
        :rtype: dict[str, Any]
        """
        return {"latest": self.latest.as_json(), "previous": None if self.previous is None else self.previous.as_json()}


def _announced_version(body: Any) -> AnnouncedVersion | None:
    """a stored version, or ``None`` for none.

    :raises KeyError, TypeError, ValueError: when it is not one
    """
    if body is None:
        return None
    return AnnouncedVersion(str(body["version"]), int(body["order"]), dict(body.get("detail") or {}))


def _announced_from(row: dict[str, Any] | None, table: str) -> Announced | None:
    """the record a row holds; ``None`` for none, and for one that cannot be read (it is then replaced).

    :param row: the index row, or ``None``
    :ptype row: dict[str, Any] | None
    :param table: the index's table, for the log
    :ptype table: str
    :return: the record
    :rtype: Announced | None
    """
    if row is None:
        return None
    try:
        body = row["members"]
        latest = _announced_version(body["latest"])
        return None if latest is None else Announced(latest, _announced_version(body.get("previous")))
    except (KeyError, TypeError, ValueError) as exc:
        # NOSILENT: an unreadable record is replaced by the next announcement
        log.warning("an announced-version record could not be read; it is replaced: table=%s error=%s", table, exc)
        return None


@dataclass
class _Retirement:
    """one scope's background retirement on this replica.

    :ivar retired_below: the highest order this replica has retired below, or ``None``
    :ivar wanted: the highest order it has been asked to retire below, or ``None``
    :ivar task: the retirement running, or ``None``
    """

    retired_below: int | None = None
    wanted: int | None = None
    task: asyncio.Task[None] | None = None


@dataclass
class _Pending:
    """the compute one :meth:`VersionedAnswers.answer` call brought, whether it ran, and after a peer."""

    compute: Callable[[], Awaitable[str | bytes | bytearray]]
    ran: bool = False
    after_peer: bool = False


#: the compute of the :meth:`VersionedAnswers.answer` call this task is inside, for ``compute`` to find:
#: a context variable, so concurrent callers never see each other's and a cancelled one leaves nothing
_PENDING: ContextVar[_Pending | None] = ContextVar("versioned_answer_pending", default=None)


class VersionedAnswer(BaseEntity):
    """one answer: ``version``, ``request`` (a digest), ``body`` (base64 gzip)."""

    primary_key_field: str = "request"


#: the most list items :func:`answer_json` writes in one encoder call
_JSON_PIECE: Final = 256

#: the most work :func:`answer_json` gives one encoder call, in units of :func:`_heft`: a value, a
#: mapping's key, or 64 characters of text each one. The C encoder writes about ten thousand a millisecond (a row of
#: twenty mixed values is about forty), so a call holds the interpreter lock for a millisecond or two
_JSON_BUDGET: Final = 8192

#: how many of a list's items :func:`answer_json` weighs whole to learn how deep its items run
_JSON_SAMPLES: Final = 16

#: the characters of text one unit of :func:`_heft` stands for
_JSON_TEXT_UNIT: Final = 64


def _json_default(value: Any) -> Any:
    """what an answer's JSON writes for a value ``json`` has no type for.

    A ``Decimal`` is a JSON number, the one its ``float`` writes: a reader parses a number into a
    double either way, and an exact L1 hands back the ``Decimal`` a float copy used to hold. A
    ``NaN`` or infinite one, which JSON has no number for, is ``null``. Anything else is its ``str``.

    :param value: the value
    :ptype value: Any
    :return: the value ``json`` writes in its place
    :rtype: Any
    """
    written: Any = str(value)
    if isinstance(value, Decimal):
        written = float(value) if value.is_finite() else None
    return written


def _finite(value: Any) -> Any:
    """``value`` with every ``NaN`` or infinite ``float`` in it, at any depth, replaced by ``None``.

    :param value: an answer, or a part of one
    :ptype value: Any
    :return: the same structure, every float in it finite or ``None``
    :rtype: Any
    """
    result: Any = value
    if isinstance(value, float) and not math.isfinite(value):
        result = None
    elif isinstance(value, dict):
        result = {key: _finite(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = [_finite(item) for item in value]
    return result


def _dumps(value: Any) -> str:
    """``value`` as JSON, written by :func:`_json_default`, with a non-finite float as ``null``.

    ``json`` writes a ``NaN`` or infinite float as ``NaN`` or ``Infinity``, which no JSON parser
    reads. The common answer holds none and is written in one pass; one that does is written again
    with each replaced by ``null``.

    :param value: the value
    :ptype value: Any
    :return: its JSON
    :rtype: str
    """
    try:
        text = json.dumps(value, default=_json_default, allow_nan=False)
    except ValueError:
        text = json.dumps(_finite(value), default=_json_default, allow_nan=False)
    return text


def _heft(value: Any, cap: int) -> int:
    """the work of writing ``value`` as JSON, in :data:`_JSON_BUDGET`'s units, counted until it passes ``cap``.

    A list or mapping longer than :data:`_JSON_PIECE` counts as past ``cap`` whatever it holds, so no one call
    writes one.

    :param value: an answer, or a part of one
    :ptype value: Any
    :param cap: where counting stops
    :ptype cap: int
    :return: the work, or more than ``cap`` once it passes
    :rtype: int
    """
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            total += 1 + len(item) // _JSON_TEXT_UNIT
        elif isinstance(item, dict):
            if len(item) > _JSON_PIECE or len(item) > cap - total:
                return cap + 1
            total += 1 + len(item)
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            if len(item) > _JSON_PIECE or len(item) > cap - total:
                return cap + 1
            total += 1
            stack.extend(item)
        else:
            total += 1
        if total > cap:
            return total
    return total


def _key_text(key: Any) -> str:
    """a mapping's key as ``json.dumps`` writes it: text quoted, a number, ``true``/``false`` or ``null`` as text.

    :param key: the key
    :ptype key: Any
    :return: the key's JSON, quoted
    :rtype: str
    :raises TypeError: for a key ``json`` refuses (a tuple, say)
    """
    if isinstance(key, str):
        return json.dumps(key)
    # json turns a number, a bool or None into text the way it writes any mapping's key
    return json.dumps({key: 0}, allow_nan=False)[1:-4]


#: the types whose values are one unit each, whatever they hold
_JSON_SCALARS: Final = frozenset({int, float, bool, type(None)})

#: the scalars and text
_JSON_FLAT: Final = _JSON_SCALARS | {str}


def _flat_work(held: list[Any]) -> int | None:
    """a run's work (:func:`_heft`'s units) when its items hold nothing deeper than plain values and
    text, counted by C-level passes over it (``map``, ``chain``), never an item at a time; None for any other run.

    Text held inside an item counts as a unit, however long: how long a list's text runs is weighed from the
    samples that size its runs (:func:`_write_run`). What this catches is an item deeper than the samples -- a row
    holding a list or a mapping -- which is then weighed whole.

    :param held: the run's items, or its entries' values
    :ptype held: list[Any]
    :return: the work, or None when an item holds a list or a mapping, or is of a type this does not weigh
    :rtype: int | None
    """
    kinds = set(map(type, held))
    work: int | None = None
    if kinds <= _JSON_SCALARS:
        work = len(held)
    elif kinds == {str}:
        work = len(held) + sum(map(len, held)) // _JSON_TEXT_UNIT
    elif (kinds <= {list, tuple} or kinds == {dict}) and max(map(len, held)) <= _JSON_PIECE:
        mappings = kinds == {dict}
        inner = itertools.chain.from_iterable(map(dict.values, held) if mappings else held)
        if set(map(type, inner)) <= _JSON_FLAT:
            # a mapping's key is a unit beside its value, as in _heft
            work = len(held) + sum(map(len, held)) * (2 if mappings else 1)
    return work


class _Run:
    """a list's items, or a mapping's entries, written into the answer's text as pieces with a comma between.

    :param out: the answer's text so far
    :ptype out: _AnswerJson
    :param mapping: whether the items are a mapping's ``(key, value)`` entries
    :ptype mapping: bool
    """

    def __init__(self, out: _AnswerJson, *, mapping: bool) -> None:
        self.out = out
        self.mapping = mapping
        self.started = False

    def _comma(self) -> None:
        if self.started:
            self.out.append(", ")
        self.started = True

    def dump(self, items: list[Any]) -> None:
        """write ``items`` in one encoder call.

        :param items: items, or entries
        :ptype items: list[Any]
        :return: nothing
        :rtype: None
        """
        if items:
            self._comma()
            self.out.append(_dumps(dict(items) if self.mapping else items)[1:-1])

    def open(self, item: Any) -> None:
        """write one item too large for a run by :func:`_write`, which opens it.

        :param item: an item, or an entry
        :ptype item: Any
        :return: nothing
        :rtype: None
        """
        self._comma()
        if self.mapping:
            key, item = item
            self.out.append(_key_text(key))
            self.out.append(": ")
        _write(item, self.out)

    def weighed(self, items: list[Any], values: list[Any]) -> None:
        """write ``items`` weighing each whole (:func:`_heft`): together until the next would pass the
        budget, and one past the budget alone, opened.

        :param items: items, or entries
        :ptype items: list[Any]
        :param values: the items, or the entries' values
        :ptype values: list[Any]
        :return: nothing
        :rtype: None
        """
        start = 0
        weight = 0
        for index, value in enumerate(values):
            cost = _heft(value, _JSON_BUDGET)
            if index > start and weight + cost > _JSON_BUDGET:
                self.dump(items[start:index])
                start, weight = index, 0
            if cost > _JSON_BUDGET:
                self.open(items[index])
                start, weight = index + 1, 0
            else:
                weight += cost
        self.dump(items[start:])


def _write_run(value: dict[Any, Any] | list[Any] | tuple[Any, ...], out: _AnswerJson) -> None:
    """a mapping or a list too large for one encoder call, written into ``out`` a run of entries or items at a time.

    A few items, spread through it, are weighed whole, and a run is as many items as half the budget holds at the
    heaviest of them, never more than :data:`_JSON_PIECE`. A run whose items hold only plain values and text is
    weighed by C-level passes (:func:`_flat_work`) and, within the budget, written in one encoder call: rows,
    as lists or as mappings, cost no Python an item. Any other run -- items holding lists or mappings, or heavier
    than the samples said -- is weighed an item at a time (:meth:`_Run.weighed`), and an item past the budget is
    opened in turn.

    :param value: the mapping or list
    :ptype value: dict[Any, Any] | list[Any] | tuple[Any, ...]
    :param out: the text so far
    :ptype out: _AnswerJson
    :return: nothing
    :rtype: None
    """
    mapping = isinstance(value, dict)
    count = len(value)
    # pointers copied, never an item made: a mapping's entries are taken a run at a time from its own iterator
    values: list[Any] = (
        list(value.values()) if isinstance(value, dict) else value if type(value) is list else list(value)
    )
    entries = iter(value.items()) if isinstance(value, dict) else None
    stride = max(1, count // _JSON_SAMPLES)
    heaviest = max(_heft(values[index], _JSON_BUDGET) for index in (*range(0, count, stride), count - 1))
    # half the budget at the heaviest sample: a run of items a little heavier than the samples still fits
    size = max(1, min(_JSON_PIECE, _JSON_BUDGET // (2 * max(1, heaviest))))
    run = _Run(out, mapping=mapping)
    out.append("{" if mapping else "[")
    for start in range(0, count, size):
        held = values[start : start + size]
        items = held if entries is None else list(itertools.islice(entries, size))
        work = _flat_work(held)
        if work is not None and work <= _JSON_BUDGET:
            run.dump(items)
        else:
            run.weighed(items, held)
    out.append("}" if mapping else "]")


def _write(value: Any, out: _AnswerJson) -> None:
    """``value``'s JSON appended to ``out``: in one encoder call when it is within :data:`_JSON_BUDGET`, else a
    run of entries or items at a time (:func:`_write_run`).

    :param value: an answer, or a part of one
    :ptype value: Any
    :param out: the text so far
    :ptype out: _AnswerJson
    :return: nothing
    :rtype: None
    """
    if isinstance(value, (dict, list, tuple)) and _heft(value, _JSON_BUDGET) > _JSON_BUDGET:
        _write_run(value, out)
    else:
        out.append(_dumps(value))


#: the most characters :meth:`_AnswerJson.__eq__` compares in one call: a slice of a large answer, never all of it
_COMPARED_SLICE: Final = 1 << 20


class _AnswerJson(bytearray):
    """an answer's JSON: ``json.dumps``' text, which is all ASCII, held as its bytes, one byte a character.

    :func:`answer_json` appends each piece as it is written (:meth:`append`), so no one call ever copies the whole
    answer: a ``"".join`` of the pieces into one ``str`` of 51 MB held the interpreter lock for 5 ms on a laptop
    and 32-40 ms on a 4-core CI runner (fresh pages faulted in, one copy), and an ``encode`` of that ``str`` for
    gzip was a second copy as long. This grows a few kilobytes a call, and its bytes go to gzip (``zlib`` lets go
    of the lock while it compresses) as they are.

    It reads as the text too: it equals the ``str`` it holds (compared a slice at a time), ``in`` finds text in
    it, ``len`` is its characters, :meth:`encode` to UTF-8 or ASCII is itself, and ``json.loads`` and ``str``
    read it (each a copy of the whole: for a small answer, or a test).
    """

    def append(self, piece: str) -> None:  # type: ignore[override]
        """add ``piece``, a run of the answer's JSON, to the end.

        :param piece: JSON text, ASCII
        :ptype piece: str
        :return: nothing
        :rtype: None
        """
        self.extend(piece.encode("ascii"))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            size = len(self)
            return len(other) == size and all(
                self[start : start + _COMPARED_SLICE] == other[start : start + _COMPARED_SLICE].encode("utf-8")
                for start in range(0, size, _COMPARED_SLICE)
            )
        return bytearray.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __contains__(self, key: object, /) -> bool:
        # the text is ASCII, so text that is not never matches its UTF-8; anything else is bytearray's to refuse
        needle = key.encode("utf-8") if isinstance(key, str) else key
        return super().__contains__(cast("SupportsIndex | Buffer", needle))

    def __str__(self) -> str:
        return self.decode("ascii")

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes | bytearray:
        """the text in ``encoding``: itself for UTF-8, ASCII or Latin-1, which spell ASCII as ASCII.

        :param encoding: the encoding
        :ptype encoding: str
        :param errors: as ``str.encode``'s
        :ptype errors: str
        :return: the encoded text
        :rtype: bytes | bytearray
        """
        if codecs.lookup(encoding).name in _ASCII_SUPERSETS:
            return self
        return str(self).encode(encoding, errors)


#: the codecs that write ASCII text as its ASCII bytes
_ASCII_SUPERSETS: Final = frozenset({"utf-8", "ascii", "latin-1", "iso8859-1"})


def answer_json(answer: Any) -> _AnswerJson:
    """``json.dumps(answer, default=str)``, the same text, written a piece at a time; a ``Decimal`` is a number.

    The C JSON encoder holds the interpreter lock for its whole call, so an answer of megabytes written in
    one call holds every other thread for as long, the event loop among them, whichever thread writes it.
    Each piece here is a call of at most :data:`_JSON_BUDGET` (a millisecond or two) and never more than
    :data:`_JSON_PIECE` list items, and the loop runs between them: a mapping or a list too large for one
    call is written a run of entries or items at a time (whatever its keys' types), and an item too large for a
    run is opened in turn, however deep it sits (a short list of long series, say). The bytes are ``json``'s
    own: each run is one ``json.dumps`` call with its brackets taken off. Each piece is added to the answer as
    it is written, so nothing ever copies the whole answer in one call (:class:`_AnswerJson`).

    :param answer: the answer
    :ptype answer: Any
    :return: its JSON, exactly as ``json.dumps`` writes it with :func:`_json_default`, but a ``NaN``
        or infinite number, ``float`` or ``Decimal``, is ``null``: JSON has no number for it
    :rtype: _AnswerJson
    """
    out = _AnswerJson()
    _write(answer, out)
    return out


def _gzipped(text: str | bytes | bytearray) -> bytes:
    """an answer's text gzipped with no time in its header, so equal answers are equal bytes.

    Run on a worker thread (``asyncio.to_thread``), never the event loop: a large answer takes tens of
    milliseconds, and zlib lets go of the interpreter lock while it compresses. Bytes (:func:`answer_json`'s)
    are compressed as they are, never copied first.

    :param text: the answer's text
    :ptype text: str | bytes | bytearray
    :return: the gzip bytes
    :rtype: bytes
    """
    return gzip.compress(text if isinstance(text, bytes | bytearray) else text.encode("utf-8"), mtime=0)


def _row_body(text: str | bytes | bytearray) -> str:
    """an answer's text as its row keeps it: gzipped (:func:`_gzipped`), base64. Run on a worker thread.

    :param text: the answer's text
    :ptype text: str | bytes | bytearray
    :return: the row's ``body``
    :rtype: str
    """
    return base64.b64encode(_gzipped(text)).decode("ascii")


def _answers_schema(table_name: str) -> TableSchema:
    """the L1 table of the answers: every column either tier writes (the row's, and the times a swap stamps).

    :param table_name: the answers' table
    :ptype table_name: str
    :return: the schema
    :rtype: TableSchema
    """
    return TableSchema(
        name=table_name,
        primary_key=("version", "request"),
        columns=[
            Column("version", STRING_TYPE),
            Column("request", STRING_TYPE),
            Column("body", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, nullable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
    )


def _index_schema(table_name: str) -> TableSchema:
    """the L1 table of an index: a row's name, its members (JSON) and, on ``versions``, the floor.

    :param table_name: the index's table
    :ptype table_name: str
    :return: the schema
    :rtype: TableSchema
    """
    return TableSchema(
        name=table_name,
        primary_key="name",
        columns=[
            Column("name", STRING_TYPE),
            Column("members", JSONB_TYPE, nullable=True),
            Column("floor", BIGINT_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, nullable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
    )


def _l1_metadata(schema: TableSchema) -> Any:
    """the metadata an L1 backend is initialised with for one table.

    :param schema: the table
    :ptype schema: TableSchema
    :return: SQLAlchemy metadata holding it
    :rtype: Any
    """
    return schema.to_sqlalchemy_table(MetaData()).metadata


class _IndexRow(BaseEntity):
    """one row of the index: ``name``, ``members``, and on ``versions`` the ``floor``."""

    primary_key_field: str = "name"


class _AnswerIndex(BaseCollection[_IndexRow]):
    """the index of a :class:`VersionedAnswers`: rows in L2 and L1, changed only by compare-and-set."""

    primary_key_column: str | tuple[str, ...] = ("name",)
    #: a copy in L1 is coherent: every change is a compare-and-set, whose broadcast evicts the peers'
    #: copies, and the generation kept beside the rows catches a broadcast missed
    caches_in_l1: ClassVar[bool] = True
    write_generation: ClassVar[WriteGenerationDeclaration] = L2_WRITE_GENERATION
    #: derived: the index is rebuilt by the answers it indexes, and it changes only by
    #: compare-and-set, which raises on its own
    l2_is_the_store: bool = False

    def __init__(self, registry: CollectionRegistry, config: CoreConfig, nats_client: Any, *, table_name: str) -> None:
        self._table_name = table_name
        super().__init__(registry, config, nats_client, None)
        # L2 alone: the index is rebuilt by the answers it indexes, never read through from a store
        self.l3_pool = None
        if self._l1 is not None:
            self._l1.initialize(_l1_metadata(_index_schema(table_name)))

    async def read_l2(self, name: str) -> dict[str, Any] | None:
        """one row as L2 holds it, past this replica's L1: for retirement, which must see every digest.

        :param name: the row's name
        :ptype name: str
        :return: the row, or ``None`` when L2 holds none or cannot be read
        :rtype: dict[str, Any] | None
        """
        return await self._get_from_l2(name)

    @property
    def table_name(self) -> str:
        """the table the index's keys are named by.

        :return: the table name
        :rtype: str
        """
        return self._table_name

    @property
    def entity_class(self) -> type[_IndexRow]:
        """the entity class.

        :return: :class:`_IndexRow`
        :rtype: type[_IndexRow]
        """
        return _IndexRow

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """nothing: there is no store behind the index.

        :param entity_id: unused
        :ptype entity_id: Any
        :return: ``None``
        :rtype: dict[str, Any] | None
        """
        del entity_id
        return None

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """nothing: there is no store behind the index.

        :param data: unused
        :ptype data: dict[str, Any]
        :param original_timestamp: unused
        :ptype original_timestamp: Any
        :param conn: unused
        :ptype conn: Any
        :return: ``0``
        :rtype: int
        """
        del data, original_timestamp, conn
        return 0

    async def delete_from_store(self, entity_id: Any) -> None:
        """nothing: there is no store behind the index.

        :param entity_id: unused
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        del entity_id

    def serialize(self, data: dict[str, Any]) -> bytes:
        """a row as L2 holds it: JSON.

        :param data: the row
        :ptype data: dict[str, Any]
        :return: JSON bytes
        :rtype: bytes
        """
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """a row back from L2.

        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        return row


class VersionedAnswers(DerivedCollection[VersionedAnswer]):
    """gzip answers keyed by ``(version, request digest)``, in L2 and L1, retired when the version moves.

    Without an L1 backend on the registry the answers and index are kept in L2 alone.

    :param registry: the owner's registry, carrying its L2 client and key scope, and the L1 backend each
        table binds, when it keeps one
    :ptype registry: CollectionRegistry
    :param config: core configuration
    :ptype config: CoreConfig
    :param nats_client: the L2 client; the registry's when omitted
    :ptype nats_client: Any
    :param table_name: the table the keys are named by: one per kind of answer (the index is
        ``{table_name}_index``)
    :ptype table_name: str
    :param build_lock: the cross-pod build lock; a tool pod passes a ``LeaseBuildLock``
    :ptype build_lock: BuildLock | None
    :param purger: purges retired keys so they leave no marker; a tool pod passes one asking the hub.
        None keeps the markers (logged once)
    :ptype purger: KeyPurger | None
    """

    primary_key_column: str | tuple[str, ...] = ("version", "request")
    #: an answer is a pure function of its key, so a copy in L1 is never stale; retirement deletes by
    #: compare-and-set, whose broadcast evicts the peers' copies
    caches_in_l1: ClassVar[bool] = True
    write_generation: ClassVar[WriteGenerationDeclaration] = L2_WRITE_GENERATION
    #: a miss that finds a peer's answer reads it from L2, where answers live
    derived_values_tier: ClassVar[ReadTier] = ReadTier.L2
    #: derived: a miss computes the answer again, and answers are written by compare-and-set,
    #: which raises on its own
    l2_is_the_store: bool = False

    def __init__(
        self,
        registry: CollectionRegistry,
        config: CoreConfig,
        nats_client: Any = NATS_CLIENT_FROM_REGISTRY,
        *,
        table_name: str,
        build_lock: BuildLock | None = None,
        purger: KeyPurger | None = None,
    ) -> None:
        self._table_name = table_name
        self._purger = purger
        self._markers_kept_logged = False
        self._background: set[asyncio.Task[None]] = set()
        super().__init__(registry, config, nats_client, None, build_lock=build_lock)
        # no durable tier to pull through (a miss computes)
        self.l3_pool = None
        if self._l1 is not None:
            self._l1.initialize(_l1_metadata(_answers_schema(table_name)))
        self._index = _AnswerIndex(registry, config, nats_client, table_name=f"{table_name}_index")
        #: per scope, this replica's background retirement
        self._retirements: dict[str, _Retirement] = {}

    @property
    def table_name(self) -> str:
        """the table the answers' keys are named by.

        :return: the table name
        :rtype: str
        """
        return self._table_name

    @property
    def entity_class(self) -> type[VersionedAnswer]:
        """the entity class.

        :return: :class:`VersionedAnswer`
        :rtype: type[VersionedAnswer]
        """
        return VersionedAnswer

    # ------------------------------------------------------------------
    # the public surface
    # ------------------------------------------------------------------

    async def answer(
        self,
        version: str,
        request: str,
        compute: Callable[[], Awaitable[str | bytes | bytearray]],
        *,
        order: int,
        scope: str = "",
    ) -> bytes:
        """the answer to ``request`` at ``version``, gzip-compressed: cached, or computed once and cached.

        :meth:`answer_with_outcome`, without the outcome.

        :param version: the data's version the answer is at
        :ptype version: str
        :param request: the request, whole (digested for the key): two requests asking the same thing
            must spell it the same
        :ptype request: str
        :param compute: computes the answer's JSON at exactly ``version``, as text or as its UTF-8 bytes
            (:func:`answer_json`'s, a ``bytearray``); it raises rather than answer at another, and whatever
            it raises reaches the caller uncached
        :ptype compute: Callable[[], Awaitable[str | bytes | bytearray]]
        :param order: where ``version`` stands among the data's versions: higher is newer
        :ptype order: int
        :param scope: the line of versions ``version`` belongs to: each scope's orders are compared and
            retired on their own, so one scope's move never retires another's answers
        :ptype scope: str
        :return: the answer, gzip bytes
        :rtype: bytes
        :raises ValueError: when ``version`` cannot be a key segment
        """
        return (await self.answer_with_outcome(version, request, compute, order=order, scope=scope)).body

    async def answer_with_outcome(
        self,
        version: str,
        request: str,
        compute: Callable[[], Awaitable[str | bytes | bytearray]],
        *,
        order: int,
        scope: str = "",
    ) -> GivenAnswer:
        """the answer to ``request`` at ``version``, gzip-compressed, and how it was given.

        A key this replica's L1 holds is answered from it: no L2 round trip and no index write, since
        whoever stored the answer recorded it. Anything else records the request in the index, then
        reads L2 or computes.

        :param version: the data's version the answer is at
        :ptype version: str
        :param request: the request, whole (digested for the key): two requests asking the same thing
            must spell it the same
        :ptype request: str
        :param compute: computes the answer's JSON at exactly ``version``, as text or as its UTF-8 bytes
            (:func:`answer_json`'s, a ``bytearray``); it raises rather than answer at another, and whatever
            it raises reaches the caller uncached
        :ptype compute: Callable[[], Awaitable[str | bytes | bytearray]]
        :param order: where ``version`` stands among the data's versions: higher is newer
        :ptype order: int
        :param scope: the line of versions ``version`` belongs to: each scope's orders are compared and
            retired on their own, so one scope's move never retires another's answers
        :ptype scope: str
        :return: the answer, gzip bytes, and its :class:`AnswerOutcome`
        :rtype: GivenAnswer
        :raises ValueError: when ``version`` cannot be a key segment
        """
        key = self.key_of(version, request)
        # no await between this read and the answer: a broadcast evicting the key cannot land between
        held = self._select_from_l1(key)
        if held is not None:
            record_tier_read(self.table_name, ReadTier.L1)
            return GivenAnswer(base64.b64decode(held["body"]), AnswerOutcome.READ)
        refused = await self._recorded(version, key[1], order, scope)
        if refused is not None:
            # retired, older than what was retired, or the index unreachable: answered, not stored
            return GivenAnswer(await asyncio.to_thread(_gzipped, await compute()), refused)
        pending = _Pending(compute)
        token = _PENDING.set(pending)
        try:
            row = await self.ensure(key)
        finally:
            _PENDING.reset(token)
        if row is None:
            raise RuntimeError(f"{self.table_name}: no answer for {key}; compute returned nothing")
        outcome = AnswerOutcome.READ
        if pending.ran:
            if await self._confirm(version, key[1], scope):
                outcome = AnswerOutcome.RETIRED
            else:
                outcome = AnswerOutcome.PEER_FALLBACK if pending.after_peer else AnswerOutcome.COMPUTED
        return GivenAnswer(base64.b64decode(row["body"]), outcome)

    def _in_background(self, work: Coroutine[Any, Any, None]) -> None:
        """run ``work`` off the read path, keeping a reference until it ends.

        :param work: the coroutine
        :ptype work: Coroutine[Any, Any, None]
        :return: nothing
        :rtype: None
        """
        task = asyncio.get_running_loop().create_task(self._awaited_in_background(work))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _awaited_in_background(self, work: Coroutine[Any, Any, None]) -> None:
        """await ``work``, recording what it raises: nothing else awaits a background task.

        :param work: the coroutine
        :ptype work: Coroutine[Any, Any, None]
        :return: nothing
        :rtype: None
        """
        try:
            await work
        except Exception as exc:  # noqa: BLE001 -- no caller awaits background work, so what it raises is recorded here or nowhere
            contain(exc, log=log, what=f"background work failed: table={self.table_name}", level="error")

    def key_of(self, version: str, request: str) -> tuple[str, str]:
        """the key of ``request``'s answer at ``version``.

        :param version: the data's version
        :ptype version: str
        :param request: the request, whole
        :ptype request: str
        :return: ``(version, sha256 hex of request)``
        :rtype: tuple[str, str]
        :raises ValueError: when ``version`` is not KV-grammar characters without ``_`` or ``.``
        """
        if not _VERSION.match(version):
            raise ValueError(f"{version!r} cannot be a version key segment: KV-grammar characters, no '_' or '.'")
        return version, hashlib.sha256(request.encode("utf-8")).hexdigest()

    def current_version(self, version: str, order: int, *, scope: str = "") -> None:
        """note the data's current version; when it is newer than any seen, retire the scope's older ones.

        Cheap to call on every read that learns the version: it does nothing for an order this
        replica has already retired below or asked to in the scope, and otherwise starts (or extends)
        the scope's one background retirement. A retirement that fails is tried again on the next call.

        :param version: the version the caller's read is at
        :ptype version: str
        :param order: its order: higher is newer
        :ptype order: int
        :param scope: the line of versions ``version`` belongs to: each scope's orders are compared and
            retired on their own, so one scope's move never retires another's answers
        :ptype scope: str
        :return: nothing
        :rtype: None
        """
        del version
        retirement = self._retirements.setdefault(scope, _Retirement())
        if retirement.wanted is not None and order <= retirement.wanted:
            return
        retirement.wanted = order
        if retirement.task is None or retirement.task.done():
            retirement.task = asyncio.get_running_loop().create_task(self._retire_until_current(retirement, scope))

    async def announce(
        self, version: str, order: int, *, scope: str = "", detail: Mapping[str, Any] | None = None
    ) -> tuple[AnnouncedVersion, Announced | None]:
        """the version an index names for ``scope``: the newer of this replica's and the recorded one.

        The record holds the newest version any replica announced for the scope, and the one it
        replaced. This replica's version is recorded when it is newer than the record's, by
        compare-and-set, so the record only rises; otherwise the recorded one is followed. Retirement
        then goes below the version the record replaced (:meth:`current_version`), so a page still
        reading at it is answered; a host that announces leaves retirement to this call. Each scope is
        its own line of versions, recorded and retired on its own (pass the same ``scope`` to
        :meth:`answer`); anything that makes two versions incomparable -- another conversion of the
        same data -- belongs in the scope.

        A record that cannot be read or written is logged, and this replica's own version is announced.

        :param version: the version this replica would announce on its own
        :ptype version: str
        :param order: its order: higher is newer
        :ptype order: int
        :param scope: which index the version is announced in
        :ptype scope: str
        :param detail: the host's own JSON kept beside the version, for a replica that follows it
        :ptype detail: Mapping[str, Any] | None
        :return: the version to announce, and the record as it stands after (``None`` when it could not
            be read or written)
        :rtype: tuple[AnnouncedVersion, Announced | None]
        """
        own = AnnouncedVersion(version, order, dict(detail or {}))
        name = f"{_ANNOUNCED}{scope}"
        found: Announced | None = None
        proposed: Announced | None = None

        def rise(row: dict[str, Any] | None) -> _Action:
            nonlocal found, proposed
            found = _announced_from(row, self._index.table_name)
            if found is not None and found.latest.order >= order:
                proposed = None
                return ("noop", None)
            proposed = Announced(own, None if found is None else found.latest)
            return ("upsert", {"name": name, "members": proposed.as_json()})

        try:
            await self._index.l2_cas_mutate(name, rise)
        except _INDEX_ERRORS as exc:
            # NOSILENT: unrecorded, this replica announces its own version, as with no record at all
            log.warning(
                "the announced version could not be shared; this replica's own is announced: "
                "table=%s scope=%s error=%s",
                self.table_name,
                scope,
                exc,
            )
            return own, None
        record = proposed if proposed is not None else found
        assert record is not None  # narrow: a noop needs a record found
        if record.previous is not None:
            self.current_version(record.previous.version, record.previous.order, scope=scope)
        announced = record.latest if record.latest.order > order else own
        return announced, record

    async def announced(self, scope: str = "") -> Announced | None:
        """the announced-version record for ``scope``, as :meth:`announce` keeps it.

        :param scope: which index
        :ptype scope: str
        :return: the record; ``None`` when there is none, or it cannot be read
        :rtype: Announced | None
        """
        return _announced_from(await self._index.ensure(f"{_ANNOUNCED}{scope}"), self._index.table_name)

    async def retire_older_than(self, order: int, *, scope: str = "") -> int:
        """delete every answer this owner holds at a version of the scope of a lower order.

        :param order: the order of the version now current
        :ptype order: int
        :param scope: the line of versions ``version`` belongs to: each scope's orders are compared and
            retired on their own, so one scope's move never retires another's answers
        :ptype scope: str
        :return: how many answers were deleted
        :rtype: int
        :raises KvError: when the index or an entry could not be read or changed
        :raises ConcurrentModificationError: when the index kept changing past the retry budget
        :raises Exception: whatever the purger raises outside its contract (:data:`KeyPurger`), once
            every retired version is cleared
        """
        taken: dict[str, int] = {}
        row_name = _versions_row(scope)

        def raise_floor(row: dict[str, Any] | None) -> _Action:
            nonlocal taken
            members: dict[str, int] = dict(row["members"]) if row else {}
            floor = row.get("floor") if row else None
            taken = {version: older for version, older in members.items() if older < order}
            new_floor = order - 1 if floor is None else max(floor, order - 1)
            if row is not None and new_floor == floor:
                return ("noop", None)
            return ("upsert", {"name": row_name, "members": members, "floor": new_floor})

        # first the floor: from here no replica records a version below it, so what is listed is final
        await self._index.l2_cas_mutate(row_name, raise_floor)
        # the purge runs once every version is cleared, so a purger that fails outside its contract
        # cannot leave a version's answers behind; keys already deleted are purged even when a later
        # version fails, since a forgotten version is never visited again
        touched: list[str] = []
        try:
            deleted = await self._clear_versions(taken, touched, row_name)
        except Exception:
            await self._purge(touched)
            raise
        await self._purge(touched)
        if deleted:
            log.info("retired old answers: table=%s below=%d deleted=%d", self.table_name, order, deleted)
        return deleted

    async def _clear_versions(self, taken: dict[str, int], touched: list[str], row_name: str) -> int:
        """delete every answer of the versions taken below the floor, forgetting each one emptied.

        :param taken: each version below the floor, with its order
        :ptype taken: dict[str, int]
        :param touched: the keys deleted, relative to the owner's scope, to purge
        :ptype touched: list[str]
        :param row_name: the index row holding the versions and the floor
        :ptype row_name: str
        :return: how many answers were deleted
        :rtype: int
        """
        deleted = 0
        for version, older in sorted(taken.items()):
            emptied = True
            for shard in _SHARDS:
                answers, shard_emptied = await self._clear_shard(f"{version}.{shard}", touched)
                deleted += answers
                emptied = emptied and shard_emptied
            if emptied:
                # every digest is accounted for: the version is forgotten; otherwise the next retirement
                # finishes it, once the writers still at work have written or gone
                await self._index.l2_cas_mutate(row_name, _forget(row_name, version, older))
        return deleted

    # ------------------------------------------------------------------
    # the index
    # ------------------------------------------------------------------

    async def _recorded(self, version: str, digest: str, order: int, scope: str) -> AnswerOutcome | None:
        """record ``version`` and the digest in the index before computing, unless it is retired.

        :param version: the version
        :ptype version: str
        :param digest: the request's digest
        :ptype digest: str
        :param order: the version's order
        :ptype order: int
        :param scope: the version's scope
        :ptype scope: str
        :return: ``None`` when the answer may be stored; why not, when its version is at or below the
            floor (:attr:`AnswerOutcome.RETIRED`) or the index could not be written (logged,
            :attr:`AnswerOutcome.INDEX_UNWRITABLE`)
        :rtype: AnswerOutcome | None
        """
        refused = False
        row_name = _versions_row(scope)

        def add_version(row: dict[str, Any] | None) -> _Action:
            nonlocal refused
            members: dict[str, int] = dict(row["members"]) if row else {}
            floor = row.get("floor") if row else None
            refused = floor is not None and order <= floor
            if refused or members.get(version) == order:
                return ("noop", None)
            members[version] = order
            return ("upsert", {"name": row_name, "members": members, "floor": floor})

        try:
            await self._index.l2_cas_mutate(row_name, add_version)
            if not refused:
                shard = f"{version}.{digest[0]}"
                await self._index.l2_cas_mutate(shard, _add_member(shard, digest))
        except _INDEX_ERRORS as exc:
            # NOSILENT: the read is answered uncached; an answer the index does not name is never stored
            log.warning("answer index unwritable; answering uncached: table=%s error=%s", self.table_name, exc)
            return AnswerOutcome.INDEX_UNWRITABLE
        return AnswerOutcome.RETIRED if refused else None

    async def _confirm(self, version: str, digest: str, scope: str) -> bool:
        """after this replica's answer landed: if its version was retired meanwhile, take the answer back.

        The floor is read from L2, past this replica's L1, as retirement reads it: a copy a broadcast
        has not yet evicted could hide the retirement. A floor that cannot be read counts as retired:
        the answer was already returned to its caller, and taking back an answer still wanted costs
        one recompute, while keeping one retired would outlive its version. The purge of what was
        taken back runs off the read path.

        :param version: the version
        :ptype version: str
        :param digest: the request's digest
        :ptype digest: str
        :param scope: the version's scope
        :ptype scope: str
        :return: whether the answer was taken back
        :rtype: bool
        """
        retired = False
        try:
            versions = await self._index.read_l2(_versions_row(scope))
            floor = versions.get("floor") if versions else None
            retired = versions is None or floor is not None and versions["members"].get(version, floor) <= floor
            if retired:
                await self.l2_cas_mutate((version, digest), _delete_present)
                shard = f"{version}.{digest[0]}"
                removed = await self._index.l2_cas_mutate(shard, _remove_member(shard, digest))
                taken_back = [self._relative(self.l2_key((version, digest)))]
                if removed.action == "deleted":
                    taken_back.append(self._relative(self._index.l2_key(shard)))
                self._in_background(self._purge(taken_back))
        except _INDEX_ERRORS as exc:
            retired = False
            # NOSILENT: the digest stays in its shard, so the next retirement deletes this answer
            log.warning(
                "answer landed on a retired version and could not be taken back now: table=%s version=%s error=%s",
                self.table_name,
                version,
                exc,
            )
        return retired

    async def _clear_shard(self, name: str, touched: list[str]) -> tuple[int, bool]:
        """delete every answer a shard names and drop the digests accounted for; keep those still being written.

        :param name: the shard's name, ``{version}.{digit}``
        :ptype name: str
        :param touched: the keys deleted, relative to the owner's scope, to purge
        :ptype touched: list[str]
        :return: how many answers were deleted, and whether the shard is gone
        :rtype: tuple[int, bool]
        """
        version = name.split(".", 1)[0]
        row = await self._index.read_l2(name)
        deleted = 0
        handled: set[str] = set()
        for digest in sorted(row["members"]) if row else ():
            settled, answered = await self._clear_answer(version, digest, touched)
            deleted += answered
            if settled:
                handled.add(digest)
        emptied = False

        def drop_handled(current: dict[str, Any] | None) -> _Action:
            nonlocal emptied
            members = [member for member in current["members"] if member not in handled] if current else []
            emptied = not members
            if current is None or len(members) == len(current["members"]) and members:
                return ("noop", None)
            return ("delete", None) if emptied else ("upsert", {"name": name, "members": members})

        outcome = await self._index.l2_cas_mutate(name, drop_handled)
        if outcome.action == "deleted":
            touched.append(self._relative(self._index.l2_key(name)))
        return deleted, emptied

    async def _clear_answer(self, version: str, digest: str, touched: list[str]) -> tuple[bool, bool]:
        """delete one digest's answer, or learn that its writer is still at work.

        :param version: the version
        :ptype version: str
        :param digest: the digest
        :ptype digest: str
        :param touched: the keys deleted, relative to the owner's scope
        :ptype touched: list[str]
        :return: whether the digest is accounted for (its answer deleted, or none and no writer at
            work), and whether an answer was deleted
        :rtype: tuple[bool, bool]
        """
        key = (version, digest)
        outcome = await self.l2_cas_mutate(key, _delete_present)
        answered = outcome.action == "deleted"
        settled = answered
        if not answered:
            try:
                # no answer yet: only a writer holding the key's build lock can still write one
                async with self.derivation_paused(key):
                    answered = (await self.l2_cas_mutate(key, _delete_present)).action == "deleted"
                    settled = True
            except BuildLockHeld:
                settled = False
        if answered:
            touched.append(self._relative(self.l2_key(key)))
        return settled, answered

    def _relative(self, key: str) -> str:
        """a key of this owner's, without the scope that leads it: what a purge names.

        :param key: the full key, ``{scope}.{rest}``
        :ptype key: str
        :return: ``{rest}``
        :rtype: str
        """
        return key.removeprefix(f"{self._registry.kv_key_scope}.")

    async def _purge(self, keys: list[str]) -> None:
        """purge deleted keys so they leave no marker; keep the markers (logged once) when it cannot.

        :param keys: the keys, relative to the owner's scope
        :ptype keys: list[str]
        :return: nothing
        :rtype: None
        :raises Exception: whatever the purger raises outside its contract (:data:`KeyPurger`)
        """
        if not keys:
            return
        try:
            if self._purger is None:
                raise CollectionKeysRequestError("no purger: this owner keeps the delete markers")
            for start in range(0, len(keys), MAX_PURGED_KEYS):
                await self._purger(keys[start : start + MAX_PURGED_KEYS])
        except _PURGE_ERRORS as exc:  # a marker left is space, never a wrong answer
            # NOSILENT: said once, not per retirement
            if not self._markers_kept_logged:
                self._markers_kept_logged = True
                contain(
                    exc,
                    log=log,
                    what=f"retired answers' keys could not be purged; their delete markers stay: table={self.table_name} error={exc}",
                )

    async def _retire_until_current(self, retirement: _Retirement, scope: str) -> None:
        """retire ``scope`` below the highest order asked for, until it is reached or a retirement fails.

        :param retirement: the scope's retirement state
        :ptype retirement: _Retirement
        :param scope: the scope
        :ptype scope: str
        :return: nothing
        :rtype: None
        """
        while (target := retirement.wanted) is not None and (
            retirement.retired_below is None or target > retirement.retired_below
        ):
            try:
                await self.retire_older_than(target, scope=scope)
            except Exception as exc:  # noqa: BLE001 -- a retirement that fails for any reason must be tried again, not wedge the next ones
                # NOSILENT: the old answers wait for the next call, which tries again
                contain(
                    exc,
                    log=log,
                    what=f"retiring old answers failed: table={self.table_name} scope={scope} below={target} error={exc}",
                )
                retirement.wanted = retirement.retired_below
                return
            retirement.retired_below = target

    # ------------------------------------------------------------------
    # DerivedCollection's contract
    # ------------------------------------------------------------------

    def derive_key(self, request: Any) -> Any:
        """a ``(version, request)`` pair onto its key.

        :param request: ``(version, request text)``
        :ptype request: Any
        :return: the key
        :rtype: Any
        """
        version, text = request
        return self.key_of(version, text)

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        """the answer a peer has put in L2, or ``None``.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row, or ``None``
        :rtype: dict[str, Any] | None
        """
        return await self._get_from_l2(entity_id)

    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        """compute the answer for ``entity_id`` with the calling :meth:`answer`'s ``compute``, and compress it.

        :param entity_id: the key
        :ptype entity_id: Any
        :return: the row
        :rtype: dict[str, Any] | None
        :raises AnswerNotComputable: when reached by a read other than :meth:`answer` (``get``,
            ``get_for``), which brings no way to compute
        """
        pending = _PENDING.get()
        if pending is None:
            raise AnswerNotComputable(
                f"{self.table_name}: an answer is computed only through answer(), which brings its compute"
            )
        pending.ran = True
        pending.after_peer = self.deriving_after_peer()
        version, digest = self.normalize_pk(entity_id)
        text = await pending.compute()
        return {"version": version, "request": digest, "body": await asyncio.to_thread(_row_body, text)}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        """write the answer to L2 while the computing replica still holds the key's build lock.

        Written here, under the lock, rather than by the read's seed after it, so
        a retirement that finds no answer and can take the lock knows none is still coming.

        :param data: the row
        :ptype data: dict[str, Any]
        :param original_timestamp: unused
        :ptype original_timestamp: Any
        :param conn: unused
        :ptype conn: Any
        :return: ``1``
        :rtype: int
        """
        del original_timestamp, conn
        try:
            await self.l2_cas_mutate((data["version"], data["request"]), lambda _row: ("upsert", data))
        except GenerationUnavailableError as exc:
            # NOSILENT: the answer landed; only a replica that misses its broadcast cannot tell (logged
            # where the advance failed), and what it keeps is an answer that never changes
            log.warning("answer stored without advancing its write generation: table=%s error=%s", self.table_name, exc)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        """nothing: there is no durable tier; retirement deletes from L2.

        :param entity_id: unused
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        del entity_id

    def serialize(self, data: dict[str, Any]) -> bytes:
        """a row as L2 holds it: JSON.

        :param data: the row
        :ptype data: dict[str, Any]
        :return: JSON bytes
        :rtype: bytes
        """
        return json.dumps({key: data[key] for key in ("version", "request", "body")}).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """a row back from L2.

        :param data: JSON bytes
        :ptype data: bytes
        :return: the row
        :rtype: dict[str, Any]
        """
        row: dict[str, Any] = json.loads(data)
        return row


def _delete_present(row: dict[str, Any] | None) -> _Action:
    """delete the row when there is one.

    :param row: the row, or ``None``
    :ptype row: dict[str, Any] | None
    :return: the action
    :rtype: _Action
    """
    return ("delete", None) if row is not None else ("noop", None)


def _add_member(name: str, member: str) -> Callable[[dict[str, Any] | None], _Action]:
    """a shard change adding ``member``.

    :param name: the shard's name
    :ptype name: str
    :param member: the digest
    :ptype member: str
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: list[str] = list(row["members"]) if row else []
        if member in members:
            return ("noop", None)
        return ("upsert", {"name": name, "members": [*members, member]})

    return change


def _remove_member(name: str, member: str) -> Callable[[dict[str, Any] | None], _Action]:
    """a shard change removing ``member``, deleting the shard when it empties.

    :param name: the shard's name
    :ptype name: str
    :param member: the digest
    :ptype member: str
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: list[str] = list(row["members"]) if row else []
        if member not in members:
            return ("noop", None)
        rest = [kept for kept in members if kept != member]
        return ("delete", None) if not rest else ("upsert", {"name": name, "members": rest})

    return change


def _forget(row_name: str, version: str, order: int) -> Callable[[dict[str, Any] | None], _Action]:
    """a versions-row change taking ``version`` off, once its shards are empty.

    :param row_name: the versions row
    :ptype row_name: str
    :param version: the version retired
    :ptype version: str
    :param order: its order, as retirement read it
    :ptype order: int
    :return: the change
    :rtype: Callable[[dict[str, Any] | None], _Action]
    """

    def change(row: dict[str, Any] | None) -> _Action:
        members: dict[str, int] = dict(row["members"]) if row else {}
        if members.get(version) != order or row is None:
            return ("noop", None)
        del members[version]
        return ("upsert", {"name": row_name, "members": members, "floor": row.get("floor")})

    return change
