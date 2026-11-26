"""The key pool's shared store: a budget ledger of accounts, and versioned records.

:class:`KeyPoolStore` is written in generic ledger terms, not pool terms, so a store outside this package
can satisfy it structurally, with no import of ``threetears.models``. Its parameters and returns are plain
types only: ``str``, ``int``, ``bool``, ``decimal.Decimal``, ``datetime``, and dicts, lists and tuples of
those. Snapshots are typed as :class:`AccountSnapshot`, a ``TypedDict``: a plain ``dict`` at runtime, and
one a store elsewhere satisfies by declaring a ``TypedDict`` of the same shape, which type checkers compare
structurally.

It has two parts.

**Accounts.** An account is an opaque string the caller builds; the pool builds
``"<provider>/<pool>/<label>/<generation>/<period id>"``, where the generation changes when a label's secret
does. So a key's spend is kept per secret and per billing period, and a new period's account starts at zero
spend the first time it is seen, with no rollover job. An account holds:

- ``spent``, the money settled into it;
- its open reservations, each an id with an amount and an expiry;
- its orphaned reservations, those the expiry sweep settled at their full amount, each id with that amount.

A reservation holds budget for one call in flight: :meth:`KeyPoolStore.reserve` admits it only when
``spent + reserved + amount <= budget``, checked and recorded as one atomic step, so concurrent callers
never together take an account past its budget. :meth:`KeyPoolStore.settle` replaces it with the call's
actual cost, and :meth:`KeyPoolStore.release` drops it with no spend. A reservation whose caller crashed is
settled at its full amount when :meth:`KeyPoolStore.sweep_expired` finds it past its expiry, and is
reported as orphaned: counting it as spend is the conservative choice.

**Records.** A record is a JSON-safe document under a name, with a version that starts at 0 for an absent
record and rises by one on every write. :meth:`KeyPoolStore.compare_and_set_record` writes only over the
version the caller read, so two processes moving a pool's active key at once agree: the loser is handed
the winner's version and document to adopt. The pool keeps its active key, key states, holds,
fingerprints and configured budgets in records.

Money is :class:`decimal.Decimal` throughout: budgets are compared by value at the line, where float
error would admit a reservation the sum says doesn't fit.

:class:`InMemoryKeyPoolStore` is the single-process store, the production choice for a script or CLI, and
the store the pool's unit tests run on.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, TypedDict, runtime_checkable

__all__ = [
    "AccountSnapshot",
    "InMemoryKeyPoolStore",
    "KeyPoolStore",
    "KeyPoolStoreError",
    "ReservationAlreadyOpen",
    "ReservationNotFound",
    "ReservationSnapshot",
]


class KeyPoolStoreError(Exception):
    """The key pool's store failed, or refused an operation its ledger can't honour.

    The pool wraps a store's storage failures in this, with the store's own exception as its cause, and
    fails the call closed rather than running it unbudgeted. It is not a ``ModelProviderError``: the
    provider never saw the call.
    """


class ReservationNotFound(KeyPoolStoreError, LookupError):
    """A settle or release named a reservation the account doesn't hold.

    The id was never reserved on that account, or its reservation was already settled or released. A
    reservation the expiry sweep closed is not this: settling or releasing it late is accepted (see
    :meth:`KeyPoolStore.settle`).

    :param account: the account named
    :ptype account: str
    :param reservation_id: the reservation id named
    :ptype reservation_id: str
    """

    def __init__(self, account: str, reservation_id: str) -> None:
        self.account = account
        self.reservation_id = reservation_id
        super().__init__(f"account {account!r} holds no reservation {reservation_id!r}; it is unknown or closed")


class ReservationAlreadyOpen(KeyPoolStoreError, ValueError):
    """A reserve reused an id the account still holds, open or orphaned and not yet settled late.

    Reservation ids are unique per call, so this is a caller's mistake; admitting it would overwrite a
    hold some call in flight still owns.

    :param account: the account named
    :ptype account: str
    :param reservation_id: the reservation id reused
    :ptype reservation_id: str
    """

    def __init__(self, account: str, reservation_id: str) -> None:
        self.account = account
        self.reservation_id = reservation_id
        super().__init__(f"account {account!r} already holds reservation {reservation_id!r}")


class ReservationSnapshot(TypedDict):
    """One open reservation, as :meth:`KeyPoolStore.snapshot` reports it."""

    #: the money held
    amount: Decimal
    #: when the expiry sweep may settle it at ``amount``; timezone-aware
    expires_at: datetime


class AccountSnapshot(TypedDict):
    """One account's ledger, as :meth:`KeyPoolStore.snapshot` reports it. An unseen account is all zero."""

    #: money settled into the account, orphaned reservations included
    spent: Decimal
    #: the sum of the open reservations' amounts
    reserved: Decimal
    #: the open reservations, by id; their count is the calls in flight on the account
    open_reservations: dict[str, ReservationSnapshot]
    #: reservations the expiry sweep settled at their full amount and no late settle has corrected,
    #: by id, each with the amount the sweep counted as spend
    orphaned: dict[str, Decimal]


@runtime_checkable
class KeyPoolStore(Protocol):
    """The ledger and record store a key pool shares between its processes.

    Every method is a coroutine, and each one is atomic: a concurrent caller sees an operation entirely
    or not at all. Two stores built over the same storage see each other's accounts and records.

    Errors are stated by their builtin base, which a store outside this package raises a subclass of:

    - ``LookupError`` for a settle or release of a reservation the account doesn't hold
      (:class:`ReservationNotFound` here);
    - ``ValueError`` for a reserve reusing an id the account holds (:class:`ReservationAlreadyOpen`
      here), and for invalid arguments: money that is negative, not finite or not a ``Decimal``, a naive
      ``datetime``, a negative version, or a document that isn't JSON-safe.

    Anything else a store raises is a storage failure, which the pool wraps in
    :class:`KeyPoolStoreError`.
    """

    async def reserve(
        self, account: str, reservation_id: str, amount: Decimal, budget: Decimal, expires_at: datetime
    ) -> bool:
        """Hold ``amount`` on ``account`` for one call, if it fits under ``budget``.

        It fits when ``spent + reserved + amount <= budget``. The check and the hold are one atomic step,
        so concurrent reserves on one account never together exceed the budget. A reservation past its
        expiry that no sweep has closed still counts as reserved. An account seen for the first time
        starts with nothing spent.

        :param account: the account to hold the money on
        :ptype account: str
        :param reservation_id: the call's reservation id, unique per call
        :ptype reservation_id: str
        :param amount: the money to hold; ``Decimal``, finite, not negative
        :ptype amount: Decimal
        :param budget: the account's budget for its period; ``Decimal``, finite, not negative. Passed on
            each call, not stored, so a budget the operator changes applies at once
        :ptype budget: Decimal
        :param expires_at: when the expiry sweep may settle this reservation at ``amount``;
            timezone-aware
        :ptype expires_at: datetime
        :return: ``True`` when the reservation is held; ``False``, holding nothing, when it doesn't fit
        :rtype: bool
        :raises ValueError: (:class:`ReservationAlreadyOpen` here) when ``reservation_id`` is open or
            orphaned on ``account``; when ``amount`` or ``budget`` is invalid, or ``expires_at`` naive
        """
        ...

    async def settle(self, account: str, reservation_id: str, amount: Decimal) -> None:
        """Replace a reservation with the call's actual cost.

        ``amount`` may be above or below what was reserved: spend records the truth, and the next
        reserve sees it.

        A reservation the expiry sweep already settled at its full amount is settled late, not refused:
        the swept amount is replaced by ``amount`` and the id leaves the orphaned set. A slow call can
        outlive its expiry, and refusing its settle would leave the conservative estimate standing as
        spend while handing the pool an error for a call that succeeded. A second settle or release of
        the same id is refused.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to settle
        :ptype reservation_id: str
        :param amount: the call's actual cost; ``Decimal``, finite, not negative
        :ptype amount: Decimal
        :raises LookupError: (:class:`ReservationNotFound` here) when the account holds no such
            reservation, open or orphaned: it was never made there, or it is already closed
        :raises ValueError: when ``amount`` is invalid
        """
        ...

    async def release(self, account: str, reservation_id: str) -> None:
        """Drop a reservation with no spend, for a call the provider can't have billed.

        A reservation the expiry sweep already settled is released late: the amount the sweep counted
        as spend is taken back out, as a settle at zero would, and the id leaves the orphaned set.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to drop
        :ptype reservation_id: str
        :raises LookupError: (:class:`ReservationNotFound` here) when the account holds no such
            reservation, open or orphaned
        """
        ...

    async def sweep_expired(self, accounts: Iterable[str], now: datetime) -> dict[str, list[str]]:
        """Settle every open reservation on ``accounts`` whose expiry is at or before ``now``.

        Each is settled at its full amount and moved to the account's orphaned set, where a late settle or
        release can still correct it. The store reads no clock; the caller passes ``now``.

        :param accounts: the accounts to sweep; a bare ``str`` is refused rather than read as characters
        :ptype accounts: Iterable[str]
        :param now: the sweep's instant; timezone-aware
        :ptype now: datetime
        :return: the ids this sweep orphaned, by account, in the order they were reserved; an account
            with none is absent
        :rtype: dict[str, list[str]]
        :raises ValueError: when ``now`` is naive
        :raises TypeError: when ``accounts`` is a ``str``
        """
        ...

    async def snapshot(self, accounts: Iterable[str]) -> dict[str, AccountSnapshot]:
        """Read the ledger of each of ``accounts``.

        :param accounts: the accounts to read; a bare ``str`` is refused rather than read as characters
        :ptype accounts: Iterable[str]
        :return: every requested account's ledger, by account; an account never seen reads as zero
            spent, zero reserved and no reservations
        :rtype: dict[str, AccountSnapshot]
        :raises TypeError: when ``accounts`` is a ``str``
        """
        ...

    async def read_record(self, name: str) -> tuple[int, dict[str, Any] | None]:
        """Read a record and its version.

        :param name: the record's name
        :ptype name: str
        :return: ``(version, document)``; ``(0, None)`` when the record was never written. The document
            is the caller's own copy
        :rtype: tuple[int, dict[str, Any] | None]
        """
        ...

    async def compare_and_set_record(
        self, name: str, expected_version: int, document: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Write ``document`` under ``name`` if the record is still at ``expected_version``.

        A first write passes ``expected_version=0``. Records are never deleted; an operator reset writes
        a new document.

        :param name: the record's name
        :ptype name: str
        :param expected_version: the version the caller read, 0 for an absent record
        :ptype expected_version: int
        :param document: the new document: a dict that survives a JSON round trip unchanged (string keys;
            ``str``, ``int``, finite ``float``, ``bool``, ``None``, lists and dicts; money as a string)
        :ptype document: dict[str, Any]
        :return: ``(True, new_version, document)`` when written. ``(False, version, current)`` when
            another writer got there first: the winner's version and document, for the caller to adopt
            (``current`` is ``None`` only when the record is absent)
        :rtype: tuple[bool, int, dict[str, Any] | None]
        :raises ValueError: when ``expected_version`` is negative or ``document`` isn't JSON-safe
        """
        ...


@dataclass(slots=True)
class _Reservation:
    amount: Decimal
    expires_at: datetime


@dataclass(slots=True)
class _Account:
    spent: Decimal = Decimal(0)
    open: dict[str, _Reservation] = field(default_factory=dict)
    orphaned: dict[str, Decimal] = field(default_factory=dict)

    def reserved(self) -> Decimal:
        return sum((held.amount for held in self.open.values()), Decimal(0))


class InMemoryKeyPoolStore:
    """A :class:`KeyPoolStore` held in this process's memory.

    The production store for a single-process consumer, a script or a CLI, and the store the pool's unit
    tests run on. It holds plain values and nothing survives the process.

    Safe across concurrent asyncio tasks and across threads: every operation runs, with no ``await`` in
    it, under one ``threading.Lock``, so a pool driven from a synchronous call in a worker thread and an
    async call on the loop still share one consistent ledger. The lock is held for microseconds and never
    across an ``await``, so the event loop is never blocked on it.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accounts: dict[str, _Account] = {}
        self._records: dict[str, tuple[int, str]] = {}

    async def reserve(
        self, account: str, reservation_id: str, amount: Decimal, budget: Decimal, expires_at: datetime
    ) -> bool:
        """See :meth:`KeyPoolStore.reserve`.

        :param account: the account to hold the money on
        :ptype account: str
        :param reservation_id: the call's reservation id, unique per call
        :ptype reservation_id: str
        :param amount: the money to hold
        :ptype amount: Decimal
        :param budget: the account's budget for its period
        :ptype budget: Decimal
        :param expires_at: when the expiry sweep may settle this reservation; timezone-aware
        :ptype expires_at: datetime
        :return: whether the reservation is held
        :rtype: bool
        :raises ReservationAlreadyOpen: when ``reservation_id`` is open or orphaned on ``account``
        :raises ValueError: when ``amount`` or ``budget`` is invalid, or ``expires_at`` naive
        """
        _check_money("amount", amount)
        _check_money("budget", budget)
        _check_aware("expires_at", expires_at)
        with self._lock:
            ledger = self._accounts.setdefault(account, _Account())
            if reservation_id in ledger.open or reservation_id in ledger.orphaned:
                raise ReservationAlreadyOpen(account, reservation_id)
            if ledger.spent + ledger.reserved() + amount > budget:
                return False
            ledger.open[reservation_id] = _Reservation(amount, expires_at)
            return True

    async def settle(self, account: str, reservation_id: str, amount: Decimal) -> None:
        """See :meth:`KeyPoolStore.settle`.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to settle
        :ptype reservation_id: str
        :param amount: the call's actual cost
        :ptype amount: Decimal
        :raises ReservationNotFound: when the account holds no such reservation, open or orphaned
        :raises ValueError: when ``amount`` is invalid
        """
        _check_money("amount", amount)
        with self._lock:
            ledger = self._accounts.get(account)
            if ledger is not None and reservation_id in ledger.open:
                del ledger.open[reservation_id]
                ledger.spent += amount
                return
            if ledger is not None and reservation_id in ledger.orphaned:
                ledger.spent += amount - ledger.orphaned.pop(reservation_id)
                return
            raise ReservationNotFound(account, reservation_id)

    async def release(self, account: str, reservation_id: str) -> None:
        """See :meth:`KeyPoolStore.release`.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to drop
        :ptype reservation_id: str
        :raises ReservationNotFound: when the account holds no such reservation, open or orphaned
        """
        await self.settle(account, reservation_id, Decimal(0))

    async def sweep_expired(self, accounts: Iterable[str], now: datetime) -> dict[str, list[str]]:
        """See :meth:`KeyPoolStore.sweep_expired`.

        :param accounts: the accounts to sweep
        :ptype accounts: Iterable[str]
        :param now: the sweep's instant; timezone-aware
        :ptype now: datetime
        :return: the ids this sweep orphaned, by account; an account with none is absent
        :rtype: dict[str, list[str]]
        :raises ValueError: when ``now`` is naive
        :raises TypeError: when ``accounts`` is a ``str``
        """
        names = _account_names(accounts)
        _check_aware("now", now)
        swept: dict[str, list[str]] = {}
        with self._lock:
            for account in names:
                ledger = self._accounts.get(account)
                if ledger is None:
                    continue
                expired = [rid for rid, held in ledger.open.items() if held.expires_at <= now]
                for rid in expired:
                    held = ledger.open.pop(rid)
                    ledger.spent += held.amount
                    ledger.orphaned[rid] = held.amount
                if expired:
                    swept[account] = expired
        return swept

    async def snapshot(self, accounts: Iterable[str]) -> dict[str, AccountSnapshot]:
        """See :meth:`KeyPoolStore.snapshot`.

        :param accounts: the accounts to read
        :ptype accounts: Iterable[str]
        :return: every requested account's ledger, by account
        :rtype: dict[str, AccountSnapshot]
        :raises TypeError: when ``accounts`` is a ``str``
        """
        names = _account_names(accounts)
        with self._lock:
            return {account: _snapshot_of(self._accounts.get(account) or _Account()) for account in names}

    async def read_record(self, name: str) -> tuple[int, dict[str, Any] | None]:
        """See :meth:`KeyPoolStore.read_record`.

        :param name: the record's name
        :ptype name: str
        :return: ``(version, document)``; ``(0, None)`` when absent
        :rtype: tuple[int, dict[str, Any] | None]
        """
        with self._lock:
            return _decoded(self._records.get(name))

    async def compare_and_set_record(
        self, name: str, expected_version: int, document: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """See :meth:`KeyPoolStore.compare_and_set_record`.

        :param name: the record's name
        :ptype name: str
        :param expected_version: the version the caller read, 0 for an absent record
        :ptype expected_version: int
        :param document: the new document, JSON-safe
        :ptype document: dict[str, Any]
        :return: ``(True, new_version, document)`` when written, else ``(False, version, current)``
        :rtype: tuple[bool, int, dict[str, Any] | None]
        :raises ValueError: when ``expected_version`` is negative or ``document`` isn't JSON-safe
        """
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
            raise ValueError(f"expected_version must be an int of 0 or more, got {expected_version!r}")
        encoded = _encoded(document)
        with self._lock:
            current = self._records.get(name)
            version = current[0] if current is not None else 0
            if version != expected_version:
                return (False, *_decoded(current))
            self._records[name] = (version + 1, encoded)
            return True, version + 1, json.loads(encoded)


def _check_money(what: str, value: Decimal) -> None:
    if not isinstance(value, Decimal):
        raise ValueError(f"{what} must be a decimal.Decimal, got {type(value).__name__}")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{what} must be finite and not negative, got {value}")


def _check_aware(what: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware, got the naive {value.isoformat()}")


def _account_names(accounts: Iterable[str]) -> list[str]:
    # a lone account string is iterable too, and would be swept or read as one-character accounts
    if isinstance(accounts, str):
        raise TypeError(f"accounts must be a collection of account strings, not the one string {accounts!r}")
    return list(accounts)


def _snapshot_of(ledger: _Account) -> AccountSnapshot:
    return {
        "spent": ledger.spent,
        "reserved": ledger.reserved(),
        "open_reservations": {
            rid: {"amount": held.amount, "expires_at": held.expires_at} for rid, held in ledger.open.items()
        },
        "orphaned": dict(ledger.orphaned),
    }


def _encoded(document: dict[str, Any]) -> str:
    if not isinstance(document, dict):
        raise ValueError(f"a record's document must be a dict, got {type(document).__name__}")
    try:
        encoded = json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"a record's document must be JSON-safe: {exc}") from exc
    # json.dumps quietly turns int keys into strings and tuples into lists; a document that reads back
    # different from what was written is refused rather than stored changed
    if json.loads(encoded) != document:
        raise ValueError("a record's document must survive a JSON round trip unchanged: string keys, lists not tuples")
    return encoded


def _decoded(stored: tuple[int, str] | None) -> tuple[int, dict[str, Any] | None]:
    if stored is None:
        return 0, None
    version, encoded = stored
    document: dict[str, Any] = json.loads(encoded)
    return version, document
