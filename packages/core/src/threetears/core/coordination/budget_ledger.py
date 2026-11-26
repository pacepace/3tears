"""BudgetLedger -- a durable budget ledger of accounts, and versioned records, over the coordination tables.

Sibling to :class:`~threetears.core.coordination.windowed_counter.WindowedCounter`: state kept in L3,
with L2's compare-and-swap as the fence every replica writes through. It answers "may this call hold
this much money against this account's budget" for every process sharing one store::

    ledger = BudgetLedger(registry, purpose="model-key-pool")
    if await ledger.reserve(account, reservation_id, estimate, budget, expires_at):
        try:
            cost = await make_the_call()
        except NothingBilled:
            await ledger.release(account, reservation_id)
            raise
        await ledger.settle(account, reservation_id, cost)

It satisfies ``threetears.models.key_pool.store.KeyPoolStore`` structurally, with no import of
``threetears.models``: that Protocol's docstrings are the contract, and this class keeps it exactly.

**Accounts.** One row per ``(purpose, account)`` in ``coordination_ledger_accounts`` holds the money
spent, the open reservations and the orphaned ones. A reservation is admitted only when ``spent +
reserved + amount <= budget``, decided and recorded in one compare-and-swap of that row, so concurrent
reserves -- in one process or in many -- never together pass the budget. The budget is passed on every
call and never stored, so a budget an operator changes applies at once.

**Records.** One row per ``(purpose, name)`` in ``coordination_documents`` holds a JSON document and its
version. A write names the version it read, and loses to any writer that got there first, which hands it
the winner's version and document.

**Durable before it returns.** Both tables persist each won swap to L3 synchronously: when a call
returns, its change is in L3, or a later swap that built on it is. A NATS wipe therefore loses no spend:
the next swap on the key finds L2 empty and continues from L3's row. A persist that fails withdraws the
won L2 value and raises, so a failed call is never left answering from L2. That is the trade the key
pool's plan accepted: a database write per reserve and per settle.

**Why it refuses a registry with no L2.** The compare-and-swap is the guarantee. Without L2 the
collection falls back to a read-modify-write through L3, and two processes could both admit a
reservation the budget held room for once. A registry with no L3 is not refused, as no coordination
primitive refuses it, and is logged once per table: the ledger then lives in L2 alone and a broker
restart loses it, which is a wiring gap for anything holding money.

**Errors.** As the Protocol states them by builtin base: a settle or release of a reservation the
account doesn't hold raises :class:`LedgerReservationNotFound` (a ``LookupError``); a reserve reusing an
id the account holds raises :class:`LedgerReservationAlreadyOpen`, and an invalid argument
:class:`LedgerArgumentError` (both ``ValueError``); a bare ``str`` given as ``accounts`` raises
``TypeError``. A storage failure -- L2 (``KvError``), L3, or every compare-and-swap round lost
(``ConcurrentModificationError``) -- propagates as it is, for the caller to fail closed on.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal
from typing import Any, Final, Literal, TypedDict

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import CoreConfig
from threetears.core.coordination.tables import (
    CoordinationDocumentsCollection,
    CoordinationLedgerAccountsCollection,
    coordination_collection,
)

__all__ = ["BudgetLedger"]

#: compare-and-swap rounds per operation. An account's contention is a burst of calls in flight on one
#: key, which is exactly when a budget matters; ``WindowedCounter`` measured 8 rounds raising a conflict
#: on most of a 25-way burst against one key, and 30 raising none.
_MAX_CAS_ATTEMPTS: Final = 30

_Action = Literal["upsert", "delete", "noop"]


class LedgerReservationNotFound(LookupError):
    """A settle or release named a reservation the account doesn't hold, open or orphaned.

    :param account: the account named
    :ptype account: str
    :param reservation_id: the reservation id named
    :ptype reservation_id: str
    """

    def __init__(self, account: str, reservation_id: str) -> None:
        self.account = account
        self.reservation_id = reservation_id
        super().__init__(f"account {account!r} holds no reservation {reservation_id!r}; it is unknown or closed")


class LedgerReservationAlreadyOpen(ValueError):
    """A reserve reused an id the account still holds, open or orphaned and not yet settled late.

    :param account: the account named
    :ptype account: str
    :param reservation_id: the reservation id reused
    :ptype reservation_id: str
    """

    def __init__(self, account: str, reservation_id: str) -> None:
        self.account = account
        self.reservation_id = reservation_id
        super().__init__(f"account {account!r} already holds reservation {reservation_id!r}")


class LedgerArgumentError(ValueError):
    """An argument the ledger can't record: bad money, a naive datetime, a bad version, a non-JSON document."""


class ReservationSnapshot(TypedDict):
    """One open reservation, as :meth:`BudgetLedger.snapshot` reports it."""

    #: the money held
    amount: Decimal
    #: when the expiry sweep may settle it at ``amount``; timezone-aware
    expires_at: datetime


class AccountSnapshot(TypedDict):
    """One account's ledger, as :meth:`BudgetLedger.snapshot` reports it. An unseen account is all zero."""

    #: money settled into the account, orphaned reservations included
    spent: Decimal
    #: the sum of the open reservations' amounts
    reserved: Decimal
    #: the open reservations, by id, in the order they were made
    open_reservations: dict[str, ReservationSnapshot]
    #: reservations the expiry sweep settled at their full amount and no late settle has corrected,
    #: by id, each with the amount the sweep counted as spend
    orphaned: dict[str, Decimal]


class BudgetLedger:
    """A durable budget ledger and record store, shared by every process on one registry's L2 and L3.

    Every method is one compare-and-swap of one row per account or record, so each is atomic: a
    concurrent caller sees an operation entirely or not at all.
    """

    def __init__(self, registry: CollectionRegistry, *, purpose: str, config: CoreConfig | None = None) -> None:
        """configure the ledger over its registry's coordination tables.

        :param registry: the collection registry whose tiers the ledger reads and writes. It needs L2,
            the fence, and should have L3, the record (see the module docstring)
        :ptype registry: CollectionRegistry
        :param purpose: what this ledger budgets (``"model-key-pool"``), carried in every row's key so
            ledgers over one table never share an account or a record
        :ptype purpose: str
        :param config: core config forwarded when this process first builds the ledger's collections;
            defaults to the framework defaults, which the tables' declared write policy overrides
        :ptype config: CoreConfig | None
        :raises ValueError: when ``purpose`` is empty, or the registry has no L2 client
        """
        if not purpose.strip():
            raise ValueError("BudgetLedger purpose must be a non-empty name, e.g. 'model-key-pool'")
        self._purpose = purpose
        self._accounts = coordination_collection(registry, CoordinationLedgerAccountsCollection, config)
        self._documents = coordination_collection(registry, CoordinationDocumentsCollection, config)
        self._accounts.require_l2_fence("BudgetLedger")
        self._documents.require_l2_fence("BudgetLedger")

    @property
    def purpose(self) -> str:
        """what this ledger budgets, carried in every row it writes.

        :return: the purpose
        :rtype: str
        """
        return self._purpose

    async def reserve(
        self, account: str, reservation_id: str, amount: Decimal, budget: Decimal, expires_at: datetime
    ) -> bool:
        """Hold ``amount`` on ``account`` for one call, if ``spent + reserved + amount <= budget``.

        The check and the hold are one compare-and-swap of the account's row. A reservation past its
        expiry that no sweep has closed still counts as reserved. An account seen for the first time
        starts with nothing spent.

        :param account: the account to hold the money on
        :ptype account: str
        :param reservation_id: the call's reservation id, unique per call
        :ptype reservation_id: str
        :param amount: the money to hold; ``Decimal``, finite, not negative
        :ptype amount: Decimal
        :param budget: the account's budget; ``Decimal``, finite, not negative. Never stored
        :ptype budget: Decimal
        :param expires_at: when the expiry sweep may settle this reservation at ``amount``; timezone-aware
        :ptype expires_at: datetime
        :return: ``True`` when the reservation is held; ``False``, holding nothing, when it doesn't fit
        :rtype: bool
        :raises LedgerReservationAlreadyOpen: when ``reservation_id`` is open or orphaned on ``account``
        :raises LedgerArgumentError: when ``amount`` or ``budget`` is invalid, or ``expires_at`` naive
        """
        _check_money("amount", amount)
        _check_money("budget", budget)
        _check_aware("expires_at", expires_at)
        admitted = [False]

        def _reserve(current: dict[str, Any] | None) -> tuple[_Action, dict[str, Any] | None]:
            # re-run from a fresh read on every round, so the verdict is the last round's
            admitted[0] = False
            state = _AccountState.of(current)
            if reservation_id in state.open or reservation_id in state.orphaned:
                raise LedgerReservationAlreadyOpen(account, reservation_id)
            if state.spent + state.reserved() + amount > budget:
                return "noop", None
            state.open[reservation_id] = (amount, expires_at)
            admitted[0] = True
            return "upsert", state.row(self._purpose, account, current)

        await self._accounts.l2_cas_mutate((self._purpose, account), _reserve, max_retries=_MAX_CAS_ATTEMPTS)
        return admitted[0]

    async def settle(self, account: str, reservation_id: str, amount: Decimal) -> None:
        """Replace a reservation with the call's actual cost, above or below what was reserved.

        A reservation the expiry sweep already settled is settled late, once: the swept amount is
        replaced by ``amount`` and the id leaves the orphaned set. A second settle or release is refused.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to settle
        :ptype reservation_id: str
        :param amount: the call's actual cost; ``Decimal``, finite, not negative
        :ptype amount: Decimal
        :return: nothing
        :rtype: None
        :raises LedgerReservationNotFound: when the account holds no such reservation, open or orphaned
        :raises LedgerArgumentError: when ``amount`` is invalid
        """
        _check_money("amount", amount)

        def _settle(current: dict[str, Any] | None) -> tuple[_Action, dict[str, Any] | None]:
            state = _AccountState.of(current)
            if reservation_id in state.open:
                del state.open[reservation_id]
                state.spent += amount
            elif reservation_id in state.orphaned:
                state.spent += amount - state.orphaned.pop(reservation_id)
            else:
                raise LedgerReservationNotFound(account, reservation_id)
            return "upsert", state.row(self._purpose, account, current)

        await self._accounts.l2_cas_mutate((self._purpose, account), _settle, max_retries=_MAX_CAS_ATTEMPTS)

    async def release(self, account: str, reservation_id: str) -> None:
        """Drop a reservation with no spend; one the sweep already settled is released late, as a settle at zero.

        :param account: the account the reservation was made on
        :ptype account: str
        :param reservation_id: the reservation to drop
        :ptype reservation_id: str
        :return: nothing
        :rtype: None
        :raises LedgerReservationNotFound: when the account holds no such reservation, open or orphaned
        """
        await self.settle(account, reservation_id, Decimal(0))

    async def sweep_expired(self, accounts: Iterable[str], now: datetime) -> dict[str, list[str]]:
        """Settle every open reservation on ``accounts`` whose expiry is at or before ``now``, at its full amount.

        Each one swept moves to the account's orphaned set, where a late settle or release can still
        correct it. The ledger reads no clock; the caller passes ``now``.

        :param accounts: the accounts to sweep; a bare ``str`` is refused rather than read as characters
        :ptype accounts: Iterable[str]
        :param now: the sweep's instant; timezone-aware
        :ptype now: datetime
        :return: the ids this sweep orphaned, by account, in the order they were reserved; an account
            with none is absent
        :rtype: dict[str, list[str]]
        :raises LedgerArgumentError: when ``now`` is naive
        :raises TypeError: when ``accounts`` is a ``str``
        """
        names = _account_names(accounts)
        _check_aware("now", now)
        swept: dict[str, list[str]] = {}
        for account in names:
            orphaned: list[str] = []

            def _sweep(
                current: dict[str, Any] | None, account: str = account, orphaned: list[str] = orphaned
            ) -> tuple[_Action, dict[str, Any] | None]:
                orphaned.clear()
                if current is None:
                    return "noop", None
                state = _AccountState.of(current)
                orphaned.extend(rid for rid, (_, expires) in state.open.items() if expires <= now)
                if not orphaned:
                    return "noop", None
                for rid in orphaned:
                    held, _ = state.open.pop(rid)
                    state.spent += held
                    state.orphaned[rid] = held
                return "upsert", state.row(self._purpose, account, current)

            await self._accounts.l2_cas_mutate((self._purpose, account), _sweep, max_retries=_MAX_CAS_ATTEMPTS)
            if orphaned:
                swept[account] = list(orphaned)
        return swept

    async def snapshot(self, accounts: Iterable[str]) -> dict[str, AccountSnapshot]:
        """Read the ledger of each of ``accounts``.

        :param accounts: the accounts to read; a bare ``str`` is refused rather than read as characters
        :ptype accounts: Iterable[str]
        :return: every requested account's ledger, by account; an account never seen reads as zero
            spent, zero reserved and no reservations
        :rtype: dict[str, AccountSnapshot]
        :raises TypeError: when ``accounts`` is a ``str``
        """
        names = _account_names(accounts)
        result: dict[str, AccountSnapshot] = {}
        for account in names:
            entity = await self._accounts.get((self._purpose, account))
            result[account] = _AccountState.of(None if entity is None else entity.to_dict()).snapshot()
        return result

    async def read_record(self, name: str) -> tuple[int, dict[str, Any] | None]:
        """Read a record and its version.

        :param name: the record's name
        :ptype name: str
        :return: ``(version, document)``; ``(0, None)`` when the record was never written. The document
            is the caller's own copy
        :rtype: tuple[int, dict[str, Any] | None]
        """
        entity = await self._documents.get((self._purpose, name))
        return _record_of(None if entity is None else entity.to_dict())

    async def compare_and_set_record(
        self, name: str, expected_version: int, document: dict[str, Any]
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Write ``document`` under ``name`` if the record is still at ``expected_version``.

        A first write passes ``expected_version=0``. Records are never deleted.

        :param name: the record's name
        :ptype name: str
        :param expected_version: the version the caller read, 0 for an absent record
        :ptype expected_version: int
        :param document: the new document: a dict that survives a JSON round trip unchanged
        :ptype document: dict[str, Any]
        :return: ``(True, new_version, document)`` when written. ``(False, version, current)`` when
            another writer got there first (``current`` is ``None`` only when the record is absent)
        :rtype: tuple[bool, int, dict[str, Any] | None]
        :raises LedgerArgumentError: when ``expected_version`` is negative or ``document`` isn't JSON-safe
        """
        if isinstance(expected_version, bool) or not isinstance(expected_version, int) or expected_version < 0:
            raise LedgerArgumentError(f"expected_version must be an int of 0 or more, got {expected_version!r}")
        encoded = _encoded(document)
        outcome: list[tuple[bool, int, dict[str, Any] | None]] = []

        def _swap(current: dict[str, Any] | None) -> tuple[_Action, dict[str, Any] | None]:
            outcome.clear()
            version, held = _record_of(current)
            if version != expected_version:
                outcome.append((False, version, held))
                return "noop", None
            outcome.append((True, version + 1, json.loads(encoded)))
            return "upsert", {
                **(current or {}),
                "purpose": self._purpose,
                "key": name,
                "expires_at": None,
                "version": version + 1,
                "document": encoded,
            }

        await self._documents.l2_cas_mutate((self._purpose, name), _swap, max_retries=_MAX_CAS_ATTEMPTS)
        return outcome[0]


class _AccountState:
    """One account's row, decoded into Decimals and datetimes, and encoded back."""

    def __init__(
        self, spent: Decimal, open_reservations: dict[str, tuple[Decimal, datetime]], orphaned: dict[str, Decimal]
    ) -> None:
        self.spent = spent
        self.open = open_reservations
        self.orphaned = orphaned

    @classmethod
    def of(cls, row: dict[str, Any] | None) -> _AccountState:
        """decode a stored row; an absent one is an account with nothing in it.

        :param row: the stored row, or ``None``
        :ptype row: dict[str, Any] | None
        :return: the decoded account
        :rtype: _AccountState
        """
        if row is None:
            return cls(Decimal(0), {}, {})
        held: dict[str, dict[str, str]] = json.loads(row["open_reservations"])
        swept: dict[str, str] = json.loads(row["orphaned"])
        return cls(
            Decimal(row["spent"]),
            {rid: (Decimal(h["amount"]), datetime.fromisoformat(h["expires_at"])) for rid, h in held.items()},
            {rid: Decimal(amount) for rid, amount in swept.items()},
        )

    def reserved(self) -> Decimal:
        """the sum of the open reservations' amounts.

        :return: the money held
        :rtype: Decimal
        """
        return sum((amount for amount, _ in self.open.values()), Decimal(0))

    def row(self, purpose: str, account: str, current: dict[str, Any] | None) -> dict[str, Any]:
        """the row to write for this account, over the stored one's other columns.

        :param purpose: the ledger's purpose
        :ptype purpose: str
        :param account: the account
        :ptype account: str
        :param current: the stored row, whose creation time and order columns ride along
        :ptype current: dict[str, Any] | None
        :return: the full row
        :rtype: dict[str, Any]
        """
        return {
            **(current or {}),
            "purpose": purpose,
            "key": account,
            "expires_at": None,
            "spent": str(self.spent),
            "reserved": str(self.reserved()),
            "open_reservations": json.dumps(
                {rid: {"amount": str(amount), "expires_at": at.isoformat()} for rid, (amount, at) in self.open.items()}
            ),
            "orphaned": json.dumps({rid: str(amount) for rid, amount in self.orphaned.items()}),
        }

    def snapshot(self) -> AccountSnapshot:
        """this account as :meth:`BudgetLedger.snapshot` reports it.

        :return: the snapshot
        :rtype: AccountSnapshot
        """
        return {
            "spent": self.spent,
            "reserved": self.reserved(),
            "open_reservations": {rid: {"amount": amount, "expires_at": at} for rid, (amount, at) in self.open.items()},
            "orphaned": dict(self.orphaned),
        }


def _record_of(row: dict[str, Any] | None) -> tuple[int, dict[str, Any] | None]:
    """a stored document row as ``(version, document)``.

    :param row: the stored row, or ``None``
    :ptype row: dict[str, Any] | None
    :return: ``(0, None)`` for an absent row
    :rtype: tuple[int, dict[str, Any] | None]
    """
    if row is None:
        return 0, None
    document: dict[str, Any] = json.loads(row["document"])
    return int(row["version"]), document


def _check_money(what: str, value: Decimal) -> None:
    """refuse money that is not a finite, non-negative ``Decimal``.

    :param what: the argument's name, for the message
    :ptype what: str
    :param value: the argument
    :ptype value: Decimal
    :return: nothing
    :rtype: None
    :raises LedgerArgumentError: when it is not
    """
    if not isinstance(value, Decimal):
        raise LedgerArgumentError(f"{what} must be a decimal.Decimal, got {type(value).__name__}")
    if not value.is_finite() or value < 0:
        raise LedgerArgumentError(f"{what} must be finite and not negative, got {value}")


def _check_aware(what: str, value: datetime) -> None:
    """refuse a naive datetime.

    :param what: the argument's name, for the message
    :ptype what: str
    :param value: the argument
    :ptype value: datetime
    :return: nothing
    :rtype: None
    :raises LedgerArgumentError: when it is naive
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise LedgerArgumentError(f"{what} must be timezone-aware, got the naive {value.isoformat()}")


def _account_names(accounts: Iterable[str]) -> list[str]:
    """the accounts asked for, refusing one bare string.

    :param accounts: the accounts
    :ptype accounts: Iterable[str]
    :return: them, as a list
    :rtype: list[str]
    :raises TypeError: when ``accounts`` is a ``str``, which would otherwise be read as one-character accounts
    """
    if isinstance(accounts, str):
        raise TypeError(f"accounts must be a collection of account strings, not the one string {accounts!r}")
    return list(accounts)


def _encoded(document: dict[str, Any]) -> str:
    """a record's document as JSON text, refusing one that would read back different.

    :param document: the document
    :ptype document: dict[str, Any]
    :return: the JSON text
    :rtype: str
    :raises LedgerArgumentError: when it is not a dict, or does not survive a JSON round trip unchanged
    """
    if not isinstance(document, dict):
        raise LedgerArgumentError(f"a record's document must be a dict, got {type(document).__name__}")
    try:
        encoded = json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LedgerArgumentError(f"a record's document must be JSON-safe: {exc}") from exc
    # json.dumps quietly turns int keys into strings and tuples into lists; a document that reads back
    # different from what was written is refused rather than stored changed
    if json.loads(encoded) != document:
        raise LedgerArgumentError(
            "a record's document must survive a JSON round trip unchanged: string keys, lists not tuples"
        )
    return encoded
