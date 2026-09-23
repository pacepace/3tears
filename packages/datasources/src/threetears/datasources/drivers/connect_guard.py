"""stop logging in with a credential the warehouse has already refused.

a warehouse counts every failed login against the account and locks it after a handful
(Redshift: five, and it never unlocks by itself). a refused credential stays refused, so
every further login with it is another failure counted toward that lock -- and a platform
has many callers that connect on their own schedule: a reaper, an introspection sweep, a
coverage pass, an agent's query. one wrong stored password locked a production account in
about 75 minutes of background passes.

a driver given a :class:`ConnectGuard` logs in through :func:`guarded_connect`, which asks
the guard before every login and tells it of every refusal and every success. the guard
:mod:`threetears.datasources` ships, :class:`CredentialRefusalGuards`, does two things:

- **it pauses a refused credential for the fleet.** a refusal is remembered in a
  :class:`~threetears.core.coordination.WindowedCounter` -- shared by every replica through
  L2, durable in L3 -- so after it no replica logs in with that credential again: a connect
  raises :class:`DriverCredentialPausedError` without contacting the warehouse.
- **it lets one login at a time through until one succeeds.** a burst of connects on a
  process that has not yet logged in with a credential -- a restart, a pool's first fill, a
  fan-out of queries -- would otherwise send every login before the first refusal is
  recorded, and a burst of five is a Redshift lock on its own. once a login with the
  credential succeeds, logins run concurrently again. the serialization is per process, so a
  burst across replicas costs at most one login per replica.

**the pause belongs to a CREDENTIAL, not to a datasource.** the owner names the credential
with a revision it chooses -- anything that changes when the credential is replaced and is
not the secret itself. replacing a credential therefore needs no step to lift the pause: the
new revision was never refused. and a holder still using the superseded credential -- a pool
not yet rebuilt, a replica not yet told -- can only pause the credential it holds, never the
one that replaced it.

a pause is lifted by :meth:`CredentialRefusalGuards.clear`, which its owner calls when an
explicit probe with the credential succeeds. a probe is a driver built WITHOUT a guard -- it
is the one login allowed through a pause, and it is a person asking. a pause also ends by
itself after ``pause_seconds``, so a credential left wrong and unattended costs one login per
window rather than one per pass.

**it fails open.** if the coordination store cannot be read or written, the connect goes
ahead (and a refusal still reaches the caller) and the counter logs why. failing closed would
turn a storage outage into an outage of every datasource; failing open costs a login only when
the store is down AND the credential is wrong, and the warehouse's own lock is still five
failures away.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import Final, Protocol, TypeVar
from uuid import UUID

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination import WindowedCounter
from threetears.observe import get_logger

from threetears.datasources.drivers.errors import (
    DriverAuthError,
    DriverCredentialPausedError,
    DriverMissingCredentialError,
)

__all__ = [
    "DEFAULT_PAUSE_SECONDS",
    "ConnectGuard",
    "CredentialRefusalGuards",
    "guarded_connect",
]

log = get_logger(__name__)

#: how long a refused credential stays paused when nobody acts: thirty days. long enough
#: that an unattended wrong credential cannot lock an account in any realistic span (five
#: windows), short enough that a pause nobody knew to clear does not outlive the reason.
DEFAULT_PAUSE_SECONDS: Final[int] = 30 * 24 * 60 * 60

_PURPOSE: Final[str] = "datasource_credential_refused"

T = TypeVar("T")


class ConnectGuard(Protocol):
    """what a driver asks before a login and tells after one, for ONE credential."""

    def serialized(self) -> AbstractAsyncContextManager[None]:
        """hold one login's slot: one at a time until a login with the credential succeeds.

        :return: a context held for the whole login, the admission check included
        :rtype: AbstractAsyncContextManager[None]
        """
        ...

    async def admit(self) -> None:
        """let a login go ahead, or refuse it without contacting the warehouse.

        :return: nothing
        :rtype: None
        :raises DriverCredentialPausedError: when this credential is paused
        """

    async def record_refusal(self, error: DriverAuthError) -> None:
        """remember that the warehouse refused this credential.

        :param error: the refusal, carrying the server's SQLSTATE and message
        :ptype error: DriverAuthError
        :return: nothing
        :rtype: None
        """

    def record_success(self) -> None:
        """remember that a login with this credential succeeded, so logins stop queuing.

        :return: nothing
        :rtype: None
        """


async def guarded_connect(guard: ConnectGuard | None, connect: Callable[[], Awaitable[T]]) -> T:
    """run one login under ``guard``: queued, admitted, and its outcome recorded.

    the one place a driver's login meets its guard, so every driver pauses the same way.
    a :class:`DriverMissingCredentialError` or :class:`DriverCredentialPausedError` is not
    recorded: each is raised before any login, so the warehouse counted nothing.

    :param guard: the credential's guard, or ``None`` for an unguarded login (a probe)
    :ptype guard: ConnectGuard | None
    :param connect: the login, raising :class:`DriverAuthError` when the server refuses it
    :ptype connect: Callable[[], Awaitable[T]]
    :return: whatever ``connect`` returns
    :rtype: T
    :raises DriverCredentialPausedError: when the guard refuses the login
    :raises DriverAuthError: when the server refuses it; the refusal is recorded first
    """
    if guard is None:
        return await connect()
    async with guard.serialized():
        await guard.admit()
        try:
            result = await connect()
        except DriverMissingCredentialError, DriverCredentialPausedError:
            raise
        except DriverAuthError as exc:
            await guard.record_refusal(exc)
            raise
        guard.record_success()
    return result


class CredentialRefusalGuards:
    """fleet-wide pauses for refused datasource credentials, over a :class:`WindowedCounter`.

    one instance per process, over the registry whose tiers every replica shares. a driver
    gets the guard for the credential it logs in with from :meth:`for_credential`.

    a credential is named by ``(datasource_id, credential_revision)``. the revision is the
    owner's: any string that changes when the credential is replaced -- a digest of the
    stored ciphertext, a reference's name. it must NOT be the secret or an unkeyed digest of
    it: the key lands in L2 and L3, where a bare digest of a password can be reversed by
    guessing.

    :param registry: the collection registry the pause lives in; L2 is what makes it
        shared across replicas, L3 what makes it survive a broker wipe
    :ptype registry: CollectionRegistry
    :param pause_seconds: how long a pause lasts when nobody clears it
    :ptype pause_seconds: int
    """

    def __init__(self, registry: CollectionRegistry, *, pause_seconds: int = DEFAULT_PAUSE_SECONDS) -> None:
        """build the counter the pauses are kept in.

        :param registry: the shared collection registry
        :ptype registry: CollectionRegistry
        :param pause_seconds: how long an unattended pause lasts
        :ptype pause_seconds: int
        :return: None
        :rtype: None
        :raises ValueError: when ``pause_seconds`` is not positive
        """
        self._counter = WindowedCounter(registry, purpose=_PURPOSE, window_seconds=pause_seconds, fail_open=True)
        self._local = _ProvenCredentials()

    def for_credential(
        self,
        datasource_id: UUID,
        *,
        credential_revision: str,
        datasource_name: str,
    ) -> ConnectGuard:
        """the guard a driver logging in with one credential carries.

        :param datasource_id: the datasource the driver connects for
        :ptype datasource_id: UUID
        :param credential_revision: names the credential the driver logs in with; see the
            class docstring for what it may and may not be
        :ptype credential_revision: str
        :param datasource_name: the datasource's name, for the refusal message and the log
        :ptype datasource_name: str
        :return: a guard bound to that credential
        :rtype: ConnectGuard
        :raises ValueError: when ``credential_revision`` is empty
        """
        if not credential_revision:
            raise ValueError(
                f"datasource {datasource_name}: a connect guard needs the revision of the credential it "
                "guards; an empty one would let every credential this datasource ever has share one pause"
            )
        return _CredentialGuard(
            counter=self._counter,
            local=self._local,
            key=_key(datasource_id, credential_revision),
            datasource_id=datasource_id,
            datasource_name=datasource_name,
        )

    async def refused_at(self, datasource_id: UUID, credential_revision: str) -> datetime | None:
        """when this credential was refused, if it is paused now.

        :param datasource_id: the datasource
        :ptype datasource_id: UUID
        :param credential_revision: the credential's revision
        :ptype credential_revision: str
        :return: the pausing refusal's time, or ``None`` when it is not paused
        :rtype: datetime | None
        """
        return await _refused_at(self._counter, _key(datasource_id, credential_revision))

    async def clear(self, datasource_id: UUID, credential_revision: str) -> None:
        """lift a pause: a probe with the credential succeeded.

        :param datasource_id: the datasource
        :ptype datasource_id: UUID
        :param credential_revision: the credential's revision
        :ptype credential_revision: str
        :return: nothing
        :rtype: None
        """
        await self._counter.clear(_key(datasource_id, credential_revision))

    async def record(self, datasource_id: UUID, credential_revision: str) -> None:
        """record a refusal of this credential -- a probe's, which ran without a guard.

        :param datasource_id: the datasource
        :ptype datasource_id: UUID
        :param credential_revision: the credential's revision
        :ptype credential_revision: str
        :return: nothing
        :rtype: None
        """
        await _record(self._counter, self._local, _key(datasource_id, credential_revision))


class _ProvenCredentials:
    """this process's own record: which credentials it has logged in with, and their queues.

    per process on purpose. the queue orders THIS process's logins; the fleet's knowledge of
    a refusal is the counter's. a credential leaves the proven set when it is refused, so its
    logins queue again.
    """

    def __init__(self) -> None:
        """start with nothing proven and no queues.

        :return: None
        :rtype: None
        """
        self._proven: set[str] = set()
        self._queues: dict[str, asyncio.Lock] = {}

    def mark_proven(self, key: str) -> None:
        """a login with this credential succeeded here, so its logins stop queuing.

        :param key: the credential's key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        self._proven.add(key)
        # not needed again unless the credential is later refused; a caller already waiting
        # on it holds its own reference and is let through by the re-check in slot().
        self._queues.pop(key, None)

    def forget(self, key: str) -> None:
        """the credential was refused, so its logins queue again.

        :param key: the credential's key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        self._proven.discard(key)

    @asynccontextmanager
    async def slot(self, key: str) -> AsyncIterator[None]:
        """hold one login's place in the credential's queue, until it is proven.

        :param key: the credential's key
        :ptype key: str
        :return: a context held for the whole login
        :rtype: AsyncIterator[None]
        """
        queue: asyncio.Lock | None = None
        if key not in self._proven:
            queue = self._queues.setdefault(key, asyncio.Lock())
            await queue.acquire()
            if key in self._proven:
                # proven while this caller waited: go ahead alongside everyone else.
                queue.release()
                queue = None
        try:
            yield
        finally:
            if queue is not None:
                queue.release()


class _CredentialGuard:
    """:class:`ConnectGuard` for one credential, over the process's counter and record."""

    def __init__(
        self,
        *,
        counter: WindowedCounter,
        local: _ProvenCredentials,
        key: str,
        datasource_id: UUID,
        datasource_name: str,
    ) -> None:
        """bind the shared pause and this process's record to one credential.

        :param counter: the fleet's pauses
        :ptype counter: WindowedCounter
        :param local: this process's proven credentials and queues
        :ptype local: _ProvenCredentials
        :param key: the credential's counter key
        :ptype key: str
        :param datasource_id: the datasource the credential belongs to
        :ptype datasource_id: UUID
        :param datasource_name: its name, for messages
        :ptype datasource_name: str
        :return: None
        :rtype: None
        """
        self._counter = counter
        self._local = local
        self._key = key
        self._datasource_id = datasource_id
        self._datasource_name = datasource_name

    def serialized(self) -> AbstractAsyncContextManager[None]:
        """hold one login's slot until the credential is proven in this process.

        :return: the slot
        :rtype: AbstractAsyncContextManager[None]
        """
        return self._local.slot(self._key)

    async def admit(self) -> None:
        """refuse the login when the credential is paused.

        :return: nothing
        :rtype: None
        :raises DriverCredentialPausedError: when it is paused
        """
        refused_at = await _refused_at(self._counter, self._key)
        if refused_at is not None:
            raise DriverCredentialPausedError(
                f"datasource {self._datasource_name}: the warehouse refused its credential at "
                f"{refused_at.isoformat()}, so no login was attempted; replace the credential or run a "
                "successful connection test to resume",
                refused_at=refused_at,
            )

    async def record_refusal(self, error: DriverAuthError) -> None:
        """pause the credential, and say so where an operator will look.

        :param error: the refusal
        :ptype error: DriverAuthError
        :return: nothing
        :rtype: None
        """
        await _record(self._counter, self._local, self._key)
        log.error(
            "datasource credential refused; pausing every login with it until it is replaced or a connection test "
            "succeeds",
            extra={
                "extra_data": {
                    "datasource_id": f"{self._datasource_id}",
                    "datasource_name": self._datasource_name,
                    "sqlstate": error.sqlstate,
                    "server_message": error.server_message,
                }
            },
        )

    def record_success(self) -> None:
        """the credential worked here, so later logins with it run concurrently.

        :return: nothing
        :rtype: None
        """
        self._local.mark_proven(self._key)


async def _refused_at(counter: WindowedCounter, key: str) -> datetime | None:
    """when the credential under ``key`` was refused, if it is paused now.

    :param counter: the fleet's pauses
    :ptype counter: WindowedCounter
    :param key: the credential's key
    :ptype key: str
    :return: the pausing refusal's time, or ``None``
    :rtype: datetime | None
    """
    state = await counter.state(key)
    return None if state is None else datetime.fromtimestamp(state.window_start, tz=UTC)


async def _record(counter: WindowedCounter, local: _ProvenCredentials, key: str) -> None:
    """pause the credential under ``key`` for the fleet, and stop trusting it here.

    :param counter: the fleet's pauses
    :ptype counter: WindowedCounter
    :param local: this process's record
    :ptype local: _ProvenCredentials
    :param key: the credential's key
    :ptype key: str
    :return: nothing
    :rtype: None
    """
    local.forget(key)
    await counter.record_attempt(key)


def _key(datasource_id: UUID, credential_revision: str) -> str:
    """the counter key for one credential.

    :param datasource_id: the datasource
    :ptype datasource_id: UUID
    :param credential_revision: the credential's revision
    :ptype credential_revision: str
    :return: its key
    :rtype: str
    """
    return f"{datasource_id}:{credential_revision}"
