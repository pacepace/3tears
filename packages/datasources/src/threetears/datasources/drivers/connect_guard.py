"""stop logging in with a credential the warehouse has already refused.

a warehouse counts every failed login against the account and locks it after a handful
(Redshift: five, and it never unlocks by itself). a refused credential stays refused, so
every further login with it is another failure counted toward that lock -- and a platform
has many callers that connect on their own schedule: a reaper, an introspection sweep, a
coverage pass, an agent's query. one wrong stored password locked a production account in
about 75 minutes of background passes.

a driver given a :class:`ConnectGuard` asks it before every login and tells it about every
refusal. the guard :mod:`threetears.datasources` ships, :class:`CredentialRefusalGuards`,
remembers a refusal in a :class:`~threetears.core.coordination.WindowedCounter` -- shared
by every replica through L2, durable in L3 -- so after the first refusal no replica logs in
with that credential again: a connect raises :class:`DriverCredentialPausedError` without
contacting the warehouse.

a pause is lifted by :meth:`CredentialRefusalGuards.clear`, which its owner calls when the
credential is replaced or when an explicit probe with it succeeds. a probe is a driver
built WITHOUT a guard -- it is the one login allowed through a pause, and it is a person
asking. a pause also ends by itself after ``pause_seconds``, so a credential left wrong and
unattended costs one login per window rather than one per pass.

**it fails open.** if the coordination store cannot be read, the connect goes ahead and the
counter logs why. failing closed would turn a storage outage into an outage of every
datasource; failing open costs a login only when the store is down AND the credential is
wrong, and the warehouse's own lock is still five failures away.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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
    """what a driver asks before a login and tells after a refusal, for ONE datasource."""

    async def admit(self) -> None:
        """let a login go ahead, or refuse it without contacting the warehouse.

        :return: nothing
        :rtype: None
        :raises DriverCredentialPausedError: when this datasource's credential is paused
        """

    async def record_refusal(self, error: DriverAuthError) -> None:
        """remember that the warehouse refused this datasource's credential.

        :param error: the refusal, carrying the server's SQLSTATE and message
        :ptype error: DriverAuthError
        :return: nothing
        :rtype: None
        """


async def guarded_connect(guard: ConnectGuard | None, connect: Callable[[], Awaitable[T]]) -> T:
    """run one login under ``guard``: admitted first, a refusal recorded after.

    the one place a driver's login meets its guard, so every driver pauses the same way.
    a :class:`DriverMissingCredentialError` or :class:`DriverCredentialPausedError` is not
    recorded: each is raised before any login, so the warehouse counted nothing.

    :param guard: the datasource's guard, or ``None`` for an unguarded login (a probe)
    :ptype guard: ConnectGuard | None
    :param connect: the login, raising :class:`DriverAuthError` when the server refuses it
    :ptype connect: Callable[[], Awaitable[T]]
    :return: whatever ``connect`` returns
    :rtype: T
    :raises DriverCredentialPausedError: when the guard refuses the login
    :raises DriverAuthError: when the server refuses it; the refusal is recorded first
    """
    if guard is not None:
        await guard.admit()
    try:
        return await connect()
    except DriverMissingCredentialError, DriverCredentialPausedError:
        raise
    except DriverAuthError as exc:
        if guard is not None:
            await guard.record_refusal(exc)
        raise


class CredentialRefusalGuards:
    """fleet-wide pauses for refused datasource credentials, over a :class:`WindowedCounter`.

    one instance per process, over the registry whose tiers every replica shares. a driver
    gets the guard for its own datasource from :meth:`for_datasource`.

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

    def for_datasource(self, datasource_id: UUID, *, datasource_name: str) -> ConnectGuard:
        """the guard a driver for one datasource carries.

        :param datasource_id: the datasource the driver connects for
        :ptype datasource_id: UUID
        :param datasource_name: its name, for the refusal message and the log
        :ptype datasource_name: str
        :return: a guard bound to that datasource
        :rtype: ConnectGuard
        """
        return _DatasourceGuard(self, datasource_id, datasource_name)

    async def refused_at(self, datasource_id: UUID) -> datetime | None:
        """when this datasource's credential was refused, if it is paused now.

        :param datasource_id: the datasource to look up
        :ptype datasource_id: UUID
        :return: the pausing refusal's time, or ``None`` when it is not paused
        :rtype: datetime | None
        """
        state = await self._counter.state(_key(datasource_id))
        return None if state is None else datetime.fromtimestamp(state.window_start, tz=UTC)

    async def clear(self, datasource_id: UUID) -> None:
        """lift a pause: the credential was replaced, or a probe with it succeeded.

        :param datasource_id: the datasource whose pause to lift
        :ptype datasource_id: UUID
        :return: nothing
        :rtype: None
        """
        await self._counter.clear(_key(datasource_id))

    async def record(self, datasource_id: UUID) -> None:
        """record a refusal of this datasource's credential.

        :param datasource_id: the datasource the warehouse refused
        :ptype datasource_id: UUID
        :return: nothing
        :rtype: None
        """
        await self._counter.record_attempt(_key(datasource_id))


class _DatasourceGuard:
    """:class:`ConnectGuard` for one datasource, over :class:`CredentialRefusalGuards`."""

    def __init__(self, guards: CredentialRefusalGuards, datasource_id: UUID, datasource_name: str) -> None:
        """bind the shared pauses to one datasource.

        :param guards: the process's pauses
        :ptype guards: CredentialRefusalGuards
        :param datasource_id: the datasource this guard answers for
        :ptype datasource_id: UUID
        :param datasource_name: its name, for messages
        :ptype datasource_name: str
        :return: None
        :rtype: None
        """
        self._guards = guards
        self._datasource_id = datasource_id
        self._datasource_name = datasource_name

    async def admit(self) -> None:
        """refuse the login when the credential is paused.

        :return: nothing
        :rtype: None
        :raises DriverCredentialPausedError: when it is paused
        """
        refused_at = await self._guards.refused_at(self._datasource_id)
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
        await self._guards.record(self._datasource_id)
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


def _key(datasource_id: UUID) -> str:
    """the counter key for one datasource.

    :param datasource_id: the datasource
    :ptype datasource_id: UUID
    :return: its key
    :rtype: str
    """
    return f"{datasource_id}"
