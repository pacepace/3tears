"""datasource driver subpackage: ABC + factory + shared helpers.

public surface (DS-09-09):

- :class:`Driver` -- abstract base every concrete driver subclasses
- :func:`create_driver` -- the factory; the ONLY public way to
  construct a concrete driver instance
- :class:`TableRow` / :class:`ColumnRow` -- pinned row shapes returned
  by ``list_tables`` / ``list_columns``
- :class:`DriverConnectError` / :class:`DriverAuthError` /
  :class:`DriverMissingCredentialError` /
  :class:`DriverCredentialPausedError` -- the connection-failure types
  EVERY driver raises, so a caller catches them once rather than once
  per backend (see :mod:`threetears.datasources.drivers.errors`)
- :class:`ConnectGuard` / :class:`CredentialRefusalGuards` /
  :func:`guarded_connect` -- a credential the warehouse refused is not
  tried again by any replica until it is replaced or a probe succeeds
  (see :mod:`threetears.datasources.drivers.connect_guard`)

concrete driver classes (``AsyncpgDriver``, ``RedshiftDriver``,
``SnowflakeDriver``, ``BigQueryDriver``) are NOT re-exported here: they
are built exclusively via :func:`create_driver`.

import contract (owner ruling, 2026-10-24: no lazy imports in serving
code): importing :mod:`threetears.datasources.drivers.factory` imports
every driver module, so building a driver on the event loop loads
nothing. Redshift's module is imported under ``try/except ImportError``,
so without the redshift extra the package still imports and a Redshift
config raises the ImportError naming it; the Snowflake and BigQuery
drivers are stubs that load no backend library. a runtime audit
(``test_lazy_imports.py``) verifies both in clean subprocesses.
"""

from __future__ import annotations

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow
from threetears.datasources.drivers.connect_guard import (
    DEFAULT_PAUSE_SECONDS,
    ConnectGuard,
    CredentialRefusalGuards,
    guarded_connect,
)
from threetears.datasources.drivers.errors import (
    DriverAuthError,
    DriverConnectError,
    DriverCredentialPausedError,
    DriverMissingCredentialError,
    DriverPoolBusyError,
)
from threetears.datasources.drivers.factory import create_driver

__all__ = [
    "DEFAULT_PAUSE_SECONDS",
    "ColumnRow",
    "ConnectGuard",
    "CredentialRefusalGuards",
    "Driver",
    "DriverAuthError",
    "DriverConnectError",
    "DriverCredentialPausedError",
    "DriverMissingCredentialError",
    "DriverPoolBusyError",
    "TableRow",
    "create_driver",
    "guarded_connect",
]
