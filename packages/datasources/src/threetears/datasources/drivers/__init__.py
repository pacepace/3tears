"""datasource driver subpackage: ABC + factory + shared helpers.

public surface (DS-09-09):

- :class:`Driver` -- abstract base every concrete driver subclasses
- :func:`create_driver` -- the factory; the ONLY public way to
  construct a concrete driver instance
- :class:`TableRow` / :class:`ColumnRow` -- pinned row shapes returned
  by ``list_tables`` / ``list_columns``

concrete driver classes (``AsyncpgDriver``, ``RedshiftDriver``,
``SnowflakeDriver``, ``BigQueryDriver``) are NOT re-exported here. they
ship in shards 10 / 11 / 12 and are accessed exclusively via
:func:`create_driver`. re-exporting them at this layer would break the
lazy-import contract (any import of ``threetears.datasources.drivers``
would pull every backend library into ``sys.modules``).

lazy-import contract: importing :mod:`threetears.datasources`,
:mod:`threetears.datasources.drivers`, or
:mod:`threetears.datasources.drivers.factory` MUST NOT load any of
``asyncpg``, ``redshift_connector``, ``snowflake.connector``, or
``google.cloud.bigquery`` into ``sys.modules``. backend libs are
imported only inside the matching ``case`` arm of
:func:`create_driver`. a runtime audit (``test_lazy_imports.py``)
verifies this in a clean subprocess on every test run.
"""

from __future__ import annotations

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow
from threetears.datasources.drivers.factory import create_driver

__all__ = [
    "ColumnRow",
    "Driver",
    "TableRow",
    "create_driver",
]
