"""who may read each data-source table's change records: the declarations its collection classes carry.

The data-source modules speak of customers, never of tenants (``test_capability_source.py`` bans the word
there), so the change log's declarations (:mod:`threetears.core.tenancy`) are built here once, under the
words those modules use.
"""

from __future__ import annotations

from typing import Final

from threetears.core.tenancy import Tenancy, no_tenant, tenant_column

__all__ = ["BY_CUSTOMER", "COLUMN_CATALOGUE", "TABLE_CATALOGUE"]

#: a table whose rows each belong to one customer, named by ``customer_id``
BY_CUSTOMER: Final[Tenancy] = tenant_column("customer_id")

#: a data source's table catalogue: no customer column; it is keyed by ``datasource_id``
TABLE_CATALOGUE: Final[Tenancy] = no_tenant(
    "a data source's table catalogue carries no customer column; it is keyed by datasource_id"
)

#: a data source's column catalogue: no customer column; it is keyed by ``datasource_id``
COLUMN_CATALOGUE: Final[Tenancy] = no_tenant(
    "a data source's column catalogue carries no customer column; it is keyed by datasource_id"
)
