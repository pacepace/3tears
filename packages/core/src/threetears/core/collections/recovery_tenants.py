"""the tenants a lost write's recovery advances a table under, when its record names no row
(:mod:`threetears.core.collections.lost_writes`: a keys-unknown window).

A table's restore declaration names its tenancy; one that names none takes its class's, and a table no class declares
(a tool pod's own, undeclared on purpose, known only from the epoch it carries) goes under every tenant it may be read
under, asked of no registry, so recovery reports no undeclared lookup for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from threetears.core.collections.generation import tenancy_of_table
from threetears.core.collections.undeclared_tables import is_undeclared_table
from threetears.core.tenancy import TENANCY_UNDECLARED, Tenancy, schema_tenant

__all__ = ["recovery_tenants"]

#: the scope a lost write's recovery gives a table it knows only from the epoch it carries (no class declares it: a
#: tool pod's own table, undeclared on purpose): every tenant it may be read under -- each schema holding it (its
#: pods' key, were their class to declare it by schema) and ``_all`` (every other declaration's followers read it).
#: The conservative scope: a broader invalidation never misses a follower. Passed explicitly in place of a tenancy, so
#: recovery asks no registry for the table and reports no undeclared lookup (``threetears_undeclared_table_total``
#: stays at zero for it, and next release's refusal of an undeclared lookup never breaks recovery)
_TRIGGER_ONLY: Final = "trigger_only"


def _tenancy_of(table: str, declared: Tenancy) -> Tenancy | str:
    """what a lost write's recovery advances ``table`` under: the tenancy its restore declaration names; for one that
    names none, its class's, or :data:`_TRIGGER_ONLY` for a table no class declares (asked quietly,
    :func:`~threetears.core.collections.undeclared_tables.is_undeclared_table`).

    :param table: the table
    :ptype table: str
    :param declared: the tenancy its restore declaration names
    :ptype declared: Tenancy
    :return: the tenancy, or :data:`_TRIGGER_ONLY`
    :rtype: Tenancy | str
    """
    if declared.kind != TENANCY_UNDECLARED.kind:
        return declared
    return _TRIGGER_ONLY if is_undeclared_table(table) else tenancy_of_table(table)


def recovery_tenants(
    table: str, declared: Tenancy, schemas: Sequence[str], *, platform_schema: str, schema: str | None
) -> tuple[str | None, ...]:
    """the tenants a rowless advance of ``table`` goes under.

    For an agent-schema table, each schema holding it (``schema`` alone, when the window is one schema's: a migration
    run's); ``_all`` for any other declared table. A table known only from the epoch it carries (no class declares it)
    goes under each pod schema holding it and ``_all``: every tenant. A table the registry answers undeclared for (the
    startup loader never ran) lives in its pod's schema; one in the platform schema stays on ``_all``.

    :param table: the table
    :ptype table: str
    :param declared: the tenancy its restore declaration names (``TENANCY_UNDECLARED`` for none)
    :ptype declared: Tenancy
    :param schemas: the schemas holding it
    :ptype schemas: Sequence[str]
    :param platform_schema: the platform schema
    :ptype platform_schema: str
    :param schema: the one schema the window's writes ran in, or ``None`` for any
    :ptype schema: str | None
    :return: the tenants, ``None`` for ``_all``
    :rtype: tuple[str | None, ...]
    """
    tenancy = _tenancy_of(table, declared)
    kind = tenancy if isinstance(tenancy, str) else tenancy.kind
    held = [each for each in schemas if schema is None or each == schema]
    pods = tuple(schema_tenant(each) for each in held if each != platform_schema)
    tenants: tuple[str | None, ...] = (None,)
    if kind == _TRIGGER_ONLY:
        tenants = (*pods, None)
    elif kind == "schema":
        tenants = tuple(schema_tenant(each) for each in held) or (None,)
    elif kind == TENANCY_UNDECLARED.kind and held and platform_schema not in held:
        tenants = pods
    return tenants
