"""datasource namespace helpers: deterministic uuid5 ids + canonical names.

both helpers are runtime primitives consumed by Hub-side datasource
authorize wiring (``aibots/hub/datasources/authorize.py``) and the
agent-side datasource tool pod (``aibots/hub/datasources/tool_pod.py``).
the shapes they produce match the namespace-task-01 contract: every
datasource row materializes a ``datasource``-type namespace whose name
matches :func:`datasource_namespace_name` and whose UUID matches
:func:`datasource_namespace_id`.
"""

from __future__ import annotations

from uuid import NAMESPACE_DNS, UUID, uuid5

from threetears.core.namespaces import PLURAL_PREFIX_DATASOURCE, build_namespace_name

__all__ = [
    "DATASOURCE_NAMESPACE_TYPE",
    "datasource_namespace_id",
    "datasource_namespace_name",
]


DATASOURCE_NAMESPACE_TYPE = "datasource"


def datasource_namespace_id(datasource_id: UUID) -> UUID:
    """return deterministic namespace UUID for a datasource row.

    :param datasource_id: owning datasource UUID
    :ptype datasource_id: UUID
    :return: deterministic namespace UUID
    :rtype: UUID
    """
    return uuid5(NAMESPACE_DNS, f"threetears.namespaces.datasource.{datasource_id.hex}")


def datasource_namespace_name(datasource_name: str) -> str:
    """return canonical namespace name for a datasource.

    matches the lookup key :class:`DataSource*Tool.execute` uses via
    :meth:`NamespaceCollection.get_by_name` so every caller agrees on
    the shape. shape: ``datasources.<datasource_name>`` per the
    canonical plural-prefix + dot-separator form pinned by
    :func:`threetears.core.namespaces.build_namespace_name`. the
    datasource name segment is sanitized (any ``.`` replaced with
    ``-``) before interpolation.

    :param datasource_name: datasource's ``platform.datasources.name``
    :ptype datasource_name: str
    :return: canonical namespace name string
    :rtype: str
    """
    return build_namespace_name(PLURAL_PREFIX_DATASOURCE, datasource_name)
