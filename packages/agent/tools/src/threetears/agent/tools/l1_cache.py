"""L1 SQLite factory for a tool pod's collection stack (``coll-task-07c`` TP-05).

The sanctioned per-process construction site, in the same shape the registry
(:mod:`threetears.registry.l1_cache`) and channels presence
(:mod:`threetears.channels.presence.l1_cache`) already use: ONE factory that constructs and
initializes a shared :class:`~threetears.core.cache.sqlite.SQLiteBackend`, which the caller hands
to :meth:`~threetears.core.collections.registry.CollectionRegistry.configure` as the default L1
tier. Every other construction site is a bespoke cache wrapper in disguise, and
``tests/enforcement/test_cache_primitive_usage.py`` enforces that by file.

**Why this exists rather than "the pod builds its own".** A tool pod may live in a partner-operated
fourth repository, where this repo's per-repo cache-primitive allowlist cannot see it at all. So the
question "is this bespoke SQLite cache sanctioned?" is made MOOT rather than exempted: the pod never
constructs a backend, it declares its tables and calls
:func:`~threetears.agent.tools.bootstrap.build_tool_pod_collection_stack`, which comes here.

Unlike the registry's factory this one takes the caller's :class:`~sqlalchemy.MetaData`: a tool
pod's Collection-backed surfaces are defined by the pod, not by this package, so there is no fixed
table set to declare here. What is fixed is the SHAPE -- one named in-memory database per process,
every declared table mirrored into it before the backend is handed out.

**The runtime's own tables are the exception, and they are added rather than expected.** The pod
declares what IT holds; :func:`register_tool_pod_runtime_tables` adds what the RUNTIME holds --
today, ``object_resolutions``, the two-tier store behind
:class:`~threetears.agent.tools.object_resolver.HubObjectResolver`. It runs inside
:func:`create_tool_pod_l1_backend` so a pod cannot forget it: a runtime table missing from L1 is
not an error, it is a collection whose reads silently miss forever, which is the failure mode this
whole surface exists to stop.
"""

from __future__ import annotations

from typing import Final

from sqlalchemy import BigInteger, Column, MetaData, String, Table
from sqlalchemy.dialects.postgresql import TIMESTAMP

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.observe import get_logger

from threetears.agent.tools.object_resolution_collection import OBJECT_RESOLUTIONS_TABLE

__all__ = [
    "TOOL_POD_L1_DB_NAME",
    "create_tool_pod_l1_backend",
    "register_tool_pod_runtime_tables",
]

log = get_logger(__name__)

#: the named in-memory SQLite database one tool-pod process shares across every collection on its
#: registry. Named rather than anonymous because ``SQLiteBackend`` keys its connection on it, so two
#: collections asking for the same name genuinely share one database within the process.
TOOL_POD_L1_DB_NAME: Final[str] = "tool_pod_l1_cache"


def register_tool_pod_runtime_tables(metadata: MetaData) -> MetaData:
    """declare every table the tool-pod runtime itself holds onto ``metadata``.

    Idempotent, because a host may hold one :class:`~sqlalchemy.MetaData` for the life of the
    process and hand it here more than once; a second ``Table(...)`` on the same name would raise
    :class:`~sqlalchemy.exc.InvalidRequestError` rather than merge.

    ``object_resolutions`` mirrors
    :class:`~threetears.agent.tools.object_resolution_collection.ObjectResolutionCollection`: the
    composite ``(customer_id, object_id)`` primary key is the collection's declared key, both
    columns holding stringified UUIDs. There is no L3 table behind it -- the hub is the system of
    record for an object's id -> key mapping, and this tier only remembers what it answered.

    :param metadata: the metadata the pod's L1 database is built from
    :ptype metadata: MetaData
    :return: the same metadata, with the runtime's tables present
    :rtype: MetaData
    """
    if OBJECT_RESOLUTIONS_TABLE not in metadata.tables:
        Table(
            OBJECT_RESOLUTIONS_TABLE,
            metadata,
            Column("customer_id", String(36), primary_key=True),
            Column("object_id", String(36), primary_key=True),
            Column("s3_key", String(2048), nullable=False),
            Column("mime_type", String(255), nullable=False),
            Column("size_bytes", BigInteger, nullable=False),
            Column("summary", String(4096), nullable=True),
            Column("category", String(255), nullable=True),
            Column("date_created", TIMESTAMP(timezone=True), nullable=False),
            Column("date_updated", TIMESTAMP(timezone=True), nullable=False),
        )
    return metadata


def create_tool_pod_l1_backend(metadata: MetaData, *, db_name: str = TOOL_POD_L1_DB_NAME) -> SQLiteBackend:
    """create and initialize the shared L1 backend for one tool-pod process.

    The runtime's own tables are registered onto ``metadata`` first, so a pod that declared none
    still gets a working L1 tier for the collections the runtime holds on its behalf.

    :param metadata: the pod's declared Collection tables, mirrored into the L1 database
    :ptype metadata: MetaData
    :param db_name: name of the in-memory database; override only when one process genuinely needs
        two independent L1 tiers
    :ptype db_name: str
    :return: initialized backend, ready to pass as the registry's default L1 tier
    :rtype: SQLiteBackend
    """
    register_tool_pod_runtime_tables(metadata)
    backend = SQLiteBackend(db_name=db_name)
    backend.initialize(metadata)
    log.info(
        "tool pod L1 cache initialized",
        extra={"extra_data": {"db_name": db_name, "table_count": len(metadata.tables)}},
    )
    return backend
