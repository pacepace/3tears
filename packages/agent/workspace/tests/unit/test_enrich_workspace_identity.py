"""the customer lookup must land on the platform tables wherever they live.

:func:`enrich_workspace_identity` reads the owning customer off the hub's
``namespaces`` row for a workspace. That table is in the schema the PLATFORM
configures, which on the shipped compose stack is not called ``platform``. So
this module fixes a deployment whose platform schema is called something else
and asserts the read still resolves.

**Why the schema name is the whole point of the fixture.** The one existing
place the real helper runs against a real database
(``tests/integration/test_cross_agent_workspace.py``) creates a schema
literally named ``platform`` and binds ``search_path`` to
``"<agent schema>", platform``. Under that fixture a statement carrying a
hardcoded ``platform.`` qualifier and a statement carrying none are
indistinguishable, so it can pass whether or not the qualifier is there. The
deployment where they differ is the one nobody was testing, and it is the only
one that ships.

:class:`_BrokerLikePool` is therefore a stand-in for the platform's query
broker rather than for asyncpg, and it models the two behaviours that decide
this question:

1. a request names a NAMESPACE; the broker resolves that namespace's row and
   issues ``SET search_path TO <that row's schema_name>`` -- ONE schema, with
   no fallback entry behind it;
2. the broker rewrites no table reference, so a schema-qualified name is
   executed verbatim and fails when that schema does not exist.

Both are asserted about the fake itself in
:class:`TestTheFakeReproducesTheDeployment`, so the fixture cannot drift into
proving something easier than the real thing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid7

import pytest
from threetears.agent.workspace.tools.helpers import enrich_workspace_identity
from threetears.core.namespaces import (
    PLATFORM_RBAC_READ_NAMESPACE,
    PLURAL_PREFIX_WORKSPACE,
    build_agent_namespace_name,
    build_namespace_name,
)

from _helpers.workspace_shims import FakeWorkspaceEntity

#: the schema this deployment's platform tables actually live in. Deliberately
#: NOT ``platform``: that is the value the removed hardcoded default used, and
#: a fixture that keeps using it cannot tell a correct statement from the bug.
_PLATFORM_SCHEMA = "aibots"

#: the calling pod's own schema, which is where its default namespace resolves
#: to and where the ``namespaces`` table is NOT.
_AGENT_SCHEMA = "agent_0123456789abcdef0123456789abcdef"

#: ``FROM <schema>.<table>`` / ``FROM <table>`` in the statements under test.
_FROM = re.compile(r"\bfrom\s+(?:(?P<schema>[a-z_][a-z0-9_]*)\.)?(?P<table>[a-z_][a-z0-9_]*)", re.IGNORECASE)


class UndefinedTable(Exception):
    """what Postgres raises when a statement names a relation that is not there.

    Stands in for ``asyncpg.exceptions.UndefinedTableError``, which the
    platform broker surfaces to a caller as a data-layer failure. Modelled
    with a local type so this module needs no asyncpg import to make the
    point.
    """


class UnknownNamespace(Exception):
    """what the broker answers when the namespace a request names has no row."""


# parity-exempt: stands in for the PLATFORM's query broker, whose namespace-to-search_path resolution has no protocol in this repo to declare parity against
@dataclass
class _BrokerLikePool:
    """pool that resolves a statement the way the platform's broker does.

    :ivar default_namespace: namespace a request binds to when it names none;
        the calling pod's own, exactly as the production proxy backend does
    :ivar namespace_schemas: the ``namespaces.name -> namespaces.schema_name``
        mapping the platform holds, and the ONLY thing that decides which
        schema a statement resolves against
    :ivar schemas: schema name -> table name -> rows, for the schemas that
        exist on this deployment
    :ivar calls: every ``(query, params, namespace)`` served, for assertions
    """

    default_namespace: str
    namespace_schemas: dict[str, str]
    schemas: dict[str, dict[str, list[dict[str, Any]]]]
    calls: list[tuple[str, tuple[Any, ...], str | None]] = field(default_factory=list)

    async def fetchrow(
        self,
        query: str,
        *params: Any,
        namespace: str | None = None,
        customer_scope: UUID | None = None,
    ) -> dict[str, Any] | None:
        """resolve and serve one row, or raise the way the deployment would.

        :param query: the caller's SQL
        :ptype query: str
        :param params: bound parameter values
        :ptype params: Any
        :param namespace: namespace to bind the statement to; ``None`` means
            the caller's own
        :ptype namespace: str | None
        :param customer_scope: accepted for surface parity with the production
            proxy backend; unused here
        :ptype customer_scope: UUID | None
        :return: the matching row, or ``None`` when nothing matches
        :rtype: dict[str, Any] | None
        :raises UnknownNamespace: when no namespace row carries that name
        :raises UndefinedTable: when the statement names a relation that does
            not exist in the schema it resolved against
        """
        del customer_scope
        self.calls.append((query, params, namespace))
        bound = namespace or self.default_namespace
        if bound not in self.namespace_schemas:
            raise UnknownNamespace(f"namespace not found: {bound}")
        # the broker issues ``SET search_path TO <schema_name>`` -- one schema,
        # nothing behind it -- so this is the whole of what a bare name sees.
        search_path = self.namespace_schemas[bound]
        matched = _FROM.search(query)
        if matched is None:
            raise UndefinedTable(f"no FROM clause found in: {query}")
        table = matched.group("table")
        qualifier = matched.group("schema")
        # a qualified reference is executed VERBATIM: the broker rewrites no
        # table reference, so it reaches whatever schema the caller named.
        target = qualifier if qualifier is not None else search_path
        tables = self.schemas.get(target)
        if tables is None or table not in tables:
            named = f"{qualifier}.{table}" if qualifier is not None else table
            raise UndefinedTable(f'relation "{named}" does not exist')
        rows = tables[table]
        return next((row for row in rows if row.get("namespace_id") == params[0]), None)


@dataclass
class _FakeWorkspace(FakeWorkspaceEntity):
    """structural stand-in for :class:`Workspace` carrying what the helper touches.

    :ivar id: workspace id, which is ALSO the id of its paired namespace row
    :ivar namespace_name: canonical namespace name of the workspace itself
    :ivar customer_id: what the helper stamps
    """

    id: UUID
    namespace_name: str
    customer_id: UUID | None = None


def _make_pool(*, workspace_id: UUID, customer_id: UUID) -> _BrokerLikePool:
    """build a deployment whose platform schema is not called ``platform``.

    The ``namespaces`` table exists in :data:`_PLATFORM_SCHEMA` and nowhere
    else, and the caller's own namespace resolves to :data:`_AGENT_SCHEMA`,
    which holds only the agent's own tables.

    :param workspace_id: id of the workspace whose namespace row is seeded
    :ptype workspace_id: UUID
    :param customer_id: customer the seeded namespace row carries
    :ptype customer_id: UUID
    :return: the configured pool
    :rtype: _BrokerLikePool
    """
    agent_namespace = build_agent_namespace_name(uuid7())
    return _BrokerLikePool(
        default_namespace=agent_namespace,
        namespace_schemas={
            agent_namespace: _AGENT_SCHEMA,
            # convert at border: canonical namespace-name token segment
            build_namespace_name(PLURAL_PREFIX_WORKSPACE, str(workspace_id)): _AGENT_SCHEMA,
            PLATFORM_RBAC_READ_NAMESPACE: _PLATFORM_SCHEMA,
        },
        schemas={
            _AGENT_SCHEMA: {"workspaces": [], "workspace_files": []},
            _PLATFORM_SCHEMA: {
                "namespaces": [{"namespace_id": workspace_id, "customer_id": customer_id}],
            },
        },
    )


def _make_workspace(workspace_id: UUID) -> _FakeWorkspace:
    """build the workspace entity the helper enriches.

    :param workspace_id: workspace id
    :ptype workspace_id: UUID
    :return: fake workspace entity
    :rtype: _FakeWorkspace
    """
    return _FakeWorkspace(
        id=workspace_id,
        # convert at border: canonical namespace-name token segment
        namespace_name=build_namespace_name(PLURAL_PREFIX_WORKSPACE, str(workspace_id)),
    )


class TestTheFakeReproducesTheDeployment:
    """controls: the fixture must model the two behaviours that decide this.

    These pass before and after the fix. They exist so that the assertions
    below cannot quietly become true because the fixture got easier.
    """

    async def test_a_schema_qualified_name_reaches_a_schema_that_is_not_there(self) -> None:
        """the broker rewrites nothing, so a hardcoded qualifier fails here.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=uuid7())
        with pytest.raises(UndefinedTable, match="platform.namespaces"):
            await pool.fetchrow(
                "SELECT customer_id FROM platform.namespaces WHERE namespace_id = $1",
                workspace_id,
                namespace=PLATFORM_RBAC_READ_NAMESPACE,
            )

    async def test_a_bare_name_does_not_resolve_under_the_callers_own_namespace(self) -> None:
        """un-qualifying alone is not the fix: the read has to BIND as well.

        The caller is an agent pod whose default namespace resolves to its own
        schema, and that schema has no ``namespaces`` table. This is the
        failure the platform's own tool server records for the retired
        agent-side namespace write.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=uuid7())
        with pytest.raises(UndefinedTable, match='relation "namespaces" does not exist'):
            await pool.fetchrow(
                "SELECT customer_id FROM namespaces WHERE namespace_id = $1",
                workspace_id,
            )

    async def test_a_bare_name_resolves_when_bound_to_the_carve_out(self) -> None:
        """binding the read is what puts the platform tables on the path.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        customer_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=customer_id)
        row = await pool.fetchrow(
            "SELECT customer_id FROM namespaces WHERE namespace_id = $1",
            workspace_id,
            namespace=PLATFORM_RBAC_READ_NAMESPACE,
        )
        assert row is not None
        assert row["customer_id"] == customer_id


class TestEnrichWorkspaceIdentity:
    """the helper itself, on a deployment whose platform schema is renamed."""

    async def test_it_stamps_the_customer_where_the_schema_is_not_named_platform(self) -> None:
        """the read lands, and the entity carries the customer afterwards.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        customer_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=customer_id)
        workspace = _make_workspace(workspace_id)

        await enrich_workspace_identity(workspace, pool)

        assert workspace.customer_id == customer_id

    async def test_it_binds_the_read_to_the_platform_carve_out_namespace(self) -> None:
        """the namespace is what routes the statement, so it must be passed.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=uuid7())

        await enrich_workspace_identity(_make_workspace(workspace_id), pool)

        assert [namespace for _, _, namespace in pool.calls] == [PLATFORM_RBAC_READ_NAMESPACE]

    async def test_the_statement_names_the_table_and_not_a_schema(self) -> None:
        """a qualifier here is a hardcoded schema name, which is the defect.

        Asserted on the statement the helper actually sent, so it holds
        whatever the deployment's schema happens to be called -- a fixture
        that owns the named schema proves nothing either way.

        :return: nothing
        :rtype: None
        """
        workspace_id = uuid7()
        pool = _make_pool(workspace_id=workspace_id, customer_id=uuid7())

        await enrich_workspace_identity(_make_workspace(workspace_id), pool)

        sent = pool.calls[0][0]
        matched = _FROM.search(sent)
        assert matched is not None
        assert matched.group("schema") is None, (
            f"{sent!r} qualifies its table with a schema name. The schema a deployment "
            "puts the platform tables in is configured, so the statement names the bare "
            "table and the request names the namespace."
        )
        assert matched.group("table") == "namespaces"

    async def test_a_workspace_with_no_namespace_row_is_left_unstamped(self) -> None:
        """a miss is not an error; the authorize step rejects an unstamped entity.

        :return: nothing
        :rtype: None
        """
        pool = _make_pool(workspace_id=uuid7(), customer_id=uuid7())
        workspace = _make_workspace(uuid7())

        await enrich_workspace_identity(workspace, pool)

        assert workspace.customer_id is None
