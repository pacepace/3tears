"""factory for workspace TearsTool instances.

each tool module under :mod:`threetears.agent.workspace.tools` defines a
``_build`` callable that constructs the tool from runtime dependencies and
registers itself at import time via
:func:`~threetears.agent.workspace.tool_registry.register_tool_builder`.
this module imports the tools package at module top, so every
registration has fired before :func:`build_workspace_tools` iterates the
registry. the registry lives in its own lower module so the tools package
never imports this one.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.agent.acl import AclCache
from threetears.agent.tools.base_tool import TearsTool

from threetears.agent.workspace import tools as _tools  # noqa: F401 - registers every tool builder
from threetears.agent.workspace.tool_registry import register_tool_builder, registered_tool_builders

__all__ = [
    "build_workspace_tools",
    "register_tool_builder",
]


def build_workspace_tools(
    *,
    acl_cache: AclCache,
    namespace_collection: Any,
    workspace_collection: Any = None,
    workspace_file_collection: Any = None,
    workspace_file_version_collection: Any = None,
    sandbox: Any = None,
    lease: Any = None,
    context_provider: Any = None,
    nats_client: Any = None,
    namespace: str | None = None,
    agent_id: UUID | None = None,
    pod_id: UUID | None = None,
    config: Any = None,
    db_pool: Any = None,
    validators: Any = None,
    customer_id: UUID | None = None,
) -> list[TearsTool]:
    """
    instantiates every registered workspace tool with shared dependencies.

    the canonical kwargs span every dependency any workspace tool may
    need. each registered ``_build`` accepts ``**kwargs`` so unused deps
    are dropped. the tools package is imported at this module's top, so
    every registration side-effect has fired before iteration.

    :param workspace_collection: collection for Workspace entities
    :ptype workspace_collection: Any
    :param workspace_file_collection: collection for WorkspaceFile entities
    :ptype workspace_file_collection: Any
    :param workspace_file_version_collection: collection for WorkspaceFileVersion entities
    :ptype workspace_file_version_collection: Any
    :param sandbox: workspace sandbox enforcing path constraints
    :ptype sandbox: Any
    :param lease: KV lease coordinator for cross-pod write serialization
    :ptype lease: Any
    :param context_provider: callable returning current ToolContextManager
    :ptype context_provider: Any
    :param nats_client: NATS client for cross-pod messaging and audit publish
    :ptype nats_client: Any
    :param namespace: NATS subject namespace; write-class tools use it to
        build the ``{namespace}.audit.workspace.{action}`` subject
    :ptype namespace: str | None
    :param agent_id: identifier of owning agent
    :ptype agent_id: UUID | None
    :param pod_id: identifier of host pod
    :ptype pod_id: UUID | None
    :param config: workspace configuration; write-class tool builders
        fall back to ``config.validators`` when the explicit
        ``validators`` kwarg is omitted
    :ptype config: Any
    :param db_pool: asyncpg pool (or pool-like) for transactional lifecycle writes
    :ptype db_pool: Any
    :param validators: explicit per-pattern validator entries that win
        over ``config.validators``; used by tests and callers that wire
        dispatch without building a full WorkspaceConfig. accepts
        ``list[ValidatorEntry]`` in production, ``Any`` at the factory
        boundary because each builder defensively coerces via
        :func:`_resolve_validators`
    :ptype validators: Any
    :param acl_cache: shared :class:`AclCache` wired with membership
        + grant loaders at bootstrap; every tool threads it into
        ``authorize_workspace_access`` so cross-agent + user-scoped
        grants are enforced before any database IO. REQUIRED on every
        production and test call path
    :ptype acl_cache: AclCache
    :param namespace_collection: three-tier
        :class:`NamespaceCollection` (typed ``Any`` at this boundary
        because the concrete class lives outside the agent-workspace
        package's dep graph). **NO BUILDER CONSUMES IT.** it used to say
        :class:`WorkspaceCreateTool` wrote the paired ``namespaces`` row
        through :meth:`NamespaceCollection.save_entity` on every create;
        that write moved to the hub, which owns direct database access,
        and ``WorkspaceCreateTool``'s builder never asks for this key.
        the parameter is inert -- it is placed in the dependency bundle
        and read by nothing
    :ptype namespace_collection: Any
    :param customer_id: owning-customer UUID for the agent this tool
        bundle serves; stamped onto newly-created workspaces so the
        paired ``namespaces`` row carries the right customer
    :ptype customer_id: UUID | None
    :return: list of constructed TearsTool instances
    :rtype: list[TearsTool]
    """
    deps: dict[str, Any] = {
        "workspace_collection": workspace_collection,
        "workspace_file_collection": workspace_file_collection,
        "workspace_file_version_collection": workspace_file_version_collection,
        "sandbox": sandbox,
        "lease": lease,
        "context_provider": context_provider,
        "nats_client": nats_client,
        "namespace": namespace,
        "agent_id": agent_id,
        "pod_id": pod_id,
        "config": config,
        "db_pool": db_pool,
        "validators": validators,
        "acl_cache": acl_cache,
        "namespace_collection": namespace_collection,
        "customer_id": customer_id,
    }
    return [builder(**deps) for builder in registered_tool_builders()]
