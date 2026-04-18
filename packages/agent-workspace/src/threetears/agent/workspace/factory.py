"""factory registry for workspace TearsTool instances.

each tool module under :mod:`threetears.agent.workspace.tools` defines a
``_build`` callable that constructs the tool from runtime dependencies and
registers itself with this factory at import time via
:func:`register_tool_builder`. :func:`build_workspace_tools` lazy-imports
the tools package to ensure registration side-effects fire before
iterating the registry, while keeping this module free of top-level
imports of the tools package -- the tools package imports
``register_tool_builder`` from here at its own module load time, so a top-
level import of ``tools`` in this file would create a circular import
graph during partial-initialization windows. lazy import is the chosen
remediation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import TearsTool

__all__ = [
    "build_workspace_tools",
    "register_tool_builder",
]

_TOOL_BUILDERS: list[Callable[..., TearsTool]] = []


def register_tool_builder(builder: Callable[..., TearsTool]) -> None:
    """
    appends a tool builder to the registry.

    each tool module calls this at import time. builders take ``**kwargs``
    so :func:`build_workspace_tools` can pass every canonical dependency
    and each builder consumes only what it needs.

    dedup: if an identical ``builder`` object is already registered
    (e.g. a test harness that reloads the tools subpackage), the call is
    a no-op so :func:`build_workspace_tools` does not emit duplicates.

    :param builder: callable returning a constructed TearsTool
    :ptype builder: Callable[..., TearsTool]
    :return: None
    :rtype: None
    """
    if builder in _TOOL_BUILDERS:
        return
    _TOOL_BUILDERS.append(builder)


def build_workspace_tools(
    *,
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
) -> list[TearsTool]:
    """
    instantiates every registered workspace tool with shared dependencies.

    the canonical kwargs span every dependency any workspace tool may
    need. each registered ``_build`` accepts ``**kwargs`` so unused deps
    are dropped. lazy-imports the tools package on first call so module
    registration side-effects fire before iteration, avoiding a circular
    import between this module and the tools subpackage.

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
    :return: list of constructed TearsTool instances
    :rtype: list[TearsTool]
    """
    # lazy import: tools modules import register_tool_builder from this
    # module at their own load time; importing tools at top level here
    # would create a circular import during partial initialization.
    from threetears.agent.workspace import tools as _tools  # noqa: F401

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
    }
    return [builder(**deps) for builder in _TOOL_BUILDERS]
