"""The action catalogue: every eval action, declared once, for any transport to mount.

An :class:`Action` binds one operation (:mod:`threetears.evals.ops`) to its name, help, flat parameters,
result, permission class and agent-facing rendering. :func:`eval_catalogue` is the engine's actions plus
any a host contributes; :meth:`ActionCatalogue.mount` cuts a tool from it by permission class —
:func:`standard_tools` (``evals`` and ``evals_admin``) or :func:`read_only_tools` — and a mounted tool's
:meth:`MountedTool.call` carries a call out: ``help`` generated from the catalogue, undeclared parameters
refused, errors that teach, long work as a job to poll. A transport adapter is a thin binding of
:class:`MountedTool` to its server's tool shape (:mod:`threetears.evals.transports.fastmcp`).

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.actions.catalogue import (
    ACTION_NAME,
    DEFAULT_PREFIX,
    HELP_ACTION,
    PERMISSION_CLASSES,
    RESERVED_PARAMETERS,
    TOOL_NAME,
    Action,
    ActionCatalogue,
    ActionHandler,
    ActionOutcome,
    Caller,
    MountedTool,
    PermissionClass,
    ToolHints,
    ToolSpec,
    read_only_tools,
    standard_tools,
)
from threetears.evals.actions.engine import engine_actions, eval_catalogue

__all__ = [
    "ACTION_NAME",
    "DEFAULT_PREFIX",
    "HELP_ACTION",
    "PERMISSION_CLASSES",
    "RESERVED_PARAMETERS",
    "TOOL_NAME",
    "Action",
    "ActionCatalogue",
    "ActionHandler",
    "ActionOutcome",
    "Caller",
    "MountedTool",
    "PermissionClass",
    "ToolHints",
    "ToolSpec",
    "engine_actions",
    "eval_catalogue",
    "read_only_tools",
    "standard_tools",
]
