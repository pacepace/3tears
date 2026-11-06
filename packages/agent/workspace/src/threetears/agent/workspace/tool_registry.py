"""registry of workspace TearsTool builders.

each tool module under :mod:`threetears.agent.workspace.tools` registers
its ``_build`` callable here at import time via
:func:`register_tool_builder`. the registry sits below both the tools
package and :mod:`threetears.agent.workspace.factory`, so the factory can
import the tools package at module top without an import cycle.
"""

from __future__ import annotations

from collections.abc import Callable

from threetears.agent.tools.base_tool import TearsTool

__all__ = [
    "register_tool_builder",
    "registered_tool_builders",
]

_TOOL_BUILDERS: list[Callable[..., TearsTool]] = []


def register_tool_builder(builder: Callable[..., TearsTool]) -> None:
    """
    appends a tool builder to the registry.

    each tool module calls this at import time. builders take ``**kwargs``
    so :func:`~threetears.agent.workspace.factory.build_workspace_tools`
    can pass every canonical dependency and each builder consumes only
    what it needs.

    dedup: if an identical ``builder`` object is already registered
    (e.g. a test harness that reloads the tools subpackage), the call is
    a no-op so the factory does not emit duplicates.

    :param builder: callable returning a constructed TearsTool
    :ptype builder: Callable[..., TearsTool]
    :return: None
    :rtype: None
    """
    if builder in _TOOL_BUILDERS:
        return
    _TOOL_BUILDERS.append(builder)


def registered_tool_builders() -> tuple[Callable[..., TearsTool], ...]:
    """
    returns the registered tool builders in registration order.

    :return: snapshot of the registry
    :rtype: tuple[Callable[..., TearsTool], ...]
    """
    return tuple(_TOOL_BUILDERS)
