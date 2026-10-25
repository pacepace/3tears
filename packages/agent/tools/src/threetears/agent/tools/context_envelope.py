"""unified call-context envelope for NATS tool dispatch, under its agent-tools name.

:class:`CallContext` and :func:`bind_log_context` are defined in
:mod:`threetears.core.tool_wire.context`: the registry reads the envelope as
well as the pod, and core is the package both depend on. this module
re-exports the same objects, so an import from here and one from core name
one class, and ``isinstance`` and nested validation hold across both.
"""

from __future__ import annotations

from threetears.core.tool_wire.context import CallContext, bind_log_context

__all__ = [
    "CallContext",
    "bind_log_context",
]
