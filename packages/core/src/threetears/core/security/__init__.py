"""policy-driven access-control primitives.

public surface:

- :class:`Sandbox` — abstract base; subclasses implement ``check``,
  inherit ``enforce`` which raises :class:`SandboxDenied` on DENY
- :class:`PathSandbox` — concrete filesystem-shaped sandbox with named
  roots, symlink-escape rejection, glob allow-lists
- :class:`SandboxDecision` — str-enum with ``ALLOW`` / ``DENY``
- :class:`SandboxDenied` — :class:`PermissionError` subclass carrying
  ``action``, ``target``, ``reason`` attributes
"""

from threetears.core.security.sandbox import (
    PathSandbox,
    Sandbox,
    SandboxDecision,
    SandboxDenied,
)

__all__ = [
    "PathSandbox",
    "Sandbox",
    "SandboxDecision",
    "SandboxDenied",
]
