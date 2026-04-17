"""WorkspaceSandbox — thin builder atop core :class:`PathSandbox`.

translates a :class:`WorkspaceConfig` (declared in ``agent.yaml``) into a
concrete :class:`PathSandbox` with named filesystem roots (``templates``
and ``bind``) and glob allow-lists for ``read`` / ``write``. overrides
:meth:`PathSandbox._deny_reason` to produce workspace-specific,
actionable error messages that reference the agent's configured globs —
this lets an LLM tool caller self-correct (e.g. "write glob list does
not include ``*.json``, let me pick a yaml path").

no new access-control logic is introduced here: validation and glob
matching live in the core sandbox; this module is a factory plus
messaging surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from threetears.agent.workspace.config import WorkspaceConfig
from threetears.core.security import PathSandbox


class WorkspaceSandbox(PathSandbox):
    """workspace-aware :class:`PathSandbox` built from :class:`WorkspaceConfig`.

    subclass only overrides :meth:`_deny_reason` for workspace-specific
    messaging; all validation and glob-matching logic is inherited from
    core :class:`PathSandbox`. action vocabulary is fixed at
    ``{"read", "write"}`` per parent contract.
    """

    @classmethod
    def from_config(cls, config: WorkspaceConfig) -> WorkspaceSandbox:
        """build workspace sandbox from a :class:`WorkspaceConfig`.

        named fs roots are populated only when the corresponding config
        field is set:

        - ``templates`` root from ``config.templates_dir`` when not None
        - ``bind`` root from ``config.bind_root`` when not None

        missing roots are simply absent from :attr:`_fs_roots` — callers
        that :meth:`resolve_fs_path` against an unregistered name receive
        :class:`KeyError` (programmer error, not a policy denial).

        :param config: workspace configuration loaded from ``agent.yaml``
        :ptype config: WorkspaceConfig
        :return: workspace sandbox with fs roots and allow-lists applied
        :rtype: WorkspaceSandbox
        """
        fs_roots: dict[str, Path] = {}
        if config.templates_dir is not None:
            fs_roots["templates"] = config.templates_dir
        if config.bind_root is not None:
            fs_roots["bind"] = config.bind_root
        return cls(
            fs_roots=fs_roots,
            allow_read=config.allow.read,
            allow_write=config.allow.write,
        )

    def _deny_reason(self, action: str, target: str) -> str:
        """return workspace-specific, actionable deny reason for LLM feedback.

        routes through the parent :meth:`PathSandbox._classify_relative_key`
        so the validation classification is a single source of truth; this
        method only reshapes the reason text for agent-visible messaging.

        failure modes surfaced:

        - unknown action verb -> ``"action {action!r} not understood; expected 'read' or 'write'"``
        - empty / oversize / control-char key -> ``"invalid relative_path: {detail}"``
        - absolute path or ``..`` segment -> ``"relative_path must not be absolute or contain '..'"``
        - no glob match -> ``"no glob in workspace.allow.{action} matched {target!r}; configured globs: [...]"``

        :param action: action verb that was denied
        :ptype action: str
        :param target: target relative key that was denied
        :ptype target: str
        :return: reason string carried on :class:`SandboxDenied.reason`
        :rtype: str
        """
        result: str
        if action not in {"read", "write"}:
            result = f"action {action!r} not understood; expected 'read' or 'write'"
        else:
            mode: Literal["read", "write"] = "read" if action == "read" else "write"
            _, core_reason = self._classify_relative_key(target, mode)
            result = self._reshape_reason(action, target, mode, core_reason)
        return result

    def _reshape_reason(
        self,
        action: str,
        target: str,
        mode: Literal["read", "write"],
        core_reason: str,
    ) -> str:
        """map core validation reason to workspace-flavored message.

        dispatches on the short-form core reason strings emitted by
        :meth:`PathSandbox._classify_relative_key`; every possible core
        reason is handled explicitly so new core classifications fail
        loudly via the fallback branch rather than silently leaking
        generic text to the agent.

        :param action: action verb that was denied
        :ptype action: str
        :param target: target relative key that was denied
        :ptype target: str
        :param mode: read/write mode (matches action for read/write verbs)
        :ptype mode: Literal["read", "write"]
        :param core_reason: reason string from core classifier
        :ptype core_reason: str
        :return: workspace-flavored reason string
        :rtype: str
        """
        result: str
        if core_reason == "key is empty":
            result = "invalid relative_path: empty"
        elif core_reason.startswith("key length "):
            detail = core_reason.replace("key ", "", 1)
            result = f"invalid relative_path: {detail}"
        elif core_reason == "key contains NUL or control character":
            result = "invalid relative_path: contains NUL or control character"
        elif core_reason in (
            "absolute path not allowed",
            "parent-ref (..) not allowed in key",
        ):
            result = "relative_path must not be absolute or contain '..'"
        elif core_reason.startswith("no "):
            globs = self._allow_read if mode == "read" else self._allow_write
            result = f"no glob in workspace.allow.{action} matched {target!r}; configured globs: {globs}"
        else:
            result = f"invalid relative_path: {core_reason}"
        return result
