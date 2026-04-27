"""WorkspaceSandbox — thin builder atop core :class:`PathSandbox`.

after namespace-task-01 phase 7 the sandbox is a **syntactic** and
**filesystem-jail** primitive only. path-glob allow-lists are no
longer the authority for workspace file access — the unified rbac
evaluator is, via the custom action types ``read_file_matching:<glob>``
and ``write_file_matching:<glob>`` (see
:data:`threetears.agent.acl.READ_FILE_MATCHING_PREFIX` /
:data:`threetears.agent.acl.WRITE_FILE_MATCHING_PREFIX`). the legacy
``workspace.allow.read`` / ``workspace.allow.write`` config fields are
read at bootstrap by
:func:`aibots_agents.runtime.access_translation.translate_workspace_allow_to_assignments`
and translated into role assignments; this module no longer consults
those fields.

:class:`WorkspaceSandbox` therefore keeps only:

- named filesystem roots (``templates`` and ``bind``) with root-jail
  containment via :meth:`PathSandbox.resolve_fs_path`.
- syntactic relative-key validation (empty, length, control chars,
  absolute, parent-ref) via :meth:`PathSandbox.validate_syntax`.

the glob-matching :meth:`PathSandbox.check` /
:meth:`PathSandbox.check_relative_key` / :meth:`PathSandbox.enforce`
surface is no longer used by workspace tools. tools now call
:meth:`PathSandbox.validate_syntax` followed by
:func:`threetears.agent.workspace.authorize.authorize_workspace_file_access`.
"""

from __future__ import annotations

from pathlib import Path

from threetears.agent.workspace.config import WorkspaceConfig
from threetears.core.security import PathSandbox

__all__ = [
    "WorkspaceSandbox",
]


class WorkspaceSandbox(PathSandbox):
    """workspace-aware :class:`PathSandbox` built from :class:`WorkspaceConfig`.

    subclass carries only the named ``templates`` / ``bind`` filesystem
    roots and inherits :meth:`PathSandbox.validate_syntax` for the
    syntactic relative-key check. the ``allow`` globs on
    :class:`WorkspaceConfig` are retired from enforcement as of
    namespace-task-01 phase 7; the translator reads them at bootstrap
    and encodes them as rbac role assignments.

    the inherited glob-based :meth:`PathSandbox.check` /
    :meth:`PathSandbox.enforce` vocabulary stays available for callers
    outside the workspace tool path that still rely on the glob-driven
    semantics (none exist in the 3tears monorepo after phase 7), but
    workspace file-access enforcement routes through the rbac
    :func:`threetears.agent.workspace.authorize.authorize_workspace_file_access`
    helper.
    """

    @classmethod
    def from_config(cls, config: WorkspaceConfig) -> WorkspaceSandbox:
        """build a workspace sandbox from a :class:`WorkspaceConfig`.

        named fs roots are populated only when the corresponding config
        field is set:

        - ``templates`` root from ``config.templates_dir`` when not None
        - ``bind`` root from ``config.bind_root`` when not None

        missing roots are simply absent from :attr:`_fs_roots`; callers
        that :meth:`resolve_fs_path` against an unregistered name
        receive :class:`KeyError` (programmer error, not a policy
        denial).

        globs are intentionally NOT threaded into the underlying
        :class:`PathSandbox`: namespace-task-01 phase 7 retired the
        ``workspace.allow`` enforcement in favor of the rbac
        path-level gate. ``allow_read=[]`` and ``allow_write=[]``
        are passed in so the inherited glob-driven methods (unused
        from the workspace tool path) fail-closed if a stray caller
        still invokes them.

        :param config: workspace configuration loaded from ``agent.yaml``
        :ptype config: WorkspaceConfig
        :return: workspace sandbox with fs roots registered and empty
            glob allow-lists (rbac-enforced glob matching lives in the
            evaluator, not here)
        :rtype: WorkspaceSandbox
        """
        fs_roots: dict[str, Path] = {}
        if config.templates_dir is not None:
            fs_roots["templates"] = config.templates_dir
        if config.bind_root is not None:
            fs_roots["bind"] = config.bind_root
        return cls(
            fs_roots=fs_roots,
            allow_read=[],
            allow_write=[],
        )
