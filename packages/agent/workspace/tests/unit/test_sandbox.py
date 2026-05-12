"""unit tests for threetears.agent.workspace.sandbox.WorkspaceSandbox.

covers ``from_config`` construction semantics against the inherited core
``PathSandbox`` contract. namespace-task-01 phase 7 retires the legacy
glob-driven ``workspace.allow`` enforcement; these tests now assert that
:meth:`WorkspaceSandbox.from_config` wires up named filesystem roots
without threading the allow globs into the underlying :class:`PathSandbox`.
the path-level rbac gate now lives in
:func:`threetears.agent.acl.evaluate_file_access` and is exercised by
:mod:`tests.unit.test_file_access` in ``packages/agent-acl`` and the
workspace tool tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.agent.workspace.config import AllowConfig, WorkspaceConfig
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.core.security import PathSandbox, SandboxDenied


class TestFromConfigRoots:
    """``from_config`` populates named fs_roots from config path fields."""

    def test_both_paths_populate_templates_and_bind_roots(self, tmp_path: Path) -> None:
        """both templates_dir and bind_root set -> both named roots present."""
        templates = tmp_path / "templates"
        templates.mkdir()
        bind = tmp_path / "bind"
        bind.mkdir()
        config = WorkspaceConfig(
            templates_dir=templates,
            bind_root=bind,
            allow=AllowConfig(read=["**/*"], write=["out/**"]),
        )
        sandbox = WorkspaceSandbox.from_config(config)
        resolved_templates = sandbox.resolve_fs_path("a.txt", "templates")
        resolved_bind = sandbox.resolve_fs_path("b.txt", "bind")
        assert resolved_templates.is_relative_to(templates.resolve())
        assert resolved_bind.is_relative_to(bind.resolve())

    def test_only_templates_set_bind_not_registered(self, tmp_path: Path) -> None:
        """templates_dir set, bind_root None -> only templates root registered."""
        templates = tmp_path / "templates"
        templates.mkdir()
        config = WorkspaceConfig(templates_dir=templates, allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        sandbox.resolve_fs_path("a.txt", "templates")
        with pytest.raises(KeyError):
            sandbox.resolve_fs_path("a.txt", "bind")

    def test_only_bind_set_templates_not_registered(self, tmp_path: Path) -> None:
        """bind_root set, templates_dir None -> only bind root registered."""
        bind = tmp_path / "bind"
        bind.mkdir()
        config = WorkspaceConfig(bind_root=bind, allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        sandbox.resolve_fs_path("b.txt", "bind")
        with pytest.raises(KeyError):
            sandbox.resolve_fs_path("b.txt", "templates")

    def test_empty_config_produces_empty_fs_roots(self) -> None:
        """WorkspaceConfig() -> sandbox with no registered roots; both lookups KeyError."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(KeyError):
            sandbox.resolve_fs_path("x", "templates")
        with pytest.raises(KeyError):
            sandbox.resolve_fs_path("x", "bind")


class TestFromConfigGlobsRetired:
    """``from_config`` no longer threads allow globs into the sandbox.

    namespace-task-01 phase 7: the ``workspace.allow`` globs are read at
    agent bootstrap by the access translator and materialized into rbac
    role assignments; the sandbox itself is no longer the authority for
    path-level access. the inherited glob allow-lists stay empty so a
    stray caller that still invokes the glob-driven surface fails
    closed.
    """

    def test_from_config_does_not_forward_allow_globs(self) -> None:
        """``from_config`` does not thread allow.read / allow.write globs.

        behavioral assertion: the inherited :meth:`PathSandbox.check`
        fails closed for every input because neither the explicit
        globs nor the defaults are applied. callers must use the new
        rbac gate, not :meth:`check` / :meth:`enforce`, for path-level
        decisions.
        """
        from threetears.core.security import SandboxDecision

        config = WorkspaceConfig(
            allow=AllowConfig(read=["docs/**"], write=["out/**/*.yaml"]),
        )
        sandbox = WorkspaceSandbox.from_config(config)
        # a path that would have matched the explicit glob fails closed
        # because the translator now owns the read/write glob decision.
        assert sandbox.check("read", "docs/readme.md") is SandboxDecision.DENY
        assert sandbox.check("write", "out/a.yaml") is SandboxDecision.DENY

    def test_validate_syntax_permits_valid_relative_path(self) -> None:
        """``validate_syntax`` passes a well-formed relative key."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        sandbox.validate_syntax("docs/readme.md")

    def test_validate_syntax_rejects_absolute_path(self) -> None:
        """``validate_syntax`` rejects absolute paths."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(SandboxDenied):
            sandbox.validate_syntax("/etc/passwd")

    def test_validate_syntax_rejects_parent_ref(self) -> None:
        """``validate_syntax`` rejects ``..`` path segments."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(SandboxDenied):
            sandbox.validate_syntax("../etc/passwd")

    def test_validate_syntax_rejects_empty(self) -> None:
        """``validate_syntax`` rejects empty keys."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(SandboxDenied):
            sandbox.validate_syntax("")

    def test_validate_syntax_rejects_control_char(self) -> None:
        """``validate_syntax`` rejects keys with NUL bytes."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(SandboxDenied):
            sandbox.validate_syntax("bad\x00name")

    def test_validate_syntax_rejects_oversize_key(self) -> None:
        """``validate_syntax`` rejects keys larger than ~512 bytes.

        the cap lives on :class:`PathSandbox` as a module-private
        constant (mirrored in
        ``packages/core/tests/unit/security/test_path_sandbox.py``);
        this test picks a hard-coded size well above the documented
        default so the assertion does not couple to the private
        attribute name.
        """
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        oversize = "a" * 2048
        with pytest.raises(SandboxDenied):
            sandbox.validate_syntax(oversize)


class TestSubclass:
    """WorkspaceSandbox is a PathSandbox subclass (not a reimplementation)."""

    def test_workspace_sandbox_is_path_sandbox_subclass(self) -> None:
        """direct subclass relationship preserved."""
        assert issubclass(WorkspaceSandbox, PathSandbox)
