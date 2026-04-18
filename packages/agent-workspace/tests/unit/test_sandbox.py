"""unit tests for threetears.agent.workspace.sandbox.WorkspaceSandbox.

covers ``from_config`` construction semantics and ``deny_reason`` messaging
against the inherited core ``PathSandbox`` contract.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from threetears.agent.workspace.config import AllowConfig, WorkspaceConfig
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.core.security import PathSandbox, SandboxDecision, SandboxDenied


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


class TestFromConfigGlobs:
    """``from_config`` forwards config.allow.read / write glob lists."""

    def test_allow_read_and_write_propagated_to_path_sandbox(self) -> None:
        """read and write globs are applied by the inherited check method."""
        config = WorkspaceConfig(
            allow=AllowConfig(read=["docs/**"], write=["out/**/*.yaml"]),
        )
        sandbox = WorkspaceSandbox.from_config(config)
        assert sandbox.check("read", "docs/readme.md") is SandboxDecision.ALLOW
        assert sandbox.check("read", "other/readme.md") is SandboxDecision.DENY
        assert sandbox.check("write", "out/a/b.yaml") is SandboxDecision.ALLOW
        assert sandbox.check("write", "out/a/b.txt") is SandboxDecision.DENY

    def test_default_allow_read_matches_everything_and_write_denies(self) -> None:
        """default AllowConfig reads everything, writes nothing (fail-closed)."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        assert sandbox.check("read", "anything.md") is SandboxDecision.ALLOW
        assert sandbox.check("write", "anything.md") is SandboxDecision.DENY


class TestDenyReasonMessaging:
    """deny_reason returns workspace-specific, actionable messages."""

    def test_unknown_action_message_names_action_and_expected_actions(self) -> None:
        """unknown action deny message names the offending action verb."""
        sandbox = WorkspaceSandbox.from_config(WorkspaceConfig())
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("delete", "foo.yaml")
        reason = exc_info.value.reason
        assert "'delete'" in reason
        assert "read" in reason
        assert "write" in reason

    def test_empty_key_message_flags_invalid_relative_path(self) -> None:
        """empty relative_path produces an invalid-relative_path reason."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("read", "")
        reason = exc_info.value.reason
        assert "invalid relative_path" in reason
        assert "empty" in reason

    def test_control_char_message_flags_invalid_relative_path(self) -> None:
        """NUL or low-ASCII control char in key produces invalid reason."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("read", "bad\x00name")
        reason = exc_info.value.reason
        assert "invalid relative_path" in reason
        assert "control" in reason or "NUL" in reason

    def test_oversize_key_message_flags_invalid_relative_path(self) -> None:
        """oversize key (> _MAX_KEY_LEN) produces an invalid-relative_path reason."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        oversize = "a" * (PathSandbox._MAX_KEY_LEN + 1)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("read", oversize)
        reason = exc_info.value.reason
        assert "invalid relative_path" in reason
        assert "length" in reason

    def test_absolute_path_message_forbids_absolute_and_parent_ref(self) -> None:
        """absolute path produces the absolute-or-dotdot reason."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("read", "/etc/passwd")
        reason = exc_info.value.reason
        assert "relative_path must not be absolute or contain '..'" == reason

    def test_parent_ref_message_forbids_absolute_and_parent_ref(self) -> None:
        """parent-ref (..) in key produces the absolute-or-dotdot reason."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"]))
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("read", "../etc/passwd")
        reason = exc_info.value.reason
        assert "relative_path must not be absolute or contain '..'" == reason

    def test_no_glob_match_mentions_action_target_and_globs(self) -> None:
        """no glob match in allow.write lists the configured write globs."""
        config = WorkspaceConfig(
            allow=AllowConfig(read=["**/*"], write=["out/**/*.yaml", "tmp/*.json"]),
        )
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("write", "foo.txt")
        reason = exc_info.value.reason
        assert "workspace.allow.write" in reason
        assert "'foo.txt'" in reason
        assert re.search(r"out/\*\*/\*\.yaml", reason) is not None
        assert re.search(r"tmp/\*\.json", reason) is not None

    def test_no_glob_match_on_empty_write_list_still_lists_globs(self) -> None:
        """fail-closed write path mentions the empty configured globs list."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"], write=[]))
        sandbox = WorkspaceSandbox.from_config(config)
        with pytest.raises(SandboxDenied) as exc_info:
            sandbox.enforce("write", "out/a.yaml")
        reason = exc_info.value.reason
        assert "workspace.allow.write" in reason
        assert "'out/a.yaml'" in reason
        assert "[]" in reason


class TestAllowPath:
    """matching globs produce ALLOW and enforce does not raise."""

    def test_matching_write_glob_allows_enforce(self) -> None:
        """a write key matching an allow.write glob returns ALLOW."""
        config = WorkspaceConfig(allow=AllowConfig(read=["**/*"], write=["out/**/*.yaml"]))
        sandbox = WorkspaceSandbox.from_config(config)
        assert sandbox.check("write", "out/sub/foo.yaml") is SandboxDecision.ALLOW
        sandbox.enforce("write", "out/sub/foo.yaml")


class TestSubclass:
    """WorkspaceSandbox is a PathSandbox subclass (not a reimplementation)."""

    def test_workspace_sandbox_is_path_sandbox_subclass(self) -> None:
        """direct subclass relationship preserved."""
        assert issubclass(WorkspaceSandbox, PathSandbox)
