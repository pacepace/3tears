"""tests for threetears.core.security.sandbox.PathSandbox.

covers relative-key validation (empty, oversize, absolute, ``..``,
control chars), glob allow-lists with ``**`` recursion, hidden-file
policy, and filesystem-path resolution with root containment and
symlink-escape rejection.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from threetears.core.security.sandbox import (
    PathSandbox,
    SandboxDecision,
    SandboxDenied,
)


def _make(
    *,
    tmp_path: Path,
    allow_read: list[str] | None = None,
    allow_write: list[str] | None = None,
    extra_roots: dict[str, Path] | None = None,
) -> PathSandbox:
    """construct a :class:`PathSandbox` with one ``templates`` root.

    :param tmp_path: per-test temporary directory fixture root
    :ptype tmp_path: Path
    :param allow_read: explicit read-mode glob allow-list
    :ptype allow_read: list[str] | None
    :param allow_write: explicit write-mode glob allow-list
    :ptype allow_write: list[str] | None
    :param extra_roots: additional named roots to register
    :ptype extra_roots: dict[str, Path] | None
    :return: configured sandbox
    :rtype: PathSandbox
    """
    roots: dict[str, Path] = {"templates": tmp_path}
    if extra_roots is not None:
        roots.update(extra_roots)
    return PathSandbox(
        fs_roots=roots,
        allow_read=list(allow_read) if allow_read is not None else [],
        allow_write=list(allow_write) if allow_write is not None else [],
    )


class TestConstructor:
    """constructor resolves roots and stores copies of allow-lists."""

    def test_roots_resolved_at_construction(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        sb = PathSandbox(
            fs_roots={"templates": sub},
            allow_read=[],
            allow_write=[],
        )
        # verify resolved-root invariant through the public API: the
        # resolver must join ``<resolved_root>/<key>`` -- any un-resolved
        # root would produce a different absolute path.
        assert sb.resolve_fs_path("x.txt", "templates") == sub.resolve() / "x.txt"

    def test_allow_lists_copied(self, tmp_path: Path) -> None:
        read_allow: list[str] = ["*.yaml"]
        sb = PathSandbox(
            fs_roots={"templates": tmp_path},
            allow_read=read_allow,
            allow_write=[],
        )
        # mutate the original input after construction; defensive-copy
        # invariant means the sandbox must still match ``*.yaml`` (the
        # appended ``*.json`` must not leak into the sandbox's list).
        read_allow.append("*.json")
        assert sb.check_relative_key("foo.yaml", "read") == SandboxDecision.ALLOW
        assert sb.check_relative_key("foo.json", "read") == SandboxDecision.DENY

    def test_keyword_only_arguments(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            PathSandbox({"templates": tmp_path}, [], [])  # type: ignore[misc]


class TestCheckRelativeKey:
    """`check_relative_key` input validation and glob matching."""

    def test_empty_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("", "read") == SandboxDecision.DENY

    def test_oversize_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        oversize = "a" * (PathSandbox.MAX_KEY_LEN + 1)
        assert sb.check_relative_key(oversize, "read") == SandboxDecision.DENY

    def test_max_len_allowed(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        at_limit = "a" * PathSandbox.MAX_KEY_LEN
        assert sb.check_relative_key(at_limit, "read") == SandboxDecision.ALLOW

    def test_absolute_path_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("/abs/path", "read") == SandboxDecision.DENY

    def test_parent_ref_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("..", "read") == SandboxDecision.DENY

    def test_parent_ref_in_middle_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("a/../b", "read") == SandboxDecision.DENY

    def test_nul_byte_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("a/b\x00c", "read") == SandboxDecision.DENY

    def test_control_char_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("a\x01b", "read") == SandboxDecision.DENY

    def test_tab_is_allowed_char(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("a\tb", "read") == SandboxDecision.ALLOW

    def test_newline_is_allowed_char(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key("a\nb", "read") == SandboxDecision.ALLOW

    def test_hidden_file_no_special_gate(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        assert sb.check_relative_key(".hidden", "read") == SandboxDecision.ALLOW

    def test_fail_closed_on_empty_allow(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_write=[])
        assert sb.check_relative_key("foo.yaml", "write") == SandboxDecision.DENY

    def test_single_segment_glob_allow(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_write=["*.yaml"])
        assert sb.check_relative_key("foo.yaml", "write") == SandboxDecision.ALLOW

    def test_single_segment_glob_does_not_match_multi(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_write=["*.yaml"])
        assert sb.check_relative_key("subdir/foo.yaml", "write") == SandboxDecision.DENY

    def test_recursive_glob_matches_multi_segment(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_write=["**/*.yaml"])
        assert sb.check_relative_key("subdir/foo.yaml", "write") == SandboxDecision.ALLOW

    def test_recursive_glob_deep(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*.yaml"])
        assert sb.check_relative_key("a/b/c.yaml", "read") == SandboxDecision.ALLOW

    def test_read_and_write_lists_are_independent(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["*.yaml"], allow_write=[])
        assert sb.check_relative_key("foo.yaml", "read") == SandboxDecision.ALLOW
        assert sb.check_relative_key("foo.yaml", "write") == SandboxDecision.DENY


class TestCheck:
    """`check` handles action vocabulary and delegates by mode."""

    def test_unknown_action_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"], allow_write=["**/*"])
        assert sb.check("exec", "foo.yaml") == SandboxDecision.DENY

    def test_empty_action_denied(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"], allow_write=["**/*"])
        assert sb.check("", "foo.yaml") == SandboxDecision.DENY

    def test_read_delegates(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["*.yaml"])
        assert sb.check("read", "foo.yaml") == SandboxDecision.ALLOW
        assert sb.check("read", "foo.json") == SandboxDecision.DENY

    def test_write_delegates(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_write=["*.yaml"])
        assert sb.check("write", "foo.yaml") == SandboxDecision.ALLOW
        assert sb.check("write", "foo.json") == SandboxDecision.DENY


class TestEnforce:
    """`enforce` raises :class:`SandboxDenied` with actionable reason."""

    def test_enforce_allow_does_not_raise(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        sb.enforce("read", "foo.yaml")

    def test_enforce_unknown_action_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("exec", "foo.yaml")
        assert info.value.action == "exec"
        assert info.value.target == "foo.yaml"
        assert "action" in info.value.reason

    def test_enforce_empty_key_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "")
        assert "empty" in info.value.reason

    def test_enforce_absolute_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "/etc/passwd")
        assert "absolute" in info.value.reason

    def test_enforce_parent_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "a/../b")
        assert ".." in info.value.reason or "parent" in info.value.reason

    def test_enforce_control_char_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "a\x00b")
        assert "control" in info.value.reason or "NUL" in info.value.reason

    def test_enforce_no_matching_glob_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["*.yaml"])
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", "foo.json")
        assert "glob" in info.value.reason or "match" in info.value.reason

    def test_enforce_oversize_reason(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        oversize = "a" * (PathSandbox.MAX_KEY_LEN + 1)
        with pytest.raises(SandboxDenied) as info:
            sb.enforce("read", oversize)
        assert "length" in info.value.reason or "long" in info.value.reason


class TestResolveFsPath:
    """`resolve_fs_path` root containment, symlink escape, missing root."""

    def test_missing_root_raises_key_error(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(KeyError):
            sb.resolve_fs_path("anything", "does_not_exist")

    def test_happy_path_returns_resolved(self, tmp_path: Path) -> None:
        inner = tmp_path / "a"
        inner.mkdir()
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        result = sb.resolve_fs_path("a", "templates")
        assert result == inner.resolve()

    def test_nested_relative_path(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c.yaml"
        deep.parent.mkdir(parents=True)
        deep.write_text("x")
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        result = sb.resolve_fs_path("a/b/c.yaml", "templates")
        assert result == deep.resolve()

    def test_path_object_relative_ok(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        result = sb.resolve_fs_path(Path("a"), "templates")
        assert result == (tmp_path / "a").resolve()

    def test_parent_escape_rejected(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.resolve_fs_path("../escape", "templates")
        assert "escape" in info.value.reason or "root" in info.value.reason

    def test_absolute_path_object_rejected(self, tmp_path: Path) -> None:
        sb = _make(tmp_path=tmp_path, allow_read=["**/*"])
        with pytest.raises(SandboxDenied) as info:
            sb.resolve_fs_path(Path("/etc/passwd"), "templates")
        assert "absolute" in info.value.reason

    def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        link = root / "link"
        os.symlink(outside, link)
        sb = PathSandbox(
            fs_roots={"root": root},
            allow_read=["**/*"],
            allow_write=[],
        )
        with pytest.raises(SandboxDenied) as info:
            sb.resolve_fs_path("link", "root")
        assert "escape" in info.value.reason or "root" in info.value.reason

    def test_symlink_inside_root_ok(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        target = root / "target.txt"
        target.write_text("ok")
        link = root / "link"
        os.symlink(target, link)
        sb = PathSandbox(
            fs_roots={"root": root},
            allow_read=["**/*"],
            allow_write=[],
        )
        result = sb.resolve_fs_path("link", "root")
        assert result == target.resolve()


class TestValidateSyntax:
    """``validate_syntax`` enforces steps 1-5 without glob matching."""

    def test_empty_key_raises(self, tmp_path: Path) -> None:
        """empty key -> SandboxDenied with ``key is empty`` reason."""
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        with pytest.raises(SandboxDenied) as info:
            sb.validate_syntax("")
        assert "empty" in info.value.reason

    def test_oversize_key_raises(self, tmp_path: Path) -> None:
        """key longer than ``MAX_KEY_LEN`` -> denied."""
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        oversize = "a" * (PathSandbox.MAX_KEY_LEN + 1)
        with pytest.raises(SandboxDenied) as info:
            sb.validate_syntax(oversize)
        assert "length" in info.value.reason

    def test_control_char_raises(self, tmp_path: Path) -> None:
        """key containing NUL byte -> denied."""
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        with pytest.raises(SandboxDenied):
            sb.validate_syntax("bad\x00key")

    def test_absolute_path_raises(self, tmp_path: Path) -> None:
        """absolute path -> denied."""
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        with pytest.raises(SandboxDenied) as info:
            sb.validate_syntax("/etc/passwd")
        assert "absolute" in info.value.reason

    def test_parent_ref_raises(self, tmp_path: Path) -> None:
        """parent-ref (..) segment in key -> denied."""
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        with pytest.raises(SandboxDenied) as info:
            sb.validate_syntax("../etc/passwd")
        assert "parent" in info.value.reason

    def test_valid_relative_passes_even_with_empty_globs(
        self,
        tmp_path: Path,
    ) -> None:
        """syntactically valid key passes regardless of glob allow-lists.

        the rbac path-level gate takes over glob matching in
        namespace-task-01 phase 7; the sandbox's syntactic check runs
        whether or not any glob is configured.
        """
        sb = _make(tmp_path=tmp_path, allow_read=[], allow_write=[])
        sb.validate_syntax("docs/readme.md")
