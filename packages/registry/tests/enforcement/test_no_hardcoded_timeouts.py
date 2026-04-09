"""Enforcement test -- no hardcoded timeout literals in source code.

Scans all Python source files under the registry, agent-tools, and core
packages for numeric literals used as timeout values. Timeouts must be
sourced from configuration (env var, agent config, or tool declaration),
never hardcoded as magic numbers in constructor defaults or function
signatures.

Detects:
  - Function/method parameters named *timeout* with a numeric default
  - Module-level constants matching *TIMEOUT* with a numeric assignment
  - Keyword arguments named *timeout* passed with a numeric literal

Allowlist entries exist for genuine platform defaults that live in a
designated config layer (e.g. AgentEnvironment, DefaultCoreConfig).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGES_ROOT = Path(__file__).resolve().parent.parent.parent.parent

_SCAN_DIRS: list[tuple[str, Path]] = [
    ("registry", _PACKAGES_ROOT / "registry" / "src"),
    ("agent-tools", _PACKAGES_ROOT / "agent-tools" / "src"),
    ("core", _PACKAGES_ROOT / "core" / "src"),
    ("observe", _PACKAGES_ROOT / "observe" / "src"),
]

# Files that are designated config layers where sensible defaults belong.
# Use relative-to-src paths so the allowlist is readable.
_ALLOWED_DEFAULT_FILES: set[str] = {
    # Agent environment config -- platform defaults live here
    "threetears/core/config.py",
    # Registry config -- single source of truth for registry timeout defaults
    "threetears/registry/config.py",
}

# Specific (file_relative, param_name, line) tuples for narrow exceptions.
# Each must include a comment explaining why the exception exists.
_ALLOWED_EXCEPTIONS: set[tuple[str, str]] = set()


def _collect_src_files() -> list[tuple[str, Path, Path]]:
    """Collect all Python source files across scanned packages.

    :return: list of (package_name, src_root, file_path) tuples
    :rtype: list[tuple[str, Path, Path]]
    """
    results: list[tuple[str, Path, Path]] = []
    for pkg_name, src_dir in _SCAN_DIRS:
        if not src_dir.exists():
            continue
        for p in sorted(src_dir.rglob("*.py")):
            if p.name == "__init__.py":
                continue
            results.append((pkg_name, src_dir, p))
    return results


_SRC_FILES = _collect_src_files()
_SRC_IDS = [f"{pkg}:{path.relative_to(src)}" for pkg, src, path in _SRC_FILES]


def _is_allowed(file_relative: str) -> bool:
    """Check whether a file is in the config-layer allowlist.

    :param file_relative: path relative to src root
    :ptype file_relative: str
    :return: True if file is an allowed config layer
    :rtype: bool
    """
    return file_relative in _ALLOWED_DEFAULT_FILES


def _is_timeout_name(name: str) -> bool:
    """Check whether a parameter or variable name relates to timeouts.

    :param name: identifier name to check
    :ptype name: str
    :return: True if name contains 'timeout'
    :rtype: bool
    """
    return "timeout" in name.lower()


def _get_numeric_value(node: ast.expr) -> float | None:
    """Extract numeric value from an AST node if it is a literal.

    Handles plain numbers and negative numbers (UnaryOp with USub).

    :param node: AST expression node
    :ptype node: ast.expr
    :return: numeric value if node is a literal, None otherwise
    :rtype: float | None
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -float(node.operand.value)
    return None


class TestNoHardcodedTimeouts:
    """Timeout values must come from configuration, not magic numbers."""

    @pytest.mark.parametrize(
        "pkg_name, src_root, src_file",
        _SRC_FILES,
        ids=_SRC_IDS,
    )
    def test_no_hardcoded_timeout_defaults(
        self,
        pkg_name: str,
        src_root: Path,
        src_file: Path,
    ) -> None:
        """Function/method parameters named *timeout* must not have numeric defaults."""
        file_relative = str(src_file.relative_to(src_root))
        if _is_allowed(file_relative):
            return

        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        violations: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Check each parameter default
            all_args = node.args.args + node.args.kwonlyargs
            # defaults apply to the LAST len(defaults) of args
            num_args = len(node.args.args)
            num_defaults = len(node.args.defaults)
            offset = num_args - num_defaults

            for i, default in enumerate(node.args.defaults):
                arg_index = offset + i
                if arg_index < len(node.args.args):
                    arg = node.args.args[arg_index]
                    if _is_timeout_name(arg.arg):
                        val = _get_numeric_value(default)
                        if val is not None:
                            violations.append(
                                f"  line {node.lineno}: {node.name}() param '{arg.arg}' "
                                f"has hardcoded default {val}. "
                                f"Source timeout from config instead."
                            )

            for j, default in enumerate(node.args.kw_defaults):
                if default is None:
                    continue
                kwarg = node.args.kwonlyargs[j]
                if _is_timeout_name(kwarg.arg):
                    val = _get_numeric_value(default)
                    if val is not None:
                        violations.append(
                            f"  line {node.lineno}: {node.name}() param '{kwarg.arg}' "
                            f"has hardcoded default {val}. "
                            f"Source timeout from config instead."
                        )

        if violations:
            detail = "\n".join(violations)
            pytest.fail(
                f"{pkg_name}:{file_relative} has hardcoded timeout defaults:\n{detail}"
            )

    @pytest.mark.parametrize(
        "pkg_name, src_root, src_file",
        _SRC_FILES,
        ids=_SRC_IDS,
    )
    def test_no_hardcoded_timeout_constants(
        self,
        pkg_name: str,
        src_root: Path,
        src_file: Path,
    ) -> None:
        """Module-level constants matching *TIMEOUT* must not be numeric literals."""
        file_relative = str(src_file.relative_to(src_root))
        if _is_allowed(file_relative):
            return

        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        violations: list[str] = []

        for node in ast.iter_child_nodes(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if _is_timeout_name(target.id):
                    val = _get_numeric_value(node.value)
                    if val is not None:
                        violations.append(
                            f"  line {node.lineno}: {target.id} = {val} -- "
                            f"hardcoded timeout constant. "
                            f"Source from config instead."
                        )

        if violations:
            detail = "\n".join(violations)
            pytest.fail(
                f"{pkg_name}:{file_relative} has hardcoded timeout constants:\n{detail}"
            )
