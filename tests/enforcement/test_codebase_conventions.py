"""Thin shell over the canonical codebase-conventions walkers.

Four AST-level conventions, applied to **every** package in the workspace: no bare
``print()``, no stdlib ``logging.getLogger``, ``from __future__ import annotations`` at
module top, and a return annotation on every non-dunder, non-test function.

This replaces a hand-rolled copy that lived in ``packages/core/tests/enforcement/`` and
scanned ``threetears.core`` alone. The scanners themselves have been in
``packages/enforcement`` the whole time with no caller at all, so the monorepo was
simultaneously shipping the shared implementation and running a private one over part of
the tree. The exemptions below each name a module that legitimately does the thing.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.codebase_conventions import (
    CodebaseConventionsConfig,
    run_codebase_conventions_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every package's source tree. Explicit rather than discovered: discovery walks path-deps
#: and would pull in whatever a package happens to depend on, which is not the set this
#: repo is responsible for.
_SRC_ROOTS = tuple(
    sorted(
        {
            path.parent
            for pattern in ("packages/*/src/threetears", "packages/agent/*/src/threetears/agent")
            for path in _REPO_ROOT.glob(pattern)
            if path.is_dir()
        }
    )
)

_CONFIG = CodebaseConventionsConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    print_exempt_files={
        # Every enforcement runner ends by writing its rendered report to stderr. That IS
        # the domain's output: a scanner nobody can read the results of is a scanner that
        # fails silently. One entry per runner rather than a directory-wide pass, so a
        # print that appears in a walker or a config still fails.
        f"packages/enforcement/src/threetears/enforcement/{domain}/runner.py": "emits the enforcement report to stderr"
        for domain in (
            "cache",
            "codebase_conventions",
            "coercion_coverage",
            "dependency_alignment",
            "dict_state_detection",
            "fake_parity",
            "jwt_alg_pinning",
            "logger_coverage",
            "migration_yugabyte_safety",
            "nats_wrapper_usage",
            "no_silent_swallow",
            "no_stdlib_logging",
            "single_return",
            "underscore_access",
        )
    },
    getlogger_exempt_files={
        # The module that IMPLEMENTS get_logger. It has to reach stdlib logging; that is
        # the entire point of every other module being forbidden from doing so.
        "packages/observe/src/threetears/observe/logging.py": "implements get_logger over stdlib logging",
    },
)


class TestCodebaseConventions:
    def test_no_bare_print(self) -> None:
        """Output goes through the logger, so it carries correlation tags and structure."""
        run_codebase_conventions_enforcement(_CONFIG, walker="print")

    def test_no_stdlib_getlogger(self) -> None:
        """stdlib logging bypasses the ContextFormatter and the correlation tags, silently
        downgrading a whole package's logs to unstructured lines."""
        run_codebase_conventions_enforcement(_CONFIG, walker="stdlib_getlogger")

    def test_future_annotations_on_every_module(self) -> None:
        """Annotations stay lazy strings (PEP 563), so a forward reference never
        NameErrors at runtime."""
        run_codebase_conventions_enforcement(_CONFIG, walker="future_annotations")

    def test_every_function_declares_a_return_type(self) -> None:
        """The hand-rolled predecessor skipped private functions; this does not. An
        unannotated private helper is exactly where mypy stops being able to help."""
        run_codebase_conventions_enforcement(_CONFIG, walker="return_type")
