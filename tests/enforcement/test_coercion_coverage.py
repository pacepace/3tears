"""Thin shell over the canonical coercion-coverage walker.

A Tool subclass overrides ``execute``, never ``run``. ``run`` is where
``normalize_kwargs`` coerces the model's raw argument dict into the types the tool
declares, so a subclass that overrides ``run`` receives whatever the model happened to
emit -- a string where an int was declared, a null where a list was -- and the coercion
step it skipped is invisible at the call site.

This locks in an invariant that is true across every package, so the first ``run`` override
fails at commit rather than at the first malformed model response in production. The walker
reads a base as Tool-ish by the suffix of its name, so LangChain's ``BaseTool`` and
``StructuredTool`` qualify; the two LangChain wrappers that override ``run`` without being
TearsTools are exempted, by class, in ``_coercion_coverage_exemptions.txt``.
"""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.coercion_coverage import (
    CoerceCoverageConfig,
    run_coercion_enforcement,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

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

_CONFIG = CoerceCoverageConfig(
    repo_root=_REPO_ROOT,
    src_roots=_SRC_ROOTS,
    exemptions_path=Path(__file__).with_name("_coercion_coverage_exemptions.txt"),
)


def test_tool_subclasses_override_execute_not_run() -> None:
    """Overriding ``run`` bypasses ``normalize_kwargs``, so the tool sees the model's raw
    argument types rather than the ones it declared."""
    run_coercion_enforcement(_CONFIG)
