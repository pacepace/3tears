"""Shared scaffolding for 3tears-enforcement domain scanners.

Public API re-exports the helpers that domain modules depend on so a
domain can ``from threetears.enforcement.common import ...`` without
reaching into submodule paths.
"""

from threetears.enforcement.common.ast_helpers import (
    iter_python_files,
    note_unscanned,
    parse_python_file,
    relative_posix_path,
    is_private_name,
    is_logger_call,
    is_suppress_call,
)
from threetears.enforcement.common.collection_registry import (
    CLIENT_KEYWORDS,
    CLIENT_SPELLINGS,
    L2_BINDER_METHODS,
    REGISTRY_CTOR,
    argument_spellings,
    callee_names,
    constructed_registries,
    constructed_registry_lines,
    dotted,
    l2_live_registries,
    names_a_live_client,
    receiver,
)
from threetears.enforcement.common.repo_layout import (
    find_repo_root,
    find_local_src_roots,
)
from threetears.enforcement.common.pyproject_discovery import (
    discover_src_roots,
    PyprojectError,
)
from threetears.enforcement.common.git_layout import (
    find_worktree_root,
    repo_identity,
)
from threetears.enforcement.common.inheritance import (
    ClassBaseGraph,
    collect_class_base_graph,
    extract_base_names,
    transitively_subclasses_any,
)
from threetears.enforcement.common.exemptions import (
    Exemption,
    parse_exemptions_with_rationale,
    apply_exemptions,
    ExemptionError,
)
from threetears.enforcement.common.modes import (
    MODE_REPORT,
    MODE_STRICT,
    resolve_mode,
    ModeError,
)
from threetears.enforcement.common.violations import (
    Violation,
)
from threetears.enforcement.common.reports import (
    emit_report,
)

__all__ = [
    "CLIENT_KEYWORDS",
    "CLIENT_SPELLINGS",
    "ClassBaseGraph",
    "Exemption",
    "ExemptionError",
    "L2_BINDER_METHODS",
    "MODE_REPORT",
    "MODE_STRICT",
    "ModeError",
    "PyprojectError",
    "REGISTRY_CTOR",
    "Violation",
    "apply_exemptions",
    "argument_spellings",
    "callee_names",
    "collect_class_base_graph",
    "constructed_registries",
    "constructed_registry_lines",
    "discover_src_roots",
    "dotted",
    "emit_report",
    "extract_base_names",
    "find_local_src_roots",
    "find_repo_root",
    "find_worktree_root",
    "is_logger_call",
    "is_private_name",
    "is_suppress_call",
    "iter_python_files",
    "l2_live_registries",
    "names_a_live_client",
    "note_unscanned",
    "parse_exemptions_with_rationale",
    "parse_python_file",
    "receiver",
    "relative_posix_path",
    "repo_identity",
    "resolve_mode",
    "transitively_subclasses_any",
]
