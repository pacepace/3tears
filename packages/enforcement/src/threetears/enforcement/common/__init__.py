"""Shared scaffolding for 3tears-enforcement domain scanners.

Public API re-exports the helpers that domain modules depend on so a
domain can ``from threetears.enforcement.common import ...`` without
reaching into submodule paths.
"""

from threetears.enforcement.common.ast_helpers import (
    iter_python_files,
    parse_python_file,
    relative_posix_path,
    is_private_name,
    is_logger_call,
    is_suppress_call,
)
from threetears.enforcement.common.repo_layout import (
    find_repo_root,
    find_local_src_roots,
)
from threetears.enforcement.common.pyproject_discovery import (
    discover_src_roots,
    PyprojectError,
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
    "ClassBaseGraph",
    "Exemption",
    "ExemptionError",
    "MODE_REPORT",
    "MODE_STRICT",
    "ModeError",
    "PyprojectError",
    "Violation",
    "apply_exemptions",
    "collect_class_base_graph",
    "discover_src_roots",
    "emit_report",
    "extract_base_names",
    "find_local_src_roots",
    "find_repo_root",
    "is_logger_call",
    "is_private_name",
    "is_suppress_call",
    "iter_python_files",
    "parse_exemptions_with_rationale",
    "parse_python_file",
    "relative_posix_path",
    "resolve_mode",
    "transitively_subclasses_any",
]
