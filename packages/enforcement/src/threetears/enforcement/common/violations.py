"""violation dataclass shared across every enforcement domain.

every walker produces ``Violation`` records; the ``category`` field
namespaces them by domain (``cache.missing_collection``, ``underscore.A``,
etc.) so reports and exemptions key off a single uniform shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common.ast_helpers import relative_posix_path

__all__ = ["Violation"]


@dataclass(frozen=True)
class Violation:
    """one detected violation, uniformly described across domains.

    :ivar category: fully-qualified domain.subcategory identifier
        (``cache.missing_collection``, ``underscore.A``, etc.)
    :ivar file: absolute source path of the offending site
    :ivar line: 1-based line number of the offending site
    :ivar symbol: offending symbol — class name, table name, file
        basename, or other walker-specific identifier
    :ivar reason: human-readable explanation of the violation
    """

    category: str
    file: Path
    line: int
    symbol: str
    reason: str

    def format(self, repo_root: Path) -> str:
        """render this violation as a single line keyed for reports.

        format is ``[category] relpath:line:symbol  -- reason``. the
        relative path uses forward slashes regardless of host OS so
        report output is stable across platforms.

        :param repo_root: repo root for relative-path rendering
        :ptype repo_root: Path
        :return: single-line rendering
        :rtype: str
        """
        rel = relative_posix_path(self.file, repo_root)
        return f"[{self.category}] {rel}:{self.line}:{self.symbol}  -- {self.reason}"
