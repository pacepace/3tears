"""multi-line human-readable report builder shared across every domain.

every domain's pytest entry point prints the same shape of diagnostic
report — repo anchor, src-root inventory, mode, violation breakdown by
category, then the violations themselves. this module produces that
report from the standardised ``Violation`` shape.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from threetears.enforcement.common.exemptions import Exemption
from threetears.enforcement.common.violations import Violation

__all__ = ["emit_report"]


def emit_report(
    violations: list[Violation],
    src_roots: tuple[Path, ...],
    exemptions: list[Exemption],
    mode: str,
    repo_root: Path,
    domain: str,
) -> str:
    """build a multi-line report describing the run and its findings.

    the report shape is::

        repo_root: <abs path>
        src_roots: [<abs>, <abs>, ...]
        domain: <domain>
        mode: <strict|report>
        exemptions_loaded: <count>
        violations_total: <count>
          <category-1>: <count>
          <category-2>: <count>
          ...
        [<category>] <relpath>:<line>:<symbol>  -- <reason>
        [<category>] <relpath>:<line>:<symbol>  -- <reason>
        ...

    category breakdown lines appear in alphabetical order. per-violation
    lines are sorted by ``(category, file, line, symbol)`` for
    determinism. when ``violations`` is empty only the header block is
    emitted.

    :param violations: filtered violations to render (already passed
        through :func:`apply_exemptions
        <threetears.enforcement.common.exemptions.apply_exemptions>`)
    :ptype violations: list[Violation]
    :param src_roots: src roots that were scanned
    :ptype src_roots: tuple[Path, ...]
    :param exemptions: exemption entries that were loaded
    :ptype exemptions: list[Exemption]
    :param mode: resolved enforcement mode (``strict`` / ``report``)
    :ptype mode: str
    :param repo_root: repo root used for relative-path rendering
    :ptype repo_root: Path
    :param domain: domain identifier for the header
    :ptype domain: str
    :return: rendered multi-line report
    :rtype: str
    """
    counter: Counter[str] = Counter(v.category for v in violations)
    lines: list[str] = [
        f"repo_root: {repo_root}",
        f"src_roots: {[str(r) for r in src_roots]}",
        f"domain: {domain}",
        f"mode: {mode}",
        f"exemptions_loaded: {len(exemptions)}",
        f"violations_total: {len(violations)}",
    ]
    for category in sorted(counter):
        lines.append(f"  {category}: {counter[category]}")
    for violation in sorted(
        violations,
        key=lambda v: (v.category, str(v.file), v.line, v.symbol),
    ):
        lines.append(violation.format(repo_root))
    return "\n".join(lines)
