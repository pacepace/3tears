"""Generic configurable severity system for report modules.

Provides a dataclass-based severity configuration so consumers
can define their own severity levels without depending on any
domain-specific enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeverityConfig:
    """Configurable severity ordering and display labels.

    :param order: severity keys from most to least severe
    :ptype order: list[str]
    :param labels: mapping from severity key to display label
    :ptype labels: dict[str, str]
    """

    order: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)


DEFAULT_SEVERITY_CONFIG = SeverityConfig(
    order=["critical", "high", "medium", "low", "informational"],
    labels={
        "critical": "Critical",
        "high": "High",
        "medium": "Medium",
        "low": "Low",
        "informational": "Informational",
    },
)


def sort_by_severity(
    findings: list[dict[str, Any]],
    config: SeverityConfig | None = None,
) -> list[dict[str, Any]]:
    """Sort findings by severity, most severe first.

    :param findings: list of finding dicts, each with a ``severity`` key
    :ptype findings: list[dict[str, Any]]
    :param config: severity configuration; uses default if *None*
    :ptype config: SeverityConfig | None
    :return: new list sorted by severity rank
    :rtype: list[dict[str, Any]]
    """
    cfg = config or DEFAULT_SEVERITY_CONFIG

    def _rank(finding: dict[str, Any]) -> int:
        """Return numeric rank for a finding's severity.

        :param finding: finding dictionary
        :ptype finding: dict[str, Any]
        :return: sort rank (lower is more severe)
        :rtype: int
        """
        sev = str(finding.get("severity", "")).lower()
        try:
            return cfg.order.index(sev)
        except ValueError:
            return len(cfg.order)

    return sorted(findings, key=_rank)


def severity_label(
    sev: str,
    config: SeverityConfig | None = None,
) -> str:
    """Return the display label for a severity key.

    :param sev: severity key (e.g. ``"critical"``)
    :ptype sev: str
    :param config: severity configuration; uses default if *None*
    :ptype config: SeverityConfig | None
    :return: human-readable label, or the raw key if not configured
    :rtype: str
    """
    cfg = config or DEFAULT_SEVERITY_CONFIG
    return cfg.labels.get(sev.lower(), sev)
