"""Mermaid diagram generator for structured reports.

Generates Mermaid-syntax diagrams for severity distribution,
network topology, and attack flow visualization using a
configurable severity system.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from threetears.agent.tools.reports.severity import (
    DEFAULT_SEVERITY_CONFIG,
    SeverityConfig,
    severity_label,
    sort_by_severity,
)


class MermaidGenerator:
    """Static methods producing Mermaid diagram blocks.

    Each method returns a string containing valid Mermaid syntax
    ready for embedding in Markdown fenced code blocks.
    """

    @staticmethod
    def severity_pie_chart(
        findings: list[dict[str, Any]],
        severity_config: SeverityConfig | None = None,
    ) -> str:
        """Generate a Mermaid pie chart of finding severity distribution.

        :param findings: list of finding dictionaries with ``severity`` key
        :ptype findings: list[dict[str, Any]]
        :param severity_config: severity configuration; uses default if *None*
        :ptype severity_config: SeverityConfig | None
        :return: Mermaid pie chart syntax
        :rtype: str
        """
        cfg = severity_config or DEFAULT_SEVERITY_CONFIG
        counts: Counter[str] = Counter(str(f["severity"]).lower() for f in findings)
        lines = ["pie title Finding Severity Distribution"]
        for sev in cfg.order:
            count = counts.get(sev, 0)
            if count > 0:
                label = severity_label(sev, cfg)
                lines.append(f'    "{label}" : {count}')
        return "\n".join(lines)

    @staticmethod
    def network_topology(findings: list[dict[str, Any]]) -> str:
        """Generate a Mermaid graph of network topology from findings.

        Groups findings by ``affected_target``, showing affected
        components (ports/services) as connected nodes.

        :param findings: list of finding dictionaries
        :ptype findings: list[dict[str, Any]]
        :return: Mermaid graph syntax
        :rtype: str
        """
        lines = ["graph LR"]
        if not findings:
            lines.append("    scanner[Scanner]")
            return "\n".join(lines)

        targets: dict[str, list[str]] = {}
        for f in findings:
            target = f["affected_target"]
            component = f.get("affected_component")
            if target not in targets:
                targets[target] = []
            if component and component not in targets[target]:
                targets[target].append(component)

        for target, components in targets.items():
            safe_target = target.replace(".", "_").replace("/", "_").replace(":", "_")
            lines.append(f"    scanner[Scanner] --> {safe_target}[{target}]")
            for comp in components:
                safe_comp = f"{safe_target}_{comp.replace('/', '_').replace(':', '_')}"
                lines.append(f"    {safe_target} --> {safe_comp}[{comp}]")

        return "\n".join(lines)

    @staticmethod
    def attack_flow(
        findings: list[dict[str, Any]],
        severity_config: SeverityConfig | None = None,
    ) -> str:
        """Generate a Mermaid graph showing attack flow ordered by severity.

        Findings are displayed as a directed graph from highest
        to lowest severity, representing the logical progression
        of discovered vulnerabilities.

        :param findings: list of finding dictionaries
        :ptype findings: list[dict[str, Any]]
        :param severity_config: severity configuration; uses default if *None*
        :ptype severity_config: SeverityConfig | None
        :return: Mermaid graph syntax
        :rtype: str
        """
        cfg = severity_config or DEFAULT_SEVERITY_CONFIG
        lines = ["graph TD"]
        if not findings:
            lines.append("    start[No findings]")
            return "\n".join(lines)

        sorted_findings = sort_by_severity(findings, cfg)

        prev_node = None
        for i, f in enumerate(sorted_findings):
            node_id = f"f{i}"
            title = f["title"]
            sev = str(f["severity"]).lower()
            sev_lbl = severity_label(sev, cfg)
            label = f"{title}\\n[{sev_lbl}]"
            lines.append(f'    {node_id}["{label}"]')
            if prev_node is not None:
                lines.append(f"    {prev_node} --> {node_id}")
            prev_node = node_id

        return "\n".join(lines)

    @staticmethod
    def chain_attack_flow(
        chain_name: str,
        findings: list[dict[str, Any]],
        severity_escalation: str,
    ) -> str:
        """Generate Mermaid attack flow diagram for a vulnerability chain.

        Produces a ``graph TD`` diagram showing each finding as a node
        connected in ``step_order``, with a final Impact node displaying
        the escalated severity.

        :param chain_name: name of the vulnerability chain
        :ptype chain_name: str
        :param findings: list of finding dicts with title, severity, step_order
        :ptype findings: list[dict[str, Any]]
        :param severity_escalation: the escalated severity for the chain
        :ptype severity_escalation: str
        :return: Mermaid graph syntax
        :rtype: str
        """
        _severity_colors: dict[str, str] = {
            "critical": "#ff0000",
            "high": "#ff4444",
            "medium": "#ff8800",
            "low": "#ffcc00",
            "informational": "#4488ff",
        }

        lines = ["graph TD"]
        if not findings:
            lines.append(f'    empty["{chain_name}: No findings"]')
            return "\n".join(lines)

        sorted_findings = sorted(findings, key=lambda f: f.get("step_order", 0))

        style_defs: list[str] = []
        prev_node = None
        for i, f in enumerate(sorted_findings):
            node_id = f"step{i}"
            title = f["title"]
            sev = str(f["severity"]).lower()
            label = f"{title}\\n[{sev}]"
            lines.append(f'    {node_id}["{label}"]')
            color = _severity_colors.get(sev)
            if color:
                style_defs.append(f"    style {node_id} fill:{color},color:#fff")
            if prev_node is not None:
                lines.append(f"    {prev_node} --> {node_id}")
            prev_node = node_id

        impact_node = "impact"
        esc_label = str(severity_escalation).lower()
        lines.append(f'    {prev_node} --> {impact_node}["Impact: {severity_escalation}"]')
        esc_color = _severity_colors.get(esc_label)
        if esc_color:
            style_defs.append(f"    style {impact_node} fill:{esc_color},color:#fff")

        lines.extend(style_defs)
        return "\n".join(lines)
