"""Markdown report compiler for structured findings.

Compiles findings and metadata into a structured Markdown report
with executive summary, findings table, technical details,
remediation guidance, and Mermaid diagrams using a configurable
severity system.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from threetears.agent.tools.reports.mermaid_generator import MermaidGenerator
from threetears.agent.tools.reports.severity import (
    DEFAULT_SEVERITY_CONFIG,
    SeverityConfig,
    severity_label,
    sort_by_severity,
)

__all__ = [
    "MarkdownCompiler",
    "ReportMetadata",
]


@dataclass
class ReportMetadata:
    """Metadata for a report.

    :param title: report title
    :ptype title: str
    :param client_name: client organization name
    :ptype client_name: str
    :param target_scope: description of the target scope
    :ptype target_scope: str
    :param report_date: date of the report
    :ptype report_date: datetime
    :param assessor_name: name of the assessor or team
    :ptype assessor_name: str
    """

    title: str
    client_name: str
    target_scope: str
    report_date: datetime
    assessor_name: str = "Security Team"


class MarkdownCompiler:
    """Compiles findings and metadata into a structured Markdown report.

    All methods are static -- the compiler is stateless. Findings
    are sorted by severity (most severe first) throughout the report.
    """

    @staticmethod
    def compile(
        findings: list[dict[str, Any]],
        metadata: ReportMetadata,
        chains: list[dict[str, Any]] | None = None,
        severity_config: SeverityConfig | None = None,
    ) -> str:
        """Compile findings into a complete Markdown report.

        :param findings: list of finding dictionaries
        :ptype findings: list[dict[str, Any]]
        :param metadata: report metadata
        :ptype metadata: ReportMetadata
        :param chains: detected vulnerability chains with Mermaid diagrams
        :ptype chains: list[dict[str, Any]] | None
        :param severity_config: severity configuration; uses default if *None*
        :ptype severity_config: SeverityConfig | None
        :return: complete Markdown report
        :rtype: str
        """
        cfg = severity_config or DEFAULT_SEVERITY_CONFIG
        chains = chains or []
        sorted_findings = sort_by_severity(findings, cfg)
        sections = [
            MarkdownCompiler._header(metadata),
            MarkdownCompiler._executive_summary(sorted_findings, metadata, cfg),
            MarkdownCompiler._severity_chart(sorted_findings, cfg),
            MarkdownCompiler._findings_table(sorted_findings, cfg),
            MarkdownCompiler._technical_details(sorted_findings, cfg),
        ]
        if chains:
            sections.append(MarkdownCompiler._chain_section(chains))
        # Filter empty sections before joining
        return "\n\n".join(s for s in sections if s)

    @staticmethod
    def _header(metadata: ReportMetadata) -> str:
        """Generate the report header section.

        :param metadata: report metadata
        :ptype metadata: ReportMetadata
        :return: Markdown header
        :rtype: str
        """
        date_str = metadata.report_date.strftime("%Y-%m-%d")
        return (
            f"# {metadata.title}\n\n"
            f"**Client:** {metadata.client_name}\n\n"
            f"**Target Scope:** {metadata.target_scope}\n\n"
            f"**Date:** {date_str}\n\n"
            f"**Assessor:** {metadata.assessor_name}"
        )

    @staticmethod
    def _executive_summary(
        findings: list[dict[str, Any]],
        metadata: ReportMetadata,
        config: SeverityConfig,
    ) -> str:
        """Generate the executive summary section.

        :param findings: sorted findings
        :ptype findings: list[dict[str, Any]]
        :param metadata: report metadata
        :ptype metadata: ReportMetadata
        :param config: severity configuration
        :ptype config: SeverityConfig
        :return: Markdown executive summary
        :rtype: str
        """
        lines = ["## Executive Summary"]

        if not findings:
            lines.append(
                f"\nThe assessment of {metadata.target_scope} for {metadata.client_name} "
                "identified no vulnerabilities. No findings were recorded during this engagement."
            )
            return "\n".join(lines)

        counts: Counter[str] = Counter(str(f["severity"]).lower() for f in findings)
        summary_parts = []
        for sev in config.order:
            count = counts.get(sev, 0)
            if count > 0:
                label = severity_label(sev, config)
                summary_parts.append(f"{count} {label}")

        total = len(findings)
        lines.append(
            f"\nThe assessment of {metadata.target_scope} for {metadata.client_name} "
            f"identified **{total} findings**: {', '.join(summary_parts)}."
        )

        return "\n".join(lines)

    @staticmethod
    def _severity_chart(
        findings: list[dict[str, Any]],
        config: SeverityConfig,
    ) -> str:
        """Generate the Mermaid severity distribution chart.

        :param findings: sorted findings
        :ptype findings: list[dict[str, Any]]
        :param config: severity configuration
        :ptype config: SeverityConfig
        :return: Markdown with Mermaid chart
        :rtype: str
        """
        if not findings:
            return ""
        chart = MermaidGenerator.severity_pie_chart(findings, config)
        return f"## Severity Distribution\n\n```mermaid\n{chart}\n```"

    @staticmethod
    def _findings_table(
        findings: list[dict[str, Any]],
        config: SeverityConfig,
    ) -> str:
        """Generate the findings summary table.

        :param findings: sorted findings
        :ptype findings: list[dict[str, Any]]
        :param config: severity configuration
        :ptype config: SeverityConfig
        :return: Markdown table
        :rtype: str
        """
        if not findings:
            return ""
        lines = [
            "## Findings Summary",
            "",
            "| # | Title | Severity | Target | CVSS |",
            "|---|-------|----------|--------|------|",
        ]
        for i, f in enumerate(findings, 1):
            sev = severity_label(str(f["severity"]).lower(), config)
            cvss = f.get("cvss_score")
            cvss_str = str(cvss) if cvss is not None else "N/A"
            target = f["affected_target"]
            lines.append(f"| {i} | {f['title']} | {sev} | {target} | {cvss_str} |")
        return "\n".join(lines)

    @staticmethod
    def _technical_details(
        findings: list[dict[str, Any]],
        config: SeverityConfig,
    ) -> str:
        """Generate technical detail sections for each finding.

        :param findings: sorted findings
        :ptype findings: list[dict[str, Any]]
        :param config: severity configuration
        :ptype config: SeverityConfig
        :return: Markdown technical details
        :rtype: str
        """
        if not findings:
            return ""
        sections = ["## Technical Details"]
        for i, f in enumerate(findings, 1):
            sev = severity_label(str(f["severity"]).lower(), config)
            section = [f"### {i}. {f['title']}"]
            section.append(f"\n**Severity:** {sev}")

            cvss = f.get("cvss_score")
            if cvss is not None:
                section.append(f"\n**CVSS Score:** {cvss}")

            section.append(f"\n**Target:** {f['affected_target']}")
            if f.get("affected_component"):
                section.append(f" ({f['affected_component']})")

            cve_ids = f.get("cve_ids", [])
            if cve_ids:
                section.append(f"\n**CVE IDs:** {', '.join(cve_ids)}")

            section.append(f"\n#### Description\n\n{f['description']}")

            remediation = f.get("remediation")
            if remediation:
                section.append(f"\n#### Remediation\n\n{remediation}")

            sections.append("\n".join(section))
        return "\n\n".join(sections)

    @staticmethod
    def _chain_section(chains: list[dict[str, Any]]) -> str:
        """Generate the vulnerability chains section with Mermaid diagrams.

        :param chains: list of chain dicts with name, escalated_severity,
            description, and attack_flow_mermaid fields
        :ptype chains: list[dict[str, Any]]
        :return: Markdown vulnerability chains section
        :rtype: str
        """
        if not chains:
            return ""
        lines = ["## Vulnerability Chains"]
        for i, chain in enumerate(chains, 1):
            name = chain.get("name", "Unknown Chain")
            severity = chain.get("escalated_severity", "unknown")
            description = chain.get("description", "")
            lines.append(f"\n### Chain {i}: {name}")
            lines.append(f"\n**Escalated Severity:** {severity}")
            if description:
                lines.append(f"\n{description}")
            mermaid = chain.get("attack_flow_mermaid", "")
            if mermaid:
                lines.append(f"\n```mermaid\n{mermaid}\n```")
        return "\n".join(lines)
