"""Tests for the report builder modules."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from threetears.agent.tools.reports.markdown_compiler import MarkdownCompiler, ReportMetadata
from threetears.agent.tools.reports.mermaid_generator import MermaidGenerator
from threetears.agent.tools.reports.pdf_renderer import PandocNotFoundError, PdfRenderer
from threetears.agent.tools.reports.severity import (
    SeverityConfig,
    severity_label,
    sort_by_severity,
)


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture
def sample_findings() -> list[dict[str, Any]]:
    """Return a small set of findings across severities."""
    return [
        {
            "title": "SQL Injection",
            "severity": "critical",
            "affected_target": "10.0.0.1",
            "affected_component": "port 80/http",
            "description": "SQL injection in login form.",
            "cvss_score": 9.8,
            "cve_ids": ["CVE-2021-1234"],
            "remediation": "Use parameterised queries.",
        },
        {
            "title": "Missing HSTS",
            "severity": "low",
            "affected_target": "10.0.0.1",
            "description": "HSTS header not set.",
            "cvss_score": 3.1,
        },
        {
            "title": "XSS Reflected",
            "severity": "high",
            "affected_target": "10.0.0.2",
            "affected_component": "port 443/https",
            "description": "Reflected XSS in search parameter.",
            "cvss_score": 7.5,
        },
    ]


@pytest.fixture
def metadata() -> ReportMetadata:
    """Return sample report metadata."""
    return ReportMetadata(
        title="Test Report",
        client_name="Acme Corp",
        target_scope="10.0.0.0/24",
        report_date=datetime(2026, 3, 18),
        assessor_name="Test Team",
    )


# -- Severity config sorting --------------------------------------------------


class TestSortBySeverity:
    """Tests for severity ordering logic."""

    def test_sort_default_config(self, sample_findings: list[dict[str, Any]]) -> None:
        """Findings are sorted critical > high > low with default config."""
        result = sort_by_severity(sample_findings)
        severities = [f["severity"] for f in result]
        assert severities == ["critical", "high", "low"]

    def test_sort_custom_config(self) -> None:
        """Custom severity config re-orders findings accordingly."""
        config = SeverityConfig(
            order=["low", "high", "critical"],
            labels={"low": "Low", "high": "High", "critical": "Critical"},
        )
        findings = [
            {"severity": "critical", "title": "A"},
            {"severity": "low", "title": "B"},
            {"severity": "high", "title": "C"},
        ]
        result = sort_by_severity(findings, config)
        assert [f["severity"] for f in result] == ["low", "high", "critical"]

    def test_unknown_severity_sorts_last(self) -> None:
        """Findings with unrecognised severity sort after known ones."""
        findings = [
            {"severity": "unknown", "title": "A"},
            {"severity": "critical", "title": "B"},
        ]
        result = sort_by_severity(findings)
        assert [f["severity"] for f in result] == ["critical", "unknown"]

    def test_severity_label_known(self) -> None:
        """Known severity key returns its display label."""
        assert severity_label("critical") == "Critical"
        assert severity_label("CRITICAL") == "Critical"

    def test_severity_label_unknown(self) -> None:
        """Unknown severity key returns the raw key."""
        assert severity_label("exotic") == "exotic"


# -- Markdown compilation with findings ----------------------------------------


class TestMarkdownCompilation:
    """Tests for MarkdownCompiler.compile."""

    def test_compile_with_findings(
        self,
        sample_findings: list[dict[str, Any]],
        metadata: ReportMetadata,
    ) -> None:
        """Compiled report contains expected sections and content."""
        report = MarkdownCompiler.compile(sample_findings, metadata)

        assert "# Test Report" in report
        assert "**Client:** Acme Corp" in report
        assert "## Executive Summary" in report
        assert "**3 findings**" in report
        assert "1 Critical" in report
        assert "## Severity Distribution" in report
        assert "```mermaid" in report
        assert "## Findings Summary" in report
        assert "| 1 |" in report
        assert "## Technical Details" in report
        assert "SQL Injection" in report
        assert "CVE-2021-1234" in report
        assert "Use parameterised queries." in report

    def test_compile_no_findings(self, metadata: ReportMetadata) -> None:
        """Report with no findings has header and no-findings summary."""
        report = MarkdownCompiler.compile([], metadata)

        assert "# Test Report" in report
        assert "identified no vulnerabilities" in report
        assert "## Findings Summary" not in report
        assert "## Technical Details" not in report

    def test_compile_with_chains(
        self,
        sample_findings: list[dict[str, Any]],
        metadata: ReportMetadata,
    ) -> None:
        """Report includes vulnerability chains section when chains provided."""
        chains = [
            {
                "name": "Auth Bypass Chain",
                "escalated_severity": "critical",
                "description": "Combined vulns escalate impact.",
                "attack_flow_mermaid": "graph TD\n    step0 --> step1",
            },
        ]
        report = MarkdownCompiler.compile(sample_findings, metadata, chains=chains)

        assert "## Vulnerability Chains" in report
        assert "Auth Bypass Chain" in report
        assert "Combined vulns escalate impact." in report

    def test_findings_sorted_in_report(
        self,
        sample_findings: list[dict[str, Any]],
        metadata: ReportMetadata,
    ) -> None:
        """Findings appear in severity order (critical first) in report."""
        report = MarkdownCompiler.compile(sample_findings, metadata)
        # In findings table, SQL Injection (critical) before XSS (high) before Missing HSTS (low)
        sql_pos = report.index("SQL Injection")
        xss_pos = report.index("XSS Reflected")
        hsts_pos = report.index("Missing HSTS")
        assert sql_pos < xss_pos < hsts_pos


# -- Mermaid pie chart ---------------------------------------------------------


class TestMermaidPieChart:
    """Tests for MermaidGenerator.severity_pie_chart."""

    def test_pie_chart_content(self, sample_findings: list[dict[str, Any]]) -> None:
        """Pie chart includes title and severity counts."""
        chart = MermaidGenerator.severity_pie_chart(sample_findings)

        assert chart.startswith("pie title Finding Severity Distribution")
        assert '"Critical" : 1' in chart
        assert '"High" : 1' in chart
        assert '"Low" : 1' in chart
        # Medium and Informational should not appear (count=0)
        assert "Medium" not in chart
        assert "Informational" not in chart

    def test_pie_chart_custom_config(self) -> None:
        """Pie chart respects custom severity config labels."""
        config = SeverityConfig(
            order=["p1", "p2"],
            labels={"p1": "Priority 1", "p2": "Priority 2"},
        )
        findings = [
            {"severity": "p1", "title": "A"},
            {"severity": "p1", "title": "B"},
            {"severity": "p2", "title": "C"},
        ]
        chart = MermaidGenerator.severity_pie_chart(findings, config)
        assert '"Priority 1" : 2' in chart
        assert '"Priority 2" : 1' in chart

    def test_network_topology(self) -> None:
        """Network topology diagram includes targets and components."""
        findings = [
            {"affected_target": "10.0.0.1", "affected_component": "port 80"},
            {"affected_target": "10.0.0.1", "affected_component": "port 443"},
            {"affected_target": "10.0.0.2"},
        ]
        topo = MermaidGenerator.network_topology(findings)
        assert "graph LR" in topo
        assert "10_0_0_1[10.0.0.1]" in topo
        assert "port 80" in topo
        assert "10_0_0_2[10.0.0.2]" in topo

    def test_network_topology_empty(self) -> None:
        """Empty findings produce a scanner-only topology."""
        topo = MermaidGenerator.network_topology([])
        assert "scanner[Scanner]" in topo


# -- PDF renderer command building ---------------------------------------------


class TestPdfRenderer:
    """Tests for PdfRenderer command building and error handling."""

    def testbuild_command_basic(self) -> None:
        """Basic command includes pandoc, output, and pdf engine."""
        renderer = PdfRenderer()
        cmd = renderer.build_command("/tmp/out.pdf", None, None)

        assert cmd[0] == "pandoc"
        assert "-o" in cmd
        assert "/tmp/out.pdf" in cmd
        assert "pdflatex" in cmd
        assert "mermaid-filter" in cmd

    def testbuild_command_with_template(self) -> None:
        """Template path is appended to command."""
        renderer = PdfRenderer()
        cmd = renderer.build_command("/tmp/out.pdf", "/tpl/report.tex", None)

        assert "--template" in cmd
        assert "/tpl/report.tex" in cmd

    def testbuild_command_with_variables(self) -> None:
        """Variables are escaped and added as -V flags."""
        renderer = PdfRenderer()
        cmd = renderer.build_command("/tmp/out.pdf", None, {"author": "John & Jane"})

        # Find the variable argument
        v_indices = [i for i, arg in enumerate(cmd) if arg == "-V"]
        var_args = [cmd[i + 1] for i in v_indices if cmd[i + 1].startswith("author=")]
        assert len(var_args) == 1
        # & should be escaped
        assert r"\&" in var_args[0]

    def test_render_calls_subprocess(self) -> None:
        """Render invokes subprocess.run with correct command."""
        renderer = PdfRenderer()
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("threetears.agent.tools.reports.pdf_renderer.subprocess.run", return_value=mock_result) as mock_run:
            renderer.render("# Hello", "/tmp/out.pdf")

        mock_run.assert_called_once()
        args = mock_run.call_args
        cmd = args[0][0]
        assert cmd[0] == "pandoc"
        assert "/tmp/out.pdf" in cmd

    def test_render_raises_on_missing_pandoc(self) -> None:
        """Render raises PandocNotFoundError when pandoc is missing."""
        renderer = PdfRenderer()

        with patch(
            "threetears.agent.tools.reports.pdf_renderer.subprocess.run",
            side_effect=FileNotFoundError("No such file"),
        ):
            with pytest.raises(PandocNotFoundError, match="Pandoc is not installed"):
                renderer.render("# Hello", "/tmp/out.pdf")

    def test_render_raises_on_nonzero_exit(self) -> None:
        """Render raises RuntimeError when pandoc fails."""
        renderer = PdfRenderer()
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"something went wrong"

        with patch(
            "threetears.agent.tools.reports.pdf_renderer.subprocess.run",
            return_value=mock_result,
        ):
            with pytest.raises(RuntimeError, match="Pandoc failed"):
                renderer.render("# Hello", "/tmp/out.pdf")
