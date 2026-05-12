"""Report builder modules for generating structured reports.

Provides configurable severity ordering, Mermaid diagram generation,
Markdown compilation, and PDF rendering via Pandoc.
"""

from __future__ import annotations

from threetears.agent.tools.reports.markdown_compiler import MarkdownCompiler, ReportMetadata
from threetears.agent.tools.reports.mermaid_generator import MermaidGenerator
from threetears.agent.tools.reports.pdf_renderer import PandocNotFoundError, PdfRenderer
from threetears.agent.tools.reports.severity import DEFAULT_SEVERITY_CONFIG, SeverityConfig

__all__ = [
    "DEFAULT_SEVERITY_CONFIG",
    "MarkdownCompiler",
    "MermaidGenerator",
    "PandocNotFoundError",
    "PdfRenderer",
    "ReportMetadata",
    "SeverityConfig",
]
