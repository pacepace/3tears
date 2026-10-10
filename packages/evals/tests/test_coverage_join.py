"""The report joins the coverage map to the findings and the next steps, on stored lever names (#630, #631).

- **A gap is shown beside the step that would measure it** (#630). Each ``thin`` / ``unswept`` lever sits beside
  the next steps whose ``lever`` names it; a gap no step names says so; a step naming a lever with no coverage
  row still renders, unattached.
- **A lever no finding names is shown, and counted** (#631). The coverage table states each lever's findings
  or "no finding", and the methods section counts the levers no finding names. Nothing refuses or acts on it.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.analysis.report import DisclosureBlock, Report, TableBlock, TextBlock, build_report
from threetears.evals.kernel.authored import NO_CHART, AuthoredAnalysis, Finding, NextStep
from threetears.evals.kernel.campaign import CoverageLens, LeverCoverage
from packages.evals.tests.factories import make_analysis


def _lever(name: str, status: str) -> LeverCoverage:
    return LeverCoverage(name=name, cells=1 if status == "unswept" else 2, k=3, n=6, dispersion="±0.01", status=status)


def _finding(title: str, axes: list[str]) -> Finding:
    return Finding.model_validate(
        {
            "title": title,
            "body": "b",
            "confidence": "medium",
            "axes": axes,
            "evidence": [],
            "chart": {"type": NO_CHART, "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
            "caveats": [],
            "invalidates": [],
            "durable": "",
        }
    )


def _step(title: str, lever: str) -> NextStep:
    return NextStep(title=title, why="", leverage="high", lever=lever)


def _report() -> Report:
    """Four levers: one measured and found, one measured with no finding, one unswept with a step, one without."""
    analysis = make_analysis(
        coverage=CoverageLens(
            levers=[
                _lever("chunk_tokens", "measured"),
                _lever("retriever_top_k", "measured"),
                _lever("extraction_schema", "unswept"),
                _lever("ocr_engine_version", "thin"),
            ]
        ),
        document=AuthoredAnalysis(
            headline="h",
            summary="",
            findings=[_finding("wider chunks extract more", ["chunk_tokens"])],
            decisions=[],
            questions=[],
            next=[
                _step("sweep the schema", "extraction_schema"),
                _step("try a reranker", "reranker"),
            ],
        ),
    )
    return build_report(analysis)


def _coverage(report: Report) -> dict[str, dict[str, Any]]:
    (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "coverage"]
    return {row["lever"]: row for row in table.rows}


def test_an_unswept_lever_a_step_names_sits_beside_that_step_and_one_no_step_names_says_so() -> None:
    rows = _coverage(_report())

    assert rows["extraction_schema"]["status"] == "unswept"
    assert rows["extraction_schema"]["next"] == "sweep the schema"
    assert rows["ocr_engine_version"]["next"] == "no next step names it"
    assert rows["chunk_tokens"]["next"] is None, "a measured lever no step names is no gap"


def test_a_step_whose_lever_has_no_coverage_row_still_renders_unattached() -> None:
    report = _report()

    steps = {block.body: block for block in report.blocks if isinstance(block, TextBlock) and block.role == "next_step"}

    assert "reranker" not in _coverage(report)
    assert [fact.value for fact in steps["try a reranker"].facts if fact.name == "Lever"] == [
        "reranker (no coverage row)"
    ]
    assert [fact.value for fact in steps["sweep the schema"].facts if fact.name == "Lever"] == [
        "extraction_schema (unswept)"
    ]


def test_a_lever_no_finding_names_reads_no_finding_and_is_counted_in_the_methods() -> None:
    report = _report()

    rows = _coverage(report)
    methods = [
        block.text for block in report.blocks if isinstance(block, DisclosureBlock) and block.section == "methods"
    ]

    assert rows["chunk_tokens"]["findings"] == "1"
    assert {lever for lever, row in rows.items() if row["findings"] == "no finding"} == {
        "retriever_top_k",
        "extraction_schema",
        "ocr_engine_version",
    }
    assert (
        "3 of the 4 lever(s) in the coverage map are named by no finding: retriever_top_k, extraction_schema, "
        "ocr_engine_version."
    ) in methods
