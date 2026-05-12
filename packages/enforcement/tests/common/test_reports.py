"""tests for ``reports`` module."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common.exemptions import Exemption
from threetears.enforcement.common.reports import emit_report
from threetears.enforcement.common.violations import Violation


def _make_violation(category: str, file: Path, line: int, symbol: str) -> Violation:
    return Violation(
        category=category,
        file=file,
        line=line,
        symbol=symbol,
        reason=f"reason for {symbol}",
    )


class TestEmitReport:
    def test_empty_violations(self, tmp_path: Path) -> None:
        repo = tmp_path
        report = emit_report(
            violations=[],
            src_roots=(repo / "src",),
            exemptions=[],
            mode="strict",
            repo_root=repo,
            domain="cache",
        )
        assert "domain: cache" in report
        assert "violations_total: 0" in report
        assert "exemptions_loaded: 0" in report
        assert "mode: strict" in report

    def test_header_includes_all_inputs(self, tmp_path: Path) -> None:
        repo = tmp_path
        report = emit_report(
            violations=[],
            src_roots=(repo / "src", repo / "packages" / "core" / "src"),
            exemptions=[
                Exemption(file="a.py", line=1, symbol="_x", rationale="r"),
                Exemption(file="b.py", line=2, symbol="_y", rationale="r"),
            ],
            mode="report",
            repo_root=repo,
            domain="underscore_access",
        )
        assert f"repo_root: {repo}" in report
        assert "domain: underscore_access" in report
        assert "mode: report" in report
        assert "exemptions_loaded: 2" in report
        # both src roots appear in the rendered ``src_roots:`` line
        assert str(repo / "src") in report
        assert str(repo / "packages" / "core" / "src") in report

    def test_violations_sorted_and_categorised(self, tmp_path: Path) -> None:
        repo = tmp_path
        f1 = repo / "src" / "a.py"
        f2 = repo / "src" / "b.py"
        f1.parent.mkdir(parents=True)
        f1.write_text("")
        f2.write_text("")
        violations = [
            _make_violation("zeta", f2, 30, "z"),
            _make_violation("alpha", f1, 10, "a"),
            _make_violation("alpha", f1, 5, "b"),
            _make_violation("beta", f2, 1, "x"),
        ]
        report = emit_report(
            violations=violations,
            src_roots=(repo / "src",),
            exemptions=[],
            mode="strict",
            repo_root=repo,
            domain="d",
        )
        # category breakdown lines, sorted
        assert "  alpha: 2" in report
        assert "  beta: 1" in report
        assert "  zeta: 1" in report
        # ordering: alpha categories appear before beta
        alpha_idx = report.index("[alpha] ")
        beta_idx = report.index("[beta] ")
        zeta_idx = report.index("[zeta] ")
        assert alpha_idx < beta_idx < zeta_idx
        # within alpha: sorted by file then line then symbol; both same file,
        # so by line: 5 then 10
        first_alpha_line = report[alpha_idx:beta_idx]
        assert first_alpha_line.index(":5:") < first_alpha_line.index(":10:")

    def test_violations_total_and_per_category_counts(self, tmp_path: Path) -> None:
        repo = tmp_path
        f = repo / "x.py"
        f.write_text("")
        violations = [
            _make_violation("a", f, 1, "x"),
            _make_violation("a", f, 2, "y"),
            _make_violation("b", f, 1, "z"),
        ]
        report = emit_report(
            violations=violations,
            src_roots=(),
            exemptions=[],
            mode="strict",
            repo_root=repo,
            domain="d",
        )
        assert "violations_total: 3" in report
        assert "  a: 2" in report
        assert "  b: 1" in report

    def test_renders_each_violation_via_format(self, tmp_path: Path) -> None:
        repo = tmp_path
        f = repo / "src" / "thing.py"
        f.parent.mkdir(parents=True)
        f.write_text("")
        v = Violation(
            category="domain.cat",
            file=f,
            line=42,
            symbol="Sym",
            reason="explanation",
        )
        report = emit_report(
            violations=[v],
            src_roots=(),
            exemptions=[],
            mode="strict",
            repo_root=repo,
            domain="domain",
        )
        assert "[domain.cat] src/thing.py:42:Sym  -- explanation" in report
