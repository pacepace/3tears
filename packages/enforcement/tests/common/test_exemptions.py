"""tests for ``exemptions`` module."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.common.exemptions import (
    Exemption,
    ExemptionError,
    apply_exemptions,
    parse_exemptions_with_rationale,
)
from threetears.enforcement.common.violations import Violation


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


VALID_RATIONALE = "framework-stable internal field tests legitimately read"


class TestParseWellFormed:
    def test_simple_entry(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "src/pkg/mod.py:42:_helper\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert len(entries) == 1
        e = entries[0]
        assert e.file == "src/pkg/mod.py"
        assert e.line == 42
        assert e.symbol == "_helper"
        assert e.rationale == VALID_RATIONALE

    def test_multiple_entries(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "a.py:1:_a\n"
            f"# rationale: {VALID_RATIONALE} two\n"
            "b.py:2:_b\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert [e.symbol for e in entries] == ["_a", "_b"]

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            "\n\n"
            f"# rationale: {VALID_RATIONALE}\n"
            "\n"
            "a.py:1:_a\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert len(entries) == 1

    def test_non_rationale_comments_pass_through(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            "# header note about this file\n"
            "# another comment line\n"
            f"# rationale: {VALID_RATIONALE}\n"
            "a.py:1:_a\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert len(entries) == 1

    def test_star_line_means_zero(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "any/file.py:*:_helper\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert entries[0].line == 0


class TestParseRejects:
    def test_missing_rationale_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "ex.txt", "a.py:1:_x\n")
        with pytest.raises(ExemptionError, match="no preceding"):
            parse_exemptions_with_rationale(path)

    def test_empty_rationale_raises(self, tmp_path: Path) -> None:
        path = _write(tmp_path / "ex.txt", "# rationale: \na.py:1:_x\n")
        with pytest.raises(ExemptionError, match="non-empty reason"):
            parse_exemptions_with_rationale(path)

    def test_too_short_rationale_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            "# rationale: short\n"
            "a.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="at least"):
            parse_exemptions_with_rationale(path)

    def test_blanket_internal_access_rejected(self, tmp_path: Path) -> None:
        # 30+ chars but starts with the blanket phrase "internal access"
        rationale = "internal access for the helper function only"
        assert len(rationale) >= 30
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {rationale}\n"
            "a.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="blanket phrase"):
            parse_exemptions_with_rationale(path)

    def test_blanket_tests_need_this_rejected(self, tmp_path: Path) -> None:
        # 30+ chars; the "tests need this" prefix triggers the blanket check
        rationale = "tests need this for mocking out the receiver"
        assert len(rationale) >= 30
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {rationale}\n"
            "a.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="blanket phrase"):
            parse_exemptions_with_rationale(path)

    def test_malformed_entry_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "not-a-valid-entry\n",
        )
        with pytest.raises(ExemptionError, match="malformed"):
            parse_exemptions_with_rationale(path)

    def test_non_int_line_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "a.py:notnumber:_x\n",
        )
        with pytest.raises(ExemptionError, match="must be an integer"):
            parse_exemptions_with_rationale(path)

    def test_zero_line_explicitly_rejected(self, tmp_path: Path) -> None:
        # the ``*`` form is the canonical "any line"; literal 0 must be rejected
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\n"
            "a.py:0:_x\n",
        )
        with pytest.raises(ExemptionError, match="must be positive"):
            parse_exemptions_with_rationale(path)

    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_exemptions_with_rationale(tmp_path / "missing.txt")


class TestApplyExemptions:
    def test_filters_matching(self, tmp_path: Path) -> None:
        repo = tmp_path
        file_path = repo / "src" / "a.py"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("")
        violations = [
            Violation(
                category="x.y",
                file=file_path,
                line=10,
                symbol="_helper",
                reason="r",
            ),
            Violation(
                category="x.y",
                file=file_path,
                line=20,
                symbol="_other",
                reason="r2",
            ),
        ]
        exemptions = [
            Exemption(file="src/a.py", line=10, symbol="_helper", rationale="r"),
        ]
        result = apply_exemptions(violations, exemptions, repo)
        assert len(result) == 1
        assert result[0].symbol == "_other"

    def test_preserves_order(self, tmp_path: Path) -> None:
        repo = tmp_path
        f = repo / "a.py"
        f.write_text("")
        violations = [
            Violation(category="c", file=f, line=i, symbol=f"_v{i}", reason="r")
            for i in range(5)
        ]
        result = apply_exemptions(violations, [], repo)
        assert [v.line for v in result] == [0, 1, 2, 3, 4]

    def test_line_zero_matches_any_line(self, tmp_path: Path) -> None:
        repo = tmp_path
        file_path = repo / "src" / "a.py"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("")
        violations = [
            Violation(category="x", file=file_path, line=10, symbol="_h", reason="r"),
            Violation(category="x", file=file_path, line=99, symbol="_h", reason="r"),
            Violation(
                category="x",
                file=file_path,
                line=10,
                symbol="_other",
                reason="r",
            ),
        ]
        exemptions = [
            Exemption(file="src/a.py", line=0, symbol="_h", rationale="r"),
        ]
        result = apply_exemptions(violations, exemptions, repo)
        assert len(result) == 1
        assert result[0].symbol == "_other"

    def test_line_zero_does_not_match_other_files(self, tmp_path: Path) -> None:
        repo = tmp_path
        file_a = repo / "a.py"
        file_b = repo / "b.py"
        file_a.write_text("")
        file_b.write_text("")
        violations = [
            Violation(category="x", file=file_a, line=10, symbol="_h", reason="r"),
            Violation(category="x", file=file_b, line=10, symbol="_h", reason="r"),
        ]
        exemptions = [
            Exemption(file="a.py", line=0, symbol="_h", rationale="r"),
        ]
        result = apply_exemptions(violations, exemptions, repo)
        assert len(result) == 1
        assert result[0].file == file_b
