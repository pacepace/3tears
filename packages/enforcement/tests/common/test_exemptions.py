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
            f"# rationale: {VALID_RATIONALE}\nsrc/pkg/mod.py:42:_helper\n",
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
            f"# rationale: {VALID_RATIONALE}\na.py:1:_a\n# rationale: {VALID_RATIONALE} two\nb.py:2:_b\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert [e.symbol for e in entries] == ["_a", "_b"]

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"\n\n# rationale: {VALID_RATIONALE}\n\na.py:1:_a\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert len(entries) == 1

    def test_non_rationale_comments_pass_through(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# header note about this file\n# another comment line\n# rationale: {VALID_RATIONALE}\na.py:1:_a\n",
        )
        entries = parse_exemptions_with_rationale(path)
        assert len(entries) == 1

    def test_star_line_means_zero(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\nany/file.py:*:_helper\n",
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
            "# rationale: short\na.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="at least"):
            parse_exemptions_with_rationale(path)

    def test_blanket_internal_access_rejected(self, tmp_path: Path) -> None:
        # 30+ chars but starts with the blanket phrase "internal access"
        rationale = "internal access for the helper function only"
        assert len(rationale) >= 30
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {rationale}\na.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="blanket phrase"):
            parse_exemptions_with_rationale(path)

    def test_blanket_tests_need_this_rejected(self, tmp_path: Path) -> None:
        # 30+ chars; the "tests need this" prefix triggers the blanket check
        rationale = "tests need this for mocking out the receiver"
        assert len(rationale) >= 30
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {rationale}\na.py:1:_x\n",
        )
        with pytest.raises(ExemptionError, match="blanket phrase"):
            parse_exemptions_with_rationale(path)

    def test_malformed_entry_raises(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\nnot-a-valid-entry\n",
        )
        with pytest.raises(ExemptionError, match="malformed"):
            parse_exemptions_with_rationale(path)

    def test_a_malformed_occurrence_suffix_raises(self, tmp_path: Path) -> None:
        """A non-integer middle field is a SCOPE key now, but ``#`` still means an ordinal.

        Replaces an earlier test that required any non-integer field to raise. That
        contract was retired deliberately when the scope form was added; what survives
        of it is this: a field that reaches for the ordinal syntax and gets it wrong is
        a typo, not a scope named ``thing#x``.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\na.py:only#x:_x\n",
        )
        with pytest.raises(ExemptionError, match="malformed scope key"):
            parse_exemptions_with_rationale(path, allow_scope_keys=True)

    def test_zero_line_explicitly_rejected(self, tmp_path: Path) -> None:
        # the ``*`` form is the canonical "any line"; literal 0 must be rejected
        path = _write(
            tmp_path / "ex.txt",
            f"# rationale: {VALID_RATIONALE}\na.py:0:_x\n",
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
        violations = [Violation(category="c", file=f, line=i, symbol=f"_v{i}", reason="r") for i in range(5)]
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


class TestTheScopeKeyForm:
    """The third key form: `path:qualname[#N]:symbol`.

    A line number describes the file's layout, not the thing exempted, so an edit
    anywhere above an entry stops it matching and the domain reports a violation
    against code nobody touched. A scope key survives that edit.

    Added additively: the line and `*` forms parse and match exactly as before, which
    the rest of this module still pins.
    """

    def test_a_bare_scope_parses_with_no_occurrence(self, tmp_path: Path) -> None:
        """No ordinal means "any occurrence in this scope".

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:TestThing.test_it:_x\n")

        entry = parse_exemptions_with_rationale(path, allow_scope_keys=True)[0]

        assert (entry.scope, entry.occurrence, entry.line) == ("TestThing.test_it", None, 0)

    def test_an_occurrence_suffix_parses(self, tmp_path: Path) -> None:
        """`#N` distinguishes two accesses of one symbol in one scope.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:only#2:_x\n")

        entry = parse_exemptions_with_rationale(path, allow_scope_keys=True)[0]

        assert (entry.scope, entry.occurrence) == ("only", 2)

    def test_a_module_level_scope_parses(self, tmp_path: Path) -> None:
        """`<module>` is a scope like any other, and its angle brackets are not special.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:<module>:_x\n")

        assert parse_exemptions_with_rationale(path, allow_scope_keys=True)[0].scope == "<module>"

    def test_a_scope_key_is_not_treated_as_file_wide(self, tmp_path: Path) -> None:
        """The hazard this form introduces, pinned.

        A scope-keyed entry carries line 0 because it names no line, which is the
        same 0 the `*` form uses. Bucketing on the line first would exempt the symbol
        EVERYWHERE in the file: silently wider than what was written, in a mechanism
        whose whole job is to be narrow.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        exemptions = [Exemption(file="a.py", line=0, symbol="_x", rationale="r", scope="wanted", occurrence=None)]
        inside = Violation(category="d.c", file=tmp_path / "a.py", line=2, symbol="_x", reason="r")
        elsewhere = Violation(category="d.c", file=tmp_path / "a.py", line=9, symbol="_x", reason="r")

        def _scope_of(_path: Path, line: int) -> str:
            return "wanted" if line == 2 else "other"

        kept = apply_exemptions([inside, elsewhere], exemptions, tmp_path, scope_of=_scope_of)

        assert [v.line for v in kept] == [9], "a scope key exempted a symbol outside its scope"

    def test_a_scope_key_is_inert_without_a_resolver(self, tmp_path: Path) -> None:
        """A domain that does not opt in cannot be silently changed by this feature.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        exemptions = [Exemption(file="a.py", line=0, symbol="_x", rationale="r", scope="only", occurrence=None)]
        violation = Violation(category="d.c", file=tmp_path / "a.py", line=2, symbol="_x", reason="r")

        assert apply_exemptions([violation], exemptions, tmp_path) == [violation]

    def test_an_occurrence_exempts_only_its_own_access(self, tmp_path: Path) -> None:
        """Two accesses in one scope keep their own entries, and their own reasons.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        exemptions = [Exemption(file="a.py", line=0, symbol="_x", rationale="r", scope="only", occurrence=1)]
        first = Violation(category="d.c", file=tmp_path / "a.py", line=2, symbol="_x", reason="r")
        second = Violation(category="d.c", file=tmp_path / "a.py", line=3, symbol="_x", reason="r")

        kept = apply_exemptions([first, second], exemptions, tmp_path, scope_of=lambda _p, _ln: "only")

        assert [v.line for v in kept] == [2], "the wrong occurrence was exempted"

    def test_an_entry_without_an_ordinal_covers_every_occurrence(self, tmp_path: Path) -> None:
        """The convenience half of the same rule.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        exemptions = [Exemption(file="a.py", line=0, symbol="_x", rationale="r", scope="only", occurrence=None)]
        first = Violation(category="d.c", file=tmp_path / "a.py", line=2, symbol="_x", reason="r")
        second = Violation(category="d.c", file=tmp_path / "a.py", line=3, symbol="_x", reason="r")

        assert apply_exemptions([first, second], exemptions, tmp_path, scope_of=lambda _p, _ln: "only") == []

    def test_the_key_survives_an_edit_above_it(self, tmp_path: Path) -> None:
        """The whole point, stated as the thing a line key cannot do.

        The access moves from line 2 to line 6 because six lines were inserted above
        it. A line-keyed entry stops matching; the scope key does not move.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        exemptions = [Exemption(file="a.py", line=0, symbol="_x", rationale="r", scope="only", occurrence=0)]
        before = Violation(category="d.c", file=tmp_path / "a.py", line=2, symbol="_x", reason="r")
        after = Violation(category="d.c", file=tmp_path / "a.py", line=8, symbol="_x", reason="r")

        def _scope_of(_path: Path, _line: int) -> str:
            return "only"

        assert apply_exemptions([before], exemptions, tmp_path, scope_of=_scope_of) == []
        assert apply_exemptions([after], exemptions, tmp_path, scope_of=_scope_of) == []


class TestScopeKeysAreOptIn:
    """A line-keyed domain must keep getting an error for a typo in its key field.

    Every other domain sharing this parser is line-keyed. Accepting
    any non-numeric field as a scope everywhere would turn `a.py:1O2:_x` -- an O for a
    zero -- into an inert exemption that matches nothing, silently, instead of an error
    naming the line. The first version of this change did exactly that.
    """

    def test_a_typo_still_raises_for_a_domain_that_did_not_opt_in(self, tmp_path: Path) -> None:
        """The default, which is what every other domain gets.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:1O2:_x\n")

        with pytest.raises(ExemptionError, match="must be an integer"):
            parse_exemptions_with_rationale(path)

    def test_the_same_file_parses_when_the_domain_opts_in(self, tmp_path: Path) -> None:
        """The control, so the test above cannot pass by rejecting everything.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:1O2:_x\n")

        assert parse_exemptions_with_rationale(path, allow_scope_keys=True)[0].scope == "1O2"

    def test_a_negative_line_does_not_reach_int(self, tmp_path: Path) -> None:
        """`--5` used to slip past the digit guard and raise a bare ValueError.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:--5:_x\n")

        with pytest.raises(ExemptionError, match="must be an integer"):
            parse_exemptions_with_rationale(path)

    def test_a_non_decimal_digit_does_not_escape_as_a_bare_valueerror(self, tmp_path: Path) -> None:
        """`isdigit` is true for a superscript two; `int()` on one raises.

        The same escape `--5` took, one character class over. `isdecimal` is the
        predicate that agrees with `int`.

        :param tmp_path: pytest temp directory
        :ptype tmp_path: Path
        :return: nothing
        :rtype: None
        """
        path = _write(tmp_path / "ex.txt", f"# rationale: {VALID_RATIONALE}\na.py:²:_x\n")

        with pytest.raises(ExemptionError, match="must be an integer"):
            parse_exemptions_with_rationale(path)
