"""meta-tests for the partition-column enforcement walker.

exercises the walker's three contracts -- (1) clean source surfaces no
violation, (2) source touching a partitioned table without the partition
column predicate produces a violation that names the file + line, (3)
source matching one of the documented exemption fragments is accepted
-- against synthetic source files in ``tmp_path``.

review-task-01 finding D-8: every AST walker should ship with at least
one positive + negative + exemption fixture so a buggy walker that
silently passes everything is caught at CI time. partition-hardening-
task-01 sub-task 4 closes the gap for the partition walker.

:see: tests/enforcement/test_partition_column_enforcement.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

__all__: list[str] = []


def _load_walker() -> Any:
    """import the partition walker module by file path.

    pytest's ``--import-mode=importlib`` does not give sibling test
    modules a stable importable name, so the walker module is loaded
    by file location. avoids any package-layout assumption.

    :return: the imported walker module
    :rtype: Any
    """
    here = Path(__file__).resolve().parent
    target = here / "test_partition_column_enforcement.py"
    spec = importlib.util.spec_from_file_location(
        "_partition_walker_under_test", target,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_walker = _load_walker()
_violations_in_file = _walker._violations_in_file


def _write(path: Path, text: str) -> Path:
    """write ``text`` to ``path``, creating parent dirs; return ``path``.

    :param path: target file
    :ptype path: Path
    :param text: file contents
    :ptype text: str
    :return: same path, for chaining
    :rtype: Path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestPartitionWalkerPositive:
    """clean source surfaces no walker violation."""

    def test_select_with_partition_predicate_passes(
        self, tmp_path: Path,
    ) -> None:
        """``WHERE agent_id = $1`` on memories produces no violation."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM memories WHERE agent_id = $1'\n",
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_unrelated_select_produces_no_violation(
        self, tmp_path: Path,
    ) -> None:
        """SQL on a non-partitioned table is ignored."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM unrelated_table WHERE id = $1'\n",
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_docstring_mentioning_table_is_not_sql(
        self, tmp_path: Path,
    ) -> None:
        """a docstring that mentions ``FROM memories`` is not flagged."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            '"""prose: this module reads FROM memories."""\n',
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []


class TestPartitionWalkerNegative:
    """deliberate violation surfaces a clear walker message."""

    def test_select_without_partition_predicate_flagged(
        self, tmp_path: Path,
    ) -> None:
        """``SELECT * FROM memories WHERE user_id = $1`` is a violation."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = _violations_in_file(target)
        assert strict != [] or deferred != []
        all_msgs = strict + deferred
        joined = "\n".join(all_msgs)
        assert "memories" in joined
        assert "agent_id" in joined
        assert str(target) in joined

    def test_violation_message_carries_file_and_line(
        self, tmp_path: Path,
    ) -> None:
        """walker output identifies path:line for the offending literal."""
        target = _write(
            tmp_path / "src" / "pkg" / "deep.py",
            (
                "# leading comment\n"
                "SQL = 'SELECT id FROM memories'\n"
            ),
        )
        strict, deferred = _violations_in_file(target)
        all_msgs = strict + deferred
        assert len(all_msgs) >= 1
        assert ":2:" in all_msgs[0]


class TestPartitionWalkerExemption:
    """documented exemption fragments are accepted by the walker."""

    def test_dynamic_scope_clause_exempt(self, tmp_path: Path) -> None:
        """``websearch_to_tsquery`` matches the dynamic-scope-clause exemption."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            (
                "SQL = '"
                "SELECT * FROM memories WHERE search_vector @@ "
                "websearch_to_tsquery(\\'english\\', $1)"
                "'\n"
            ),
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_migrations_subtree_is_exempt(self, tmp_path: Path) -> None:
        """source in a ``migrations/`` subtree is exempt by directory."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_tests_subtree_is_exempt(self, tmp_path: Path) -> None:
        """source in a ``tests/`` subtree is exempt by directory."""
        target = _write(
            tmp_path / "tests" / "unit" / "test_x.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = _violations_in_file(target)
        assert strict == []
        assert deferred == []
