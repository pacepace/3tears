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

from pathlib import Path

# the walker lives in a support module beside both test modules, under a public name, so this
# meta-test exercises exactly the function the enforcement test runs.
from packages.core.tests.enforcement.partition_column_walker import (  # type: ignore[import-not-found]
    tied_columns,
    violations_in_file,
)
from threetears.enforcement.no_whole_reads.schema import collect_schema

__all__: list[str] = []


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
        self,
        tmp_path: Path,
    ) -> None:
        """``WHERE agent_id = $1`` on memories produces no violation."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM memories WHERE agent_id = $1'\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_unrelated_select_produces_no_violation(
        self,
        tmp_path: Path,
    ) -> None:
        """SQL on a non-partitioned table is ignored."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM unrelated_table WHERE id = $1'\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_docstring_mentioning_table_is_not_sql(
        self,
        tmp_path: Path,
    ) -> None:
        """a docstring that mentions ``FROM memories`` is not flagged."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            '"""prose: this module reads FROM memories."""\n',
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []


class TestPartitionWalkerNegative:
    """deliberate violation surfaces a clear walker message."""

    def test_select_without_partition_predicate_flagged(
        self,
        tmp_path: Path,
    ) -> None:
        """``SELECT * FROM memories WHERE user_id = $1`` is a violation."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict != [] or deferred != []
        all_msgs = strict + deferred
        joined = "\n".join(all_msgs)
        assert "memories" in joined
        assert "agent_id" in joined
        assert str(target) in joined

    def test_violation_message_carries_file_and_line(
        self,
        tmp_path: Path,
    ) -> None:
        """walker output identifies path:line for the offending literal."""
        target = _write(
            tmp_path / "src" / "pkg" / "deep.py",
            ("# leading comment\nSQL = 'SELECT id FROM memories'\n"),
        )
        strict, deferred = violations_in_file(target)
        all_msgs = strict + deferred
        assert len(all_msgs) >= 1
        assert ":2:" in all_msgs[0]


class TestPartitionWalkerExemption:
    """documented exemption fragments are accepted by the walker."""

    def test_dynamic_scope_clause_exempt(self, tmp_path: Path) -> None:
        """``websearch_to_tsquery`` matches the dynamic-scope-clause exemption."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            ("SQL = 'SELECT * FROM memories WHERE search_vector @@ websearch_to_tsquery(\\'english\\', $1)'\n"),
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_migrations_subtree_is_exempt(self, tmp_path: Path) -> None:
        """source in a ``migrations/`` subtree is exempt by directory."""
        target = _write(
            tmp_path / "src" / "pkg" / "migrations" / "v001.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []

    def test_tests_subtree_is_exempt(self, tmp_path: Path) -> None:
        """source in a ``tests/`` subtree is exempt by directory."""
        target = _write(
            tmp_path / "tests" / "unit" / "test_x.py",
            "SQL = 'SELECT * FROM memories WHERE user_id = $1'\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict == []
        assert deferred == []


#: the parent: a schedule per conversation, its id unique on its own
_PARENT_DDL = (
    "CREATE TABLE agent_wake_schedules (conversation_id UUID NOT NULL, schedule_id UUID NOT NULL, "
    "PRIMARY KEY (conversation_id, schedule_id), UNIQUE (schedule_id))"
)
#: the parent with no unique id of its own: only the pair is a key
_PARENT_PAIR_ONLY_DDL = (
    "CREATE TABLE agent_wake_schedules (conversation_id UUID NOT NULL, schedule_id UUID NOT NULL, "
    "PRIMARY KEY (conversation_id, schedule_id))"
)
_CHILD_DDL = (
    "CREATE TABLE wake_fires (conversation_id UUID NOT NULL, fire_id UUID NOT NULL, schedule_id UUID, "
    "webhook_subscription_id UUID, PRIMARY KEY (conversation_id, fire_id))"
)
#: the composite key tying a fire's schedule to its conversation, added NOT VALID as an expand step
_PAIR_FK_DDL = (
    "ALTER TABLE wake_fires ADD CONSTRAINT wake_fires_schedule_conversation_fk "
    "FOREIGN KEY (conversation_id, schedule_id) REFERENCES agent_wake_schedules (conversation_id, schedule_id) "
    "ON DELETE CASCADE NOT VALID"
)
#: a key on the schedule alone: it fixes the schedule, not the conversation the fire is stored in
_SCHEDULE_FK_DDL = (
    "ALTER TABLE wake_fires ADD CONSTRAINT wake_fires_schedule_fk "
    "FOREIGN KEY (schedule_id) REFERENCES agent_wake_schedules (schedule_id)"
)
#: a composite key over other columns
_OTHER_PAIR_DDL = (
    "CREATE TABLE webhook_subscriptions (conversation_id UUID NOT NULL, subscription_id UUID NOT NULL, "
    "PRIMARY KEY (conversation_id, subscription_id), UNIQUE (subscription_id))",
    "ALTER TABLE wake_fires ADD CONSTRAINT wake_fires_subscription_conversation_fk "
    "FOREIGN KEY (conversation_id, webhook_subscription_id) "
    "REFERENCES webhook_subscriptions (conversation_id, subscription_id)",
)
_COUNT_BY_SCHEDULE = "SQL = 'SELECT COUNT(*) FROM wake_fires WHERE schedule_id = $1 AND actual_fired_at > $2'\n"


def _tied(tmp_path: Path, *statements: str) -> dict[str, frozenset[str]]:
    """the tied columns a migration of ``statements`` declares, read through the no_whole_reads catalog.

    :param tmp_path: the tree root
    :ptype tmp_path: Path
    :param statements: the DDL, in order
    :ptype statements: str
    :return: table onto its tied columns
    :rtype: dict[str, frozenset[str]]
    """
    body = "".join(f"S{n} = {statement!r}\n" for n, statement in enumerate(statements))
    _write(tmp_path / "src" / "pkg" / "migrations" / "v001.py", body)
    return tied_columns(collect_schema(tmp_path, (tmp_path / "src",)))


class TestPartitionWalkerForeignKeyTie:
    """a read bound by a column a declared foreign key ties to the partition column reads one partition.

    Ruling 2026-11-12 (wake fire count): the key must cover the column AND the partition column, onto a table where
    the column's referenced column alone is unique. Anything less stays a violation.
    """

    def test_tied_column_read_passes(self, tmp_path: Path) -> None:
        """the composite key onto a unique schedule id: a count bound by ``schedule_id`` passes."""
        tied = _tied(tmp_path, _PARENT_DDL, _CHILD_DDL, _PAIR_FK_DDL)
        assert tied == {"wake_fires": frozenset({"schedule_id"})}
        target = _write(tmp_path / "src" / "pkg" / "mod.py", _COUNT_BY_SCHEDULE)
        assert violations_in_file(target, tied) == ([], [])

    def test_unique_column_without_the_key_stays_red(self, tmp_path: Path) -> None:
        """the schedule id is unique on the parent, but the only key is on it alone: still a violation."""
        tied = _tied(tmp_path, _PARENT_DDL, _CHILD_DDL, _SCHEDULE_FK_DDL)
        assert tied == {}
        target = _write(tmp_path / "src" / "pkg" / "mod.py", _COUNT_BY_SCHEDULE)
        strict, _ = violations_in_file(target, tied)
        assert len(strict) == 1
        assert "conversation_id" in strict[0]

    def test_key_on_other_columns_stays_red(self, tmp_path: Path) -> None:
        """a composite key ties the subscription, not the schedule: a read by schedule is still a violation."""
        tied = _tied(tmp_path, _PARENT_DDL, _CHILD_DDL, *_OTHER_PAIR_DDL)
        assert tied == {"wake_fires": frozenset({"webhook_subscription_id"})}
        target = _write(tmp_path / "src" / "pkg" / "mod.py", _COUNT_BY_SCHEDULE)
        strict, _ = violations_in_file(target, tied)
        assert len(strict) == 1

    def test_key_onto_a_non_unique_column_stays_red(self, tmp_path: Path) -> None:
        """the parent's schedule id is unique only with its conversation: one id may span conversations."""
        tied = _tied(tmp_path, _PARENT_PAIR_ONLY_DDL, _CHILD_DDL, _PAIR_FK_DDL)
        assert tied == {}
        target = _write(tmp_path / "src" / "pkg" / "mod.py", _COUNT_BY_SCHEDULE)
        strict, _ = violations_in_file(target, tied)
        assert len(strict) == 1

    def test_tied_column_not_bound_stays_red(self, tmp_path: Path) -> None:
        """naming the tied column without binding it to a value reads every partition."""
        tied = _tied(tmp_path, _PARENT_DDL, _CHILD_DDL, _PAIR_FK_DDL)
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "SQL = 'SELECT COUNT(*) FROM wake_fires WHERE schedule_id IS NOT NULL'\n",
        )
        strict, _ = violations_in_file(target, tied)
        assert len(strict) == 1


class TestPartitionWalkerHelperCalls:
    """a keyed helper call is a read or write of its table too: it pins the partition column in ``by=``."""

    def test_a_helper_call_missing_the_partition_pin_is_flagged(self, tmp_path: Path) -> None:
        """Flips if a keyed helper call reading or writing a partitioned table without its partition column in
        ``by=`` (or with no ``by=`` at all) passes, as a statement written as SQL text does not."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "from threetears.core.keyed import read_by, update_keyed, holds_any\n"
            "async def by_user(conn, user_id):\n"
            "    return await read_by(conn, 'memories', by={'user_id': user_id}, records=True)\n"
            "async def every(conn):\n"
            "    return await holds_any(conn, 'conversations', order=['agent_id'])\n"
            "async def mark(conn, fire_id):\n"
            "    await update_keyed(conn, 'wake_fires', values={'status': 'done'}, by={'fire_id': fire_id})\n",
        )
        strict, _ = violations_in_file(target)
        assert len(strict) == 3, strict
        assert all("partition column" in message for message in strict), strict

    def test_a_helper_call_pinning_the_partition_column_passes(self, tmp_path: Path) -> None:
        """Flips if a helper call pinning the partition column in ``by=``, or handing ``by=`` built elsewhere (opaque,
        as an f-string placeholder is), or naming its table at run time, or under a statement whose comment block
        declares it ``__SPANS_PARTITIONS__``, is flagged."""
        target = _write(
            tmp_path / "src" / "pkg" / "mod.py",
            "from threetears.core.keyed import read_by, keyset_page\n"
            "async def by_user(conn, agent_id, user_id):\n"
            "    return await read_by(conn, 'memories', by={'agent_id': agent_id, 'user_id': user_id})\n"
            "async def page(conn, by, table):\n"
            "    await keyset_page(conn, 'conversations', columns=None, by=by, order=[], limit=5)\n"
            "    await read_by(conn, table, by={'user_id': 1})\n"
            "async def due(conn):\n"
            "    # __SPANS_PARTITIONS__: the tick reads due jobs across every partition\n"
            "    async with conn.transaction():\n"
            "        rows = (\n"
            "            await read_by(conn, 'scheduled_jobs', by={'status': 'active'})\n"
            "        )\n",
        )
        strict, deferred = violations_in_file(target)
        assert strict == [] and deferred == [], strict
