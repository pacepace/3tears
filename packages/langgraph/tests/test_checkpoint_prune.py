"""``ThreeTierCheckpointSaver.aprune`` refuses a call it cannot honour before touching the database.

The behaviour against real rows is ``integration/test_checkpoint_prune_live.py``; these are the
refusals, which need no database, and the maintenance command line's own refusal.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.langgraph import checkpoint_maintenance
from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_maintenance.__main__ import main
from threetears.langgraph.checkpoint_scope import CheckpointScope

_UNSCOPED = CheckpointScope.unscoped(reason="tests pin the un-tenanted keyspace")


def _executor() -> Any:
    """an executor stand-in that records every call.

    :return: mock executor with async fetch/fetchrow/execute
    :rtype: Any
    """
    executor = MagicMock()
    executor.fetch = AsyncMock(return_value=[])
    executor.fetchrow = AsyncMock(return_value=None)
    executor.execute = AsyncMock(return_value="DELETE 0")
    return executor


def _callers(package: Path, names: set[str]) -> dict[str, set[str]]:
    """each named function -> the functions of the package's source calling it by name.

    :param package: the package's source directory
    :ptype package: Path
    :param names: the callees
    :ptype names: set[str]
    :return: callee -> its callers
    :rtype: dict[str, set[str]]
    """
    found: dict[str, set[str]] = {}
    for source in sorted(package.rglob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call):
                    callee = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                    if callee in names:
                        found.setdefault(callee, set()).add(function.name)
    return found


def _touched(executor: Any) -> int:
    """
    :param executor: the stand-in
    :ptype executor: Any
    :return: calls made on it
    :rtype: int
    """
    return int(executor.fetch.await_count + executor.fetchrow.await_count + executor.execute.await_count)


class TestRefusals:
    async def test_an_unknown_strategy_is_refused_before_any_statement(self) -> None:
        executor = _executor()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        with pytest.raises(ValueError, match="keep_latest, delete"):
            await saver.aprune(["t-1"], strategy="keep_oldest")

        assert _touched(executor) == 0

    async def test_a_bare_string_is_refused(self) -> None:
        """a string is a sequence of one-character thread ids; pruning those is never meant."""
        executor = _executor()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        with pytest.raises(TypeError, match="bare string"):
            await saver.aprune("t-1")

        assert _touched(executor) == 0

    async def test_no_threads_runs_no_statement(self) -> None:
        executor = _executor()
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        result = await saver.aprune([])

        assert (result.threads, result.checkpoints_deleted, result.writes_deleted) == (0, 0, 0)
        assert _touched(executor) == 0

    async def test_a_status_tag_without_a_count_is_an_error_not_a_zero(self) -> None:
        executor = _executor()
        executor.execute = AsyncMock(return_value="OK")
        saver = ThreeTierCheckpointSaver(executor, scope=_UNSCOPED)

        with pytest.raises(ValueError, match="no row count"):
            await saver.aprune(["t-1"])

    def test_the_sync_prune_says_to_use_aprune(self) -> None:
        saver = ThreeTierCheckpointSaver(_executor(), scope=_UNSCOPED)

        with pytest.raises(NotImplementedError, match="aprune"):
            saver.prune(["t-1"])


class TestCommandLine:
    @pytest.mark.parametrize("action", ["prune", "delete-unowned"])
    def test_a_deleting_action_needs_yes_or_dry_run(self, action: str, capsys: pytest.CaptureFixture[str]) -> None:
        """refused at the command line, before any connection is opened."""
        args = [action, "--dsn", "postgresql://nowhere.invalid/none", "--owned-sql", "SELECT 'x'"]

        with pytest.raises(SystemExit) as exit_info:
            main(args)

        assert exit_info.value.code == 2
        assert "--yes" in capsys.readouterr().err

    def test_delete_unowned_needs_the_owned_query(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exit_info:
            main(["delete-unowned", "--dsn", "postgresql://nowhere.invalid/none", "--dry-run"])

        assert exit_info.value.code == 2
        assert "--owned-sql" in capsys.readouterr().err

    def test_the_whole_walks_are_reached_only_from_the_operator_s_command_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """the walks over every stored thread are an operator's command (``@operator_command``): reached in the
        package only from ``main``, the command line, which runs a deleting action only with --yes."""
        package = Path(checkpoint_maintenance.__file__).parents[1]
        walks = {"_thread_page", "_thread_pages", "checkpoint_usage", "prune_threads", "delete_unowned_threads", "_run"}

        assert _callers(package, walks) == {
            "_thread_page": {"_thread_pages"},
            "_thread_pages": {"checkpoint_usage", "prune_threads", "delete_unowned_threads"},
            "checkpoint_usage": {"_run"},
            "prune_threads": {"_run"},
            "delete_unowned_threads": {"_run"},
            "_run": {"main"},
        }
        with pytest.raises(SystemExit):
            main(["prune", "--dsn", "postgresql://nowhere.invalid/none"])
        assert "--yes" in capsys.readouterr().err
