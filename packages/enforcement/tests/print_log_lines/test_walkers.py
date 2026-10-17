"""tests for the print-log-lines walkers: what counts as a log line, and which modules are in scope."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.common import Violation
from threetears.enforcement.print_log_lines import find_print_log_lines, find_scoped_modules, imports_any_root

_MARKER = "# print-log: ok"
_USES = "from threetears.observe import get_logger\n"


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    src = repo / "src"
    src.mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo, src


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _scan(
    src: Path, repo: Path, *, exempt: dict[str, str] | None = None, roots: tuple[str, ...] = ("threetears",)
) -> list[Violation]:
    return find_print_log_lines((src,), repo, exempt or {}, _MARKER, roots)


class TestScope:
    @pytest.mark.parametrize(
        "source",
        [
            "import threetears.core\n",
            "from threetears.observe import get_logger\n",
            "import threetears\n",
            "def f():\n    from threetears.core import sync\n",
        ],
    )
    def test_a_module_importing_the_root_is_in_scope(self, source: str) -> None:
        assert imports_any_root(ast.parse(source), ("threetears",))

    @pytest.mark.parametrize(
        "source",
        ["import threetearsx\n", "from . import threetears\n", "import os\n", "x = 'threetears'\n"],
    )
    def test_a_module_not_importing_it_is_out_of_scope(self, source: str) -> None:
        assert not imports_any_root(ast.parse(source), ("threetears",))

    def test_an_empty_root_list_puts_every_module_in_scope(self) -> None:
        assert imports_any_root(ast.parse("import os\n"), ())

    def test_a_module_without_3tears_may_print_to_stderr(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "helper.py", "import sys\nprint('x', file=sys.stderr)\n")
        assert _scan(src, repo) == []
        assert len(_scan(src, repo, roots=())) == 1

    def test_scoped_modules_skip_exempt_and_out_of_scope_files(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "app.py", _USES)
        _write(src / "cli.py", _USES)
        _write(src / "helper.py", "import os\n")
        scoped = find_scoped_modules((src,), repo, {"src/cli.py": "the command line"}, ("threetears",))
        assert [p.name for p in scoped] == ["app.py"]


class TestStderrPrint:
    @pytest.mark.parametrize(
        "source",
        [
            "import sys\nprint('lost', file=sys.stderr)\n",
            "import sys\nprint('lost', file=sys.__stderr__)\n",
            "import sys as s\nprint('lost', file=s.stderr)\n",
            "from sys import stderr\nprint('lost', file=stderr)\n",
            "from sys import stderr as err\nprint('lost', flush=True, file=err)\n",
            "import sys\ndef f():\n    print(\n        'lost',\n        file=sys.stderr,\n    )\n",
        ],
    )
    def test_each_spelling_is_flagged(self, tmp_path: Path, source: str) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "mod.py", _USES + source)
        (found,) = _scan(src, repo)
        assert found.category == "print_log_lines.stderr_print"
        assert found.symbol == "print"
        assert "get_logger" in found.reason

    @pytest.mark.parametrize(
        "source",
        [
            "print('the answer')\n",
            "import sys\nprint('the answer', file=sys.stdout)\n",
            "import sys\nwith open('f') as fh:\n    print('row', file=fh)\n",
            "stderr = open('f')\nprint('row', file=stderr)\n",
            "import sys\nconsole.print('x', file=sys.stderr)\n",
        ],
    )
    def test_output_that_is_not_a_stderr_log_line_is_left_alone(self, tmp_path: Path, source: str) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "mod.py", _USES + source)
        assert _scan(src, repo) == []

    def test_the_line_number_is_the_calls(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "mod.py", _USES + "import sys\n\nprint('x', file=sys.stderr)\n")
        (found,) = _scan(src, repo)
        assert found.line == 4


class TestStdlibLogging:
    @pytest.mark.parametrize(
        ("source", "symbol"),
        [
            ("import logging\nlog = logging.getLogger(__name__)\n", "logging.getLogger"),
            ("import logging\nlogging.basicConfig(level=logging.INFO)\n", "logging.basicConfig"),
            ("import logging as lg\nlg.warning('x')\n", "logging.warning"),
            ("from logging import getLogger\nlog = getLogger('x')\n", "logging.getLogger"),
            ("from logging import basicConfig as setup\nsetup()\n", "logging.basicConfig"),
            ("import logging.handlers\nlogging.info('x')\n", "logging.info"),
        ],
    )
    def test_each_direct_use_is_flagged(self, tmp_path: Path, source: str, symbol: str) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "mod.py", _USES + source)
        (found,) = _scan(src, repo)
        assert found.category == "print_log_lines.stdlib_logging"
        assert found.symbol == symbol

    @pytest.mark.parametrize(
        "source",
        [
            "log = get_logger(__name__)\nlog.warning('x')\nlog.info('y')\n",
            "import logging\nlevel = logging.INFO\n",
            "obj.getLogger('x')\n",
            "def getLogger(n):\n    return n\ngetLogger('x')\n",
        ],
    )
    def test_observe_loggers_and_lookalikes_are_left_alone(self, tmp_path: Path, source: str) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "mod.py", _USES + source)
        assert _scan(src, repo) == []


class TestEscapeHatches:
    def test_an_exempt_file_is_not_scanned(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path)
        _write(src / "cli.py", _USES + "import sys\nprint('usage', file=sys.stderr)\n")
        assert _scan(src, repo, exempt={"src/cli.py": "the command line answers on stderr"}) == []

    def test_the_marker_excuses_one_call_on_any_line_it_spans(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path)
        source = (
            _USES
            + "import sys\n"
            + "print(\n    'ok',\n    file=sys.stderr,  # print-log: ok\n)\n"
            + "print('flagged', file=sys.stderr)\n"
        )
        _write(src / "mod.py", source)
        (found,) = _scan(src, repo)
        assert found.line == 7
