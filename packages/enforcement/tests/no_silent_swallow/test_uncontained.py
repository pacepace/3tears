"""tests for the ``find_uncontained`` walker and the runner's pending list.

a broad handler re-raises or calls ``contain``; a cancellation handler re-raises. a log line or
a ``# NOSILENT`` comment excuses neither, and the pending list may only name packages that still
need it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.no_silent_swallow import (
    NoSilentSwallowConfig,
    find_uncontained,
    run_no_silent_swallow_enforcement,
)

_CONTAIN = frozenset({"contain"})


def _scan(tmp_path: Path, source: str) -> list[tuple[str, int, str]]:
    module = tmp_path / "src" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(source)
    return [(v.category, v.line, v.symbol) for v in find_uncontained((tmp_path / "src",), _CONTAIN)]


class TestBroad:
    def test_a_logged_broad_handler_is_flagged(self, tmp_path: Path) -> None:
        source = "def f():\n    try:\n        g()\n    except Exception as exc:\n        log.exception('g failed')\n"
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "Exception")]

    def test_a_nosilent_comment_does_not_excuse_it(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except Exception:\n        # NOSILENT: best effort, the caller does not care\n        return None\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "Exception")]

    def test_exception_inside_a_tuple_is_broad(self, tmp_path: Path) -> None:
        source = "def f():\n    try:\n        g()\n    except (ValueError, Exception):\n        return 0\n"
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "(ValueError, Exception)")]

    def test_contain_satisfies_it(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except Exception as exc:\n        contain(exc, log=log, what='g failed')\n        return None\n"
        )
        assert _scan(tmp_path, source) == []

    def test_contain_called_through_its_module_satisfies_it(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except Exception as exc:\n        observe.contain(exc, log=log, what='g failed')\n"
        )
        assert _scan(tmp_path, source) == []

    def test_a_raise_on_any_path_satisfies_it(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except Exception as exc:\n        if fatal(exc):\n            raise\n        retry()\n"
        )
        assert _scan(tmp_path, source) == []

    def test_a_raise_or_contain_in_a_nested_function_does_not_count(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except Exception as exc:\n"
            "        def later():\n            contain(exc, log=log, what='x')\n            raise exc\n"
            "        schedule(later)\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "Exception")]

    def test_base_exception_must_re_raise_and_cannot_be_contained(self, tmp_path: Path) -> None:
        source = (
            "def f():\n    try:\n        g()\n"
            "    except BaseException as exc:\n        contain(exc, log=log, what='g failed')\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "BaseException")]

    def test_a_narrow_handler_is_not_this_walkers_business(self, tmp_path: Path) -> None:
        source = "def f():\n    try:\n        g()\n    except KeyError:\n        return None\n"
        assert _scan(tmp_path, source) == []

    def test_a_bare_except_is_left_to_the_silent_body_walker(self, tmp_path: Path) -> None:
        source = "def f():\n    try:\n        g()\n    except:\n        pass\n"
        assert _scan(tmp_path, source) == []

    def test_except_star_is_scanned(self, tmp_path: Path) -> None:
        source = "def f():\n    try:\n        g()\n    except* Exception:\n        log.warning('x')\n"
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 4, "Exception")]

    def test_suppress_exception_is_flagged(self, tmp_path: Path) -> None:
        source = (
            "import contextlib\n\ndef f():\n"
            "    # NOSILENT: best effort\n    with contextlib.suppress(Exception):\n        g()\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.broad", 5, "Exception")]


class TestCancel:
    def test_an_absorbed_cancellation_is_flagged(self, tmp_path: Path) -> None:
        source = (
            "import asyncio\n\nasync def stop(task):\n    task.cancel()\n    try:\n        await task\n"
            "    except asyncio.CancelledError:\n        pass\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.cancel", 7, "CancelledError")]

    def test_a_re_raised_cancellation_is_fine(self, tmp_path: Path) -> None:
        source = (
            "import asyncio\n\nasync def run():\n    try:\n        await work()\n"
            "    except asyncio.CancelledError:\n        await tidy()\n        raise\n"
        )
        assert _scan(tmp_path, source) == []

    def test_suppress_cancelled_error_is_flagged_marker_or_not(self, tmp_path: Path) -> None:
        source = (
            "import asyncio, contextlib\n\nasync def stop(task):\n    task.cancel()\n"
            "    # NOSILENT: the cancellation requested on the line above\n"
            "    with contextlib.suppress(asyncio.CancelledError):\n        await task\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.cancel", 6, "CancelledError")]

    def test_a_cancellation_caught_with_exception_is_reported_once_as_cancel(self, tmp_path: Path) -> None:
        source = (
            "import asyncio\n\nasync def f():\n    try:\n        await g()\n"
            "    except (asyncio.CancelledError, Exception) as exc:\n        contain(exc, log=log, what='g')\n"
        )
        assert _scan(tmp_path, source) == [("no_silent_swallow.cancel", 6, "(CancelledError, Exception)")]


def _repo(tmp_path: Path, packages: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    for name, source in packages.items():
        module = repo / "packages" / name / "src" / name / "mod.py"
        module.parent.mkdir(parents=True)
        module.write_text(source)
    return repo


_UNCONTAINED = "def f():\n    try:\n        g()\n    except Exception:\n        log.warning('g failed')\n"
_CLEAN = "def f():\n    return g()\n"


def _run(repo: Path, pending: tuple[str, ...]) -> None:
    src_roots = tuple(sorted((repo / "packages").glob("*/src")))
    run_no_silent_swallow_enforcement(
        NoSilentSwallowConfig(repo_root=repo, src_roots=src_roots, uncontained_pending=pending)
    )


class TestPendingList:
    def test_a_pending_package_is_not_checked_yet(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, {"old": _UNCONTAINED, "new": _CLEAN})
        _run(repo, ("packages/old",))

    def test_a_package_not_listed_is_checked(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, {"old": _UNCONTAINED, "new": _UNCONTAINED})
        with pytest.raises(pytest.fail.Exception, match=r"no_silent_swallow\.broad.*packages/new"):
            _run(repo, ("packages/old",))

    def test_an_entry_with_nothing_left_to_convert_must_go(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, {"old": _CLEAN})
        with pytest.raises(pytest.fail.Exception, match="packages/old has no uncontained handler left"):
            _run(repo, ("packages/old",))

    def test_an_entry_naming_no_directory_must_go(self, tmp_path: Path) -> None:
        repo = _repo(tmp_path, {"old": _CLEAN})
        with pytest.raises(pytest.fail.Exception, match="packages/gone is not a directory"):
            _run(repo, ("packages/gone",))
