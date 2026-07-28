"""tests for the single-return ``find_multiple_business_returns`` walker."""

from __future__ import annotations

from pathlib import Path

from threetears.enforcement.common import Violation
from threetears.enforcement.single_return.config import SingleReturnConfig
from threetears.enforcement.single_return.walkers import find_multiple_business_returns

_EXCLUDED = SingleReturnConfig(repo_root=Path("/nowhere")).excluded_function_names


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _make_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    return repo_root


def _scan(
    src: Path,
    repo: Path,
    excluded: frozenset[str] = _EXCLUDED,
    exempt_files: dict[str, str] | None = None,
) -> list[Violation]:
    return find_multiple_business_returns((src,), repo, excluded, exempt_files or {})


class TestCompliantShapes:
    def test_a_single_return_is_clean(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "mod.py", "def f(x):\n    y = x + 1\n    return y\n")
        assert _scan(src, repo) == []

    def test_leading_guards_do_not_count(self, tmp_path: Path) -> None:
        # Guards are the shape that makes one business return achievable, so counting
        # them would penalise exactly the style the rule exists to produce.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "def f(x):\n"
            "    if x is None:\n        return 0\n"
            "    if x < 0:\n        return -1\n"
            "    if x > 99:\n        return 99\n"
            "    return x\n",
        )
        assert _scan(src, repo) == []

    def test_a_docstring_does_not_end_the_guard_prologue(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            'def f(x):\n    """Doc."""\n    if x is None:\n        return 0\n    return x\n',
        )
        assert _scan(src, repo) == []

    def test_excluded_dunders_may_branch_return(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "class C:\n"
            "    def __eq__(self, other):\n"
            "        if isinstance(other, C):\n            pass\n"
            "        if other:\n            return True\n"
            "        return False\n",
        )
        assert _scan(src, repo) == []


class TestViolations:
    def test_two_business_returns_are_reported(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "def f(x):\n    y = x\n    if y:\n        return 1\n    return 2\n",
        )
        found = _scan(src, repo)
        assert len(found) == 1
        assert found[0].symbol == "f"
        # The function's own def line, not a return line: the function is the unit
        # that has to be restructured.
        assert found[0].line == 1
        assert "lines [4, 5]" in found[0].reason

    def test_a_late_guard_shaped_if_is_business_logic(self, tmp_path: Path) -> None:
        # The first non-guard statement ends the prologue permanently. Without that,
        # any function could launder returns by opening with one guard.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "def f(x):\n"
            "    if x is None:\n        return 0\n"
            "    y = x * 2\n"
            "    if y > 10:\n        return 10\n"
            "    return y\n",
        )
        assert len(_scan(src, repo)) == 1

    def test_an_excluded_name_can_be_narrowed_per_repo(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "class C:\n    def __eq__(self, other):\n        x = other\n        if x:\n            return True\n        return False\n",
        )
        assert _scan(src, repo, excluded=frozenset()) != []


class TestNestedScopes:
    def test_a_nested_def_is_charged_to_itself(self, tmp_path: Path) -> None:
        # THE bug this walker exists to hold: an ast.walk implementation descends past
        # the scope boundary, charges the inner returns to the outer function, and
        # reports the same returns twice. Both hand-rolled copies had to be fixed
        # separately when it was found.
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "def outer(x):\n"
            "    def inner(y):\n"
            "        z = y\n"
            "        if z:\n            return 1\n"
            "        return 2\n"
            "    return inner(x)\n",
        )
        # `inner` owns its two returns; `outer` has exactly one and must stay clean.
        assert [v.symbol for v in _scan(src, repo)] == ["inner"]

    def test_a_nested_lambda_does_not_leak(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "mod.py", "def f(x):\n    g = lambda a: a + 1\n    return g(x)\n")
        assert _scan(src, repo) == []

    def test_an_async_nested_def_is_charged_to_itself(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(
            src / "mod.py",
            "async def outer(x):\n"
            "    async def inner(y):\n"
            "        z = y\n"
            "        if z:\n            return 1\n"
            "        return 2\n"
            "    return await inner(x)\n",
        )
        assert [v.symbol for v in _scan(src, repo)] == ["inner"]


class TestFileFilters:
    def test_an_exempt_file_is_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "mod.py", "def f(x):\n    y = x\n    if y:\n        return 1\n    return 2\n")
        assert _scan(src, repo, exempt_files={"src/mod.py": "legacy"}) == []

    def test_an_empty_file_is_skipped(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path / "repo")
        src = repo / "src"
        _write(src / "empty.py", "")
        assert _scan(src, repo) == []
