"""tests for shape J: a src class that overrides, or reads through ``super()``, a third-party private member.

The third-party base throughout is the standard library's ``argparse.HelpFormatter``, whose
``_format_usage``, ``_split_lines`` and ``_fill_text`` are private (only the class's name is its public
API), and which is importable everywhere. The repo's own namespace is ``acme``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.underscore_access import (
    UnderscoreAccessConfig,
    run_underscore_enforcement,
    shape_j_file_violations,
    shape_j_violations,
    unresolved_entries,
)

_RATIONALE = "argparse.HelpFormatter (verified on this interpreter) keeps the usage line private"


def _write(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    """a single-package repo whose ``src`` holds ``files``, by path under ``src``."""
    repo = tmp_path / "repo"
    _write(repo / "pyproject.toml", '[project]\nname = "synthetic"\n')
    src = repo / "src"
    _write(src / "acme" / "__init__.py", "")
    for relative, source in files.items():
        _write(src / relative, source)
    return repo, src


def _found(tmp_path: Path, files: dict[str, str]) -> list[tuple[str, int, str]]:
    repo, src = _repo(tmp_path, files)
    return [(v.file.relative_to(src).as_posix(), v.line, v.symbol) for v in shape_j_violations((src,), repo)]


_OVERRIDE = """\
import argparse


class Formatter(argparse.HelpFormatter):
    def _format_usage(self, usage, actions, groups, prefix):
        return super()._format_usage(usage, actions, groups, prefix)
"""


class TestWhatIsReported:
    def test_an_override_of_a_third_party_private_member(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path, {"acme/fmt.py": _OVERRIDE})
        [violation] = shape_j_violations((src,), repo)
        assert (violation.category, violation.line, violation.symbol) == ("underscore_access.J", 5, "_format_usage")
        assert "argparse.HelpFormatter" in violation.reason
        assert "overrides" in violation.reason

    def test_an_override_and_its_super_read_are_one_violation_at_the_definition(self, tmp_path: Path) -> None:
        assert _found(tmp_path, {"acme/fmt.py": _OVERRIDE}) == [("acme/fmt.py", 5, "_format_usage")]

    def test_a_super_read_of_a_member_the_class_does_not_define(self, tmp_path: Path) -> None:
        source = (
            "from argparse import HelpFormatter as Base\n\n\n"
            "class Formatter(Base):\n"
            "    def wrapped(self, text, width):\n"
            "        return super()._split_lines(text, width)\n"
        )
        repo, src = _repo(tmp_path, {"acme/fmt.py": source})
        [violation] = shape_j_violations((src,), repo)
        assert (violation.line, violation.symbol) == (6, "_split_lines")
        assert "through super()" in violation.reason

    def test_a_class_body_assignment_over_a_private_member(self, tmp_path: Path) -> None:
        source = "import argparse\n\n\nclass Formatter(argparse.HelpFormatter):\n    _fill_text = None\n"
        assert _found(tmp_path, {"acme/fmt.py": source}) == [("acme/fmt.py", 5, "_fill_text")]

    def test_a_base_that_cannot_be_imported_is_reported_not_passed(self, tmp_path: Path) -> None:
        source = (
            "from no_such_library_anywhere import Thing\n\n\nclass Sub(Thing):\n    def _helper(self):\n        pass\n"
        )
        repo, src = _repo(tmp_path, {"acme/sub.py": source})
        [violation] = shape_j_violations((src,), repo)
        assert violation.symbol == "_helper"
        assert "could not be imported" in violation.reason


class TestWhatIsNot:
    def test_a_new_private_helper_the_base_does_not_have(self, tmp_path: Path) -> None:
        source = (
            "import argparse\n\n\nclass Formatter(argparse.HelpFormatter):\n    def _own_helper(self):\n        pass\n"
        )
        assert _found(tmp_path, {"acme/fmt.py": source}) == []

    def test_dunders_and_mangled_names(self, tmp_path: Path) -> None:
        source = (
            "import argparse\n\n\n"
            "class Formatter(argparse.HelpFormatter):\n"
            "    def __repr__(self):\n        return 'f'\n"
            "    def __secret(self):\n        pass\n"
        )
        assert _found(tmp_path, {"acme/fmt.py": source}) == []

    def test_a_class_with_only_repo_bases(self, tmp_path: Path) -> None:
        files = {
            "acme/base.py": "class Base:\n    def _hook(self):\n        pass\n",
            "acme/sub.py": "from acme.base import Base\n\n\nclass Sub(Base):\n    def _hook(self):\n        pass\n",
        }
        assert _found(tmp_path, files) == []

    def test_a_hook_langchain_documents_for_subclasses_is_not_a_reach_but_its_other_privates_are(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("langchain_core")
        source = (
            "from langchain_core.tools import BaseTool\n"
            "from langchain_core.language_models import BaseChatModel\n\n\n"
            "class Tool(BaseTool):\n"
            "    def _run(self, *args):\n        pass\n"
            "    def _parse_input(self, *args):\n        pass\n\n\n"
            "class Model(BaseChatModel):\n"
            "    def _generate(self, *args):\n        pass\n"
            "    def _stream(self, *args):\n        pass\n"
            "    def _combine_llm_outputs(self, *args):\n        pass\n"
        )
        assert [symbol for _, _, symbol in _found(tmp_path, {"acme/lc.py": source})] == [
            "_parse_input",
            "_combine_llm_outputs",
        ]

    def test_a_base_that_is_not_a_class(self, tmp_path: Path) -> None:
        source = "from json import dumps\n\n\nclass Odd(dumps):\n    def _helper(self):\n        pass\n"
        assert _found(tmp_path, {"acme/odd.py": source}) == []


class TestTheRepoIsFollowedStatically:
    """an in-repo base is read from its source, never imported, so the chain to a library is followed
    through it and its first definer decides."""

    _BASE = (
        "import argparse\n\n\nclass Base(argparse.HelpFormatter):\n    def _format_usage(self, *args):\n        pass\n"
    )

    def test_the_repo_class_that_overrides_first_is_the_one_reported(self, tmp_path: Path) -> None:
        files = {
            "acme/base.py": self._BASE,
            "acme/sub.py": "from .base import Base\n\n\nclass Sub(Base):\n    def _format_usage(self, *args):\n        pass\n",
        }
        assert _found(tmp_path, files) == [("acme/base.py", 5, "_format_usage")]

    def test_a_library_private_reached_through_a_repo_base(self, tmp_path: Path) -> None:
        files = {
            "acme/base.py": self._BASE,
            "acme/sub.py": "from acme.base import Base\n\n\nclass Sub(Base):\n    def _split_lines(self, *args):\n        pass\n",
        }
        assert ("acme/sub.py", 5, "_split_lines") in _found(tmp_path, files)

    def test_a_repo_base_reached_through_a_package_re_export(self, tmp_path: Path) -> None:
        files = {
            "acme/base.py": self._BASE,
            "acme/api/__init__.py": "from acme.base import Base\n",
            "acme/sub.py": "from acme.api import Base\n\n\nclass Sub(Base):\n    def _fill_text(self, *args):\n        pass\n",
        }
        assert ("acme/sub.py", 5, "_fill_text") in _found(tmp_path, files)

    def test_a_class_defined_in_a_function_with_its_base_imported_there(self, tmp_path: Path) -> None:
        source = (
            "def build():\n"
            "    from argparse import HelpFormatter\n\n"
            "    class Formatter(HelpFormatter):\n"
            "        def _split_lines(self, text, width):\n"
            "            return []\n\n"
            "    return Formatter\n"
        )
        assert _found(tmp_path, {"acme/fmt.py": source}) == [("acme/fmt.py", 5, "_split_lines")]


class TestTheLedger:
    def test_one_file_reports_what_the_repo_walk_reports(self, tmp_path: Path) -> None:
        repo, src = _repo(tmp_path, {"acme/fmt.py": _OVERRIDE})
        assert shape_j_file_violations(src / "acme" / "fmt.py", repo) == shape_j_violations((src,), repo)
        assert shape_j_file_violations(repo / "pyproject.toml", repo) == []

    def test_an_override_entry_resolves_while_the_base_has_the_member_and_not_after(self, tmp_path: Path) -> None:
        """An override is a definition, not an attribute read: only shape J's own findings resolve its
        entry, and once the base no longer has the member the entry is stale."""
        source = "import argparse\n\n\nclass Formatter(argparse.HelpFormatter):\n    def _split_lines(self, text, width):\n        return []\n"
        repo, src = _repo(tmp_path, {"acme/fmt.py": source})
        entry = "src/acme/fmt.py:Formatter._split_lines#0:_split_lines"
        ledger = _write(repo / "exemptions.txt", f"# rationale: {_RATIONALE}\n{entry}\n")
        assert unresolved_entries(ledger, repo) == []
        _write(src / "acme" / "fmt.py", source.replace("argparse.HelpFormatter", "argparse.Namespace"))
        assert unresolved_entries(ledger, repo) == [entry]


class TestTheRunner:
    def test_shape_j_fails_the_gate_and_is_part_of_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNDERSCORE_AUDIT_MODE", raising=False)
        repo, _ = _repo(tmp_path, {"acme/fmt.py": _OVERRIDE})
        config = UnderscoreAccessConfig(repo_root=repo, enable_shape_b_ruff=False)
        with pytest.raises(pytest.fail.Exception, match="underscore_access.J"):
            run_underscore_enforcement(config, walker="shape_j")
        with pytest.raises(pytest.fail.Exception, match="_format_usage"):
            run_underscore_enforcement(config, walker="all")

    def test_a_scope_keyed_entry_exempts_it(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNDERSCORE_AUDIT_MODE", raising=False)
        repo, _ = _repo(tmp_path, {"acme/fmt.py": _OVERRIDE})
        ledger = _write(
            repo / "exemptions.txt",
            f"# rationale: {_RATIONALE}\nsrc/acme/fmt.py:Formatter._format_usage#0:_format_usage\n",
        )
        config = UnderscoreAccessConfig(repo_root=repo, enable_shape_b_ruff=False, exemptions_path=ledger)
        run_underscore_enforcement(config, walker="shape_j")
