"""tests for ``ast_helpers`` module."""

from __future__ import annotations

import ast
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from threetears.enforcement.common import ast_helpers
from threetears.enforcement.common.ast_helpers import (
    argument_spellings,
    callee_names,
    clear_parse_cache,
    dotted,
    is_logger_call,
    is_private_name,
    is_suppress_call,
    iter_python_files,
    parse_python_file,
    parse_python_path,
    receiver,
    relative_posix_path,
)


def _touch(path: Path, content: str = "") -> Path:
    """create a parent-tree and write ``content`` to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


class TestIterPythonFiles:
    def test_skips_excluded_dirs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "src" / "pkg" / "good.py")
        _touch(tmp_path / ".venv" / "lib" / "skipped.py")
        _touch(tmp_path / "node_modules" / "skipped.py")
        _touch(tmp_path / "build" / "skipped.py")
        _touch(tmp_path / "_build" / "skipped.py")
        _touch(tmp_path / ".git" / "hooks" / "skipped.py")
        _touch(tmp_path / "src" / "pkg" / "__pycache__" / "skipped.py")
        _touch(tmp_path / "src" / "pkg" / "sub" / "good2.py")

        results = sorted(p.name for p in iter_python_files(tmp_path))
        assert results == ["good.py", "good2.py"]

    def test_skips_cookiecutter_template_dirs(self, tmp_path: Path) -> None:
        _touch(tmp_path / "src" / "real.py")
        _touch(tmp_path / "{{cookiecutter.project_slug}}" / "src" / "tpl.py")
        _touch(tmp_path / "templates" / "{{cookiecutter.x}}" / "tpl.py")

        results = sorted(p.name for p in iter_python_files(tmp_path))
        assert results == ["real.py"]

    def test_deterministic_order(self, tmp_path: Path) -> None:
        _touch(tmp_path / "z" / "z.py")
        _touch(tmp_path / "a" / "a.py")
        _touch(tmp_path / "m" / "m.py")

        results = [p.name for p in iter_python_files(tmp_path)]
        assert results == ["a.py", "m.py", "z.py"]

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        results = list(iter_python_files(tmp_path / "missing"))
        assert results == []

    def test_only_python_files(self, tmp_path: Path) -> None:
        _touch(tmp_path / "src" / "real.py")
        _touch(tmp_path / "src" / "data.json")
        _touch(tmp_path / "src" / "README.md")
        results = sorted(p.name for p in iter_python_files(tmp_path))
        assert results == ["real.py"]


class TestParsePythonFile:
    def test_parses_valid_source(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "good.py", "def f():\n    return 1\n")
        tree = parse_python_file(path)
        assert tree is not None
        assert isinstance(tree, ast.Module)

    def test_returns_none_on_syntax_error(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "bad.py", "def f(:\n")
        assert parse_python_file(path) is None

    def test_returns_none_on_unicode_error(self, tmp_path: Path) -> None:
        path = tmp_path / "binary.py"
        path.write_bytes(b"\xff\xfe\xff")
        assert parse_python_file(path) is None

    def test_parses_empty_file(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "empty.py", "")
        tree = parse_python_file(path)
        assert tree is not None
        assert tree.body == []


class TestParsePythonFileCache:
    """every gate parses every file; an unchanged file is parsed once for all of them."""

    def test_an_unchanged_file_is_parsed_once(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "mod.py", "x = 1\n")
        assert parse_python_file(path) is parse_python_file(path)

    def test_a_rewritten_file_is_parsed_again(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "mod.py", "x = 1\n")
        first = parse_python_file(path)
        path.write_text("def rewritten():\n    return 2\n")
        second = parse_python_file(path)
        assert first is not None and second is not None
        assert isinstance(second.body[0], ast.FunctionDef)
        assert second.body[0].name == "rewritten"

    def test_a_rewrite_that_keeps_size_and_mtime_is_still_seen(self, tmp_path: Path) -> None:
        # a same-size rewrite inside one mtime tick: the replaced file is a new inode
        path = _touch(tmp_path / "mod.py", "a = 1\n")
        first = parse_python_file(path)
        stat = path.stat()
        replacement = _touch(tmp_path / "replacement.py", "b = 2\n")
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(path)
        second = parse_python_file(path)
        assert first is not None and second is not None
        target = second.body[0]
        assert isinstance(target, ast.Assign) and isinstance(target.targets[0], ast.Name)
        assert target.targets[0].id == "b"

    def test_a_cached_failure_is_reported_on_every_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        notes: list[str] = []
        monkeypatch.setattr(ast_helpers, "note_unscanned", lambda target, reason: notes.append(reason))
        path = _touch(tmp_path / "bad.py", "def f(:\n")
        assert parse_python_file(path) is None
        assert parse_python_file(path) is None
        assert len(notes) == 2
        assert notes[0] == notes[1]
        assert notes[0].startswith("syntax error at line")

    def test_a_file_that_becomes_valid_parses(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "mod.py", "def f(:\n")
        assert parse_python_file(path) is None
        path.write_text("def f():\n    return 1\n")
        assert parse_python_file(path) is not None

    def test_clearing_releases_the_cached_tree(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "mod.py", "x = 1\n")
        first = parse_python_file(path)
        clear_parse_cache()
        assert parse_python_file(path) is not first

    def test_a_missing_file_still_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_python_file(tmp_path / "absent.py")


class TestParsePythonPath:
    """the raising form: what ``ast.parse(path.read_text(encoding="utf-8"))`` raises, every time."""

    def test_a_syntax_error_names_the_file_on_every_call(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "bad.py", "x = 1\ndef f(:\n")
        for _ in range(2):
            with pytest.raises(SyntaxError) as raised:
                parse_python_path(path)
            assert raised.value.filename == str(path)
            assert raised.value.lineno == 2

    def test_each_caller_gets_its_own_error(self, tmp_path: Path) -> None:
        """a cached failure raised as one shared object would carry one caller's traceback to the next."""
        path = _touch(tmp_path / "bad.py", "def f(:\n")
        raised: list[SyntaxError] = []
        for _ in range(2):
            with pytest.raises(SyntaxError) as caught:
                parse_python_path(path)
            raised.append(caught.value)
        assert raised[0] is not raised[1]
        assert raised[0].args == raised[1].args

    def test_invalid_utf8_raises_on_every_call(self, tmp_path: Path) -> None:
        path = tmp_path / "binary.py"
        path.write_bytes(b"\xff\xfe\xff")
        for _ in range(2):
            with pytest.raises(UnicodeDecodeError):
                parse_python_path(path)

    def test_shares_its_cache_with_parse_python_file(self, tmp_path: Path) -> None:
        path = _touch(tmp_path / "mod.py", "x = 1\n")
        assert parse_python_path(path) is parse_python_file(path)


class TestRelativePosixPath:
    def test_under_root_uses_forward_slashes(self, tmp_path: Path) -> None:
        root = tmp_path
        file_path = tmp_path / "src" / "pkg" / "mod.py"
        file_path.parent.mkdir(parents=True)
        file_path.write_text("")
        result = relative_posix_path(file_path, root)
        assert result == "src/pkg/mod.py"

    def test_outside_root_returns_absolute_posix(self, tmp_path: Path) -> None:
        root = tmp_path / "consumer"
        root.mkdir()
        other = tmp_path / "neighbour" / "file.py"
        other.parent.mkdir()
        other.write_text("")
        result = relative_posix_path(other, root)
        assert result.startswith("/")
        assert result.endswith("/neighbour/file.py")

    def test_path_objects_work_for_pure_paths(self) -> None:
        # use PurePosix to check that POSIX inputs round-trip correctly
        result = relative_posix_path(
            Path("/abs/repo/src/pkg/mod.py"),
            Path("/abs/repo"),
        )
        # cannot resolve nonexistent paths; relative_to via .resolve() will
        # still produce a relative result if both paths are absolute and one
        # is a prefix of the other under the resolver. since neither exists,
        # ``resolve()`` is identity on non-strict mode and the relativisation
        # works.
        assert "mod.py" in result


class TestIsPrivateName:
    def test_underscore_prefix(self) -> None:
        assert is_private_name("_x") is True
        assert is_private_name("_helper") is True
        assert is_private_name("_a_b_c") is True

    def test_dunder_excluded(self) -> None:
        assert is_private_name("__x__") is False
        assert is_private_name("__init__") is False
        assert is_private_name("__all__") is False

    def test_trailing_underscore_excluded(self) -> None:
        # PEP 8: ``class_`` is the keyword-escape convention, not private
        assert is_private_name("class_") is False
        assert is_private_name("id_") is False
        # ``_x_`` ends with ``_``, so excluded
        assert is_private_name("_x_") is False

    def test_throwaway_underscore_excluded(self) -> None:
        assert is_private_name("_") is False

    def test_public_name_excluded(self) -> None:
        assert is_private_name("foo") is False
        assert is_private_name("Foo") is False
        assert is_private_name("FOO") is False


class TestIsLoggerCall:
    LOGGER_NAMES = frozenset({"log", "logger", "_logger"})
    METHOD_NAMES = frozenset({"debug", "info", "warning", "error", "exception", "critical"})

    def test_bare_logger_name_call(self) -> None:
        node = ast.parse("log.debug('hi')").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is True

    def test_attribute_logger_receiver(self) -> None:
        # self.logger.error(...) — receiver is an Attribute, not a Name
        node = ast.parse("self.logger.error('hi')").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is True

    def test_unknown_method_rejected(self) -> None:
        node = ast.parse("log.something('hi')").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is False

    def test_non_call_rejected(self) -> None:
        node = ast.parse("x = 1").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is False

    def test_unknown_receiver_name_with_known_method_accepted(self) -> None:
        # Attribute receiver always accepted because indirect-receiver
        # logger calls (ctx.log.error) are legitimate.
        node = ast.parse("ctx.log.error('hi')").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is True

    def test_bare_call_no_attribute_rejected(self) -> None:
        node = ast.parse("debug('hi')").body[0]
        assert is_logger_call(node, self.LOGGER_NAMES, self.METHOD_NAMES) is False


class TestIsSuppressCall:
    def test_bare_suppress(self) -> None:
        node = ast.parse("suppress(KeyError)", mode="eval").body
        assert is_suppress_call(node) is True

    def test_attribute_suppress(self) -> None:
        node = ast.parse("contextlib.suppress(KeyError)", mode="eval").body
        assert is_suppress_call(node) is True

    def test_other_call_rejected(self) -> None:
        node = ast.parse("foo(1)", mode="eval").body
        assert is_suppress_call(node) is False

    def test_non_call_rejected(self) -> None:
        node = ast.parse("x", mode="eval").body
        assert is_suppress_call(node) is False


class TestDotted:
    """the spelling function every other helper is built on."""

    def test_a_bare_name_resolves(self) -> None:
        assert dotted(ast.parse("registry").body[0].value) == "registry"  # type: ignore[attr-defined]

    def test_an_attribute_chain_resolves(self) -> None:
        assert dotted(ast.parse("self._registry").body[0].value) == "self._registry"  # type: ignore[attr-defined]

    def test_a_deep_chain_resolves(self) -> None:
        assert dotted(ast.parse("a.b.c.d").body[0].value) == "a.b.c.d"  # type: ignore[attr-defined]

    def test_a_non_name_rooted_expression_does_not(self) -> None:
        """``build()[0]`` has no static name, and guessing one would be worse than None."""
        assert dotted(ast.parse("build()[0]").body[0].value) is None  # type: ignore[attr-defined]


class TestCalleeAndReceiver:
    def test_a_bare_call_has_a_callee_and_no_receiver(self, parse_call: Callable[[str], ast.Call]) -> None:
        call = parse_call("configure(x)")

        assert callee_names(call) == frozenset({"configure"})
        assert receiver(call) is None

    def test_a_method_call_has_both(self, parse_call: Callable[[str], ast.Call]) -> None:
        call = parse_call("self._registry.configure(x)")

        assert callee_names(call) == frozenset({"configure"})
        assert receiver(call) == "self._registry"


class TestArgumentSpellings:
    def test_positional_and_keyword_values_are_both_collected(self, parse_call: Callable[[str], ast.Call]) -> None:
        call = parse_call("configure(nc, l3_pool=pool)")

        assert argument_spellings(call) == frozenset({"nc", "pool"})

    def test_unspellable_arguments_are_dropped_rather_than_guessed(self, parse_call: Callable[[str], ast.Call]) -> None:
        call = parse_call("configure(build()[0], nc)")

        assert argument_spellings(call) == frozenset({"nc"})


class TestDottedUnwrapsGenericBases:
    """the fold: `fake_parity` carried a third copy of this walk for base-class lists.

    Its only difference was unwrapping `ast.Subscript`, so that
    ``BaseCollection[FakeRefEntity]`` spells as ``BaseCollection``. That is now a
    parameter rather than a copy.
    """

    def test_a_generic_base_spells_as_its_origin(self) -> None:
        node = ast.parse("BaseCollection[FakeRefEntity]").body[0].value  # type: ignore[attr-defined]

        assert dotted(node, unwrap_subscript=True) == "BaseCollection"

    def test_a_dotted_generic_base_spells_as_its_dotted_origin(self) -> None:
        node = ast.parse("mod.BaseCollection[FakeRefEntity]").body[0].value  # type: ignore[attr-defined]

        assert dotted(node, unwrap_subscript=True) == "mod.BaseCollection"

    def test_a_nested_generic_unwraps_to_the_outermost_origin(self) -> None:
        node = ast.parse("Outer[Inner[X]]").body[0].value  # type: ignore[attr-defined]

        assert dotted(node, unwrap_subscript=True) == "Outer"

    def test_it_is_off_by_default(self) -> None:
        """a subscript is not a name in most contexts; dropping the parameter silently
        would make `registry[0]` read as `registry`, which is a different object."""
        node = ast.parse("registry[0]").body[0].value  # type: ignore[attr-defined]

        assert dotted(node) is None

    def test_a_plain_name_is_unaffected_by_the_flag(self) -> None:
        node = ast.parse("BaseCollection").body[0].value  # type: ignore[attr-defined]

        assert dotted(node, unwrap_subscript=True) == "BaseCollection"
