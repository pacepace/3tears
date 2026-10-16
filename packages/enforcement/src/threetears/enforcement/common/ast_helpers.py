"""ast-walking helpers shared across every enforcement domain.

these helpers exist because every domain walker re-implemented the same
file-iteration / parse / private-name / logger-call recognisers slightly
differently. consolidating them here keeps the walkers focused on the
shape they're actually checking.

**The dotted-name spelling family lives here too** -- :func:`dotted`,
:func:`callee_names`, :func:`receiver`, :func:`argument_spellings`. Look here
before writing another one: this walk has been reimplemented three times in this
package already, and each copy differed by one detail that turned out to be a
parameter rather than a reason. :func:`dotted`'s ``unwrap_subscript`` is the
surviving example.

One near-relative deliberately stays separate: ``inheritance._base_name`` does
the same unwrap but returns the LAST segment, because the class-base graph keys
by bare name. Folding it would change that graph's keys, so it is a different
function rather than a fifth copy -- but note the two therefore disagree on a
dotted base (``pkg.Base`` here, ``Base`` there), which is a real difference to
know about before comparing their outputs.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from threetears.observe import BuildOnce, get_logger

__all__ = [
    "SKIP_DIRS",
    "clear_parse_cache",
    "iter_python_files",
    "nodes_by_type",
    "note_unscanned",
    "parse_python_file",
    "parse_python_path",
    "relative_posix_path",
    "is_private_name",
    "is_logger_call",
    "is_suppress_call",
    "dotted",
    "callee_names",
    "receiver",
    "argument_spellings",
]

log = get_logger(__name__)


def note_unscanned(target: Path | str, reason: str) -> None:
    """record something an enforcement walker could not examine.

    every walker skips files it cannot read or parse, and a skipped file is indistinguishable in
    the result from a file that was read and found clean -- the gate reports a pass over source it
    never looked at. that is the whole failure mode these gates exist to catch, so the skips go
    through one function: ``grep note_unscanned`` finds every place coverage can silently thin,
    and the warnings say which files it actually thinned on in a given run.

    :param target: the file, directory, or dotted name that was not examined
    :ptype target: Path | str
    :param reason: why it could not be examined
    :ptype reason: str
    :return: nothing
    :rtype: None
    """
    log.warning(
        "enforcement walker skipped a target it could not examine",
        extra={"extra_data": {"target": str(target), "reason": reason}},
    )


#: directories never walked for source-scanning passes. covers virtual
#: environments, build artefacts, IDE caches, vcs metadata, and the
#: legacy sphinx ``_build`` output dir. cookiecutter template trees are
#: detected separately because their names contain ``{{`` placeholders.
#:
#: public because layout discovery shares the exclusion: a tooling cache
#: such as ``packages/registry/.mypy_cache/3.14/src`` is a directory named
#: ``src`` that is not a source tree, and both the file iterator and the
#: src-root finder must agree on that.
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "__pycache__",
        "node_modules",
        ".git",
        "dist",
        "build",
        ".eggs",
        "_build",
    }
)


def _has_cookiecutter_marker(path: Path) -> bool:
    """true iff any path component contains a cookiecutter placeholder.

    cookiecutter directories embed ``{{cookiecutter.x}}`` in their path
    components; the files inside are jinja2 templates rather than valid
    python source.

    :param path: path to classify
    :ptype path: Path
    :return: whether any path component contains ``{{``
    :rtype: bool
    """
    return any("{{" in part for part in path.parts)


def iter_python_files(root: Path) -> Iterable[Path]:
    """yield every ``.py`` file under ``root``, deterministically ordered.

    skips :data:`SKIP_DIRS` (venv / build / cache / vcs) and any path
    that lives inside a cookiecutter template tree (component contains
    ``{{``). entries within each directory are sorted so report output
    is stable across runs.

    :param root: directory to walk
    :ptype root: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    if not root.is_dir():
        return
    yield from _walk_sorted(root)


def _walk_sorted(directory: Path) -> Iterable[Path]:
    """recursively yield ``.py`` files in deterministic order.

    :param directory: directory to walk
    :ptype directory: Path
    :return: iterator of python files
    :rtype: Iterable[Path]
    """
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except (PermissionError, FileNotFoundError) as exc:
        # A whole subtree drops out of every walker that uses this iterator, silently.
        note_unscanned(directory, f"directory could not be listed: {type(exc).__name__}")
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name in SKIP_DIRS:
                continue
            if "{{" in entry.name:
                continue
            yield from _walk_sorted(entry)
        elif entry.is_file() and entry.suffix == ".py":
            if _has_cookiecutter_marker(entry):
                continue
            yield entry


#: the file's identity on disk when it was parsed: ``(st_mtime_ns, st_size, st_ino)``.
_FileStamp = tuple[int, int, int]


@dataclass(frozen=True)
class _Parse:
    """one file's parse: the stamp it was made at, and the tree or the error it raised.

    :ivar path: the file
    :ivar stamp: the file's :data:`_FileStamp` when it was read
    :ivar outcome: the parsed module, or the error parsing raised
    """

    path: Path
    stamp: _FileStamp
    outcome: ast.Module | SyntaxError | UnicodeDecodeError


def _stamp(path: Path) -> _FileStamp:
    """the file's identity on disk now.

    :param path: the file
    :ptype path: Path
    :return: its modification time, size and inode
    :rtype: _FileStamp
    :raises OSError: when the file cannot be read
    """
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _parse(path: Path) -> _Parse:
    """read and parse *path* now.

    :param path: the file
    :ptype path: Path
    :return: the parse, failed or not
    :rtype: _Parse
    :raises OSError: when the file cannot be read
    """
    stamp = _stamp(path)
    outcome: ast.Module | SyntaxError | UnicodeDecodeError
    try:
        outcome = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        outcome = exc
    return _Parse(path=path, stamp=stamp, outcome=outcome)


def _unchanged_on_disk(parse: _Parse) -> bool:
    """whether the file is still the one *parse* read.

    :param parse: a cached parse
    :ptype parse: _Parse
    :return: ``True`` while the file's stamp is the one it was parsed at
    :rtype: bool
    :raises OSError: when the file can no longer be read
    """
    return _stamp(parse.path) == parse.stamp


#: every parse this process has made, by path. every gate walks every file, so without this a run
#: parses each file once per gate. unbounded on purpose: the gates walk the same files in the
#: same order, so any bound smaller than the file set evicts each entry just before it is wanted
#: again. a file whose stamp moved is parsed again in place. a ``BuildOnce`` rather than a dict
#: because a hand-filled module-level cache is a race between threads that ask first
#: (``tests/enforcement/test_build_once_is_the_only_lazy_fill.py``); its one lock serialises
#: parses, which costs nothing here because the gates parse from one thread.
_PARSES: BuildOnce[Path, _Parse] = BuildOnce(is_current=_unchanged_on_disk)


@dataclass(frozen=True)
class _NodeIndex:
    """one parse's nodes grouped by their exact type, each group in :func:`ast.walk` order.

    :ivar parse: the parse the nodes came from; the index is current while it is
    :ivar by_type: node type -> its nodes
    """

    parse: _Parse
    by_type: dict[type[ast.AST], tuple[ast.AST, ...]]


def _index(path: Path) -> _NodeIndex:
    """walk *path*'s cached tree once and group its nodes by type.

    :param path: the file
    :ptype path: Path
    :return: the index
    :rtype: _NodeIndex
    :raises SyntaxError: when the file is not valid python
    :raises UnicodeDecodeError: when the file is not valid utf-8
    :raises OSError: when the file cannot be read
    """
    parse = _PARSES.get(path, lambda: _parse(path))
    if not isinstance(parse.outcome, ast.Module):
        raise type(parse.outcome)(*parse.outcome.args) from None
    groups: dict[type[ast.AST], list[ast.AST]] = {}
    for node in ast.walk(parse.outcome):
        groups.setdefault(type(node), []).append(node)
    return _NodeIndex(parse=parse, by_type={kind: tuple(nodes) for kind, nodes in groups.items()})


def _index_unchanged_on_disk(index: _NodeIndex) -> bool:
    """whether the file is still the one *index* was built from.

    :param index: a cached index
    :ptype index: _NodeIndex
    :return: ``True`` while the file's stamp is the one its parse was made at
    :rtype: bool
    :raises OSError: when the file can no longer be read
    """
    return _unchanged_on_disk(index.parse)


#: every file's nodes by type, beside :data:`_PARSES` and released with it. a gate that keeps one
#: node type reads it here instead of walking the whole tree, so the walk is paid once per file
#: however many gates ask.
_NODE_INDEXES: BuildOnce[Path, _NodeIndex] = BuildOnce(is_current=_index_unchanged_on_disk)


def nodes_by_type[T: ast.AST](path: Path, node_type: type[T]) -> tuple[T, ...]:
    """every node of exactly *node_type* in *path*'s tree, in :func:`ast.walk` order.

    the same nodes, in the same order, as ``[n for n in ast.walk(parse_python_path(path)) if
    isinstance(n, node_type)]`` for a node type no other node type subclasses -- which is every
    concrete node :func:`ast.parse` produces. the tree is walked once per file and the grouping
    cached beside the parse, current while the parse is and released by
    :func:`clear_parse_cache`. the nodes are the shared tree's: read them, never mutate them.

    :param path: file to read
    :ptype path: Path
    :param node_type: the concrete node class wanted, e.g. ``ast.ClassDef``
    :ptype node_type: type[T]
    :return: the nodes
    :rtype: tuple[T, ...]
    :raises SyntaxError: when the file is not valid python
    :raises UnicodeDecodeError: when the file is not valid utf-8
    :raises OSError: when the file cannot be read
    """
    index = _NODE_INDEXES.get(path, lambda: _index(path))
    return cast(tuple[T, ...], index.by_type.get(node_type, ()))


def parse_python_path(path: Path) -> ast.Module:
    """parse a python file as utf-8: ``ast.parse(path.read_text(encoding="utf-8"))``, once.

    the result is cached by path, and reused while the file's modification
    time, size and inode are unchanged, so a file rewritten in place is
    parsed again -- unless the rewrite keeps the same size inside one tick of
    the filesystem's timestamp clock, which on Linux can be milliseconds wide.
    a scan over files that are not being edited under it never meets that; a
    caller that rewrites a file and re-parses it at once should change its size
    or call :func:`clear_parse_cache` first. a file that does not parse raises
    an equal error on every call. the returned tree is shared with every other caller: read it, never
    mutate it. :func:`clear_parse_cache` releases the cache.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module AST
    :rtype: ast.Module
    :raises SyntaxError: when the file is not valid python
    :raises UnicodeDecodeError: when the file is not valid utf-8
    :raises OSError: when the file cannot be read; never cached
    """
    outcome = _PARSES.get(path, lambda: _parse(path)).outcome
    if isinstance(outcome, ast.Module):
        return outcome
    # a fresh error per caller: re-raising the cached one would hand each caller the traceback
    # and context the previous caller's raise left on it
    raise type(outcome)(*outcome.args) from None


def parse_python_file(path: Path) -> ast.Module | None:
    """parse a python file as utf-8, returning None on parse failure.

    absorbs :class:`SyntaxError` and :class:`UnicodeDecodeError` since
    walkers must be robust to template / non-python files that slipped
    past the iterator filters. every other I/O error propagates. both
    absorbed cases go through :func:`note_unscanned` -- a file no walker
    could parse is a file no gate has actually checked -- on every call,
    so each walker that skips it says so.

    cached and shared as :func:`parse_python_path` is: read the tree, never
    mutate it.

    :param path: file to parse
    :ptype path: Path
    :return: parsed module AST, or None on syntax / unicode error
    :rtype: ast.Module | None
    """
    try:
        return parse_python_path(path)
    except UnicodeDecodeError:
        note_unscanned(path, "not valid utf-8")
        return None
    except SyntaxError as exc:
        note_unscanned(path, f"syntax error at line {exc.lineno}")
        return None


def clear_parse_cache() -> None:
    """release every tree :func:`parse_python_path` has cached, and every :func:`nodes_by_type` index.

    a whole repository's trees run to hundreds of megabytes; a process that
    has finished scanning, and goes on to other work, calls this to give them
    back.

    :return: nothing
    :rtype: None
    """
    _PARSES.clear()
    _NODE_INDEXES.clear()


def relative_posix_path(path: Path, root: Path) -> str:
    """return ``path`` relative to ``root`` with forward-slash separators.

    falls back to the absolute posix string when ``path`` is not under
    ``root`` (so cross-repo paths still render cleanly in reports). the
    forward-slash output is stable across linux / mac / windows hosts,
    which matters because exemption files store paths in the canonical
    posix form.

    :param path: target path
    :ptype path: Path
    :param root: reference root for relativisation
    :ptype root: Path
    :return: forward-slash relative path, or absolute posix path on
        cross-root
    :rtype: str
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.as_posix()
    return rel.as_posix()


def is_private_name(name: str) -> bool:
    """true iff ``name`` is a single-leading-underscore identifier.

    excludes:

    - the throwaway ``_``
    - dunders (``__foo__`` — both starts and ends with double underscore)
    - trailing-underscore identifiers (``class_`` / ``id_`` — the
      escape-keyword convention; also catches single-char variants like
      ``_x_`` since these end with ``_``)

    matches PEP 8 / SLF001 semantics: ``_helper``, ``_foo``,
    ``__name_mangled`` (starts with ``__`` but does not end with
    ``__``) — those are private. note that name-mangled ``__foo`` is
    private under this classifier because the dunder check requires
    both the start and end pattern.

    :param name: identifier to classify
    :ptype name: str
    :return: whether ``name`` is a private (underscore-prefixed)
        identifier
    :rtype: bool
    """
    if name == "_":
        return False
    if not name.startswith("_"):
        return False
    if name.startswith("__") and name.endswith("__"):
        return False
    if name.endswith("_"):
        return False
    return True


def is_logger_call(
    node: ast.AST,
    logger_names: frozenset[str],
    method_names: frozenset[str],
) -> bool:
    """true iff ``node`` is a ``logger.<method>(...)`` call expression.

    accepts either of two shapes:

    - ``Expr(Call(Attribute(Name(logger_name), method_name)))`` — a bare
      logger reference such as ``log.debug(...)`` whose receiver is one
      of the known logger names (``log`` / ``logger`` / ``_logger`` /
      etc., as configured per domain).
    - ``Expr(Call(Attribute(_, method_name)))`` — any attribute call
      whose method name is in ``method_names``. this catches indirect
      receivers (``self.log.error(...)`` / ``ctx.logger.warning(...)``)
      where the receiver is not a bare name.

    :param node: ast node to test (typically an ``Expr`` or ``Call``)
    :ptype node: ast.AST
    :param logger_names: known receiver names (``log``, ``logger``, ...)
    :ptype logger_names: frozenset[str]
    :param method_names: known logger method names (``debug``, ``info``,
        ``warning``, ``error``, ``exception``, ``critical``)
    :ptype method_names: frozenset[str]
    :return: whether the node is a logger-method call
    :rtype: bool
    """
    if isinstance(node, ast.Expr):
        node = node.value
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr not in method_names:
        return False
    # not `receiver`: that is a module-level function here now, and shadowing it inside
    # a helper that does not call it is the kind of thing that reads fine until someone
    # adds the call.
    called_on = func.value
    if isinstance(called_on, ast.Name) and called_on.id in logger_names:
        return True
    if isinstance(called_on, ast.Attribute):
        return True
    return False


def is_suppress_call(node: ast.AST) -> bool:
    """true iff ``node`` is a ``suppress(...)`` / ``contextlib.suppress(...)`` call.

    used by the no-silent-swallow walker to recognise the canonical
    "silently allow this exception" idiom. recognises both the bare-name
    callee (``suppress(KeyError)``) and the attribute callee
    (``contextlib.suppress(KeyError)``); does not require the import to
    be detected.

    :param node: ast node to test
    :ptype node: ast.AST
    :return: whether the node is a suppress call
    :rtype: bool
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name) and func.id == "suppress":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "suppress":
        return True
    return False


def dotted(node: ast.expr, *, unwrap_subscript: bool = False) -> str | None:
    """return the dotted spelling of a Name-rooted expression.

    ``registry`` and ``self._registry`` resolve; ``build()[0]`` does not, because there is
    no static name for it.

    :param node: expression to spell
    :ptype node: ast.expr
    :param unwrap_subscript: spell ``Base[Param]`` as ``Base``. OFF by default, because a
        subscript is not a name in most contexts and silently dropping the parameter
        would be a lie; ON for base-class lists, where ``BaseCollection[FakeEntity]``
        genuinely names ``BaseCollection``
    :ptype unwrap_subscript: bool
    :return: dotted spelling, or ``None`` when the expression is not name-rooted
    :rtype: str | None
    """
    parts: list[str] = []
    current: ast.expr = node
    while unwrap_subscript and isinstance(current, ast.Subscript):
        current = current.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    spelled: str | None = None
    if isinstance(current, ast.Name):
        parts.append(current.id)
        spelled = ".".join(reversed(parts))
    return spelled


def callee_names(call: ast.Call) -> frozenset[str]:
    """return the bare and attribute spellings of a call's callee.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: callee names
    :rtype: frozenset[str]
    """
    callee = call.func
    names: set[str] = set()
    if isinstance(callee, ast.Name):
        names.add(callee.id)
    elif isinstance(callee, ast.Attribute):
        names.add(callee.attr)
    return frozenset(names)


def receiver(call: ast.Call) -> str | None:
    """return the dotted spelling of a method call's receiver.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: receiver spelling, or ``None`` when the call is not a method call
    :rtype: str | None
    """
    callee = call.func
    spelled: str | None = None
    if isinstance(callee, ast.Attribute):
        spelled = dotted(callee.value)
    return spelled


def argument_spellings(call: ast.Call) -> frozenset[str]:
    """return the dotted spelling of every positional and keyword argument value.

    :param call: call to inspect
    :ptype call: ast.Call
    :return: argument spellings
    :rtype: frozenset[str]
    """
    spellings: set[str] = set()
    for argument in [*call.args, *(keyword.value for keyword in call.keywords)]:
        spelled = dotted(argument)
        if spelled is not None:
            spellings.add(spelled)
    return frozenset(spellings)
