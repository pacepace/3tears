"""Find free text written after an indent with only its first line indented.

The enforcement half of :mod:`threetears.observe.text_block`. A value whose
name says it is free text (an excerpt, a snippet, a summary, a body, a reason)
interpolated into a line that starts with an indent or a list marker --
``f"      {excerpt}"``, ``f"- {summary}"``, ``f"1. {title}"``, ``"   " + snippet``
-- keeps the item's place on its first line only. Such a value goes through ``under`` (every line indented) or
``one_line`` (one line), or carries a ``# one line: <why>`` comment on its line
or the line above, naming why it cannot hold a line break.

A log call, a raised error, a database statement and a docstring are not text
a model reads as a list, and are not looked at.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path

from threetears.enforcement.common import iter_python_files, parse_python_file

__all__ = ["FREE_TEXT", "FREE_TEXT_WORDS", "MARKER", "unbroken_text", "unbroken_text_in"]

#: A name that says its value is text someone wrote, and so can hold line breaks.
FREE_TEXT = re.compile(
    r"(?i)(excerpt|snippet|summary|content|text|body|description|desc|preview|reason|error|note|message|detail|"
    r"prose|answer|reply|said|words|output|title|label|headline|gist|plan|step|quote)"
)

#: A name that is free text only as a whole word: ``shown`` and ``value`` are, ``deadline`` is no ``line``.
FREE_TEXT_WORDS = re.compile(r"(?i)(?:^|_)(value|shown|what|cut|said|line)(?:_|$)")

#: The comment that says a matched value is one line by what it is.
MARKER = "# one line:"

#: The calls that make a text safe to put after an indent. A fence opens and closes on lines of its
#: own, so what it holds is marked off however many lines it has.
_SAFE = frozenset({"one_line", "under", "untrusted_fence", "len", "int", "float", "round", "format_offload_handle"})
#: A name that is an id, whatever words it holds: ``message_id_end`` is not a message.
_AN_ID = re.compile(r"(?:^|_)ids?(?:_|$)")
#: The calls whose text is a log line or an error, not a list a model reads.
_NOT_READ = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log", "print", "fail", "skip"}
    # and a statement sent to a database
    | {"fetch", "fetchrow", "fetchval", "execute", "executemany"}
)
#: A line that is an item of a list or sits under one: indented, or opened by a list marker.
_INDENTED_LINE = re.compile(r"(?:^|\n)(?: {2,}|\t| *(?:[-*\u2022]|\d+\.) )[^\n]*$")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    return ""


def _what(value: ast.expr) -> list[str]:
    """The names that say what a value is: ``hit.excerpt`` is an excerpt, ``note.path`` a path.

    An attribute by its own name, a name by itself, ``row["body"]`` by its key,
    ``a or b`` and ``a if c else b`` by either side, and a call by what it was
    handed (``str(x)`` and ``cut_whole(x, 80)`` are ``x``). A call that makes text
    safe names nothing.
    """
    if isinstance(value, ast.Attribute):
        # ``x.value`` is what ``x`` is: ``kind.value`` is an enum's name, not a text
        return _what(value.value) if value.attr == "value" else [value.attr]
    if isinstance(value, ast.Name):
        return [value.id]
    if isinstance(value, ast.Subscript):
        key = value.slice
        return [key.value] if isinstance(key, ast.Constant) and isinstance(key.value, str) else _what(value.value)
    if isinstance(value, ast.BoolOp):
        return [name for operand in value.values for name in _what(operand)]
    if isinstance(value, ast.IfExp):
        return _what(value.body) + _what(value.orelse)
    if isinstance(value, ast.Call):
        name = _call_name(value)
        if name in _SAFE:
            return []
        # a method of the text keeps its line breaks (``excerpt.strip()``), and so does a call that
        # shortens it (``cut_whole(excerpt, 200)``): a call is what it was handed, not what it is
        # called -- ``_message_head(row)`` builds a head line, ``cut_whole(body)`` is still the body
        receiver = _what(value.func.value) if isinstance(value.func, ast.Attribute) else []
        return [*receiver, *(_what(value.args[0]) if value.args else [])]
    return []


def _free(value: ast.expr) -> bool:
    """Whether an interpolated expression is free text that did not pass a call that makes it safe."""
    return any(
        (FREE_TEXT.search(name) or FREE_TEXT_WORDS.search(name)) and not _AN_ID.search(name) for name in _what(value)
    )


def _after_indent(node: ast.JoinedStr) -> list[ast.FormattedValue]:
    """The values of an f-string that sit on a line it starts with an indent."""
    found: list[ast.FormattedValue] = []
    text = ""
    for part in node.values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            text += part.value
        elif isinstance(part, ast.FormattedValue):
            # ``{x!r}`` is a repr, which writes a line break as \\n: one line whatever x holds
            if _INDENTED_LINE.search(text) and part.conversion != ord("r"):
                found.append(part)
            # a digit: ``f"{n}. {title}"`` is a numbered item, its number interpolated
            text += "0"
    return found


def unbroken_text(source: str) -> list[int]:
    """The lines of ``source`` that put free text after an indent with no ``under``, ``one_line`` or marker.

    :param source: a Python module's source
    :ptype source: str
    :return: the line numbers, ascending
    :rtype: list[int]
    """
    return _unbroken_lines(ast.parse(source), source.splitlines())


def _unbroken_lines(tree: ast.Module, lines: list[str]) -> list[int]:
    """The lines of a parsed module that put free text after an indent with no ``under``, ``one_line`` or marker.

    :param tree: the module, parsed
    :ptype tree: ast.Module
    :param lines: its source lines, for the ``# one line:`` marker
    :ptype lines: list[str]
    :return: the line numbers, ascending
    :rtype: list[int]
    """
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    def _read_as_a_list(node: ast.AST) -> bool:
        current: ast.AST | None = node
        while current is not None:
            # a text handed to one_line or under is made safe by the call around it
            if isinstance(current, ast.Raise) or _call_name(current) in _NOT_READ | _SAFE:
                return False
            current = parents.get(current)
        return True

    def _one_of_its_lines(value: ast.expr) -> bool:
        # ``"  " + line for line in body.splitlines()`` indents every line: the name a loop over
        # ``splitlines()`` binds is one line of the text, whatever it is called
        if not isinstance(value, ast.Name):
            return False
        current: ast.AST | None = parents.get(value)
        while current is not None:
            loops: list[tuple[ast.expr, ast.expr]] = []
            if isinstance(current, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                loops = [(gen.target, gen.iter) for gen in current.generators]
            elif isinstance(current, (ast.For, ast.AsyncFor)):
                loops = [(current.target, current.iter)]
            for target, source in loops:
                if isinstance(target, ast.Name) and target.id == value.id:
                    return _call_name(source) == "splitlines"
            current = parents.get(current)
        return False

    def _marked(line: int) -> bool:
        return any(MARKER in lines[n - 1] for n in (line, line - 1) if 0 < n <= len(lines))

    hits: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.JoinedStr, ast.BinOp)):
            continue
        values: list[ast.expr] = []
        if isinstance(node, ast.JoinedStr):
            values = [part.value for part in _after_indent(node)]
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str) and _INDENTED_LINE.search(left.value):
                values = [node.right]
        flagged = [value for value in values if _free(value) and not _one_of_its_lines(value)]
        if flagged and _read_as_a_list(node) and not _marked(node.lineno):
            hits.add(node.lineno)
    return sorted(hits)


def unbroken_text_in(roots: Iterable[Path]) -> list[str]:
    """Every such line under ``roots``, as ``path:line: code``.

    The files are the ones every enforcement walker reads (:func:`iter_python_files`), decoded as
    utf-8 and parsed once per run through the shared parse cache; a file that does not parse is
    reported through ``note_unscanned`` and skipped.

    :param roots: directories whose ``*.py`` files are read
    :ptype roots: Iterable[Path]
    :return: one entry per line found
    :rtype: list[str]
    """
    found: list[str] = []
    for root in roots:
        for path in iter_python_files(root):
            tree = parse_python_file(path)
            if tree is None:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            found.extend(f"{path}:{n}: {lines[n - 1].strip()}" for n in _unbroken_lines(tree, lines))
    return found
