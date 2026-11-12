"""Enforcement: the evals package's prose cites no plan, chunk, review or question that lives only in a gitignored file.

The evals package is built from plans kept under ``.prawduct/``, which is gitignored. A comment that says "(Q3)",
"owner ruling Q3", "plan §2.3", "moved at C5b", "chunk 6" or "rev-20261110T032147Z" points every reader who is not
the author at nothing: the plan is not in the clone, and the id names no commit, issue or document they can open. The
reason belongs in the comment itself, said in the present tense.

What is read is prose: a Python module's comments and docstrings (a bare string statement, wherever it sits), the
kit's script and stylesheet comments, and Markdown outside ``tests/fixtures/`` (which holds recorded documents a test
reads as data). A string a test reads as data, "(see C2)" in a document under test, is code and is left alone.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections.abc import Iterator
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVALS = _REPO_ROOT / "packages" / "evals"
_ROOTS = ("src", "tests", "examples")

#: Recorded documents and replies a test reads as data: their text is what a host or a writer produced.
_DATA = Path("tests") / "fixtures"

#: Each citation, by what it points at.
_CITATIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a section of a plan", re.compile(r"plan §")),
    ("the plan by name", re.compile(r"number-forms plan")),
    ("a review id", re.compile(r"rev-\d{8}T\d{6}Z")),
    # A question id however it is cited ("(Q3)", "owner ruling Q3", "Q4 of 2026-11-07"); a quoted "Q3" is a word a
    # check matches, not a citation.
    ("a question id", re.compile(r"(?<![\"'])\bQ\d+\b(?![\"'])")),
    ("a chunk id", re.compile(r"\bC\d{1,2}[a-g]?(-\d)?\b")),
    # A chunk by number ("presentation build, chunk 1"); one of a numbered issue ("#1009 chunk 06") names what a reader
    # can open.
    ("a chunk number", re.compile(r"(?<!\d )\bchunks? \d+\b")),
)

#: A JavaScript string, template or comment, leftmost first, so a ``//`` inside a string is never read as a comment.
#: A regex literal is not told apart; the kit's hold no quote or comment opener.
_JS_TOKENS = re.compile(r"\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'|`(?:\\.|[^`\\])*`|//[^\n]*|/\*.*?\*/", re.S)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def _python_prose(text: str) -> Iterator[tuple[int, str]]:
    """Each line of a module's comments and docstrings, with its line number."""
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for offset, line in enumerate(node.value.value.splitlines()):
                yield node.lineno + offset, line


def _comments(text: str, pattern: re.Pattern[str]) -> Iterator[tuple[int, str]]:
    for match in pattern.finditer(text):
        if match.group().startswith(("//", "/*")):
            first = text.count("\n", 0, match.start()) + 1
            for offset, line in enumerate(match.group().splitlines()):
                yield first + offset, line


def _prose(relative: Path, text: str) -> Iterator[tuple[int, str]]:
    """The prose of one file under the package, by its kind; nothing for a file that holds none."""
    if relative.suffix == ".py":
        yield from _python_prose(text)
    elif relative.suffix == ".js":
        yield from _comments(text, _JS_TOKENS)
    elif relative.suffix == ".css":
        yield from _comments(text, _CSS_COMMENT)
    elif relative.suffix == ".md" and not relative.is_relative_to(_DATA):
        yield from enumerate(text.splitlines(), start=1)


def _citations(relative: Path, text: str) -> Iterator[str]:
    for line, prose in _prose(relative, text):
        for what, pattern in _CITATIONS:
            for match in pattern.finditer(prose):
                yield f"{relative.as_posix()}:{line}: {what}, {match.group()!r}: {prose.strip()}"


def _read_files() -> list[Path]:
    return [
        path
        for root in _ROOTS
        for path in sorted((_EVALS / root).rglob("*"))
        if path.is_file() and path.suffix in (".py", ".js", ".css", ".md")
    ]


def test_evals_prose_cites_no_gitignored_plan() -> None:
    files = _read_files()
    assert len(files) > 1000  # src, tests and examples are all read, not one of them
    found = [
        citation
        for path in files
        for citation in _citations(path.relative_to(_EVALS), path.read_text(encoding="utf-8"))
    ]
    assert not found, (
        "these cite a plan, chunk, review or question that no reader of the clone can open; state the reason "
        "itself, in the present tense:\n  " + "\n  ".join(found)
    )


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        ("src/a.py", "x = 1  # the cap moved at C5b\n", ["a chunk id"]),
        ("src/a.py", '"""Refuses rather than degrades (plan §2.3)."""\n', ["a section of a plan"]),
        ("src/a.py", "def f():\n    '''Ruled on 2026-11-07 (Q11).'''\n", ["a question id"]),
        ("src/a.py", "x = 1\n'''Found by rev-20261110T032147Z-fc0c4c5d.'''\n", ["a review id"]),
        ("tests/a.py", "# the number-forms plan's §2\n", ["the plan by name"]),
        ("src/a.py", "# cents whatever its size (owner ruling Q3, 2026-11-07)\n", ["a question id"]),
        ("src/a.py", "# the money ruling, Q4 of 2026-11-07; the check-name ruling, Q11\n", ["a question id"] * 2),
        ("src/a.py", "# at C10 and C12b-2\n", ["a chunk id"] * 2),
        ("tests/a.py", "# (presentation build, chunk 1), chunks 4\n", ["a chunk number"] * 2),
        ("src/k.js", 'const a=1; // from C6-2\nconst b="//C3";\n', ["a chunk id"]),
        ("src/k.js", "/* C1b\n   (Q4) */ const a=1;\n", ["a chunk id", "a question id"]),
        ("src/k.css", "/* C2 */ .a{background:url(http://x/C3)}\n", ["a chunk id"]),
        ("examples/README.md", "| On | the C5c pages |\n", ["a chunk id"]),
    ],
)
def test_a_citation_in_prose_is_found(name: str, text: str, expected: list[str]) -> None:
    assert [citation.split(": ")[1].split(",")[0] for citation in _citations(Path(name), text)] == expected


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("tests/a.py", 'DOC = "A finding (see C2) under test."\n'),
        ("tests/a.py", "assert check('(Q3)') == []\n"),
        ("src/k.js", 'const s="(Q3) C5b";\n'),
        ("src/k.js", "const s=`C2 ${x}`;\n"),
        ("tests/fixtures/report_docs/a/report.md", "A writer's page about C5b.\n"),
        ("src/a.py", "# CSV C100 C2x ABC1 SC2 Q3x (Q) plan chunked 4\n"),
        ("src/a.py", '"""A case ``q3`` is not the word "Q3"."""\n'),
        ("tests/a.py", "# run live by hand (#1009 chunk 06)\n"),
        ("src/a.json", '{"note": "C5b"}\n'),
    ],
)
def test_code_data_and_lookalikes_are_not_citations(name: str, text: str) -> None:
    assert list(_citations(Path(name), text)) == []
