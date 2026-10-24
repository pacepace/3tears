"""A tool states its notes before its body: "this is a part", "results are partial", "read by OCR".

A reader that keeps only the head of a long result (a prompt budget, a stored
copy, a digest) keeps what comes first. A note written after the body is the
first thing lost, and what is lost is exactly the sentence saying the body is
incomplete: a page cut at its bound was discussed as if whole, and a recall that
lost a branch was presented as everything there was. A consumer cannot repair
that afterwards without parsing a tool's prose to find its last line, so the
tool states its notes first, in its own text.

Four shapes put a note after a body, and this walks every package's ``src`` for
each:

- a text window rendered without ``note_first=True``
  (``threetears.agent.tools.text_window.TextWindow.rendered``);
- a note glued on behind something: a literal that opens with a line break and
  then a note's words;
- a note appended to a list of lines after lines that are not notes;
- a note built into a string after its body: an f-string, a ``+`` chain or a
  ``.join([...])`` whose note piece comes after a piece that is not. In a string
  a note piece is a literal opening like a note, a call to a note-maker (a
  function or method whose name says ``note``, such as ``window.note(...)``),
  or a name assigned from one -- not every name containing "note", which in a
  string is as often a person's note or an error's detail.

A window's note placed by hand (``.note(...)``) is listed in ``HAND_PLACED`` with
where it goes. A literal that reads like a note and is not part of a tool's
result is listed in ``NOT_A_TOOL_RESULT`` with what it is. A string that puts
its note after something on purpose is listed in ``NOTE_BUILT_AFTER_ON_PURPOSE``
with why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCES = sorted(p for p in (WORKSPACE / "packages").rglob("src") if p.is_dir() and "/tests/" not in str(p))

#: Where the window and its two renderings are defined.
WINDOW_MODULE = "packages/agent/tools/src/threetears/agent/tools/text_window.py"

#: How a note opens, after any bracket or emphasis mark.
NOTE = re.compile(
    r"^[\(\[\*_]*(partial results|note\b|warning\b|caution\b|truncated\b|cut short|only the first|showing the first)",
    re.IGNORECASE,
)

#: ``file:function`` -> where the hand-placed window note goes. A new one is added here with its place.
HAND_PLACED: dict[str, str] = {
    "packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py:_handle_document": (
        "in the prompt an analyser model reads, not in a tool result"
    ),
    "packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py:_handle_audio_video": (
        "first, above the transcript's heading"
    ),
    "packages/agent/tools/src/threetears/agent/tools/document.py:parse_document_tool": (
        "first, above the document's facts and its text"
    ),
}

#: ``file:opening`` -> what the literal is, when it is not a note in a tool's result.
NOT_A_TOOL_RESULT: dict[str, str] = {
    "packages/agent/wake/src/threetears/agent/wake/dispatch.py:\n[truncated:": (
        "the byte-cap mark on a wake's context_from block, which is prompt material built by dispatch"
    ),
    "packages/channels/src/threetears/channels/formatting.py: [truncated]": (
        "the mark on a Discord embed cut to the platform's length limit, read by a person"
    ),
}


#: ``file:function`` -> why a string built there puts its note after something. A new one is added with its reason.
NOTE_BUILT_AFTER_ON_PURPOSE: dict[str, str] = {
    "packages/search/src/threetears/search/bind_answer.py:_head": (
        "the one-line attribution AR-G2 requires first, naming whose synthesis follows; the note comes right "
        "under it, before the answer text and its sources"
    ),
    "packages/agent/tools/src/threetears/agent/tools/builtin/calculator.py:_evaluate": (
        "a one-line result whose parenthetical names its units, on the same line as the number it qualifies, "
        "so no head of the result keeps the number without it"
    ),
    "packages/agent/tools/src/threetears/agent/tools/builtin/analyze_media.py:_handle_document": (
        "the prompt an analyser model reads, not a tool result: the window note follows the document fence it "
        "describes, as HAND_PLACED records"
    ),
}


def _opening(node: ast.AST | None) -> str | None:
    """The literal text a value opens with, up to its first placeholder."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = ""
        for value in node.values:
            if not isinstance(value, ast.Constant):
                break
            out += str(value.value)
        return out
    return None


def _is_note(node: ast.AST | None) -> bool:
    """Whether a value is a note: a literal that opens like one, a list of them, or a name saying so."""
    if isinstance(node, ast.List):
        return bool(node.elts) and all(_is_note(e) for e in node.elts)
    if isinstance(node, ast.Name):
        return "note" in node.id.lower()
    text = _opening(node)
    return text is not None and bool(NOTE.match(text))


def _functions(tree: ast.AST) -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef | ast.FunctionDef)]


def _owner(tree: ast.AST, node: ast.AST) -> str:
    """The innermost function holding ``node``, by name; ``<module>`` outside any."""
    best, span = "<module>", None
    for fn in _functions(tree):
        end = fn.end_lineno or fn.lineno
        if fn.lineno <= getattr(node, "lineno", 0) <= end and (span is None or end - fn.lineno < span):
            best, span = fn.name, end - fn.lineno
    return best


def _trailing_windows(tree: ast.AST, where: str) -> list[str]:
    """Every ``.rendered(...)`` that does not put the note first."""
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "rendered"):
            continue
        first = next((kw.value for kw in node.keywords if kw.arg == "note_first"), None)
        if not (isinstance(first, ast.Constant) and first.value is True):
            hits.append(f"{where}:{node.lineno}: a window rendered without note_first=True")
    return hits


def _hand_placed(tree: ast.AST, where: str) -> list[str]:
    """Every window note a function places by hand, as ``file:function``."""
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "note"):
            continue
        if any(kw.arg in {"tool", "how"} for kw in node.keywords):
            found.append(f"{where}:{_owner(tree, node)}")
    return found


def _glued_on(tree: ast.AST, where: str) -> list[str]:
    """Every literal that opens with a line break and then a note's words: a note behind something."""
    hits = []
    inside = {id(part) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for part in node.values}
    for node in ast.walk(tree):
        if id(node) in inside:
            continue  # a piece of an f-string, read with the whole of it
        text = _opening(node) if isinstance(node, ast.Constant | ast.JoinedStr) else None
        if not text or text.lstrip(" \n") == text or not NOTE.match(text.lstrip(" \n")):
            continue
        if any(f"{where}:{text}".startswith(key) for key in NOT_A_TOOL_RESULT):
            continue
        hits.append(f"{where}:{node.lineno}: a note glued on behind what it is about: {text.strip()[:60]!r}")
    return hits


def _appended_late(tree: ast.AST, where: str) -> list[str]:
    """Every note appended to a list of lines after a line that is not a note."""
    hits = []
    for fn in _functions(tree):
        grown: dict[str, list[tuple[int, bool]]] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign | ast.AnnAssign) and isinstance(node.value, ast.List):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        for element in node.value.elts:
                            grown.setdefault(target.id, []).append((node.lineno, _is_note(element)))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"append", "extend"}
                and isinstance(node.func.value, ast.Name)
                and node.args
            ):
                grown.setdefault(node.func.value.id, []).append((node.lineno, _is_note(node.args[0])))
        for name, lines in grown.items():
            ordered = sorted(lines)
            for position, (lineno, is_note) in enumerate(ordered):
                if is_note and any(not earlier for _, earlier in ordered[:position]):
                    hits.append(f"{where}:{lineno}: a note appended to {name!r} after lines that are not notes")
    return hits


def _pieces(node: ast.AST) -> list[ast.AST] | None:
    """The pieces a string is built from, in order, when it is built: an f-string, a ``+`` chain, a joined list.

    :return: the pieces, or None when the node builds no string from pieces
    """
    if isinstance(node, ast.JoinedStr):
        return [part.value if isinstance(part, ast.FormattedValue) else part for part in node.values]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        flat: list[ast.AST] = []
        for side in (node.left, node.right):
            flat.extend(_pieces(side) or [side])
        return flat
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and node.args
        and isinstance(node.args[0], ast.List)
    ):
        return list(node.args[0].elts)
    return None


def _piece_kind(piece: ast.AST, made: set[str]) -> str:
    """``note``, ``blank`` (whitespace only) or ``body``, for one piece of a built string."""
    if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
        text = piece.value.lstrip(" \n")
        if not text:
            return "blank"
        return "note" if NOTE.match(text) else "body"
    if isinstance(piece, ast.Name) and piece.id in made:
        return "note"
    return "note" if _makes_note(piece) else "body"


def _makes_note(node: ast.AST) -> bool:
    """Whether a value is a call to a note-maker: a function or method whose name says ``note``."""
    if not isinstance(node, ast.Call):
        return False
    callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
    return "note" in callee.lower()


def _built_late(tree: ast.AST, where: str) -> list[str]:
    """Every string built with a note piece after a piece that is not a note."""
    made = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and _makes_note(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    hits = []
    for node in ast.walk(tree):
        pieces = _pieces(node)
        if not pieces:
            continue
        seen_body = False
        for piece in pieces:
            kind = _piece_kind(piece, made)
            if kind == "note" and seen_body:
                if f"{where}:{_owner(tree, node)}" not in NOTE_BUILT_AFTER_ON_PURPOSE:
                    hits.append(f"{where}:{node.lineno}: a note built into a string after what it is about")
                break
            seen_body = seen_body or kind == "body"
    return hits


def _findings(source: str, where: str) -> list[str]:
    tree = ast.parse(source)
    windows = [] if where == WINDOW_MODULE else _trailing_windows(tree, where)
    built = [] if where == WINDOW_MODULE else _built_late(tree, where)
    found = windows + _glued_on(tree, where) + _appended_late(tree, where) + built
    # one finding per line: a nested function is walked with the one that holds it, and one line can
    # match two shapes (a glued literal is also a string built late)
    by_line: dict[str, str] = {}
    for hit in found:
        by_line.setdefault(":".join(hit.split(":", 2)[:2]), hit)
    return list(by_line.values())


def _each_source() -> list[tuple[str, str]]:
    return [
        (path.read_text(encoding="utf-8"), str(path.relative_to(WORKSPACE)))
        for root in SOURCES
        for path in sorted(root.rglob("*.py"))
    ]


def test_no_tool_writes_a_note_after_its_body() -> None:
    hits = [hit for source, where in _each_source() for hit in _findings(source, where)]
    assert not hits, (
        "these write a note after the body it is about, where a reader that keeps only the head of a result "
        "loses it. State the note first (TextWindow.rendered(note_first=True), or append it before the body), "
        "or add the literal to NOT_A_TOOL_RESULT with what it is:\n" + "\n".join(hits)
    )


def test_every_hand_placed_window_note_is_listed_with_its_place() -> None:
    found = sorted(
        {
            site
            for source, where in _each_source()
            if where != WINDOW_MODULE
            for site in _hand_placed(ast.parse(source), where)
        }
    )
    assert found == sorted(HAND_PLACED), (
        "a window's note is placed by hand (.note(tool=...) or .note(how=...)) somewhere HAND_PLACED does not "
        "list, or HAND_PLACED lists one that is gone. Render with rendered(note_first=True), or list the site "
        f"with where its note goes.\nfound: {found}\nlisted: {sorted(HAND_PLACED)}"
    )


def test_the_tables_name_what_is_there() -> None:
    for key in NOT_A_TOOL_RESULT:
        file, opening = key.split(":", 1)
        path = WORKSPACE / file
        assert path.is_file() and opening.strip() in path.read_text(encoding="utf-8"), f"names what is gone: {key}"
    for key in NOTE_BUILT_AFTER_ON_PURPOSE:
        file, function = key.split(":", 1)
        path = WORKSPACE / file
        assert path.is_file() and f"def {function}(" in path.read_text(encoding="utf-8"), f"names what is gone: {key}"
    assert len(SOURCES) > 20, f"the walk found only {SOURCES}"


def test_the_walk_has_teeth() -> None:
    """Each of the four shapes is found, and each of their note-first forms is not."""
    late = (
        "def tool(window, found, degraded):\n"
        "    parts = []\n"
        "    parts.append(f'Found {found} memories:')\n"
        "    if degraded:\n"
        "        parts.append(f'(Partial results: {degraded} failed, so more may exist.)')\n"
        "    text = window.rendered(tool='web_fetch')\n"
        "    stored = window.rendered(how='kept as chunks', note_first=False)\n"
        "    body = text + f'\\n\\n[Truncated at {found} characters]'\n"
        "    lines = ['**Title:** x']\n"
        "    lines.append('**Note:** OCR was used for some pages')\n"
        "    return parts, stored, body, lines\n"
    )
    assert sorted(int(h.split(":")[1]) for h in _findings(late, "x.py")) == [5, 6, 7, 8, 10]
    first = (
        "def tool(window, found, degraded):\n"
        "    sections = []\n"
        "    if degraded:\n"
        "        sections.append([f'(Partial results: {degraded} failed, so more may exist.)'])\n"
        "    sections.append([f'Found {found} memories:'])\n"
        "    text = window.rendered(tool='web_fetch', note_first=True)\n"
        "    lines = []\n"
        "    window_note = window.note(tool='parse_document')\n"
        "    lines.append(window_note)\n"
        "    lines.append('**Note:** OCR was used for some pages')\n"
        "    lines.append('**Title:** x')\n"
        "    return sections, text, lines\n"
    )
    assert _findings(first, "x.py") == []
    assert _hand_placed(ast.parse(first), "x.py") == ["x.py:tool"]
    built_late = (
        "def tool(body, window, candidates):\n"
        "    footnotes = _footnotes(candidates)\n"
        "    joined = f'{body}\\n\\n{footnotes}' if footnotes else body\n"
        "    added = body + '\\n\\n' + window.note(tool='t')\n"
        "    listed = '\\n\\n'.join([body, _notes(candidates)])\n"
        "    inline = f'{body}\\n\\n(Partial results: two engines failed.)'\n"
        "    return joined, added, listed, inline\n"
    )
    assert sorted(int(h.split(":")[1]) for h in _findings(built_late, "x.py")) == [3, 4, 5, 6]
    built_first = (
        "def tool(body, window, candidates, note, notes_file):\n"
        "    footnotes = _footnotes(candidates)\n"
        "    joined = f'{footnotes}\\n\\n{body}' if footnotes else body\n"
        "    added = window.note(tool='t') + '\\n\\n' + body\n"
        "    users = f'{body} ({note})'\n"
        "    missing = f'the release notes {notes_file} do not exist'\n"
        "    return joined, added, users, missing\n"
    )
    assert _findings(built_first, "x.py") == [], "a note stated first, or a name that is not a made note, passes"
    assert _findings("x = window.rendered(tool='t')\n", WINDOW_MODULE) == [], "the window's own module defines both"
