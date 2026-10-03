"""A tool that did not do what it was asked returns ``[TOOL ERROR]``, or every caller records it as done.

A consumer reads a tool's failure only from that prefix, the platform's own
convention (``threetears.agent.tools.utils.tool_error``): any other text is a call
that worked. This walks every tool in every package's ``src``: each function
decorated ``@tool`` and each handed to ``StructuredTool.from_function``, and every
string it returns that opens with a refusal's words. An answer that only looks
like one goes in ``ANSWERS``, with why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
SOURCES = sorted(p for p in (WORKSPACE / "packages").rglob("src") if p.is_dir() and "/tests/" not in str(p))

#: How a refusal opens. A line that opens this way and is not one belongs in ANSWERS.
REFUSAL = re.compile(r"^(No |Nothing|Invalid|Not |Cannot|Can't|Failed|Import failed|Unknown|Only |Could not|Unable)")

#: ``file:opening`` -> why it is an answer and not a refusal.
ANSWERS: dict[str, str] = {
    "packages/agent/intention/src/threetears/agent/intention/tools.py:No open wants to raise now.": "a listing that is empty",
    "packages/agent/memory/src/threetears/agent/memory/tools.py:No relevant memories or documents found": "a search that ran",
    "packages/agent/memory/src/threetears/agent/memory/tools.py:No passages matched.": "a search that ran",
    "packages/agent/skills/src/threetears/agent/skills/tools.py:No skills available.": "a listing that is empty",
    "packages/agent/tools/src/threetears/agent/tools/relevance.py:No matching tools found.": "a search that ran",
    "packages/agent/tools/src/threetears/agent/tools/todo.py:No todos in this conversation.": "a listing that is empty",
}


def _opening(node: ast.AST | None) -> str | None:
    """The literal text a returned value opens with, up to its first placeholder."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = ""
        for value in node.values:
            if not isinstance(value, ast.Constant):
                break
            out += str(value.value)
        return out
    if isinstance(node, ast.Tuple) and node.elts:
        return _opening(node.elts[0])
    if isinstance(node, ast.BinOp):
        return _opening(node.left)
    return None


def _tool_functions(tree: ast.AST) -> list[ast.AsyncFunctionDef | ast.FunctionDef]:
    """Every function in a module that is a tool: decorated ``@tool``, or handed to ``from_function``."""
    handed: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_function":
            for kw in node.keywords:
                if kw.arg in {"coroutine", "func"} and isinstance(kw.value, ast.Name):
                    handed.add(kw.value.id)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        decorated = any(
            (isinstance(t, ast.Name) and t.id == "tool") or (isinstance(t, ast.Attribute) and t.attr == "tool")
            for t in (d.func if isinstance(d, ast.Call) else d for d in node.decorator_list)
        )
        if decorated or node.name in handed:
            found.append(node)
    return found


def _plain_refusals(source: str, where: str) -> list[str]:
    hits = []
    for fn in _tool_functions(ast.parse(source)):
        for node in ast.walk(fn):
            if not isinstance(node, ast.Return):
                continue
            text = _opening(node.value)
            if not text or not REFUSAL.match(text) or text.startswith("[TOOL ERROR]"):
                continue
            if any(f"{where}:{text}".startswith(key) for key in ANSWERS):
                continue
            hits.append(f"{where}:{node.lineno}: {text[:80]!r}")
    return hits


def test_every_refusal_a_tool_returns_is_a_tool_error() -> None:
    hits = []
    for root in SOURCES:
        for path in sorted(root.rglob("*.py")):
            hits += _plain_refusals(path.read_text(), str(path.relative_to(WORKSPACE)))
    assert not hits, (
        "these return a refusal as plain text, which the turn records as done: start it with [TOOL ERROR], "
        "or add it to ANSWERS with why it is an answer:\n" + "\n".join(hits)
    )


def test_the_walk_has_teeth() -> None:
    """A decorated tool and a from_function tool, each refusing in plain text, are both found."""
    source = (
        "from langchain_core.tools import tool, StructuredTool\n"
        "@tool('a')\n"
        "async def a(x: str) -> str:\n"
        "    return f'Nothing changed: {x}'\n"
        "async def b() -> str:\n"
        "    return 'Invalid id'\n"
        "t = StructuredTool.from_function(coroutine=b, name='b', description='b')\n"
        "@tool('c')\n"
        "async def c() -> str:\n"
        "    return '[TOOL ERROR] Nothing changed'\n"
    )
    assert [h.split(":")[1] for h in _plain_refusals(source, "x.py")] == ["4", "6"]
    for key in ANSWERS:
        path = WORKSPACE / key.split(":", 1)[0]
        assert path.is_file() and key.split(":", 1)[1] in path.read_text(), f"ANSWERS names what is not there: {key}"
