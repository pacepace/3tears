"""Guard: exactly one module of the engine renders the significance vocabulary.

A source walk (``rglob`` + ``ast.parse``) over the package rather than a test of each surface,
because the renderer that matters is the next one, in a module nobody thought to test.
"""

from __future__ import annotations

import ast
from pathlib import Path


from threetears.evals.analysis import significance as _significance


def test_no_module_writes_the_reader_facing_significance_words_for_itself():
    """`format_significance` claims to be the only server-side renderer; keep it true.

    The claim went into its docstring while a fourth hand-written copy was still
    live in the MCP history table, printing a bare verdict for a row that carries
    no statistic. The grep that would have caught it is this: the reader-facing
    words are declared once, and a module spelling one of them itself is by
    definition a second renderer.

    Field NAMES are not labels — ``row.get("significant")`` reads a key and makes
    no claim to a reader — so only the two multi-word labels are looked for, which
    is the pair a hand-written branch cannot avoid spelling.
    """
    labels = {_significance.NOT_SIGNIFICANT_LABEL, _significance.NOT_TESTED_LABEL}
    owner = Path(_significance.__file__).resolve()
    offenders: dict[str, list[int]] = {}
    for path in sorted(Path(_significance.__file__).resolve().parents[1].rglob("*.py")):
        if path == owner:
            continue
        source = path.read_text(encoding="utf-8")
        # Cheap text filter first: parsing every module in the package to find the
        # two or three that mention the vocabulary at all is seconds of work for
        # an answer a substring search settles.
        if not any(label in source for label in labels):
            continue
        tree = ast.parse(source)
        # Docstrings quote the vocabulary to explain it, which is the opposite of
        # rendering it — every one of them points AT the shared renderer.
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        hits = sorted(
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value in labels and id(node) not in docstrings
        )
        if hits:
            offenders[str(path)] = hits

    assert not offenders, f"a second significance renderer: {offenders} — call significance.format_significance instead"
