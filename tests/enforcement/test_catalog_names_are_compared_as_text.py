"""enforcement: a catalogue ``name`` column is compared with an array only as ``text``.

YugabyteDB 2026.1.1.1 mis-evaluates a ``name`` column compared with a ``text[]``:
``relname = ANY(ARRAY['a', 'b'])`` fails ``compressed lz4 data is corrupt``, intermittently, once the
server has taken some load, while ``relname::text = ANY(...)`` and ``relname = ANY(...::name[])`` do
not. The platform migration chain's write-mark backstop compared ``information_schema.tables.table_name``
(a domain over ``name``) with ``ANY($1::text[])`` and stopped every fresh bring-up on that version.

Every SQL string in the packages that reads the catalogue (``information_schema`` or a ``pg_``
catalogue) must compare its name columns with ``ANY(...)`` as ``text`` (``t.table_name::text = ANY(...)``).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: a string reads the catalogue when it names one of these
_CATALOGUE = re.compile(
    r"\binformation_schema\b|\bpg_(?:class|namespace|trigger|proc|attribute|type|index|constraint)\b"
)

#: the catalogue's ``name``-typed columns (``information_schema``'s are domains over ``name``), compared
#: with ``ANY(`` directly: no cast between the column and the ``=``
_UNCAST = re.compile(
    r"\b(?:\w+\.)?(relname|nspname|tgname|proname|attname|typname|conname|table_name|table_schema|column_name)"
    r"\s*=\s*ANY\s*\(",
    re.IGNORECASE,
)


def violations_in(sql: str) -> list[str]:
    """the uncast name comparisons in one SQL string that reads the catalogue.

    :param sql: the SQL text
    :ptype sql: str
    :return: each offending column
    :rtype: list[str]
    """
    found: list[str] = []
    if _CATALOGUE.search(sql):
        found = [match.group(1) for match in _UNCAST.finditer(sql)]
    return found


def _strings(path: Path) -> Iterator[tuple[int, str]]:
    """every string literal in a module, f-string text included, with its line.

    :param path: the module
    :ptype path: Path
    :return: ``(line, text)`` pairs
    :rtype: Iterator[tuple[int, str]]
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(part.value for part in node.values if isinstance(part, ast.Constant))
            yield node.lineno, text


def _sources() -> Iterator[Path]:
    for pattern in ("packages/*/src", "packages/agent/*/src"):
        for root in sorted(_REPO_ROOT.glob(pattern)):
            yield from sorted(root.rglob("*.py"))


def test_every_catalogue_name_is_compared_with_an_array_as_text() -> None:
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}:{line} {column}"
        for path in _sources()
        for line, text in _strings(path)
        for column in violations_in(text)
    ]
    assert offenders == [], (
        "compare a catalogue name column with ANY(...) as text (column::text = ANY(...)); YugabyteDB "
        "2026.1.1.1 fails the name-to-text[] comparison:\n" + "\n".join(offenders)
    )


def test_the_gate_reads_the_shapes_it_must() -> None:
    reads = "SELECT 1 FROM information_schema.tables t WHERE t.table_name = ANY($1::text[])"
    assert violations_in(reads) == ["table_name"]
    assert violations_in("SELECT 1 FROM pg_class WHERE relname = ANY(ARRAY['a'])") == ["relname"]
    assert violations_in("SELECT 1 FROM information_schema.tables t WHERE t.table_name::text = ANY($1)") == []
    # an application table's own table_name column is not the catalogue's
    assert violations_in("SELECT id FROM collection_write_marks WHERE table_name = ANY($1::text[])") == []
