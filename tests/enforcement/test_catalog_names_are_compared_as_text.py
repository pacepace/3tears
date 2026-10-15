"""enforcement: a catalogue ``name`` column is compared with an array only as ``text``.

YugabyteDB 2026.1.1.1 mis-evaluates a ``name`` column compared with a ``text[]``:
``relname = ANY(ARRAY['a', 'b'])`` fails ``compressed lz4 data is corrupt``, intermittently, once the
server has taken some load, while ``relname::text = ANY(...)`` and ``relname = ANY(...::name[])`` do
not. The platform migration chain's write-mark backstop compared ``information_schema.tables.table_name``
with ``ANY($1::text[])`` and stopped every fresh bring-up on that version. The walker is the shared
one (:mod:`threetears.enforcement.catalog_name_comparison`), which the hub's gate runs too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from threetears.enforcement.catalog_name_comparison import find_catalog_name_violations, violations_in

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_catalogue_name_is_compared_with_an_array_as_text() -> None:
    roots = sorted(path for pattern in ("packages/*/src", "packages/agent/*/src") for path in _REPO_ROOT.glob(pattern))
    offenders = find_catalog_name_violations(roots, relative_to=_REPO_ROOT)
    assert offenders == [], (
        "compare a catalogue name column with an array as text (column::text = ANY(...)); YugabyteDB "
        "2026.1.1.1 fails the name-to-text[] comparison:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    ("sql", "found"),
    [
        ("SELECT 1 FROM information_schema.tables t WHERE t.table_name = ANY($1::text[])", ["table_name"]),
        ("SELECT 1 FROM pg_class WHERE relname = ANY(ARRAY['a'])", ["relname"]),
        # a placeholder naming the catalogue table does not hide a catalogue-only column
        ("SELECT 1 FROM {catalog} c WHERE c.relname = ANY($1)", ["relname"]),
        ("SELECT 1 FROM pg_attribute a WHERE a.attname <> ALL($1)", ["attname"]),
        ("SELECT 1 FROM pg_tables WHERE tablename != SOME($1)", ["tablename"]),
        ("SELECT 1 FROM information_schema.tables t WHERE t.table_name::text = ANY($1)", []),
        ("SELECT 1 FROM pg_class WHERE relname IN ('a', 'b')", []),
        # an application table's own table_name column is not the catalogue's
        ("SELECT id FROM collection_write_marks WHERE table_name = ANY($1::text[])", []),
    ],
)
def test_the_walker_reads_the_shapes_it_must(sql: str, found: list[str]) -> None:
    assert violations_in(sql) == found


def test_a_text_built_across_lines_is_read_whole(tmp_path: Path) -> None:
    module = tmp_path / "queries.py"
    module.write_text(
        'A = "SELECT c.relname FROM pg_catalog.pg_class c " + "WHERE c.relname = ANY($1::text[])"\n'
        'B = " ".join(["SELECT 1 FROM information_schema.columns", "WHERE column_name = ANY($1)"])\n',
        encoding="utf-8",
    )
    assert [entry.split(" ")[-1] for entry in find_catalog_name_violations([tmp_path], relative_to=tmp_path)] == [
        "relname",
        "column_name",
    ]
