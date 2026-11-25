# mdtable

The staff handbook is Markdown in a repository, and its tables (the on-call rota, who to ask, expense limits, time
off by office) are edited by people who don't want to line up pipes by hand. `mdtable.py` writes them:
`format_table(header, rows, align)` returns the table's Markdown with every column padded to the same width, so the
source reads like the rendered page. The rules (widths, alignment, escaping, what counts as a character) are in the
module docstring. Names on the rota come from the HR system and from people's own email signatures, so plenty of them
are accented, and they must line up like any other.

Until now the docs team aligned the tables by hand. `data/tables.jsonl` is every table in the handbook as it stands:
the cells as the table's editor supplied them (`header`, `rows`, and `align` where the table sets one) and the
Markdown the docs team signed off (`expected`). The formatter should write exactly that Markdown for each table.

Tests: `python -m unittest discover tests` from this directory.
