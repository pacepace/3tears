# docnames

Names for the restaurant group's new document store (`naming.py`). Every upload is filed under
`stored_name(title, uploaded, extension)`: the upload date, a slug of the title, and the extension, for example
`2026-03-04-spring-menu-costings.pdf`. The rules for the slug are in the module docstring.

We are moving last year's documents over from the old archive, and links to them are all over our emails and
spreadsheets, so every document has to keep the name the archive gave it. `data/archive_index.jsonl` is the archive's
index: each document's title, upload date and extension as entered, and the name the archive filed it under
(`archive_name`). The archive used the same naming rules.

Tests: `python -m pytest tests` (or `python -m unittest discover tests`) from this directory.
