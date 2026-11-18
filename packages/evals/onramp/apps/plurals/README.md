# plurals

The notification texts in our app are templates, so the copy team can change the words without a release, and
`messages.py` fills them in: `render(template, values)` replaces each field with its value, and a plural field picks
the right English form for the number (`{count|one file|# files}`, with an optional zero form in front). The syntax,
how numbers are written, and what happens to a field with no value are in the module docstring.

Every template goes past the copy team before it ships. `data/strings.jsonl` is their review sheet from the last
release: the string's key, the template, the values they tried it with, and the text they approved (`approved`). For
each line, `render(template, values)` should give exactly the approved text.

Tests: `python -m unittest discover tests` from this directory.
