# nutrilabel

Front-of-pack nutrition labels for our own-brand range, printed from each product's spec sheet. `label(spec)` gives
the label line; `labels_file(path)` does a file of spec sheets, one JSON sheet per line. A sheet has the product's
`sku`, `name`, `form` (`food` or `drink`), shelf `category`, and its values per 100 g (food) or 100 ml (drink):
`energy_kj`, `fat`, `saturates`, `sugars` and `salt` in grams.

The label's layout and colour bands are in the module docstring of `label.py`. Each of fat, saturates, sugars and
salt is LOW, MED or HIGH against the government's front-of-pack thresholds. Drinks have their own thresholds, per
100 ml, half the food ones (salt's LOW line is the same): for sugars, a drink is LOW at 2.5 g or less and HIGH above
11.25 g, where a food is LOW at 5 g or less and HIGH above 22.5 g.

`data/approved.jsonl` is this season's spec sheets, each with `approved`, the label line the compliance team signed
off for print. The labels the script prints must match the signed-off text exactly.

Tests: `python -m unittest discover tests` from this directory.
