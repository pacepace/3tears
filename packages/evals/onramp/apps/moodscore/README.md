# moodscore

The sentiment score on our feedback dashboard (`sentiment.py`). Reviews come in from the ordering app and the
dashboard shows how warm or cold each one reads: `score(text)` is a number from -1 (as cold as it gets) to 1 (as
warm), and `mood(text)` names its side (`positive`, `negative` or `neutral`). The dashboard averages scores by week and
by branch, so the number matters, not just its side. `raw_score(text)` gives the sum before it is scaled, and
`tokens(text)` the words it reads.

The scoring rules are in the module docstring: a lexicon of weighted words (`data/lexicon.tsv`), negators that turn
round the first sentiment word after them, intensifiers, and clause breaks.

Before we put the score on the dashboard, three people on the customer team each rated a sample of real reviews from
-1 to 1, by hand; `data/rated_reviews.jsonl` has each review and the panel's mean rating (`panel`). Three people do not
agree to the second decimal place, so we count a score as agreeing with the panel when it is within 0.25 of the
panel's mean. The score should agree with the panel on every review in the sample, and the closer the better.

Tests: `python -m unittest discover tests` from this directory.
