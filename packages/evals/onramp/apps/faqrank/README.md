# faqrank

The search box on our invoicing app's help centre (`search.py`). `search(query, k=5)` ranks our help articles
(`data/articles.jsonl`) for what a customer typed and returns the best `k` as `Hit`s (article id, title, score), best
first. How an article is scored is in the module docstring. `Index(articles)` builds the same search over any list of
articles.

`data/queries.jsonl` is a sample of what customers really typed into the box last month. For each one, the support
team marked which articles help and how much (`data/judgments.jsonl`): 3 answers the question, 2 is closely related,
1 is of some use, and an article not listed is no use for that query. A good ranking puts the 3s first and gets the
useful articles into the top five; the team reads results the way a customer does, so the first place matters most.

Tests: `python -m pytest tests` (or `python -m unittest discover tests`) from this directory.
