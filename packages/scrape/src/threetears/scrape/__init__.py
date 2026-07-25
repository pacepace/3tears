"""3tears-scrape — general-purpose, AI-native scraping platform component.

Originally built inside a forecasting application (faidh) as
``src/faidh/scrape/`` and lifted out into this package on 2026-07-15. It was
written domain-agnostic from its first commit specifically so that move
could be a directory move plus import-path updates in the consumer, not a
rewrite: nothing here has ever known what a scraped field *means*, only what
type it was declared as. That application's WARN Act plugin remains the
running example of a consumer throughout this package's docstrings -- it is
not part of this package, and it is the only place domain meaning lives.

The discipline that made the lift cheap is still load-bearing and still
observed: no module here imports a consuming application's config, store, or
entities. Every real dependency (collections, drivers, API keys, sidecar
URLs) is passed in by the caller.
"""

from __future__ import annotations

__all__: list[str] = []
