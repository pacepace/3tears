---
artifact: build-plan
version: 2
scope: search-extract-off-loop
branch: fix/search-extract-off-loop
depends_on: []
partition: "serial -- one module (packages/search extract.py) and its tests; a second agent would cost more in coordination than it saves"
last_validated: 2026-11-24
---

## Requirements Confidence

**Level:** High

**Why:** a measured production defect with a repro. Discodon reported it, and the owner
approved the work in 3tears ("yes we can update 3tears and submit a PR. Best to have @3tears
handle it, and we just consume here", relayed by the Discodon session on 2026-11-24).

Problem: `extract()` runs its CPU stage on the caller's event loop: it decodes the page,
extracts the text with trafilatura, then runs `trafilatura.metadata.extract_metadata` as a
second full parse, with author extraction and htmldate's extensive date search. When several
extractions run at once in one process, timers fire late and reads die at their bound after
the network has already answered. In production, 47 of 240 reads timed out on Rockchip arm64.

Goals (the three properties Discodon asked for):
1. A caller can move the CPU stage onto an executor it supplies, including a
   `ProcessPoolExecutor`: module-level functions over bytes and str, with picklable results.
   With no executor, the stage runs inline as before.
2. Title and date come from the tree that is already parsed. No second `extract_metadata`
   pass, no author extraction, and no extensive date search.
3. A cut-off or failed parse keeps its honest record. Cancellation propagates, and nothing
   stands in for an unread page.

Out of scope: an executor seam on the `web_fetch` builtin (`agent/tools`) and on the standalone
pipeline. Flagged to the owner as a follow-up, not dropped.

**Decisions:**
- [DECISION: drop htmldate's extensive search, and pass the page's own URL to it | measured on
  the 24 pages in Discodon's repro. The shipped path dates kexp.org/read/2020/3/25/... as
  2010-01-01, and passing the URL gives 2020-03-25. The extensive search alone finds dates on 3
  of the 24 pages, and all 3 are wrong: a citation date in the article body (en-academic,
  2005-05-29), "In 1999/2000" in running text (fodderstompf), and a 2024 that appears nowhere as
  a date in the page (scaruffi). Titles were identical on all 24 | user can veto]
- [DECISION: the executor is one optional keyword, `executor: concurrent.futures.Executor | None`,
  on `extract()` | it is the stdlib's own seam, and `loop.run_in_executor` takes any Executor,
  so a thread pool and a process pool both work with nothing new to learn | user can veto]
- [DECISION: the thin-page frame scan (decode plus `HTMLParser`) runs on the same executor |
  it is the same kind of CPU work on the same bytes | user can veto]

## Chunks

### Chunk 01: extract CPU stage off the loop, single parse

Critic mode: cumulative

Delivers: the `executor` parameter; a module-level page stage (decode, then either text as
written or one lxml parse that serves text, title and date); the frame scan on the executor;
tests for each goal (a process pool run, the loop kept free during a slow stage, cancellation,
the URL date, the metadata-failure note on the new seam); and updated docstrings.

Done when: the search package's tests pass on the pool, lint and the search typecheck are
clean, the benchmark shows the CPU drop on the 24 pages, the cumulative Critic is clean, and
the PR is open into develop.
