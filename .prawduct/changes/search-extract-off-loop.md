---
schema: 1
type: fix
scope: search-extract-off-loop
date: 2026-11-24
title: Extract's page work can run off the event loop, and a page is parsed once
---
`threetears.search.extract.extract()` ran a page's CPU work on the caller's event loop: the decode, trafilatura's
text extraction, then `trafilatura.metadata.extract_metadata` as a second full parse, with author extraction and
htmldate's extensive date search. A host making several reads at once had their timers fire late, and reads died at
their bound after the network had answered. Chunk 01 of the `search-extract-off-loop` plan, in two commits:

- `extract(..., executor=)` takes a `concurrent.futures.Executor`. The decode, the text, the title and date, and a
  thin page's frame scan are module-level functions over bytes and str with picklable results, so a
  `ProcessPoolExecutor` works. With no executor the work runs inline, as before.
- A failure the page stage absorbs is recorded in the caller's process, once, wherever the stage ran.
- The title and date are read from the tree the text came from. The date is htmldate's ordinary search, given the
  address that answered.
- The `[extract]` extra names `htmldate`.

## Consumer note
`extract()` takes an optional `executor`; pass a `ProcessPoolExecutor` to keep page parsing off the event loop.
Nothing else in its signature, `Candidate` or the transport changes. A page's `published_at` is now None where only
htmldate's extensive search found a date, and a page whose address carries the day reads that day. A broken executor
raises; a cancelled read raises `CancelledError` at once while work already handed to the executor runs out there.
