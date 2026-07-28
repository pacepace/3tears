# 3tears-scrape

`threetears.scrape` -- AI-driven, self-healing web scraping: LLM-proposed
extraction strategies validated against real page content, a durable circuit
that stops paying for a walled target, and a human-handover surface for the
pages no unattended fetch will pass.

## Problem

Scrapers are written per site and die per site. A selector moves, extraction
returns nothing, and somebody edits a parser -- so the cost of a hundred sources
is a hundred parsers and a hundred small repairs. Onboarding a source is an
engineering task rather than a configuration change.

Two failures follow from that shape and neither is a parsing problem. A page
that has been *redesigned* and a page that is showing a *bot challenge* both
look like "extraction returned nothing", and they need opposite responses:
regenerate the strategy, or stop fetching and fetch a human. Told apart wrongly,
a challenge burns a working recipe and a redesign waits for somebody to notice.

And a target behind a wall costs money forever. Every poll fetches a page that
fails, and every failure invites another LLM call to work out why -- so the
cheapest thing to do with a blocked target is the thing a naive loop never does,
which is not fetch it.

## What it does

- **`ScrapeDriver`** -- a backend-agnostic render contract (`RenderedPage`,
  `NavStep`, `NetworkCall`). Implementations ship for a real headful Chromium
  behind an isolated sidecar, in-process stealth Firefox, documents
  (PDF/DOCX/XLSX/CSV), plain JSON APIs, authenticated in-session XHR capture, the
  listing-plus-detail and multi-document shapes, and a forced download for a
  document sitting behind a real challenge. The `drivers/` package is the current
  set -- read the modules there rather than trusting a list here, since neither a
  count nor an enumeration survives the next one being added. (The package's
  `__all__` is deliberately empty: each driver is imported from its own module, so
  nothing is re-exported at the namespace level.)
- **The eval loop** (`run_eval_loop`, `run_eval_loop_multi_row`) -- propose
  candidates, validate each structurally against the real page, have a judge pick
  the winner by comparing extracted values against page content, persist the
  winner as a recipe, and reuse it with zero LLM calls until it stops validating.
  That is the `css` and `regex` path.

  The `per_document` and `multi_row_vision` strategies deliberately have no
  reusable pattern, for different reasons. `per_document`'s targets share no
  template a recipe could generalise across; `multi_row_vision`'s problem is that
  `find_tables()`, the text substrate a cached pattern would key against, fails on
  its table.

  Both therefore pay LLM calls on every poll, never fewer than two per unit of
  work. `per_document` spends, per document, an extraction plus an unconditional
  grounding judge -- and the extraction's cost depends on the document's shape,
  since a born-digital one is chunked by field count while an OCR'd one is a single
  vision call. `multi_row_vision` spends two over the whole table: an extraction,
  then the same kind of judge.

  Both also persist a recipe, and it is a marker for operational visibility rather
  than something reused to skip a call -- so a recipe row existing is not evidence
  that a poll is free. Worth knowing before costing a target, because it is the
  difference between paying once and paying every poll.
- **`classify_failed_page`** -- asks what a failed page IS. A wall leaves the
  recipe byte-identical and records `blocked`; a genuinely changed page
  regenerates on the first failure rather than waiting for a threshold.
- **`TargetCircuit`** -- the durable fetch gate. Repeated blocks trip it open, an
  open circuit suppresses the fetch, and each probe that finds the wall still
  standing doubles the wait to a ceiling. It writes no state machine of its own:
  the row is hydrated into a real `CircuitBreaker` from
  [`models`](models.md) via `restore()`, the transition is driven by calling it,
  and the result is written back.
- **The human-handover surface** -- `build_operator_router()` returns a FastAPI
  `APIRouter` carrying the operator page, a vendored noVNC client, and the
  WebSocket that relays the display's pixels. `claim_session` takes a lease so
  two pods cannot both serve one display; `serve_session` answers that session's
  control messages on a subject keyed to the session id, so a caller addresses
  the session and never a pod. A completed target's solve comes back **sealed**.
- **Session-state reuse** (`seal_session_state`, `record_session_state`) -- the
  cookies a human earned are sealed under an operator key, stored with an expiry,
  and handed back to later unattended fetches, so one solve is not paid for twice.
- **`RobotsGate`** -- honours a target's `Crawl-delay` and escalates a `Disallow`
  to a human rather than obeying or ignoring it silently. Both on by default.
- **Egress** -- which exit a fetch leaves by, through
  [`core`](core.md)'s `EgressDriver` seam.

## Design philosophy

**Seams, not services.** Nothing here runs itself. `ScrapeTool` is registered by
a platform, `list_walled()` and `record_human_cleared()` are called by one,
`ScrapeDriver` is a protocol a platform picks an implementation of, and the
operator surface is a router a platform mounts into an app it already has. There
is no daemon to deploy and no poller: the package will not fetch anything you did
not ask it to.

**The domain stays out.** The core never learns what a field *means* -- only that
whatever a candidate extracted for a caller-supplied field name parses as its
declared type. There is no per-site code and no vendor-shaped pattern matching.

**A human is part of the design, not a fallback.** Some pages are cleared by a
person in seconds and by no unattended fetch ever. That path is first-class: a
bounded working set, a hard TTL, per-target browser-context isolation, and the
person's work captured and reused. Nothing is held while a target waits -- it is
reported and forgotten, then re-driven from `url` plus `nav_steps` when somebody
actually arrives, so waiting costs no container resources.

**Authorization lives where identity lives.** The sidecar authenticates nobody
and never will: it holds no identity and cannot evaluate a policy. It honours a
capability; deciding who was entitled to one is the consuming platform's call.

**The AGPL boundary is a process boundary.** `nodriver` is AGPL-3.0, so it runs
in its own container that imports nothing from `threetears` -- Xvfb, Chromium,
`x11vnc` and nothing else. Everything needing this family's identity,
coordination or NATS primitives lives on the MIT side, which reaches the display
over a shared loopback. `CamoufoxDriver` is a fully in-process alternative if you
do not want the sidecar at all.

## When to adopt

Adopt it when you have more than a handful of sources and the per-site parser is
already the maintenance cost, or when your targets are behind bot challenges and
somebody is currently clearing them by hand with nothing captured. Adopt the
circuit on its own if you have targets you already know are walled and are still
paying to find out.

Do **not** adopt it for one stable page you control -- an `httpx` call and a
selector is cheaper and always will be. It also assumes an LLM budget: the
self-healing is LLM-driven, and while a validating recipe costs nothing, a
regeneration is a real call. And it assumes PostgreSQL through
[`core`](core.md), because the recipes and health rows are durable by design.

## Composes with

- [`core`](core.md) -- three-tier collections for recipes, extractions and
  health rows; `KVLease` for the session claim; `EgressDriver` for exits;
  `security.encryption` for sealing a human's solve.
- [`models`](models.md) -- the LLM factory the eval loop proposes and judges
  through, and `CircuitBreaker`, which the fetch circuit hydrates rather than
  reimplements.
- [`agent-tools`](agent-tools.md) -- `parse_document` for the non-HTML drivers,
  and the tool surface `ScrapeTool` is exposed through.
- [`nats`](nats.md) -- `serve_owner` / `forward` route a session's control
  messages to the pod holding its display. Optional, behind the `hitl` extra.
- [`scheduled-jobs`](scheduled-jobs.md) -- books a blocked target's next probe
  for an event-driven caller. Optional, behind the `reprobe` extra; a polling
  caller needs none of it.
- [`observe`](observe.md) -- structured logging throughout.

## Install

```bash
pip install 3tears-scrape

# The human-handover router: adds FastAPI and 3tears-nats. Optional because a
# deployment that never needs a person never needs a web framework.
pip install "3tears-scrape[hitl]"

# Books a blocked target's next probe as a scheduled job, for an event-driven
# caller rather than a polling one.
pip install "3tears-scrape[reprobe]"
```

The nodriver sidecar is a container, not a pip install -- see
[`packages/scrape/sidecar/`](../../packages/scrape/sidecar/) for its definition,
its own AGPL-3.0 licence, and its contract tests.

`3tears-scrape` redistributes noVNC (MPL-2.0) as package data, so its declared
licence is `MIT AND MPL-2.0 AND LicenseRef-noVNC-DES` rather than plain MIT. The
licence texts travel with the files and in the wheel's `dist-info`; provenance is
recorded in `operator_assets/novnc-provenance.json`.
