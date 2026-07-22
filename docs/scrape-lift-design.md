# 3tears-scrape: lift from faidh + page-finding agent + schema-discovery mode — design + decisions

**Status:** DECIDED (2026-07-15) — design captured; build sharded below. Task 01 (the lift) is
approved to start now; tasks 02/03 are scoped here but explicitly sequenced *after* task 01 ships.
**Driver:** faidh's WARN Act scraper (`src/faidh/scrape/*`) was built domain-agnostic from Chunk 1
on, on the explicit premise that it would eventually lift into 3tears as a real, general-purpose
package — every file already carries a "Zero faidh imports (see `scrape/__init__.py`)" docstring
line and a "directory move, not a disentangling exercise" comment. Other 3tears consumers need this
capability today, not hypothetically — this design turns the standing promise into a real package.

> Captured because this is now four distinct, real pieces of work (the lift itself, the sidecar's
> relocation, a page-finding agent, a schema-discovery mode) that piled up across one conversation
> — written down before any of it is built, per this project's own "requirements precede code" rule.

> **Design rules for this whole effort:**
> 1. **The lift is a move, not a rewrite.** `src/faidh/scrape/*` already has zero faidh imports
>    (verified: `grep -rn "^from faidh\|^import faidh" src/faidh/scrape/` returns nothing). Don't
>    redesign working code while moving it — file relocation + import-path updates + a real test
>    pass, not a rearchitecture.
> 2. **Preserve the AGPL isolation boundary exactly.** The nodriver sidecar stays a genuinely
>    separate process/container (HTTP, not an in-process import) — that arm's-length boundary is
>    the entire reason it's safe to depend on nodriver at all. Moving *where* it's maintained
>    doesn't change *that* it's isolated.
> 3. **Sequence strictly: task 01 ships and is reviewed before 02/03 start.** The page-finder agent
>    and discovery mode are real, separately-scoped features, not a rider on the lift PR.
> 4. **Every new capability stays independently callable, never a forced pipeline.** A caller with
>    a known URL skips the page-finder; a caller with a known schema skips discovery mode. Chaining
>    is a convenience, not a requirement — this mirrors the existing driver/eval-loop split, where
>    each stage already takes plain data in and returns plain data out.

---

## The requirement (decided — user direction, 2026-07-15)

Four pieces, in this priority order:

1. **Lift `3tears-scrape` for real.** A new top-level 3tears package, not a promise in a docstring.
2. **The nodriver sidecar moves too, maintained once.** "The nodriver stuff needs to lift into
   3tears as well... it needs to have the info to build the sidecar there, and that is where that
   should be maintained, not in both. People who use 3tears need to know how to build and use that
   sidecar... that is how we're dealing with AGPL." — 3tears gets its own real, buildable sidecar
   deployment (Dockerfile + docs), the single source of truth; faidh (and any other consumer)
   builds/runs *that*, not a duplicated copy.
3. **A page-finding research agent.** "We need a new AI loop that does the research to find the
   right page... an agent that can use the web search and web fetch tools in a loop up to a certain
   amount of turns until it gives us the right page... you should be able to call that and then
   chain the result into the next stage which is find the set of variables on that page. But it
   shouldn't always have to go in this order — you should be able to unchain all of the scraper
   features and just run them individually, or start the chain from there." Verification of its own
   answer before returning it: agreed, not optional.
4. **A schema-discovery mode on the same mechanism.** "We also want a 'capture every variable on
   this page' mode, and that determination... should be placed on this same mechanism: either it is
   already looking for certain variables, or it is going to outline all it can capture for use in
   the variable list." Not a new tool — a mode flag on the existing propose→validate→judge
   extraction pipeline: schema *given* (today's behavior) vs. schema *discovered* (new).

---

## Current state (VERIFIED in the faidh checkout, not assumed)

`src/faidh/scrape/` (2026-07-15, `feature/3tears-scrape-chunk-01` branch, Chunk 22 shipped):

| File | What it is |
|---|---|
| `driver.py` | `ScrapeDriver` ABC, `RenderedPage`, `NetworkCall`, `NavStep` — the whole backend-agnostic contract |
| `drivers/nodriver_sidecar.py` | HTTP client for the sidecar service (below) |
| `drivers/camoufox.py` | In-process browser backend (MPL-2.0-safe, no sidecar needed) |
| `drivers/document.py` | Static-file backend (PDF/DOCX/XLSX/CSV/TXT/MD/LaTeX via `threetears.agent.tools.document.parse_document`) |
| `drivers/api.py` | Stateless JSON-API backend |
| `drivers/network_capture.py` | Authenticated in-session XHR capture backend (newest, Chunk 22) |
| `extraction.py` | LLM candidate generation (CSS-selector *and* regex/text-block shapes) + structural validation |
| `eval_loop.py` | propose → validate → judge → persist, single-record and multi-row |
| `collections.py` | `ScrapeTarget`/`ScrapeRecipe`/`ScrapeExtraction` + their `BaseCollection`s |
| `migrations.py` | Already registered under `PACKAGE_NAME = "3tears_scrape"` (forward-looking naming, done) |
| `target_source.py` | Pluggable target config (`StaticTargetSource`/`YamlTargetSource`/`CollectionTargetSource`) |
| `tool.py` | `ScrapeTool` — the ad-hoc, one-off MCP-exposed entry point |
| `enrichment.py` | Secondary free-form LLM notes pass |
| `llm_retry.py` | Shared bounded-retry structured-call helper |

`tests/scrape/` mirrors every file above 1:1, plus `test_driver_contract.py` (the shared,
backend-agnostic `ScrapeDriver` contract suite). `grep -rn "^from faidh\|^import faidh"
src/faidh/scrape/` returns **nothing** — the zero-faidh-imports discipline documented in every
file's own docstring is real, not aspirational.

`services/nodriver-sidecar/` (faidh repo, separate deployable): `Dockerfile`, `main.py` (the HTTP
API nodriver itself is wrapped behind), `docker-compose.yml`, its own `pyproject.toml`/`uv.lock`,
its own `tests/`. AGPL-3.0 (nodriver's own license) — isolated by being a genuinely separate
process/container faidh's MIT-licensed code only ever talks to over HTTP, never imports.

**3tears' own build conventions** (verified: `docker-bake.hcl`, `docker/Dockerfile`): one shared
`threetears-base` image built from `3tears/docker/Dockerfile`, consumed by per-repo *consumer*
targets (`hub`, `admin`, `schema`, `agent`) each with their own Dockerfile in their own sibling
repo. No existing `services/<name>/` convention — every deployable is a bake target with a context.

**3tears package layout** (verified: `packages/*`): every domain package is a same-named directory
directly under `packages/` (`packages/channels/`, `packages/mcp/`, `packages/scheduled-jobs/`,
etc.), each with its own `pyproject.toml`/`README.md`/`src/`/`tests/`. No package is nested under
another (e.g. nothing lives under `packages/agent/` that isn't itself agent-specific).

---

## Scope boundary — what lifts, what stays in faidh

| Stays in faidh | Lifts to `3tears-scrape` |
|---|---|
| `src/faidh/intake/plugins/warn_act.py` — Tier 2 signal mapping, `ArbitrarySignalEntity`, WARN-domain field interpretation | Everything in the table above |
| `src/faidh/intake/plugins/seeds/warn_act_targets.yaml` — WARN-specific target config | — |
| `src/faidh/scrape/tool.py`'s faidh-side wrapper, `src/faidh/tools/scrape_tool.py` (`FaidhScrapeTool`, the zero-arg `FAIDH_TOOLS`-registered shim) | `tool.py`'s own `ScrapeTool` (the real, reusable, DI'd implementation it wraps) |
| `src/faidh/intake/runner.py`'s `poll_scrape_targets` (decided 2026-07-15 — see D2a below) | — |

**Decided (2026-07-15): `poll_scrape_targets` stays in faidh.** Domain-agnostic in principle
(takes plain `ScrapeTarget`s/field schemas, no WARN-specific knowledge), but it's faidh's own
scheduling orchestration, living alongside CongressGov/GDELT/Telegram's own cadence logic in the
same file — that's faidh's call to make about its own intake layer, not a `3tears-scrape` platform
primitive. It already consumes `3tears-scrape`'s real driver/eval-loop/collections primitives as a
plain library consumer post-lift; no further extraction needed.

After the lift, faidh depends on `3tears-scrape` exactly the way it already depends on
`threetears.agent.tools.document` — a real package import, not a copied module.

---

## Decisions

- **D1 — New top-level package `packages/scrape/` → `3tears-scrape`.** A directory move of
  `src/faidh/scrape/*` (verbatim logic, updated import paths only), matching the flat
  `packages/<domain>/` convention every other 3tears package already uses. Ships with its own
  `pyproject.toml`/`README.md`/`LICENSE`, mirroring `packages/scheduled-jobs/`'s shape.
- **D2 — WARN-Act domain logic stays in faidh.** `3tears-scrape` never learns what `employer`/
  `county`/etc. mean — that discipline (already true today, see `extraction.py`'s own "domain-
  agnostic core never hardcodes what a field means" docstring) doesn't change, it just moves to a
  package boundary instead of a directory boundary.
- **D2a — `poll_scrape_targets` stays in faidh.** It's faidh's own intake-scheduling orchestration
  (alongside CongressGov/GDELT/Telegram's own cadence logic in the same file), not a `3tears-scrape`
  platform primitive, even though it only consumes plain `ScrapeTarget`s/schemas today. Faidh keeps
  it and calls into `3tears-scrape`'s driver/eval-loop/collections as a normal library consumer.
- **D3 — The nodriver sidecar moves to 3tears, maintained once.** Proposed home:
  `packages/scrape/sidecar/` (source + Dockerfile alongside the package that depends on it, not a
  new top-level `services/` convention this repo doesn't otherwise have) plus a new
  `docker-bake.hcl` target (`nodriver-sidecar`) so any consumer builds it the same way they'd build
  any other bake target — `docker buildx bake nodriver-sidecar`. faidh's own
  `services/nodriver-sidecar/` is deleted, not kept as a second copy, once the 3tears one is real
  and faidh points at it.
- **D4 — AGPL isolation is structural, not maintainer-dependent.** The sidecar is a separate
  process/container regardless of which repo hosts its source — `NodriverSidecarDriver` (the
  in-process side) only ever speaks HTTP to it. `CamoufoxDriver` remains the in-process,
  license-safe alternative for consumers who don't want the sidecar dependency at all.
- **D5 — Page-finding agent is a new, independent stage (task 02, not yet built).** Bounded-turn
  loop (WebSearch/WebFetch tools, a real turn cap, not unbounded), self-verifying its own answer
  (fetch the candidate URL and check for real structure — a table, a document link, *something* —
  before returning it, not just trusting a search-snippet match) before handing back a
  `ScrapeTarget`-shaped result. Independently callable — never a forced prerequisite to extraction.
- **D6 — Schema-discovery is a mode, not a new tool (task 03, not yet built).** The existing
  propose→validate→judge extraction mechanism gains a second mode: no caller-supplied
  `field_schema` in ⇒ a discovered field list (names, inferred types, sample values) out, using the
  same structured-LLM-call infrastructure (`bounded_retry_structured_call`) the schema-*consuming*
  candidate generators already use, just inverted.
- **D7 — Strict sequencing.** Task 01 (the lift) ships, is Critic-reviewed, and lands before task
  02/03 design gets fleshed out further, let alone built. Building the page-finder/discovery-mode
  work *inside* `3tears-scrape` post-lift avoids building it in faidh only to re-lift a second time.

---

## Build (task shards — `scrape-task-NN` convention, in this `docs/` dir)

- **`scrape-task-01`** — Lift `src/faidh/scrape/*` into `packages/scrape/` (`3tears-scrape`) +
  relocate the nodriver sidecar into 3tears + repoint faidh as a real dependency consumer. Full
  design in `docs/scrape-task-01-lift-core-package.md`. **Approved to start now.**
- **`scrape-task-02`** — Page-finding research agent (bounded WebSearch/WebFetch loop +
  self-verification), producing a `ScrapeTarget`-shaped result, independently callable. Scoped
  above (D5); full design deferred until after task 01 ships.
- **`scrape-task-03`** — Schema-discovery mode on the existing extraction mechanism. Scoped above
  (D6); full design deferred until after task 01 ships.

---

## Anti-patterns

- **DO NOT redesign extraction/driver logic while lifting it.** Task 01 is a move + import-path
  update + a real test pass, not a rewrite — if something looks worth improving mid-lift, note it
  and finish the move first.
- **DO NOT let faidh keep a second copy of the sidecar "just in case."** Once 3tears' sidecar is
  real and faidh points at it, faidh's `services/nodriver-sidecar/` is deleted. One source of
  truth, per direct instruction.
- **DO NOT import nodriver in-process anywhere.** The sidecar's separate-process boundary is the
  AGPL isolation itself, not an implementation detail — this holds regardless of which repo hosts
  the source.
- **DO NOT build the page-finder or discovery mode before task 01 ships.** Sequencing is a decision
  (D7), not a suggestion.
- **DO NOT force page-finder → extraction into a single non-optional pipeline.** Every stage takes
  plain data in, returns plain data out, exactly like the existing driver/eval-loop split already
  does — chaining is convenience, not architecture.
