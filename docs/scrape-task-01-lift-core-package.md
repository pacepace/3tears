# scrape-task-01: Lift `src/faidh/scrape/*` into `packages/scrape/` (`3tears-scrape`) + relocate the nodriver sidecar

**Status:** APPROVED TO START. **Foundational** — `scrape-task-02` (page-finding agent) and
`scrape-task-03` (schema-discovery mode) build inside this package once it exists; neither starts
before this ships.
**Scope:** a new top-level 3tears package (`3tears-scrape`) + a new sidecar build target, sourced
from faidh's already-domain-agnostic `src/faidh/scrape/*` and `services/nodriver-sidecar/`. A move
+ import-path update + a real test pass — not a rewrite (see the parent design's Design Rule 1).
**Origin:** `docs/scrape-lift-design.md` (D1-D4, D7).

---

## Objective

Move every file in faidh's `src/faidh/scrape/` (verified zero-faidh-imports, see the parent
design's "Current state" table) into a new `packages/scrape/` package in this repo, publish it as
`3tears-scrape`, relocate `services/nodriver-sidecar/` alongside it as a new buildable unit, and
repoint faidh to consume the published package instead of its own copy. When this ships, faidh's
`src/faidh/scrape/` and `services/nodriver-sidecar/` directories no longer exist — there is exactly
one copy of this code, in 3tears.

---

## Design constraints (verified, not assumed — do not relitigate)

- **Zero faidh imports already true.** `grep -rn "^from faidh\|^import faidh" src/faidh/scrape/`
  (faidh repo) returns nothing. This move is mechanical: relocate files, update the package's own
  internal import paths (`from .driver import ...` stays relative and needs no change; anything
  currently reachable only via `faidh.scrape.X` from *outside* the package is what faidh's own
  consumer code must update).
- **Migrations are already correctly namespaced.** `migrations.py`'s `PACKAGE_NAME = "3tears_scrape"`
  was chosen for exactly this move — no renaming needed, just registration under 3tears' own
  `MigrationRunner` instead of faidh's.
- **3tears package layout is flat.** Every domain package is `packages/<name>/` directly (verified:
  `packages/channels/`, `packages/scheduled-jobs/`, `packages/mcp/`, etc.) — `packages/scrape/`,
  not nested under `packages/agent/`.
- **3tears has no `services/<name>/` convention.** Deployables are `docker-bake.hcl` targets with a
  context pointing at a sibling repo's own Dockerfile (verified: `hub`/`admin`/`schema`/`agent`
  targets). The sidecar becomes a new target, not a new top-level directory pattern this repo
  doesn't otherwise have.
- **The AGPL isolation boundary is structural, not aspirational.** The sidecar must remain a
  genuinely separate process/container after the move — `NodriverSidecarDriver` only ever speaks
  HTTP to it. Do not collapse it into an in-process import under any circumstance.
- **Package `pyproject.toml` shape**, per `packages/scheduled-jobs/pyproject.toml`'s own precedent:
  `dependencies` lists real `3tears-*` package deps (this package will need `3tears`,
  `3tears-agent-tools` for `DocumentDriver`'s `parse_document` reuse, `3tears-models` for
  `LlmPurpose`, `3tears-observe`), plus the scrape-specific third-party deps faidh's own
  `pyproject.toml` already declares (`httpx[socks]`, `beautifulsoup4`, `pydantic`, `camoufox`,
  `playwright<1.61` — carry the version pin and its `daijro/camoufox#653` comment forward verbatim,
  the bug is still open until proven otherwise).

---

## Files to move (source: faidh repo, verified inventory)

### `packages/scrape/src/threetears/scrape/` (from `src/faidh/scrape/`)

`__init__.py`, `driver.py`, `drivers/__init__.py`, `drivers/nodriver_sidecar.py`,
`drivers/camoufox.py`, `drivers/document.py`, `drivers/api.py`, `drivers/network_capture.py`,
`extraction.py`, `eval_loop.py`, `collections.py`, `migrations.py`, `target_source.py`, `tool.py`,
`enrichment.py`, `llm_retry.py`.

### `packages/scrape/tests/` (from `tests/scrape/`)

`__init__.py`, `test_collections.py`, `test_driver_api.py`, `test_driver_camoufox.py`,
`test_driver_contract.py`, `test_driver_document.py`, `test_driver_network_capture.py`,
`test_driver_nodriver_sidecar.py`, `test_enrichment.py`, `test_eval_loop.py`, `test_extraction.py`,
`test_migrations_drift.py`, `test_target_source.py`, `test_tool.py`.

### Sidecar (from `services/nodriver-sidecar/`)

`Dockerfile`, `main.py`, `docker-compose.yml`, `entrypoint.sh`, `pyproject.toml`, `uv.lock`,
`.dockerignore`, `LICENSE`, `conftest.py`, `tests/`. Proposed home: `packages/scrape/sidecar/`
(source alongside the package that depends on it). **Resolve exact placement during implementation
if a different convention turns out to fit better — this is a proposal, not a mandate.**

### Stays in faidh (do not move)

`src/faidh/intake/plugins/warn_act.py`, `src/faidh/intake/plugins/seeds/warn_act_targets.yaml`,
`src/faidh/tools/scrape_tool.py` (`FaidhScrapeTool`, the thin `FAIDH_TOOLS` wrapper — it constructs
and delegates to `3tears-scrape`'s real `ScrapeTool`, doesn't reimplement it).

---

## Decided: `poll_scrape_targets` stays in faidh

`poll_scrape_targets` (`src/faidh/intake/runner.py`) is domain-agnostic in principle (plain
`ScrapeTarget`s/schemas, no WARN-specific knowledge), but it's faidh's own intake-scheduling
orchestration, living alongside CongressGov/GDELT/Telegram's own cadence logic in the same file —
faidh's call to make about its own intake layer, not a `3tears-scrape` platform primitive (see the
parent design's D2a). It doesn't move; it just calls into `3tears-scrape`'s real
driver/eval-loop/collections primitives as a normal library consumer after the repoint, same as
`warn_act.py` does.

---

## Implementation Notes

1. **Move first, update imports second, run tests third — in that order, don't interleave.** A
   file-for-file move (`git mv` where possible, to preserve blame/history) keeps the diff reviewable
   as "this moved" rather than "this changed," per Design Rule 1.
2. **`packages/scrape/pyproject.toml`** — model directly on `packages/scheduled-jobs/pyproject.toml`
   (build-system, project metadata, `dependencies`, `[project.optional-dependencies]` shape).
   `name = "3tears-scrape"`, `license = "MIT"` (matches every other 3tears package — the AGPL
   surface is the *separate* sidecar container, never this package's own license).
3. **Migrations registration** — `migrations.py`'s `register()`/`apply_migrations()` need to plug
   into 3tears' own `MigrationRunner` wiring (wherever 3tears' other packages register their own
   migrations — follow that existing pattern, don't invent a new one).
4. **faidh-side repoint** — `pyproject.toml`'s `[tool.uv.sources]` entry for whatever currently
   resolves `faidh.scrape.*` changes to a real `3tears-scrape` dependency declaration (mirrors how
   faidh already depends on `3tears-agent-tools` for `parse_document`). Every faidh import of
   `faidh.scrape.X` becomes `threetears.scrape.X` (or whatever the actual published module path
   ends up being — match 3tears' own naming convention, verified via an existing package's
   `src/threetears/<name>/` layout before assuming the path).
5. **Sidecar bake target** — new `docker-bake.hcl` target (`nodriver-sidecar`), context pointing at
   `packages/scrape/sidecar/` (in-repo, unlike the cross-repo consumer targets), own Dockerfile.
   Document the build/run command in the package's own `README.md` — "people who use 3tears need to
   know how to build and use that sidecar" is a documentation requirement, not just a build one.
6. **Delete, don't duplicate.** Once faidh's own test suite is green against the published
   `3tears-scrape` package, delete `src/faidh/scrape/`, `tests/scrape/`, and
   `services/nodriver-sidecar/` from the faidh repo in the same change that repoints the
   dependency — never leave both copies live "just in case."

---

## Anti-patterns

- **DO NOT rewrite or "improve" logic while moving it.** If something looks worth changing, note it
  for a follow-up chunk and move the working code as-is first.
- **DO NOT leave a second copy of the sidecar in faidh "for now."** One source of truth, per direct
  instruction — this is the whole point of the move.
- **DO NOT collapse the sidecar into an in-process import.** The separate-process boundary is the
  AGPL isolation mechanism itself.
- **DO NOT lift `poll_scrape_targets`.** Decided: it stays in faidh (D2a) — don't move it "for
  consistency" with everything else in the inventory.
- **DO NOT start `scrape-task-02`/`scrape-task-03` work before this ships and is reviewed.**

---

## Acceptance Criteria

- [ ] `packages/scrape/` exists in 3tears with every file from the inventory above, `3tears-scrape`
      installable/importable as a real package.
- [ ] `packages/scrape/sidecar/` (or wherever placement lands) builds via a new `docker-bake.hcl`
      target; its README documents how to build and run it.
- [ ] Migrations register and apply cleanly under 3tears' own `MigrationRunner`.
- [ ] faidh's `pyproject.toml` depends on `3tears-scrape` as a real package (not a copied
      directory); every `faidh.scrape.X` import updated to the published module path.
- [ ] `src/faidh/scrape/`, `tests/scrape/`, `services/nodriver-sidecar/` deleted from faidh in the
      same change that completes the repoint.
- [ ] `warn_act.py`/`warn_act_targets.yaml`/`poll_scrape_targets` (all staying in faidh) still work
      end-to-end against a real state's WARN page through the published package — not just unit
      tests passing, a real live fetch.
- [ ] Both repos: mypy/ruff clean, full test suites green (no weakened tests to make the move
      "pass").

---

## Verification

```bash
# 3tears side
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scrape
uv run pytest packages/scrape/tests/ -q
./scripts/lint.sh && ./scripts/typecheck.sh
docker buildx bake nodriver-sidecar   # confirm it actually builds

# faidh side, after the repoint
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/faidh
uv sync --group dev --reinstall
uv run pytest tests/ -m "not live" -q
uv run ruff check src/faidh/ tests/ && uv run mypy src/faidh/
```
