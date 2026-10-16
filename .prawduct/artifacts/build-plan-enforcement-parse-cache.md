---
artifact: build-plan
version: 1
scope: enforcement-parse-cache
branch: perf/enforcement-parse-cache
depends_on: []
partition: "serial -- one shared helper and its call sites; a second agent would only contend for the same files"
last_validated: 2026-10-15
---

## Requirements Confidence

**Level:** High

**Why:** Measured, not inferred. On `develop` at `4819c7f31`, `tests/enforcement` takes about 135 s
and calls `ast.parse` 84,418 times over 5,424 distinct sources (most `src/` files 60 times) and
`Path.read_text` 94,104 times over 2,603 files; 54 s of the run is `ast.parse` alone. Parsing every
file in the repo once takes 3.2 s. `scan_private_bindings` issues 1,045,907 `stat` calls for 2,142
files: `_resolve_absolute` tries every import against every search root, twice per root. On a
Linux-under-Docker-Desktop box the same tree's `test_private_binding` scan fixture took 461 s and
`test_underscore_access` 192 s, inside a 78-minute tests phase.

Problem: every enforcement gate re-reads and re-parses the whole workspace, and the private-binding
scan re-stats the same candidate module files for every import of every file.

Success: `tests/enforcement` passes and fails exactly the same tests; `ast.parse` runs about once
per distinct file per run; the private-binding scan's `stat` count drops by orders of magnitude;
the suite's wall time drops substantially.

Out of scope: `pytest-xdist`, a cache that persists across runs, caching directory walks, the
ruff shell-out in shape B, and the non-enforcement package tests.

**Decisions:**
- [DECISION: an unbounded per-process dict, not an LRU | every gate walks every file in the same
  order, so any bound smaller than the file set misses on every lookup; the whole repo's trees
  cost about 800 MB, released by `clear_parse_cache()` when `tests/enforcement` finishes]
- [DECISION: an entry is valid while the file's `mtime_ns`, size and inode match | a test that
  rewrites a file in place re-parses; one `stat` replaces a read and a parse]
- [DECISION: a cached tree is shared, never copied | no gate mutates a parsed tree (no
  `NodeTransformer`, `fix_missing_locations`, parent links or body edits under
  `packages/enforcement/src` or `tests/enforcement`); a copy would cost what the cache saves]
- [DECISION: the private-binding module memo lives for one scan | a library consumer scanning
  twice in one process must see files created in between]
- [DECISION: a raising `parse_python_path` beside `parse_python_file` | the direct call sites
  raise, absorb or decode leniently, each on purpose; one raising primitive keeps each site's
  behaviour exact]
- [ASSUMPTION: the new public names ship in the next minor | `parse_python_path` and
  `clear_parse_cache` are API growth; whether they ride 0.70.0 (PR 800) or 0.71.0 is the
  user's call at merge time, and the API-growth gate fails until then]
- [DECISION: the cache is released by a `pytest_runtest_teardown` hook, not a package-scoped
  fixture | `tests/enforcement` has no `__init__.py`, and pytest 9 holds such a fixture to the end
  of the session]

## Chunks

### Chunk 01: Parse once per run

- [x] `parse_python_path` and `parse_python_file` cached in a `BuildOnce`, `clear_parse_cache()`,
  the private-binding per-scan memo, direct `ast.parse(path.read_text())` sites over repo files
  routed through the cache, a teardown hook clearing it. Verify: `tests/enforcement` and
  `packages/enforcement/tests` same pass/fail as baseline, bar the API-growth gate; parse and
  stat counts re-measured.

Not routed, deliberately: gates whose helper takes source text, so the same function checks the
repo file and a planted snippet (`test_build_once_is_the_only_lazy_fill.py` and five more), and
the two sites that parse a git blob rather than a file.

Critic mode: cumulative

## Status

Chunk 01 built and reviewed (final: 0 blocking). Draft PR pending; version bump pending the user's call; integration suite to run before the PR leaves draft.
