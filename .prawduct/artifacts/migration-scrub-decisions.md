# Backlog migration scrub decisions

The markdown backlog moved to GitHub Issues on 2026-11-26. This records the decisions behind that
one-time migration, so they can be audited without the session that made them.

## Target

- **Repo:** `pacepace/3tears`, confirmed by the owner 2026-11-26 ("Yes to all recommendations",
  `.prawduct/owner-rulings.md`). Recorded as `backlog_service_repo` in `project-state.yaml`.
- **Archive scope:** `all` (the default). The only archived item is minted as a closed issue so its
  id still resolves on the tracker.
- **prawduct:** 3.10.0. Labels provisioned with `prawduct-hook backlog provision` (7 created).

## Survey

Surveyed through 2026-11-26. The source held two items, both filed that day on
`feat/model-key-pool` and carried onto #1305's branch for the import. Before importing, the
tracker was searched for each one's subject.

| id | disposition | issue | reason |
|---|---|---|---|
| MOD-7R2K | keep | #1311, open | An OpenRouter `finish_reason: "error"` chunk ends the stream silently; no existing issue covers it. |
| MOD-4QW9 | drop (duplicate) | #1312, closed not planned | #776 (open since 2026-10-13) tracks the same `max_retries=0` defect. The one fact MOD-4QW9 added, that a caller-supplied `client=` ignores `max_retries`, was posted on #776 (issuecomment-7182853553). #965 also asks for `max_retries=0` to be a single attempt. |

Neither item was split or merged. Both ids were valid PFX ids.

## Restructure

Both titles were tightened to the `area: summary` shape and both bodies were set in the bug template
(Problem, Repro, Actual, Expected, Evidence, Env). `restructure-preview` reported 0 title failures,
0 lint findings and 0 collisions. Each issue keeps its pre-migration title and body verbatim in its
`prawduct` block (`original_title`, `original_body`).

## Import and verification

- `import`: 2 created, 0 skipped, 0 rejected, 0 collisions.
- `export`: 2 items, taken as a backup right after the import.
- `verify-migration`: exit 0. `missing`, `unaliasable`, `collisions`, `status_mismatch`,
  `duplicate_alias` and `unencodable_status` are all empty; 2 source items, 2 aliased.
