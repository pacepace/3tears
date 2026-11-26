# Backlog — 3tears

> **Frozen history since 2026-11-26.** The live backlog is GitHub Issues in
> [pacepace/3tears](https://github.com/pacepace/3tears/issues): read it with `/prawduct:backlog`,
> `prawduct-hook backlog list --repo pacepace/3tears`, or `gh issue list -R pacepace/3tears`.
> Both items below migrated: MOD-7R2K is #1311 (open) and MOD-4QW9 is #1312 (closed as a duplicate
> of #776). Nothing here is read any more, and changes made on the tracker are not copied back, so
> this file will drift from it. File new items on the tracker, never here.

<!-- Structured backlog. Manage it with /prawduct:backlog — pick, add, find <q>, list,
     update ID, migrate. Items move between the three sections below via
     /prawduct:backlog update ID status=...; status is never inferred from build plans
     or change logs — an agent or human sets it explicitly.

== Item shape ==

  EVERY ITEM STARTS AT COLUMN 0. The parser takes a column-0 `- ` bullet as the
  item boundary and treats everything indented as that item's body, so an item
  written with leading spaces is invisible: it parses as no item at all, and a
  whole backlog written that way migrates as zero issues. The example below is
  therefore flush-left on purpose — do not indent it back to match this comment.

- **[PFX-XXXX]** One-line title
  `effort: M · impact: M · area: stop-hook · source: reflection · added: 2026-05-29 · status: open`

  Free-form body of any length — a single sentence or multi-paragraph analysis
  with file refs, fix-shape, and open questions. The author chooses what fits.

  ID format `[PFX-XXXX]`:
    PFX = 2–3 uppercase letters naming the work-space the item was filed from.
          Derive a sensible prefix from the item's area; reuse existing ones so
          related items share a prefix. Starter vocabulary (extend freely):
            STH stop-hook · CRT critic · SYN sync · LLM prompt/LLM · BKL backlog
            MIG migration · JNT janitor · MET methodology · DOC docs · TST tests
          A project may optionally declare its prefix vocabulary as
          `backlog_prefixes:` in project-state.yaml for validation — not required.
    XXXX = 4-char random alphanumeric (base36). Random IDs avoid cross-branch
           collisions; ~1.7M combinations per prefix.

  Metadata bar (one backticked, dot-separated line; required on new items):
    effort: S | M | L     S = <30 min · M = hours · L = multi-chunk
    impact: S | M | L     S = cosmetic · M = quality-of-life · L = user-felt/structural
    area:   <tag>         free-form topic tag; reuse existing tags to enable grouping
    source: builder | critic | reflection | janitor | user
    added:  YYYY-MM-DD
    status: open | promoted | shipped | dropped
  Optional, on the same line (distinct concepts — keep them straight):
    related:   PFX-XXXX, PFX-XXXX   cross-references to related items
    closes:    PFX-XXXX             this item supersedes another backlog item (item → item)
    closed-by: scope:<slug> | PR | release tag  what shipped this item (item → release), set on
                                    status=shipped; a handle that exists before the commit —
                                    never a bare chunk id (names no plan), a bare commit SHA
                                    (dangles on --amend) or a branch name (refused). Form:
                                    the backlog skill's `update`
    reviewed:  YYYY-MM-DD           last-touched timestamp (auto-set on any update)
    accepted-by: @actor             soft claim "someone is on this" so others don't
                                    double-pick; pick/list exclude claimed items.
                                    Does NOT auto-expire; auto-cleared on ship/drop.
                                    Not a lock (backlog.md is eventually-consistent).
    stage: <lifecycle>              idea | research | requirements | design | ready.
                                    Where the item sits in the feature lifecycle;
                                    only `ready` is implementable. Absent/early =>
                                    pick routes to discovery/planning, not code.
    refs: <doc#section>, <doc>      links to governing artifacts (requirements /
                                    arch / design docs). Distinct from `related:`
                                    (which is item -> item).
    revisit: YYYY-MM-DD | <event>   expiry on a norm exception or stopgap, or a dated
                                    re-check; a past date raises an advisory while
                                    the item is open.

  Legacy items (no metadata) remain valid — tools treat them as
  `effort: ? · impact: ? · area: untagged · status: open` and rank them lower.
  Run `/prawduct:backlog migrate` to add structure at your own pace; nothing is forced. -->

## Open

<!-- Items available to pick up. -->

- **[MOD-7R2K]** OpenRouter's documented mid-stream error ends the stream silently
  `effort: M · impact: L · area: models · source: builder · added: 2026-11-26 · status: open`

  OpenRouter documents a mid-stream error as a chunk whose `choices` carry `finish_reason: "error"`.
  langchain-openrouter 0.2.9 raises only when `choices` is empty, so with the documented shape the stream just
  ends: the caller gets a truncated answer as if it were whole, and the key pool (KP-H5) never sees the error.
  Found while building the key pool's classifiers (chunk 03, `key_pool/classify.py`). Fix in
  `providers/openrouter.py`: raise the provider error for a `finish_reason: "error"` chunk, with a test on the
  documented shape.

## Promoted

<!-- Items currently being addressed in an active build plan. /prawduct:backlog pick
     skips these by default (work is already in flight). -->

## Archive

<!-- Shipped and dropped items, kept for searchability. Never deleted. -->

- **[MOD-4QW9]** OpenRouter `max_retries=0` does not turn retries off
  `effort: S · impact: M · area: models · source: builder · added: 2026-11-26 · status: dropped · reviewed: 2026-11-26`

  With `max_retries=0` langchain-openrouter sets no retry config, so the SDK's default applies: 5xx and connection
  errors retried for up to an hour. A caller-supplied `client=` ignores `max_retries` entirely. A test run hung on
  it while building the key pool (chunk 03). Fix in `providers/openrouter.py`: map `max_retries=0` to an explicit
  no-retry config, with a test that a 5xx surfaces after one request.

  Dropped 2026-11-26: duplicate of pacepace/3tears#776, which tracks the same defect. The one fact this item adds
  (a caller-supplied `client=` ignores `max_retries`) is posted on #776.

