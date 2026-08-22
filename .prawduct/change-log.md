# Change Log — 3tears

<!-- Append new entries at the top. Each entry is a ## section.
     This file is separate from project-state.yaml to reduce merge conflicts
     when multiple branches add entries simultaneously.

     A tag-line under the ## header carries the two machine-read keys:

         <!-- prawduct: scope=<build-plan scope> | release=vX.Y.Z -->

     `scope` matches the governing build plan's `scope:` frontmatter.
     `release` is added AT RELEASE — its absence is what marks an entry
     release-pending, so a feature branch writes none. -->

## 2026-08-22: every L2 key is scoped to the principal that wrote it

<!-- prawduct: scope=full-collection-support -->

**Why:** a tool pod got L1 and nothing else, and the shared `{ns}-collections`
KV bucket was flat. Every key was `{table}.{body}`, every static NATS user held
`$KV.>` and `$JS.>` on both directions, and the bucket ran with `allow_direct`
unset — which puts every KV read on the body-carried `$JS.API.STREAM.MSG.GET`
form, where the key travels in the request body and no subject grant can
constrain it. So any principal could read any other principal's cached rows, and
the per-principal isolation the rest of the platform assumes did not exist at
the cache tier at all.

**What changed:** keys are now `{scope}.{table}.{body}`, where `{scope}` is the
principal the writing process authenticates as. Minted grants narrow to that
scope, the hub declares the bucket with `allow_direct: true` so reads are
subject-addressed and therefore constrainable, and tool pods get a real L2 tier
through a `ToolServer.add_connected_callback` seam. The static-user grants in
every NATS conf are now GENERATED from one source
(`aibots.hub.security.static_nats_grants`, in the hub repo) rather than
hand-written — hand-deriving these subject families had already failed twice, and
both drafts read green.

**Cold cutover, no shim.** There is no dual-read path and none will be added. For
a collection with an L3 tier a re-key is one cache miss; for a collection whose
L3 is `None` — the identity fence among them — L2 IS the source of truth, so the
keys must be copied before the new code rolls.
`14-eng-ai-bot/scripts/copy_l2_keys_into_scope.py` does that, and
`docs/runbook-l2-key-scope-cutover.md` is the procedure.

**Also in this branch:** `family_from_base.py` moved into `threetears-base` so
the hub and identity-core run one copy of the family-version check; the
`identity` bake target inherits that base and tags the names compose actually
consumes; the capabilities loader stopped discarding a legitimately-zero cost
(`a or b` treats `Decimal("0")` as absent, which silently unpriced every
embedding model); and the `_hub` principal gained the subscribe grant for a
subject its own responder serves.

**Known residual, pinned not hidden:** a deny-closed user keeps the coarse
`$JS.>`, so it can create a stream that `sources` the collections KV — the source
name rides in the request body where no subject permission sees it. Closing it
needs `$JS.API.STREAM.CREATE` constrained for `tool_server` and `identity`, which
needs identity's buckets enumerable from here. Recorded in
`static_nats_grants.py` and pinned by
`test_the_body_carried_sourcing_residual_is_pinned` so the set cannot grow
unnoticed.
