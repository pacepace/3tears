---
artifact: build-plan
version: 1
scope: guard-anchor-and-read-all-fingerprint
branch: feat/durable-coordination
depends_on: [durable-coordination]
partition: "serial -- two unrelated chunks that share one release; each is small enough that a second agent would spend more on coordination than it saves, and Chunk 01 must land before identity and hub can be rewired"
last_validated: 2026-09-16
---

## Requirements Confidence

**Level:** High

**Why:** Both problems were observed, not inferred. Chunk 01's was reproduced from identity's
integration suite: 16 failures, every one logging `ReplayGuard refused an artifact issued before
its bucket was created` followed by `dpop proof rejected at mint` and `password login denied`,
clustered in the first minute of the run. Chunk 02's was raised by Pace reviewing PR 166 in
`bl-eng-client-delivery` and relayed with the mechanism spelled out.

Problem: `ReplayGuard` cannot distinguish a bucket created for the FIRST TIME EVER from one
recreated after a wipe, so it applies its fail-closed watermark to both -- and on a first
creation nothing was ever recorded, so the refusal protects nothing while refusing every
artifact for the verifier's tolerance plus the drift allowance. Separately, `read_all` proves
completeness with a row COUNT, which a delete plus an insert during the read leaves unchanged.

Success: a guard given a durable anchor accepts immediately on a first-ever creation, still
refuses for the full window after a genuine wipe, and identity's integration suite is whole.
`read_all` raises when the relation changed under it, including when the row count did not.

Out of scope: the registry server and the tool pod keep today's anchorless behaviour (see the
decision below). Where DPoP nonces are stored does not change -- they stay in memory-backed KV,
and no coordination write moves onto a per-request path.

**Decisions:**
- [DECISION: the watermark stays; the anchor only tells the guard whether there is anything to
  fail closed ABOUT | `build-plan-durable-coordination.md` records "nonces stay memory, fail
  closed by creation-time watermark" as Pace's call on 2026-09-15, and that holds for a wipe.
  A first-ever creation is not a wipe: no artifact was ever recorded, so no replay is possible,
  and the refusal is pure cost | user can veto]
- [DECISION: the anchor is an optional collaborator, not a required one | the registry server
  (`registry/server.py:739`) and the tool pod (`agent/tools/server.py`) deliberately take only a
  NATS client -- the tool pod's own comment says it holds no creds of its own. Requiring an
  anchor means wiring durable storage into two components built to avoid it, to save a minute of
  refused internal RPC that retries. Identity and the hub carry human-facing login and already
  hold pools | Pace, 2026-09-16]
- [DECISION: the anchor is a narrow Protocol, with a coordination-table implementation shipped
  beside it | keeps `ReplayGuard` free of a `CollectionRegistry` dependency, so a consumer
  without L3 still constructs one. "3tears first, never bespoke" is satisfied by shipping the
  collection-backed implementation rather than leaving each consumer to invent one | user can veto]
- [DECISION: the fingerprint REPLACES the count rather than joining it | same number of queries,
  so the change is cost-neutral, and a count that the fingerprint does not already subsume is
  not worth a second round trip | relayed from Pace via the delivery session, 2026-09-16]

**Open assumptions:**
- ~~[ASSUMPTION: one fingerprint expression is portable across Redshift, Snowflake, BigQuery,
  Postgres and Yugabyte]~~ **WRONG, and it changed the design.** Turning a hash into a summable
  number has no shared spelling: Postgres casts through `bit(32)`, Redshift has `STRTOL`,
  Snowflake has `TO_NUMBER` with a format model. `read_all` also cannot choose per dialect --
  the query wire carries no datasource type. So the fingerprint became a DRIVER method with a
  declarative ask on the wire, which is Pace's option C, chosen 2026-09-16 over a weaker
  portable approximation. Two further facts found while settling it: BigQuery is a STUB (every
  method raises), so the live set is Postgres/Yugabyte, Redshift and Snowflake; and BigQuery
  will need a dialect seam in the shared key-expression builder, because its `MD5` returns BYTES
  and its cast is `STRING` rather than `VARCHAR`. Recorded at that driver's call site.
- [ASSUMPTION: NULL key values contribute to the fingerprint | HIGH impact -- a NULL row swap
  goes unseen otherwise | Chunk 02 tests it explicitly]
- [ASSUMPTION: identity and hub both have a coordination-backed registry available at the point
  their DPoP guards are constructed | MED impact -- identity's is confirmed (`self.collections`
  exists by then); hub's is not yet read | Chunk 03 settles it]

**What would raise confidence:** Chunk 02's fingerprint expression exercised against a live
instance of each dialect, rather than against a fake.

## Status

- [x] Chunk 01: The first-creation anchor
- [x] Chunk 02: read_all proves completeness by fingerprint
- [ ] Chunk 03: Wire identity and hub onto the anchor

## Chunk 01: The first-creation anchor

**Delivers.** A `ReplayAnchor` Protocol and a coordination-table implementation, plus an
optional `anchor=` on `ReplayGuard`. With an anchor, the guard records the moment this purpose's
ledger first existed and reads it back on the first record. No anchor row means the ledger has
never existed, so the watermark is skipped. An anchor older than the bucket's creation time
means the bucket was wiped, and the watermark applies exactly as it does today.

**Files.** `packages/core/src/threetears/core/coordination/replay_guard.py` (the Protocol and
the optional collaborator), a new module beside it for the collection-backed implementation,
`packages/core/src/threetears/core/coordination/__init__.py` (exports), and unit tests under
`packages/core/tests/unit/coordination/`.

**Acceptance.**
- A guard with no anchor behaves exactly as today, watermark included. The existing tests prove
  this without modification -- they are the regression contract for the anchorless path.
- A guard with an empty anchor accepts an artifact issued now, against a bucket created now.
- A guard whose anchor predates the bucket's creation still refuses inside the window.
- The anchor is read once per guard, not once per record.
- Fail-closed is preserved: an anchor that cannot be read denies, never admits.

**Done when.** The above are tested, `uv run pytest packages/core/tests/unit/coordination/` is
green, ruff and mypy are clean, and the CHANGELOG names the behaviour change.

## Chunk 02: read_all proves completeness by fingerprint

**Delivers.** `read_all` asks the datasource for one fingerprint of the relation computed over
the ordering key, before paging and again after the last page, and raises when the two differ.
This replaces the `COUNT` query.

**Files.** `packages/datasources/src/threetears/datasources/query_client.py` and its tests.

**Acceptance.**
- A relation mutated during the read raises, including a delete plus an insert that leaves the
  row count unchanged -- the case a count cannot see.
- NULL key values contribute to the fingerprint, so a NULL row swap is detected.
- The expression used is valid on each admitted dialect. Settle this BEFORE implementing:
  row-constructor syntax is already known not to be portable.
- The docstring stops promising what the count could not deliver.

**Done when.** The above are tested, the datasources suite is green, ruff and mypy are clean,
and the CHANGELOG names the contract change.

## Chunk 03: Wire identity and hub onto the anchor

**Delivers.** Identity's DPoP guard and the hub's two guards construct a coordination-backed
anchor. Identity's integration suite goes whole.

**Files.** `identity_core/server.py`, and in the hub `aibots/hub/app.py` and
`aibots/hub/security/dpop_binding.py`.

**Acceptance.**
- Identity's integration suite passes as a whole run AND file by file -- the ordering dependence
  is the symptom that started this, so a green full run alone does not close it.
- The registry server and tool pod are untouched, with the reason recorded at their call sites.

**Done when.** Identity is green both ways, the hub's own suite is green, and both repos are
repinned to the released 3tears version.
