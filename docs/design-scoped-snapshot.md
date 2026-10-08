# Scoped snapshot: whole tables in L2

`threetears.core.collections.scoped_snapshot.ScopedSnapshot` keeps tables that a pod needs whole
in NATS (the L2 tier), and loads them into a DuckDB L1. It also covers the pod-owned NATS pieces
under it: `threetears.nats.object_store`, `NatsKvBucket.watch_prefix`, a tool pod's Object Store
grant, and `threetears.nats.object_store_requests`.

## Decision

**Pace chose NATS L2 over S3 or parquet files (2026-10-07).** The question came from the ENR tool
pod. A starting replica waited 22 minutes reading its report tables from L3, a thousand rows a
statement. After every refresh it spent about 5 minutes rebuilding its whole DuckDB copy. The first
proposal was a parquet file in object storage. Pace asked instead how 3tears' L2 could hold the
copy properly.

A probe on the local stack (nats-server 2.12.6) settled the shape:

- One epoch of the candidates table plus its geography, split by state, is **16.9 MB** as Arrow
  IPC with zstd. Per-state chunks run from 7.8 KB to 1.69 MB, so all six tables are likely under
  50 MB. That fits the local NATS and cobalt's 1 GiB memory store per server with no setting
  changed.
- Loading 102 chunks into DuckDB took **0.57 to 0.59 s**.
- Swapping in one state took **62 to 83 ms** end to end. About 10,000 reader queries ran during the
  swaps, and none saw a torn read.

S3 or parquet stays a fallback only. L3 is the truth, and NATS is a cache rebuilt from it.

## Shape

- **One chunk per scope and table.** Each is one Arrow IPC stream, zstd-compressed, in the pod's
  own Object Store. It is named `{snapshot}/{scope}/{epoch}/{table}.{column digest}`. The digest
  means a rebuild under other columns never reuses an old chunk's name.
- **One pointer per scope.** It is a small KV entry naming the scope's epoch and its chunk for
  every table, so a scope's change across tables is one swap.
- **An index.** It names every scope, so a missing pointer is noticed, not skipped.
- **A rebuild claim.** It is a KV entry with a TTL, renewed while the holder works and released
  only by compare-and-set.
- **Compare-and-set on pointers and the index.** A pointer never moves to a lower epoch. The
  index merges what racing writers added. A replica's L1 never goes back to a lower epoch of a
  scope.
- **Every row has a scope.** A publish with no scope is refused. A rebuild that finds rows with a
  null scope fails and names the table, instead of serving a copy that claims to be whole and
  isn't.
- **Retirement.** Superseded chunks are deleted by the bucket's declarer (the hub), in batches of
  at most `MAX_RETIRED_OBJECTS`. A pod holds no purge. A reader that loses a race to a retired
  chunk reads the scope's pointer again.

## The pod's own buckets

A tool pod gets an Object Store and a pointer bucket only when its registry row opts in. The hub
declares both when the pod asks. Both are bounded: the Object Store by `max_bytes` with
`discard: new`, and the pointer bucket by `max_bytes` too. Neither allows rollup headers, because a
rollup is a purge done by publishing. The hub deletes both when the pod is removed.

The pod's grant (`JsCapability.OBJECT_STORE_OBJECTS`) allows:

- `STREAM.INFO` (bind)
- a direct get of metadata
- a named-consumer create filtered inside the bucket
- the `$O.` publish

It never carries a stream-management verb, `STREAM.MSG.GET` or `STREAM.MSG.DELETE`.

## Known residual: a pod can put its own bytes on any subject

**This is a platform hole. No grant closes it.** It is pinned by
`packages/nats/tests/integration/test_a_pod_can_bounce_its_bytes_to_any_subject_residual_live.py`.
That test asserts the hole exists, so when it starts failing, the hole has closed: remove the test
and this section.

Probed on 2026-10-07 against nats-server 2.12.6, with a real minted tool-pod JWT applied as config
permissions:

1. **Push-consumer delivery is not checked.** A named push consumer on the pod's own bucket is
   admitted by the grant, because its filter rides in the create subject. Its `deliver_subject`,
   though, rides in the request body, which no subject permission sees. The server delivered the
   pod's stored bytes to `bounce.tools.internal.<another pod>` and to another principal's inbox,
   although the pod may publish to neither.
2. **Reply subjects are not checked.** Every request the grant admits is answered on the request's
   reply subject, and NATS never checks that subject against the requester's permissions. A direct
   get of a value the pod wrote came back verbatim on a subject the pod may not publish. The same
   holds for pull `MSG.NEXT` and any JetStream API call.

So changing the read shape (pull consumers, ordered consumers, direct gets) closes nothing. Any pod
that can write a value and read it back has this, through every L2 grant:

- its own KV buckets and Object Store
- its owner keys in the shared buckets
- its scope of the collections bucket

**The defence belongs to the receiver.** If a subject's receiver trusts the subject alone (no
signed token, no correlation id it minted itself), a pod can feed it the pod's own bytes. Pace
decided on 2026-10-07 to audit those receivers; their fixes are separate work. The other route is
separate NATS accounts per principal with explicit exports and imports, since deliveries and
replies stay inside an account.
