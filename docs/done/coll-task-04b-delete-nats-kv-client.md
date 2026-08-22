# coll-task-04b: Delete the retired `NatsKvClient`

## Objective

Delete `NatsKvClient` and everything that points at it, across two repos.

Split out of `coll-task-04` (now `coll-task-04a`), which carried this as KVC-09.
It shares no code path with that shard's reconcile primitive: 04a builds a
JetStream create-or-reconcile arm whose failure mode is a startup crash-loop,
and this is a deletion whose only risk is a patch target disappearing from
under a shared test fixture. Reviewing them together is how the second one gets
skimmed.

Facts cited here live in the evidence ledger
(`14-eng-ai-bot/docs/collection-support-evidence.md`).

---

## Why it is retired, and why deletion rather than repair

`NatsKvClient` in `packages/core/src/threetears/core/cache/kv.py` has **zero
production construction sites**. The SDK marks it retired in
`bootstrap/phases/backend.py` and `devx/workspace_runtime.py`. Every live KV
path goes through `NatsClient.kv_bucket` / `NatsKvBucket`.

It is not inert, though. It carries three things worth removing rather than
leaving:

- **A stale storage declaration.** Its docstrings claim `file` storage while
  `BucketConfig.storage` defaults to memory. Its TTL half is self-consistent;
  only storage is stale. This matters beyond tidiness because the hub's
  CLAUDE.md cache carve-out *cites `NatsKvClient.storage` as the authority* for
  the memory-storage decision. The real authority is `NatsClient.kv_bucket`'s
  `storage: str = "memory"`. **04a corrects that citation**; this shard removes
  the thing it used to point at, so the two must not be reordered.
- **A fail-open swallow.** `NatsKvClient.connect` wraps its open in
  `except Exception: log.warning(...)`, commented "fail-open". `coll-task-04a`
  deliberately leaves it alone on the grounds that this shard deletes the whole
  class. If this shard is ever dropped, that catch must instead be narrowed -- 
  do not leave it as-is on the assumption the deletion is coming.
- **The 7200 s TTL** that `coll-task-04a` warns against citing as evidence of
  config drift. Deleting the class removes the misleading exhibit.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| KVD-01 | `NatsKvClient` is deleted; no import, reference or patch target remains in any repo | P0 |
| KVD-02 | `BucketConfig`'s fate is decided explicitly -- deleted with it, or kept with a recorded reason and its stale storage docstring corrected | P0 |
| KVD-03 | The SDK's shared conftest no longer patches the deleted path | P0 |
| KVD-04 | The enforcement catalog entry naming it is removed | P1 |

---

## Decision: `BucketConfig` is deleted with the class

Checked before deciding. Outside `cache/kv.py` itself and its own unit test,
`BucketConfig` had no consumer in any of the seven repos -- the only other hits
were prose in the CHANGELOG, these shards, and `14-eng-ai-bot-agent-map`'s
`docs/geo-tile-caching-design.md`. The three `TestBucketConfiguration` classes
under `packages/core/tests/unit/coordination/` are unrelated: they name a test
class, not this type.

It existed solely to describe a bucket for `NatsKvClient.connect`, and the whole
of what it described -- suffix, TTL, storage -- is now expressed as keyword
arguments to `NatsClient.kv_bucket`. Keeping it would have left an orphan
dataclass no code constructs, which is the same "reader follows a citation and
finds no live code path" failure the stale storage docstring already caused. So
the whole module goes: `BucketConfig`, `NatsKvClient`, `_PING_TIMEOUT_SECONDS`,
and `packages/core/src/threetears/core/cache/kv.py` itself.
`threetears/core/cache/__init__.py` never exported either name, so no
`__all__` edit was needed.

---

## Files to Modify

- `packages/core/src/threetears/core/cache/kv.py` -- delete `NatsKvClient`; decide `BucketConfig`.
- `packages/core/tests/test_kv_client.py` -- deleted with it (24 tests).
- `packages/core/src/threetears/core/coordination/replay_guard.py` and
  `packages/nats/src/threetears/nats/client.py` -- module docstrings name the
  class as the fail-open counterexample / migration ancestor.
- `3tears/tests/enforcement/test_dict_state_detection.py` -- a catalog entry names `NatsKvClient`.
- `14-eng-ai-bot-agents/src/aibots_agents/bootstrap/phases/backend.py` and
  `devx/workspace_runtime.py` -- the comments marking it retired now name a class
  that does not exist. Correct or remove them.
- `14-eng-ai-bot-agents/tests/unit/runtime/conftest.py` -- **the sharp edge.** It
  patches `threetears.core.cache.kv.NatsKvClient` *by path*, and
  `mock.patch` raises on a target that no longer exists. This is a SHARED
  fixture, so getting it wrong fails a broad swathe of the SDK suite rather
  than one test, and the traceback names the fixture rather than the deletion.

---

## Anti-patterns

- DO NOT leave a deprecation alias or re-export. The no-shims rule is absolute; a caller updates in the same commit.
- DO NOT narrow the `connect` swallow instead of deleting. That is the fallback if this shard is abandoned, not a substitute for it.
- DO NOT grep only for the symbol. It is patched **by dotted path string** in at least one conftest, so a symbol-only search misses the site most likely to break the suite.

---

## Success criteria

- [x] `NatsKvClient` appears nowhere in any of the seven repos, as symbol or as a dotted-path string. The symbol is gone entirely; the surviving hits are past-tense prose in records of the deletion itself (this shard, `coll-task-04a`, `coll-sequence.md`, the CHANGELOG, `14-eng-ai-bot/docs/done/*` and the evidence ledger), which are history and stay
- [x] `BucketConfig`'s fate is decided and the decision is recorded -- deleted with the class; see **Decision** above
- [x] `./scripts/check-all.sh` green -- **15843 -> 15811 passed, 3 skipped, 410 deselected**, plus 139 sidecar unchanged
- [x] SDK unit suite green -- **2984 -> 2983 passed**, the delta being `test_bootstrap_does_not_construct_nats_kv_client`, which asserted the absence of a class that can no longer exist
- [x] The hub CLAUDE.md carve-out no longer cites the deleted class as its authority (04a does this; confirmed at `14-eng-ai-bot/CLAUDE.md:375`, which now cites `NatsClient.kv_bucket`)

### The drop is 32, not 24

`test_kv_client.py` carried 24 tests, and four enforcement suites parametrize
one case **per source file**, so deleting `cache/kv.py` removes eight more:
three in `packages/core/tests/enforcement/test_docstring_conventions.py`, two in
`packages/registry/tests/enforcement/test_no_hardcoded_timeouts.py`, one in
`tests/enforcement/test_uuid_stringification.py`, and two in
`tests/enforcement/test_uuidv7_enforcement.py`. Verified by diffing
`--collect-only` output either side of the deletion: every one of the 32 is
accounted for and nothing else moved.

Removing the `NatsKvClient` entry from the dict-state allowlist changes no
count -- that catalog is not parametrized -- but it is not optional either:
`TestDictStateDetection.test_no_stale_allowlist_entries` fails on an entry
naming a file that no longer exists.

---

## Verification

```bash
cd 3tears
./scripts/check-all.sh
cd ../14-eng-ai-bot-agents
uv run pytest tests/unit/ -q
```

A repo-wide search for the dotted path, not only the symbol, is the check that
matters:

```bash
grep -rn "NatsKvClient" --include=*.py --include=*.md .
```
