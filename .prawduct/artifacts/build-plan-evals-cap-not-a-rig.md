---
artifact: build-plan
version: 1
scope: evals-cap-not-a-rig
branch: fix/evals-cap-not-a-rig
depends_on: [evals-run-resume]
partition: "serial -- one small defect fix in one package; a second agent would cost more in coordination than it saves"
last_validated: 2026-10-24
---

## Requirements Confidence

**Level:** High

**Why:** The defect was reported by a consumer whose round-1 memo it blocks, and traced in code. Two launches on one
template with identical judge, cases, apparatus and k differed only in the launch argument `max_cost_usd` (0.75 and
3). Their arms read "no interval could be read: <arm> and <arm> ran under no test setup in common".

Cause: the core declares `max_cost_usd` as an apparatus `Sweepable`. `_apparatus_levels` records every apparatus
declaration, and `_apparatus_classes` digests them into `apparatus_class_id`, so the cap split cells. The documented
intent is the opposite: `docs/cost-and-budgets.md` says "The cap is not part of the measurement, so raising it is no
difference", and the owner's ruling (MIGRATIONS, 2026-10-23) is that nothing outside what decides a result is a
confound. `docs/concepts.md` contradicted this by listing the cap among the rig's inputs.

Problem: a cap difference splits the rig and is named as a confound.

Success: runs that differ only in their cap pool in one cell per arm, pair and read an interval. No confound list
names the cap. A real apparatus difference (the judge) still splits. The quick path's per-arm share of an `Eval`'s
cap splits nothing. The bisection still states the cap difference as a plain fact.

Out of scope: removing `max_cost_usd` as a declaration (kinds seat it: `JUDGE_KIND_CONTRACT`, `_REPORTER_SEATS`).
Changing how resume stores a raised cap. No package version bump.

**Decisions:**
- [DECISION: a `Sweepable.no_rig_coordinate: str | None` field, prose rather than a bool, mirroring
  `no_own_coordinate` | the module's idiom: taking an input out of an identity costs a sentence. Valid on apparatus
  only, refused beside `confounds` (which nothing would read) and when blank | user can veto]
- [DECISION: such inputs are left out of `_apparatus_levels` itself | that one map feeds the apparatus class, the
  campaign-wide confound scan, the lever cohorts, the divergences and factor aliasing, so one filter keeps all of
  them agreeing that a cap difference is no confound. The bisection reads the registry directly and still reports it |
  user can veto]
- [DECISION: `CELL_MODEL_VERSION` 13 -> 14 beside bundle shape 62 -> 63 | cells.py's rule: bump when a core dimension
  joins or leaves the apparatus coordinate. `host_declarations_digest` reads the flag, since it digests what
  partitions observations | user can veto]
- [DECISION: `test_uncapped_spend_ceiling`'s "unrecorded ceiling raises an undecided confound" changes to "raises no
  spend-ceiling confound" | an undecided confound claims the cap might have differed, and a differing cap is no
  confound. A changed requirement, stated in the report, not a weakened test | user can veto]

## Chunks

### Chunk 01: the cap is no dimension of the rig

**Delivers:** the flag and its registration checks, the filter in `_apparatus_levels`, the digest and the version
bumps, docs (concepts, cost-and-budgets), CHANGELOG, MIGRATIONS, the regenerated surface snapshot and reference page,
and tests: `test_spend_ceiling_is_no_rig.py`, plus a digest test in `test_bundle_mismatch_cause.py`.

**Acceptance:** the new tests pass and fail without the filter. The touched selections pass. Lint and the evals
typecheck are clean.

## Status

- Chunk 01: built, verified by selections, committed on fix/evals-cap-not-a-rig (not pushed). Critic not yet run:
  the coordinator runs it.
