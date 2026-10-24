---
artifact: build-plan
version: 1
scope: evals-depth-disclosure
branch: evals/depth-disclosure
depends_on:
  - artifact: no-assumed-pool   # #922's no-pooling-by-assumption ruling (old runs cut off, future runs pool forward)
partition: "serial -- chunk 02 reads the depth facts chunk 01 computes, and chunk 03 is an opt-in estimator over the same per-case counts; one builder, three commits"
last_validated: 2026-10-24
---

## Requirements Confidence

**Level:** High

**Why:** The owner asked for this on 2026-10-24, and discodon passed the request on with six numbered properties. The code
was read at c7e2ac24f the same day:

- pass^k per case is `C(c,k)/C(n,k)` (`kernel/scoring.py:395-404`), averaged over cases with n>=k.
- Cases short of k are dropped without being counted or disclosed.
- The frontier's k is the subject's PLANNED minimum `k_runs` (`lenses/frontier.py:1931`), not the depth actually delivered.
- The paired rate interval weights every shared case equally, whatever its depth (`stats.py:1355-1382`, `:1495-1541`). The "x of y" counts beside it are trial-pooled, and no Methods line says either.
- `variance_components` / `paired_case_variance` exist (`stats.py:2701-2799`), but only pre-launch `detectable_effect` uses them.
- No memo check is depth-aware. `memo_style.py` only refuses a proposed k that no run used.
- No model-based pass^k exists.

Problem: now that old runs pool forward, unequal depth is the normal case. Arms will have different run counts, and also
different counts across the cases within one arm. Readings, the campaign page and the memo bundle neither state the
depths nor say what a depth difference does and doesn't mean.

Success:
- Every reading and frontier point states each arm's depth.
- Every pass^k comparison names its common depth, and a depth some arm lacks is "unmeasured for <arm>", never silently
  dropped.
- Interval width is attributed to the side that limits it.
- The bundle carries this as structure, and prompt rule 5 tells the writer not to rank above the common depth. No code
  checks the memo's prose (owner, 2026-10-24: "as long as it does not bring back code checking prose").
- No existing value changes. This is disclosure, plus one opt-in, labelled model.

**Decisions:**
- [DECISION: the comparison depth is min(k): the fewest runs any arm gave a case every arm shares, over every shared
  case, so no shared case is left out for depth and a case some arm never ran is left out and counted. The page and the
  writer's view state it, and say where the arms share no case that pass^k compares nothing; the frontier still
  ranks at the planned k, and no code reads or refuses a memo sentence (owner, 2026-10-24: bad judgment is fixed by
  prompts and oversight, not code). | owner 2026-10-24: "measure pass^k for
  min(k)", and arms with more runs are read at that lower k. `C(c,k)/C(n,k)` is the exact mean over every k-subset of
  a case's n runs, so a deeper arm uses ALL its runs at the lower k with no model, and that tightens its own estimate.
  An earlier draft took the shallowest of the arms' deepest cases, which let one arm's single deep case leave every
  other case out; 6635f9e95 replaced it with the owner's rule | owner stated]
- [DECISION: when the common depth is 1, the disclosure says the comparison measures pass rate, not reliability | pass^1
  is the per-case pass rate | user can veto]
- [DECISION: the model-based estimate above an arm's depth is a beta-binomial over per-case p, OFF unless a host or
  launch asks for it. It is labelled "model-based, assumes ..." on every surface and is never ranked on or gated on. |
  owner request item 4 | user can veto; descoping chunk 03 does not affect 01-02]
- [DECISION: shape bumps -- `AnalysisContextBundle.schema_version` 62->64 (63 was the spend ceiling's; this branch folds into 64 unreleased), and `CAMPAIGN_BUNDLE_SHAPE` /
  `CAMPAIGN_JSON_SHAPE` wherever their emitted shape grows, each with a CHANGELOG line and a MIGRATIONS.toml entry. No
  package version bump. | #919 precedent; no version bumps in work PRs]

## Chunks

### Chunk 01: Depth facts
- `kernel/scoring.py`: the pass^k entry counts the cases short of k (`n_cases_short_of_k`) alongside `n_cases_at_k`.
- `AspectReading` / `PairReading` (`campaign_readings.py`) gain a depth record per arm: cases, trials, and repeats per
  case min / median / max.
- `contrast_samples` / `comparisons.py` carry each case's n alongside its share, so a reading can say how many shared
  cases differ in depth.
- `FrontierPoint` states the DELIVERED depth per arm, the common depth the comparison uses, and the count of cases left
  out at that depth. A point above an arm's depth is "unmeasured for <arm>".
- Precision attribution: each side's interval half-width, the comparison's half-width, which side limits it, and the
  between-case / within-case split from `variance_components` as the floor that extra repeats can't lower.
- Tests: hand-computed fixtures. Arm A k=2 and arm B k=1 on the same cases, mixed depths within one arm, a truncated
  run, and a k=1-only campaign. Every existing value stays byte-identical, and a test pins that.

### Chunk 02: Surfaces
- The campaign page shows depth beside each reading and on each frontier point, as a table, not prose (report style
  rulings).
- A Methods line states the pairing: per-case shares, equal case weights, and counts that are trial-pooled.
- The writer view (`campaign_bundle.py`) carries per-arm depth, the common depth, the cases left out and the precision
  attribution as fields.
- [DECISION: no memo check reads the writer's prose for depth; a built one was removed | owner 2026-10-24: "Bad
  judgment is fixed by improved prompts and/or oversight and calibration, not by code." | owner stated]
  The writer view carries the depth disclosure's plain words (e.g. "1017 was run twice on character, the others once;
  that tightens 1017's own estimate, not the comparison much") for the writer to say, and the page says them itself.
- `gen_prompt.py` / `GEN_PROMPT_RULES.md` get the rule.
- Shape bumps, CHANGELOG and MIGRATIONS. Docs: `docs/concepts.md` defines depth once.

### Chunk 03: Opt-in model
- A beta-binomial pass^k at an unreached depth, off by default and switched on by a host setting
  (`HostProfile.model_pass_hat_k`); no launch-level switch.
- [DECISION: method of moments, not empirical Bayes by maximum likelihood | closed form and deterministic (two builds
  read one figure), no optimiser to fail on a few cases, and it reads only the shares and depths the page states. Cost:
  noisier shapes than a likelihood fit on few cases, and two clamps at the edges (no spread reads μ^k; a variance past
  the beta's limit is held inside it). Opt-in, labelled and never ranked, so accepted | builder, 2026-10-24]
- Every value carries `basis: "model"` and its assumption text. Nothing ranks on it, and the page
  labels it.
