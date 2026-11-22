# Review: the re-root by role, as one cut-over (#1199)

Against `evals/bp-v` 8e2709c14 (on `evals/main` d8f423000) and the host's `/opt/venv` a78b15b66, read only, by
scripted passes and imports of the bp-v package in its own venv. No tests were run.

## Verdict: BUILD WITH THESE CHANGES

The role split is right: a first eval stays two names from one place, and a stdlib script with a text map fits a
host that pins commits. The stored-data claim mostly holds for the re-root (no defining module moves, `doc_type`s are
literals, digests and shipped response formats carry no old-root spelling, `owner_module` is not persisted), but
not as written in four places, and the script's idempotency proof has a hole.

## Findings, by severity

### 1. Six old-root names shadow submodules (C, B; high)

`run` exports the functions `regrade`, `definition_drift` and `judge_kind`; `analysis` exports `campaign_readings`,
`judge_drift` and `judge_kind_readings`. Each shares its name with a submodule of the same root (checked: each
root's `__all__` against `pkgutil.iter_modules`). Three consequences:

- **Not idempotent.** In token mode (§3.3), `threetears.evals.run.regrade.regrade` matches the `name` record
  `threetears.evals.run.regrade` as its longest prefix. Under the home target, the new value has the old value as a
  dotted prefix, so every run appends another `.regrade`. §3.4 checks the columns are disjoint as exact strings,
  which misses prefix matching. Twelve such tokens exist (`run/definition_seed.py` ×3, `analysis/said.py`,
  `report/build.py`, `campaign/{page,scorecard,aggregate}.py`, others).
- **Wrong under the public target.** `analysis.judge_drift.X` becomes `host.judge_drift.X`, and the exact-string rule
  would rewrite a logger name `"threetears.evals.analysis.judge_drift"` the same way.
- **Not loud.** With the inits emptied, `from threetears.evals.run import definition_drift` binds the module and fails
  only when called. The host's tests do this (`test_definitions_drift.py:26`, `test_seed_upgrade_path.py:304,343,716`,
  `test_persona_case_names.py:311`).

Change: R3 renames the six submodules (`module` lines; none may define a stored class or a pinned logger). R0
asserts that no old spelling is a dotted prefix of a new spelling or a module path.

### 2. "Every stored document class is on `shapes` by construction" is false (C; high)

16 of the 29 `EVAL_DOC_TYPES` classes are provisional, not declared: `CaseSet`, `EvalActionJob`,
`EvalAnalysisAttempt`, `EvalAnalystGap`, `EvalCassette`, `EvalGenerationSlot`, `EvalJudgeTemperatureCheck`,
`EvalLaunch`, `EvalReportDoc`, `EvalReportSession`, `EvalReportSnapshot`, `EvalSecondJudgePass`, `EvalSweep`, the two
tombstones, and `EvalReportPlan`, which no root exports at all. The host's walk (`discodon/audit/registry.py:340`,
`:372`) finds them only because importing a root loads their modules. Under Q1 = yes an unreached one leaves every
root, and done-when (7) passes by accident of transitive imports. Criterion 1 holds for every stored class (a host's
store receives it), but the walk cannot see that through the untyped store port.

Change: declare all 29 `received` on `shapes`, and ship an explicit doc_type-to-class mapping for the host's walk in
place of the `__subclasses__` side effect.

### 3. An eager `shapes` breaks package contracts (B; high)

§1.4 makes `shapes` eager for the doc-type walk. Three tests stand in the way, and the design names none:
`tests/test_import_time.py:51` (the job-lease numbers load nothing beyond `schema`; as `received` names they would
load every shape), `:45` (`shapes` would load `quick`: `Result`, `quick.HostCommand`) and `:77` (a budget of 10
ladder names, exceeded by `as_host`, `mount_fastmcp`, three harness names and the taught provisional names). The host
mirrors `DEFAULT_JOB_LEASE_S` to avoid the import (`discodon/db/pool_policy.py:45-46`), and `import
threetears.evals.schema` alone costs 1.15 s today.

Change: with finding 2's mapping, make `shapes` lazy like the package root. Keep `:51`, re-pointed. Replace `:77`
with the admission test, recorded as a replacement, never a weakening.

### 4. The script would rewrite history and its own fixtures (B; high)

Nothing excludes `MIGRATIONS.toml` (158 old-root spellings, including the `from`/`to` fields, plus the section R3
generates), the root `CHANGELOG.md` (140), R1's `tests/fixtures/reroot/before/` and `half/`, or an R3 test that spells
an old import to assert `ImportError` (done-when 2). So done-when (12), "zero changes after the cut", cannot hold.

Change: a default exclusion list, a `# reroot: keep` line pragma, and files marked `GENERATED` / "do not edit"
reported, never rewritten.

### 5. The host's generated artifacts are missing from the handover (B; medium-high)

`web/src/evals/ops-types.ts` is generated from `.prawduct/artifacts/openapi.json` (header lines 4-5), which is built
from evals model docstrings; `tests/unit/docs/test_openapi_drift.py` fails on drift. The script rewrites the `.ts`
file (the design's "4 `.ts` prose tokens") but skips `.prawduct`, so the drift test goes red after the re-pin.

Change: report generated files (finding 4). R4 tells the host to regenerate both files after re-pinning.

### 6. `src` prose should get the public spelling, not the home module (A, C; medium)

§3.2 rewrites every `name` occurrence in `src` to the defining module, docstrings included. Those docstrings reach
agents and hosts. Nine engine actions' parameter schemas already carry a root spelling (`run_launch` carries
`threetears.evals.ops.LaunchArguments`; also `sweep_launch`, `case_set_mint`, `campaign_update` and the `judge_*`
actions). They are served over MCP as `evals` and `evals_admin`, and they build the host's OpenAPI. A home spelling
teaches internal paths and ties wire prose to modules S6 and S9 move. In a response format it would also re-key
recordings; today the proposal schema names only `run.*` module paths, and the analyst mount names none.

Change: in `src`, imports go to the home module and prose to the public spelling.

### 7. The collision rule breaks convergence on a merged branch (B; medium)

§3.3 reports a file where `evals_<role>` is already bound. On a branch merged after the cut, the script's own earlier
pass made that binding, which is the case §3.4 promises converges. Change: reuse an identical binding
(`from threetears.evals import host as evals_host`), report any other, and add the case to the `half/` fixture.

### 8. The map needs versioned lines and a prefix rule (B; medium)

S6, S9 and #1013 add lines to one map, and a host pinned between moves would apply moves its pin lacks. A prefix
record whose new path extends its old path never converges. Change: each line carries the SHA it takes effect at; the
script takes `--to <SHA>`; the generator refuses a new value that has an old value as a dotted prefix (finding 1).

### 9. Import forms and strings §3.3 does not classify (B; medium)

Sampled in `/opt/venv` and the package:

| Form | Where | Gap |
|---|---|---|
| Prose strings that are not docstrings (attribute docstrings, `Field(description=...)`, assertion messages) | `discodon/config/sections/eval.py:38,119,246,358,371,415`; `test_openrouter_generator_client.py:633` | In no class, so silently skipped and left stale; they feed the host's config docs |
| Trailing comments on a split statement | 26 statements with `# noqa: E402`, `scripts/measure_read_budgets.py:37-48` | Canonical lines drop them |
| An old-root statement that imports a submodule | `from threetears.evals.analysis import service as analysis_service` (`:38`) | No record; must be left alone |
| Bare `import threetears.evals.kernel`, no attribute read | Side-effect imports | Unspecified; must be reported |
| 3.14-only syntax | PEP 758 `except A, B:` at `analysis/service.py:747`, `providers/retry.py:109`, 18 more | Python 3.12 (§3.1) cannot parse it. Require 3.14, and refuse a file that fails to parse before rewriting |

Covered on the host: `TYPE_CHECKING` (8), `mock.patch` targets, `importlib`, paired `seed_module`, the logger pin.

### 10. The emptied inits are loud only for `from X import name` (B, A; medium)

`import threetears.evals.schema` and `importlib.import_module("threetears.evals.kernel")` still succeed. That is what
`registry.py:340` does, and its comment relies on failing. `from threetears.evals import ops` then `ops.X`
(`discodon/web/routes/evals.py:26`) fails at request time, not at startup. Change: each emptied init gets a
`__getattr__` raising `ImportError("<old> moved to <new>; run reroot.py (map at <SHA>)")` for mapped names, and
`AttributeError` otherwise, so submodule imports keep working. It serves nothing, so it is not a shim.

### 11. Some host-test replacements in §4.1 are unproven (B; medium)

Gate G1, the keyless matrix (`tests/integration/database/test_keyless_matrix_live.py:50,417-438`), runs every kind
as one mixed group through `launch_as_group`, `LaunchGroup` and `start_run`; nothing shows `run_launch` or
`sweep_launch` can. R-a admits the reader-conformance kit, but the host builds its sample runs with `start_run` and
`mint_case_set` (`test_reader_conformance.py:29,75-116`), so the kit loses its public way to make runs.
`start_run(LaunchHost, ...)` returns runs; `run_launch(OpsHost, LaunchArguments, scope_id)` returns `JobsStarted` and
needs a poll loop, and `await_job` is demoted too.

Change: before publishing the map, port the 14 host files that use `start_run`, plus G1 and reader conformance, on a
scratch branch that never merges. Any name with no real replacement stays public as the only door, or a kit-level
helper is admitted under R-a.

### 12. `shapes` is a bucket, not a role (A; medium)

The 258 `received` names are 89 models, 50 dataclasses, 54 constants, 29 exceptions, 15 `Literal` aliases, 10
protocols (`JobStore`, `ResultStore`, `RunStore`, ...), 2 seed templates and 2 registries (`SHARED_CORE`). Q1 = no
would add about 650 more. The host raises `ValidationFailedError` 33 times and `ApparatusError` 12 times, raises
`StoreConflict` from its store (the port's own contract), and subclasses `StorageError`. That is bound use, which IA
§8.1's tie-break ("an exception is received") misplaces. Shapes should stay importable (mypy-strict hosts annotate
with them; finding 2 needs them), but a reader should meet a shape under the name that returns it.

Change: R0 places by observed use where V's tie-break disagrees. An error a host raises or subclasses, a protocol it
implements, and a registry, constant or seed template it passes all go to `host`. `surface.md` and `llms.txt` fold
each shape under its reacher (IA §8.4, row 3).

### 13. `action`-kind functions sit in `host`, not `actions` (A; low-medium)

`run_launch`, `job_poll`, `report_read` and 10 more go to `host`, where an agent told "the actions root" will not
look. Change: one generated line, "the same operation in Python is `host.<action name>`"; `llms.txt`'s KINDS becomes
WHERE NAMES LIVE (five lines), NEEDS A SHIPPED EXAMPLE goes, and done-when keeps it within today's 136 lines.

### 14. The map freeze conflicts with work in flight (process; medium)

On-ramp O4 removes `quick.measure`/`Measure` and "rides the re-root's demotion list" (`design-on-ramp.md:397`);
this design places both at the package root (§1.1 line 37, §1.3). The on-ramp says renumbering rides the cut-over
(`:78`, `:420`); this design's done-when (11) assumes today's rungs 4 and 5. O2 "can start now" and adds
`providers.ModelRates` (`:110`); judges R1 (`Check`) and the remedy actions add public names on branches cut from
f9ab6f74d.

Change: O4 rides R3. O6 renumbers after R3, and the on-ramp text is corrected. The map is generated at R3's base and
frozen there. A late name never gets a map line, since it never shipped at an old root. It is placed by `role_root()`,
which R0 wires into the admission test. O2 places `ModelRates` by rule, at the package root because it is typed.

### 15. design-structure.md is superseded without saying so (process; medium)

The cut-over contradicts three parts of it: §4.3 (tests import from roots or subpackage inits; one-minor shims), Q6
(a ratchet on below-root test imports that may only fall, now 3,052, which `--home-for-left` raises by about 177
files) and §7 "The pinned host" (no declared name moves). Change: amend all three in R3's PR, saying what replaces
each. Never silently drop a requirement.

### 16. R2 moves goldens its done-when does not name (B, C; medium)

The courier feeds `test_hot_path_oracle`, `test_identity_golden_keys`, `test_evidence_fingerprint` and 15 more test
files. Replacing `model.endswith("-pro")` (`courierhost/__init__.py:300`) with a `solver` knob and giving
`COURIER_CONTRACT` (`:235`) an overlay changes sweepables and identity. Change: write each use so its default
reproduces today's cells (the solver default equals today's rule, `outage` off unless a variation sets it, the action
additive). Done-when: courier goldens unchanged, or the list stated with reasons (the 11-06 goldens ruling).

### 17. R5 is mostly busywork (maintainer; low-medium)

`--home-for-left` already keeps tests honest. Tests whose subject is the function legitimately stay on it: the 91
bundle tests, the 32 HTML-page tests, the CLI tests, runner mechanics, `regrade`'s arithmetic, `closest_settings`.
Change: replace the per-file list with one rule a test holds: each action replacing a demoted function has a
catalogue-level test of its result and of one refusal with its remedy. Write those where missing; migrate nothing
else.

### 18. Smaller items

- **Names.** Rules R1–R6, chunks R0–R5, judges R1, remedy R0–R3 and ruling R-a collide. The ledger already calls the
  chunks X0–X5; use those in the design, and "rule 1–6" for the rules.
- **Snapshot format.** R0's format 4 conflicts with every branch that regenerates `surface.json`. R0 computes
  `defined_in` in its own generator; format 4 lands in R3.
- **Identifiers that follow names or paths, not roots.** The re-root moves none, but §2 and S6/S8/S9's done-when
  should name them: `contained_site` = `module:qualname` in every `contain()` log line (`schema/logs.py:73,102`);
  `result_model`, a class name stored on job records (`run/job_work.py:549,611`); stored error text
  `type(fault).__name__` (`run/runner.py:1619-1623`); package-data paths `files("threetears.evals.analysis.analyst")`
  and `...analysis.report` (`skill_ledger.py:101`, `skill.py:213`, `kit_colours.py:21`, `html_parts.py:34,39`). S8's
  private renames change the first three for new records. The data paths fail loudly, and `module` lines cover them.
- **Measurements (my AST pass).** Production 59 files / 146 statements / 392 names (design 69 / 141 / 386); tests
  108 / 366 / 601 (design 113 / 369 / 601). Close enough; the dry-run replaces them.

## The stored-data claim, for the later moves

For the re-root it survives once findings 1 and 2 are fixed: the shipped response formats (proposal, simulator,
variation; judge and blind are built by hand) carry only paths of unmoved `run.*` modules. S6 and S9 are safe if §6's
zero-miss, unchanged-digest assertion covers S8 too and finding 6 keeps home paths out of `src` prose. The second
distribution (design-structure §8) moves the data paths and the writer's tool schemas; design that then.

## The two owner questions

- **Q1 (about 630 unreached provisional names leave the roots): yes, by criterion 1, with three conditions.** The walk
  counts the doors it cannot see as reached (stored classes, finding 2; action result models served as JSON). R0 uses
  the closed-surface test's own walk, so P3 and Q1 cannot disagree. The exact list, with the host's 26 test imports,
  goes to Discodon before R3.
- **Q2 (`providers` rides the cut-over): no.** #1013 crosses distributions and has its own design. Put `ProviderError`
  in `host`, since #1013 keeps it in evals, and keep `Model` and `ModelRates` at the package root, served lazily from
  wherever #1013 puts them. Then only `Catalogue` and `PriceRefresh` move twice, which needs finding 8's versioned map.

## Changes to make to the design before a builder starts

1. R3 renames the six shadowing submodules; R0 refuses dotted-prefix overlap (findings 1, 8).
2. Declare all 29 stored classes on `shapes`; ship a doc_type-to-class mapping for the host's walk (finding 2).
3. Make `shapes` lazy; keep `test_import_time.py:51`; replace `:77` with the admission test (finding 3).
4. Script: an exclusion list (CHANGELOG, MIGRATIONS, the map, the reroot fixtures), a `# reroot: keep` pragma,
   generated files reported only, and done-when (12) restated over that (findings 4, 5).
5. In `src`, imports go home and prose gets the public spelling (finding 6).
6. Reuse the script's own earlier binding; add the case to `half/` (finding 7).
7. Map lines carry an effective SHA; add `--to <SHA>` (finding 8).
8. Classify finding 9's forms: non-docstring prose strings, trailing comments, submodule imports, bare imports;
   require Python 3.14 and refuse a pre-parse failure.
9. Emptied inits raise `ImportError` naming the new spelling for mapped names (finding 10).
10. R4 ports G1, reader conformance and the host's `start_run` uses on a scratch branch before the map is published
    (finding 11).
11. R0 places by observed use (raised, subclassed, implemented or passed = `host`). Docs fold each shape under its
    reacher. `llms.txt` gains WHERE NAMES LIVE and the actions-in-host line, within 136 lines (findings 12, 13).
12. Freeze: the map is generated at R3's base. O4 rides R3; O6 renumbers after it; O2 places `ModelRates` by rule;
    `role_root()` in the admission test holds late names (finding 14).
13. Amend design-structure §4.3, Q6 and §7 in R3's PR (finding 15).
14. R2 keeps today's cells; its done-when names the courier goldens (finding 16).
15. R5 becomes the per-action catalogue-test rule (finding 17).
16. Chunks X0–X5, rules 1–6; snapshot format 4 in R3; finding 18's rows in §2.

## Order, and what can start now

Cut-over first, then S6, S8 and S9: right. R1 can start once changes 4 and 6–8 are in §3; R0 once changes 1, 2 and 11
are in §1; R2 once change 14 is written (V is on `evals/main`). R3 waits for all three, batch 2 and the change 10
port, then lands at the next quiet point with O4 folded in; wave-3 branches merge after it and place names by rule.
