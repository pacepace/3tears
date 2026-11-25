# Design: the re-root by role, as one cut-over (#1199)

Read against `evals/bp-v` at 8e2709c14 and the production host's checkout at `/opt/venv` (develop, a78b15b66), read
only, by scripted passes and imports of the bp-v source. Decisions already made (single cut-over, no re-exports, an
idempotent script, a text map, the registered demotions, shipped uses for the four, evals roots only) are not
reopened. Chunks are X0 to X5; the placement rules are "rule 1" to "rule 6".

## Revision 2 (2026-11-15)

The review (`design-reroot-cutover-review.md`) was checked finding by finding against the code and the host. All 18
findings hold, and none is rejected. Two are clarified rather than taken as written (finding 11, finding 12's
"passed"). What changed:

| # | Change | Why (verified at) |
|---|---|---|
| 1 | The cut-over renames six submodules whose names are also public functions of the same old root (§1.5); a structural test forbids it recurring | `run` exports `regrade`, `definition_drift`, `judge_kind`, and `analysis` exports `campaign_readings`, `judge_drift`, `judge_kind_readings`, each also a module of that package (`__all__` against `pkgutil.iter_modules`). Prefix rewriting appended a segment every run, and the host's `from threetears.evals.run import definition_drift` (`tests/unit/evals/test_seed_upgrade_path.py:304,343,716`, `test_persona_case_names.py:311`) would bind the module silently |
| 2 | All 29 stored document classes are declared on `shapes`; `shapes.EVAL_DOC_CLASSES` maps each `doc_type` to its class for the host's walk. Rule: a stored document class and an action's result type are reached by definition | 13 of 29 were declared, 15 provisional, and `EvalReportPlan` is exported by no root (checked by import) |
| 3 | Every sub-root is lazy (name → home table), as the package root is; `test_import_time.py:51` and `:45` are kept and re-pointed; `:77` is replaced by the admission test | `test_import_time.py:51` (the lease numbers load only `schema`); the host mirrors the lease to avoid the import cost (`discodon/db/pool_policy.py:45-46`) |
| 4 | Script: default exclusions, a `# reroot: keep` pragma, generated files reported and regenerated instead, done-when (12) restated | `MIGRATIONS.toml` and `CHANGELOG.md` record history; the host's `web/src/evals/ops-types.ts:4` is "GENERATED ... do not edit" and drift-tested (`tests/unit/docs/test_openapi_drift.py`) |
| 5 | Map lines carry the epoch (SHA) they take effect at; `--to <SHA>`; the generator refuses a new value that equals an old value or has one as a dotted prefix; the script reuses its own earlier binding; Python 3.14 required | PEP 758 `except A, B:` at `analysis/service.py:747`, `providers/retry.py:109` |
| 6 | Host-test replacements are proved (X4, part 1) before X3 freezes the demotion list. By the criteria, `start_run`, `launch_as_group` and `assemble_context_bundle` stay public as `tests` (§4.2) | G1 forms one mixed-kind group (`tests/integration/database/test_keyless_matrix_live.py:417-440`); `start_run` carries `launch_group`, `cassette_mode`, `judge_config_ids`, `max_metered_calls`, `margins` (`run/launch.py:2007-2040`), which `LaunchArguments` (`ops/runs.py:190-248`) does not; `inspect_campaign_bundle` returns a `BundleInspection`, not the bundle the host's tests read |
| 7 | Later moves (S6, S8, S9) carry stored-name pins: `result_model`, error-text class names, `contained_site`, package-data paths (§6) | `run/job_work.py:549,611` stores `type(result).__name__`; `run/runner.py:1619-1623` stores `type(fault).__name__`; `schema/logs.py:73,102` |
| 8 | Placement by observed use: what a host raises, subclasses, implements or passes goes to `host`, not `shapes`; docs fold each shape under its reacher; `shapes` is kept importable (§1.6) | 11 received names are raised or subclassed by the host or the courier (scan: `ValidationFailedError`, `ApparatusError`, `StorageError`, `StoreConflict` and 7 more) |
| 9 | In `src`, imports go to the home module and prose to the public spelling | Nine engine actions' parameter schemas carry root spellings, served over MCP and into the host's OpenAPI |
| 10 | Emptied inits raise `ImportError` naming the new spelling for a mapped name | `import threetears.evals.schema` alone still succeeds; `registry.py:340` relies on failure |
| 11 | Freeze: the map is generated at X3's base. O4 (`@measure` removed) rides X3; O6 renumbers after X3; `ModelRates` is typed, at the root | `design-on-ramp.md:78,105-110,397,420` |
| 12 | X2 keeps today's courier cells by default, and its done-when names the courier goldens | The courier feeds `test_identity_golden_keys`, `test_hot_path_oracle` and others |
| 13 | X5 is one rule: a catalogue-level test per replacing action. design-structure §4.3, Q6 and §7 are amended in X3's PR | The review's findings 15 and 17 |
| 14 | Q1 is marked recommended yes, awaiting the owner, and X0's rule works either way. Q2 is answered no: `Model` and `ModelRates` stay at the root and `ProviderError` goes in `host` | Coordinator, 2026-11-15 |

## 0. What the cut-over is

Fifteen layer roots become five role roots. Every declared name has exactly one public spelling, assigned by rule
(§1). Only six submodules are renamed (§1.5); every other defining module stays where it is. The thirteen former
roots keep their modules, and their `__init__.py` exports nothing: a moved name raises `ImportError` naming its new
spelling. The demotions and the shipped uses land with it (§4, §5). A stdlib script and a generated, epoch-versioned
text map carry every consumer across, the host's 46 branches included, by idempotent re-runs (§3).

## 1. The five roots

### 1.1 The rule

The first rule that holds places the name. Kinds are V's (`surface_tiers.json`, `kinds`); "observed use" is read
from the criterion-2 sources only (host production, shipped examples, other packages' source), never from tests.

| Rule | Condition | Root |
|---|---|---|
| 1 | Package metadata: `__version__`, `PUBLIC_ROOTS`, `EXTRAS`, `Extra`, `surface_tier`, `declared_names` | `threetears.evals` |
| 2 | Kind `typed` (`Model` and `ModelRates` among them) | `threetears.evals` |
| 3 | Kind `tests` | `threetears.evals.testing` |
| 4 | Defined under `actions/` or `transports/`: the catalogue, its parts, its transport, its outcome types | `threetears.evals.actions` |
| 5 | Kind `bound`, `invoked` or `action`; or a `received` name that observed use shows a host raising, subclassing, implementing or passing to the engine | `threetears.evals.host` |
| 6 | Kind `received` otherwise: a value a host only reads or annotates with | `threetears.evals.shapes` |

**Reached by definition.** A stored document class (someone's store receives it) and an action's result type
(someone's caller receives it) count as reached, whatever imports them. Action parameter models do not count: the
catalogue serves their JSON schema.

**Provisional names.** X0's rule takes a switch for Q1 (§9). Under the recommended answer, a provisional name that a
declared signature reaches, transitively, or that is reached by definition, is placed by `kind_by_rule` where that
decides and as `received` otherwise. It keeps its provisional tier. Any other provisional name leaves the roots.
Under the other answer, every provisional name is placed the same way and none leaves. `Log` is typed, at the root,
and still provisional. `quick.measure` and `quick.Measure` are removed under O4, which rides X3.

### 1.2 Exceptions

| Name | Rule says | Placed | Why |
|---|---|---|---|
| `providers.Model` | 5 | collapses into `threetears.evals.Model` | The same object exported twice (`__init__.py:87`) |
| `ops.EvalSummary` | 6 | collapses into `shapes.EvalSummary` | The same object as `analysis.EvalSummary` (`ops/__init__.py:253`) |
| `ROOT_EXTRAS`, `RootExtra` | 1 | renamed `EXTRAS`, `Extra`, keyed by qualified name | No root fails to import without an extra once fastmcp names are lazy; a root-keyed map would be empty and false |
| `mount_fastmcp` | 2 over 4 | `threetears.evals` | Rung 4 types it: the newcomer's one-place rule outranks the catalogue's grouping |

### 1.3 Counts and spellings

Declared after the cut, by my reading. X0 computes the exact figures, and they are reported, never targets.
Arithmetic: 488, less 19 demoted (§4), less 4 shapes that only demoted names reach, less 2 duplicates, plus 7 kits
admitted under R-a, plus 16 stored classes: **486**.

| Root | Declared | By kind |
|---|---|---|
| `threetears.evals` | 17 (18 with O2's `ModelRates`) | typed 12, metadata 5 |
| `threetears.evals.host` | about 177 | bound 77, invoked 76, action 13, plus 11 received names a host raises or subclasses, plus those it passes (X0 finds them) |
| `threetears.evals.shapes` | about 252 | received |
| `threetears.evals.testing` | 27 | tests: 17, plus 7 kits, plus 3 kept doors (§4.2) |
| `threetears.evals.actions` | 13 | bound 9, received 4; reached provisional names (`ToolSpec`, `MountedTool`, ...) beside them |

Provisional names: my transitive walk with V's `edges` reaches 653 distinct objects, and 627 names are reached by
nothing (about 630 leave under Q1 = yes). The adopter goals do not change: a first eval needs 2 names, rungs 0 to 3
need 5, all at `threetears.evals` and spelled as today. On today's rung numbering (O6 renumbers after X3), five
spellings change: `as_host`, `mount_fastmcp`, `AnalystSkill`, `Writer` and `WriterEval` move to `threetears.evals`.

### 1.4 Root mechanics

- `host.py` and `shapes.py` are new modules. The `testing/` and `actions/` inits and the root `__init__` are
  rewritten. Every root is lazy: a generated table maps each name to its home module, served by `__getattr__` (the
  root's pattern today, `__init__.py:80-110`), with `__all__` and `__dir__` listing them. Importing a root loads
  nothing until a name is read.
- `shapes.EVAL_DOC_CLASSES: Mapping[str, type[EvalBaseModel]]` (declared, received) maps every `doc_type` to its
  class and loads the 29 homes when it is read. It replaces the host's `__subclasses__` walk, which depended on side
  effects (`discodon/audit/registry.py:340-372`). `EVAL_DOC_TYPES` stays: the cheap tuple of strings the host's
  production code tests membership against. A test holds `tuple(EVAL_DOC_CLASSES) == EVAL_DOC_TYPES`.
- The five fastmcp names are lazy like every other name. Each raises the existing message
  (`transports/fastmcp/__init__.py:94`) without the extra, and `EXTRAS` lists them.
- Each of the thirteen former roots keeps a docstring and one `__getattr__`. It raises
  `ImportError("<old> moved to <new>; run reroot.py (map epoch <SHA>)")` for a mapped name, and `AttributeError`
  otherwise, so submodule imports still work. It serves no object, so it is not a shim.
  `import threetears.evals.kernel` on its own still succeeds; the script reports such imports (§3.3).
- Source inside the package never imports a role root (a walker test holds it). The 142 statements that import a
  layer init are re-pointed to home modules.
- `PUBLIC_ROOTS` lists the four sub-roots. `surface_tier`'s docstring changes from "moves only with ... a deprecation
  cycle" to "a MIGRATIONS record, plus a deprecation cycle or a cut-over its host agreed".
- `tests/test_import_time.py`: `:51` re-pointed (`from threetears.evals.shapes import DEFAULT_JOB_LEASE_S, ...` loads
  nothing outside `schema`); `:45` re-pointed to `shapes`. `:77` (a ladder of at most 10 names) is replaced by the
  admission test: each typed name is taught by a rung. The PR records this as a replacement, not a weakening.

### 1.5 Six submodules renamed

A public name never shares its spelling with a module of the package that exports it, or exported it.
`tests/test_no_name_shadows_a_module.py` walks every role root, the thirteen former roots and the map's old column
against `pkgutil.iter_modules`. None of the six defines a stored class, holds package data, or names a logger a host
pins. Three of them have module loggers (`run/regrade.py:77`, `run/judge_kind.py:123`, `run/definition_drift.py:115`),
whose names move by a `module` line.

| Module (lines) | Renamed (proposed; the test decides acceptability) | Function it shadowed |
|---|---|---|
| `run/regrade.py` (426) | `run/regrading.py` | `run.regrade` (demoted) |
| `run/definition_drift.py` (1,306) | `run/drift.py` | `run.definition_drift` (host-contract) |
| `run/judge_kind.py` (1,570) | `run/judge_as_kind.py` | `run.judge_kind` (provisional) |
| `analysis/campaign_readings.py` (1,950) | `analysis/campaign_read.py` | `analysis.campaign_readings` (provisional) |
| `analysis/judge_drift.py` (333) | `analysis/judge_drift_read.py` | `analysis.judge_drift` (stable) |
| `analysis/judge_kind_readings.py` (381) | `analysis/judge_kind_read.py` | `analysis.judge_kind_readings` (stable) |

69 in-repo files name one of these module paths; the script re-points them from the `module` lines.

### 1.6 `shapes`: a role, and importable

`shapes` means "what the engine hands you to read". It earns that only once rule 5 takes out what a host does
something with: errors it raises or subclasses (the host raises `ValidationFailedError` 33 times and `ApparatusError`
17), protocols it implements, registries and seed templates it passes. Those are bound use, and IA §8.1's tie-break
("an exception is received") misplaced them.

It stays importable, for three reasons. A mypy-strict host annotates with these types: 127 of Discodon's 246
production names land here. Stored classes need a public mapping. And reaching a type only through what returns it
gives no spelling for a parameter annotation. A reader should still never meet it as a flat list of about 250
names. `surface.md` and `llms.txt` fold each shape under the name that reaches it (IA §8.4, row 3). `llms.txt`
replaces its KINDS block with WHERE NAMES LIVE (five lines, one per root) and adds "the same operation in Python is
`host.<action name>`". The NEEDS A SHIPPED EXAMPLE block goes, and the file stays within today's 136 lines. Laziness
makes the size free at import.

### 1.7 The rule against the north star

| Question | Answer |
|---|---|
| Does a first eval import from one place? | Yes: every typed name, every rung, is at `threetears.evals` |
| Does a host find what it binds in one place? | Yes for embedding the engine (`host`, including what it raises and subclasses). Serving the catalogue is a second job (`actions`). It annotates from `shapes` |
| Does an agent find the catalogue without reading the rest? | The catalogue listing needs no Python. Its Python door is `actions`, and `llms.txt` says where each action's Python function is |

## 2. Stored and wire identifiers

| Identifier | Where | Moves in the cut? | How it is kept |
|---|---|---|---|
| `doc_type` | 29 `doc_type: Literal[...]`; `EVAL_DOC_TYPES` (`schema/models.py:242`) | No | String literals |
| Fingerprints, digests, judge identity | `schema/hashing.py`, `bundle/evidence.py`, `run/definition_digests.py` and others | No | No hashed input holds a module path, qualname or class name (searched); `lenses/program_budget.py:54` names a derived model that is never stored |
| Schema pins | `test_core_schema_pins.py:192,215-222` | No | Descriptions, titles and examples are dropped from the digest |
| Recordings | `testing/recorded_completions.py` (S4: input-keyed) | No | Prose fixes are wording, and S4's keys survive wording; X3 re-asserts zero misses |
| Response formats | `providers/model.py:1310`, `run/simulator.py:207`, `gen/proposal_contract.py:36` | No | `RubricProposal`'s schema names `run.authoring`, `run.conversation` and `run.simulator` module paths, none renamed |
| `result_model` on action-job records | `run/job_work.py:549,611`: `type(result).__name__` | No | No class is renamed. Later moves pin it (§6) |
| Stored error text | `run/runner.py:1619-1623`: `type(fault).__name__` | No | Same |
| `contained_site` in log records | `schema/logs.py:73,102`: `module:qualname` | Only for the six renamed modules | Log-only and operator-visible; never stored in a document |
| Package-data paths | `files("threetears.evals.analysis.analyst")` (`skill.py:213`, `skill_ledger.py:101`), `...analysis.report` (`html_parts.py:34,39`, `kit_colours.py:21`) | No | Fail loudly if moved; `module` lines cover a later move |
| Action and MCP tool names | `actions/engine.py:2043` on; host mounts `evals`, `evals_admin`, `evals_analyst` (`discodon/evals/mcp.py:1,607`) | No | Explicit strings |
| Command paths | `python -m threetears.evals` (`quick/cli.py:204`); `python -m threetears.evals.harness` (`harness/__main__.py:74`; the host's `EVAL_SYSTEM.md:168`, `host.py:705`) | No | Operator identifiers; no move may take `__main__.py` or `harness/__main__.py` |
| Logger names | `get_logger(__name__)` in 79 modules; fixed `threetears.evals` (`schema/logs.py:47`), `threetears.evals.analyst` (`ops/analyst.py:63`) | Three, the renamed modules' | The stable contract is the two fixed names. Module loggers follow modules through `module` lines. The host pins only `threetears.evals.kernel.host.profile` (`test_apparatus_seats.py:28`), which is not moved |
| `owner_module` | Host: `cls.__module__` (`registry.py:423`); literal `"threetears.evals.storage.routed"` (`:461`) | No | No stored class's module moves; not persisted (read only in host tests) |
| Seed references | Engine `seed_module` names internal modules (`kernel/prompts/catalogue.py:46,52`), never stored; host's `seed_module="threetears.evals.gen"` + constant (`discodon/evals/prompts.py:163,169`) | Host string only | Resolved by the pairing rule to `threetears.evals.shapes`; `test_prompts.py:89` holds |
| Tier map, `surface.json`, `PUBLIC_ROOTS`, `surface_tier` | `__init__.py:122-209` | Yes, wholesale | They are the contract, read live by the host's gate; an old name raises `KeyError` |
| Printed root spelling | `analysis/summary.py:937` | Yes | Becomes `threetears.evals.host.list_results`; goldens move by that substitution only |

## 3. The rewrite script

### 3.1 Where it lives, how a host runs it

`packages/evals/scripts/reroot.py` is dev tooling, never installed: one file, standard library only, and it never
imports `threetears`. It requires Python 3.14: the package and the host use PEP 758 syntax, and the script parses
with the running interpreter. Before rewriting, it refuses any file that fails to parse, and reports it untouched.
The map is `packages/evals/api/reroot-map.txt`.

```
git -C <3tears clone> show <SHA>:packages/evals/scripts/reroot.py > reroot.py
git -C <3tears clone> show <SHA>:packages/evals/api/reroot-map.txt > reroot-map.txt
python3.14 reroot.py --map reroot-map.txt --to <pinned SHA> --dry-run .
python3.14 reroot.py --map reroot-map.txt --to <pinned SHA> .
```

### 3.2 The map

Generated by `scripts/reroot_map.py` from two API snapshots. The header lists epochs in order, each a SHA and a
title: epoch 1 is the cut-over; S6, S9 and #1013 each append one. Lines are tab-separated: epoch, record, old, new,
note. `--to <SHA>` applies epochs up to and including that SHA, and defaults to all. The header carries a SHA-256
of the body, and the script refuses a mismatch.

| Record | Public target (host, docs, examples, in-repo tests) | Home target (package `src` imports) |
|---|---|---|
| `name`: old spelling → role-root spelling | rewrite | imports: home module; prose: public spelling |
| `renamed`: `ROOT_EXTRAS` → `EXTRAS`, `RootExtra` → `Extra` | rewrite the token and report (shape changed) | same |
| `demoted`: old spelling → home, note = replacement | report `file:line name -> replacement`, exit 2; in-repo tests with `--home-for-left`: rewrite to home | imports: home; prose in an action or field description: reported (wire prose names the action) |
| `left`: unreached provisional → home | as `demoted` | as `demoted` |
| `root`: an old root → the roots its names went to | bare-root strings: resolve when paired, else report | same |
| `module`: an internal module path → new path | strings, prose and imports by dotted-prefix match | same |

**Invariant, checked by the generator and again by the script on load.** No new value of any line equals an old
value of a line at or before its epoch, or has one as a dotted prefix. The rewrite of each epoch then never touches
its own output or an earlier epoch's output, and the composition is idempotent. The six renames (§1.5) are what make
this hold for `run.regrade` and its kin.

### 3.3 What it rewrites

| Form | Treatment |
|---|---|
| `from OLD import a, b as c`, at any depth (`if TYPE_CHECKING:`, functions, `try`) | Split by target root, names and aliases in their original order; a trailing comment (`# noqa: E402`) stays on the first emitted line; `demoted`/`left` names stay on a residual old-root statement and are reported |
| `from OLD import submodule [as x]` (`from threetears.evals.analysis import service as analysis_service`) | Untouched: no record names it. A renamed submodule follows its `module` line |
| `from threetears.evals import ops` (module binding), then `ops.X` | Each read becomes `evals_<role>.X`, bound by `from threetears.evals import <role> as evals_<role>`. An identical binding already present (from an earlier run) is reused; any other binding of that alias is reported. Host production: 2 files, 12 reads |
| `import threetears.evals.kernel [as k]`, dotted chains | Chains are rewritten by their longest `name` prefix. A bare import with no attribute read is reported |
| `import *` from an old root; `getattr(old_binding, "...")`; `__import__` | Reported |
| String equal to a qualified name or `module:attr` | Exact match on `name`/`renamed`; `module` by prefix. This covers `mock.patch`, string annotations, `owner_module=`, `getLogger(...)` and `import_module(...)` with a name |
| String equal to a bare old root | Resolved when the call or tuple pairs it with a string naming an attribute of that root (`seed_module=` + `constant=`); else reported (host: `registry.py:340`, which the doc-type mapping replaces) |
| String holding code (`"from threetears.evals.kernel import X\n"`) | Reported, never rewritten |
| Other strings: attribute docstrings, `Field(description=...)`, assertion messages | Prose token mode (host: `discodon/config/sections/eval.py:38,119,246,358,371,415`) |
| Docstrings, comments, Markdown, `.toml`/`.yaml`/`.json`/`.cfg`/`.ts` | Token mode, by longest `name` or `module` prefix; `root` records only as whole tokens; `--no-prose` skips this class. Markdown `python` fences are parsed as code |
| Generated files (`GENERATED` or "do not edit" in the first 10 lines) | Reported only; the report names how each is regenerated |
| Excluded by default | Dot directories, `site-packages`, `CHANGELOG*`, `MIGRATIONS.toml`, the map, `tests/fixtures/reroot/**`, `--exclude` paths; a line carrying `# reroot: keep` |

The script does not format. It emits canonical import lines and says to run the host's formatter afterwards.

### 3.4 Exit codes, dry run, convergence

Exit 0: nothing left. Exit 2: rewrites applied and hand edits listed. Exit 1: refused (a bad map, a pre-parse
failure, or a post-rewrite parse failure, which restores the file). `demoted`, `left`, generated and code-string
sites are reported on every run and never changed, so a re-run's exit 2 repeats the same list and redoes nothing.
A half-rewritten tree, or a branch merged after the cut, converges in one pass, because every occurrence is
rewritten independently and the binding rule reuses earlier output. `--dry-run` prints counts by form and by report
type; `--verbose` adds every site; `--json` gives the same for a gate.

### 3.5 Its tests

| Test | Holds |
|---|---|
| `before/` → `after/` | One fixture file per form of §3.3, plus Markdown, `.toml`, a generated file and a pragma line, rewrites byte-equal under a two-epoch fixture map holding every record type |
| Idempotent | `after/` again: no change, exit 0 |
| Converges | `half/`: every second occurrence pre-rewritten, one file already holding `evals_host`, one with a merge-shaped duplicate import → `after/` |
| Epochs | `--to <epoch-1 SHA>` applies only epoch 1; a second run with `--to <epoch-2 SHA>` equals one run with it |
| Demoted | exit 2 with `file:line name -> replacement`; the rest of the file rewritten |
| Refusals | Map lines that overlap by equality or dotted prefix, a wrong digest, a file that fails to parse: exit 1, nothing written |
| Strings | Paired root resolved, unpaired reported, code strings reported, module paths untouched |
| Dry run | Writes nothing; counts equal a pinned table |
| Stdlib only | An AST walk finds no non-stdlib import |

## 4. The demotions

### 4.1 Decided demotions (19)

A demoted name stays a non-underscore name in its home module, unpublic by its module. A host's replacement is an
action, called from Python through the catalogue:
`await eval_catalogue().mount(standard_tools()[0]).call({"action": "<name>", ...}, host=ops_host, caller=caller)`
(`actions/catalogue.py:779,1029`; `actions/engine.py:3372`). Otherwise it is a kept name. The "host tests" column
marks the names whose replacement X4 part 1 proves on the host's own call before X3.

| Name | Home | Replacement | Repo test files | Host tests |
|---|---|---|---|---|
| `analysis.report_html` | `analysis/report/serialize_html.py` | `host.report_read(..., format="html")` / the `report_read` action | 32 | |
| `ops.await_job` | `ops/jobs.py` | `host.job_manager.wait_for(ids)`, which G1 already uses (`test_keyless_matrix_live.py:440`), or `host.job_poll` | 13 | yes |
| `quick.run_cli` | `quick/cli.py` | the command `python -m threetears.evals` | 12 | |
| `run.quote_launch` | `run/launch.py` | the `launch_estimate` action | 8 | yes |
| `ops.judge_case_author` | `ops/judge_cases.py` | the `judge_case_author` action | 5 | yes |
| `run.mint_case_set` | `run/case_sets.py` | the `case_set_mint` action | 3 | yes |
| `run.regrade` | `run/regrading.py` | the `result_regrade` action | 2 | |
| `negotiation.closest_settings` | `negotiation/settings.py` | `host.negotiate` | 2 | |
| `ops.case_rename` | `ops/cases.py` | the `cases_rename` action | 2 | yes |
| `analysis.gate_verdicts` | `analysis/gate.py` | `Result.gate()` (`quick/result.py:896`) | 2 | |
| `ops.campaign_export` | `ops/campaign_closure.py` | the `campaign_export` action, or `python -m threetears.evals.harness export` | 2 | yes |
| `run.admit_template` | `run/authoring.py` | `host.create_template` / `host.update_template` | 1 | |
| `kernel.CappedCompletion` | `kernel/capped_completion.py` | none needed: the engine applies the cap | 1 | |
| `run.load_seed_corpus` | `run/definition_seed.py` | `host.seed_eval_definitions`; the `definitions_drift`/`definitions_reset` actions | 1 | |
| `actions.render_jobs_started` | `actions/render.py` | none needed: the catalogue renders results | 1 | |
| `ops.case_set_mint` | `ops/runs.py` | the `case_set_mint` action | 1 | yes |
| `ops.case_sets_list` | `ops/runs.py` | the `case_sets_list` action | 1 | yes |
| `kernel.host.load_bar_registry` | `kernel/host/bars.py` | build `host.BarRegistry` | 1 | |
| `transports.fastmcp.CatalogueTool` | `transports/fastmcp/__init__.py` | FastMCP's own tool listing on the mounted server | 0 | yes |

With them go the shapes that only demoted names reach and that are not reached by definition: `ops.CaseSetMint`
and `ops.JudgeCaseAuthor` (action parameters), `quick.HostCommand` and `quick.HostFactory` (`run_cli`'s). If the
action results `ops.CampaignExport`, `ops.CaseRenamed` and `ops.CaseSetLine` are what X0 confirms them to be, they
stay on `shapes`, reached by definition.

### 4.2 Kept as `tests`, by the criteria (3)

What a host's own gate and conformance tests need, with no action that does it:

| Name | Root | Criterion | The capability no door has |
|---|---|---|---|
| `run.start_run` | `testing` | 1 and 2 under R-a (a door a host's own gate and conformance kit use to build their inputs, used in 14 host test files); 3: the only door | Launching on a `LaunchHost` with `launch_group`, `cassette_mode`, `judge_config_ids`, `max_metered_calls` and `margins` (`run/launch.py:2007-2040`; `LaunchArguments`, `ops/runs.py:190-248`, has none of them), returning the runs |
| `run.launch_as_group` | `testing` | as above | One group holding every kind, the shape of G1 (`test_keyless_matrix_live.py:417-440`) |
| `analysis.assemble_context_bundle` | `testing` | as above | The assembled `AnalysisContextBundle` a host's kind tests read (`test_persona_kind.py:883,945`, `test_run_machine.py:21`); `inspect_campaign_bundle` returns a `BundleInspection` |

X4 part 1 may overturn any of the three by proof: when a port to an action works on the host's real call, the name
is demoted instead.

### 4.3 In-repo tests and pages

The cut re-points tests mechanically. A demoted import becomes a home-module import (`--home-for-left`), and test
bodies do not change. 177 test files import a demoted or kept name from an old root, 91 of them
`assemble_context_bundle`, now `testing`. X5 holds one rule instead of a migration: each action that replaces a
demoted function has a catalogue-level test of its result and of one refusal with its remedy.

Pages: `adopting-a-host.md` (13 mentions), `serving.md` (7), `concepts.md` (5), `cost-and-budgets.md` (4),
`reading-reports.md` (2), and one each in `judges-and-calibration.md`, `building-an-inventory.md` and
`open-problems.md`. A code block that calls a demoted function calls its replacement; prose names the action;
`start_run` is taught under testing. `test_no_page_names_a_removed_name` holds this.

### 4.4 Provisional kits admitted to `tests` (7)

`testing.READER_CONFORMANCE_CASES`, `testing.ReaderConformanceCase`, `testing.ReaderSample`,
`testing.RESULT_READINGS_CONFORMANCE_CASES`, `testing.ResultReadingsConformanceCase`, `run.resolve_constructor`,
`run.callers_missing_the_constructor`. Ruling R-a applies: the host's tests run them (`test_reader_conformance.py:31`,
`test_keyless_matrix_live.py:43-52`).

## 5. The four awaiting a shipped use (X2)

All four go into the courier host (`tests/fixtures/courierhost/__init__.py`). Each default reproduces today's cells:
the solver default is today's rule, `outage` is off unless a case's variation sets it, and the action is additive.

| Name | Smallest real use |
|---|---|
| `failed_call_output` | `RoutePlannerKind.invoke` asks its scripted planner model through a `CompletionClient`; a case whose variation sets `outage` makes it decline, and `invoke` routes the failure through `failed_call_output` with `withhold_failure_detail` |
| `Ordinal` | `COURIER_CONTRACT` (`:235`) gains an overlay: `solver: Annotated[Literal["auto", "greedy", "exhaustive"], Ordinal()] = "auto"`. `auto` keeps today's `model.endswith("-pro")` rule (`:300`); the other levels override it |
| `MemberActsOn` | Same overlay: `zone_slack_min: Annotated[dict[str, int], MemberActsOn({"downtown": ON_TIME_RATE})] = {}`, which the planner adds per zone; empty by default |
| `start_action_job` | One courier action, `courier_cases_propose`: the scripted model proposes stop cases, run through `start_action_job`, served by `eval_catalogue(host_actions=...)`, and driven in a test through the catalogue and `job_poll` |

An overlay adds courier levers to every lens. Done-when: the courier goldens (`test_hot_path_oracle`,
`test_identity_golden_keys`, `test_evidence_fingerprint` and the rest that build the courier) are unchanged. Or,
where an overlay lever's presence moves one, each moved golden is listed with that reason and regenerated
deliberately (the 11-06 goldens ruling). `AWAITING_SHIPPED_USE` is then deleted.

## 6. Order with the structure track, and the stored-name pins

Cut-over first, then S6, S8 and S9. None of them touches a public root after the cut, and the host's module-path
strings (`kernel.host.profile`, `storage.routed`) are not in their scope. S6 and S9 drop their one-minor shims
(design-structure §4.3): each appends an epoch of `module` lines, and tests are re-pointed by the script. They must
not move `__main__.py`, `harness/__main__.py`, `kernel/host/profile.py`, `storage/routed.py`, any package-data
directory, or a module defining a stored class, without a `module` line and a host note.

Designed now, built in X0 and run by every later move:

| Pin | Holds |
|---|---|
| `test_stored_class_names.py` | The `__name__` of every class that can land in a stored field (every action result type, read into `result_model`; every exception type the runner writes into error text) is pinned by value. A rename fails; S8's private renames of such classes need a migration note |
| `contained_site` | Every move asserts its `module` lines cover each moved module, so an operator's log filter on `module:qualname` can be updated from the map |
| Data paths | Every `files("threetears.evals...")` literal resolves in a fresh interpreter |
| Recordings and digests | Rung 5 and `judge.jsonl` replay with zero misses; every identity golden, fingerprint and schema pin is unchanged |

The in-repo churn is not paid twice. The cut re-points about 1,700 root-import statements in 523 test files. S6 and
S9 re-point different statements (home imports of the modules they move) by re-running the script.

## 7. Build chunks

| Chunk | Scope | Files owned (disjoint) | Size | Needs | Can start |
|---|---|---|---|---|---|
| X0 | The rule (rules 1 to 6, observed use, reached by definition, the Q1 switch); `defined_in` computed in its own generator (snapshot format 4 waits for X3); `reroot_map.py` with the epoch and prefix invariant; the stored-name pins of §6; a test placing every declared name, late ones included | new `scripts/role_roots.py`, new `scripts/reroot_map.py`, new `tests/test_role_roots_rule.py`, new `tests/test_stored_class_names.py`, new `tests/test_no_name_shadows_a_module.py` (marked xfail-strict until X3 renames the six) | about 450, 350 test | this revision | **Now** |
| X1 | The script and its tests | new `scripts/reroot.py`, new `tests/test_reroot_script.py`, new `tests/fixtures/reroot/**` | about 900, 600 test | this revision | **Now** |
| X2 | Four courier uses; `AWAITING_SHIPPED_USE` deleted | `tests/fixtures/courierhost/**`, new `tests/test_courier_shipped_uses.py`, the register in `scripts/surface_kinds.py`, `tests/test_declared_holdbacks.py` | about 220, 150 test | V on `evals/main` | **Now** |
| X4, part 1 | Prove §4.1's host-test replacements and §4.2's three kept doors on a throwaway clone of the host, outside `/opt/venv`, never pushed | none in this repo; findings to `.prawduct/artifacts/reroot-host-proof.md` | small | X0's planned map | After X0 |
| X3 | The cut-over: one PR, three commits. (a) Roots, lazy tables, `EVAL_DOC_CLASSES`, emptied inits with their `__getattr__`, six renames, `EXTRAS`, `PUBLIC_ROOTS`, demotions, kits, 16 stored classes declared, O4, registers deleted, pages, the tier promise, `test_import_time` re-pointed, design-structure §4.3/Q6/§7 amended. (b) Script output over `src`, tests, examples, docs, README. (c) Regenerated files, snapshot format 4 | every evals file | hand about 2,000; mechanical about 1,900 statements | X0, X1, X2, X4 part 1, S5a-2's batch | At the next quiet point |
| X4, part 2 | Handover | a note on #1199 | small | X3 | |
| X5 | A catalogue-level test of result and refusal per replacing action, where missing | new `tests/test_replacing_actions.py` | about 300 test | X3 | |

**The judges and remedy chunks** add public names now (judges' `Check`, the remedy actions, O2's `ModelRates`). X0,
X1 and X2 need nothing from them. X0's placement test judges any late name by rule, so a name that no rule places
fails at once rather than at X3. **The map is frozen at X3's base:** a name already on `evals/main` there gets a line;
a name that merges after gets none, because it never shipped at an old root. That name is placed by `role_root()`,
and its branch re-runs the script after merging `evals/main`, as the host's 46 branches do. If X3's base moves, the
map is regenerated (done-when 3).

### Done-when (a test holds each)

| Chunk | Done when |
|---|---|
| X0 | Every declared name has exactly one root under rules 1 to 6 with §1.2's exceptions, under both Q1 settings; no two names collide; the planned map satisfies §3.2's invariant; every stored class and action result type is reached by definition; the stored-name pins pass on today's tree |
| X1 | §3.5 whole |
| X2 | `AWAITING_SHIPPED_USE` is gone, and the admission test passes with each of the four used; courier goldens unchanged, or listed with reasons |
| X4, part 1 | For each "host tests" name of §4.1 and each of §4.2's three, the host's real call is either ported and passing (the replacement's call written out), or recorded as not portable, naming the missing capability. The demotion list X3 builds from is this file's |
| X3 | (1) Each declared name is served by exactly the root its rule assigns, and by no other. (2) In a fresh interpreter, one moved name from each former root raises `ImportError` naming the new spelling, and `from threetears.evals.run import launch` still imports the submodule. (3) `reroot-map.txt` equals the map generated at X3's base, and every before-name is unchanged or has one line. (4) Both registers are gone; admission passes. (5) Every frozen stored fixture reads and re-dumps byte-identically; no identity golden, fingerprint or schema pin is regenerated. (6) Zero recording misses. (7) `EVAL_DOC_CLASSES` maps all 29 doc types and equals `EVAL_DOC_TYPES` in keys; `from threetears.evals.shapes import DEFAULT_JOB_LEASE_S` loads nothing outside `schema`. (8) The recorded host production list, rewritten through the map, imports cleanly; each host-test name is rewritten-and-importable or has a line with a proved replacement. (9) With fastmcp missing, all five roots import and the five lazy names raise naming the extra. (10) No `src` module imports a role root. (11) Examples import only from `threetears.evals`; goals 2 and 5 hold. (12) The script over the repository makes zero changes and reports only pragma-kept lines, excluded files and the listed generated files. (13) Every moved golden equals its old content with the map applied. (14) `test_no_name_shadows_a_module` passes |
| X5 | Each replacing action has its result test and one refusal-with-remedy test |

Expected to move in X3: `surface.json` (format 4), `surface_tiers.json`, `surface.md`, `llms.txt`, `reference.md`,
a generated `MIGRATIONS.toml` section, and goldens printing an old-root spelling, by substitution only. No version
bump; the release carrying it is a minor, bundled by the maintainer.

### Handover (X4, part 2)

The cut SHA and the map's epoch; `reroot-map.txt`; `reroot.py`; Python 3.14; the dry-run on the host's develop with
every exit-2 site; X4 part 1's proved replacement calls. Then the host regenerates its generated files instead of
editing them: `.prawduct/artifacts/openapi.json` with its own builder, then `web/scripts/gen-ops-types.mjs`
(`ops-types.ts:4`), held by `test_openapi_drift.py`. Pre-measured on a78b15b66; the review counted production at
59 files / 146 statements / 392 names, and the dry-run settles it.

| | Production | Tests |
|---|---|---|
| Files / statements / names | 69 / 141 / 386 | 113 / 369 / 601 |
| Distinct names used | 246: 235 move, 9 keep, 2 module bindings | 234: most move; hand edits below |
| Module bindings | 2 files, 12 reads | 10 files |
| Strings | `seed_module` pairs resolved; `registry.py:340-372` rewritten by hand to `EVAL_DOC_CLASSES` | about 30 code samples by hand; the logger pin unchanged |
| Hand edits | `registry.py`'s walk; `ops-types.ts` regenerated | demoted names with proved replacements; about 26 unreached provisional imports under Q1 = yes; `ROOT_EXTRAS` → `EXTRAS` (`test_public_api_only.py:97`); 3 names V removed |

## 8. Decided by principle

| Decision | Principle |
|---|---|
| Former roots serve nothing and raise naming the new spelling | Loud failure; one public spelling |
| Six shadowing modules renamed, and a test forbids recurrence | One spelling, one thing; idempotency |
| Stored classes and action results reached by definition | Criterion 1: someone outside receives them, even when no signature says so |
| Placement by observed use from criterion-2 sources | Kind is what a host does with a name; tests never decide it |
| Every root lazy, with an explicit doc-type mapping | Contracts by declaration, not by import side effects; a lease constant costs nothing |
| `shapes` importable, never shown flat | Strict hosts annotate; a reader meets a shape under its reacher |
| `start_run`, `launch_as_group`, `assemble_context_bundle` kept as `tests` | Criteria 1 to 3 under R-a, unless X4's proof shows a door; never kept by count |
| Prose in `src` takes the public spelling | Wire prose must not teach internal paths |
| Typed names, `mount_fastmcp` included, at the root | The newcomer's single place |
| A mechanical cut; test-door rules after | A reviewable diff |

## 9. Owner questions

**Q1. Do provisional names that nothing declared reaches, that no production code uses, and that are not reached by
definition leave the public roots?** About 630 names. **Recommended yes, awaiting the owner.** The cost to Discodon
is about 26 test imports; its production code imports none of them. X0's rule works either way, and X3 takes the
answer as a switch.

**Q2. Does `providers` ride the cut-over?** **No** (answered). `Model` and `ModelRates` stay at the root, served
lazily from wherever #1013 later puts them. `ProviderError` goes in `host`, since #1013 keeps it in evals.
`Catalogue` and `PriceRefresh` move again under #1013, as a later epoch of the same map.

## 10. Not verified

- Kinds are V's; counts in §1.3 inherit any misplacement and are my reading until X0 computes them.
- "Passed" observed use (registries, seed templates, constants a host hands the engine) was not scanned; raised and
  subclassed names were (11, by regex over host production, the courier and the examples).
- Whether `ops.CampaignExport`, `CaseRenamed` and `CaseSetLine` are action results (X0 confirms).
- That `job_manager.wait_for` is a public door for a host that only holds an `OpsHost`.
- The six proposed module names were checked only for collisions with existing files.
- No `$defs` key carries a module-qualified name; the renamed modules define no model used in a response format
  (assumed from a grep for stored classes, not for response formats).
- The host's 46 branches were not read.
- Which catalogue-listing goldens embed parameter-schema prose.
