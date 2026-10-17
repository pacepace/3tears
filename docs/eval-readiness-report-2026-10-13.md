# 3tears-evals: readiness review before the develop merge

**Date:** 2026-10-13. **Tree reviewed:** `feature/eval-adopt` at `v0.66.0-evaladopt.21` (merge `ea51d1ae2`).
**Consumer reviewed:** Discodon at its `.21` adoption (PR #2873, `63043673a`).
**Scope:** architecture, separation of concerns, adoption, usability, forward compatibility. Not a bug hunt.

**Status since this review.** The report is kept as written against `.21`. The three blockers landed on the line
as `v0.66.0-evaladopt.22` (PR #770). S5 (surface tiers), S7 (provisional marks) and S8 (the judge-wording policy)
landed as `v0.66.0-evaladopt.23` (PR #775). S1 is designed in `eval-providers-design-2026-10-13.md` and is being
built for `.24`. Read the verdict below as the state of `.21`, not of the line today.

## BLUF

**Merge after three small fixes on the line. Do not merge as-is.** The engine is far better than any consumer
would write: paired per-case tests with family correction, verdicts that refuse to overclaim, guardrails decided
apart from capability, content-addressed identity, a declared world with a conformance kit, and a compatibility
apparatus (frozen schema fixtures, git-backed API snapshot, migrations ledger) stronger than most 1.0 packages
ship. The boundary with Discodon is clean: zero internal imports, zero monkeypatches, pydantic as the only
dependency, a pluggable 8-method store proven by a conformance kit.

Three choices are wrong in a way that cannot be repaired for data written after the merge, and each is a
day's work now:

1. **A campaign's control is stored as a bare hash with no pre-image.** A lever rename or an identity bump orphans
   every stored campaign's control, and no upgrader can recover it. The run already stores its lever map beside its
   key; the campaign must too.
2. **Cassettes share a version number with analyses.** Any change to the analysis shape refuses every recorded tool
   corpus, which exists to protect third-party quota. Give cassettes their own version space.
3. **The 0.66 reader refuses every document one core version newer, and an added optional field is a core bump.**
   Whatever tolerance 0.66 ships with is the tolerance every future rolling upgrade has on its old side. Split the
   core version into `major.minor`, let a reader accept any newer minor of its own major (round-tripping the
   fields it does not know), and gate a minor on a mechanically-checked additive schema diff. Strictness stays
   everywhere drift can be detected; it is relaxed only where a tested contract makes the unknown field safe.

The adoption risk the owner named — consumers frightened off into hand-rolling something worse — is real, but it
does not sit at rung zero. `run_eval` → `compare` is a genuinely excellent first hour. The fright happens at hour
two, at four specific places: wanting a real LLM judge (the consumer must write a ten-field provider adapter
first), wanting a CI gate (`accuracy >= 0.9` has no one-liner), reading anything back from the store (engine
threading plumbing appears in user code), and the cliff from `compare` to a 300–500-line host with nothing in
between. None is architectural; all four are additive, and they should land before the second consumer onboards.

Everything else in this report is a strong recommendation or backlog. The package should merge to develop as
0.66.0 with the three blockers fixed and the four hour-two cliffs scheduled for 0.67.

## How this was reviewed

Four independent read-throughs, each with a different lens, then verified against source before anything here
was ranked: (1) the host boundary and both directions of leakage, reading the package's host contract against
Discodon's `discodon/evals/` seam; (2) day-one usability, reading README → examples → tutorial → concepts →
reference as a newcomer; (3) forward compatibility, reading the versioning, identity and API-surface machinery
and walking three plausible future changes through it; (4) Discodon's eval content and adoption history
(`.1`→`.21` in six days, 81 commits), to learn what the next consumers inherit. File references are to
`packages/evals/` on the reviewed tree unless prefixed.

## The picture in numbers

| | |
|---|---|
| Package source | 149,752 lines; tests 122,086 lines; docs 9,958 lines |
| Public surface | **1,482 names across 14 roots** (`api/surface.json`): kernel 330, analysis 321, run 210, schema 184, ops 182, kernel.host 83, quick 48 |
| Discodon's integration | 18,119 lines (host seam ~5.4k; domain content ~6.9k; generic machinery it had to write ~2.7k; glue the package's shape forces ~3.0k) |
| Discodon imports | 212 names from 12 public roots; **0** internal paths, **0** private reaches, **0** monkeypatches |
| Dependencies | `pydantic` only; `[vega]` and `[fastmcp]` extras; no other 3tears package |
| Rung zero | 21 lines (`README.md:66-89`); six concepts |
| "Minimal" own host | `tests/fixtures/courierhost/__init__.py`, 537 lines; a one-endpoint consumer needs ~250–350 |
| Version spaces | core 14, regenerable 9, bundle 50, cell model 12, report 9, identity 24, intent 1, goal-check proof rules 2 |

## Verdict by lens

**Separation of concerns: sound.** The package names no host noun (`tests/host_vocabulary.py` forbids 13 of
them), installs no ambient host (`tests/test_two_hosts_one_process.py`), and reaches into the host only through
declared ports proven by four conformance kits (store, completion, jobs, world). Discodon's YugabyteDB adapter is
220 lines using nothing but the port. Two qualifications: the surface is wide with no stability tier (every one of
1,482 names "may move, recorded, never shimmed"), and the dependency arrow is inverted inside Discodon's product
(nine production modules import the eval package: `tools/base.py` for `ExternalSpend`, `quality/llm_scorer.py`
for `run_judge_llm`, `audit/registry.py` for `EvalBaseModel`). The package's rule that evals depends on nothing
holds; its converse does not.

**Adoption and usability: excellent first hour, cliffs at hour two.** The tutorial (550 lines, one running
example, every term defined on first use, every output shown, "read your misses and decide whose fault it is"
taught before any number) and the nine-rung examples ladder are the best part of the package. Rung zero is a
real hello world. The value no consumer could write starts at `compare`: paired tests, Holm, "not separated"
never "no difference", TOST equivalence only against a declared margin, guardrails as non-inferiority apart from
the contrasts. The cliffs are listed under Strong recommendations S1–S4.

**Forward compatibility: strong apparatus, wrong granularity in three places.** The baseline reset is clean and
verifiable: `CORE_BASELINE_VERSION = CORE_SCHEMA_VERSION = 14` with an empty upgrader chain, frozen fixture
digests so history cannot be edited, pre-release documents refused by name. Keys are stored beside their
pre-images on runs, addressing fields are pinned so adapter indexes survive upgrades, host payloads are
structurally untouchable, the report ships a published JSON Schema with a versioned `$id`. The three blockers
are the places this apparatus will bite once stored data matters.

## Blockers: irreversible for data written after the merge

### B1. The campaign's control is a bare variant-key hash with no pre-image

`kernel/declaration.py:451-459`: `CampaignDesign.control: str | None`, "a 64-hex digest over the resolved
contestant stack". `control_from_run_id` resolves the key and drops the run id (`analysis/campaigns.py:246-259`).
The campaign is a kept core document. Variant keys are never re-derived at read (`kernel/identity.py:921-970`): the
run survives a rename or an `IDENTITY_VERSION` bump because it stores `variant_levers` beside `variant_key`
(`schema/models.py:3343`). The campaign has no such field, so after a host renames a lever, a kind or an overlay
field, or the engine bumps identity, every stored campaign's control matches no new run of the same arm, and an
upgrader cannot reconstruct it ("becomes None, read as 'not recorded'", `schema/versioning.py:35`).

**Fix:** store the control's lever map (or the resolving run id, which `control_from_run_id` already has) beside
the key, and re-resolve from it at read when the identity version differs. Since this is pre-release, fold it into
the v14 baseline; no upgrader needed. An operator can re-declare a control by hand from a surviving run, which is
why this is close to STRONG rather than a hard stop, but a control declared between the merge and a later fix is
lost.

### B2. Cassettes share `REGENERABLE_SCHEMA_VERSION` with analyses

`schema/models.py:5203`: `EvalCassette.schema_version = REGENERABLE_SCHEMA_VERSION`, the same strict constant as
`EvalAnalysis`, `EvalInsight` and the judge profile. A new chart type on a finding (`kernel/campaign.py:429,727`)
bumps regenerable, which refuses every cassette corpus; re-capturing one spends the third-party quota the cassette
exists to protect (`models.py:5181-5185`). Cassettes are `record` origin (`schema/doc_types.py:103`) — evidence of
what a tool answered — and nothing about their shape is tied to the analysis shape. The docs admit 24 bundle bumps
in two months (`docs/measuring-soundly.md:203-204`); analyses are cheap to regenerate by design, corpora are not.

**Fix:** a cassette version space of its own (or make cassettes core with upgraders, which their origin suggests).
One constant and one doc-type fact. A later release can baseline old cassettes, but every corpus captured in
between is paid for twice.

### B3. Additive schema changes must not be stop-the-world, and only the 0.66 reader can decide that

**The problem.** `schema/versioning.py:175-179` refuses any document above `CORE_SCHEMA_VERSION` ("written by a
newer build"). `versioning.py:28`: every change to a core shape is a bump, "an added optional field included".
Rewrite is lazy: the first new-build process that touches a run writes it at v15, and every old-build process
sharing the store refuses it from then on. So every core bump, including a purely additive one, is a stop-all for
a multi-process host. Discodon absorbed four in six days (v11–v14; its `DEPLOYING.md` says "migration is not
attempted anywhere"); a consumer with a stricter data owner will not. And the family rolls versions often, so the
common change is additive. The old side of every future rolling upgrade is whatever reader 0.66 ships with:
tolerance added in 0.70 does nothing for the 0.66 process, which is why this is a blocker and not a backlog item.

Strictness is the right principle ("tolerant readers hide drift until it corrupts a comparison",
`docs/principles.md`). The defect is that it is applied uniformly: a purely additive change pays the same price as
a rename.

**The scheme: scope tolerance to exactly the case a contract makes safe.**

1. **Split the core version into `major.minor`.** Minor = additive only: new optional fields with defaults, new
   optional sub-objects, new doc types; no upgrader. Major = everything else: rename, remove, type change, new
   required field, any semantic change; upgrader required, as today.
2. **The reader rule.** Same major, any minor → read. Fields this build does not know, on a document whose minor is
   newer than the build's → preserved opaquely and **round-tripped on write** (lazy rewrite never drops them; the
   write stamps `max(read_version, build_version)`). Unknown fields on a document at the build's minor or older →
   refused, exactly as now. Different major → refused or upgraded, as now. Drift detection is kept everywhere it
   can detect drift; it is relaxed only where the contract guarantees the unknown field is safe to ignore.
3. **Enforce additivity on the writer side, mechanically.** The per-type JSON-schema digests already pinned in
   `tests/test_core_schema_pins.py` become the gate: a minor bump's schema diff from the previous version must
   consist solely of added non-required properties; anything else fails and demands a major. This is what makes
   trusting a newer minor legitimate: the newer build proved additivity before it could ship.
4. **The one thing the digest cannot check.** A field can be schema-additive and semantically breaking: a new
   `faults` field meaning "exclude this cell" is read wrong by an old build that ignores it. The ledger entry for a
   minor must assert, in words, "an older build that ignores this field reads the document correctly", and review
   holds it there. When that sentence is false, it is a major.
5. **Stop using closed `Literal`s for vocabularies that will grow.** A new enum member is breaking for any reader
   that branches on it, so under this scheme it is a major. `UsageRole` (`schema/models.py:3994`), the
   provenance/origin literals (`:2418-2431`) and their like should become open strings with a registered
   vocabulary and an explicit "unknown to this build" branch in readers. Convert the growth-prone ones before the
   merge; it is the other half of making additive evolution cheap.
6. **For majors, expand/contract through a write-version pin** (later; not a blocker). `EvalStorage(
   write_core_version=...)`: a new build reads v15 but writes v14 during a rolling deploy, and the host flips once
   the fleet is upgraded. It needs a downgrader per major step, but majors become rare under 1–5, and the pin
   constrains only newer builds, so it can ship in any later release.

**What it buys.** Additive changes, the common case, roll without a stop-all, in either direction, with mixed
fleets. The package's own cost per change falls too: a minor needs a digest regeneration and a ledger sentence, not
a fixture-corpus copy and an upgrader, so evolution gets cheaper exactly where it is most frequent. The strictness
principle survives, scoped.

**Known cost.** During a roll, an old build reads new evidence as if the new fields were not there, so two builds
can print different numbers from one store for a few minutes. Bundles are fingerprinted with the package version,
so the mismatch is disclosed rather than hidden; the semantic rule in step 4 is what keeps it from being worse.

**Before the merge (irreversible, in `.22`):** the version representation, the reader's same-major tolerance rule,
round-tripping of newer-minor fields, and the open-vocabulary conversion of growth-prone literals. These are what
0.66 readers hard-code forever; roughly one to two days on the line. **Later (constrains only newer builds):** the
writer-side additive gate, the write-version pin and downgraders, and, optionally, per-type versions so a change to
`RubricScore` does not restamp runs and campaigns.

## Strong recommendations: expensive after consumers onboard

Ordered by how much adoption they buy. S1–S4 are the hour-two cliffs; they are additive, and they are the
difference between a consumer who stays on the ladder and one who hand-rolls.

### S1. Ship provider completion clients as extras

The judge rung requires a `CompletionClient` (`schema/completion.py:229`) whose results carry ten attributes
(`content, input_tokens, output_tokens, reasoning_tokens, cost_usd, price_source, model, served_model, stop_reason,
temperature`), including a `cost_usd` the consumer prices from a table they maintain. The examples share a 93-line
Anthropic adapter (`examples/_live.py`) that is "not an example of its own" and not in the wheel. A consumer who
wants "judge my 20 answers with Claude" copies that file on day one and now owns a price table and a stop-reason
mapping. That is the moment they ask whether calling the SDK from a loop is simpler. Ship `3tears-evals[anthropic]`
and `[openai]` with adapters and list prices; the protocol stays, the burden moves.

### S2. A threshold gate and a baseline gate on the quick path

`Comparison.gate()` and CLI `gate` are good (typed outcomes, `--fail-on`, exit 4) but need a control arm run in the
same campaign ("a run of the incumbent from last month is not a control", `docs/choosing-a-design.md:142-145`).
The two things every consumer wants in CI on day one — `accuracy >= 0.9` and "did this PR regress from main's last
run" — have no one-liner: a bar lives only in a host's `declared_design`, and `scope_history` flags regressions but
gates nothing. Consumers will write the `assert` themselves and lose every verdict guarantee. Add
`run_eval(..., bars={"accuracy": 0.9})` → `summary.gate()`, and either a gate against a stored run or a documented
CI recipe that re-runs the control at k=1 and says why that is the honest choice.

### S3. A middle rung between `compare` and "write a host"

The ladder goes from `run_eval` (in-memory, no launch) straight to a `LaunchHost` (courierhost 537 lines, a
13-field `KindWiring`, a launcher that must call `admit` from both `plan_arm` and `launch`). The thing most
products want first — a stored case bank over one production function, launchable, with a subject and a report —
cost Discodon 1,813 lines (`classifier.py` + `classifier_actions.py`), of which the domain part is ~50. Three
pieces close most of it: a package-owned launchable case-bank kind (cases on `host_payload`, graded by
`expected`, add/retire/seed actions); a helper deriving component-lever `Sweepable`s and their readers from a list
of names (Discodon wrote the reader three times: `levers.py:316-390`, `classifier.py:916-948`,
`reporter.py:162-203`); and a `LaunchableKind` slot for the shared admit.

### S4. Async facades for reading back, so engine threading never appears in user code

`run_blocking(host.blocking_executor, scope_history, host, …)` plus rebuilding `callable_host(<same scorers>,
store=<same store>)` is what the tutorial teaches at step 8 (`docs/tutorial.md:469-489`) and what
`choosing-a-design.md`, `examples/reports.py` and `examples/world.py` repeat. `StoreCalledOnEventLoop` is the right
rule for the engine; it is the wrong thing to meet at "keep your runs". Put facades on the objects the consumer
already holds: `await comparison.bundle()`, `await summary.history(metric=…)`, `result.trace()`. Once this idiom is
in consumers' CI scripts it is a breaking change to remove.

### S5. Tier the public surface before anyone holds it

1,482 names, with `tests/test_public_surface_is_closed.py` driving closure (every type reachable from a public
signature must be exported, so `EvalRun`'s 60 fields alone pull dozens in). The policy is honest — "any release may
break it", recorded in `MIGRATIONS.toml`, never shimmed — but with the family's exact-pin rule every move is a
forced consumer change, and the sole consumer touches 212 names. The practical host contract is perhaps 40
(`EvalHost`, `HostProfile`, the registries, `KindContract`, `DocumentStore`, `STORE_CONFORMANCE_CASES`,
`run_eval`/`compare`, `Report`, the refusal kinds). Add a `tier` to each snapshot entry (`host-contract` vs
`closure`) and promise the first tier MIGRATIONS-only treatment with a deprecation cycle. The machinery in
`scripts/public_surface.py` already supports it. Even `quick`'s `__all__` exports `CALLABLE_UNSEATED`,
`JUDGED_CALLABLE_KIND_CONTRACT` and `SHARED_ARM_MODEL`, plumbing a consumer should never see.

### S6. Naming hygiene before the vocabulary freezes

Each is small; together they are what a consumer repeats to a teammate, and renaming after three consumers onboard
breaks every saved report. The first `render()` prints `confusion_cell` (an internal measure key) and a run UUID;
`compare` reports name arms `candidate=v2` at model `callable`; *cell*, *kind*, *candidate* and *label* each carry
two meanings (`docs/concepts.md:117-123, 247-252`), and every example then uses `stratum_key="kind"`;
`metric=`/`reading=`/`measure` and `k`/`k_runs`/`intended_repetitions` coexist; `ArmKey`'s docstring
(`quick/compare.py:107-108`) contradicts `compare`'s (`:814-818`). Pick one word per concept, hide the internals in
`render()`, and move README's eighteen research nouns (`README.md:11-40`) below "First result".

### S7. First-consumer-shaped mechanisms sit in the kept schema

The host-noun canary proves the words are gone; it cannot see shape. Each of these exists because Discodon's
persona needed it, generalised honestly, and unproven by a second consumer: `AsyncDelivery`/`AsyncExternalSpend`
and the `inner_agent`/`external` members of `UsageRole` (`schema/models.py:3831-3875, 3994`; on `EvalResult`, a
core document; the runner has a dedicated refusal for it, `run/runner.py:1935`); `MeteredCallLedger` (search
credits); triggered world dimensions (seeded findings); `ChartFont` per-character metrics on a *host profile*
(brand font); `SeedSection` with primacy/reference/recency zones (`kernel/prompts/seed.py:47-62`, Discodon's
sectioned prompt registry, which three of four package-owned prompts are declared in). Not a blocker — upgraders
exist and the docstrings generalise — but mark them provisional in the docs, and treat the second consumer's
adoption as the test of whether each is an engine concept. Do not let a third consumer onboard before that test.

### S8. The release policy does not know that judge identity is a prompt fingerprint

`builtin:<sha12>` digests the engine's own judge prompt wording and `<config>@<sha12>` the reply-format suffix
(`schema/models.py:1610-1626, 1643-1665`), per result. Any wording improvement makes every prior judged result
"another judge" for agreement, profiles, drift and evidence tiers, so a consumer's `calibrated` tier drops to
`undetermined` on upgrade. Right design; the gap is that `MIGRATIONS.toml`'s `notes` convention does not list
"built-in judge wording changed" as a breaking change, so it will ship unannounced. Add it to the policy, keep a
wording ledger beside `builtin_judge_config_id`, and confirm whether run-level `judge_config_ids`
(`run/launch.py:3782`) carries the suffixed form, which would also re-key the context.

### S9. Put the dependency arrow right inside Discodon

Nine Discodon production modules import the eval package. `ApparatusError` was deliberately built as an
import-free leaf for the tool layer (`kernel/host/apparatus.py:40-44`); `ExternalSpend` was not, and
`run_judge_llm`, `extract_json`, `format_number` and `EvalBaseModel` are borrowed utilities. Either carve the
metering vocabulary into a leaf the package *intends* production code to import, or move it out of evals; and stop
production code using the engine's judge and JSON helpers. Cheap now, expensive once more product code leans on it.
This is the one finding addressed to Discodon rather than the package.

### S10. The free report should lead with what the evidence supports

The code-only report (what a consumer gets without a paid `analysis_generate`, which is MCP-only) has no
headline, no decisions and no "what to run next": a 13-column contrasts table with no cue which column to read. An
operator fluent in arm/control/guardrail/bar can act; one who is not gets facts without a recommendation. The CLI
`gate` exit code is the only built-in "should I ship" answer. A verdict-summary block in the serialisers, built from
`Report.verdicts`, closes most of this and costs no model call.

## Backlog

Grouped; none is foundational. The first group is worth scheduling as an "adoption kit" release, because every
item is something Discodon wrote, debugged in production, and the next consumer will otherwise write again.

**Adoption kit (lift from Discodon).** Boot lifecycle (`EvalHostBoot`: build off-loop, sweep abandoned runs,
seed, 503 while booting, cancel jobs before closing the pool; ~180 lines); REST error envelope and per-chart
compilation (`rest.py`, ~120 generic lines; check whether `ops.report_charts` now duplicates it); prompt drift
actions and the boot warning (~400 lines over the package's `prompt_drift`); `CallTiming` (wall-clock ceiling =
retries × (cap/tps + prefill), re-derived by every host today); an OpenTelemetry `TraceSink` as an `[otel]` extra;
the provider-fault exclusion rule (`provider_faults.py`, bought with four fix commits where outages were scored as
model quality); the seed file format, enum-axis case expansion and a `seed` CLI subcommand (the package's seeder
admits definitions but reads no files and writes no cases); `dump_scope`/`load_scope` in `storage`; the
resolve-once-per-launch client-policy pattern (`PolicyPinnable`, because `EvalHost.clients` is process-wide).

**API warts on the host route.** The circular `LaunchHost` construction (`launching: list = []; lambda:
launching[0]`, in both Discodon and courierhost); `reporter_profile(HostProfile(...))` wrapping to register a
package-owned kind; per-kind `variant_levers` merged by hand; `authoring_checks` as three callbacks the host must
remember to chain, one derivable from `LaunchHost.kinds`; `KindWiring`'s 13 fields; `control_end_states` as hand-
written ceremony for a host with two goal checks.

**Version sprawl.** Fold `CELL_MODEL_VERSION` into the bundle `schema_version` (same bump triggers,
`analysis/cells.py:57-65` vs `analysis/bundle/schema.py:1896-1898`); list `GOAL_CHECK_PROOF_RULES` in the
versioning table; document that `scope_id`, lever names, kind names and rubric-dim names are permanent once a run
exists (scope is inside the context hash, `kernel/identity.py:329-333`); tell hosts to stamp a version inside
`host_payload`; note that `cassette.params_hash` canonicalisation has no version and a change produces silent
replay misses.

**Quick-path utilities.** No retry/backoff and no `concurrency=` on `run_eval` (a 429 counts against the
candidate); no candidate-response cache (cassettes cover tools only, so adding a scorer re-pays every call); no
scorer helpers (every live example normalises by hand; `match` is whitespace-exact); no case-file loader or
schema; case sets and traces documented but with no example.

**Docs.** Lab-notebook sections inside how-to guides (`judges-and-calibration.md:463-533`,
`cost-and-budgets.md:116-211`) belong under Background; the examples and both reference hosts are not in the wheel,
so a PyPI user cannot follow "start here" for hosts; the runtime model (asyncio only; sync stores on a thread pool;
async drivers through a bridge on its own thread; `price_ceiling` runs on the loop) is stated in Discodon's code,
not the package's docs; Discodon's `test_public_api_only.py` re-implements the package's import-boundary walker
and should become one call to `nonpublic_evals_imports`.

## What is right, and should not be simplified away

- The tutorial and the examples ladder. One question and one new capability per file; all offline; stand-ins
  named as stand-ins. This is the model for the rest of the docs.
- Statistics that refuse to overclaim, each method proven by seeded simulation at the sample sizes the engine
  actually sees, with the misses kept in the suite as strict xfails.
- Guardrails as a separate pillar: non-inferiority with a required margin and direction, never traded against a
  gain.
- Content-addressed identity with pre-images stored beside keys on runs; apparatus tracked so a moved rig never
  pools with a still one; `served_model` disclosure; unpriced never shown as zero.
- The declared world and its conformance kit — the thing the README correctly calls new.
- The compatibility apparatus: frozen fixture corpus with pinned digests, git-backed surface snapshot that fails
  rather than passes without tags, the migrations ledger printed as release notes, the import-boundary walker
  shipped to hosts as a product, reads strict in both directions.
- The store port: eight methods, exact semantics, four shipped implementations, one conformance kit. SQLite on
  the standard library works out of the box.
- No ambient host, no process-global state, pydantic as the only dependency.
- Discodon's two patterns the next consumers should copy: the seed corpus as data reviewed like code, and
  `FidelityContract` declarations holding each kind to production's own constructors.

## Suggested sequencing

1. **`.22` on the line, one tag:** B1 (control pre-image, folded into the v14 baseline), B2 (cassette version
   space), B3's before-the-merge half (`major.minor`, the reader rule, round-tripping, open vocabularies). Discodon
   re-adopts; its data is wiped at the stop-all anyway.
2. **The develop PR as 0.66.0**, with S5 (surface tiers), S6 (naming) and S8 (judge-identity policy) decided in
   that PR or immediately after, because each gets more expensive with every consumer that holds a name.
3. **0.67, before the second consumer onboards:** S1–S4 and S10. These are what keep a new consumer on the
   ladder at hour two.
4. **Adoption kit** from the backlog, in Discodon's order of pain: provider faults, boot lifecycle, prompt drift
   actions, seed format and CLI, REST envelope, OTel sink.
5. **S7** is tested, not scheduled: the second consumer's adoption decides which provisional mechanisms are engine
   concepts.
