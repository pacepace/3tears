# Declarations Nothing Reads

> Status: pattern note, written 2026-08-21 from one product wave built against
> this library.
>
> Worked example: `14-eng-ai-survey` wave 3 — forty task shards and a
> known-defect ledger whose entries run to L250, at that repo's
> `docs/admin-platform/00-overview.md`. Line references below of the form
> **L###** are into that ledger's Known-Defect table, cited by entry id rather
> than by line because the file is still being edited; everything else names a
> file in that repo or in this one. A few ids are reused for two entries, so each
> citation below also carries enough of the entry's headline to disambiguate it.

One defect shape produced more of that wave's findings than any other, and no
amount of ordinary review caught it, because **every instance of it compiles,
type-checks, and passes its tests.**

The shape is a **declaration that nothing reads**: a model field, a database
column, a config flag, a function, a module, a whole subsystem — declared,
documented, often unit-tested, and connected to no live code path. The ledger
numbered them as it went and reached **twenty-one** before giving up on the
count (ledger **L231**, "`core/alerting_system.py` is the TWENTY-FIRST unwired
declaration"; **L247** calls its own entry "the twenty-somethingth"). Two
entries call it "this wave's most-repeated defect" (**L185**, **L250**).

This page is what that cost bought: the variants, why they survive, the four
checks that find them, and what this library can and cannot do about it today.

---

## The five variants

Only the first is what people picture. The other four are the ones that hide.

**1. Declared, never read.** The plain case.
`14-eng-ai-survey/src/survey_engine/config/survey_models.py:462` declares
`PersistenceConfig.retention_days` with a default of **90** and has zero
production readers, so the default was chosen while the field was inert
(ledger **L227**, "the TWENTIETH declaration-nothing-reads defect"). Seventeen
other instrument fields sit in the same file in the same state (**L220**).

**2. Read, never declared.** Worse, because it is silent *and* it looks wired.
`services/survey_execution_service.py:2215` and `:2230` gate option shuffling
and scale flipping on `getattr(question_config, "randomize_options", False)` and
`getattr(question_config, "flip_options", False)`. Neither field was declared on
`QuestionConfig` (`config/survey_models.py:131`) — they were declared on a
different model, `RandomizationConfig`
(`config/randomization_models.py:25-26`), that `QuestionConfig` never
references. The branch was therefore unreachable. `RandomizationConfig`'s own
fields had no production reader either (ledger **L220**), so **both**
spellings were dead and every committed instrument that asked for shuffling
silently got none and no warning — **26 lines across 8 files** under `configs/`,
counted at that commit with
`grep -cE '^[[:space:]]+randomize_options: true$'`. (The ledger entry names four
files; that is an undercount.) **A read with a default cannot fail.** That is
the whole mechanism.

**3. Written, never read**, and **4. read, never written.** `collectors.metadata_allow_list`
had a reader (`filter_entry_metadata`) and no writer, so every allow-list was
permanently empty (ledger **L181**). Its mirror image, `quality_flags` in
the export builder, was read out of an analysis blob nothing wrote, so it was
permanently `[]` (ledger **L186**).

**5. Read by exactly one caller, off the path that matters.** This is the
expensive one, because a reader-count of one reads as "wired".
`survey_versions.config_json` — the frozen copy of an instrument, the thing that
guarantees a respondent's questions cannot change under them mid-interview — had
exactly **one** production reader — `admin/tools/results.py:708`,
`snapshot = version.get("config_json")`, which computes quota cells and is not on
the respondent path. Every call site of `get_survey_config` read the mutable draft
instead, and that function took no version argument at all. **Freezing an instrument
did nothing, and the freeze guard protected a row nothing read**
(ledger **L218**, "Freezing an instrument did nothing, verified"; the reader
count is checkable at that branch's commit `22a1410^`).

A sixth case is the same shape one level up: a *process* nothing starts.
`survey_engine/admin/main.py` declared `def main()` at line 234 with no
`if __name__ == "__main__"` guard and no `[project.scripts]` entry, and the image
ran a different app — so a tool pod with twenty-six registered tools was
unreachable in principle (ledger **L166**; both facts checkable at that commit's parent).

---

## Why it survives everything

- **An optional read cannot fail.** `getattr(x, "y", False)`, `config.get("y", 90)`
  and `hasattr` guards all degrade to the default rather than raising. Variant 2
  is invisible for exactly this reason, and so is a `hasattr` gate that is
  always false (ledger **L137**, which found one sitting in front of a
  662-line manager that had zero production callers).
- **A unit test can assert the mock.** If the test constructs a `Mock()` shaped
  like the API you *meant* to write, it passes against an implementation that
  never had that API. Three such files were found in this one wave; the ledger
  puts the total at five (ledger **L232**).
- **A test that imports nothing it claims to test can never fail.**
  `src/tests/unit/test_alerting_system.py` is 625 lines long and contains no
  reference to the module at all: line 13 is the comment
  `# Import the actual alerting system module` followed by nothing. It
  manufactured the appearance of coverage over a **931-line** module
  (`core/alerting_system.py`) that had zero importers anywhere under
  `src/survey_engine/`, and three of whose mechanisms were broken as a result —
  a cooldown keyed on an id minted fresh per trigger so it never matched, an
  empty suppression rule that silenced every alert forever, and severity
  escalation comparing enum `.value` strings, under which `"critical" > "warning"`
  is False (ledger **L231**, which cites the internal line numbers).
- **A benchmark or performance suite the gate does not run is not coverage.**
  That module's only importer outside `src/tests/` was
  `scripts/create_performance_baselines.py:54`, and its only other real
  importers were four files under `src/tests/performance/`, which the repo's
  preflight does not execute.
- **Documentation asserts the wiring, and documentation is not checked.** That
  repo's own `CLAUDE.md` claimed "PII Detection: 6 integration points". The
  measured number was one, and `detect_pii` itself had zero call sites — it still
  does; what got wired was `redact_pii` (ledger **L132**, verified at HEAD
  against `src/survey_engine/analysis/pii.py:42`).

---

## Four checks that find them

### 1. Count readers, not references

A grep for a symbol name returns its declaration, its tests, its docstring
mentions and its entry in a changelog. None of those is a reader. The
measurement that works is: *every* reference, classified, with tests and docs
excluded, and the remainder read to confirm it is a call and not a comment.

Doing that honestly is what produced the wave's hardest numbers —
`core/alerting_system.py` (931 LOC), `core/advanced_metrics.py` (996) and
`core/monitoring_config.py` (751) came to **2,678 lines with no production
importer between them** (ledger **L232**), and the branch went on to delete
57 files totalling 17,220 lines outright (measured with
`git diff --diff-filter=D --numstat <merge-base> HEAD`).

Note the trap in the other direction: the same wave recorded a subsystem as
"dead at both ends" and was wrong, because `data/collections/export_jobs_data.py:67`
imported the publisher **at module level**, cascading into the service registry
and the bootstrap (ledger **L241**). A module-level import is a reader even
when nothing calls what it imports.

### 2. A/B every guard: revert it and watch the test fail

A guard that has never been observed to fail is a guard that may not
discriminate. Removing it and re-running the test that supposedly proves it is
cheap, and this wave found guards that passed either way more than once:

- A single-use-link proof raced twenty redeemers and passed — **and went on
  passing with the concurrency handler disabled entirely**, because a shared L1
  handed every loser the winner's already-redeemed row. Rule left behind: *a
  concurrency claim about a three-tier collection is not tested until the racers
  have separate L1s* (ledger **L108**).
- The rebuilt proof was still too weak: with all fencing off it admitted 19, 15
  and 5 racers across three runs, so **a run landing on 1 would have passed with
  no fence in the build** (ledger **L113**).
- An accessibility test asserting a live region is "always mounted" passed
  against a deliberately-broken conditional version, because the two are
  textually identical (ledger **L236**).
- Two guards could never have discriminated at all and were found by reading:
  one was written `A if is_list else B == pk_cols`, which Python parses as
  `A if is_list else (B == pk_cols)`, so the comparison never ran (ledger **L46**).

And the A/B can lie to you if the thing you edited is not the thing under test.
That repo installs this library as an editable overlay, and its type-check
shelled out to `uv run mypy` **without `--no-sync`**, silently re-syncing and
dropping the overlay — so an A/B probe measured the old code and reported a
false pass (ledger **L114** and **L179**). If you A/B across a package boundary,
prove the edit is in the interpreter you are measuring.

The A/B is also worth running when you expect it to *confirm* the guard, because
sometimes it comes back against the design. On one derived-key table, the CAS
fence let 2 of 20 concurrent writers land while **no fence at all** let 20 of 20
land and still produced exactly one row: the one-row guarantee came from the
derived primary key, not the fence, and the fence was actively harmful because
its caller did not retry. It shipped without one (ledger **L138**).

### 3. Derive inventories from the running system, never by hand

Any enforcement rule that begins with a hand-written list of what to check is a
rule whose coverage is a guess. Stated plainly in the ledger: a hand-written
module list "is precisely how three surfaces were missed", and replacing it with
a structural derivation — any module importing a concrete chat-model type or
constructing a framework message, minus a four-entry allow-list each carrying a
written rationale — immediately found two more, which were fixed rather than
allow-listed (ledger **L141**). The same entry names the precedent: "same
failure mode as the route allow-list before it was derived from the running app."

That precedent is worth stating exactly, because the honest version is worse than
the tidy one. **Eight configuration endpoints on that product's public app carried
no authentication at all** — they were public *by omission rather than by decision*,
and no inventory of any kind existed to notice (ledger **L101**). The
derived route-auth test that now fails any unauthenticated operation missing from a
justified allow-list is what closed it. And the same entry records its own
correction, measured rather than reasoned: on a production-wired app those eight
returned 500 anyway, because the service registered behind them implemented none of
the methods the routes called (ledger **L106**) — so the exposure was real at the route level
and not at the data level. Both halves belong in the record. A guide that keeps only
the alarming half is doing the thing this page is about.

The derivations that wave ended up with are the transferable part:

| Inventory | Derived from |
|---|---|
| Which HTTP routes may be unauthenticated | the app's own route table, failing any operation with no security requirement that is not on a justified allow-list |
| Which routes belong to which of two pods | both apps' own OpenAPI documents, asserting a set relation; import rules read from the AST, because the import that defeated the earlier attempt was function-local |
| Which modules may reach a model | AST: modules importing a concrete chat-model type or constructing a framework message |
| Which stores a backup must cover | the collections' own `TableSchema` set, plus the tables `threetears.langgraph` owns, plus the compose file's named volumes |
| Which reads must honour a pilot flag | which interface methods declare the keyword, not a list of call sites |
| Whether the generated data section still matches | byte-identity against the generator's own output |
| Whether a foreign key has the shape it claims | a live `information_schema`, not the source — "precisely because this hazard's failure mode is silence" |

Two properties make these hold up. They **fail in both directions**: a frozen
set of known leaks may shrink and must not grow, *and* an entry that has been
fixed also fails, so the list cannot rot. And every exemption carries a written
reason, so a stale exemption is visibly stale rather than merely old.

### 4. Treat "declared" and "wired" as separate claims in review

The cheapest fix is a habit rather than a tool. When a change adds a field, a
column or a flag, the reviewer's question is not "is this declared correctly"
but **"name the line that reads it, on the path that matters."** Variant 5 above
is the reason the second half of that sentence is there: `survey_versions.config_json`
had a reader, and the interview still ran against the wrong instrument.

One more, because it caused a wasted shard: **an escalation inherits its
author's framing.** A shard escalated a choice between two seams for relocating
an instrument's draft body. Both options were answerable and neither would have
helped, because nothing read the *frozen* body either — moving it would have
moved one unread body to another table. The ledger's own generalisation is worth
copying: *"an escalation inherits its author's framing, and the checker should
re-derive the problem before picking between the options offered"*
(ledger **L219**, "an escalation inherits its author's framing").

---

## What `threetears.enforcement` covers today

The package ships **fourteen** static-analysis domains — each a directory under
`packages/enforcement/src/threetears/enforcement/` with `config.py`, `runner.py`
and `walkers.py` — plus `common/` for shared scaffolding:

`cache`, `codebase_conventions`, `coercion_coverage`, `dependency_alignment`,
`dict_state_detection`, `fake_parity`, `jwt_alg_pinning`, `logger_coverage`,
`migration_yugabyte_safety`, `nats_wrapper_usage`, `no_silent_swallow`,
`no_stdlib_logging`, `single_return`, `underscore_access`.

Three of those are missing from the catalog table in
`packages/enforcement/README.md:13-25` (`fake_parity`, `jwt_alg_pinning`,
`single_return`), despite that README's own line 79 instructing an author to
"Document the domain in this README", and `docs/adoption/enforcement.md:17`
still says "roughly ten". The count is only derivable from the directory
listing. That is worth fixing, and it is a small instance of the same shape this
page is about: a documented inventory with no check that it is complete.

### The adoption is worth doing, and here is the measured reason

That product repo declared `3tears-enforcement` as a dependency and imported it
**nowhere**, while hand-writing the same scanners beside it — 48 enforcement
files when the ledger entry was written (ledger **L147**), 72 today. Nine
domains were adopted in one change. The single most informative result:

`threetears.enforcement.single_return`'s own module docstring warns about
exactly one hand-rolling mistake
(`packages/enforcement/src/threetears/enforcement/single_return/__init__.py:10-14`):

> nested `def` / `lambda` scopes are charged to themselves rather than to the
> enclosing function. that is the one place a hand-rolled version of this walker
> reliably goes wrong, and the reason this lives in the shared package: the fix
> had to be applied twice, in two verbatim copies, before it moved here.

The hand-written copy had that exact bug. It ran `ast.walk` over each statement
and tried to skip nested functions with `if node is not stmt: continue`, which
skips the `FunctionDef` node and then walks straight into its body — so every
return inside a nested helper was charged to the enclosing one. Swapping to the
shared walker on the same tree took **389 baselined violations across 109 files
down to 226 across 79**, and the new set was a **strict subset** of the old:
zero findings the previous baseline had not already recorded, so no detection
was traded for the drop (`14-eng-ai-survey/src/tests/enforcement/test_single_return_enforcement.py:19-24`).

That is the argument for the package in one number. A hand-rolled analyser does
not merely duplicate work; it duplicates work *and* reproduces the bug the
shared one was extracted to fix.

### Two gaps this wave hit

**No domain has a ratchet.** `SingleReturnConfig` offers `exempt_files` and an
exemptions file, and both silence permanently. Neither shrinks, and neither can
express "you touched this file, so fix its violations". The consuming repo kept
its own baseline for that reason: swapping a shrinking baseline of 226
tracked-for-repair sites for an exemption list of 79 permanently-silenced files
is not the same mechanism (`test_single_return_enforcement.py:26-33`). A
baseline-with-caps that can only tighten is a reasonable candidate for
`common/`.

**Nothing checks for a declaration with no reader at the field or module
grain.** The package already has the idea at the *dependency* grain —
`dependency_alignment` enforces "no declaration nothing imports"
(`packages/enforcement/README.md:19`), which is this defect one level up, and
which is exactly the rule that would have caught the abandoned
`3tears-enforcement` pin in the first place. (It was the one domain that
consumer could not adopt: it fails immediately on four dependency facts whose
only fix is editing `pyproject.toml`, which that wave's release policy forbade
mid-wave. Adopt it in the change that cuts a release, not in the change that
adopts the rest.) The nearest thing at the finer grain is a rule the consuming
repo wrote
and that is a fair promotion candidate: fail a **commented-out import of the
package under test** inside the test tree
(`14-eng-ai-survey/src/tests/enforcement/test_no_commented_out_imports_in_tests.py`).
Its docstring is worth reading before generalising it, because it records both
the win and the limit honestly: the rule flags the `# from x import y` decoys,
and it deliberately does **not** attempt the general form ("a unit test
importing nothing from the package"), which flagged 57 files there — nearly all
legitimately, since every static-analysis test reads source as text. "A rule
with 55 false positives is a rule that gets exempted into silence."

A genuine unread-declaration domain is harder than it looks and should be scoped
narrowly if it is attempted. Two shapes look tractable within one repository:

- a Pydantic/dataclass field whose name appears nowhere outside its own model
  definition, its tests and its docstrings;
- a `getattr(obj, "name", default)` where `obj`'s annotated type declares no
  `name` — variant 2, which is the silent one and which a type checker will not
  flag because `getattr` with a literal is not narrowed.

Both are marked here as *proposals, not implementations.* Neither has been
written, and the cross-repository case (a column read by a different service) is
out of reach of static analysis entirely.

---

## Three primitives this library offers that one consumer could not use

Recorded because a refusal with evidence is more useful than a silent
reimplementation, and because two of the three are close to usable.

**`threetears.observe.resilience.retry_with_backoff`** (`packages/observe/src/threetears/observe/resilience.py:22`)
cannot back a compare-and-swap loop. It takes `Callable[[], Awaitable[None]]`,
so it cannot carry a value out; it catches bare `Exception` at `:84`, so a real
driver error becomes a retry and then a swallowed `False`; and its docstring
states it "never raises" (`:33`). A CAS loop must retry *only*
`ConcurrentModificationError` and propagate everything else. The consumer kept
five hand-written loops rather than lose that distinction. A value-returning,
selectively-catching, raising variant would be additive and would absorb them.

**`threetears.agent.acl.query_visibility.customer_scope_visibility_clause`**
(`packages/agent/acl/src/threetears/agent/acl/query_visibility.py:148`) is an
RBAC visibility filter, not a partition predicate. Its signature takes a caller
`user_id`, and the SQL it emits joins `role_assignments`, `group_members` and
`namespaces` (`:131-133`) — tables a product's own database does not have.
Substituting it for a `WHERE customer_id = $1` on an already-verified customer
would also discard a primary-key-leading equality. Ninety-seven such predicates
stayed hand-written, correctly.

**`threetears.backup.BackupEngine`** cannot perform a pool-to-pool move.
`create_backup(source_dsn, ...)` (`packages/backup/src/threetears/backup/engine.py:83`)
is keyed by a bare DSN with no table or row predicate, and it unconditionally
writes the dump to the `ObjectStore` at `:95` before returning a key;
`restore_into` reads back from that key (`:102`). There is no source-to-target
path that skips the artifact. Relocating one tenant's rows between two live
pools is a different operation, and the consuming repo streamed it rather than
producing an encrypted extract nobody asked for. A per-tenant extract remains a
reasonable additive request; whole-table `pg_dump --table` selection would not
satisfy it, since the selection needed is by row.

---

## The short version

- A declaration is not wired until you can name the line that reads it **on the
  path that matters**.
- A read with a default cannot fail, so it cannot tell you it is unreachable.
- A guard you have never watched fail may not discriminate. Revert it and re-run.
- Any inventory a rule checks should be derived from the running system, and
  should fail both when it grows and when an entry goes stale.
- A test that imports nothing it claims to test can never fail, and is worse
  than no test, because it manufactures the appearance of coverage.
- Before hand-rolling an analyser, read the shared one's docstring. It probably
  names the bug you are about to write.

---

## See also

- `14-eng-ai-bot-agents/docs/guides/building-a-product-service.md` — the other
  half of what that wave learned, written for the next product author rather
  than for this library: choosing a shape, tool faces, the two authorization
  grains, tenancy the framework enforces, and the known platform edges. This
  page tells you how to know whether what you built is running; that one tells
  you what to build.
- `packages/enforcement/README.md` and `docs/adoption/enforcement.md` — the
  domain catalog, with the caveats about its completeness noted above.
