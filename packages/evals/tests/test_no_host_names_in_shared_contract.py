"""No-host-names-in-the-shared-contract canary.

**The norm this enforces:** *new eval machinery lands host-agnostic by construction.* Three
enforcers hold it — a toy host in CI, an AST import gate, and a no-host-names-in-shared-registry
test. This is the third.

It exists because the other two cannot see this failure. An import gate goes green on every
module below, because they import nothing from the first host's packages — and a module with
zero host imports can still be dense with host concepts. The concept crosses the boundary where
the import does not.

Rule (AST walk, no imports of the files under test). Across the declared
shared-contract module set, none of these may contain a host noun as a whole word:

- a class name
- a function name
- an assignment target at module or class level — this is where a pydantic field name
  lives
- a parameter of a ``def``, an ``async def`` or a ``lambda``, including ``*args`` /
  ``**kwargs`` — a signature must not escape the scan by the form it was written in
- a ``Literal[...]`` member, wherever the annotation sits: a field, a parameter, or a
  return type
- a **token-shaped** string constant inside a declared value: a field default, an enum
  member, a parameter default, or an entry in a declaration table

Host nouns are :data:`~packages.evals.tests.host_vocabulary.HOST_NOUNS` — ``persona``,
``discord``, ``music``, ``dj``, ``universe``, ``segue``, ``research``, ``tavily``,
``openrouter`` and a room's vocabulary (``listener``, ``arrival``, ``presence``,
``audience``) — case-insensitive, with an optional plural ``s``. Matching is **whole-word after splitting identifiers** into their parts
(snake_case, kebab-case, camelCase, digit boundaries). Whole-word is load-bearing:
``dj`` as a substring matches ``adjust``, ``adjacent`` and ``adjective``, and a canary
that cries wolf gets switched off.

**What is deliberately NOT scanned, and why:**

- *Docstrings, comments and* ``Field(description=...)`` *prose.* Scoped out here, and not
  unguarded: ``test_host_vocabulary_ceiling.py`` counts host terms in every module's full source against
  the per-module, per-term ceilings in ``host_vocabulary_register.py``, which only fall. That scope-out is also why a string
  counts only when it is **token-shaped** — no whitespace, at most
  :data:`_MAX_TOKEN_LENGTH` characters. ``EVAL_ANALYSIS_GEN_DEFAULT`` in
  ``analysis/gen_prompt.py`` is the live demonstration: a module-level assignment whose
  value is the whole shipped generation prompt, so without the token-shape rule every
  sentence of it is scanned as a "field default". It contains ``research.research_model``
  — whole-word ``research``, a registered host noun — in a sentence teaching a model NOT to
  compare one role's value with another's. So the constant would register as a violation
  on prose alone, and the same constant demonstrates the whole-word rule below: it contains
  ``dj`` as a substring, inside ``adjective``, and ``dj`` is a registered host noun. (Derive a claim like this one from
  :data:`~packages.evals.tests.host_vocabulary.HOST_NOUNS` rather than from a hand-typed noun
  list: a recalled list is how an example of this kind goes false.)
- *Every string outside the three surfaces that carry one.* A string value is scanned when it is
  the right-hand side of an ``=`` or an annotated ``=`` (``declared-value``), a parameter default
  (``param-default``), or a ``Literal[...]`` member of any annotation — a parameter's, a
  variable's, or a return type's (``literal-member``). Those are the three ``surface`` labels the
  scanner emits for a value, and :class:`TestWhichStringsTheScannerCanSee` reads that vocabulary
  out of :class:`_ContractScanner`'s own source, so a fourth producer fails a test rather than
  waiting for someone to notice this sentence. Position within the file does not matter: ``visit_FunctionDef`` ends in
  ``generic_visit``, so an assignment INSIDE a function body is scanned exactly like a
  module-level one — ``reporting.py``'s
  ``coordinates: dict[str, Any] = {"subject_id": …, "scope_id": …}`` is a live instance.
  Everything else escapes: a ``return``, a call argument, a subscript target, a ``for`` iterable,
  a comparison, an augmented assignment and a walrus — the last two despite binding a name.

  **This bullet has been wrong five times running**, each version claiming a wider blind spot
  than exists, and the fourth was used as the recorded justification for a design decision on
  another module. The fifth said "anything that is not the right-hand side of an ``=``", which
  :func:`test_the_scanner_flags_a_synthetic_violation_on_every_surface` already contradicted with
  a scanned parameter default and a scanned return-type ``Literal``. So it is not maintained as
  prose:
  :class:`TestWhichStringsTheScannerCanSee` enumerates every shape, and the sentence above is a
  summary of that class rather than an independent claim.
- *Log format strings and the values interpolated into them.* A declared name is scanned; an
  operator-facing MESSAGE is prose, and scanning prose would flag every sentence that mentions
  a host. The package's logging rule is that **a log key names the VALUE it carries, not the
  module that emits it**, so every ``eval.*`` line in the package keys the scope
  ``scope=``, after the ``scope_id`` it interpolates. Nothing will redden if a later edit puts a
  host noun in a shared-contract log line; the review question the norm states is the only check
  there is.
- *Attribute access on host objects.* The canary asks what a module **declares**, not what it
  reads, so a host-named attribute read inside a function body is invisible here. That is a
  different coupling class with its own item, and folding it in would bury the register this
  canary exists to keep short. No live instance is left in a scanned module (the search was for a
  dotted read whose attribute contains a host noun, in code rather than prose); an example named
  here goes stale as soon as the read it names goes away, so the next reader should search for one
  rather than trust this sentence.

**The allowlist is a debt register, and it may only shrink. It is empty**, and the four
properties below are what keep it from growing back by
anyone's oversight. Every entry needs a one-line reason and a linked issue — never a bare pass
list. Four properties make that structural rather than aspirational, each pinned by its own
test below, and all four still hold over an empty dict:

1. A new violation fails this canary unless an entry is added for it.
2. An entry matching no live violation **fails**. So the list cannot rot, and paying a
   debt without deleting its entry goes red — the shrink is forced, not remembered.
3. An entry with a blank reason fails.
4. An entry with no ``#NNNN`` issue reference fails.

Plus :data:`_ALLOWLIST_CEILING`, which must equal the register's size — so growing the
list is a two-line, diff-visible, deliberate act. Being honest about its limit: that is
a speed bump, not a lock. The only unbeatable form compares against the merge base, and
a `source_scan` canary has to give the same verdict in a worktree, a shallow CI clone and
a released tarball. What is enforced is that growth cannot be **silent**.

The module set is **declared**, not derived. "The modules with no host imports" is the
tempting derivation, and it does not hold in either direction: a host-agnostic module can import
one shared host base class, and a module with no host imports can be host-coupled by vocabulary
or role. Deriving the set from an import count would let it drift with an unrelated import;
declaring it means a module leaves this contract only by someone saying so.

**A declared set has to be extended, or it silently stops covering the package.** A module
that lands after the declaration with nobody classifying it leaves the canary passing, which
proves only that it is not looking — so the completeness test below refuses an unclassified
module. Three classifications are worth their reasons:

- ``contracts/provider.py`` — **shared contract, declared.** It is the extraction seam's own port:
  the completion protocol every eval consumer is constructed with, plus the provider
  trivia the host's LLM layer used to own. A second consumer implements
  ``CompletionClient`` and inherits the rest verbatim, so nothing in it may name a host.
- ``run/fidelity.py`` — **shared contract, declared, and only after a split.** The mechanism
  (``FidelityContract`` plus the two checkers) is generic, but a registry naming real host
  modules belongs to the host, so the package holds none. A registry of dotted host paths is
  also the measure of how little this canary can see of string-carried coupling: a dotted path
  is caught only when one of its parts is a host noun, and the rest are invisible to every name
  gate. ``test_hostneutral_fidelity.py`` carries the text-level check; this canary is not the
  right instrument for it.
- a candidate kind's host half — **host-side, never declared.** Every kind IMPLEMENTATION sits
  on the host side of the kind seam, and an extraction deletes it rather than ports it.
  Declaring one would also read as a false green — it scans clean because its coupling is
  *imports*, not names, and a pass on a file whose whole purpose is host coupling teaches the
  wrong lesson about what a pass here means. A host's own fidelity registry is out for the same
  reason.

**A per-module classification cannot express "this one type", and a host noun survived in the
gap.** ``models.py`` is classified host-coupled, which is right for ``EvalRun`` and
``EvalResult`` and wrong for ``ContextComponents`` — a value type that ``identity.py``,
declared shared contract, both produces and consumes, and whose FIELD NAMES sit in the pre-image
of every context key the engine mints. So a component called ``universe`` lived in declared
shared-contract arithmetic while every scan of the shared contract went green, and it survived a
repo-wide ``universe`` -> ``scope_id`` rename for exactly that reason.

:data:`_SHARED_CONTRACT_TYPES` closes it: a host-coupled module may declare the individual
classes inside it that a second consumer inherits verbatim, and those class bodies are scanned
under a ``module.py::ClassName`` label as if they were their own file. This is a widening of the
instrument rather than a new rule — the register's ceiling is unchanged, and a host noun inside a
declared type fails the same test everything else does. Recording the gap instead was the
alternative, and it was rejected on the evidence the gap already produced: what a recorded gap
buys is a reader who knows, and what closed it buys is the next ``universe``-shaped noun failing
a test on the day it is typed. (``models.py`` itself is now declared shared contract whole, so
the register holds no entry today.)

The entry is deliberately narrow. A type belongs here when the ENGINE's own shared code reads or
writes it — not when it merely crosses the wire — because that is the property the module
classification is a proxy for, and a list that grew to "every persisted type" would be
``models.py`` reclassified by the back door with none of the review a reclassification gets.
"""

from __future__ import annotations

import ast
import re
import shutil
import sys
from pathlib import Path
from typing import NamedTuple

import pytest

from packages.evals.tests.host_vocabulary import HOST_NOUNS
from packages.evals.tests.identifier_words import identifier_words
from packages.evals.tests.source_tree import tree_subdirectories
from packages.evals.tests.import_resolution import absolute_module


_REPO_ROOT = Path(__file__).resolve().parents[1] / "src"
EVAL_ROOT = _REPO_ROOT / "threetears" / "evals"

#: The declared shared-contract module set — the code a second consumer inherits
#: verbatim. Paths are relative to ``threetears/evals/``. ``analysis`` is a whole tree, so a file
#: added to it is covered the day it lands rather than the day someone remembers.
_SHARED_CONTRACT_MODULES: tuple[str, ...] = (
    # `python -m threetears.evals`: hands over to the command line in `quick`, a whole shared tree.
    "__main__.py",
    # The stored analysis shapes, which are contracts rather than analysis (the stored campaign,
    # analysis and insight models, the campaign declaration, the decision surface, the authored
    # document shape, and the analysis measures). `contracts/` is split, so each is listed.
    "contracts/campaign.py",
    "contracts/declaration.py",
    "contracts/surface.py",
    "contracts/authored.py",
    "contracts/analysis_measures.py",
    # The engine's own Pydantic base. Shared contract by construction — it exists precisely so
    # that the models above it can be constructed in a host that has none of the first host installed —
    # and every module in this list sits on top of it, so a host noun reaching here reaches all
    # of them. It names the host base it was copied from, in prose, and that is the point of
    # scanning it: the next edit is the one that reaches for a host TYPE instead of a sentence.
    "contracts/base.py",
    # The contracts package's public root: its ``__all__`` is what a
    # second consumer imports, so every name it exports is scanned here. Declared module by module,
    # like `gen/__init__.py`, because `contracts/` was classified module by module while it held
    # host-coupled modules, and the list stays the record of each one's move.
    "contracts/__init__.py",
    "analysis/bundle.py",
    "analysis/gen_prompt.py",
    # The gen package's public root, its exports scanned. Declared module by module, like
    # `contracts/__init__.py`, because `gen/` was classified module by module while it held a
    # host-coupled module, and the list stays the record of each one's move.
    "gen/__init__.py",
    # The rubric proposers. Shared contract by construction: they take the subject and catalog feeds
    # as text the host renders, so a second consumer drafts with them verbatim, and a host noun here
    # would be the subject's shape leaking back into the drafting code it was rendered out of.
    "gen/proposers.py",
    # The run package's public root, its exports scanned; declared module by module for the
    # reason `gen/__init__.py` is.
    "run/__init__.py",
    "contracts/dsl.py",
    # The model-prose marker. Shared contract by construction: the models above
    # declare their prose fields with it and `dsl.py` reads its schema helpers.
    "contracts/prose.py",
    # Where one JSON Schema can sit inside another: the one answer the honoured-subset audit, the
    # registry's self-contradiction check and the prose gate all walk.
    "contracts/schema_nesting.py",
    "contracts/errors.py",
    # The write-seam refusal of an undeclared authoring field, and the lock every writer of an
    # existing campaign holds. Each is shared by two packages that may not import each other —
    # authoring in run and in analysis, campaign writes in analysis and run's delete cascade — so
    # each lives in contracts, and a host noun here would reach both sides at once.
    "contracts/authoring_fields.py",
    "contracts/campaign_writes.py",
    # The run-status filter's vocabulary, its two defaults and its refusal: read by the run listing
    # in run and by the comparison lenses in analysis, which may not import each other.
    "contracts/status_filter.py",
    # The blank-argument rule, on the same terms one argument over: read by the analysis lenses
    # and by the run package's launch, which may not import each other.
    "contracts/arguments.py",
    # Which model scored each dim, and what a run may claim about it: a rule over two fields the
    # stored run carries, read by the context key, the comparison badge and every renderer.
    "contracts/judge_attribution.py",
    "contracts/metrics.py",
    "contracts/identity.py",
    # The shared pre-image of every eval key. Nothing in it is host-shaped and nothing in it may
    # become so: it is in the dependency closure of `identity.py` and the whole `analysis` tree,
    # so a host noun reaching here reaches every key and every bundle at once.
    "contracts/hashing.py",
    "contracts/provider.py",
    "run/fidelity.py",
    # The criteria judge. It left the host-coupled set when its last host reach went: the JSON
    # parser it calls now lives in contracts, and it logs under its own module name rather than the
    # host's cost-logger family. Scanned so the next host reach is a red build, not a quiet one.
    "run/judge.py",
    # The external-spend vocabulary. It is the one place a provider name could most plausibly be
    # hardcoded again — the shape it replaced held one provider's configured rate as a module
    # constant — so it is scanned rather than trusted.
    "contracts/spend.py",
    # The seeder, not the corpus it ships. It resolves occupancy by each type's natural key and
    # writes through the storage port; the host-specific part is the corpus a host hands it.
    # A host noun appearing HERE would mean the empty-slots-only rule
    # had been written against one host's vocabulary, so it is scanned rather than exempted.
    "run/definition_seed.py",
    # The authoring family — template, catalog rubric-dim and judge-config CRUD and the
    # engine's authoring refusals — over the storage it writes through. What only a host can
    # answer (its tool catalog, its world's seed walk, its kind capability table) arrives as a
    # callable, so a host noun appearing HERE would mean one host's vocabulary had been written
    # into the authoring contract every host inherits. A template's kind content is the kind's
    # spec model's to validate, and arrives through the host's kind contract.
    "run/authoring.py",
    # The check-controls gate authoring calls: a goal check is proven against the template's own
    # seed and an author's control end state, through the run's own grading function. It reads a
    # host's world and tools only through the profile it is handed, so a host noun here would be one
    # host's vocabulary written into the proof every host's checks must pass.
    "run/check_controls.py",
    # How the engine hands a blocking call to an executor its host chooses. Shared contract by
    # construction: it exists so the engine names no executor, pool or config of its own, and a
    # host noun here would be the engine choosing one host's pool for every host.
    "run/offload.py",
    # The run listing and the result reads every read surface stands on. They take the scope as
    # ``scope_id``, so nothing here names the host's partition, and a host noun arriving would be a
    # host's listing rule written into the engine's.
    "run/reads.py",
    # `stats.py`, `numbers.py` and `reporting.py` (the aggregation core the extraction boundary is
    # defined around, shared contract since its host-named parameters and operator string went)
    # live in the `analysis` tree and are scanned with it.
    # The four below moved off _HOST_COUPLED_MODULES when the coupling that put them there was
    # paid, on the same condition reporting.py moved under: they import no first-host module and
    # carry no host noun anywhere this scanner looks. Moving a module
    # here only ever ADDS scanning, so the risk of the call is one direction — and the risk of
    # NOT making it is the one this file already names: a module that has become shared contract
    # while the scan keeps skipping it, where the next edit can put a host noun in a field name
    # and stay green.
    #
    # `budget.py` and `metering.py` read host configuration until the ceilings arrived as values,
    # and the run cost cap is engine machinery a second consumer inherits rather than a port.
    # `result_condition.py` inherited the host's model base until the engine had its own, and its
    # models are now eval-owned by declaration. `store_port.py` was
    # never coupled at all — its own docstring calls it the one module in the package that imports
    # nothing, so an extraction keeps it verbatim, which is the definition of shared contract
    # rather than an exception to it.
    "run/budget.py",
    "run/metering.py",
    # The cascade those two used to hold a copy of each: override-or-default, the
    # enforcement-off short circuit, and which tier answered. It reads no configuration and
    # names no currency beyond `float` and `int`, so it is shared contract from the day it
    # existed rather than by promotion — and it is scanned because the host noun that would
    # most plausibly arrive here is a third ceiling named after whatever the host meters.
    "run/ceilings.py",
    "contracts/result_condition.py",
    # Which of a result's judged dims failed, and how a re-judge's outcomes land on it: pure
    # functions over eval models and the judge's outcome type, with the I/O left to the service.
    "run/rejudge.py",
    "contracts/store_port.py",
    # The curation family, the variation generator and the job manager moved off
    # _HOST_COUPLED_MODULES when the runtime tier's host-named partition became `scope_id`: that
    # parameter name was the only host coupling any of them exhibited where this scanner looks,
    # so with it gone they are scanned like the rest.
    "run/curation.py",
    "gen/variation_gen.py",
    "run/jobs.py",
    # What a result scored, what a set of results scored together, and whether a run's loop
    # delivered the matrix it promised. It carries `CellSummary` with it — the bounded per-cell
    # record `summarize_completeness` counts. Shared contract from the day it existed rather than
    # by promotion: these are pure functions over eval models, and they are here because they sat
    # in `runner.py`, which an extraction replaces wholesale, so an engine module wanting a
    # composite had to import the one file that cannot travel. Scanned because the host noun that
    # would most plausibly arrive here is a scoring rule stated in one host's vocabulary.
    "contracts/scoring.py",
    # The candidate-kind seam: the protocol a kind implements, the object it hands back, and
    # the two failures the dispatch site tells apart. Shared contract from the day it existed
    # rather than by promotion — every kind IMPLEMENTATION lives host-side and this file is the
    # declaration they implement, so a second consumer inherits it verbatim and writes its own
    # kinds against it. Scanned because the standing objection to this seam is that a host's
    # shapes would cross it: this canary is what makes that a failing test rather than a
    # paragraph, and it is the check that moved the async-delivery record off
    # `CandidateTelemetry` and into engine vocabulary (`AsyncDelivery`) beside the kind's opaque
    # `kind_payload`.
    "contracts/candidate_kind.py",
    # The cassette seams a kind implements and the cell handle ``prepare`` is handed. Split out of
    # `run/cassette_proxy.py` so the protocol that declares ``prepare`` could name its argument: it
    # is what every host's kind is written against, so a host noun here would be one host's
    # candidate shape written into every host's contract.
    "contracts/cassettes.py",
    # The safe-edit protocol for a run document, and the two-method port it composes. It moved
    # out of `storage.py` because the policy names no backend, no tier and no host — `EvalStorage`
    # is one implementation of the two methods it drives and a test double is another — while
    # living beside `EvalStorage` made every engine caller import the module an extraction leaves
    # behind. Its scope parameter is `scope_id`, the engine's word for a partition it never
    # interprets.
    "run/run_document.py",
    # What happens to a run after its launch: reading it, cancelling it, reclaiming abandoned ones,
    # re-judging a result and recording completeness. Shared contract from the day it existed rather
    # than by promotion — it left the host's service so a second consumer manages its runs with it, and
    # takes `scope_id` and a host-built judge subject.
    "run/lifecycle.py",
    # Tool recording and replay. It moved off _HOST_COUPLED_MODULES when the lane stopped reaching
    # into a host's candidate and started wiring only the seams a kind hands it (now declared in
    # `contracts/cassettes.py`); what is here records and replays through them for every host alike.
    "run/cassette_proxy.py",
    # The three below moved off _HOST_COUPLED_MODULES when the judge stopped rendering one host's
    # subject and transcript. The judge now places the evidence a kind renders
    # and reads none of it, the simulator drives a template's `conversation` block against "a
    # candidate", and the world's seed is data rather than references into a subject record. A
    # host noun arriving in any of them would be one host's subject written back into the engine.
    "run/judge_service.py",
    "run/simulator.py",
    # The speaker-round loop over the simulator above: it hands a kind's own delivery and candidate
    # callables the turns, and a host noun here would be one host's table written into every kind's loop.
    "run/conversation.py",
    # The call ledger every kind fills and every goal check's call predicates read: tool and action
    # names are the host's words carried as data, never declared here. A host noun in this module
    # would be one host's tool written into the record every kind keeps.
    "contracts/call_ledger.py",
    # The re-check that re-grades a stored run from the ledgers its cells stored, through the run's
    # own grading function. It reads no kind's trace shape, so a host noun here would be one kind's
    # replay written back into the engine.
    "run/recheck.py",
    # Moved off _HOST_COUPLED_MODULES when a launch's overlays became one map the kind's own model
    # validates: the launch names no overlay of any host's, so a host noun arriving
    # here would be one host's knob written into every host's launch signature.
    "run/launch.py",
    # The engine's storage, placed in contracts and importing nothing of the host. Moved off
    # _HOST_COUPLED_MODULES when the rubric-dim catalog's subject-tag filter left with the
    # template's kind spec: what a template states for its kind is the kind's
    # model's to validate, so a host noun arriving here would be one host's taxonomy written into
    # the store every host reads through.
    "contracts/storage.py",
    # The generic trial loop and the per-result covariate derivation, both importing nothing of the
    # host. Moved off _HOST_COUPLED_MODULES when a host tool's delivery record became
    # `AsyncDelivery` and its conclusion-path covariate was deleted: the runner stores a kind's
    # background-work record and payload without naming either, so a host noun arriving here would
    # be one host's tool read by every host's run loop.
    "run/runner.py",
    "contracts/covariates.py",
    # The per-role usage ledger. Moved off _HOST_COUPLED_MODULES when its rows stopped defaulting
    # their price source to the first host's provider: a completion client says
    # where its price came from and the ledger stores that, so a provider name arriving here would
    # be one host's pricing claimed for every host's dollars.
    "contracts/usage_capture.py",
    # A cell's world events and the session every world-bearing kind seeds, fires and reads back
    # through: dimension names and conditions are the host's words carried as data. A host noun here
    # would be one host's world written into every kind's cell.
    "contracts/world_events.py",
    "contracts/world_session.py",
    # The stored eval models. Host-coupled while ``EvalRun`` and ``EvalResult`` carried a host's
    # subject, overlays and async tool record; the last of those has left, and with it the per-type entry ``ContextComponents`` needed while the module around it was not
    # scanned. A host noun arriving here would be one host's shape stored by every host.
    "contracts/models.py",
)
#: Individual types inside a HOST-COUPLED module that are nevertheless shared contract, scanned as
#: if each were its own file. Module path (relative to ``threetears/evals/``) -> the class names in it.
#:
#: The escape hatch a per-module list structurally cannot provide. ``models.py`` is host-coupled
#: because ``EvalRun`` and ``EvalResult`` are, and ``ContextComponents`` is not: the identity
#: predicate — declared shared contract above — both produces and consumes it, and its FIELD NAMES
#: are the pre-image of every context key. One of them was ``universe`` for as long as the class
#: existed, and no scan could see it.
#:
#: **Narrow by intent.** A type earns an entry when the engine's own shared code reads or writes
#: it. Reclassifying the whole module is the other way to fix this, and it is not the same act:
#: that decision belongs in :data:`_HOST_COUPLED_MODULES` where a reviewer meets it.
#:
#: **Empty today:** ``models.py`` — the one module that needed it — is no longer host-coupled and
#: is declared shared contract whole. The mechanism stays for the next
#: host-coupled module with a shared type inside it.
_SHARED_CONTRACT_TYPES: dict[str, tuple[str, ...]] = {}

#: Whole trees under ``threetears/evals/`` that are shared contract. ``contracts/host`` is the registry
#: package: it ships the classification and the algebra, never the inputs, so a host noun
#: appearing there is the exact failure this canary exists for.
#:
#: ``contracts/prompts`` is the vocabulary the eval seed prompts are declared in (``seed.py``,
#: which is contracts: the declaration type is shared, the corpus is not; a host's list of its
#: seeds is its own). What is scanned in the prompt trees is the DECLARATION
#: — the section names, the formatter and data keys, the registry types, and the path each seed is
#: found at — because those are the vocabulary a second host inherits. The prompt BODIES are prose and escape by the token-shape rule, which is the
#: right split here and not an oversight: what the shipped text says is counted by
#: ``test_host_vocabulary_ceiling.py``, which reads prose. A host noun reaching a section name or a
#: declared key is a different failure and is the one this covers.
#:
#: ``gen/prompts`` is the two proposers' seed prompts, which belong to the gen package and are
#: scanned on exactly the terms above. It is a nested tree because its parent
#: ``gen`` is not classifiable whole yet: see :func:`test_every_eval_module_is_classified_one_way_or_the_other`.
#: ``contracts/host`` and ``contracts/prompts`` are nested for the same reason: ``contracts`` still
#: holds modules on :data:`_HOST_COUPLED_MODULES` (every ``contracts/`` entry there).
#:
#: ``storage`` and ``testing`` are whole trees from birth: the adapters the engine ships behind the
#: store port (the in-memory reference store) and the conformance kits any host runs against its own
#: adapter. Each exists to serve every host, so a host noun in either would be a leak. ``quick``, the
#: batteries (``run_eval`` and the command line), is whole from birth on the same terms.
#:
#: A host's adapter tree never belongs here. Its whole purpose is host coupling — it names its
#: host's concepts on purpose, and an extraction deletes it rather than porting it. Scanning it
#: would report the design as a violation.
_SHARED_CONTRACT_TREES: tuple[str, ...] = (
    "contracts/host",
    "analysis",
    "contracts/prompts",
    "gen/prompts",
    "storage",
    "testing",
    "quick",
)

_HOST_NOUN_RE = re.compile(rf"^(?:{'|'.join(HOST_NOUNS)})s?$", re.IGNORECASE)

#: A string constant is scanned as declared vocabulary only when it is token-shaped.
#: Anything longer, or carrying whitespace, is prose — a different item's sweep.
_MAX_TOKEN_LENGTH = 64


class Violation(NamedTuple):
    """One host noun reaching a name or a declared value in the shared contract."""

    module: str
    surface: str
    name: str
    lineno: int
    nouns: tuple[str, ...]

    @property
    def key(self) -> tuple[str, str, str]:
        """The allowlist key — deliberately carries no line number.

        A line-keyed register rots on every unrelated edit above it, and a ratchet that
        goes red for reasons unrelated to the debt is a ratchet somebody deletes. The
        cost is that a second occurrence of the same name on the same surface of the
        same module is covered by the existing entry: the same already-declared debt,
        in the place it was already declared.
        """
        return (self.module, self.surface, self.name)


class Debt(NamedTuple):
    """One allowlisted violation: why it is still here, and who owns retiring it."""

    reason: str
    issue: str


#: The debt register. **It may only shrink.** Adding an entry needs a linked issue whose
#: acceptance includes deleting the entry again; the canary fails on an entry that no
#: longer matches, so the deletion is enforced rather than remembered.
_ALLOWLIST: dict[tuple[str, str, str], Debt] = {}

#: The register's size, pinned so that growing it cannot happen in one line. Lower it
#: when a debt is paid; raising it is the deliberate act the norm asks you not to make.
#:
#: Widening the instrument — a new host noun, a newly scanned module — can surface names that were
#: already there, and registering them is a tightening, not a relaxation: the alternative is
#: leaving the scan narrow so the number stays low, which is the number governing the code instead
#: of the reverse.
#:
#: **Zero is the fragile number, not the comfortable one.** At 0 the next addition should be argued
#: for on its issue before it is written here. And a zero says only that nothing is exempt from the
#: rule this file enforces, not that the shared contract is free of host vocabulary: the canary
#: classifies declared names, params and literals, and the module docstring lists the coupling
#: classes it cannot see.
_ALLOWLIST_CEILING = 0

_ISSUE_RE = re.compile(r"^#\d+$")


def _host_nouns_in(text: str) -> tuple[str, ...]:
    """Every host noun appearing as a whole word in ``text``, deduplicated and sorted."""
    return tuple(sorted({word for word in identifier_words(text) if _HOST_NOUN_RE.match(word)}))


def _is_token_shaped(value: object) -> bool:
    """Whether a constant is declared vocabulary rather than prose."""
    return isinstance(value, str) and 0 < len(value) <= _MAX_TOKEN_LENGTH and not re.search(r"\s", value)


class _ContractScanner(ast.NodeVisitor):
    """Collects host nouns reaching a name or a declared value in one module."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.violations: list[Violation] = []

    def _record(self, surface: str, lineno: int, name: str) -> None:
        nouns = _host_nouns_in(name)
        if nouns:
            self.violations.append(Violation(self.module, surface, name, lineno, nouns))

    def _record_declared_values(self, node: ast.AST, surface: str) -> None:
        """Every token-shaped string anywhere inside a declared value expression.

        Walks the whole expression rather than only its top level, because a
        declaration table's vocabulary sits inside calls, tuples and dicts —
        ``VersionedInput(name="subject_id", ...)`` is the shape this exists to catch.
        """
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and _is_token_shaped(child.value):
                self._record(surface, child.lineno, child.value)

    def _record_signature(self, args: ast.arguments) -> None:
        """Record a callable's parameters, their annotations' Literal members, and their defaults.

        Shared by ``def``, ``async def`` and ``lambda`` so a signature cannot escape the
        scan by the form it was written in — a lambda in a declaration table is still
        contract surface.
        """
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg is None:
                continue
            self._record("param", arg.lineno, arg.arg)
            if arg.annotation is not None:
                self._record_literal_members(arg.annotation)
        for default in (*args.defaults, *(d for d in args.kw_defaults if d is not None)):
            self._record_declared_values(default, "param-default")

    def _record_literal_members(self, annotation: ast.AST) -> None:
        for child in ast.walk(annotation):
            if not isinstance(child, ast.Subscript):
                continue
            target = child.value
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
            if name != "Literal":
                continue
            for member in ast.walk(child.slice):
                if isinstance(member, ast.Constant) and _is_token_shaped(member.value):
                    self._record("literal-member", member.lineno, member.value)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record the class name, then descend."""
        self._record("class", node.lineno, node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record the function name, its signature and its return Literal members, then descend."""
        self._record("function", node.lineno, node.name)
        self._record_signature(node.args)
        if node.returns is not None:
            self._record_literal_members(node.returns)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Async functions carry the same surfaces as sync ones."""
        self.visit_FunctionDef(node)  # type: ignore[arg-type]

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """A lambda has no name, but its parameters and defaults are still surface."""
        self._record_signature(node.args)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """An annotated assignment: a pydantic field, its default, and its Literal members."""
        if isinstance(node.target, ast.Name):
            self._record("name", node.lineno, node.target.id)
        self._record_literal_members(node.annotation)
        if node.value is not None:
            self._record_declared_values(node.value, "declared-value")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """A plain assignment: a constant, an enum member, or a declaration table."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._record("name", node.lineno, target.id)
        self._record_declared_values(node.value, "declared-value")
        self.generic_visit(node)


def _shared_contract_files() -> list[Path]:
    """Every file in the declared shared-contract set, module files first."""
    files = [EVAL_ROOT / module for module in _SHARED_CONTRACT_MODULES]
    for tree in _SHARED_CONTRACT_TREES:
        files.extend(sorted((EVAL_ROOT / tree).rglob("*.py")))
    return files


def _scan_source(module: str, source: str) -> list[Violation]:
    """Scan one module's source text. Split out so the self-tests need no files."""
    scanner = _ContractScanner(module)
    scanner.visit(ast.parse(source, filename=module))
    # Deduplicated on the full tuple: one violation can be reached twice when a
    # declared value nests inside another declared value.
    return sorted(set(scanner.violations))


def _class_def(source: str, module: str, name: str) -> ast.ClassDef | None:
    """The top-level ``class name`` node in ``source``, or ``None`` when it does not exist.

    Top-level only: a declared type is a module's public shape, and reaching a nested class would
    let an entry silently cover something nobody named.

    Args:
        source: The module's source text.
        module: The module path, for the parser's error messages.
        name: The class to find.

    Returns:
        The node, or ``None``.
    """
    return next(
        (
            node
            for node in ast.parse(source, filename=module).body
            if isinstance(node, ast.ClassDef) and node.name == name
        ),
        None,
    )


def _scan_type(module: str, source: str, name: str) -> list[Violation]:
    """Scan one class body inside an otherwise-unscanned module.

    Args:
        module: The module path, relative to ``threetears/evals/``.
        source: The module's source text.
        name: The class to scan.

    Returns:
        The violations, labelled ``module::ClassName`` so a register entry names the type rather
        than the file it happens to live in.
    """
    node = _class_def(source, module, name)
    if node is None:
        return []
    scanner = _ContractScanner(f"{module}::{name}")
    scanner.visit(node)
    return sorted(set(scanner.violations))


def _scan_shared_contract() -> list[Violation]:
    """Every violation across the declared set — whole modules, whole trees, and declared types."""
    found: list[Violation] = []
    for path in _shared_contract_files():
        module = path.relative_to(EVAL_ROOT).as_posix()
        found.extend(_scan_source(module, path.read_text()))
    for module, names in _SHARED_CONTRACT_TYPES.items():
        source = (EVAL_ROOT / module).read_text()
        for name in names:
            found.extend(_scan_type(module, source, name))
    return found


#: Top-level ``threetears/evals/*.py`` modules that are deliberately NOT shared contract — the scan
#: below does not read them. A module belongs here when it is a host's own machinery, which an
#: extraction ports or deletes rather than lifting.
#:
#: This list exists only so the completeness check below can tell "classified as host-coupled"
#: from "nobody has classified this yet". It grants no exemption from anything else.
#:
#: **A module on this list is always one paid debt away from belonging on the other.** Every
#: decoupling retires a module's IMPORTS or its VOCABULARY, and the aggregation core is the worked
#: example of what happens when nobody notices: ``analysis/reporting.py`` sat here while it became
#: the module the extraction boundary is defined around, so the scan skipped the one module the
#: boundary is named after. :func:`test_every_host_coupled_module_still_exhibits_host_coupling` is
#: the gate that catches that now.
_HOST_COUPLED_MODULES: frozenset[str] = frozenset(
    {
        "__init__.py",
    }
)


def _host_module_imports(source: str, path: Path) -> set[str]:
    """The ``discodon`` modules outside ``threetears.evals`` that ``source`` names.

    A local walk for a local question, and the distinction is worth stating because
    ``test_extraction_import_boundary.py`` walks the same edges. That canary ENFORCES the rule
    and owns it, with the registers and ceilings that go with enforcement; this one is asking
    whether a *classification* in this file is still true, and needs only to know whether any
    such import exists. Neither reads the other's answer, which is the same deliberate
    independence the two files already keep — see that module's docstring on why no code is
    shared.

    Relative imports are resolved rather than skipped: ``from ..config import x`` in an eval
    module is a host reach spelled differently, and a walk that skipped it would report a module
    as decoupled on the strength of an import form. The arithmetic is
    :func:`~packages.evals.tests.import_resolution.absolute_module`, shared with the enforcing canary
    — not because the two files share a subject, which they deliberately do not, but because
    what a dotted name resolves to is Python's decision rather than either canary's. The copy
    this replaced sliced the package by a computed index, so a level that walked above the
    repository root produced a *truncated* package name instead of nothing.

    Args:
        source: The module's source text.
        path: Its path, used to anchor a relative import against its own package.

    Returns:
        The absolute ``discodon.*`` module names it reaches outside ``threetears.evals``.
    """
    root = EVAL_ROOT.parent.parent
    found: set[str] = set()

    def crossing(name: str) -> bool:
        return name == "discodon" or (name.startswith("discodon.") and not name.startswith("threetears.evals"))

    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if crossing(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = absolute_module(path, node, root=root)
            if crossing(module):
                found.add(module)
    return found


#: The modules on :data:`_HOST_COUPLED_MODULES` that exhibit no host coupling TODAY, each with
#: the reason it is classified by role rather than by evidence.
#:
#: **The residue of an anti-rot check, and the reason that check has one.** A module lands on the
#: host-coupled list because it names a host's concepts or imports a host's packages, and
#: `reporting.py` is the worked example of what happens when that stops being true and nobody
#: notices: it sat here while it *became* the aggregation core the extraction boundary is defined
#: around, so the scan skipped the one module the boundary is named after. What moved it was a
#: person looking, not a gate.
#:
#: :func:`test_every_host_coupled_module_still_exhibits_host_coupling` is that gate, and this is
#: what it admits: a module whose host-coupling is a fact about its ROLE that its source does not
#: yet spell. That is a real category and not a loophole — a stub holds no code to name anything,
#: and a driver of host-shaped work can name none of it directly — but it is the category a paid
#: debt decays into, so each entry says which it is and an entry with no reason fails.
_HOST_COUPLED_BY_ROLE: dict[str, str] = {
    "__init__.py": (
        "the package's front door. It exports the eval surface and describes it, so what "
        "makes it host-coupled is the inventory it publishes rather than any name in it"
    ),
}


def test_every_host_coupled_module_still_exhibits_host_coupling() -> None:
    """A module classified host-coupled must still BE host-coupled, or be declared by role.

    The direction the completeness check cannot see. That one asks whether every module is
    classified; this asks whether a classification is still TRUE — and the failure it catches is
    silent by construction, because a module that sheds its host coupling and keeps its listing
    simply goes on being skipped. ``reporting.py`` did exactly that while it became the module
    the extraction boundary is defined around, and a person moved it, not a gate.

    That is now the standing case rather than the exceptional one: every requirement this
    extraction has left pays off a coupling, so each one turns some module on that list into
    shared contract. Without this check, meeting a requirement quietly REMOVES scanning from the
    module the requirement just made portable.

    Exhibiting host coupling means either naming a ``discodon`` module outside ``threetears.evals``
    or carrying a host noun where the scanner looks. A module that does neither is either shared
    contract now — move it, which only ever adds scanning — or host-coupled by a role its source
    does not spell, which is :data:`_HOST_COUPLED_BY_ROLE` and needs a reason.
    """
    undeclared: list[str] = []
    for module in sorted(_HOST_COUPLED_MODULES):
        path = EVAL_ROOT / module
        source = path.read_text()
        if _host_module_imports(source, path) or _scan_source(module, source):
            continue
        if module not in _HOST_COUPLED_BY_ROLE:
            undeclared.append(module)

    assert not undeclared, (
        "These modules are classified host-coupled but exhibit no host coupling — they name no "
        "`discodon` module outside `threetears.evals` and carry no host noun. Each has either become "
        "shared contract (move it to _SHARED_CONTRACT_MODULES; that only adds scanning) or is "
        "host-coupled by a role its source does not spell (declare it in _HOST_COUPLED_BY_ROLE "
        "with the reason). Leaving it here exempts it from the scan on a rationale that has "
        "expired:\n  " + "\n  ".join(undeclared)
    )


def test_every_by_role_entry_is_still_needed_and_carries_a_reason() -> None:
    """The by-role register may not name a module that now shows its coupling, or carry a blank.

    Anti-rot on the residue itself. An entry that outlives its need is a module carrying a
    written explanation for an exemption it no longer has, which reads as considered and is not;
    a blank reason is the bare pass list every register in this file exists to refuse.
    """
    stale = [
        module
        for module in sorted(_HOST_COUPLED_BY_ROLE)
        if module in _HOST_COUPLED_MODULES
        and (
            _host_module_imports((EVAL_ROOT / module).read_text(), EVAL_ROOT / module)
            or _scan_source(module, (EVAL_ROOT / module).read_text())
        )
    ]
    assert not stale, (
        "These are declared host-coupled BY ROLE but now exhibit host coupling directly, so the "
        "declaration is answering a question nobody is asking. Remove the entry:\n  " + "\n  ".join(stale)
    )

    unlisted = sorted(set(_HOST_COUPLED_BY_ROLE) - _HOST_COUPLED_MODULES)
    assert not unlisted, (
        "These are declared host-coupled by role but are not on _HOST_COUPLED_MODULES at all:\n  "
        + "\n  ".join(unlisted)
    )

    blank = [module for module, reason in sorted(_HOST_COUPLED_BY_ROLE.items()) if not reason.strip()]
    assert not blank, "These by-role entries carry no reason, which makes them a pass list:\n  " + "\n  ".join(blank)


#: Subpackages under ``threetears/evals/`` that are deliberately NOT shared contract, the tree-level
#: sibling of :data:`_HOST_COUPLED_MODULES`. EMPTY in the package: the first host's adapter tree,
#: which named its concepts by design, was left behind by the cut, so a tree that arrives here is
#: classified like any other rather than inheriting that exemption by name.
_HOST_COUPLED_TREES: frozenset[str] = frozenset()


def test_every_eval_module_is_classified_one_way_or_the_other() -> None:
    """A new module must be declared shared or declared host-coupled — never merely unlisted.

    The direction the declared-set check above cannot see. That one asks whether everything
    DECLARED still exists, which catches a rename; this asks whether everything that EXISTS has
    been classified, which catches the case that actually happens — somebody adds a module and
    nobody decides what it is. An unlisted module is scanned by nothing, so a host noun in it is
    invisible to this canary however load-bearing the module turns out to be.

    ``hashing.py`` is why this exists. It is the shared pre-image of every eval key, it sat in
    the dependency closure of `identity.py` and the whole `analysis` tree, and it was in neither
    list — so the scan that exists to protect the shared contract could not see the one module
    every key passes through.

    **A tree is classified whole, or every module in it is classified one by one.** A package
    move can put modules of both classes in one directory before the debt that separates them is
    paid, so a tree in neither tree list is a SPLIT tree: each module under it (outside a nested tree that IS listed,
    such as ``gen/prompts``) joins the module population below and must be classified like a
    top-level module. That keeps the property this test exists for — every module is classified —
    and refuses a split tree that holds no module at all, since there would be nothing to
    classify and the tree would be scanned by nothing.
    """
    listed_trees = set(_SHARED_CONTRACT_TREES) | _HOST_COUPLED_TREES

    def in_listed_tree(relative: str) -> bool:
        return any(relative.startswith(tree + "/") for tree in listed_trees)

    # Subpackages, which the top-level glob cannot see at all. A whole DIRECTORY of modules can
    # be neither shared-contract nor host-coupled and the module check below stays green — the
    # exact hole the module version was written to close, one level up. A tree is classified by
    # being in _SHARED_CONTRACT_TREES or _HOST_COUPLED_TREES, or as a split tree whose modules are
    # each classified; nothing else is.
    trees_present = {path.name for path in tree_subdirectories(EVAL_ROOT)}
    split_trees = sorted(trees_present - listed_trees)
    empty_split = [tree for tree in split_trees if not list((EVAL_ROOT / tree).rglob("*.py"))]
    assert not empty_split, (
        f"unclassified eval subpackages: {empty_split}. Add each to _SHARED_CONTRACT_TREES (scanned "
        "for host nouns) or to _HOST_COUPLED_TREES (a host's own machinery). A tree in neither is scanned "
        "by nothing, however many modules it grows."
    )

    present = {path.name for path in EVAL_ROOT.glob("*.py")}
    for tree in split_trees:
        for path in (EVAL_ROOT / tree).rglob("*.py"):
            relative = path.relative_to(EVAL_ROOT).as_posix()
            if not in_listed_tree(relative):
                present.add(relative)
    classified = set(_SHARED_CONTRACT_MODULES) | _HOST_COUPLED_MODULES
    unclassified = sorted(present - classified)

    assert not unclassified, (
        f"unclassified eval modules: {unclassified}. Add each to _SHARED_CONTRACT_MODULES (the engine's "
        "own vocabulary, scanned for host nouns) or to _HOST_COUPLED_MODULES (a host's machinery, which "
        "an extraction ports or deletes) — or, for a module in a subpackage, classify its whole tree in "
        "_SHARED_CONTRACT_TREES or _HOST_COUPLED_TREES. Leaving one unlisted exempts it from the canary silently."
    )
    assert not (stale := sorted(_HOST_COUPLED_MODULES - present)), (
        f"host-coupled modules no longer exist: {stale} — the list must shrink when they do, or it starts "
        "granting cover to names nothing occupies"
    )


def test_a_cache_only_directory_is_not_an_eval_subpackage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A directory holding only ``__pycache__`` is what a move leaves behind, not a subpackage.

    ``git pull`` over a module move deletes the tracked files and keeps the untracked bytecode
    cache, so every checkout that had imported the old tree carries a directory the repository no
    longer has. The canary runs against a copy of the real tree with one such directory added, so
    the test drives the classification the canary actually performs rather than a helper beside it.

    Both directions on one fixture: the cache-only directory is ignored, and the same directory
    holding one non-cache file is still refused as an unclassified tree — the refusal ``seed`` is
    listed to satisfy, which the cache rule must not lose.
    """
    root = tmp_path / "eval"
    shutil.copytree(EVAL_ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
    cache = root / "moved_away" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "module.cpython-314.pyc").write_bytes(b"")
    monkeypatch.setattr(sys.modules[__name__], "EVAL_ROOT", root)

    test_every_eval_module_is_classified_one_way_or_the_other()

    (root / "moved_away" / "corpus.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match=r"unclassified eval subpackages: \['moved_away'\]"):
        test_every_eval_module_is_classified_one_way_or_the_other()


def test_the_declared_module_set_still_exists() -> None:
    """Every declared module resolves, and the viz tree is non-empty.

    Without this, a rename or a deletion silently narrows what the canary covers and
    the green stays green — the failure mode of every scan that derives its own input.
    """
    missing = [module for module in _SHARED_CONTRACT_MODULES if not (EVAL_ROOT / module).is_file()]
    assert not missing, f"declared shared-contract modules no longer exist: {missing}"
    for tree in _SHARED_CONTRACT_TREES:
        root = EVAL_ROOT / tree
        assert root.is_dir(), f"declared shared-contract tree no longer exists: {tree}"
        assert list(root.rglob("*.py")), f"declared shared-contract tree is empty: {tree}"


def test_every_declared_shared_contract_type_is_a_live_type_in_a_host_coupled_module() -> None:
    """A per-type entry must resolve, and must be doing work a per-module entry could not.

    Two directions, and both are the anti-rot half of the mechanism the module docstring
    describes. A name that no longer resolves scans nothing while reading as coverage — the same
    failure :func:`test_the_declared_module_set_still_exists` catches one level up. And an entry
    naming a module that is already shared contract is redundant rather than harmless: it
    double-scans the same class, and it hides that somebody reclassified the module without
    deleting the entry that existed only because it had not been.
    """
    for module, names in _SHARED_CONTRACT_TYPES.items():
        assert module in _HOST_COUPLED_MODULES, (
            f"{module} declares shared-contract types but is not classified host-coupled. Either the whole "
            "module is shared contract — declare it in _SHARED_CONTRACT_MODULES and drop this entry — or it "
            "is unclassified, which is the state _HOST_COUPLED_MODULES exists to refuse."
        )
        path = EVAL_ROOT / module
        assert path.is_file(), f"{module} declares shared-contract types but no longer exists"
        source = path.read_text()
        missing = [name for name in names if _class_def(source, module, name) is None]
        assert not missing, (
            f"{module} declares shared-contract types that are not top-level classes in it: {missing}. "
            "A name that resolves to nothing is scanned by nothing while reading as coverage."
        )


def test_a_declared_type_is_scanned_even_though_its_module_is_not() -> None:
    """Pins the detector for the per-type surface, so the widening cannot go vacuous.

    The whole point of :data:`_SHARED_CONTRACT_TYPES` is that the surrounding module is NOT
    scanned, so nothing else in this file would notice if the type scan silently stopped running.
    Written as source text, against the same entry points the real scan uses.
    """
    source = "class Kept:\n    universe_id: str = ''\n\nclass Ignored:\n    persona_id: str = ''\n"
    kept = _scan_type("probe.py", source, "Kept")
    assert {(v.module, v.surface, v.name) for v in kept} == {("probe.py::Kept", "name", "universe_id")}
    assert _scan_type("probe.py", source, "Ignored") != [], "the sibling class is scannable, just not declared"
    assert _scan_type("probe.py", source, "Absent") == [], "a name that is not a class in the module scans nothing"


def test_no_host_name_reaches_the_shared_contract() -> None:
    """No host noun names anything in the shared contract, outside the debt register.

    Fix the name. If it genuinely cannot be fixed yet, file the issue that will fix it
    and add an ``_ALLOWLIST`` entry citing it — the register may only shrink, so an
    addition is a deliberate act with an owner attached.
    """
    unlisted = [v for v in _scan_shared_contract() if v.key not in _ALLOWLIST]
    if unlisted:
        detail = "\n  ".join(
            f"{v.module}:{v.lineno}: {v.surface} '{v.name}' names {', '.join(v.nouns)}" for v in unlisted
        )
        pytest.fail(
            "Host nouns reached the shared eval contract, which must stay host-agnostic.\n"
            "  Rename it, or file an issue and add an _ALLOWLIST entry citing it.\n  " + detail
        )


def test_every_allowlist_entry_still_matches_a_live_violation() -> None:
    """The register cannot rot, and a paid debt must delete its own entry.

    This is the half that makes "may only shrink" real: fixing a name turns this test
    red until the entry goes too, so the register tracks the debt exactly rather than
    accumulating permissions nobody can audit.
    """
    live = {v.key for v in _scan_shared_contract()}
    stale = sorted(key for key in _ALLOWLIST if key not in live)
    if stale:
        detail = "\n  ".join(f"{module} | {surface} | {name}" for module, surface, name in stale)
        pytest.fail(
            "These _ALLOWLIST entries match nothing in the shared contract any more.\n"
            "  If the debt was paid, delete the entry (and lower _ALLOWLIST_CEILING) in the same commit.\n  " + detail
        )


def test_every_allowlist_entry_carries_a_reason_and_an_issue() -> None:
    """No bare pass list: each entry says why it stands and who retires it."""
    bad: list[str] = []
    for key, debt in sorted(_ALLOWLIST.items()):
        label = " | ".join(key)
        if not debt.reason.strip():
            bad.append(f"{label}: empty reason")
        if not _ISSUE_RE.match(debt.issue.strip()):
            bad.append(f"{label}: issue reference {debt.issue!r} is not of the form '#1234'")
    assert not bad, "Allowlist entries missing a reason or an issue reference:\n  " + "\n  ".join(bad)


def test_the_allowlist_has_not_grown_past_its_ceiling() -> None:
    """The register's size is pinned, so growth cannot be a one-line edit.

    A speed bump rather than a lock, and said plainly: nothing here stops someone
    raising the ceiling. What it stops is doing so **silently** — the ceiling and the
    entry land in one diff, where a reviewer sees both.
    """
    assert len(_ALLOWLIST) == _ALLOWLIST_CEILING, (
        f"_ALLOWLIST holds {len(_ALLOWLIST)} entries but _ALLOWLIST_CEILING is {_ALLOWLIST_CEILING}. "
        "Paying a debt lowers the ceiling; raising it is the deliberate act the norm asks you not to make."
    )


def test_the_scanner_flags_a_synthetic_violation_on_every_surface() -> None:
    """Pins the detector, so an AST regression cannot make the canary vacuous.

    One synthetic module carrying a host noun on each scanned surface; every surface
    must be reported. Written as source text rather than a fixture file so the test
    exercises the same entry point the real scan does.
    """
    source = (
        "from typing import Literal\n"
        "\n"
        "MUSIC_ROOT = 'x'\n"
        "\n"
        "class PersonaThing:\n"
        "    kind: Literal['persona', 'widget'] = 'widget'\n"
        "    subject_kind: str = Field(default='persona')\n"
        "    discord_id: str = ''\n"
        "\n"
        "def build_dj(universe_id: str, mode: str = 'segue') -> None:\n"
        "    pass\n"
        "\n"
        "async def fetch(scope: Literal['universe', 'global']) -> Literal['tavily', 'none']:\n"
        "    pass\n"
        "\n"
        "TABLE = (Input(name='segue_score', read=lambda discord_run: discord_run),)\n"
    )
    by_surface = {(v.surface, v.name) for v in _scan_source("synthetic.py", source)}
    for expected in (
        ("name", "MUSIC_ROOT"),
        ("class", "PersonaThing"),
        ("literal-member", "persona"),
        ("declared-value", "persona"),
        ("name", "discord_id"),
        ("function", "build_dj"),
        ("param", "universe_id"),
        ("param-default", "segue"),
        ("declared-value", "segue_score"),
        # An async signature, a Literal in a parameter annotation, one in a return
        # annotation, and a lambda parameter — each a way a name could have slipped
        # past a scanner that only understood `def` and pydantic fields.
        ("literal-member", "universe"),
        ("literal-member", "tavily"),
        ("param", "discord_run"),
    ):
        assert expected in by_surface, f"scanner missed {expected}; found {sorted(by_surface)}"


def test_the_scanner_stays_quiet_on_clean_code_and_on_prose() -> None:
    """No false positives — the half that decides whether anyone keeps this canary on.

    Three traps in one module: a host noun as a substring of an ordinary English word
    (``dj`` inside ``adjust``), a host noun in prose the scan deliberately does not read,
    and a host noun in a docstring, which is a different item's sweep.
    """
    source = (
        '"""A module docstring mentioning a persona and a DJ, which are prose."""\n'
        "\n"
        "def adjust_adjacent(adjective: str) -> str:\n"
        '    """Adjust an adjacent adjective."""\n'
        "    return adjective\n"
        "\n"
        "FIELD = Field(default='', description='Reference to a persona id, today.')\n"
        "PROSE = 'Never assume the subject is a persona; read what the bundle says.'\n"
    )
    assert _scan_source("synthetic.py", source) == []


class TestWhichStringsTheScannerCanSee:
    """The scanned/unscanned boundary, enumerated — the module docstring's bullet, as a test.

    That bullet has been wrong five times running, each version claiming a wider blind spot than
    exists — which makes the register look like it covers less than it does and invites the next
    reader to skip a check that would have fired. The fourth version was already used as the
    recorded justification for a design decision on another module, and the fifth contradicted
    ``test_the_scanner_flags_a_synthetic_violation_on_every_surface`` sixty lines above it.

    Prose cannot be kept true by care here, because the answer is a property of
    ``ast.NodeVisitor`` dispatch. There are **three** producers of a scanned string value, not one:
    ``_record_declared_values`` (the right-hand side of an ``=`` or annotated ``=``, and every
    parameter default), and ``_record_literal_members`` (any ``Literal[...]`` member of a
    parameter annotation, a variable annotation or a return type).

    **Two assertions, and they close different gaps** — the second exists because the first
    version of this class had only the first, and a review pointed out that a hand-written set
    locked the cases to a list someone had to REMEMBER to update, which is the property the class
    claimed to have replaced. :meth:`test_the_scanned_cases_cover_every_value_carrying_surface`
    holds the enumerated cases in step with the two frozensets;
    :meth:`test_the_declared_surfaces_are_the_ones_the_scanner_emits` holds the frozensets in step
    with :class:`_ContractScanner` itself, by reading the labels out of its source. A fourth
    producer fails the second one without anybody writing a case for it — **provided it names its
    label at the call site**, which is the shape every producer has today and which the second
    test also pins.
    """

    #: Every ``surface`` label the scanner emits for a VALUE.
    _VALUE_SURFACES = frozenset({"declared-value", "param-default", "literal-member"})

    #: The labels for a NAME rather than a value. Subtracted below rather than ignored, because
    #: a scanned value usually arrives beside one — ``PERSONA_ID = "persona_id"`` reports both the
    #: target's ``name`` and the right-hand side's ``declared-value``.
    _NAME_SURFACES = frozenset({"name", "class", "function", "param"})

    @pytest.mark.parametrize(
        ("shape", "source"),
        [
            ("a module-level assignment", 'PERSONA_ID = "persona_id"'),
            ("an assignment inside a function body", 'def f():\n    x = {"persona_id": 1}'),
            ("an annotated assignment inside a function body", 'def f():\n    x: dict = {"persona_id": 1}'),
            ("the constant part of an f-string", 'def f():\n    x = f"{a}persona_id"'),
            # The two the fifth version of the bullet denied. Neither is the right-hand side of
            # anything, and both are scanned.
            ("a parameter default", 'def f(mode: str = "segue"):\n    pass'),
            ("a keyword-only parameter default", 'def f(*, mode: str = "segue"):\n    pass'),
            (
                "a Literal member in a return annotation",
                'from typing import Literal\ndef f() -> Literal["tavily"]:\n    pass',
            ),
            (
                "a Literal member in a parameter annotation",
                'from typing import Literal\ndef f(x: Literal["tavily"]):\n    pass',
            ),
            (
                "a Literal member nested in an annotation",
                'from typing import Literal\ndef f(x: dict[str, Literal["tavily"]]):\n    pass',
            ),
        ],
    )
    def test_a_value_on_a_carrying_surface_is_scanned(self, shape: str, source: str) -> None:
        assert _scan_source("probe.py", source), f"expected {shape} to be scanned"

    def test_the_scanned_cases_cover_every_value_carrying_surface(self) -> None:
        """Every declared value surface has a case above, and no case reaches an undeclared one.

        Half the guard. It says the enumeration and the classification agree; it says nothing
        about whether the classification matches the scanner, which is the other test's job.
        """
        marks = [m for m in self.test_a_value_on_a_carrying_surface_is_scanned.pytestmark if m.name == "parametrize"]
        assert len(marks) == 1, "the case table is read off this mark, so a second parametrize would silently pick one"
        cases = marks[0].args[1]
        covered = {v.surface for _, source in cases for v in _scan_source("probe.py", source)}
        assert covered - self._NAME_SURFACES == self._VALUE_SURFACES

    def test_the_declared_surfaces_are_the_ones_the_scanner_emits(self) -> None:
        """The classification above is derived from :class:`_ContractScanner`, not recalled.

        The other half, and the one the first version of this class lacked. Without it a fourth
        producer emitting a NEW label passed silently: no case would exercise it, so the covered
        set would not move, so the assertion above would still hold — a guard whose advertised
        authority exceeded its real one, in the file five review rounds had been spent
        establishing as the authority on exactly this question.

        Reads the labels out of the scanner's own source rather than from a list beside it, the
        same anti-rot shape the portable-factories assertion uses. Two call forms carry a label:
        ``self._record(<surface>, …)`` names it first, and ``self._record_declared_values(node,
        <surface>)`` names it second and passes it through. ``_record_literal_members`` hardcodes
        its own and so is covered by the first form.

        **``positions`` is the one thing here still written by hand**, so a THIRD pass-through
        helper — a new method taking ``surface`` as a parameter — would carry labels this walk
        never looks at. That is closed rather than disclaimed: the second assertion below refuses
        any method of this class, sync or async, declaring a ``surface`` parameter in any position
        that this map does not name — the same
        ``posonlyargs``/``args``/``kwonlyargs`` sweep :meth:`_record_signature` already does one
        level down, since a guard that understood fewer parameter forms than the scanner it
        guards would be the file disagreeing with itself.

        Being exact about the residue, since being approximately right about this guard is what
        six review rounds were spent on: a producer that hardcoded a label inside a helper
        reached only from another helper would still escape, and nothing here claims otherwise.
        """
        scanner = next(
            node
            for node in ast.walk(ast.parse(Path(__file__).read_text(encoding="utf-8")))
            if isinstance(node, ast.ClassDef) and node.name == _ContractScanner.__name__
        )
        positions = {"_record": 0, "_record_declared_values": 1}
        emitted = {
            call.args[index].value
            for call in ast.walk(scanner)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and (index := positions.get(call.func.attr, -1)) >= 0
            and len(call.args) > index
            and isinstance(call.args[index], ast.Constant)
            and isinstance(call.args[index].value, str)
        }
        assert emitted, "found no surface labels — the call forms this reads must have changed"
        assert emitted == self._VALUE_SURFACES | self._NAME_SURFACES

        pass_through = {
            fn.name
            for fn in ast.walk(scanner)
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
            and any(a.arg == "surface" for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs))
        }
        assert pass_through <= set(positions), (
            f"{sorted(pass_through - set(positions))} takes a `surface` parameter, so its call sites name labels "
            "this walk does not read — add it to `positions` with the index of that argument"
        )

    @pytest.mark.parametrize(
        ("shape", "source"),
        [
            ("a returned literal", 'def f():\n    return {"persona_id": 1}'),
            ("a call argument", 'def f():\n    helper({"persona_id": 1})'),
            ("a subscript target", 'def f():\n    d["persona_id"] = 1'),
            ("a for-loop iterable", 'def f():\n    for x in ["persona_id"]:\n        pass'),
            ("a comparison operand", 'def f():\n    if a == "persona_id":\n        pass'),
            # These two BIND a name, which is why "literals never bound to a name" — the third
            # version of the docstring bullet — was false. They are unscanned all the same,
            # because neither is an `Assign` or an `AnnAssign`.
            ("an augmented assignment", 'def f():\n    x = []\n    x += ["persona_id"]'),
            ("a walrus", 'def f():\n    if (x := "persona_id"):\n        pass'),
        ],
    )
    def test_everything_else_escapes(self, shape: str, source: str) -> None:
        assert not _scan_source("probe.py", source), f"expected {shape} to be unscanned"
