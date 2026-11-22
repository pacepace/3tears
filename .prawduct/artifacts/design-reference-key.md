# Design: a case's reference key (#1285) and the set grader (#1286)

**Coordinator rulings, 2026-11-19 (these override the text below where they differ):**

- **Order.** A key's parts are sets. The pinned file sorts core, acceptable, aliases and facts when a key is built, so the digest never depends on the order they were written in.
- **Paths.** Engine path only. The quick path is a follow-up issue.
- **Open questions 2, 3, 5 and 6:** as recommended.
- **Open question 4:** the caller-ledger lines name `discodon/evals/kinds/persona/reference.py` (Discodon, 2026-11-19). It is the one host module that calls `grade_set`, `set_grade_descriptors` and `EntityExtractor`, and it holds Discodon's `EntityMatcher`. Discodon confirmed the schema as drafted, and that the decision eval is an engine campaign (`campaign_create`, then `sweep_launch`, then `runs_compare`).

Base: `evals/main` a1cf245e9. Paths are under `packages/evals/`. The work needs no version bump.

## 0. Current code

| What | Where |
|---|---|
| `EvalTestCase`: `variation_params`, `host_payload`, `stratum` (no identity), `content_hash` | `src/threetears/evals/schema/models.py:2554`, `:2589`, `:2591`, `:2605`, `:2636` |
| A generated case's hash is `variation_params` only | `kernel/identity.py:119` |
| An authored case's identity: `TEMPLATE_CASE_FIELDS` → `_case_digest` (an empty field is dropped) → `template_case_content_digest` → id `uuid5(template.id\|digest)` | `run/definition_digests.py:131,138,157`; `run/definition_seed.py:255,296` |
| Drift compares stored and corpus digests through the same fields | `run/definition_drift.py:483,547` |
| `compare_two_runs` pairs only shared case ids, so a new case id never pairs | `analysis/reads.py:2159` |
| The engine judge reads `JudgeEvidence.case_material`. The kind renders it and the engine reads none of it | `schema/models.py:5778,5804` |
| On the quick path, `Judge(case_material=)` gets the case, else the case as JSON | `quick/judged.py:544-566` |
| A quick case's address digests every case dict | `quick/one_call.py:1408,2380` |
| A function grader is a check unless it declares a number. A raise excludes the whole cell | `quick/graders.py:272`; `quick/one_call.py:983-995` |
| A kind's code measures are `CandidateOutput.host_measures`. A cell without a measure is outside that measure's n | `kernel/candidate_kind.py:339`; `analysis/summary.py:1648` |
| `MeasurePopulation` covers faults only. `formula` is a definition field, so its digest enters the context key (#1160) | `kernel/metrics.py:259,477,692` |
| Judge identity is `builtin:<sha12>` of the wording | `run/judge_service.py:480`; `run/judge.py:388` |
| Newer-minor tolerance needs `CoreShapeModel` on every embedded shape. Durable minor is 17.6 | `schema/base.py:341`; `schema/versioning.py:981` |

**Paths.** This batch covers the **engine path**: host kinds, seeded template cases and host-written cases. Discodon is an engine host. The quick path is left out (see Q1). It cannot run an async model step inside a grader, and it cannot leave out a single measure on a cell without excluding the whole cell.

## 1. Design

### 1.1 The key: schema text to send to the host as-is

A case may carry `reference`, beside `variation_params` and `host_payload`, in a `template_cases/*.json` corpus file or on a stored case:

```json
"reference": {
  "type": "entity_set",
  "core":       [{"name": "Band A", "aliases": ["The Band A"],
                  "facts": [{"claim": "Formed in 1975.", "source_url": "https://example.org/a"}]}],
  "acceptable": [{"name": "Band B"}]
}
```

- `type`: optional. It is always `"entity_set"` today.
- `core`: the required entities.
- `acceptable`: the other entities that also count. It may be non-exhaustive.
- `core` and `acceptable` are **disjoint**. The full key is `core` + `acceptable`. Each entity is written once, with its aliases and facts, so no id cross-references are needed and core ⊆ full holds by construction.
- Each entity has `name` (required), `aliases` (default `[]`) and `facts` (default `[]`). Each fact has `claim` and `source_url`, both required. Facts are read by #1287 and rendered for judges now.
- The key is refused if it has no entity at all.
- The key is refused if two entities share a name or alias after folding (case, Unicode NFKC, whitespace runs).
- Unknown keys are refused.
- The key is identity: editing any content of it makes a new case revision. Order is not content: the key is kept in one canonical order (entities, aliases and facts sorted), so writing it in another order is the same key and the same case.

### 1.2 Types (Chunk A; pinned, see §2)

`schema/reference.py` holds `fold_name`, `ReferenceFact`, `ReferenceEntity` and `ReferenceKey`. These are exported from `threetears.evals.schema`, all provisional. `EvalTestCase.reference: ReferenceKey | None = Field(default=None, exclude_if=lambda v: v is None)`, so a case without a key is stored byte for byte as before.

### 1.3 Identity rule

- `TEMPLATE_CASE_FIELDS` gains `"reference"`, so it joins both the drift digest and the content digest.
- `_case_digest` dumps `reference` with `mode="json"` and still drops `None`. Every existing digest and case id is therefore unchanged.
- A changed key gives a new content digest and a new case id. Drift reads it `stale`, and a reset writes a new template revision. Revisions never pool, and `compare_two_runs` never pairs the two cases.
- `SeedTemplateCase.reference` is the same field. A host writes keyed cases through `SeedTemplateCase`, which computes the content hash with the key included. `template_case_content_digest` stays unexported (Discodon, 2026-11-19: no host computes `content_hash` itself).
- `compute_content_hash` does not change, because a generated case has no key.
- `IDENTITY_VERSION` does not move, because the change is additive.

### 1.4 Case material

- `reference_material(key: ReferenceKey) -> str` (`kernel/reference.py`, provisional) renders deterministic plain text:
  - one line saying the key is non-exhaustive, so an entity outside it is unverified, not wrong;
  - then core, then acceptable, each entity with its aliases and its facts with their source.
- A kind appends it to `JudgeEvidence.case_material` for every judge that grades against the key. It then enters the label fingerprint, as all judged evidence does.
- The engine injects nothing.
- A kind never hands `test_case.reference` to the candidate.
- Code measures read `test_case.reference` directly.

### 1.5 Grader (Chunk B, `kernel/set_grade.py`)

```python
class NamedEntity(EvalBaseModel, frozen): name: str (min 1); fields: dict[str, str] = {}
class EntityMatcher(Protocol):
    @property
    def identity(self) -> str: ...            # a host changes it when its matching changes
    def match(self, named: NamedEntity, key: ReferenceKey) -> str | None: ...   # the key entity's name, or None
class FoldingMatcher:                         # identity "fold-v1": fold_name(named.name) == fold of a name or alias
def grade_set(named: Sequence[NamedEntity], key: ReferenceKey, *, matcher: EntityMatcher | None = None) -> SetGrade
class SetGrade(EvalBaseModel, frozen):
    named: int; verified: int; core_size: int; key_size: int
    found: tuple[str, ...]; core_missed: tuple[str, ...]; unverified_names: tuple[str, ...]
    unverified -> int; recall -> float; core_recall -> float | None; verified_share -> float | None   # properties
    def host_measures(self, prefix: str) -> dict[str, float]: ...
def set_grade_descriptors(prefix: str, *, reader_label: str, matcher: EntityMatcher | None = None,
                          extractor: EntityExtractor | None = None) -> tuple[MetricDescriptor, ...]
EXTRACTED_SET_FAMILY: MeasureFamily   # name "extracted_set", graded_by "judge"
```

**Inputs and matching**

- A structured surface passes `NamedEntity` values the host builds; a pick from a host tool becomes `name` plus `fields`.
- Named entities whose folded `name` is the same count once.
- A matcher that returns a name the key lacks raises `ValueError`, because that is a host defect.
- The default matcher uses no word list.
- A host plugs in its own normaliser as an `EntityMatcher`. It sees the whole key, which is what "key-aware" needs.

**Measures and their populations**

Each measure is named `<prefix>_<measure>`. A missing value means the cell is outside that measure's n, which is how the measure states its population.

| Measure | Value | Carried on |
|---|---|---|
| `_recall` | found / key size | every graded cell |
| `_core_recall` | core found / core size | cells whose key has a core; absent when the key has no core |
| `_verified_share` | verified / named | cells that named at least one entity; absent when nothing was named |
| `_named` | count | every graded cell; diagnostic, no better direction |
| `_unverified` | count | every graded cell; diagnostic, no better direction, never a fault |

- An empty output reads `named 0`, recall 0 and no verified share. A perfect output reads recall 1 and verified share 1.
- Each description states its population.
- `formula` is `set grade v1 | matcher <identity> | extractor <identity or "structured">`. A changed matcher or extractor therefore changes the measure definition and the context key, so nothing pools by assumption.
- The family is `mechanical` without an extractor and `extracted_set` with one.

### 1.6 Extraction judge (Chunk B, `kernel/entity_extraction.py`)

```python
@dataclass(frozen=True)
class EntityExtractor:
    client: CompletionClient; entity_kind: str; fields: tuple[str, ...] = ()
    identity -> str    # "<client.model_name>|extract:<sha12 of the system prompt, user template, entity_kind, fields, schema>"
    async def extract(self, text: str, *, ledger: RoleUsageLedger) -> tuple[NamedEntity, ...]
class EntityExtractionFailed(Exception)
```

**What it is.** A model step that lists the named entities. It never sees the key, so it is not primed. It sends a strict `json_schema` response format: `{"entities": [{"name", "fields": {f: str}}]}`.

**What it refuses or skips**

- A client that names no model is refused at construction, because an unnamed extractor would pool.
- A ledger whose role is not `judge` is refused.
- Blank text returns `()` and makes no call.

**How it records**

- Every completion is folded into the ledger. The kind puts `ledger.rows()` on `CandidateTelemetry.usage`, so the spend and the served model are recorded per cell.

**How it fails.** It raises `EntityExtractionFailed` in either of these cases:

- a cut-short reply, at once;
- a reply that still does not parse after 2 retries.

The kind then returns `infra_errors`. The cell is excluded and never scored 0.

## 2. Chunks

**Seam.** Chunk B depends only on `schema/reference.py`. That file is pinned below, and both chunks commit it byte for byte. Git merges two identical adds cleanly, and if they differ, A's version wins. B adds no export, register row or snapshot entry for A's names.

```python
"""A case's reference key: the entities an answer is graded against, which a host fills."""
from __future__ import annotations
import unicodedata
from typing import Literal, Self
from pydantic import Field, field_validator, model_validator
from threetears.evals.schema.base import CoreShapeModel, EvalDocumentModel

def fold_name(text: str) -> str:
    """The name folded for matching: Unicode NFKC, case folded, whitespace runs as one space, ends stripped."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

class ReferenceFact(EvalDocumentModel, CoreShapeModel):
    """One checkable fact about an entity, with where it is stated."""
    claim: str = Field(min_length=1)
    source_url: str = Field(min_length=1)

class ReferenceEntity(EvalDocumentModel, CoreShapeModel):
    """One entity of a key: its name, the other names it goes by, and facts about it."""
    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    facts: tuple[ReferenceFact, ...] = ()

    @field_validator("aliases", mode="after")
    @classmethod
    def _aliases_in_one_order(cls, aliases: tuple[str, ...]) -> tuple[str, ...]:
        """Aliases are a set: kept in one order, so writing them in another is the same key."""
        return tuple(sorted(aliases, key=lambda alias: (fold_name(alias), alias)))

    @field_validator("facts", mode="after")
    @classmethod
    def _facts_in_one_order(cls, facts: tuple[ReferenceFact, ...]) -> tuple[ReferenceFact, ...]:
        """Facts are a set: kept in one order, so writing them in another is the same key."""
        return tuple(sorted(facts, key=lambda fact: (fact.claim, fact.source_url)))

class ReferenceKey(EvalDocumentModel, CoreShapeModel):
    """The entities a case's answer is graded against: a required core and the others that also count."""
    type: Literal["entity_set"] = "entity_set"
    core: tuple[ReferenceEntity, ...] = ()
    acceptable: tuple[ReferenceEntity, ...] = ()

    @field_validator("core", "acceptable", mode="after")
    @classmethod
    def _entities_in_one_order(cls, entities: tuple[ReferenceEntity, ...]) -> tuple[ReferenceEntity, ...]:
        """Each part is a set: kept in one order, so writing it in another is the same key and the same case."""
        return tuple(sorted(entities, key=lambda entity: (fold_name(entity.name), entity.name)))

    @property
    def entities(self) -> tuple[ReferenceEntity, ...]:
        """Every entity of the key, core first."""
        return (*self.core, *self.acceptable)

    @model_validator(mode="after")
    def _one_entity_per_name(self) -> Self:
        """Refuse a key with no entity, or a name or alias that folds alike for two entities."""
        if not self.entities:
            raise ValueError("a reference key names at least one entity, in core or acceptable")
        owner: dict[str, str] = {}
        for entity in self.entities:
            for said in {fold_name(n) for n in (entity.name, *entity.aliases)}:
                if said in owner:
                    raise ValueError(f"{said!r} names both {owner[said]!r} and {entity.name!r}")
                owner[said] = entity.name
        return self
```

### Chunk A: the key, its schema, identity and case material (`evals/bp-refkey`)

**Files**

- `schema/reference.py` (the pinned file) and the `schema/__init__.py` exports
- `schema/models.py` (`EvalTestCase.reference` and the `content_hash` docstring)
- `run/definition_digests.py`, `run/definition_seed.py`
- `kernel/reference.py` and the `kernel/__init__.py` export
- `schema/versioning.py` (17.7 and its ledger line)
- `MIGRATIONS.toml`, `CHANGELOG.md`
- `docs/evaluating-agents.md:500` and `docs/judges-and-calibration.md:135`. Prose only: no `from threetears.evals… import` line, because a shown name must be stable.
- `tests/host_vocabulary_register.py` rows
- `api/` snapshot
- `tests/test_reference_key.py`

**Done when**

1. The key refuses: no entity; a folded name or alias shared by two entities (naming both); an unknown key.
2. A case without a key dumps with no `reference` key. A case with one round-trips through `to_dict`/`from_dict`.
3. A no-key case's `template_case_content_digest` and seeded id equal values pinned from a1cf245e9. Adding a key moves both, and so does changing one alias. Writing the same key's entities, aliases or facts in another order moves neither (coordinator ruling: a key's parts are sets).
4. Drift reads a corpus case whose key changed as `stale`. A reset writes a new revision, and a carried case keeps its key.
5. `reference_material` matches a pinned golden string.
6. Reach-back: a 17.7 case with a key, loaded by a1cf245e9's model, keeps the key opaquely and writes it back. This is checked once by hand, as rule 11 requires, and stated in the report.

**Gates**

| Run | Files |
|---|---|
| 5(a) | `test_reference_key.py`, `test_definition_reset.py`, `test_definition_seed_slots.py`, `test_definition_seed_scope.py`, `test_durable_schema_pins.py`, `test_identity_golden_keys.py` |
| 5(b) | all of them |
| 5(c) stored field | `test_core_minor.py`, `test_declared_names.py`, `test_held_out_cases.py`, `test_result_reading_fields.py` |
| 5(c) host contract | `test_host_contract.py` |

**Pins**

- Expected to move: the durable pins (17.6 → 17.7) and the API snapshot.
- Must not move: identity goldens, `test_declared_names` PINNED, held-out cases.

**For hosts.** A 17.6 build behaves as follows:

- It reads a stored case carrying `reference`, keeps the key opaquely and writes it back.
- Its drift recomputes the digest without the key and reads that case `customized`.
- It **refuses a corpus file carrying `reference`** (`extra_forbidden`) at boot seed.

So upgrade the package before shipping keyed corpus files, and roll the corpus back with any rollback.

### Chunk B: the grader, matching and extraction (`evals/bp-setgrade`)

**Files**

- `schema/reference.py` (the pinned copy only)
- `kernel/set_grade.py`, `kernel/entity_extraction.py`, `kernel/__init__.py` exports
- `tests/test_set_grade.py`, `tests/test_entity_extraction.py`, using the `threetears.evals.testing` scripted or recorded clients (any `Fake*` declares parity)
- `tests/enforcement/_public_api_consumers.txt` (Q4)
- register rows, `api/` snapshot, `CHANGELOG.md`
- a `docs/judges-and-calibration.md` section on grading a named set (prose, no imports)

**Done when**

1. One fixture counts both directions:
   - core recall, full recall and verified share;
   - unverified names listed and never counted as wrong.
2. The empty, perfect and no-core states give the measure sets in §1.5.
3. A host matcher replaces the default. A changed `identity` changes `measure_definition_digests`.
4. The default folds `É`/`é` and `ﬁ`/`fi`, and does not fold `é` to `e`.
5. A scripted extractor:
   - returns ordered entities;
   - records ledger rows;
   - retries twice, then raises;
   - raises on cut-short;
   - makes no call on blank text;
   - refuses an unnamed client.
6. A runner-level test: a cell that omits `_verified_share` is accepted, and the run summary's n counts only the cells carrying it.

**Gates**

| Run | Files |
|---|---|
| 5(a) | both new tests and `test_measures_*`/`test_usage_capture*`, where present |
| 5(b) | all of them, plus `test_no_matching_over_model_prose.py`: extracted names are plain `str`, not `ModelProse` |
| 5(c) | none (no stored field) |

**Expected reds in B's worktree, which clear at integration:** `test_public_surface_is_closed` and the register gate, because A's names are not yet exported.

**Pins.** Only the API snapshot moves.

**Integration.** Generated surface files (`api/surface.json`, `surface_tiers.json`, `docs/surface.md`, `llms.txt`) conflict. The integration lead regenerates them, and merges `CHANGELOG.md`.

## 3. Open questions

1. **Quick path.** Defer it, and file a follow-up for a reserved `reference` case key that is withheld from the runner, given to graders and rendered by the default case material, plus a way for one grader to leave a measure out on a cell. *Recommend: defer.*
2. **Extraction spend role.** The rows go under `judge` on the kind's telemetry. Surfaces that list a run's judges may then show the extractor's model beside the rubric judge. *Recommend `judge` now*, with an engine extraction phase as a later issue.
3. **Name of "precision".** It ships as `_verified_share` (reader name "Named, found in the key"), because "precision" implies the rest is wrong. *Recommend that name.*
4. **Callers for B's names.** No `src` module calls them, so the gate needs ledger lines naming Discodon's kinds as the caller. *Recommend* the ledger lines, after confirming with Discodon that its kinds call `grade_set`, `EntityExtractor` and `set_grade_descriptors`.
5. **Durable minor.** 17.7 assumes no other in-flight branch claims it. Integration numbers it by merge order.
6. **Accents.** The default does not strip accents, because that is language-specific. A host matcher may strip them. *Recommend as designed.*
