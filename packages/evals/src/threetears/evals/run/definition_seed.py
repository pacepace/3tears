"""Seed hand-authored eval definitions from a host's files into one scope of its store.

Four eval document types are authored in a store and nothing re-creates them. Three of
them are hand-authored — ``eval_template``, ``rubric_dim``, ``judge_config`` — and this
module seeds a host's reviewed corpus of each so a fresh scope, or one whose corpus was
dropped, comes up with a working measurement instrument instead of an empty catalogue. The
corpus is the host's and so is its location: the host passes the directory in.

**A corpus is loaded for one scope.** The files carry no scope — it is a server-owned field,
like ``id`` — so :func:`load_seed_corpus` takes the scope the definitions are built in, and
:func:`seed_eval_definitions` reads and writes that scope alone. A host sharing one corpus
across tenants seeds each scope it names; the engine never decides which scopes exist.
(The fourth, ``eval_test_case``, is generated from a template's ``variation_axes`` by
``threetears.evals.gen.variation_gen``, so seeding it would be seeding an observation; the
templates here regenerate it.)

**The ownership rule is the whole design.** Code defaults seed empty slots
only; the store is master once seeded. An operator can tune a template in production and
trust that the next boot will not stomp it — the same guarantee prompt seeding gives, and
the reason seeding from git is safe here where an unconditional write would not be.

Applying that rule needs a definition of "slot", and eval definitions do not supply one the
way prompts do. A prompt slot is a fixed, enumerable coordinate (one preset per type); a
template is one member of an open collection with a generated UUID for an id. So occupancy
is asked of a **natural key**:

- ``eval_template`` is keyed by ``name`` (``EvalStorage.load_template_by_name``).
- ``rubric_dim`` is keyed by ``key`` (``EvalStorage.load_active_rubric_dim``).
- ``judge_config`` is keyed by ``(rubric_dim_id, name)``. Not by the dim alone: a dim may carry
  more than one config — the active one a run inherits, and others a launch names by id in
  ``judge_config_ids`` (an archived config is accepted there deliberately: it is how an A/B's
  control arm runs). Keyed by the dim, every config after the first for a dim read as an occupied
  slot and was never written. The name is what an operator's re-authoring keeps
  (``update_judge_config`` archives and recreates under the same name), so one authored config's
  versions share one slot.

**A corpus that cannot be seeded as written is refused, never partly skipped.** Two documents
under one natural key, and two non-archived judge configs for one dim — a state no authoring path
can produce (``create_judge_config`` keeps one active config per dim, and which of two would score
the dim would be decided by load order) — are refused when the :class:`SeedCorpus` is built.

**Occupancy is archived-inclusive, and that is the load-bearing part.** The runtime's own
lookups (``load_active_rubric_dim``, ``load_active_judge_config``) filter ``archived=False`` —
correctly, because a retired definition should be invisible to a *run*. Reusing them here would mean an operator who archives a seeded definition, which
is the supported way to retire one, finds it back at the next boot and every boot after.
Archiving is a decision; a seed must not overturn it. This module therefore probes with the
archived-inclusive ``query_*`` methods and treats any record carrying the natural key,
active or archived, as an occupied slot.

The seeder creates. It never updates and never deletes.

Corpus files are the same create-ready shape the authoring paths accept — server-owned
fields (``id``, ``doc_type``, ``schema_version``, ``scope_id``, ``created_at``,
``updated_at``) are absent, because loading and saving assign them.

**Every template meets the gates ``create_template`` applies, before anything is written.** The
seeder admits each one through :func:`~threetears.evals.run.authoring.admit_template` — the same
function ``create_template`` calls — so the kind's spec model, the rubric's names, the host's tool
catalog, the world gate, the goal checks' discrimination proof, the host's seed walk and its kind
capabilities all apply, and a gate added to authoring reaches a corpus without anyone mirroring it.
That is why the seeder takes the host and the same three host checks ``create_template`` takes. A
refusal of any template refuses the whole seed, naming every refused template, and nothing is
written: a corpus is the host's committed code, and one the host's own gates refuse is a defect to
fix, not a partial catalogue to boot with. The whole corpus is admitted, written or not, so the
verdict depends on the corpus and the host alone and a host's CI seeding an empty store reaches the
answer its production boot would.

Two authoring refusals that read the store become something else here. A taken template name, or
rubric dim key, is an occupied slot. And ``create_judge_config``'s one-active-config-per-dim rule
against the store becomes a withheld write: a non-archived corpus
config for a dim the store already holds an active config for (one an operator authored under
another name) is not written — writing it would supersede the operator's config for every run — and
is reported as ``conflicted``, by key, on every boot until one of the two is archived.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from threetears.evals.contracts.errors import StorageError, ValidationFailedError
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.models import CatalogRubricDim, EvalTemplate, JudgeConfig
from threetears.evals.run.authoring import admit_template


@dataclass(frozen=True)
class SeedCorpus:
    """A host's authored definitions for one scope, already constructed through their production models.

    Raises:
        ValueError: A definition names a scope other than ``scope_id``. A corpus seeds one scope,
            and a definition from another would be written there by a seed that reads only this
            one — so its slot would never read occupied, and every boot would write it again.
            Also two definitions of one type under one natural key (template ``name``, rubric dim
            ``key``, judge config ``(rubric_dim_id, name)``), since the seed would write the first
            and read the second as occupied by it; and two non-archived judge configs for one
            ``rubric_dim_id``, a state no authoring path can produce — archive all but the one a
            run should inherit, and name the others by id in a launch's ``judge_config_ids``.
    """

    scope_id: str
    templates: tuple[EvalTemplate, ...] = ()
    rubric_dims: tuple[CatalogRubricDim, ...] = ()
    judge_configs: tuple[JudgeConfig, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a definition built for another scope, and a corpus that cannot be seeded as written."""
        definitions: tuple[EvalTemplate | CatalogRubricDim | JudgeConfig, ...] = (
            *self.templates,
            *self.rubric_dims,
            *self.judge_configs,
        )
        if strays := sorted({d.id for d in definitions if d.scope_id != self.scope_id}):
            raise ValueError(
                f"seed corpus for scope {self.scope_id!r} carries definitions from another scope: {strays}"
            )
        defects = [
            *(f"two templates named {k!r}" for k in _repeated(t.name for t in self.templates)),
            *(f"two rubric dims keyed {k!r}" for k in _repeated(d.key for d in self.rubric_dims)),
            *(
                f"two judge configs named {name!r} for rubric dim {dim!r}"
                for dim, name in _repeated(_config_key(c) for c in self.judge_configs)
            ),
            *(
                f"more than one non-archived judge config for rubric dim {dim!r} — archive all but the one a run "
                "should inherit; a launch names the others by id in judge_config_ids"
                for dim in _repeated(c.rubric_dim_id for c in self.judge_configs if not c.archived)
            ),
        ]
        if defects:
            raise ValueError(
                f"seed corpus for scope {self.scope_id!r} cannot be seeded as written: {'; '.join(defects)}"
            )


def _repeated[K: Hashable](keys: Iterable[K]) -> list[K]:
    """The keys occurring more than once, each once, in first-repeat order."""
    seen: set[K] = set()
    repeated: list[K] = []
    for key in keys:
        if key in seen and key not in repeated:
            repeated.append(key)
        seen.add(key)
    return repeated


def _config_key(config: JudgeConfig) -> tuple[str, str]:
    """A judge config's slot: the dim it scores and its name."""
    return (config.rubric_dim_id, config.name)


@dataclass
class SeedOutcome:
    """What one seeding pass did, per doc type.

    Every outcome is reported rather than just the writes, because the interesting
    states are the ones a single number cannot separate: a pass that created nothing
    because everything was already there, and a pass that created nothing because the
    corpus never reached the runtime, are opposite events with the same ``created`` total.

    ``failed`` counts writes the store refused with ``StorageError``. The seed does not abort
    on one — a single refused definition is no reason to leave the rest of the catalogue
    empty — so the refusal has to be counted somewhere, and counting it as created would put
    "seeded" in the boot log for records the store does not hold. ``conflicted`` names the
    definitions withheld because the store holds a live record they would contradict.
    """

    created: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    failed: dict[str, int] = field(default_factory=dict)
    #: Natural keys actually written, per doc type. Counts say how many; a partial seed is
    #: only diagnosable from the log if it says *which*.
    created_keys: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys NOT written because the store already holds a live record the write would
    #: contradict, per doc type — today only a non-archived judge config for a dim the store has an
    #: active config for. Keys rather than a count, because the operator resolves each one.
    conflicted: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total_created(self) -> int:
        """Definitions written across every doc type."""
        return sum(self.created.values())

    @property
    def total_skipped(self) -> int:
        """Slots found already occupied across every doc type."""
        return sum(self.skipped.values())

    @property
    def total_failed(self) -> int:
        """Writes the storage layer refused across every doc type."""
        return sum(self.failed.values())

    @property
    def total_conflicted(self) -> int:
        """Definitions withheld because the store holds a live record they would contradict."""
        return sum(len(keys) for keys in self.conflicted.values())

    @property
    def nothing_to_do(self) -> bool:
        """True when the pass neither wrote, skipped, failed nor withheld anything.

        That is not the reassuring case it resembles: with a corpus present, every
        definition is created, skipped, failed or withheld, so all-zero means the corpus itself was
        empty or unreachable — the state seeding exists to prevent, reported by the caller
        as a warning rather than the routine "already present" line.
        """
        return not (self.total_created or self.total_skipped or self.total_failed or self.total_conflicted)

    def summary(self) -> str:
        """One operator-readable line naming every type, including the zeroes.

        Types with nothing to do are named too: a boot that seeded no judge configs because
        the corpus carries none reads identically to one that skipped five, unless the line
        says which. Failures and conflicts are named only when there are some, so the ordinary
        line stays readable. Created keys are named because "1 created, 1 already present" does not say
        which of the two was written.
        """
        if self.nothing_to_do:
            return "no definitions in the seed corpus"
        parts = []
        for doc_type in sorted(set(self.created) | set(self.skipped) | set(self.failed) | set(self.conflicted)):
            part = f"{doc_type}: {self.created.get(doc_type, 0)} created"
            if keys := self.created_keys.get(doc_type):
                part += f" ({', '.join(keys)})"
            part += f", {self.skipped.get(doc_type, 0)} already present"
            if failed := self.failed.get(doc_type, 0):
                part += f", {failed} REFUSED BY STORAGE"
            if conflicted := self.conflicted.get(doc_type):
                part += f", {len(conflicted)} NOT WRITTEN, CONFLICTING WITH THE STORE ({', '.join(conflicted)})"
            parts.append(part)
        return "; ".join(parts)


def _load_dir[M: BaseModel](directory: Path, model: type[M], scope_id: str) -> tuple[M, ...]:
    """Construct every ``*.json`` in ``directory`` through ``model``, in ``scope_id``.

    Sorted so a corpus loads in a stable order and a seeding log is diffable between boots.

    Raises:
        ValueError: A file is not valid JSON, or is not a valid document for ``model``.
            Raised rather than skipped: a definition that silently fails to load is a
            measurement instrument missing a dimension nobody notices is gone.
    """
    if not directory.is_dir():
        return ()
    loaded: list[M] = []
    for path in sorted(directory.glob("*.json")):
        try:
            loaded.append(model(**json.loads(path.read_text(encoding="utf-8")), scope_id=scope_id))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"seed definition {path} is not a valid {model.__name__}: {e}") from e
    return tuple(loaded)


def load_seed_corpus(seed_dir: Path, scope_id: str) -> SeedCorpus:
    """Read a corpus off disk, constructing each document through its model in ``scope_id``.

    The directory is the host's to name. The run package ships the seeder, not any host's
    definitions, so it holds no path to a corpus: a host with authored definitions passes
    their directory here.

    Args:
        seed_dir: A directory holding ``templates/``, ``rubric_dims/`` and ``judge_configs/``
            subdirectories of ``*.json`` documents. A missing subdirectory loads as empty.
        scope_id: The scope the definitions are built in. A file that names a ``scope_id`` of
            its own does not construct — the field is passed twice — because the scope is the
            host's to choose at load, not the file's.

    Returns:
        The corpus. Empty tuples where a subdirectory is absent or holds no ``*.json``.

    Raises:
        FileNotFoundError: ``seed_dir`` itself is not a directory. A host naming a corpus that
            is not there has a wrong path, not an empty corpus, and loading it as empty would
            seed nothing with nothing in the log to say why.
        ValueError: A file in the corpus does not construct. This is deliberately loud —
            it means the host ships a definition the server would refuse.
    """
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"eval seed corpus directory {seed_dir} does not exist or is not a directory")
    return SeedCorpus(
        scope_id=scope_id,
        templates=_load_dir(seed_dir / "templates", EvalTemplate, scope_id),
        rubric_dims=_load_dir(seed_dir / "rubric_dims", CatalogRubricDim, scope_id),
        judge_configs=_load_dir(seed_dir / "judge_configs", JudgeConfig, scope_id),
    )


def seed_eval_definitions(
    host: EvalHost,
    corpus: SeedCorpus,
    *,
    require_known_tools_allowed: Callable[[Sequence[str] | None], None],
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None],
    refuse_undeliverable_template: Callable[[EvalTemplate], None],
) -> SeedOutcome:
    """Create any corpus definition whose natural key is absent from the corpus's scope.

    Seeding semantics: empty slots only, the store is master once seeded. Occupancy is decided
    against the archived-inclusive queries, so a definition an operator archived stays
    archived rather than being resurrected at the next boot.

    Every template is first admitted through the gates ``create_template`` applies
    (:func:`~threetears.evals.run.authoring.admit_template`), all of them before any write, and is
    stored in the shape admission resolves (its ``kind_spec`` as the kind's spec model fills it).

    Nothing is ever updated and nothing is ever deleted. An operator's own definitions, and
    their edits to seeded ones, are untouched.

    A write the storage layer refuses (``StorageError``) is counted as ``failed``, never as
    created, and does not abort the seed: one refused definition is no reason to leave the
    rest of the catalogue empty. The slot is deliberately left unoccupied, so the next boot
    re-attempts it. A non-archived judge config for a dim the store already holds an active
    config for is not written and is reported under ``conflicted``.

    Args:
        host: The host: the store being seeded, and the profile a template is held to.
        corpus: The definitions to seed and the scope they are seeded into, which the host
            loads (:func:`load_seed_corpus` over its own directory) or builds in memory.
            Required rather than defaulted, so the seeder never decides whose definitions a
            scope receives.
        require_known_tools_allowed: The host's tool-catalog check, as ``create_template`` takes it.
        refuse_undeclared_world_seed: The host's seed walk, as ``create_template`` takes it.
        refuse_undeliverable_template: The host's kind-capability check, as ``create_template`` takes it.

    Returns:
        Per-doc-type created, skipped and failed counts, the natural keys written, and the keys
        withheld as conflicting with the store.

    Raises:
        ValidationFailedError: A corpus template fails a gate ``create_template`` applies. Raised
            before anything is written, naming every refused template and why.
    """
    templates: list[EvalTemplate] = []
    refusals: list[str] = []
    for template in corpus.templates:
        try:
            templates.append(
                admit_template(
                    template,
                    profile=host.profile,
                    require_known_tools_allowed=require_known_tools_allowed,
                    refuse_undeclared_world_seed=refuse_undeclared_world_seed,
                    refuse_undeliverable_template=refuse_undeliverable_template,
                )
            )
        except ValidationFailedError as refused:
            refusals.append(f"{template.name!r}: {refused.message}")
    if refusals:
        raise ValidationFailedError(
            f"the seed corpus for scope {corpus.scope_id!r} carries templates authoring would refuse, so nothing "
            f"was seeded: {'; '.join(refusals)}"
        )

    storage = host.storage
    outcome = SeedOutcome()

    # One archived-inclusive query per type, not one probe per document: the catalogue is
    # small, and a single read is both cheaper and impossible to accidentally write with an
    # `archived=False` filter the way a per-document `load_active_*` call would be.
    existing_configs = storage.query_judge_configs(corpus.scope_id)
    dims_with_an_active_config = {c.rubric_dim_id for c in existing_configs if not c.archived}

    _seed_doc_type(
        outcome,
        "eval_template",
        templates,
        {(t.name,) for t in storage.query_templates(corpus.scope_id)},
        lambda t: (t.name,),
        storage.save_template,
    )
    _seed_doc_type(
        outcome,
        "rubric_dim",
        corpus.rubric_dims,
        {(d.key,) for d in storage.query_rubric_dims(corpus.scope_id)},
        lambda d: (d.key,),
        storage.save_rubric_dim,
    )
    _seed_doc_type(
        outcome,
        "judge_config",
        corpus.judge_configs,
        {_config_key(c) for c in existing_configs},
        _config_key,
        storage.save_judge_config,
        # The corpus holds at most one non-archived config per dim (SeedCorpus refuses more), so the
        # store's active configs are the only ones a write could contradict.
        conflicts=lambda c: not c.archived and c.rubric_dim_id in dims_with_an_active_config,
    )

    return outcome


def _seed_doc_type[D](
    outcome: SeedOutcome,
    doc_type: str,
    definitions: Sequence[D],
    occupied: set[tuple[str, ...]],
    natural_key: Callable[[D], tuple[str, ...]],
    save: Callable[[D], None],
    *,
    conflicts: Callable[[D], bool] = lambda _definition: False,
) -> None:
    """Write one doc type's definitions into their empty slots, recording the counts on ``outcome``.

    Args:
        outcome: The seed's running tally, written under ``doc_type``.
        doc_type: The doc type the counts are recorded under.
        definitions: That type's definitions from the corpus.
        occupied: The natural keys already present.
        natural_key: A definition's slot. Logged joined with ``/``.
        save: The storage write for this type.
        conflicts: Whether writing an unoccupied definition would contradict a live record; such a
            definition is not written and is recorded under ``conflicted``.
    """
    created_keys: list[str] = []
    conflicted: list[str] = []
    skipped = failed = 0
    for definition in definitions:
        key = natural_key(definition)
        # SeedCorpus refuses two definitions under one key, so `occupied` is only ever the store's.
        if key in occupied:
            skipped += 1
            continue
        if conflicts(definition):
            conflicted.append("/".join(key))
            continue
        try:
            save(definition)
        except StorageError:
            # `_save` logged the cause already. Counted, not raised, and the slot is
            # left unoccupied so the next boot retries — a refused write must not read
            # as a seeded one.
            failed += 1
            continue
        created_keys.append("/".join(key))
    outcome.created[doc_type] = len(created_keys)
    outcome.created_keys[doc_type] = created_keys
    outcome.skipped[doc_type] = skipped
    outcome.failed[doc_type] = failed
    if conflicted:
        outcome.conflicted[doc_type] = conflicted


__all__ = [
    "SeedCorpus",
    "SeedOutcome",
    "load_seed_corpus",
    "seed_eval_definitions",
]
