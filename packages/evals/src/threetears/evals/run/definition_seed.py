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

**A deleted rubric dim stays deleted.** Deleting is a decision too, and a delete empties the slot,
so on its own it would hand a seeded dim back to the next boot. ``delete_rubric_dim`` therefore
leaves a :class:`~threetears.evals.schema.models.RubricDimTombstone` for the key, and a
tombstoned key is not written: it is reported under ``deleted``, by key, on every boot.
``delete_judge_config`` does the same for a judge config's slot
(:class:`~threetears.evals.schema.models.JudgeConfigTombstone`). Templates carry none: no
operation deletes a template, only the storage port's own ``delete_template``, which writes no
tombstone, so a template deleted straight from the store still empties its slot.

**Authored judge cases seed through the door every authored case passes.** A ``judge_cases`` slot holds
:class:`SeedJudgeCase` values: ``judge_case_author``'s arguments with its two templates named by ``name``
(a corpus cannot know a generated id), plus the author. Each is built through
:func:`~threetears.evals.run.judge_kind.author_judge_case` (the criterion read from the template that words it, the
verdict held to its scale) and frozen with :func:`~threetears.evals.run.freeze_judge_cases` (``authored=``) into
its judge template, so it is the :class:`~threetears.evals.kernel.AuthoredJudgeCase` ``judge_case_author`` writes,
under the same content-derived id. Its slot is that id: a case already stored is ``already present``. A case is
written only into a judge template revision that held no cases when the seed began (since ``.30``): one whose words or
verdict changed, or a new one, for a revision already holding cases would change that revision's battery under the
same id, so it is withheld under ``needs_reset`` until ``reset_definitions`` writes the next revision. A template is resolved in the store first (the
store is master, and the template a stored case is judged against) and else in the corpus, which the same seed
writes. Every case is built before anything is written, and a case authoring would refuse refuses the whole seed,
naming each; except that a case whose criterion the stored template lacks (re-worded under the same name) is withheld
under ``needs_reset`` like one for a battery already holding cases, since a reset repairs it, and the rest seed. No operation deletes a judge case, so, as for a template, a case deleted straight from the store
empties its slot.

**A template's own cases seed with it.** A host authors stimuli for its own templates (a scenario to play out, a
set of inputs to classify) as cases rather than variation axes, and a ``template_cases`` slot holds them as
:class:`SeedTemplateCase` values: the template by name, the case's name, and what an
:class:`~threetears.evals.schema.models.EvalTestCase` carries of it (``variation_params``, the host's opaque
``host_payload``, ``stratum``). Each names a template the corpus carries, of a kind other than the judge's (a judge
template's cases are its judge cases), and is written into that template's current revision after the templates,
under an id derived from the revision and its content, carrying its corpus name as ``name`` and ``seed_name`` and the
digest of its content as ``content_hash`` (its stratum in neither, as for a generated case). The judge battery's rule holds for it, for the same reason: a template's
live cases are what a launch of it runs, so a case is written only into a revision that held no cases when the seed
began, and one whose content the revision already holds is ``already present``. A new case, or one whose content
changed, for a revision already holding cases is withheld under ``needs_reset`` until ``reset_definitions`` writes the
template's next revision with every corpus case of it.

**A case set over authored cases seeds too.** A template carries no list of its cases, and an authored case's stored
id is derived from its judge template's generated id, so a set a launch targets (``hard_negatives v1``) cannot be
implied by a template or written as stored ids. A ``case_sets`` slot holds :class:`SeedCaseSet` values instead: a
name, the judge template by name, and the corpus judge cases it lists by their corpus names, in order. Every list is
checked against the corpus when the :class:`SeedCorpus` is built (each name carried, once, and frozen into the set's
template), so a bad set refuses the whole corpus; its cases are built with the rest before anything is written. Once
they are frozen, each set is minted over their stored ids through
:func:`~threetears.evals.run.case_sets.mint_case_set`, the path ``case_set_mint`` takes. Its slot is its name: a scope
holding any version of it is already seeded, and the next version is ``case_set_mint``'s to mint.

The seeder creates. It never updates and never deletes.

**Every definition it writes records what was seeded** (since ``.30``): a template, rubric dim, judge config and case
set carry ``seed_digest``, the digest of the content written (:mod:`~threetears.evals.run.definition_digests`), and a
judge case its corpus name (``seed_name``; its ``content_hash`` is what was seeded). Bringing a seeded definition up to
date when the corpus moves is not the seed's work: :mod:`~threetears.evals.run.definition_drift` reads the drift and
resets, explicitly, writing a new revision. A template name resolves to its current revision
(``EvalStorage.load_template_by_name``), which is what a seeded judge case and case set are frozen into.

Corpus files are the same create-ready shape the authoring paths accept — server-owned
fields (``id``, ``doc_type``, ``schema_version``, ``scope_id``, ``created_at``,
``updated_at``) are absent, because loading and saving assign them.

**Every template meets the gates ``create_template`` applies, before anything is written.** The
seeder admits each one through :func:`~threetears.evals.run.authoring.admit_template` — the same
function ``create_template`` calls — so the kind's spec model, the rubric's names, the host's tool
catalog, the world gate, the goal checks' discrimination proof, the host's seed walk, its kind
capabilities and the judge declaration (every judged dimension says ``judge: "builtin"`` or
``"configured"``, a configured one honoured by a config the corpus seeds or the store holds) all
apply, and a gate added to authoring reaches a corpus without anyone mirroring it.
That is why the seeder reads the host's template checks where ``create_template`` reads them (``template_checks``). A
refusal of any template refuses the whole seed, naming every refused template, and nothing is
written: a corpus is the host's committed code, and one the host's own gates refuse is a defect to
fix, not a partial catalogue to boot with. The whole corpus is admitted, written or not, so the
verdict depends on the corpus and the host alone and a host's CI seeding an empty store reaches the
answer its production boot would (the one store read, a configured dimension's config, finds in an
empty store only what the corpus itself seeds).

Two authoring refusals that read the store become something else here. A taken template name, or
rubric dim key, is an occupied slot. And ``create_judge_config``'s one-active-config-per-dim rule
against the store becomes a withheld write: a non-archived corpus
config for a dim the store already holds an active config for (one an operator authored under
another name) is not written — writing it would supersede the operator's config for every run — and
is reported as ``conflicted``, by key, on every boot until one of the two is archived.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.kernel.errors import ConflictError, StorageError, ValidationFailedError
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.judge_cases import JUDGE_KIND, AuthoredJudgeCase, AuthoredVerdict, JudgeCase
from threetears.evals.schema.base import EvalBaseModel, VerbatimObject
from threetears.evals.schema.models import (
    CatalogRubricDim,
    EvalTemplate,
    EvalTestCase,
    JudgeConfig,
    JudgedArtifact,
)
from threetears.evals.run.authoring import active_judge_config_in, admit_template, admit_template_case
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.definition_digests import (
    TEMPLATE_CASE_FIELDS,
    case_set_digest,
    current_revisions,
    judge_config_digest,
    rubric_dim_digest,
    template_case_content_digest,
    template_case_digest,
    template_digest,
)
from threetears.evals.run.judge_kind import author_judge_case, freeze_seeded_judge_case
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

#: What a seeded template case's id is derived under, with its template revision's id and its content digest.
_TEMPLATE_CASE_NAMESPACE = uuid.UUID("dea1f6bf-3c5a-42ef-915a-49e665a6a964")


class SeedJudgeCase(EvalBaseModel):
    """One authored judge case a seed corpus writes: ``judge_case_author``'s arguments, its templates named by name.

    A corpus file holds this shape (``judge_cases/*.json``). The templates are named by ``name``, since a corpus
    cannot know the id a seeded template is given; the criterion is named, never restated, for the reason
    ``judge_case_author`` names it (its wording is what a judge profile is keyed by). The seed builds it into a
    :class:`~threetears.evals.kernel.AuthoredJudgeCase` through the same door (``author_judge_case``). Its own
    ``name`` is the corpus's handle on it, which a :class:`SeedCaseSet` lists; it is not part of the case, so two
    cases that differ only by name are one case.
    """

    model_config = ConfigDict(frozen=True)

    judge_template: str = Field(min_length=1, description="The judge template the case is frozen into, by name.")
    criterion_template: str = Field(
        min_length=1, description="The subject template whose rubric dimension the case is judged on, by name."
    )
    criterion: str = Field(min_length=1, description="The rubric dimension's name on that template.")
    output: str = Field(min_length=1, description="The output under review, as the judge reads it.")
    case_material: str = Field(min_length=1, description="What the output is judged against: its source, or its facts.")
    expected: AuthoredVerdict = Field(
        description="The verdict the output must get: 'pass' or 'fail' (or 1 / 0) on pass/fail, 1 to 5 on ordinal."
    )
    note: str = Field(min_length=1, description="Why the output must get that verdict, in the author's words.")
    author: str = Field(min_length=1, description="Who wrote the case, as the host names them: the label's rater.")
    scenario: str | None = Field(
        default=None, min_length=1, description="The scenario the judge reads; omitted reads the template's intent."
    )
    subject: str | None = Field(
        default=None, description="Who or what produced the output, as the judge should know it; omitted for none."
    )
    judged_artifact: Literal["document", "transcript"] = Field(
        default="document", description="'document' (judged against its material) or 'transcript'."
    )
    name: str | None = Field(
        default=None,
        min_length=1,
        description="The case's name in its corpus, which a seeded case set lists it by: a corpus file's name without "
        "`.json` when the file names none. None for a case no case set lists. Not part of the case.",
    )

    def authored(self, criterion_template: EvalTemplate) -> AuthoredJudgeCase:
        """The case, built against the template that words its criterion, as ``judge_case_author`` builds one.

        :raises ValidationFailedError: The template has no such dimension, or its scale cannot hold the verdict.
        """
        return author_judge_case(
            criterion_template,
            self.criterion,
            output=self.output,
            case_material=self.case_material,
            expected=self.expected,
            note=self.note,
            author=self.author,
            scenario=self.scenario,
            subject=self.subject,
            judged_artifact=JudgedArtifact(self.judged_artifact),
        )


class SeedTemplateCase(EvalBaseModel):
    """One of a template's own cases a seed corpus writes with it: a stimulus the host authored, under a name.

    A corpus file holds this shape (``template_cases/*.json``): the template by ``name`` (a corpus cannot know the id
    a seeded template is given) and what a stored :class:`~threetears.evals.schema.models.EvalTestCase` carries of the
    case. The seed writes it into its template's current revision, and a reset into the revision it writes. What
    drift compares is its ``variation_params``, ``host_payload`` and ``stratum``
    (:func:`~threetears.evals.run.definition_digests.template_case_digest`), so a relabelled stratum reads
    ``customized``; its ``content_hash`` and id are its content alone, the stratum left out as a generated case's is
    (:func:`~threetears.evals.run.definition_digests.template_case_content_digest`). Its name is the corpus's handle on
    it and the name the report gives it, not part of it.
    """

    model_config = ConfigDict(frozen=True)

    template: str = Field(
        min_length=1,
        description="The template the case belongs to, by name: one the corpus carries, of a kind other than the "
        "judge's (a judge template's cases are its judge cases).",
    )
    name: str = Field(
        min_length=1,
        description="The case's name, unique within its template: a corpus file's name without `.json` when the file "
        "names none. Stored as the case's `name` (what the report calls it) and `seed_name`. Not part of the case.",
    )
    variation_params: dict[str, str] = Field(
        default_factory=dict, description="The case's axis values, as a generated case freezes its template's axes."
    )
    host_payload: VerbatimObject = Field(
        default_factory=dict,
        description="The case's own stimulus in the host's vocabulary (a scenario, an input set), stored as written "
        "and never read by the engine.",
    )
    stratum: str | None = Field(
        default=None, min_length=1, description="The kind of case it is, which the analysis reads results by."
    )

    def digest(self) -> str:
        """The digest of the case's content, as a stored case of it digests (:func:`stored_template_case_digest`)."""
        return template_case_digest(self)

    def test_case(self, template: EvalTemplate) -> EvalTestCase:
        """The case as a test case of ``template``'s revision, under an id derived from the revision and its content,
        never its stratum."""
        digest = template_case_content_digest(self)
        return EvalTestCase(
            id=str(uuid.uuid5(_TEMPLATE_CASE_NAMESPACE, f"{template.id}|{digest}")),
            scope_id=template.scope_id,
            template_id=template.id,
            **self.model_dump(include=set(TEMPLATE_CASE_FIELDS)),
            name=self.name,
            content_hash=digest,
            seed_name=self.name,
            seed_digest=self.digest(),
        )


def stored_template_case_digest(case: EvalTestCase) -> str:
    """The digest of a stored test case's content as a template's own case: what a corpus case of it digests to."""
    return template_case_digest(case)


class SeedCaseSet(EvalBaseModel):
    """One named case set a seed corpus mints: a judge template's authored cases, listed by name, in order.

    A corpus file holds this shape (``case_sets/*.json``). A template carries no list of its cases, and an authored
    case's stored id is derived from its judge template's id, which a corpus cannot know, so a set over authored cases
    names them by their corpus names (:attr:`SeedJudgeCase.name`). The seed mints it through
    :func:`~threetears.evals.run.case_sets.mint_case_set`, the path ``case_set_mint`` takes, once its cases are frozen,
    so a fresh scope boots with ``<name> v1`` listing exactly them. Its slot is its ``name``: a scope holding any
    version of the set is already seeded, and a corpus whose list changed mints nothing over it (the store is master;
    ``case_set_mint`` mints the next version).
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, description="The set's name, as a launch targets it (`<name> v<version>`).")
    template: str = Field(min_length=1, description="The judge template whose cases it lists, by name.")
    judge_cases: tuple[str, ...] = Field(
        min_length=1,
        description="The corpus judge cases it lists, by their names, in order; each a case of `template`.",
    )
    tracked: bool = Field(
        default=True, description="Whether the set is a standing suite followed over time, or made for one launch."
    )


@dataclass(frozen=True)
class SeedCorpus:
    """A host's authored definitions for one scope, already constructed through their production models.

    :raises ValueError: A definition names a scope other than ``scope_id``. A corpus seeds one scope,
        and a definition from another would be written there by a seed that reads only this
        one — so its slot would never read occupied, and every boot would write it again.
        Also two definitions of one type under one natural key (template ``name``, rubric dim
        ``key``, judge config ``(rubric_dim_id, name)``), since the seed would write the first
        and read the second as occupied by it; and two non-archived judge configs for one
        ``rubric_dim_id``, a state no authoring path can produce — archive all but the one a
        run should inherit, and name the others by id in a launch's ``judge_config_ids``. And two
        identical judge cases, which would freeze into one stored case. And two judge cases under one name, two
        case sets under one name, and a case set listing a case twice, naming a case the corpus does not carry,
        or naming one of another judge template than its own. And a template case naming a template the corpus
        does not carry, or one of the judge kind, and two template cases of one template under one name or
        with one content.
    """

    scope_id: str
    templates: tuple[EvalTemplate, ...] = ()
    rubric_dims: tuple[CatalogRubricDim, ...] = ()
    judge_configs: tuple[JudgeConfig, ...] = ()
    judge_cases: tuple[SeedJudgeCase, ...] = ()
    case_sets: tuple[SeedCaseSet, ...] = ()
    template_cases: tuple[SeedTemplateCase, ...] = ()

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
            *(
                f"two identical judge cases for {case.criterion!r} of {case.criterion_template!r} in judge template "
                f"{case.judge_template!r}"
                for case in _repeated(case.model_copy(update={"name": None}) for case in self.judge_cases)
            ),
            *(f"two judge cases named {k!r}" for k in _repeated(c.name for c in self.judge_cases if c.name)),
            *(f"two case sets named {k!r}" for k in _repeated(s.name for s in self.case_sets)),
            *(defect for case_set in self.case_sets for defect in self._case_set_defects(case_set)),
            *self._template_case_defects(),
        ]
        if defects:
            raise ValueError(
                f"seed corpus for scope {self.scope_id!r} cannot be seeded as written: {'; '.join(defects)}"
            )

    def _template_case_defects(self) -> list[str]:
        """Why the template cases cannot be written as the corpus states them: each, or none."""
        kinds = {template.name: template.candidate_kind for template in self.templates}
        defects = [
            f"template case {case.name!r} names template {case.template!r}, which the corpus does not carry"
            for case in self.template_cases
            if case.template not in kinds
        ]
        defects += [
            f"template case {case.name!r} names judge template {case.template!r}, whose cases are its judge cases "
            "(judge_cases/)"
            for case in self.template_cases
            if kinds.get(case.template) == JUDGE_KIND
        ]
        defects += [
            f"two cases of template {template!r} named {name!r}"
            for template, name in _repeated((case.template, case.name) for case in self.template_cases)
        ]
        identical = _repeated((case.template, case.digest()) for case in self.template_cases)
        defects += [f"two identical cases of template {template!r}" for template, _digest in identical]
        return defects

    def _case_set_defects(self, case_set: SeedCaseSet) -> list[str]:
        """Why a case set cannot be minted over this corpus's judge cases: each, or none."""
        named = {case.name: case for case in self.judge_cases if case.name}
        what = f"case set {case_set.name!r}"
        defects = [f"{what} lists judge case {k!r} twice" for k in _repeated(case_set.judge_cases)]
        defects += [
            f"{what} names judge case {k!r}, which the corpus does not carry"
            for k in case_set.judge_cases
            if k not in named
        ]
        defects += [
            f"{what} is template {case_set.template!r}'s, and judge case {k!r} is frozen into {named[k].judge_template!r}"
            for k in case_set.judge_cases
            if k in named and named[k].judge_template != case_set.template
        ]
        return defects


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
    #: contradict, per doc type — a non-archived judge config for a dim the store has an active config for, and a
    #: template case whose template the store holds as a judge template. Keys rather than a count, because the
    #: operator resolves each one.
    conflicted: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys NOT written because an operator deleted them (a tombstone), per doc type — rubric
    #: dims and judge configs. Reported rather than counted as present: the slot is empty by decision.
    deleted: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys NOT written because writing them would change the content of a definition already stored, per doc
    #: type: a template case or judge case whose template's current revision already holds cases (it would join the
    #: cases runs were measured against, under the same revision id), and a case set listing such a case. Reported on
    #: every boot until ``reset_definitions`` writes the template's next revision with every corpus case of it.
    needs_reset: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys NOT written because the template they belong to is retired, per doc type: a template case whose
    #: template has no live revision (every revision archived). Archiving is how an operator says a template is gone,
    #: so it is absent to the seed, as an archived rubric dim or judge config is: nothing is written into any of its
    #: revisions, and it is neither counted as created, skipped nor failed. Reported on every boot.
    archived: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys of stored cases whose EMPTY ``name`` the seed filled from the corpus case of the same content, per
    #: doc type (``template_case``). Fill-only: a stored non-empty name is never overwritten.
    named: dict[str, list[str]] = field(default_factory=dict)
    #: Natural keys of stored nameless cases whose name fill the store refused (``StorageError``), each with the
    #: store's reason, per doc type (``template_case``). The case stays nameless, the refusal is logged at WARNING, and
    #: the next boot tries again; until one lands, the case is called by its id.
    name_fill_failed: dict[str, dict[str, str]] = field(default_factory=dict)
    #: Natural keys of stored nameless cases whose conditional name fill lost to another writer (``ConflictError``: the
    #: case was written between the seed's read and its write), per doc type (``template_case``). Nothing was undone and
    #: nothing is wrong: the next boot reads the case again and fills its name if it is still empty.
    name_fill_contended: dict[str, list[str]] = field(default_factory=dict)

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
    def total_deleted(self) -> int:
        """Definitions withheld because an operator deleted their key."""
        return sum(len(keys) for keys in self.deleted.values())

    @property
    def total_needs_reset(self) -> int:
        """Definitions withheld because only a reset can write them: they would change a stored definition."""
        return sum(len(keys) for keys in self.needs_reset.values())

    @property
    def total_archived(self) -> int:
        """Definitions withheld because their template is retired (every revision archived)."""
        return sum(len(keys) for keys in self.archived.values())

    @property
    def total_named(self) -> int:
        """Stored cases whose empty name the seed filled from the corpus."""
        return sum(len(keys) for keys in self.named.values())

    @property
    def total_name_fill_failed(self) -> int:
        """Stored nameless cases whose name fill the store refused."""
        return sum(len(keys) for keys in self.name_fill_failed.values())

    @property
    def total_name_fill_contended(self) -> int:
        """Stored nameless cases whose name fill lost to another writer."""
        return sum(len(keys) for keys in self.name_fill_contended.values())

    @property
    def nothing_to_do(self) -> bool:
        """True when the pass neither wrote, skipped, failed nor withheld anything.

        That is not the reassuring case it resembles: with a corpus present, every
        definition is created, skipped, failed, withheld or deleted, so all-zero means the corpus itself was
        empty or unreachable — the state seeding exists to prevent, reported by the caller
        as a warning rather than the routine "already present" line.
        """
        return not (
            self.total_created
            or self.total_skipped
            or self.total_failed
            or self.total_conflicted
            or self.total_deleted
            or self.total_needs_reset
            or self.total_archived
            or self.total_named
            or self.total_name_fill_failed
            or self.total_name_fill_contended
        )

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
        for doc_type in sorted(
            set(self.created)
            | set(self.skipped)
            | set(self.failed)
            | set(self.conflicted)
            | set(self.deleted)
            | set(self.needs_reset)
            | set(self.archived)
            | set(self.named)
            | set(self.name_fill_failed)
            | set(self.name_fill_contended)
        ):
            part = f"{doc_type}: {self.created.get(doc_type, 0)} created"
            if keys := self.created_keys.get(doc_type):
                part += f" ({', '.join(keys)})"
            part += f", {self.skipped.get(doc_type, 0)} already present"
            if failed := self.failed.get(doc_type, 0):
                part += f", {failed} REFUSED BY STORAGE"
            if conflicted := self.conflicted.get(doc_type):
                part += f", {len(conflicted)} NOT WRITTEN, CONFLICTING WITH THE STORE ({', '.join(conflicted)})"
            if deleted := self.deleted.get(doc_type):
                part += f", {len(deleted)} not written, deleted by an operator ({', '.join(deleted)})"
            if needs_reset := self.needs_reset.get(doc_type):
                part += (
                    f", {len(needs_reset)} NOT WRITTEN, A STORED DEFINITION WOULD CHANGE: RESET IT "
                    f"({', '.join(needs_reset)})"
                )
            if archived := self.archived.get(doc_type):
                part += f", {len(archived)} not written, its template is archived ({', '.join(archived)})"
            if named := self.named.get(doc_type):
                part += f", {len(named)} nameless, named from the corpus ({', '.join(named)})"
            if fill_failed := self.name_fill_failed.get(doc_type):
                reasons = ", ".join(f"{key}: {reason}" for key, reason in fill_failed.items())
                part += f", {len(fill_failed)} NAMELESS, NAMING REFUSED BY STORAGE ({reasons})"
            if contended := self.name_fill_contended.get(doc_type):
                part += (
                    f", {len(contended)} nameless, naming lost to another writer, retried next boot "
                    f"({', '.join(contended)})"
                )
            parts.append(part)
        return "; ".join(parts)


def _load_dir[M: BaseModel](
    directory: Path, model: type[M], scoped: Mapping[str, str], *, named_by_file: bool = False
) -> tuple[M, ...]:
    """Construct every ``*.json`` in ``directory`` through ``model``, with ``scoped``'s fields added to each.

    ``scoped`` is ``{"scope_id": ...}`` for a shape that carries a scope and empty for one that carries none: the
    caller knows the shape, so the scope is passed through as an opaque value and never inspected here. With
    ``named_by_file``, a document that names itself no ``name`` is named by its file, without ``.json``.

    Sorted so a corpus loads in a stable order and a seeding log is diffable between boots.

    :raises ValueError: A file is not valid JSON, or is not a valid document for ``model``.
        Raised rather than skipped: a definition that silently fails to load is a
        measurement instrument missing a dimension nobody notices is gone.
    """
    if not directory.is_dir():
        return ()
    loaded: list[M] = []
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if named_by_file and isinstance(document, dict) and document.get("name") is None:
                document = {**document, "name": path.stem}
            loaded.append(model(**document, **scoped))
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise ValueError(f"seed definition {path} is not a valid {model.__name__}: {e}") from e
    return tuple(loaded)


def load_seed_corpus(seed_dir: Path, scope_id: str) -> SeedCorpus:
    """Read a corpus off disk, constructing each document through its model in ``scope_id``.

    The directory is the host's to name. The run package ships the seeder, not any host's
    definitions, so it holds no path to a corpus: a host with authored definitions passes
    their directory here.

    :param seed_dir: A directory holding ``templates/``, ``rubric_dims/``, ``judge_configs/``, ``template_cases/``,
        ``judge_cases/`` and ``case_sets/`` subdirectories of ``*.json`` documents (a template case as
        :class:`SeedTemplateCase` and a judge case as :class:`SeedJudgeCase`, each named by its file unless it
        names itself, and a case set as :class:`SeedCaseSet`; none of the three carries a scope). A missing
        subdirectory loads as empty.
    :ptype seed_dir: Path
    :param scope_id: The scope the definitions are built in. A file that names a ``scope_id`` of
        its own does not construct — the field is passed twice — because the scope is the
        host's to choose at load, not the file's.
    :ptype scope_id: str
    :return: The corpus. Empty tuples where a subdirectory is absent or holds no ``*.json``.
    :rtype: SeedCorpus
    :raises FileNotFoundError: ``seed_dir`` itself is not a directory. A host naming a corpus that
        is not there has a wrong path, not an empty corpus, and loading it as empty would
        seed nothing with nothing in the log to say why.
    :raises ValueError: A file in the corpus does not construct. This is deliberately loud —
        it means the host ships a definition the server would refuse.
    """
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"eval seed corpus directory {seed_dir} does not exist or is not a directory")
    scoped = {"scope_id": scope_id}
    return SeedCorpus(
        scope_id=scope_id,
        templates=_load_dir(seed_dir / "templates", EvalTemplate, scoped),
        rubric_dims=_load_dir(seed_dir / "rubric_dims", CatalogRubricDim, scoped),
        judge_configs=_load_dir(seed_dir / "judge_configs", JudgeConfig, scoped),
        judge_cases=_load_dir(seed_dir / "judge_cases", SeedJudgeCase, {}, named_by_file=True),
        case_sets=_load_dir(seed_dir / "case_sets", SeedCaseSet, {}),
        template_cases=_load_dir(seed_dir / "template_cases", SeedTemplateCase, {}, named_by_file=True),
    )


def seed_eval_definitions(host: EvalHost, corpus: SeedCorpus) -> SeedOutcome:
    """Create any corpus definition whose natural key is absent from the corpus's scope.

    Seeding semantics: empty slots only, the store is master once seeded. Occupancy is decided
    against the archived-inclusive queries, so a definition an operator archived stays
    archived rather than being resurrected at the next boot; and a rubric dim key or judge config
    slot an operator deleted carries a tombstone, so it stays deleted and is reported under ``deleted``.

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

    A template's own cases (``corpus.template_cases``) are written into its current revision after the templates,
    each reported under ``template_case`` keyed ``<template>/<name>``: created, already present (its content is
    stored on the revision), withheld under ``needs_reset`` (the revision held cases when the pass began), withheld
    under ``conflicted`` (the store's template of that name is of the judge kind), or failed. A stored nameless case of
    a corpus case's content is given its name, reported under ``named``, or under ``name_fill_failed`` (the store
    refused the write, with its reason) or ``name_fill_contended`` (another writer wrote the case first); either leaves
    it nameless for the next boot to fill.

    Authored judge cases (``corpus.judge_cases``) are built through ``author_judge_case`` with
    the templates, before anything is written, and frozen into their judge templates after the
    templates are written; each is reported under ``judge_case``, keyed
    ``<judge template>/<test case id>``. Then each case set (``corpus.case_sets``) whose name the scope
    holds no version of is minted over its cases' stored ids through ``mint_case_set``, the path
    ``case_set_mint`` takes, reported under ``case_set`` keyed by its label (``<name> v1``); one whose
    cases did not all land (a write the store refused) is counted failed and minted on the next boot.

    :param host: The host: the store being seeded, the profile a template is held to, and the template checks
        ``create_template`` reads (``template_checks``).
    :ptype host: EvalHost
    :param corpus: The definitions to seed and the scope they are seeded into, which the host
        loads (:func:`load_seed_corpus` over its own directory) or builds in memory.
        Required rather than defaulted, so the seeder never decides whose definitions a
        scope receives.
    :ptype corpus: SeedCorpus
    :return: Per-doc-type created, skipped and failed counts, the natural keys written, the keys
        withheld as conflicting with the store, the keys withheld as deleted, and the keys withheld because only a
        reset can write them (``needs_reset``: a template case or judge case for a revision already holding cases, and
        a case set listing one).
    :rtype: SeedOutcome
    :raises ValidationFailedError: A corpus template fails a gate ``create_template`` applies, or a
        corpus judge case one ``judge_case_author`` applies (a template it names that neither the
        store nor the corpus holds, a judge template of another kind, a criterion the corpus's template lacks, a verdict its scale cannot hold). Raised before anything is written, naming
        every refused definition and why.
    :raises ValidationFailedError: (also) A corpus template case the host's check
        (``TemplateChecks.refuse_inadmissible_template_case``) refuses, or one with no name; raised before anything is
        written, with the host's reason.
    :raises RefusedInput: The host supplies no template checks.
    """
    checks = host.checks_for("the definition seed")
    templates: list[EvalTemplate] = []
    refusals: list[str] = []
    # A dimension the corpus declares "configured" is honoured by a config the same seed writes, or by one the store
    # already holds: the corpus's own active configs first (it holds at most one per dim), then the store's.
    seeded_configs = {config.rubric_dim_id: config.id for config in corpus.judge_configs if not config.archived}
    stored_config = active_judge_config_in(host.storage, corpus.scope_id)

    def active_judge_config(dim: str) -> str | None:
        return seeded_configs.get(dim) or stored_config(dim)

    for template in corpus.templates:
        try:
            templates.append(
                admit_template(template, profile=host.profile, checks=checks, active_judge_config=active_judge_config)
            )
        except ValidationFailedError as refused:
            refusals.append(f"{template.name!r}: {refused}")
    if refusals:
        raise ValidationFailedError(
            f"the seed corpus for scope {corpus.scope_id!r} carries templates authoring would refuse, so nothing "
            f"was seeded: {'; '.join(refusals)}",
            remedy="correct each template the refusals name in the seed corpus, and seed again",
        )
    case_refusals: list[str] = []
    for own in corpus.template_cases:
        try:
            admit_template_case(own, checks=checks)
        except ValidationFailedError as refused:
            case_refusals.append(f"{own.template}/{own.name}: {refused}")
    if case_refusals:
        raise ValidationFailedError(
            f"the seed corpus for scope {corpus.scope_id!r} carries template cases the host refuses, so nothing "
            f"was seeded: {'; '.join(case_refusals)}",
            remedy="correct each case the refusals name in the seed corpus, and seed again",
        )

    storage = host.storage
    stored_templates = storage.query_templates(corpus.scope_id)
    # A template is resolved in the store first (the store is master), as its current revision.
    stored_first = {**{t.name: t for t in templates}, **current_revisions(stored_templates)}
    # A case whose criterion the stored template lacks is read as the corpus words it, to name it, and withheld.
    drifted = [c for c in corpus.judge_cases if criterion_missing_from_stored(c, stored_first)]
    to_reset = authored_seed_cases(corpus, {**stored_first, **{t.name: t for t in templates}}, cases=drifted)
    cases = authored_seed_cases(corpus, stored_first, cases=[c for c in corpus.judge_cases if c not in drifted])
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
        {(t.name,) for t in stored_templates},
        lambda t: (t.name,),
        lambda t: storage.save_template(t.model_copy(update={"seed_digest": template_digest(t)})),
    )
    _seed_doc_type(
        outcome,
        "rubric_dim",
        corpus.rubric_dims,
        {(d.key,) for d in storage.query_rubric_dims(corpus.scope_id)},
        lambda d: (d.key,),
        lambda d: storage.save_rubric_dim(d.model_copy(update={"seed_digest": rubric_dim_digest(d)})),
        deleted={(t.key,) for t in storage.query_rubric_dim_tombstones(corpus.scope_id)},
    )
    _seed_doc_type(
        outcome,
        "judge_config",
        corpus.judge_configs,
        {_config_key(c) for c in existing_configs},
        _config_key,
        lambda c: storage.save_judge_config(c.model_copy(update={"seed_digest": judge_config_digest(c)})),
        # The corpus holds at most one non-archived config per dim (SeedCorpus refuses more), so the
        # store's active configs are the only ones a write could contradict.
        conflicts=lambda c: not c.archived and c.rubric_dim_id in dims_with_an_active_config,
        deleted={(t.rubric_dim_id, t.name) for t in storage.query_judge_config_tombstones(corpus.scope_id)},
    )
    if corpus.template_cases:
        _seed_template_cases(outcome, host, corpus.scope_id, corpus.template_cases)
    frozen = _seed_judge_cases(outcome, host, corpus.scope_id, cases, drifted=to_reset) if cases or to_reset else {}
    if corpus.case_sets:
        _seed_case_sets(
            outcome, host, corpus, frozen, {case.name: authored for case, authored in (*cases, *to_reset) if case.name}
        )

    return outcome


def criterion_missing_from_stored(case: SeedJudgeCase, stored_first: Mapping[str, EvalTemplate]) -> bool:
    """Whether the template the seed resolves for ``case``'s criterion lacks that criterion's rubric dimension.

    The seed resolves a template in the store first, so a stored template re-worded under the same name no longer words
    a corpus judge case's criterion. That is drift one ``reset_definitions`` repairs, not a corpus error, so the seed
    withholds such a case under ``needs_reset`` rather than refusing the corpus, and the drift read reports it the
    same way.

    :param case: The corpus judge case.
    :ptype case: SeedJudgeCase
    :param stored_first: The template each name resolves to, the store's current revision before the corpus's.
    :ptype stored_first: Mapping[str, EvalTemplate]
    :return: True when the criterion's template resolves and has no dimension named for the criterion.
    :rtype: bool
    """
    subject = stored_first.get(case.criterion_template)
    return subject is not None and all(dim.name != case.criterion for dim in subject.rubric)


def authored_seed_cases(
    corpus: SeedCorpus, by_name: Mapping[str, EvalTemplate], *, cases: Sequence[SeedJudgeCase] | None = None
) -> list[tuple[SeedJudgeCase, AuthoredJudgeCase]]:
    """Every corpus judge case built through ``author_judge_case``, before anything is written; refused as a whole.

    :param corpus: The corpus whose judge cases to build.
    :ptype corpus: SeedCorpus
    :param by_name: The template each name resolves to. The seed resolves one in the store first (the store is master:
        a corpus template whose name the store holds is never written), as its current revision, and else among
        the corpus's admitted templates, which the same seed writes; a drift read and a reset resolve one in the
        corpus first, since they read and write what the corpus carries.
    :ptype by_name: Mapping[str, EvalTemplate]
    :param cases: The corpus judge cases to build; ``None`` builds every one.
    :ptype cases: Sequence[SeedJudgeCase] | None
    :raises ValidationFailedError: A case names a template neither holds, a judge template not of the judge kind, or a
        criterion or verdict ``judge_case_author`` would refuse; naming every such case.
    """
    built: list[tuple[SeedJudgeCase, AuthoredJudgeCase]] = []
    refusals: list[str] = []
    for case in corpus.judge_cases if cases is None else cases:
        where = f"the judge case on {case.criterion!r} of {case.criterion_template!r} in {case.judge_template!r}"
        judge = by_name.get(case.judge_template)
        subject = by_name.get(case.criterion_template)
        missing = [
            name for name, found in ((case.judge_template, judge), (case.criterion_template, subject)) if not found
        ]
        if missing:
            refusals.append(f"{where}: no template named {', '.join(map(repr, missing))} in the store or the corpus")
            continue
        assert judge is not None and subject is not None
        if judge.candidate_kind != JUDGE_KIND:
            refusals.append(f"{where}: template {judge.name!r} is of kind {judge.candidate_kind!r}, not {JUDGE_KIND!r}")
            continue
        try:
            built.append((case, case.authored(subject)))
        except ValidationFailedError as refused:
            refusals.append(f"{where}: {refused}")
    if refusals:
        raise ValidationFailedError(
            f"the seed corpus for scope {corpus.scope_id!r} carries judge cases authoring would refuse, so nothing "
            f"was seeded: {'; '.join(refusals)}",
            remedy="correct each judge case the refusals name in the seed corpus, and seed again",
        )
    return built


def _seed_template_cases(
    outcome: SeedOutcome, host: EvalHost, scope_id: str, cases: Sequence[SeedTemplateCase]
) -> None:
    """Write each template case into its template's current revision, recording it under ``template_case``: created,
    already present (the revision holds its content), withheld (``needs_reset``, or ``conflicted`` for a stored judge
    template of that name), absent (``archived``: the template's every revision is archived, so an operator retired
    it, and nothing is written into any revision of it), or failed (the store refused it, or its template, written by this seed, is not stored).
    Before that, a stored nameless case of its content is named after it (:func:`_fill_empty_name`), recorded under
    ``named``, ``name_fill_failed`` or ``name_fill_contended``.

    **A case joins only an empty revision**, as a judge case joins only an empty battery and for the same reason: a
    template's live cases are what a launch of it runs, so a case added to a revision already holding cases would
    change, under the same revision id, what runs before and after this boot were measured against.
    """
    storage = host.storage
    created_keys: list[str] = []
    withheld: list[str] = []
    conflicted: list[str] = []
    archived: list[str] = []
    named: list[str] = []
    fill_failed: dict[str, str] = {}
    fill_contended: list[str] = []
    skipped = failed = 0
    held: dict[str, set[str]] = {}
    for case in cases:
        key = f"{case.template}/{case.name}"
        template = storage.load_template_by_name(case.template, scope_id)
        if template is None:
            # Its corpus template's write was refused (counted under eval_template): the next boot retries both.
            failed += 1
            continue
        if template.archived:
            # `load_template_by_name` falls back to the newest archived revision when every revision is archived: the
            # operator retired the template, so it is absent here and nothing is written into any revision of it.
            archived.append(key)
            continue
        if template.candidate_kind == JUDGE_KIND:
            # The store is master, and its template of that name is a judge's, whose cases are judge cases.
            conflicted.append(key)
            continue
        match _fill_empty_name(storage, scope_id, template, case):
            case _NameFilled():
                named.append(key)
            case _NameFillRefused(reason=reason):
                fill_failed[key] = reason
            case _NameFillContended():
                fill_contended.append(key)
        if template.id not in held:
            # Read once per revision, before this pass writes into it: what it held when the boot began.
            held[template.id] = {
                stored_template_case_digest(c) for c in storage.query_test_cases(scope_id, template_id=template.id)
            }
        if case.digest() in held[template.id]:
            skipped += 1
            continue
        if held[template.id]:
            withheld.append(key)
            continue
        try:
            storage.save_test_case(case.test_case(template))
        except StorageError:
            failed += 1
            continue
        created_keys.append(key)
    outcome.created["template_case"] = len(created_keys)
    outcome.created_keys["template_case"] = created_keys
    outcome.skipped["template_case"] = skipped
    outcome.failed["template_case"] = failed
    if withheld:
        outcome.needs_reset["template_case"] = withheld
    if conflicted:
        outcome.conflicted["template_case"] = conflicted
    if archived:
        outcome.archived["template_case"] = archived
    if named:
        outcome.named["template_case"] = named
    if fill_failed:
        outcome.name_fill_failed["template_case"] = fill_failed
    if fill_contended:
        outcome.name_fill_contended["template_case"] = fill_contended


@dataclass(frozen=True)
class _NameFilled:
    """A stored nameless case was given the corpus case's name."""


@dataclass(frozen=True)
class _NameFillRefused:
    """The store refused the name fill, for ``reason`` (its error's message)."""

    reason: str


@dataclass(frozen=True)
class _NameFillContended:
    """The conditional name fill lost to another writer of the case."""


def _fill_empty_name(
    storage: EvalStorage, scope_id: str, template: EvalTemplate, case: SeedTemplateCase
) -> _NameFilled | _NameFillRefused | _NameFillContended | None:
    """Name the stored, live, nameless case of ``template`` whose content is ``case``'s, after it; what came of it.

    Fill-only: a stored case holding any name keeps it, and a name another live case of the revision already carries
    is not given twice. Content is compared as the case's id and ``content_hash`` are derived (stratum left out), so
    the write changes neither identity nor digest: only ``name`` is written, and conditionally on the token read with
    the case, so a write landing in between is never undone.

    :param storage: The store.
    :ptype storage: EvalStorage
    :param scope_id: The scope.
    :ptype scope_id: str
    :param template: The template revision the case belongs to.
    :ptype template: EvalTemplate
    :param case: The corpus case.
    :ptype case: SeedTemplateCase
    :return: Whether a case was named, the store refused the write (logged at WARNING with its reason), or the write
        lost to another writer (logged at INFO: the next boot reads the case again); ``None`` when no stored case is
        nameless with ``case``'s content, or its name is already carried.
    :rtype: _NameFilled | _NameFillRefused | _NameFillContended | None
    """
    stored = [c for c in storage.query_test_cases(scope_id, template_id=template.id) if not c.archived]
    if any(c.name == case.name for c in stored):
        return None
    content = template_case_content_digest(case)
    for candidate in stored:
        if candidate.name or template_case_content_digest(candidate) != content:
            continue
        current, etag = storage.load_test_case_with_etag(candidate.id, scope_id)
        if current is None or current.name:
            continue
        try:
            storage.save_test_case(current.model_copy(update={"name": case.name}), if_match=etag)
        except ConflictError as lost:
            # NOSILENT: logged and recorded under name_fill_contended, which the summary names; the next boot retries.
            log.info(
                "eval.seed template_case=%s/%s scope=%s case=%s name fill lost to another writer: %s",
                case.template,
                case.name,
                scope_id,
                candidate.id,
                lost,
            )
            return _NameFillContended()
        except StorageError as refused:
            # NOSILENT: logged and recorded under name_fill_failed, with the store's reason; the next boot retries.
            log.warning(
                "eval.seed template_case=%s/%s scope=%s case=%s name fill refused by storage: %s",
                case.template,
                case.name,
                scope_id,
                candidate.id,
                refused,
            )
            return _NameFillRefused(reason=refused.message)
        return _NameFilled()
    return None


def _seed_judge_cases(
    outcome: SeedOutcome,
    host: EvalHost,
    scope_id: str,
    cases: Sequence[tuple[SeedJudgeCase, AuthoredJudgeCase]],
    *,
    drifted: Sequence[tuple[SeedJudgeCase, AuthoredJudgeCase]] = (),
) -> dict[str, str]:
    """Freeze each authored case into its judge template as ``judge_case_author`` does, recording it under
    ``judge_case``: created, already present (the same case is stored), withheld (``needs_reset``), or failed (the
    store refused it, or its judge template, written by this seed, is not stored).

    **A case joins only an empty battery.** A judge template's live cases are what a launch of it runs, so a case added
    to a revision that already holds cases would change, under the same revision id, what runs before and after this
    boot were measured against, and their results would pool. A case is therefore written only into a revision that
    held no cases when this pass began (one this seed just created, or an empty one), and is otherwise withheld under
    ``needs_reset``, keyed ``<judge template>/<corpus name>``: ``reset_definitions`` writes the template's next
    revision with the whole battery.

    :param drifted: Cases built as the corpus words them whose criterion the stored template lacks
        (:func:`criterion_missing_from_stored`). None is written: one the judge template's battery already holds is
        ``already present``, and any other is withheld under ``needs_reset``, since only a reset can word it.
    :ptype drifted: Sequence[tuple[SeedJudgeCase, AuthoredJudgeCase]]
    :return: Each named case that is stored now, created or already present, by its corpus name: its test case id.
    :rtype: dict[str, str]
    """
    storage = host.storage
    created_keys: list[str] = []
    withheld: list[str] = []
    stored: dict[str, str] = {}
    skipped = failed = 0
    held: dict[str, dict[str, str]] = {}
    for case, authored in (*cases, *drifted):
        judge = storage.load_template_by_name(case.judge_template, scope_id)
        if judge is None:
            # Its corpus template's write was refused (counted under eval_template): the next boot retries both.
            failed += 1
            continue
        if judge.id not in held:
            # Read once per revision, before this pass writes into it: what it held when the boot began.
            held[judge.id] = {
                c.content_hash or "": c.id for c in storage.query_test_cases(scope_id, template_id=judge.id)
            }
        digest = JudgeCase.from_authored(authored).content_digest()
        if (case, authored) in drifted:
            if digest in held[judge.id]:
                skipped += 1
                if case.name is not None:
                    stored[case.name] = held[judge.id][digest]
            else:
                withheld.append(f"{case.judge_template}/{case.name or digest}")
            continue
        if held[judge.id] and digest not in held[judge.id]:
            withheld.append(f"{case.judge_template}/{case.name or digest}")
            continue
        try:
            frozen = freeze_seeded_judge_case(
                storage, template=judge, scope_id=scope_id, authored=authored, seed_name=case.name
            )
        except StorageError:
            failed += 1
            continue
        if case.name is not None:
            stored[case.name] = frozen.test_case_id
        if frozen.created:
            created_keys.append(f"{case.judge_template}/{frozen.test_case_id}")
        else:
            skipped += 1
    outcome.created["judge_case"] = len(created_keys)
    outcome.created_keys["judge_case"] = created_keys
    outcome.skipped["judge_case"] = skipped
    outcome.failed["judge_case"] = failed
    if withheld:
        outcome.needs_reset["judge_case"] = withheld
    return stored


def _seed_case_sets(
    outcome: SeedOutcome,
    host: EvalHost,
    corpus: SeedCorpus,
    frozen: Mapping[str, str],
    authored: Mapping[str, AuthoredJudgeCase],
) -> None:
    """Mint each corpus case set whose name the scope holds no version of, recording it under ``case_set``: created
    (as ``<name> v<version>``), already present (some version is stored), or failed (a case of it, or its judge
    template, did not land, or the store refused the set; the next boot retries it).

    Every case a set lists was refused at build unless the corpus carries it, frozen into the set's template, so a
    set reaching here names only cases this seed froze, or failed to.
    """
    storage = host.storage
    created_keys: list[str] = []
    withheld = set(outcome.needs_reset.get("judge_case", []))
    needs_reset: list[str] = []
    skipped = failed = 0
    for case_set in corpus.case_sets:
        if storage.query_case_sets(corpus.scope_id, name=case_set.name):
            skipped += 1
            continue
        if any(f"{case_set.template}/{name}" in withheld for name in case_set.judge_cases):
            # It lists a case only a reset can write, so it is minted with that reset.
            needs_reset.append(case_set.name)
            continue
        judge = storage.load_template_by_name(case_set.template, corpus.scope_id)
        if judge is None or any(name not in frozen for name in case_set.judge_cases):
            failed += 1
            continue
        try:
            minted = mint_case_set(
                storage,
                scope_id=corpus.scope_id,
                name=case_set.name,
                template_id=judge.id,
                test_case_ids=[frozen[name] for name in case_set.judge_cases],
                tracked=case_set.tracked,
                seed_digest=seed_case_set_digest(case_set, authored),
            )
        except (StorageError, ConflictError) as refused:
            # NOSILENT: logged and counted failed, which the seed's summary names; the next boot retries an empty slot.
            log.warning("eval.seed case_set=%s scope=%s not minted: %s", case_set.name, corpus.scope_id, refused)
            failed += 1
            continue
        created_keys.append(minted.ref.label)
    outcome.created["case_set"] = len(created_keys)
    outcome.created_keys["case_set"] = created_keys
    outcome.skipped["case_set"] = skipped
    outcome.failed["case_set"] = failed
    if needs_reset:
        outcome.needs_reset["case_set"] = needs_reset


def seed_case_set_digest(case_set: SeedCaseSet, authored: Mapping[str, AuthoredJudgeCase]) -> str:
    """The digest of a corpus case set's content: its name, template, ``tracked`` and each listed case's content.

    :param case_set: The corpus set.
    :ptype case_set: SeedCaseSet
    :param authored: Each corpus judge case the set may list, built, by its corpus name.
    :ptype authored: Mapping[str, AuthoredJudgeCase]
    """
    return case_set_digest(
        name=case_set.name,
        template=case_set.template,
        tracked=case_set.tracked,
        cases=[JudgeCase.from_authored(authored[name]).content_digest() for name in case_set.judge_cases],
    )


def _seed_doc_type[D](
    outcome: SeedOutcome,
    doc_type: str,
    definitions: Sequence[D],
    occupied: set[tuple[str, ...]],
    natural_key: Callable[[D], tuple[str, ...]],
    save: Callable[[D], None],
    *,
    conflicts: Callable[[D], bool] = lambda _definition: False,
    deleted: set[tuple[str, ...]] | None = None,
) -> None:
    """Write one doc type's definitions into their empty slots, recording the counts on ``outcome``.

    :param outcome: The seed's running tally, written under ``doc_type``.
    :ptype outcome: SeedOutcome
    :param doc_type: The doc type the counts are recorded under.
    :ptype doc_type: str
    :param definitions: That type's definitions from the corpus.
    :ptype definitions: Sequence[D]
    :param occupied: The natural keys already present.
    :ptype occupied: set[tuple[str, ...]]
    :param natural_key: A definition's slot. Logged joined with ``/``.
    :ptype natural_key: Callable[[D], tuple[str, ...]]
    :param save: The storage write for this type.
    :ptype save: Callable[[D], None]
    :param conflicts: Whether writing an unoccupied definition would contradict a live record; such a
        definition is not written and is recorded under ``conflicted``.
    :ptype conflicts: Callable[[D], bool]
    :param deleted: Natural keys an operator deleted (tombstoned); an unoccupied definition under one
        is not written and is recorded under ``deleted``.
    :ptype deleted: set[tuple[str, ...]] | None
    """
    created_keys: list[str] = []
    conflicted: list[str] = []
    withheld_deleted: list[str] = []
    skipped = failed = 0
    for definition in definitions:
        key = natural_key(definition)
        # SeedCorpus refuses two definitions under one key, so `occupied` is only ever the store's.
        if key in occupied:
            skipped += 1
            continue
        if deleted and key in deleted:
            withheld_deleted.append("/".join(key))
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
    if withheld_deleted:
        outcome.deleted[doc_type] = withheld_deleted


__all__ = [
    "SeedCaseSet",
    "SeedCorpus",
    "SeedJudgeCase",
    "SeedOutcome",
    "SeedTemplateCase",
    "load_seed_corpus",
    "seed_eval_definitions",
]
