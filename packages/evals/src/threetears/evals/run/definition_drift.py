"""Whether each seeded definition is the seed corpus's, and bringing the ones that are behind up to date.

The boot seed is fill-only (:mod:`~threetears.evals.run.definition_seed`): once a definition is stored, the store is
master, so a corpus edit never reaches a store that holds the definition. Prompts solve the same problem with a drift
read and a reset (:mod:`threetears.evals.kernel.prompts.drift`); this is the same pair for seeded definitions —
templates, rubric dims, judge configs, a template's own cases, judge cases and case sets — and it is never run at
boot. A release step, or an operator, runs it deliberately.

**A drift read** (:func:`definition_drift`) reads every corpus definition against the scope's stored one, and
writes and spends nothing:

- ``current``: the stored content is the corpus's;
- ``stale``: the stored content is what was last seeded, and the corpus now differs;
- ``customized``: the stored content differs from what was seeded (an operator's edit, an archive, a delete);
- ``unknown``: there is no record of what was seeded anywhere, neither stored nor supplied (a row seeded before
  ``.30`` by a host that supplies no ``seeded_digests``, or a template case on a revision whose cases no seed
  recorded), which behaves as customized; a supplied map is the whole of what the host seeded then, so under it a
  judge case or template case nothing stored matches reads ``customized`` when the map names it (seeded, since
  removed) and ``stale`` when it does not (never seeded);
- ``absent``: nothing is stored (the seed fills it).

What was seeded is the record the seed writes (``seed_digest``; a judge case's or template case's corpus name,
``seed_name``, its ``content_hash`` being what was seeded: a template case's content, never its stratum, so one whose
stratum is not the corpus's reads customized), or, for a row seeded before the record existed, the
digest the host supplies (``seeded_digests``, keyed as a reading is: the host computes it from its own history with
:func:`corpus_digests` over the corpus it seeded then). Every digest is of content only
(:mod:`~threetears.evals.run.definition_digests`).

**A reset** (:func:`reset_definitions`) writes the corpus's content for the definitions it is named, and never in
place: a template's new content is a new **revision** (a new record under the same name, ``revision`` one past the
newest), a rubric dim's or judge config's a new record under its key, a case set's a new mint, and the record it
supersedes is archived and stays readable by id. So every run, result and case keeps pointing at the content it ran
against, a launch after the reset reads the new revision (a name resolves to its newest non-archived revision), and
results of two revisions never pool (:mod:`threetears.evals.analysis.revisions`).

- **A template's cases are part of its revision.** They are what a launch of it runs, so cases that changed under one
  template id would pool the old cases' results with the new ones'. A judge case is therefore reset with its judge
  template, and a template case with its template: the new revision holds exactly the corpus's cases for it, written
  afresh, and the corpus's case sets over a judge template are minted again over them. A case set whose cases are all
  current is minted alone. A template the corpus gives no cases (and not of the judge kind) is reset bare: its cases
  are generated, not authored.
- **Refused, all or nothing, before anything is written**, naming every refusal: a definition reading ``customized``
  or ``unknown`` (unless ``overwrite_customized``, as a prompt reset takes it); a judge template, or a template the
  corpus gives cases, whose current revision holds live cases the corpus does not carry (they would not be carried, so
  the same flag); a judge config
  contradicting a live config of another name for its dimension (always: archive one of them); and anything
  authoring would refuse (``admit_template`` under the host's template checks, ``author_judge_case``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ConfigDict, Field

from threetears.evals.kernel.errors import ConflictError, ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost, TemplateChecks
from threetears.evals.kernel.host.kinds import freeze
from threetears.evals.kernel.judge_cases import JUDGE_KIND, AuthoredJudgeCase, JudgeCase
from threetears.evals.run.authoring import admit_template, validated_kind_spec
from threetears.evals.run.case_sets import mint_case_set
from threetears.evals.run.definition_digests import (
    case_set_digest,
    current_revisions,
    judge_config_digest,
    rubric_dim_digest,
    template_case_content_digest,
    template_digest,
)
from threetears.evals.run.definition_seed import (
    SeedCaseSet,
    SeedCorpus,
    SeedJudgeCase,
    SeedTemplateCase,
    authored_seed_cases,
    seed_case_set_digest,
    stored_template_case_digest,
)
from threetears.evals.run.judge_kind import freeze_seeded_judge_case
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    CaseSet,
    CatalogRubricDim,
    EvalTemplate,
    EvalTestCase,
    JudgeConfig,
    utc_now_iso,
)

log = get_logger(__name__)

#: How a stored definition reads against the seed corpus (the module docstring).
DefinitionState = Literal["current", "stale", "customized", "unknown", "absent"]

#: The states a reset writes over only with ``overwrite_customized``: what is stored may be an operator's.
KEPT_STATES: frozenset[str] = frozenset({"customized", "unknown"})

#: The order a drift summary counts states in.
_SUMMARY_ORDER: tuple[DefinitionState, ...] = ("stale", "customized", "unknown", "absent", "current")


class DefinitionReading(EvalBaseModel):
    """How one corpus definition reads against the scope's stored one."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(
        description="The definition, as a reset names it: `<doc type>/<natural key>` — `eval_template/<name>`, "
        "`rubric_dim/<key>`, `judge_config/<dim>/<name>`, `template_case/<template>/<name>`, "
        "`judge_case/<judge template>/<corpus name>` (an unnamed case by its content digest) or `case_set/<name>`."
    )
    definition_type: str = Field(
        description="Its type: `eval_template`, `rubric_dim`, `judge_config`, `template_case`, `judge_case` or "
        "`case_set`."
    )
    state: DefinitionState = Field(description="`current`, `stale`, `customized`, `unknown` or `absent`.")
    seed_digest: str = Field(description="The digest of what the corpus carries now.")
    stored_digest: str = Field(default="", description="The digest of the stored definition; empty when none.")
    seeded_digest: str = Field(
        default="", description="The digest of what was seeded: the stored record's, or the host's; empty for none."
    )
    seeded_source: Literal["recorded", "supplied", ""] = Field(
        default="", description="Where `seeded_digest` came from: the stored record, or the host (`seeded_digests`)."
    )
    stored_id: str = Field(default="", description="The stored record read: its id; empty when none.")
    reason: str = Field(default="", description="Why it reads as it does, where the state alone does not say.")

    @property
    def line(self) -> str:
        """One printable line: the key, its state and its reason."""
        return f"{self.key}: {self.state}" + (f" ({self.reason})" if self.reason else "")


class DefinitionDrift(EvalBaseModel):
    """Every corpus definition's reading in one scope, in corpus order by doc type."""

    model_config = ConfigDict(frozen=True)

    scope_id: str = Field(description="The scope read.")
    readings: list[DefinitionReading] = Field(description="One per corpus definition.")

    def keys(self, *states: DefinitionState) -> list[str]:
        """The keys reading any of ``states``, in reading order: what a reset is handed."""
        return [reading.key for reading in self.readings if reading.state in states]

    def count(self, state: DefinitionState) -> int:
        """How many definitions read ``state``."""
        return sum(1 for reading in self.readings if reading.state == state)

    def summary(self) -> str:
        """One printable line, every state counted: ``drift: 3 stale, 0 customized, 1 unknown, 0 absent, 40 current``."""
        return "drift: " + ", ".join(f"{self.count(state)} {state}" for state in _SUMMARY_ORDER)

    def render(self) -> str:
        """The summary, then a line for every definition that is not current."""
        return "\n".join([self.summary(), *(r.line for r in self.readings if r.state != "current")])


class ResetDefinition(EvalBaseModel):
    """One definition a reset wrote, or found current and left."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(description="The definition, as its reading names it.")
    before: DefinitionState = Field(description="How it read before the reset.")
    written_id: str = Field(default="", description="The record the reset wrote; empty when it wrote none.")
    written: str = Field(
        default="", description="What it wrote, as a reader names it: `revision 2`, `v3`, a new record."
    )
    replaced_id: str = Field(
        default="", description="The record it superseded, now archived and readable by id; empty for none."
    )
    discarded: bool = Field(
        default=False,
        description="Whether a stored copy that read customized or unknown, possibly an operator's, was written over "
        "under overwrite_customized (kept archived, readable by `replaced_id`). False for a definition written fresh, "
        "with nothing stored to replace, and for one written over a stale copy, which is the seed's own.",
    )


class DefinitionReset(EvalBaseModel):
    """What a reset wrote, and how the corpus reads after it."""

    model_config = ConfigDict(frozen=True)

    scope_id: str = Field(description="The scope reset.")
    reset: list[ResetDefinition] = Field(description="Each definition named or reset with one named, in write order.")
    after: DefinitionDrift = Field(description="The drift read after the reset.")

    def render(self) -> str:
        """A printable account: each definition and what it became, then the drift after."""
        lines = []
        for entry in self.reset:
            if not entry.written_id:
                lines.append(f"{entry.key}: {entry.before}, left as it is")
                continue
            line = f"{entry.key}: {entry.before} -> {entry.written} ({entry.written_id})"
            if entry.replaced_id:
                line += f", replacing {entry.replaced_id}{' (discarded)' if entry.discarded else ''}"
            lines.append(line)
        return "\n".join([*lines, self.after.summary()])


def definition_key(doc_type: str, *natural_key: str) -> str:
    """A definition's key, as a reading names it: its doc type and natural key joined by ``/``."""
    return "/".join((doc_type, *natural_key))


def _case_key(case: SeedJudgeCase, digest: str) -> str:
    return definition_key("judge_case", case.judge_template, case.name or digest)


def _newest[R: (CatalogRubricDim, JudgeConfig)](records: Iterable[R]) -> R | None:
    """The record a key resolves to: its newest non-archived one, else its newest."""
    ordered = sorted(records, key=lambda r: (not r.archived, r.created_at, r.id))
    return ordered[-1] if ordered else None


def _state(stored: str, seed: str, seeded: str | None) -> DefinitionState:
    if stored == seed:
        return "current"
    if seeded is None:
        return "unknown"
    return "stale" if stored == seeded else "customized"


@dataclass
class _Survey:
    """One read of the store and the corpus: every reading, and what a reset writes from."""

    host: EvalHost
    corpus: SeedCorpus
    supplied: Mapping[str, str]
    templates: dict[str, list[EvalTemplate]] = field(default_factory=dict)
    current: dict[str, EvalTemplate] = field(default_factory=dict)
    resolved: dict[str, EvalTemplate] = field(default_factory=dict)
    built: dict[str, tuple[SeedJudgeCase, AuthoredJudgeCase, str]] = field(default_factory=dict)
    own: dict[str, tuple[SeedTemplateCase, str]] = field(default_factory=dict)
    cases_on: dict[str, list[EvalTestCase]] = field(default_factory=dict)
    readings: dict[str, DefinitionReading] = field(default_factory=dict)

    @classmethod
    def read(cls, host: EvalHost, corpus: SeedCorpus, supplied: Mapping[str, str] | None) -> _Survey:
        survey = cls(host=host, corpus=corpus, supplied=dict(supplied or {}))
        survey._read()
        return survey

    def _seeded(self, key: str, recorded: str | None) -> tuple[str | None, Literal["recorded", "supplied", ""]]:
        if recorded:
            return recorded, "recorded"
        if key in self.supplied:
            return self.supplied[key], "supplied"
        return None, ""

    def _reading(
        self,
        key: str,
        doc_type: str,
        seed: str,
        stored: str | None,
        stored_id: str,
        recorded: str | None,
        reason: str = "",
    ) -> DefinitionReading:
        if stored is None:
            return DefinitionReading(key=key, definition_type=doc_type, state="absent", seed_digest=seed, reason=reason)
        seeded, source = self._seeded(key, recorded)
        return DefinitionReading(
            key=key,
            definition_type=doc_type,
            state=_state(stored, seed, seeded),
            seed_digest=seed,
            stored_digest=stored,
            seeded_digest=seeded or "",
            seeded_source=source,
            stored_id=stored_id,
            reason=reason,
        )

    def stored_cases(self, template: EvalTemplate) -> list[EvalTestCase]:
        if template.id not in self.cases_on:
            self.cases_on[template.id] = self.host.storage.query_test_cases(
                self.corpus.scope_id, template_id=template.id
            )
        return self.cases_on[template.id]

    def _read(self) -> None:
        storage, scope, profile = self.host.storage, self.corpus.scope_id, self.host.profile
        for template in storage.query_templates(scope):
            self.templates.setdefault(template.name, []).append(template)
        self.current = current_revisions(t for records in self.templates.values() for t in records)
        for template in self.corpus.templates:
            # Stored as admission resolves its spec; a spec the model refuses is digested as written, and a reset of it
            # is refused by admission.
            try:
                spec = freeze(validated_kind_spec(template, profile=profile))
                self.resolved[template.name] = template.model_copy(update={"kind_spec": spec})
            except ValidationFailedError:
                self.resolved[template.name] = template
        # A corpus judge case is read, and reset, as the corpus words it: against the corpus's criterion template
        # where the corpus carries one, else the store's current revision.
        for case, authored in authored_seed_cases(self.corpus, {**self.current, **self.resolved}):
            digest = JudgeCase.from_authored(authored).content_digest()
            self.built[_case_key(case, digest)] = (case, authored, digest)
        for own_case in self.corpus.template_cases:
            self.own[definition_key("template_case", own_case.template, own_case.name)] = (own_case, own_case.digest())

        for template in self.corpus.templates:
            self._read_template(template)
        dims: dict[str, list[CatalogRubricDim]] = {}
        for dim in storage.query_rubric_dims(scope):
            dims.setdefault(dim.key, []).append(dim)
        dim_tombstones = {t.key for t in storage.query_rubric_dim_tombstones(scope)}
        for corpus_dim in self.corpus.rubric_dims:
            key = definition_key("rubric_dim", corpus_dim.key)
            stored_dim = _newest(dims.get(corpus_dim.key, []))
            self.readings[key] = self._record_reading(
                key,
                "rubric_dim",
                rubric_dim_digest(corpus_dim),
                stored_dim,
                rubric_dim_digest,
                deleted=corpus_dim.key in dim_tombstones,
            )
        configs: dict[tuple[str, str], list[JudgeConfig]] = {}
        for config in storage.query_judge_configs(scope):
            configs.setdefault((config.rubric_dim_id, config.name), []).append(config)
        config_tombstones = {(t.rubric_dim_id, t.name) for t in storage.query_judge_config_tombstones(scope)}
        for corpus_config in self.corpus.judge_configs:
            slot = (corpus_config.rubric_dim_id, corpus_config.name)
            key = definition_key("judge_config", *slot)
            self.readings[key] = self._record_reading(
                key,
                "judge_config",
                judge_config_digest(corpus_config),
                _newest(configs.get(slot, [])),
                judge_config_digest,
                deleted=slot in config_tombstones,
            )
        for key, (own_case, digest) in self.own.items():
            self.readings[key] = self._read_template_case(key, own_case, digest)
        for key, (case, _authored, digest) in self.built.items():
            self.readings[key] = self._read_case(key, case, digest)
        latest: dict[str, CaseSet] = {}
        for case_set in storage.query_case_sets(scope):
            if case_set.name not in latest or case_set.version > latest[case_set.name].version:
                latest[case_set.name] = case_set
        for corpus_set in self.corpus.case_sets:
            self.readings[definition_key("case_set", corpus_set.name)] = self._read_case_set(
                corpus_set, latest.get(corpus_set.name)
            )

    def _record_reading[R: (CatalogRubricDim, JudgeConfig)](
        self, key: str, doc_type: str, seed: str, stored: R | None, digest: Callable[[R], str], *, deleted: bool
    ) -> DefinitionReading:
        if stored is None and deleted:
            return DefinitionReading(
                key=key, definition_type=doc_type, state="customized", seed_digest=seed, reason="deleted by an operator"
            )
        if stored is None:
            return self._reading(key, doc_type, seed, None, "", None)
        reason = "archived by an operator" if stored.archived else ""
        return self._reading(key, doc_type, seed, digest(stored), stored.id, stored.seed_digest, reason)

    def _read_template(self, template: EvalTemplate) -> None:
        key = definition_key("eval_template", template.name)
        seed = template_digest(self.resolved[template.name])
        stored = self.current.get(template.name)
        if stored is None:
            self.readings[key] = self._reading(key, "eval_template", seed, None, "", None)
            return
        reason = f"revision {stored.revision}" + (", archived" if stored.archived else "")
        reading = self._reading(
            key, "eval_template", seed, template_digest(stored), stored.id, stored.seed_digest, reason
        )
        if reading.state == "current" and self.holds_cases(template):
            judge = template.candidate_kind == JUDGE_KIND
            if superseded := self.superseded_cases(stored):
                reading = reading.model_copy(
                    update={
                        "state": "stale",
                        "reason": f"{reason}; its {'battery' if judge else 'revision'} holds {len(superseded)} seeded "
                        f"case(s) the corpus has since changed, beside their current {'words' if judge else 'content'}",
                    }
                )
            elif missing := self.missing_cases(stored):
                reading = reading.model_copy(
                    update={
                        "state": "stale",
                        "reason": f"{reason}; its {'battery' if judge else 'revision'} lacks {len(missing)} corpus "
                        f"case(s), which the seed never adds to a {'battery' if judge else 'revision'} holding cases",
                    }
                )
        self.readings[key] = reading

    def holds_cases(self, template: EvalTemplate) -> bool:
        """Whether the corpus states a corpus template's cases: a judge template's battery, or a template's own cases.

        Such a template's revision is written with exactly the corpus's cases for it; any other template's cases are
        generated, not authored, and its revision is written bare.
        """
        return template.candidate_kind == JUDGE_KIND or any(c.template == template.name for c, _ in self.own.values())

    def cases_of(self, template: str) -> list[tuple[str, str | None, str]]:
        """Every corpus case of a template, by name — its judge cases or its own — as (key, corpus name, digest)."""
        judged = [
            (key, case.name, digest)
            for key, (case, _a, digest) in self.built.items()
            if case.judge_template == template
        ]
        own = [(key, case.name, digest) for key, (case, digest) in self.own.items() if case.template == template]
        return judged + own

    @staticmethod
    def case_digest(template: EvalTemplate, case: EvalTestCase) -> str:
        """A stored case's content digest, as a corpus case of its template digests: a judge case's ``content_hash``,
        and a template case's content read off the row (so a stratum relabelled in place moves it)."""
        if template.candidate_kind == JUDGE_KIND:
            return case.content_hash or ""
        return stored_template_case_digest(case)

    def missing_cases(self, template: EvalTemplate) -> list[str]:
        """The keys of the corpus cases a template's revision lacks while it holds cases: only a reset adds them."""
        stored = self.stored_cases(template)
        if not stored:
            return []
        held = {self.case_digest(template, case) for case in stored}
        return [key for key, _name, digest in self.cases_of(template.name) if digest not in held]

    def superseded_cases(self, template: EvalTemplate) -> list[EvalTestCase]:
        """The live cases of a template's revision that were seeded as a corpus case the corpus has changed."""
        cases = self.cases_of(template.name)
        wanted = {digest for _key, _name, digest in cases}
        names = {name for _key, name, _digest in cases if name}
        supplied = {self.supplied[key] for key, _name, _digest in cases if key in self.supplied}
        return [
            stored
            for stored in self.stored_cases(template)
            if not stored.archived
            and (digest := self.case_digest(template, stored)) not in wanted
            and (stored.seed_name in names or digest in supplied)
        ]

    def uncarried_cases(self, template: EvalTemplate) -> list[EvalTestCase]:
        """The live cases of a template's revision the corpus neither carries nor supersedes."""
        carried = {digest for _key, _name, digest in self.cases_of(template.name)}
        superseded = {case.id for case in self.superseded_cases(template)}
        return [
            stored
            for stored in self.stored_cases(template)
            if not stored.archived and self.case_digest(template, stored) not in carried and stored.id not in superseded
        ]

    def built_of(self, judge_template: str) -> list[tuple[str, tuple[SeedJudgeCase, AuthoredJudgeCase, str]]]:
        return [(key, built) for key, built in self.built.items() if built[0].judge_template == judge_template]

    def own_of(self, template: str) -> list[tuple[str, SeedTemplateCase]]:
        return [(key, case) for key, (case, _digest) in self.own.items() if case.template == template]

    def _read_template_case(self, key: str, case: SeedTemplateCase, digest: str) -> DefinitionReading:
        """How a template case reads on its template's current revision: matched there by content, else by the name it
        was seeded under, by the digest the host supplies, or by the name a host's own seed stored it under."""
        template = self.current.get(case.template)
        if template is None:
            return self._reading(key, "template_case", digest, None, "", None)
        where = f"{case.template!r} revision {template.revision}"
        if template.candidate_kind == JUDGE_KIND:
            return DefinitionReading(
                key=key,
                definition_type="template_case",
                state="customized",
                seed_digest=digest,
                reason=f"{where} is a judge template, whose cases are judge cases",
            )
        stored = self.stored_cases(template)
        live = [c for c in stored if not c.archived]
        if match := next((c for c in live if stored_template_case_digest(c) == digest), None):
            return self._reading(key, "template_case", digest, digest, match.id, digest)
        if match := next((c for c in stored if stored_template_case_digest(c) == digest), None):
            return self._reading(key, "template_case", digest, digest, match.id, None).model_copy(
                update={"state": "customized", "reason": "archived by an operator"}
            )
        supplied = self.supplied.get(key)
        predecessor = next((c for c in live if c.seed_name == case.name), None)
        if predecessor is None and supplied is not None:
            predecessor = next((c for c in live if stored_template_case_digest(c) == supplied), None)
        if predecessor is None:
            # A case stored under its name before seeds recorded one (by the host's own seed): no record of what was
            # seeded unless the host supplies it.
            predecessor = next((c for c in live if c.seed_name is None and c.name == case.name), None)
        if predecessor is not None:
            stored_digest = stored_template_case_digest(predecessor)
            if predecessor.seed_name is None:
                return self._reading(key, "template_case", digest, stored_digest, predecessor.id, None, f"stored on {where}")
            # A seeded row records its content (`content_hash`), never its stratum, which is no part of a case's
            # identity. Content as seeded under the corpus's stratum is the row as seeded; under another stratum it was
            # relabelled, which no record says by whom, so it reads customized.
            as_seeded = predecessor.content_hash == template_case_content_digest(predecessor)
            if as_seeded and predecessor.stratum != case.stratum:
                return self._reading(
                    key, "template_case", digest, stored_digest, predecessor.id, predecessor.content_hash
                ).model_copy(update={"state": "customized", "reason": f"seeded into {where}, its stratum relabelled"})
            recorded = stored_digest if as_seeded else predecessor.content_hash
            return self._reading(
                key, "template_case", digest, stored_digest, predecessor.id, recorded, f"seeded into {where}"
            )
        if not stored:
            # An empty revision: the boot seed fills it.
            return self._reading(key, "template_case", digest, None, "", None)
        if not any(c.seed_name for c in stored):
            # Its cases were written before seeds recorded a template's own cases. As for a judge case seeded before
            # .30, the host's map, when it supplies one, is the whole of what it seeded then.
            if supplied is not None:
                return DefinitionReading(
                    key=key,
                    definition_type="template_case",
                    state="customized",
                    seed_digest=digest,
                    seeded_digest=supplied,
                    seeded_source="supplied",
                    reason="seeded as its corpus content, since removed by an operator",
                )
            if not self.supplied:
                return DefinitionReading(
                    key=key,
                    definition_type="template_case",
                    state="unknown",
                    seed_digest=digest,
                    reason=f"{where} holds cases no seed recorded",
                )
            return DefinitionReading(
                key=key,
                definition_type="template_case",
                state="stale",
                seed_digest=digest,
                reason=f"new to the corpus; never seeded into {case.template!r} (the host's seeded digests do not name "
                "it), so only a reset writes it",
            )
        return DefinitionReading(
            key=key,
            definition_type="template_case",
            state="stale",
            seed_digest=digest,
            reason=f"new to the corpus; {where} already holds cases, so only a reset writes it",
        )

    def _read_case(self, key: str, case: SeedJudgeCase, digest: str) -> DefinitionReading:
        judge = self.current.get(case.judge_template)
        if judge is None:
            return self._reading(key, "judge_case", digest, None, "", None)
        stored = self.stored_cases(judge)
        live = [c for c in stored if not c.archived]
        if match := next((c for c in live if c.content_hash == digest), None):
            return self._reading(key, "judge_case", digest, digest, match.id, digest)
        if match := next((c for c in stored if c.content_hash == digest), None):
            return self._reading(key, "judge_case", digest, digest, match.id, None).model_copy(
                update={"state": "customized", "reason": "archived by an operator"}
            )
        predecessor = next((c for c in live if case.name and c.seed_name == case.name), None)
        supplied = self.supplied.get(key)
        if predecessor is None and supplied is not None:
            predecessor = next((c for c in live if c.content_hash == supplied), None)
        if predecessor is not None and predecessor.content_hash:
            recorded = predecessor.content_hash if predecessor.seed_name else None
            return self._reading(
                key,
                "judge_case",
                digest,
                predecessor.content_hash,
                predecessor.id,
                recorded,
                f"seeded into {case.judge_template!r} revision {judge.revision}",
            )
        if not stored:
            # An empty battery: the boot seed fills it.
            return self._reading(key, "judge_case", digest, None, "", None)
        if judge.seed_digest is None:
            # Seeded before the record existed. The host's map, when it supplies one, is the whole of what it seeded
            # then: a case it names that nothing stored matches was seeded and since removed, and one it does not
            # name was never seeded. Only with no map at all is there no record anywhere.
            if supplied is not None:
                return DefinitionReading(
                    key=key,
                    definition_type="judge_case",
                    state="customized",
                    seed_digest=digest,
                    seeded_digest=supplied,
                    seeded_source="supplied",
                    reason="seeded as its corpus content, since removed by an operator",
                )
            if not self.supplied:
                return DefinitionReading(
                    key=key,
                    definition_type="judge_case",
                    state="unknown",
                    seed_digest=digest,
                    reason=f"{case.judge_template!r} was seeded before seeds recorded their cases",
                )
            return DefinitionReading(
                key=key,
                definition_type="judge_case",
                state="stale",
                seed_digest=digest,
                reason=f"new to the corpus; never seeded into {case.judge_template!r} (the host's seeded digests do "
                "not name it), so only a reset writes it",
            )
        # The seed never adds a case to a battery already holding cases (it would change the revision runs were
        # measured against), so only a reset writes it.
        return DefinitionReading(
            key=key,
            definition_type="judge_case",
            state="stale",
            seed_digest=digest,
            reason=f"new to the corpus; {case.judge_template!r} revision {judge.revision} already holds cases, so only "
            "a reset writes it",
        )

    def _read_case_set(self, corpus_set: SeedCaseSet, stored: CaseSet | None) -> DefinitionReading:
        key = definition_key("case_set", corpus_set.name)
        authored = {case.name: built for case, built, _digest in self.built.values() if case.name}
        seed = seed_case_set_digest(corpus_set, authored)
        if stored is None:
            listed = [definition_key("judge_case", corpus_set.template, name) for name in corpus_set.judge_cases]
            if withheld := [k for k in listed if self.readings[k].state == "stale" and not self.readings[k].stored_id]:
                # The seed withholds a set listing a case it withholds (`needs_reset`): only a reset mints it.
                return DefinitionReading(
                    key=key,
                    definition_type="case_set",
                    state="stale",
                    seed_digest=seed,
                    reason=f"lists {len(withheld)} case(s) only a reset writes",
                )
            return self._reading(key, "case_set", seed, None, "", None)
        storage, scope = self.host.storage, self.corpus.scope_id
        template = storage.load_template(stored.template_id, scope)
        held = {
            case.id: case.content_hash or "" for case in storage.load_test_cases_by_ids(stored.test_case_ids, scope)
        }
        stored_digest = case_set_digest(
            name=stored.name,
            template="" if template is None else template.name,
            tracked=stored.tracked,
            cases=[held.get(case_id, "") for case_id in stored.test_case_ids],
        )
        reading = self._reading(key, "case_set", seed, stored_digest, stored.id, stored.seed_digest, stored.ref.label)
        current = self.current.get(corpus_set.template)
        if reading.state == "current" and current is not None and stored.template_id != current.id:
            reading = reading.model_copy(
                update={
                    "state": "stale",
                    "reason": f"{stored.ref.label} lists an older revision of {corpus_set.template!r}'s cases",
                }
            )
        return reading


def definition_drift(
    host: EvalHost, corpus: SeedCorpus, *, seeded_digests: Mapping[str, str] | None = None
) -> DefinitionDrift:
    """Read every definition of a seed corpus against its scope's stored one. Writes nothing, spends nothing.

    Args:
        host: The host: the store read, and the profile a template's spec is resolved by.
        corpus: The corpus, loaded for the scope it reads (:func:`~threetears.evals.run.load_seed_corpus`).
        seeded_digests: What the host seeded before ``.30``, when the store recorded nothing: the digest of each
            definition it seeded then (:func:`corpus_digests` over that corpus), by the key a reading names it by. A
            stored record's own digest wins over it. ``None`` for none, when such rows read ``unknown``.

    Returns:
        One reading per corpus definition: templates, rubric dims, judge configs, template cases, judge cases, case
        sets.

    Raises:
        ValidationFailedError: A corpus judge case cannot be built, as the boot seed refuses it.
    """
    survey = _Survey.read(host, corpus, seeded_digests)
    return DefinitionDrift(scope_id=corpus.scope_id, readings=list(survey.readings.values()))


def corpus_digests(host: EvalHost, corpus: SeedCorpus) -> dict[str, str]:
    """The digest of every definition of a corpus, by its key: what a seed of it records.

    A host whose store was seeded before ``.30`` computes this over the corpus it seeded then (its own history) and
    hands it to :func:`definition_drift` and :func:`reset_definitions` as ``seeded_digests``, so a row nobody edited
    since reads ``stale`` rather than ``unknown``. A template is digested with its spec as the host's kind resolves it
    today, so a row stored under a spec model that has since gained a default reads as differing from it.

    Args:
        host: The host whose profile resolves a template's spec, and whose store resolves a criterion template the
            corpus does not carry.
        corpus: The corpus, loaded for the scope it describes.

    Returns:
        Each definition's key -> its digest.
    """
    return {reading.key: reading.seed_digest for reading in definition_drift(host, corpus).readings}


@dataclass
class _Refusal:
    key: str
    why: str
    kept: bool = False


def _reset_call(keys: Sequence[str]) -> str:
    return f"run.reset_definitions(..., [{', '.join(repr(k) for k in keys)}], overwrite_customized=True)"


def reset_definitions(
    host: EvalHost,
    corpus: SeedCorpus,
    definitions: Sequence[str],
    *,
    overwrite_customized: bool = False,
    seeded_digests: Mapping[str, str] | None = None,
) -> DefinitionReset:
    """Write the corpus's content for the named definitions, each as a new revision or record, never in place.

    A definition already ``current`` is left as it is. A judge case resets with its judge template, whose new
    revision holds the corpus's whole battery for it, and that template's corpus case sets are minted again over the
    new revision's cases; a case set whose cases are all current is minted alone. A template case resets with its
    template, whose new revision holds every corpus case of it (the module docstring).

    Args:
        host: The host: the store written, and the profile and template checks (``template_checks``) a template is
            admitted under, as ``create_template`` admits one.
        corpus: The corpus, loaded for the scope it writes.
        definitions: The keys to reset, as a drift read names them (:meth:`DefinitionDrift.keys`).
        overwrite_customized: Write over a definition reading ``customized`` or ``unknown``, and a judge template's
            live cases the corpus does not carry. What is superseded stays stored, archived, readable by id.
        seeded_digests: As :func:`definition_drift` takes it.

    Returns:
        What was written, and the drift read after.

    Raises:
        ValidationFailedError: A key names no corpus definition, or a template or judge case authoring would refuse;
            nothing was written, every refusal named.
        ConflictError: A definition may be an operator's (customized, unknown, or uncarried live cases) without
            ``overwrite_customized``, or a judge config contradicts a live config of another name for its
            dimension; nothing was written, every refusal named.
        StorageError: A write failed part-way. Every check ran first, so this is the store failing, and the
            definitions written before it stand; the drift read says which.
        RefusedInput: The host supplies no template checks.
    """
    checks = host.checks_for("a definition reset")
    scope = corpus.scope_id
    survey = _Survey.read(host, corpus, seeded_digests)
    readings = survey.readings
    if unknown := [key for key in definitions if key not in readings]:
        raise ValidationFailedError(
            f"no definition of the seed corpus for scope {scope!r} is keyed {', '.join(map(repr, unknown))}",
            remedy="name definitions as a drift read names them (run.definition_drift)",
        )
    corpus_templates = {t.name: t for t in corpus.templates}
    corpus_sets = {s.name: s for s in corpus.case_sets}
    refusals: list[_Refusal] = []
    templates: dict[str, None] = {}
    dims = [d for d in corpus.rubric_dims if definition_key("rubric_dim", d.key) in definitions]
    configs = [
        c for c in corpus.judge_configs if definition_key("judge_config", c.rubric_dim_id, c.name) in definitions
    ]
    sets: dict[str, None] = {}
    asked = [key for key in dict.fromkeys(definitions) if readings[key].state != "current"]

    def with_template(key: str, name: str) -> None:
        if name in corpus_templates:
            templates[name] = None
        else:
            refusals.append(
                _Refusal(key, f"it resets with its judge template {name!r}, which the seed corpus does not carry")
            )

    for key in asked:
        reading = readings[key]
        if reading.definition_type == "eval_template":
            templates[key.removeprefix("eval_template/")] = None
        elif reading.definition_type == "judge_case":
            with_template(key, survey.built[key][0].judge_template)
        elif reading.definition_type == "template_case":
            with_template(key, survey.own[key][0].template)
        elif reading.definition_type == "case_set":
            corpus_set = corpus_sets[key.removeprefix("case_set/")]
            listed = [_listed_key(survey, corpus_set.template, name) for name in corpus_set.judge_cases]
            if corpus_set.template in survey.current and all(readings[k].state == "current" for k in listed):
                sets[corpus_set.name] = None
            else:
                with_template(key, corpus_set.template)
        if reading.state in KEPT_STATES and not overwrite_customized:
            refusals.append(
                _Refusal(
                    key, f"it reads {reading.state}" + (f" ({reading.reason})" if reading.reason else ""), kept=True
                )
            )
    # A template's corpus cases, and a judge template's case sets, are written with it.
    pulled: list[str] = []
    for name in templates:
        if not survey.holds_cases(corpus_templates[name]):
            continue
        pulled += [key for key, _name, _digest in survey.cases_of(name)]
        pulled += [definition_key("case_set", s.name) for s in corpus.case_sets if s.template == name]
        stored = survey.current.get(name)
        if stored is not None and (uncarried := survey.uncarried_cases(stored)) and not overwrite_customized:
            refusals.append(
                _Refusal(
                    definition_key("eval_template", name),
                    f"its revision {stored.revision} holds {len(uncarried)} live case(s) the seed corpus does not carry "
                    f"({', '.join(case.id for case in uncarried)}), which the new revision would not carry",
                    kept=True,
                )
            )
    for key in pulled:
        reading = readings[key]
        if key in asked or reading.state not in KEPT_STATES or overwrite_customized:
            continue
        # A pulled case's own `unknown` is its template's question, which the uncarried cases answer.
        if reading.definition_type == "case_set" or reading.state == "customized":
            refusals.append(
                _Refusal(key, f"it is written with its judge template and reads {reading.state}", kept=True)
            )
    for name in list(sets):
        if corpus_sets[name].template in templates:
            del sets[name]
    # The one-live-config-per-dim rule: a corpus config would supersede an operator's config of another name.
    live = {(c.rubric_dim_id, c.name): c for c in host.storage.query_judge_configs(scope, archived=False)}
    for config in configs:
        if config.archived:
            continue
        if others := sorted(n for (dim, n) in live if dim == config.rubric_dim_id and n != config.name):
            refusals.append(
                _Refusal(
                    definition_key("judge_config", config.rubric_dim_id, config.name),
                    f"dimension {config.rubric_dim_id!r} has the live judge config {', '.join(map(repr, others))} of "
                    "another name, which it would contradict; archive one of them",
                    kept=True,
                )
            )
    admitted, invalid = _admitted(host, corpus, [corpus_templates[name] for name in templates], checks)
    refusals += invalid
    for name in sets:
        corpus_set = corpus_sets[name]
        ids = _current_case_ids(survey, corpus_set)
        versions = host.storage.query_case_sets(scope, name=name)
        if versions and versions[0].test_case_ids == ids:
            refusals.append(
                _Refusal(
                    definition_key("case_set", name),
                    f"{versions[0].ref.label} already lists exactly its cases and differs only in `tracked`, which a new "
                    "version cannot record",
                )
            )
    if refusals:
        _refuse(scope, refusals)

    entries: list[ResetDefinition] = []
    now = utc_now_iso()
    for dim in dims:
        key = definition_key("rubric_dim", dim.key)
        if readings[key].state == "current":
            entries.append(ResetDefinition(key=key, before="current"))
            continue
        prior = [d for d in host.storage.query_rubric_dims(scope) if d.key == dim.key]
        written = dim.model_copy(
            update={"id": _new_id(), "created_at": now, "updated_at": now, "seed_digest": rubric_dim_digest(dim)}
        )
        host.storage.save_rubric_dim(written)
        for record in prior:
            if not record.archived:
                host.storage.save_rubric_dim(record.model_copy(update={"archived": True, "updated_at": now}))
        entries.append(_entry(readings[key], written.id, "a new record"))
    for config in configs:
        key = definition_key("judge_config", config.rubric_dim_id, config.name)
        if readings[key].state == "current":
            entries.append(ResetDefinition(key=key, before="current"))
            continue
        prior_configs = host.storage.query_judge_configs(scope, rubric_dim_id=config.rubric_dim_id)
        written_config = config.model_copy(
            update={"id": _new_id(), "created_at": now, "seed_digest": judge_config_digest(config)}
        )
        # New before old, as re-authoring does, so the dimension is never left with no live config.
        host.storage.save_judge_config(written_config)
        for prior_config in prior_configs:
            if prior_config.name == config.name and not prior_config.archived:
                host.storage.save_judge_config(prior_config.model_copy(update={"archived": True}))
        entries.append(_entry(readings[key], written_config.id, "a new record"))
    for name in templates:
        entries += _reset_template(survey, admitted[name], now)
    for name in sets:
        corpus_set = corpus_sets[name]
        judge = survey.current[corpus_set.template]
        minted = mint_case_set(
            host.storage,
            scope_id=scope,
            name=name,
            template_id=judge.id,
            test_case_ids=_current_case_ids(survey, corpus_set),
            tracked=corpus_set.tracked,
            seed_digest=readings[definition_key("case_set", name)].seed_digest,
        )
        entries.append(_entry(readings[definition_key("case_set", name)], minted.id, minted.ref.label))
    for key in dict.fromkeys(definitions):
        if key not in {entry.key for entry in entries} and readings[key].state == "current":
            entries.append(ResetDefinition(key=key, before="current"))
    for entry in entries:
        if entry.discarded:
            log.warning(
                "eval.reset_definitions scope=%s %s read %s; superseded %s, kept archived",
                scope,
                entry.key,
                entry.before,
                entry.replaced_id or "nothing",
            )
    return DefinitionReset(
        scope_id=scope, reset=entries, after=definition_drift(host, corpus, seeded_digests=seeded_digests)
    )


def _new_id() -> str:
    return str(uuid.uuid7())


def _entry(reading: DefinitionReading, written_id: str, written: str) -> ResetDefinition:
    return ResetDefinition(
        key=reading.key,
        before=reading.state,
        written_id=written_id,
        written=written,
        replaced_id=reading.stored_id,
        # A stored copy was replaced only where one was read: a case the store holds no copy of reads customized or
        # unknown by what the host's record says of it, yet is written fresh.
        discarded=bool(reading.stored_id) and reading.state in KEPT_STATES,
    )


def _listed_key(survey: _Survey, judge_template: str, name: str) -> str:
    return definition_key("judge_case", judge_template, name)


def _current_case_ids(survey: _Survey, corpus_set: SeedCaseSet) -> list[str]:
    """The stored ids of a set's corpus cases on its template's current revision, every one of which reads current."""
    return [
        survey.readings[_listed_key(survey, corpus_set.template, name)].stored_id for name in corpus_set.judge_cases
    ]


def _admitted(
    host: EvalHost, corpus: SeedCorpus, templates: Sequence[EvalTemplate], checks: TemplateChecks
) -> tuple[dict[str, EvalTemplate], list[_Refusal]]:
    """Each template admitted through ``create_template``'s gates, as the seed admits one, or why not."""
    seeded_configs = {config.rubric_dim_id: config.id for config in corpus.judge_configs if not config.archived}
    stored = {
        config.rubric_dim_id: config.id for config in host.storage.query_judge_configs(corpus.scope_id, archived=False)
    }
    admitted: dict[str, EvalTemplate] = {}
    refused: list[_Refusal] = []
    for template in templates:
        try:
            admitted[template.name] = admit_template(
                template,
                profile=host.profile,
                checks=checks,
                active_judge_config=lambda dim: seeded_configs.get(dim) or stored.get(dim),
            )
        except ValidationFailedError as e:
            refused.append(_Refusal(definition_key("eval_template", template.name), f"authoring refuses it: {e}"))
    return admitted, refused


def _reset_template(survey: _Survey, admitted: EvalTemplate, now: str) -> list[ResetDefinition]:
    """Write a template's next revision, archive the one it supersedes, and write its corpus cases into it: a judge
    template's battery, frozen, with its corpus case sets minted over it, or any other template's own cases."""
    storage, scope = survey.host.storage, survey.corpus.scope_id
    key = definition_key("eval_template", admitted.name)
    prior = survey.templates.get(admitted.name, [])
    revision = max((t.revision for t in prior), default=0) + 1
    written = admitted.model_copy(
        update={
            "id": _new_id(),
            "revision": revision,
            "seed_digest": template_digest(admitted),
            "created_at": now,
            "updated_at": now,
        }
    )
    storage.save_template(written)
    for record in prior:
        if not record.archived:
            storage.save_template(record.model_copy(update={"archived": True, "updated_at": now}))
    reading = survey.readings[key]
    entries = [_entry(reading, written.id, f"revision {revision}")]
    if written.candidate_kind != JUDGE_KIND:
        for case_key, own_case in survey.own_of(admitted.name):
            written_case = own_case.test_case(written)
            storage.save_test_case(written_case)
            entries.append(_entry(survey.readings[case_key], written_case.id, f"a case of revision {revision}"))
        return entries
    frozen: dict[str, str] = {}
    for case_key, (case, authored, _digest) in survey.built_of(admitted.name):
        stored = freeze_seeded_judge_case(
            storage, template=written, scope_id=scope, authored=authored, seed_name=case.name
        )
        if case.name:
            frozen[case.name] = stored.test_case_id
        entries.append(_entry(survey.readings[case_key], stored.test_case_id, f"a case of revision {revision}"))
    for corpus_set in survey.corpus.case_sets:
        if corpus_set.template != admitted.name:
            continue
        minted = mint_case_set(
            storage,
            scope_id=scope,
            name=corpus_set.name,
            template_id=written.id,
            test_case_ids=[frozen[name] for name in corpus_set.judge_cases],
            tracked=corpus_set.tracked,
            seed_digest=survey.readings[definition_key("case_set", corpus_set.name)].seed_digest,
        )
        entries.append(
            _entry(survey.readings[definition_key("case_set", corpus_set.name)], minted.id, minted.ref.label)
        )
    return entries


def _refuse(scope: str, refusals: Sequence[_Refusal]) -> None:
    """Raise one error naming every refusal: a conflict when each may be an operator's, else a validation failure."""
    named = "; ".join(f"{refusal.key}: {refusal.why}" for refusal in refusals)
    message = f"nothing was reset in scope {scope!r}: {named}"
    kept = [refusal.key for refusal in refusals if refusal.kept]
    if len(kept) == len(refusals):
        raise ConflictError(
            message,
            remedy=(
                f"compare each with the seed corpus; {_reset_call(kept)} replaces them, keeping each superseded record "
                "archived and readable by id (a judge config contradicting another stays refused until one is archived)"
            ),
        )
    raise ValidationFailedError(
        message, remedy="correct each definition the refusals name in the seed corpus, and reset again"
    )


__all__ = [
    "KEPT_STATES",
    "DefinitionDrift",
    "DefinitionReading",
    "DefinitionReset",
    "DefinitionState",
    "ResetDefinition",
    "corpus_digests",
    "definition_drift",
    "definition_key",
    "reset_definitions",
]
