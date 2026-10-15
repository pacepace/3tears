"""Seeded definitions are brought up to date by a drift read and an explicit reset, never in place.

The boot seed is fill-only: once a definition is stored, the store is master and a seed edit never reaches it. So a
release reads drift over its corpus (:func:`definition_drift`: every corpus definition ``current``, ``stale``,
``customized``, ``unknown`` or ``absent``, writing nothing) and resets what it chooses (:func:`reset_definitions`).

Pinned here:

- **A reset writes a new revision, and two revisions never pool** (the property held hardest, so it is first): a
  campaign whose runs span a reset is refused at creation, when a run is added, and when its bundle is assembled,
  naming both revisions; each revision's runs analyse alone. The name resolves to the new revision, the old one
  stays readable by id, and every run, result and case keeps pointing at the revision it ran against.
- **The seed records what it wrote**, and the drift read compares the store with it: a stale definition per doc
  type resets and reads current after.
- **The seed never grows a battery**: a case the corpus adds to a judge template already holding cases is withheld
  (``needs_reset``), with any case set listing it; the drift read calls the template stale, and its reset writes the
  next revision with every case. A first boot seeds a template and its cases together.
- **A customized or unknown definition is refused**, nothing written, every refusal named; ``overwrite_customized``
  accepts it.
- **A pre-.30 row** reads ``unknown``, or ``stale`` when the host supplies the digest of what it seeded then. The
  supplied map is the whole of what the host seeded, so a judge case it does not name reads ``stale`` (never seeded,
  a reset writes it without discard) and one it names that nothing stored matches reads ``customized`` (removed).
- **A template's own cases** (``SeedCorpus.template_cases``) seed with it and reset with it, under the battery's two
  rules: the seed writes a case only into a revision holding no cases (else ``needs_reset``), and on a revision whose
  cases no seed recorded (a host's own pre-.32 seed wrote them) the supplied map is the whole of what was seeded.
- **The checks come from the host the call is handed** (``EvalHost.template_checks``), as ``create_template`` reads
  them: the actions take only the corpus, and a host supplying none is refused.
- **``templates_list`` shows each template's revision.**
- **The drift read writes nothing.**
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.actions import Caller, definition_drift_actions, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.campaigns import add_runs_to_campaign, create_campaign
from threetears.evals.kernel import JUDGE_KIND, ConflictError, EvalStorage, RefusedInput, ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.host import TemplateChecks
from threetears.evals.ops import OpsHost, templates_list
from threetears.evals.run import (
    DefinitionDrift,
    DefinitionReset,
    SeedCaseSet,
    SeedCorpus,
    SeedJudgeCase,
    SeedOutcome,
    SeedTemplateCase,
    corpus_digests,
    create_template,
    definition_drift,
    freeze_judge_cases,
    load_seed_corpus,
    reset_definitions,
    seed_eval_definitions,
)
from threetears.evals.schema import EvalTemplate, EvalTestCase, RubricDim
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    make_eval_result,
    make_eval_run,
    make_judge_config,
    make_rubric_dim,
    memory_storage,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import TOY_EXTRACTOR_KIND, toyhost_template

TEMPLATE_KEY = f"eval_template/{toyhost_template().name}"


def _corpus(*templates: EvalTemplate) -> SeedCorpus:
    return SeedCorpus(scope_id=TOYHOST_SCOPE, templates=templates)


def _template(description: str = "four cases") -> EvalTemplate:
    return toyhost_template().model_copy(update={"description": description})


def _seed(storage: EvalStorage, corpus: SeedCorpus) -> None:
    seed_eval_definitions(toyhost_host(storage=storage), corpus)


def _reset(storage: EvalStorage, corpus: SeedCorpus, keys: Sequence[str], **kwargs: object):  # type: ignore[no-untyped-def]
    return reset_definitions(toyhost_host(storage=storage), corpus, keys, **kwargs)  # type: ignore[arg-type]


def _run_on(storage: EvalStorage, template: EvalTemplate, run_id: str, cases: int) -> None:
    """A completed run of ``template`` at one model over ``cases`` cases, each with a result."""
    case_ids = [f"{run_id}-case-{i}" for i in range(cases)]
    storage.save_eval_run(
        make_eval_run(
            id=run_id, scope_id=TOYHOST_SCOPE, template_id=template.id, status="completed", test_case_ids=case_ids
        )
    )
    for case_id in case_ids:
        storage.save_eval_result(
            make_eval_result(id=f"{case_id}-result", scope_id=TOYHOST_SCOPE, eval_run_id=run_id, test_case_id=case_id)
        )


def _campaign(run_ids: list[str]) -> EvalCampaign:
    return EvalCampaign(
        scope_id=TOYHOST_SCOPE,
        name="queue discipline over time",
        subject_id="subject-1",
        subject_kind="s",
        behavior="b",
        run_ids=run_ids,
        created_by="test:fixture",
    )


def _spanning_a_reset() -> tuple[EvalStorage, EvalTemplate, EvalTemplate]:
    """A store whose template was seeded, run over four cases, reset to a new revision, and run over seven."""
    storage, _ = memory_storage()
    _seed(storage, _corpus(_template("four cases")))
    first = storage.load_template_by_name(toyhost_template().name, TOYHOST_SCOPE)
    assert first is not None
    _run_on(storage, first, "run-four", 4)
    _reset(storage, _corpus(_template("seven cases")), [TEMPLATE_KEY])
    second = storage.load_template_by_name(toyhost_template().name, TOYHOST_SCOPE)
    assert second is not None
    _run_on(storage, second, "run-seven", 7)
    return storage, first, second


def test_a_campaign_spanning_a_reset_refuses_to_pool_the_two_revisions() -> None:
    storage, first, second = _spanning_a_reset()

    assert second.id != first.id and (first.revision, second.revision) == (1, 2)
    assert second.description == "seven cases" and not second.archived
    old = storage.load_template(first.id, TOYHOST_SCOPE)
    assert old is not None and old.description == "four cases", "the older revision stays readable by id, unedited"
    assert old.archived, "and is no longer what the name resolves to"
    four = storage.load_eval_run("run-four", TOYHOST_SCOPE)
    assert four is not None and four.template_id == first.id, "a run keeps pointing at the content it ran against"

    profile = toyhost_profile()
    with pytest.raises(ValidationFailedError) as refused:
        assemble_context_bundle(_campaign(["run-four", "run-seven"]), storage=storage, profile=profile)
    said = str(refused.value)
    assert repr(toyhost_template().name) in said and "revision 1" in said and "revision 2" in said, said

    for definition in ({"run_ids": ["run-four", "run-seven"]},):
        with pytest.raises(ValidationFailedError, match="revision"):
            create_campaign(
                storage,
                {"name": "spanning", "subject_id": "subject-1", "behavior": "b", **definition},
                scope_id=TOYHOST_SCOPE,
                created_by="t",
                profile=profile,
                exploratory=True,
            )
    alone = create_campaign(
        storage,
        {"name": "four only", "subject_id": "subject-1", "behavior": "b", "run_ids": ["run-four"]},
        scope_id=TOYHOST_SCOPE,
        created_by="t",
        profile=profile,
        exploratory=True,
    )
    with pytest.raises(ValidationFailedError, match="revision"):
        add_runs_to_campaign(storage, alone.id, TOYHOST_SCOPE, ["run-seven"])
    assert storage.load_campaign(alone.id, TOYHOST_SCOPE).run_ids == ["run-four"], "nothing was attached"  # type: ignore[union-attr]

    each = [
        assemble_context_bundle(_campaign([run_id]), storage=storage, profile=profile)
        for run_id in ("run-four", "run-seven")
    ]
    assert [cell.n_cases for bundle in each for cell in bundle.cells] == [4, 7], "each revision reads alone"


def test_two_templates_sharing_a_name_at_revision_one_are_not_two_revisions_and_pool() -> None:
    """Only a reset writes a revision, always one past the newest: two records under one name, each revision 1, are
    two templates that share a name, and a campaign over runs of both is no pooling of revisions."""
    storage, _ = memory_storage()
    first, second = (_template("four cases").model_copy(update={"id": f"one-name-{n}"}) for n in (1, 2))
    for template in (first, second):
        storage.save_template(template)
    assert first.id != second.id and first.name == second.name and (first.revision, second.revision) == (1, 1)
    _run_on(storage, first, "run-first", 4)
    _run_on(storage, second, "run-second", 4)
    profile = toyhost_profile()

    created = create_campaign(
        storage,
        {"name": "one name", "subject_id": "subject-1", "behavior": "b", "run_ids": ["run-first", "run-second"]},
        scope_id=TOYHOST_SCOPE,
        created_by="t",
        profile=profile,
        exploratory=True,
    )
    alone = create_campaign(
        storage,
        {"name": "first only", "subject_id": "subject-1", "behavior": "b", "run_ids": ["run-first"]},
        scope_id=TOYHOST_SCOPE,
        created_by="t",
        profile=profile,
        exploratory=True,
    )
    add_runs_to_campaign(storage, alone.id, TOYHOST_SCOPE, ["run-second"])
    bundle = assemble_context_bundle(_campaign(["run-first", "run-second"]), storage=storage, profile=profile)

    assert created.run_ids == ["run-first", "run-second"]
    assert storage.load_campaign(alone.id, TOYHOST_SCOPE).run_ids == ["run-first", "run-second"]  # type: ignore[union-attr]
    assert bundle.cells, "the bundle assembles"


# --- the record, the drift read, and a reset per doc type --------------------------------------------------------

JUDGE_NAME = "Faithfulness judge"
DIM_KEY = "extraction.layout"
CONFIG_DIM = FAITHFULNESS_DIM


def _judge_template() -> EvalTemplate:
    return EvalTemplate(
        scope_id=TOYHOST_SCOPE,
        name=JUDGE_NAME,
        intent="Score how faithfully an extraction reproduces its invoice, as a person would.",
        candidate_kind=JUDGE_KIND,
    )


def _judge_case(name: str, **changes: Any) -> SeedJudgeCase:
    words: dict[str, Any] = {
        "name": name,
        "judge_template": JUDGE_NAME,
        "criterion_template": toyhost_judged_template().name,
        "criterion": FAITHFULNESS_DIM,
        "output": json.dumps({"invoice_number": f"INV-{name}"}),
        "case_material": "Invoice INV-0042 from Acme, total 120.00 EUR, due 2026-11-01.",
        "expected": 1,
        "note": f"case {name}: three of four fields missing is not faithful",
        "author": "reviewer@example.com",
    }
    return SeedJudgeCase(**{**words, **changes})


def _full_corpus(**changes: Any) -> SeedCorpus:
    """A corpus of every seeded doc type: a subject template and a judge template, a rubric dim, a judge config, two
    judge cases and a case set over one of them; ``changes`` replaces any of them."""
    parts: dict[str, Any] = {
        "templates": (toyhost_judged_template(), _judge_template()),
        "rubric_dims": (
            make_rubric_dim(
                scope_id=TOYHOST_SCOPE,
                key=DIM_KEY,
                dim=RubricDim(name=DIM_KEY, description="Fields sit where the invoice put them.", scale="ordinal"),
            ),
        ),
        "judge_configs": (
            make_judge_config(
                scope_id=TOYHOST_SCOPE,
                name="faithfulness-v1",
                rubric_dim_id=CONFIG_DIM,
                prompt_template="Score it.",
                archived=True,
            ),
        ),
        "judge_cases": (_judge_case("a"), _judge_case("b")),
        "case_sets": (SeedCaseSet(name="hard_negatives", template=JUDGE_NAME, judge_cases=("a",)),),
    }
    return SeedCorpus(scope_id=TOYHOST_SCOPE, **{**parts, **changes})


def _drift(storage: EvalStorage, corpus: SeedCorpus, **kwargs: Any) -> DefinitionDrift:
    return definition_drift(toyhost_host(storage=storage), corpus, **kwargs)


def _states(drift: DefinitionDrift) -> dict[str, str]:
    return {reading.key: reading.state for reading in drift.readings}


def _seeded() -> tuple[EvalStorage, InMemoryDocumentStore]:
    storage, store = memory_storage()
    _seed(storage, _full_corpus())
    return storage, store


def test_the_seed_records_what_it_wrote_and_a_fresh_seed_reads_current_everywhere() -> None:
    storage, _ = _seeded()

    drift = _drift(storage, _full_corpus())

    assert set(_states(drift).values()) == {"current"}, drift.render()
    assert sorted(_states(drift)) == sorted(
        [
            f"eval_template/{toyhost_judged_template().name}",
            f"eval_template/{JUDGE_NAME}",
            f"rubric_dim/{DIM_KEY}",
            f"judge_config/{CONFIG_DIM}/faithfulness-v1",
            f"judge_case/{JUDGE_NAME}/a",
            f"judge_case/{JUDGE_NAME}/b",
            "case_set/hard_negatives",
        ]
    )
    assert drift.summary() == "drift: 0 stale, 0 customized, 0 unknown, 0 absent, 7 current"
    assert all(r.seeded_source == "recorded" for r in drift.readings), "every row the seed wrote carries its record"
    judge = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    assert judge is not None
    names = sorted(case.seed_name or "" for case in storage.query_test_cases(TOYHOST_SCOPE, template_id=judge.id))
    assert names == ["a", "b"], "a seeded case keeps the corpus name it was seeded as, readable on the row"


def test_an_absent_definition_reads_absent() -> None:
    storage, _ = memory_storage()

    drift = _drift(storage, _full_corpus())

    assert set(_states(drift).values()) == {"absent"}, drift.render()


def test_the_drift_read_writes_nothing() -> None:
    storage, store = _seeded()
    for template in storage.query_templates(TOYHOST_SCOPE):
        storage.save_template(template.model_copy(update={"seed_digest": None}))
    before = copy.deepcopy(store.documents)
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))

    _drift(storage, moved)
    _drift(storage, moved, seeded_digests=corpus_digests(toyhost_host(storage=storage), _full_corpus()))

    assert store.documents == before


def _template_moved() -> EvalTemplate:
    return toyhost_judged_template().model_copy(update={"description": "now seven cases"})


def test_a_stale_template_resets_to_a_new_revision_and_reads_current() -> None:
    storage, _ = _seeded()
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    key = f"eval_template/{toyhost_judged_template().name}"
    first = storage.load_template_by_name(toyhost_judged_template().name, TOYHOST_SCOPE)

    assert _states(_drift(storage, moved))[key] == "stale"
    reset = _reset(storage, moved, [key])

    (entry,) = reset.reset
    assert (entry.before, entry.written, entry.replaced_id) == ("stale", "revision 2", first.id)  # type: ignore[union-attr]
    assert _states(reset.after)[key] == "current" and set(_states(reset.after).values()) == {"current"}
    assert storage.load_template_by_name(toyhost_judged_template().name, TOYHOST_SCOPE).id == entry.written_id  # type: ignore[union-attr]


def test_a_stale_rubric_dim_resets_to_a_new_record_and_reads_current() -> None:
    storage, _ = _seeded()
    dim = make_rubric_dim(
        scope_id=TOYHOST_SCOPE,
        key=DIM_KEY,
        dim=RubricDim(name=DIM_KEY, description="Every field sits where the invoice put it.", scale="ordinal"),
    )
    moved = _full_corpus(rubric_dims=(dim,))
    key = f"rubric_dim/{DIM_KEY}"
    (old,) = storage.query_rubric_dims(TOYHOST_SCOPE)

    assert _states(_drift(storage, moved))[key] == "stale"
    reset = _reset(storage, moved, [key])

    assert _states(reset.after)[key] == "current"
    active = storage.load_active_rubric_dim(DIM_KEY, TOYHOST_SCOPE)
    assert active is not None and active.id == reset.reset[0].written_id != old.id
    assert storage.load_rubric_dim(old.id, TOYHOST_SCOPE).archived  # type: ignore[union-attr]


def test_a_stale_judge_config_resets_to_a_new_record_and_reads_current() -> None:
    storage, _ = _seeded()
    config = make_judge_config(
        scope_id=TOYHOST_SCOPE,
        name="faithfulness-v1",
        rubric_dim_id=CONFIG_DIM,
        prompt_template="Score it well.",
        archived=True,
    )
    moved = _full_corpus(judge_configs=(config,))
    key = f"judge_config/{CONFIG_DIM}/faithfulness-v1"

    assert _states(_drift(storage, moved))[key] == "stale"
    reset = _reset(storage, moved, [key])

    assert _states(reset.after)[key] == "current"
    configs = storage.query_judge_configs(TOYHOST_SCOPE, rubric_dim_id=CONFIG_DIM)
    assert sorted(c.prompt_template for c in configs) == ["Score it well.", "Score it."], "the old one is kept"


def _judge_case_moved() -> SeedCorpus:
    return _full_corpus(
        judge_cases=(_judge_case("a", note="case a: a missing total is not faithful"), _judge_case("b"))
    )


def _assert_a_new_battery(storage: EvalStorage, first: EvalTemplate, reset: DefinitionReset) -> None:
    """The judge template's revision 2 holds the whole corpus battery and its case set; revision 1 keeps its own."""
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()
    second = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    assert second is not None and second.revision == 2
    battery = storage.query_test_cases(TOYHOST_SCOPE, template_id=second.id)
    old = storage.query_test_cases(TOYHOST_SCOPE, template_id=first.id)
    assert len(battery) == 2 and not {c.id for c in old} & {c.id for c in battery}
    assert sorted(c.seed_name or "" for c in battery) == ["a", "b"]
    latest = storage.query_case_sets(TOYHOST_SCOPE, name="hard_negatives")[0]
    assert latest.template_id == second.id and latest.version == 2
    assert [entry.key for entry in reset.reset][:4] == [
        f"eval_template/{JUDGE_NAME}",
        f"judge_case/{JUDGE_NAME}/a",
        f"judge_case/{JUDGE_NAME}/b",
        "case_set/hard_negatives",
    ]


def test_a_stale_judge_case_resets_with_its_judge_template_and_its_case_sets() -> None:
    storage, _ = _seeded()
    moved = _judge_case_moved()
    key = f"judge_case/{JUDGE_NAME}/a"
    first = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    assert first is not None

    states = _states(_drift(storage, moved))
    assert states[key] == "stale" and states["case_set/hard_negatives"] == "stale"
    reset = _reset(storage, moved, [key])

    _assert_a_new_battery(storage, first, reset)
    assert len(storage.query_test_cases(TOYHOST_SCOPE, template_id=first.id)) == 2, "revision 1 keeps its battery"


def test_a_case_the_corpus_adds_to_a_seeded_battery_is_withheld_by_the_seed_and_written_by_a_reset() -> None:
    """The seed never grows a battery runs were measured against: a new case waits for a reset's new revision, and
    runs of the two revisions never pool."""
    storage, _ = _seeded()
    grown = _full_corpus(judge_cases=(_judge_case("a"), _judge_case("b"), _judge_case("c")))
    first = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    assert first is not None
    _run_on(storage, first, "run-two", 2)

    boot = seed_eval_definitions(toyhost_host(storage=storage), grown)

    assert boot.created["judge_case"] == 0 and boot.needs_reset == {"judge_case": [f"{JUDGE_NAME}/c"]}
    assert len(storage.query_test_cases(TOYHOST_SCOPE, template_id=first.id)) == 2, "the battery is as it was"
    drift = _drift(storage, grown)
    assert drift.keys("stale") == [f"eval_template/{JUDGE_NAME}", f"judge_case/{JUDGE_NAME}/c"], drift.render()

    reset = _reset(storage, grown, drift.keys("stale"))

    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()
    second = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    assert second is not None and second.revision == 2
    battery = storage.query_test_cases(TOYHOST_SCOPE, template_id=second.id)
    assert sorted(case.seed_name or "" for case in battery) == ["a", "b", "c"]
    _run_on(storage, second, "run-three", 3)
    with pytest.raises(ValidationFailedError, match="revision 1.*revision 2"):
        assemble_context_bundle(_campaign(["run-two", "run-three"]), storage=storage, profile=toyhost_profile())


def test_a_case_set_listing_a_withheld_case_is_withheld_with_it() -> None:
    storage, _ = _seeded()
    grown = _full_corpus(
        judge_cases=(_judge_case("a"), _judge_case("b"), _judge_case("c")),
        case_sets=(
            SeedCaseSet(name="hard_negatives", template=JUDGE_NAME, judge_cases=("a",)),
            SeedCaseSet(name="smoke", template=JUDGE_NAME, judge_cases=("a", "c")),
        ),
    )

    boot = seed_eval_definitions(toyhost_host(storage=storage), grown)

    assert boot.needs_reset["case_set"] == ["smoke"] and not storage.query_case_sets(TOYHOST_SCOPE, name="smoke")
    assert _states(_drift(storage, grown))["case_set/smoke"] == "stale"


def test_a_first_boot_seeds_a_judge_template_and_its_cases_together() -> None:
    storage, _ = memory_storage()

    boot = seed_eval_definitions(toyhost_host(storage=storage), _full_corpus())

    assert boot.created["judge_case"] == 2 and boot.created["case_set"] == 1 and not boot.needs_reset
    assert set(_states(_drift(storage, _full_corpus())).values()) == {"current"}


def test_a_stale_case_set_whose_cases_are_current_is_a_new_mint_on_the_same_revision() -> None:
    storage, _ = _seeded()
    moved = _full_corpus(case_sets=(SeedCaseSet(name="hard_negatives", template=JUDGE_NAME, judge_cases=("a", "b")),))
    key = "case_set/hard_negatives"
    judge = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)

    assert _states(_drift(storage, moved))[key] == "stale"
    reset = _reset(storage, moved, [key])

    assert _states(reset.after)[key] == "current"
    (entry,) = reset.reset
    assert entry.written == "hard_negatives v2"
    latest = storage.query_case_sets(TOYHOST_SCOPE, name="hard_negatives")[0]
    assert latest.template_id == judge.id and len(latest.test_case_ids) == 2  # type: ignore[union-attr]


# --- customized and unknown ----------------------------------------------------------------------------------------


def test_customized_definitions_are_refused_all_named_and_nothing_written_then_accepted_with_discard() -> None:
    storage, store = _seeded()
    subject = storage.load_template_by_name(toyhost_judged_template().name, TOYHOST_SCOPE)
    storage.save_template(subject.model_copy(update={"intent": "an operator's own wording"}))  # type: ignore[union-attr]
    (dim,) = storage.query_rubric_dims(TOYHOST_SCOPE)
    storage.save_rubric_dim(dim.model_copy(update={"archived": True}))
    keys = [f"eval_template/{toyhost_judged_template().name}", f"rubric_dim/{DIM_KEY}"]
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    assert [_states(_drift(storage, moved))[k] for k in keys] == ["customized", "customized"]
    before = copy.deepcopy(store.documents)

    with pytest.raises(ConflictError) as refused:
        _reset(storage, moved, keys)

    said = str(refused.value)
    assert all(key in said for key in keys) and "overwrite_customized=True" in refused.value.remedy, said
    assert store.documents == before, "a refused reset writes nothing"
    reset = _reset(storage, moved, keys, overwrite_customized=True)
    assert [entry.discarded for entry in reset.reset] == [True, True]
    assert all(entry.replaced_id for entry in reset.reset) and reset.render().count("(discarded)") == 2
    assert all(_states(reset.after)[key] == "current" for key in keys)


def test_a_pre_30_row_reads_unknown_and_stale_with_the_digest_the_host_supplies() -> None:
    storage, _ = _seeded()
    for template in storage.query_templates(TOYHOST_SCOPE):
        storage.save_template(template.model_copy(update={"seed_digest": None}))
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    key = f"eval_template/{toyhost_judged_template().name}"
    then = corpus_digests(toyhost_host(storage=storage), _full_corpus())

    unknown = _drift(storage, moved)
    supplied = _drift(storage, moved, seeded_digests=then)

    assert _states(unknown)[key] == "unknown" and _states(supplied)[key] == "stale"
    assert next(r for r in supplied.readings if r.key == key).seeded_source == "supplied"
    assert _states(unknown)[f"eval_template/{JUDGE_NAME}"] == "current", "a row that is the seed's needs no record"
    with pytest.raises(ConflictError, match="unknown"):
        _reset(storage, moved, [key])
    assert _states(_reset(storage, moved, [key], seeded_digests=then).after)[key] == "current"


def test_an_unknown_row_is_reset_with_discard() -> None:
    storage, _ = _seeded()
    for template in storage.query_templates(TOYHOST_SCOPE):
        storage.save_template(template.model_copy(update={"seed_digest": None}))
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    key = f"eval_template/{toyhost_judged_template().name}"

    reset = _reset(storage, moved, [key], overwrite_customized=True)

    assert (reset.reset[0].before, reset.reset[0].discarded) == ("unknown", True)
    assert _states(reset.after)[key] == "current"


GROWN_KEY = f"judge_case/{JUDGE_NAME}/c"


def _grown() -> SeedCorpus:
    """The full corpus with a third judge case, ``c``, the seeded battery never held."""
    return _full_corpus(judge_cases=(_judge_case("a"), _judge_case("b"), _judge_case("c")))


def _seeded_before_30() -> EvalStorage:
    """A seeded store whose templates carry no record of what seeded them, as a store seeded before .30 does."""
    storage, _ = _seeded()
    for template in storage.query_templates(TOYHOST_SCOPE):
        storage.save_template(template.model_copy(update={"seed_digest": None}))
    return storage


def test_a_case_new_to_a_pre_30_battery_reads_stale_when_the_supplied_digests_do_not_name_it() -> None:
    """The host's map is the whole of what it seeded: a case it does not name was never seeded, not unrecorded, so
    it reads stale and a reset writes it with the template's next revision, no discard asked."""
    storage = _seeded_before_30()
    then = corpus_digests(toyhost_host(storage=storage), _full_corpus())
    assert GROWN_KEY not in then

    drift = _drift(storage, _grown(), seeded_digests=then)

    reading = next(r for r in drift.readings if r.key == GROWN_KEY)
    assert reading.state == "stale", drift.render()
    assert reading.reason == (
        f"new to the corpus; never seeded into {JUDGE_NAME!r} (the host's seeded digests do not name it), so only a "
        "reset writes it"
    )
    assert drift.summary() == "drift: 2 stale, 0 customized, 0 unknown, 0 absent, 6 current", drift.render()
    reset = _reset(storage, _grown(), [GROWN_KEY], seeded_digests=then)
    entry = next(e for e in reset.reset if e.key == GROWN_KEY)
    assert (entry.before, entry.written, entry.discarded) == ("stale", "a case of revision 2", False)
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()


def test_a_case_the_supplied_digests_name_that_nothing_stored_matches_reads_customized() -> None:
    """Seeded by the host's account and gone from the store: removed by an operator, as for any other definition
    type, so a reset of it needs discard."""
    storage = _seeded_before_30()
    then = corpus_digests(toyhost_host(storage=storage), _grown())
    assert GROWN_KEY in then

    drift = _drift(storage, _grown(), seeded_digests=then)

    reading = next(r for r in drift.readings if r.key == GROWN_KEY)
    assert (reading.state, reading.seeded_source, reading.seeded_digest) == ("customized", "supplied", then[GROWN_KEY])
    assert reading.reason == "seeded as its corpus content, since removed by an operator"
    with pytest.raises(ConflictError, match="customized"):
        _reset(storage, _grown(), [GROWN_KEY], seeded_digests=then)
    reset = _reset(storage, _grown(), [GROWN_KEY], seeded_digests=then, overwrite_customized=True)
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()


def test_a_case_new_to_a_pre_30_battery_reads_unknown_with_no_supplied_digests() -> None:
    """With no map at all there is no record anywhere, and the reading stays unknown."""
    storage = _seeded_before_30()

    drift = _drift(storage, _grown())

    reading = next(r for r in drift.readings if r.key == GROWN_KEY)
    assert reading.state == "unknown", drift.render()
    assert reading.reason == f"{JUDGE_NAME!r} was seeded before seeds recorded their cases"
    with pytest.raises(ConflictError, match="unknown"):
        _reset(storage, _grown(), [GROWN_KEY])


def test_discarded_means_a_stored_copy_that_may_be_an_operator_s_was_written_over() -> None:
    """``discarded`` is true only where a stored copy reading customized or unknown was replaced under
    ``overwrite_customized``. A case written fresh, with no stored copy, reads false whatever its state says of the
    host's record, and so does one written over a stale copy: the seed's own."""
    storage = _seeded_before_30()
    template_key = f"eval_template/{JUDGE_NAME}"

    fresh = _reset(storage, _grown(), [template_key])

    grown = next(e for e in fresh.reset if e.key == GROWN_KEY)
    assert (grown.before, grown.replaced_id, grown.discarded) == ("unknown", "", False), fresh.render()
    assert not any(entry.discarded for entry in fresh.reset), fresh.render()
    assert "(discarded)" not in fresh.render()

    storage, _ = _seeded()
    stale = _reset(storage, _judge_case_moved(), [f"judge_case/{JUDGE_NAME}/a"])
    case_a = next(e for e in stale.reset if e.key == f"judge_case/{JUDGE_NAME}/a")
    assert (case_a.before, bool(case_a.replaced_id), case_a.discarded) == ("stale", True, False), stale.render()
    assert not any(entry.discarded for entry in stale.reset), stale.render()

    storage = _seeded_before_30()
    then = corpus_digests(toyhost_host(storage=storage), _grown())
    removed = _reset(storage, _grown(), [GROWN_KEY], seeded_digests=then, overwrite_customized=True)
    gone = next(e for e in removed.reset if e.key == GROWN_KEY)
    assert (gone.before, gone.replaced_id, gone.discarded) == ("customized", "", False), "nothing stored to replace"


def test_a_judge_template_whose_revision_holds_cases_the_corpus_does_not_carry_is_refused_without_discard() -> None:
    storage, _ = _seeded()
    judge = storage.load_template_by_name(JUDGE_NAME, TOYHOST_SCOPE)
    operators = _judge_case("c", note="an operator's own hard negative").authored(toyhost_judged_template())
    freeze_judge_cases(storage, template=judge, scope_id=TOYHOST_SCOPE, authored=[operators])  # type: ignore[arg-type]
    moved = _full_corpus(
        templates=(toyhost_judged_template(), _judge_template().model_copy(update={"description": "v2"}))
    )
    key = f"eval_template/{JUDGE_NAME}"

    with pytest.raises(ConflictError, match="does not carry"):
        _reset(storage, moved, [key])

    reset = _reset(storage, moved, [key], overwrite_customized=True)
    assert _states(reset.after)[key] == "current"


def test_a_key_the_corpus_does_not_carry_is_refused() -> None:
    storage, _ = _seeded()

    with pytest.raises(ValidationFailedError, match="no definition"):
        _reset(storage, _full_corpus(), ["eval_template/nobody"])


# --- a template's own cases ---------------------------------------------------------------------------------------

SUBJECT = toyhost_template().name


def _own(name: str, **changes: Any) -> SeedTemplateCase:
    """One of the subject template's own cases: a buyer persona and the invoice it sends."""
    fields: dict[str, Any] = {
        "template": SUBJECT,
        "name": name,
        "variation_params": {"invoice": f"INV-{name}"},
        "host_payload": {"persona": f"buyer {name}"},
    }
    return SeedTemplateCase(**{**fields, **changes})


def _own_key(name: str) -> str:
    return f"template_case/{SUBJECT}/{name}"


def _own_corpus(*cases: SeedTemplateCase) -> SeedCorpus:
    return SeedCorpus(scope_id=TOYHOST_SCOPE, templates=(_template(),), template_cases=cases)


def _boot(storage: EvalStorage, corpus: SeedCorpus) -> SeedOutcome:
    return seed_eval_definitions(toyhost_host(storage=storage), corpus)


def _subject(storage: EvalStorage) -> EvalTemplate:
    template = storage.load_template_by_name(SUBJECT, TOYHOST_SCOPE)
    assert template is not None
    return template


def _cases_on(storage: EvalStorage, template: EvalTemplate) -> list[EvalTestCase]:
    return sorted(storage.query_test_cases(TOYHOST_SCOPE, template_id=template.id), key=lambda c: c.name or "")


def test_a_first_boot_seeds_a_template_with_its_own_cases_and_a_second_finds_them_present() -> None:
    storage, _ = memory_storage()
    corpus = _own_corpus(_own("a"), _own("b", stratum="lookalike"))

    first = _boot(storage, corpus)
    again = _boot(storage, corpus)

    assert first.created_keys["template_case"] == [f"{SUBJECT}/a", f"{SUBJECT}/b"] and not first.needs_reset
    assert (again.created["template_case"], again.skipped["template_case"]) == (0, 2) and not again.needs_reset
    stored = _cases_on(storage, _subject(storage))
    assert [(c.name, c.seed_name, c.stratum, c.host_payload) for c in stored] == [
        ("a", "a", None, {"persona": "buyer a"}),
        ("b", "b", "lookalike", {"persona": "buyer b"}),
    ], "a case is stored as the corpus states it, named and recording the corpus name it was seeded as"
    drift = _drift(storage, corpus)
    assert _states(drift) == {TEMPLATE_KEY: "current", _own_key("a"): "current", _own_key("b"): "current"}
    assert all(r.seeded_source == "recorded" for r in drift.readings)


def test_a_case_the_corpus_adds_to_a_seeded_revision_is_withheld_and_a_reset_writes_the_next_with_every_case() -> None:
    storage, _ = memory_storage()
    _boot(storage, _own_corpus(_own("a")))
    first = _subject(storage)
    grown = _own_corpus(_own("a"), _own("b"))

    boot = _boot(storage, grown)

    assert boot.needs_reset == {"template_case": [f"{SUBJECT}/b"]} and len(_cases_on(storage, first)) == 1
    drift = _drift(storage, grown)
    assert drift.keys("stale") == [TEMPLATE_KEY, _own_key("b")], drift.render()
    assert next(r for r in drift.readings if r.key == TEMPLATE_KEY).reason == (
        "revision 1; its revision lacks 1 corpus case(s), which the seed never adds to a revision holding cases"
    )

    reset = _reset(storage, grown, [_own_key("b")])

    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()
    assert [(e.key, e.written) for e in reset.reset] == [
        (TEMPLATE_KEY, "revision 2"),
        (_own_key("a"), "a case of revision 2"),
        (_own_key("b"), "a case of revision 2"),
    ]
    second = _subject(storage)
    assert second.revision == 2 and [c.seed_name for c in _cases_on(storage, second)] == ["a", "b"]
    assert [c.seed_name for c in _cases_on(storage, first)] == ["a"], "revision 1 keeps the cases it ran"


def test_a_case_whose_content_changed_reads_stale_and_resets_with_its_template() -> None:
    storage, _ = memory_storage()
    _boot(storage, _own_corpus(_own("a")))
    moved = _own_corpus(_own("a", host_payload={"persona": "a buyer in a hurry"}))

    states = _states(_drift(storage, moved))
    reset = _reset(storage, moved, [_own_key("a")])

    assert (states[TEMPLATE_KEY], states[_own_key("a")]) == ("stale", "stale")
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()
    (case,) = _cases_on(storage, _subject(storage))
    assert case.host_payload == {"persona": "a buyer in a hurry"}


def test_a_case_an_operator_relabelled_in_place_reads_customized_and_resets_only_with_discard() -> None:
    storage, _ = memory_storage()
    corpus = _own_corpus(_own("a"))
    _boot(storage, corpus)
    (case,) = _cases_on(storage, _subject(storage))
    storage.save_test_case(case.model_copy(update={"stratum": "boundary"}))

    reading = next(r for r in _drift(storage, corpus).readings if r.key == _own_key("a"))
    with pytest.raises(ConflictError, match="customized"):
        _reset(storage, corpus, [_own_key("a")])
    reset = _reset(storage, corpus, [_own_key("a")], overwrite_customized=True)

    assert (reading.state, reading.seeded_source) == ("customized", "recorded")
    entry = next(e for e in reset.reset if e.key == _own_key("a"))
    assert (entry.before, entry.discarded) == ("customized", True)
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()


def _host_seeded_before_32() -> EvalStorage:
    """A store whose template the engine seeded and whose cases the host's own seed wrote: ``a`` as the corpus has it
    now, ``b`` as it was, neither recording a seed."""
    storage, _ = memory_storage()
    _boot(storage, _own_corpus())
    template = _subject(storage)
    for case in (_own("a"), _own("b", host_payload={"persona": "an old buyer b"})):
        storage.save_test_case(
            EvalTestCase(
                scope_id=TOYHOST_SCOPE,
                template_id=template.id,
                name=case.name,
                variation_params=case.variation_params,
                host_payload=case.host_payload,
            )
        )
    return storage


def test_cases_a_host_seeded_itself_read_by_content_then_name_and_the_supplied_map_is_the_whole_of_what_it_seeded() -> (
    None
):
    storage = _host_seeded_before_32()
    now = _own_corpus(_own("a"), _own("b"), _own("c"))
    then = corpus_digests(
        toyhost_host(storage=storage), _own_corpus(_own("a"), _own("b", host_payload={"persona": "an old buyer b"}))
    )

    unrecorded = _states(_drift(storage, now))
    supplied = _drift(storage, now, seeded_digests=then)
    named_c = _states(_drift(storage, now, seeded_digests={**then, _own_key("c"): _own("c").digest()}))

    assert [unrecorded[_own_key(n)] for n in "abc"] == ["current", "unknown", "unknown"]
    assert [_states(supplied)[_own_key(n)] for n in "abc"] == ["current", "stale", "stale"], supplied.render()
    assert named_c[_own_key("c")] == "customized", "named by the map and stored nowhere: removed since"
    boot = _boot(storage, now)
    assert (
        boot.needs_reset == {"template_case": [f"{SUBJECT}/b", f"{SUBJECT}/c"]} and boot.skipped["template_case"] == 1
    )
    with pytest.raises(ConflictError, match="unknown"):
        _reset(storage, now, supplied.keys("stale"))
    reset = _reset(storage, now, supplied.keys("stale"), seeded_digests=then)
    assert set(_states(reset.after).values()) == {"current"}, reset.after.render()
    assert [c.seed_name for c in _cases_on(storage, _subject(storage))] == ["a", "b", "c"]


def test_a_corpus_refuses_a_template_case_it_cannot_write_naming_each() -> None:
    same = {"variation_params": {}, "host_payload": {"persona": "the same buyer"}}

    with pytest.raises(ValueError) as refused:
        SeedCorpus(
            scope_id=TOYHOST_SCOPE,
            templates=(_template(), _judge_template()),
            template_cases=(
                _own("stray", template="nobody"),
                _own("judged", template=JUDGE_NAME),
                _own("twice"),
                _own("twice", host_payload={"persona": "another"}),
                _own("d", **same),
                _own("e", **same),
            ),
        )

    said = str(refused.value)
    assert "'nobody', which the corpus does not carry" in said and f"judge template {JUDGE_NAME!r}" in said, said
    assert f"two cases of template {SUBJECT!r} named 'twice'" in said and "two identical cases" in said, said


def test_a_corpus_directory_loads_a_template_s_cases_named_by_their_files(tmp_path: Path) -> None:
    (tmp_path / "templates").mkdir()
    (tmp_path / "template_cases").mkdir()
    (tmp_path / "templates" / "probe.json").write_text(
        json.dumps({"name": "probe", "intent": "i", "candidate_kind": TOY_EXTRACTOR_KIND}), encoding="utf-8"
    )
    (tmp_path / "template_cases" / "hurried-buyer.json").write_text(
        json.dumps({"template": "probe", "host_payload": {"persona": "in a hurry"}}), encoding="utf-8"
    )

    corpus = load_seed_corpus(tmp_path, TOYHOST_SCOPE)

    (case,) = corpus.template_cases
    assert (case.name, case.template, case.host_payload) == ("hurried-buyer", "probe", {"persona": "in a hurry"})


# --- the host's checks, and the listing ----------------------------------------------------------------------------


def _refuse(template: EvalTemplate) -> None:
    raise ValidationFailedError(f"no kind here delivers {template.name!r}", remedy="author it for a kind this host has")


ANOTHER = {"name": "another", "intent": "i", "candidate_kind": TOY_EXTRACTOR_KIND}


def test_a_reset_and_create_template_read_one_host_s_checks_and_a_host_with_none_is_refused() -> None:
    storage, _ = _seeded()
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    key = f"eval_template/{toyhost_judged_template().name}"
    refusing = toyhost_host(
        storage=storage,
        template_checks=TemplateChecks(
            require_known_tools_allowed=lambda _tools: None,
            refuse_undeclared_world_seed=lambda _template: None,
            refuse_undeliverable_template=_refuse,
        ),
    )
    unchecked = toyhost_host(storage=storage, template_checks=None)

    with pytest.raises(ValidationFailedError, match="no kind here delivers"):
        reset_definitions(refusing, moved, [key])
    with pytest.raises(ValidationFailedError, match="no kind here delivers"):
        create_template(refusing, ANOTHER, scope_id=TOYHOST_SCOPE)
    with pytest.raises(RefusedInput, match="supplies no template checks"):
        reset_definitions(unchecked, moved, [key])
    with pytest.raises(RefusedInput, match="supplies no template checks"):
        create_template(unchecked, ANOTHER, scope_id=TOYHOST_SCOPE)
    assert _states(definition_drift(unchecked, moved))[key] == "stale", "a drift read admits nothing, so needs none"


def test_templates_list_shows_each_template_s_revision() -> None:
    storage, first, second = _spanning_a_reset()
    host = toyhost_host(storage=storage)

    active = templates_list(host, TOYHOST_SCOPE).templates
    archived = templates_list(host, TOYHOST_SCOPE, archived=True).templates

    assert [(line.id, line.revision) for line in active] == [(second.id, 2)]
    assert [(line.id, line.revision) for line in archived] == [(first.id, 1)]


# --- the catalogue actions -----------------------------------------------------------------------------------------


def _customized() -> EvalStorage:
    """A seeded store whose subject template an operator edited in place."""
    storage, _ = _seeded()
    subject = storage.load_template_by_name(toyhost_judged_template().name, TOYHOST_SCOPE)
    storage.save_template(subject.model_copy(update={"intent": "an operator's own wording"}))  # type: ignore[union-attr]
    return storage


async def test_the_actions_read_drift_refuse_a_customized_reset_and_reset_with_discard() -> None:
    storage = off_loop(_customized)
    moved = _full_corpus(templates=(_template_moved(), _judge_template()))
    actions = definition_drift_actions(lambda scope_id: moved)
    (tool,) = eval_catalogue(actions).mount_all(standard_tools())[:1]
    eval_host = toyhost_host(storage=storage)
    host, caller = OpsHost.reading(eval_host), Caller(scope_id=TOYHOST_SCOPE, identity="operator")
    key = f"eval_template/{toyhost_judged_template().name}"
    overwrite = {"action": "definitions_reset", "definitions": [key], "overwrite_customized": True}
    checks = replace(eval_host.template_checks, refuse_undeliverable_template=_refuse)  # type: ignore[type-var]
    refusing = OpsHost.reading(replace(eval_host, template_checks=checks))

    drift = await tool.call({"action": "definitions_drift"}, host=host, caller=caller)
    refused = await tool.call({"action": "definitions_reset", "definitions": [key]}, host=host, caller=caller)
    not_deliverable = await tool.call(overwrite, host=refusing, caller=caller)
    reset = await tool.call(overwrite, host=host, caller=caller)
    listed = await tool.call({"action": "templates_list"}, host=host, caller=caller)

    assert drift.text.splitlines()[0] == "drift: 0 stale, 1 customized, 0 unknown, 0 absent, 6 current", drift.text
    assert refused.is_error and key in refused.text and "overwrite_customized" in refused.text, refused.text
    assert not_deliverable.is_error and "no kind here delivers" in not_deliverable.text, "the checks of the host called"
    assert not reset.is_error and f"{key}: customized -> revision 2" in reset.text, reset.text
    assert listed.structured is not None
    revisions = {line["name"]: line["revision"] for line in listed.structured["templates"]}
    assert revisions == {toyhost_judged_template().name: 2, JUDGE_NAME: 1}, revisions
    assert reset.text.splitlines()[-1] == "drift: 0 stale, 0 customized, 0 unknown, 0 absent, 7 current"
