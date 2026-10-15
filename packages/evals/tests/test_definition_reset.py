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
- **A customized or unknown definition is refused**, nothing written, every refusal named; ``discard_customization``
  accepts it.
- **A pre-.30 row** reads ``unknown``, or ``stale`` when the host supplies the digest of what it seeded then.
- **The drift read writes nothing.**
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from typing import Any

import pytest

from threetears.evals.actions import Caller, definition_drift_actions, eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.campaigns import add_runs_to_campaign, create_campaign
from threetears.evals.kernel import JUDGE_KIND, ConflictError, EvalStorage, ValidationFailedError
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.ops import OpsHost
from threetears.evals.run import (
    DefinitionDrift,
    DefinitionReset,
    SeedCaseSet,
    SeedCorpus,
    SeedJudgeCase,
    corpus_digests,
    definition_drift,
    freeze_judge_cases,
    reset_definitions,
    seed_eval_definitions,
)
from threetears.evals.schema import EvalTemplate, RubricDim
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
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

TEMPLATE_KEY = f"eval_template/{toyhost_template().name}"


def _admit(_definition: object) -> None:
    """A host check with nothing to refuse, so the rule under test is the engine's."""


CHECKS = {
    "require_known_tools_allowed": _admit,
    "refuse_undeclared_world_seed": _admit,
    "refuse_undeliverable_template": _admit,
}


def _corpus(*templates: EvalTemplate) -> SeedCorpus:
    return SeedCorpus(scope_id=TOYHOST_SCOPE, templates=templates)


def _template(description: str = "four cases") -> EvalTemplate:
    return toyhost_template().model_copy(update={"description": description})


def _seed(storage: EvalStorage, corpus: SeedCorpus) -> None:
    seed_eval_definitions(toyhost_host(storage=storage), corpus, **CHECKS)


def _reset(storage: EvalStorage, corpus: SeedCorpus, keys: Sequence[str], **kwargs: object):  # type: ignore[no-untyped-def]
    return reset_definitions(toyhost_host(storage=storage), corpus, keys, **CHECKS, **kwargs)  # type: ignore[arg-type]


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

    boot = seed_eval_definitions(toyhost_host(storage=storage), grown, **CHECKS)

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

    boot = seed_eval_definitions(toyhost_host(storage=storage), grown, **CHECKS)

    assert boot.needs_reset["case_set"] == ["smoke"] and not storage.query_case_sets(TOYHOST_SCOPE, name="smoke")
    assert _states(_drift(storage, grown))["case_set/smoke"] == "stale"


def test_a_first_boot_seeds_a_judge_template_and_its_cases_together() -> None:
    storage, _ = memory_storage()

    boot = seed_eval_definitions(toyhost_host(storage=storage), _full_corpus(), **CHECKS)

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
    assert all(key in said for key in keys) and "discard_customization=True" in refused.value.remedy, said
    assert store.documents == before, "a refused reset writes nothing"
    reset = _reset(storage, moved, keys, discard_customization=True)
    assert [entry.discarded for entry in reset.reset] == [True, True]
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

    reset = _reset(storage, moved, [key], discard_customization=True)

    assert (reset.reset[0].before, reset.reset[0].discarded) == ("unknown", True)
    assert _states(reset.after)[key] == "current"


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

    reset = _reset(storage, moved, [key], discard_customization=True)
    assert _states(reset.after)[key] == "current"


def test_a_key_the_corpus_does_not_carry_is_refused() -> None:
    storage, _ = _seeded()

    with pytest.raises(ValidationFailedError, match="no definition"):
        _reset(storage, _full_corpus(), ["eval_template/nobody"])


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
    actions = definition_drift_actions(lambda scope_id: moved, **CHECKS)  # type: ignore[arg-type]
    (tool,) = eval_catalogue(actions).mount_all(standard_tools())[:1]
    host, caller = OpsHost.reading(toyhost_host(storage=storage)), Caller(scope_id=TOYHOST_SCOPE, identity="operator")
    key = f"eval_template/{toyhost_judged_template().name}"

    drift = await tool.call({"action": "definitions_drift"}, host=host, caller=caller)
    refused = await tool.call({"action": "definitions_reset", "definitions": [key]}, host=host, caller=caller)
    reset = await tool.call(
        {"action": "definitions_reset", "definitions": [key], "discard_customization": True}, host=host, caller=caller
    )

    assert drift.text.splitlines()[0] == "drift: 0 stale, 1 customized, 0 unknown, 0 absent, 6 current", drift.text
    assert refused.is_error and key in refused.text and "discard_customization" in refused.text, refused.text
    assert not reset.is_error and f"{key}: customized -> revision 2" in reset.text, reset.text
    assert reset.text.splitlines()[-1] == "drift: 0 stale, 0 customized, 0 unknown, 0 absent, 7 current"
