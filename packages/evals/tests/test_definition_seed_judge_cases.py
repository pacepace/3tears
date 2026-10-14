"""The definition seed's ``judge_cases`` slot: authored judge cases seeded through the door ``judge_case_author`` uses.

Pinned here:

- **A seeded case is the case ``judge_case_author`` writes**: the same content-derived test case id, its criterion read
  from the template that words it, its verdict its one label by the case's author. A second boot finds it stored and
  writes nothing.
- **A template is named, never an id**: resolved in the store first, else in the corpus the same seed writes.
- **Every case is built before anything is written**: a template neither holds, a judge template of another kind, a
  criterion the rubric lacks and a verdict the scale cannot hold each refuse the whole seed, naming the case.
- **A corpus loads them from ``judge_cases/*.json``**, with no scope, and two identical cases are refused at build.

And the ``case_sets`` slot beside it. A template carries no list of its cases, and an authored case's stored id is
derived from its judge template's generated id, so a production set (``hard_negatives v1``) is neither implied by a
template nor writable as stored ids in git: the seed names a set's cases by their corpus names instead.

- **A seeded set is ``<name> v1`` over exactly its cases' stored ids, in order**, minted by the path
  ``case_set_mint`` takes, so a launch resolves it as any minted set. A second boot finds the name stored and mints
  nothing; a scope already holding any version keeps it, whatever the corpus now lists (the store is master).
- **A set that cannot be minted refuses the whole corpus**: a case it names twice, one the corpus does not carry, or
  one of another judge template; two sets under one name; two cases under one name.
- **A case its set lists that authoring refuses refuses the whole seed, the set with it, before anything is written.**
- **A corpus loads sets from ``case_sets/*.json``**, and a judge case file that names its case nothing is named by file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel import JUDGE_KIND, EvalStorage, ValidationFailedError, judge_case_of
from threetears.evals.run.case_sets import resolve_case_set
from threetears.evals.ops import JudgeCaseAuthor, judge_case_author
from threetears.evals.run.definition_seed import (
    SeedCaseSet,
    SeedCorpus,
    SeedJudgeCase,
    SeedOutcome,
    load_seed_corpus,
    seed_eval_definitions,
)
from threetears.evals.schema import CaseSetRef, EvalTemplate, EvalTestCase
from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template

JUDGE_NAME = "Faithfulness judge"


def _admit(_definition: object) -> None:
    """A host check with nothing to refuse, so the rule under test is the engine's."""


def _judge_template() -> EvalTemplate:
    return EvalTemplate(
        scope_id=TOYHOST_SCOPE,
        name=JUDGE_NAME,
        intent="Score how faithfully an extraction reproduces its invoice, as a person would.",
        candidate_kind=JUDGE_KIND,
    )


def _case(**changes: Any) -> SeedJudgeCase:
    words: dict[str, Any] = {
        "judge_template": JUDGE_NAME,
        "criterion_template": toyhost_judged_template().name,
        "criterion": FAITHFULNESS_DIM,
        "output": json.dumps({"invoice_number": "INV-0042"}),
        "case_material": "Invoice INV-0042 from Acme, total 120.00 EUR, due 2026-11-01.",
        "expected": 1,
        "note": "three of four fields missing is not faithful",
        "author": "reviewer@example.com",
    }
    return SeedJudgeCase(**{**words, **changes})


def _seed(storage: EvalStorage, corpus: SeedCorpus) -> SeedOutcome:
    return seed_eval_definitions(
        toyhost_host(storage=storage),
        corpus,
        require_known_tools_allowed=_admit,
        refuse_undeclared_world_seed=_admit,
        refuse_undeliverable_template=_admit,
    )


def _stored_templates(storage: EvalStorage) -> tuple[EvalTemplate, EvalTemplate]:
    subject, judge = toyhost_judged_template(), _judge_template()
    storage.save_template(subject)
    storage.save_template(judge)
    return subject, judge


def test_a_seeded_case_is_the_case_judge_case_author_writes_and_a_second_boot_writes_nothing() -> None:
    storage, _ = memory_storage()
    subject, judge = _stored_templates(storage)
    corpus = SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(),))

    first = _seed(storage, corpus)
    again = _seed(storage, corpus)

    (key,) = first.created_keys["judge_case"]
    assert key.startswith(f"{JUDGE_NAME}/") and first.created["judge_case"] == 1
    (again_created, again_skipped) = (again.created["judge_case"], again.skipped["judge_case"])
    assert (again_created, again_skipped) == (0, 1), "the same case is stored: an occupied slot"
    case_id = key.removeprefix(f"{JUDGE_NAME}/")
    stored = storage.load_test_case(case_id, TOYHOST_SCOPE)
    assert stored is not None and stored.template_id == judge.id
    case = judge_case_of(stored)
    assert case is not None and case.criterion == subject.rubric[0] and case.intent == subject.intent
    assert case.labels[0].rater == "reviewer@example.com" and case.labels[0].score == 1
    seed = _case()
    authored = JudgeCaseAuthor(
        template_id=judge.id,
        criterion_template_id=subject.id,
        criterion=seed.criterion,
        output=seed.output,
        case_material=seed.case_material,
        expected=seed.expected,
        note=seed.note,
    )
    report = judge_case_author(toyhost_host(storage=storage), authored, TOYHOST_SCOPE, author=seed.author)
    assert [(c.test_case_id, c.created) for c in report.cases] == [(case_id, False)], "one door, one case"


def test_a_case_whose_verdict_changed_is_a_new_case_beside_the_stored_one() -> None:
    storage, _ = memory_storage()
    _stored_templates(storage)
    _seed(storage, SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(),)))

    moved = _seed(storage, SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(expected=2),)))

    assert (moved.created["judge_case"], moved.skipped["judge_case"]) == (1, 0)
    assert len(storage.query_test_cases(TOYHOST_SCOPE)) == 2, "never an update: the stored case stands"


def test_the_templates_a_case_names_may_come_from_the_same_corpus() -> None:
    storage, _ = memory_storage()
    corpus = SeedCorpus(
        scope_id=TOYHOST_SCOPE,
        templates=(toyhost_judged_template().model_copy(update={"id": "seeded-subject"}),),
        judge_cases=(_case(),),
    )
    storage.save_template(_judge_template())

    outcome = _seed(storage, corpus)

    assert outcome.created["eval_template"] == 1 and outcome.created["judge_case"] == 1
    assert outcome.total_failed == 0


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        ({"judge_template": "No such judge"}, "no template named 'No such judge' in the store or the corpus"),
        ({"criterion_template": JUDGE_NAME}, "has no rubric dimension"),
        ({"criterion": "extraction.tone"}, "has no rubric dimension 'extraction.tone'"),
        ({"expected": "fail"}, f"criterion '{FAITHFULNESS_DIM}' .* cannot hold the verdict 'fail'"),
    ],
    ids=["unknown-template", "criterion-template-without-it", "unknown-criterion", "verdict-off-scale"],
)
def test_a_case_authoring_would_refuse_refuses_the_whole_seed_before_anything_is_written(
    changes: dict[str, Any], refusal: str
) -> None:
    storage, _ = memory_storage()
    _stored_templates(storage)
    corpus = SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(note="the good one"), _case(**changes)))

    with pytest.raises(ValidationFailedError, match=refusal):
        _seed(storage, corpus)
    assert storage.query_test_cases(TOYHOST_SCOPE) == [], "nothing written, the good case included"


def test_a_judge_template_of_another_kind_is_refused_by_name() -> None:
    storage, _ = memory_storage()
    _stored_templates(storage)
    subject = toyhost_judged_template().name

    with pytest.raises(ValidationFailedError, match=rf"template '{subject}' is of kind .*, not '{JUDGE_KIND}'"):
        _seed(storage, SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(judge_template=subject),)))


def test_a_corpus_loads_judge_cases_with_no_scope_and_refuses_two_identical(tmp_path: Path) -> None:
    (tmp_path / "judge_cases").mkdir()
    (tmp_path / "judge_cases" / "hard-negative.json").write_text(_case().model_dump_json(), encoding="utf-8")

    corpus = load_seed_corpus(tmp_path, TOYHOST_SCOPE)

    assert corpus.judge_cases == (_case(name="hard-negative"),), "a file that names its case nothing is named by file"
    with pytest.raises(ValueError, match="two identical judge cases"):
        SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(), _case()))
    (tmp_path / "judge_cases" / "scoped.json").write_text(
        json.dumps({**_case().model_dump(), "scope_id": TOYHOST_SCOPE}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="is not a valid SeedJudgeCase"):
        load_seed_corpus(tmp_path, TOYHOST_SCOPE)


# =============================================================================
# The case_sets slot: a named set over the corpus's authored cases
# =============================================================================


def _named(name: str, **changes: Any) -> SeedJudgeCase:
    return _case(name=name, note=f"the {name} case", **changes)


def _verdict(test_case: EvalTestCase) -> int:
    case = judge_case_of(test_case)
    assert case is not None
    return case.labels[0].score


def _set(*cases: str, **changes: Any) -> SeedCaseSet:
    return SeedCaseSet(**{"name": "hard_negatives", "template": JUDGE_NAME, "judge_cases": cases, **changes})


def _corpus(*case_sets: SeedCaseSet, cases: tuple[SeedJudgeCase, ...] | None = None) -> SeedCorpus:
    named = cases if cases is not None else (_named("hn-1"), _named("hn-2", expected=2), _named("pass-1", expected=5))
    return SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=named, case_sets=case_sets)


def test_a_seeded_set_is_v1_over_its_cases_stored_ids_in_order_and_a_launch_resolves_it() -> None:
    storage, _ = memory_storage()
    _, judge = _stored_templates(storage)

    outcome = _seed(storage, _corpus(_set("hn-2", "hn-1")))

    assert outcome.created_keys["case_set"] == ["hard_negatives v1"]
    case_ids = {key.split("/", 1)[1] for key in outcome.created_keys["judge_case"]}
    stored, cases = resolve_case_set(
        storage, CaseSetRef(name="hard_negatives", version=1), template=judge, scope_id=TOYHOST_SCOPE
    )
    assert set(stored.test_case_ids) < case_ids and len(stored.test_case_ids) == 2
    assert [_verdict(case) for case in cases] == [2, 1], "the set's order, not the corpus's"
    assert stored.tracked is True


def test_a_second_boot_mints_nothing_and_a_stored_version_stands_whatever_the_corpus_now_lists() -> None:
    storage, _ = memory_storage()
    _stored_templates(storage)
    _seed(storage, _corpus(_set("hn-1")))

    again = _seed(storage, _corpus(_set("hn-1", "hn-2")))

    assert (again.created["case_set"], again.skipped["case_set"], again.failed["case_set"]) == (0, 1, 0)
    (stored,) = storage.query_case_sets(TOYHOST_SCOPE, name="hard_negatives")
    assert stored.version == 1 and len(stored.test_case_ids) == 1, "the store is master: case_set_mint mints v2"


@pytest.mark.parametrize(
    ("corpus", "refusal"),
    [
        (lambda: _corpus(_set("hn-1", "hn-1")), "case set 'hard_negatives' lists judge case 'hn-1' twice"),
        (lambda: _corpus(_set("hn-9")), "names judge case 'hn-9', which the corpus does not carry"),
        (lambda: _corpus(_set("hn-1", template="Another judge")), "judge case 'hn-1' is frozen into"),
        (lambda: _corpus(_set("hn-1"), _set("hn-2")), "two case sets named 'hard_negatives'"),
        (lambda: _corpus(cases=(_named("hn-1"), _named("hn-1", expected=2))), "two judge cases named 'hn-1'"),
    ],
    ids=["listed-twice", "not-carried", "another-template", "two-sets-one-name", "two-cases-one-name"],
)
def test_a_set_that_cannot_be_minted_refuses_the_whole_corpus(corpus: Any, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        corpus()


def test_two_cases_that_differ_only_by_name_are_one_case() -> None:
    with pytest.raises(ValueError, match="two identical judge cases"):
        _corpus(cases=(_case(name="a"), _case(name="b")))


def test_a_listed_case_authoring_refuses_refuses_the_whole_seed_and_the_set_with_it() -> None:
    storage, _ = memory_storage()
    _stored_templates(storage)
    corpus = _corpus(_set("hn-1", "bad"), cases=(_named("hn-1"), _named("bad", criterion="extraction.tone")))

    with pytest.raises(ValidationFailedError, match="has no rubric dimension 'extraction.tone'"):
        _seed(storage, corpus)
    assert storage.query_test_cases(TOYHOST_SCOPE) == [] and storage.query_case_sets(TOYHOST_SCOPE) == []


def test_a_corpus_loads_case_sets_and_names_each_judge_case_by_its_file(tmp_path: Path) -> None:
    (tmp_path / "judge_cases").mkdir()
    (tmp_path / "case_sets").mkdir()
    for name, expected in (("hn-1", 1), ("hn-2", 2)):
        words = _case(expected=expected, note=f"the {name} case").model_dump(exclude={"name"})
        (tmp_path / "judge_cases" / f"{name}.json").write_text(json.dumps(words), encoding="utf-8")
    (tmp_path / "case_sets" / "hard_negatives.json").write_text(
        json.dumps({"name": "hard_negatives", "template": JUDGE_NAME, "judge_cases": ["hn-2", "hn-1"]}),
        encoding="utf-8",
    )

    corpus = load_seed_corpus(tmp_path, TOYHOST_SCOPE)
    storage, _ = memory_storage()
    _stored_templates(storage)
    outcome = _seed(storage, corpus)

    assert [case.name for case in corpus.judge_cases] == ["hn-1", "hn-2"]
    assert corpus.case_sets == (_set("hn-2", "hn-1"),)
    assert outcome.created_keys["case_set"] == ["hard_negatives v1"]
    assert "case_set: 1 created (hard_negatives v1)" in outcome.summary()


def test_a_judge_template_the_same_corpus_seeds_carries_its_set() -> None:
    storage, _ = memory_storage()
    subject = toyhost_judged_template()
    storage.save_template(subject)
    judge = _judge_template().model_copy(update={"id": "seeded-judge"})
    corpus = SeedCorpus(
        scope_id=TOYHOST_SCOPE, templates=(judge,), judge_cases=(_named("hn-1"),), case_sets=(_set("hn-1"),)
    )

    outcome = _seed(storage, corpus)

    assert outcome.created["eval_template"] == 1 and outcome.created_keys["case_set"] == ["hard_negatives v1"]
    (stored,) = storage.query_case_sets(TOYHOST_SCOPE)
    assert stored.template_id == "seeded-judge"
