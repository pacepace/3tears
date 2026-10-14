"""The definition seed's ``judge_cases`` slot: authored judge cases seeded through the door ``judge_case_author`` uses.

Pinned here:

- **A seeded case is the case ``judge_case_author`` writes**: the same content-derived test case id, its criterion read
  from the template that words it, its verdict its one label by the case's author. A second boot finds it stored and
  writes nothing.
- **A template is named, never an id**: resolved in the store first, else in the corpus the same seed writes.
- **Every case is built before anything is written**: a template neither holds, a judge template of another kind, a
  criterion the rubric lacks and a verdict the scale cannot hold each refuse the whole seed, naming the case.
- **A corpus loads them from ``judge_cases/*.json``**, with no scope, and two identical cases are refused at build.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel import JUDGE_KIND, EvalStorage, ValidationFailedError, judge_case_of
from threetears.evals.ops import JudgeCaseAuthor, judge_case_author
from threetears.evals.run.definition_seed import (
    SeedCorpus,
    SeedJudgeCase,
    SeedOutcome,
    load_seed_corpus,
    seed_eval_definitions,
)
from threetears.evals.schema import EvalTemplate
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

    assert corpus.judge_cases == (_case(),)
    with pytest.raises(ValueError, match="two identical judge cases"):
        SeedCorpus(scope_id=TOYHOST_SCOPE, judge_cases=(_case(), _case()))
    (tmp_path / "judge_cases" / "scoped.json").write_text(
        json.dumps({**_case().model_dump(), "scope_id": TOYHOST_SCOPE}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="is not a valid SeedJudgeCase"):
        load_seed_corpus(tmp_path, TOYHOST_SCOPE)
