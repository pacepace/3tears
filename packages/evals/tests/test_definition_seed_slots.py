"""What the definition seed writes, what it reads as occupied, and what it refuses — never what it skips in silence.

Pinned here:

- **A judge config's slot is ``(rubric_dim_id, name)``.** A dim carrying two configs — the active one a
  run inherits and an archived one a launch names by id (an A/B's control arm) — seeds both; a second
  pass writes neither, and an archived occupant is not resurrected.
- **A corpus that cannot be seeded as written is refused when it is built**: two documents of one type
  under one natural key, and two non-archived judge configs for one dim. Each refusal sits beside the
  shape that constructs.
- **A config the store would contradict is withheld and named**: a non-archived corpus config for a dim
  the store already holds an active config for is not written (it would supersede the operator's for
  every run), and the outcome and its summary say which.
- **Every template meets the gates ``create_template`` applies, before anything is written** — the world
  gate, the goal checks' discrimination proof, and each host check — and is stored as admission resolves
  it. The whole corpus is admitted, so a template whose slot is occupied is refused too.

Mutations that turn this file red (each applied to a saved copy and restored from it): the judge-config
slot keyed by ``rubric_dim_id`` alone; each of the four ``SeedCorpus`` defect lines removed; the
``conflicts=`` argument dropped from the judge-config pass; the ``admit_template`` call replaced by the
corpus template as given; and ``admit_template`` with its ``refuse_unsupplied_world`` or
``refuse_non_discriminating_checks`` line removed. (Measured: 7, 1, 1, 1, 1, 1, 5, 2 and 1 tests red.)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pytest

from threetears.evals.contracts import EvalStorage, EvalTemplate, ValidationFailedError
from threetears.evals.run import SeedCorpus, SeedOutcome, seed_eval_definitions
from packages.evals.tests.factories import make_judge_config, make_rubric_dim, memory_storage
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

SCOPE = toyhost_template().scope_id
DIM = "play.pacing"


def _admit(_definition: object) -> None:
    """A host check with nothing to refuse, so the rule under test is the engine's."""


def _seed(
    storage: EvalStorage,
    *,
    templates: Sequence[EvalTemplate] = (),
    corpus: SeedCorpus | None = None,
    refuse_undeliverable: Callable[[EvalTemplate], None] = _admit,
) -> SeedOutcome:
    return seed_eval_definitions(
        toyhost_host(storage=storage),
        corpus if corpus is not None else SeedCorpus(scope_id=SCOPE, templates=tuple(templates)),
        require_known_tools_allowed=_admit,
        refuse_undeclared_world_seed=_admit,
        refuse_undeliverable_template=refuse_undeliverable,
    )


# =============================================================================
# Judge configs: two for one dim, both seeded
# =============================================================================


def _two_pacing_judges() -> SeedCorpus:
    return SeedCorpus(
        scope_id=SCOPE,
        judge_configs=(
            make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing-strict"),
            make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing-lenient", archived=True),
        ),
    )


def test_two_judge_configs_for_one_dim_both_seed_and_a_second_pass_writes_neither() -> None:
    storage, _ = memory_storage()

    first = _seed(storage, corpus=_two_pacing_judges())
    again = _seed(storage, corpus=_two_pacing_judges())

    assert first.created["judge_config"] == 2
    assert first.created_keys["judge_config"] == [f"{DIM}/pacing-strict", f"{DIM}/pacing-lenient"]
    assert (again.created["judge_config"], again.skipped["judge_config"]) == (0, 2)
    stored = {(c.name, c.archived) for c in storage.query_judge_configs(SCOPE)}
    assert stored == {("pacing-strict", False), ("pacing-lenient", True)}
    active = storage.load_active_judge_config(DIM, SCOPE)
    assert active is not None and active.name == "pacing-strict"


def test_a_config_the_operator_archived_is_not_resurrected() -> None:
    storage, _ = memory_storage()
    _seed(storage, corpus=_two_pacing_judges())
    for config in storage.query_judge_configs(SCOPE):
        storage.save_judge_config(config.model_copy(update={"archived": True}))

    again = _seed(storage, corpus=_two_pacing_judges())

    assert (again.created["judge_config"], again.skipped["judge_config"]) == (0, 2)
    assert storage.load_active_judge_config(DIM, SCOPE) is None


# =============================================================================
# A corpus that cannot be seeded as written is refused at construction
# =============================================================================


@pytest.mark.parametrize(
    ("corpus_fields", "said"),
    [
        (
            {"templates": (toyhost_template(), toyhost_template().model_copy(update={"id": "other"}))},
            "two templates named 'Invoice field extraction'",
        ),
        (
            {"rubric_dims": (make_rubric_dim(scope_id=SCOPE), make_rubric_dim(scope_id=SCOPE))},
            "two rubric dims keyed 'conversation.tone'",
        ),
        (
            {
                "judge_configs": (
                    make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing", archived=True),
                    make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing", archived=True),
                )
            },
            f"two judge configs named 'pacing' for rubric dim '{DIM}'",
        ),
        (
            {
                "judge_configs": (
                    make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing-strict"),
                    make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing-lenient"),
                )
            },
            f"more than one non-archived judge config for rubric dim '{DIM}'",
        ),
    ],
    ids=["template name", "rubric dim key", "judge config identity", "two active configs for one dim"],
)
def test_a_corpus_that_cannot_be_seeded_as_written_is_refused(corpus_fields: dict[str, Any], said: str) -> None:
    with pytest.raises(ValueError, match=said):
        SeedCorpus(scope_id=SCOPE, **corpus_fields)


def test_the_shapes_beside_those_refusals_construct() -> None:
    """One name on two dims, and one active config beside archived ones on one dim, are distinct slots."""
    corpus = SeedCorpus(
        scope_id=SCOPE,
        judge_configs=(
            make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="v1"),
            make_judge_config(scope_id=SCOPE, rubric_dim_id="play.tension", name="v1"),
            make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="v0", archived=True),
            make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="v-1", archived=True),
        ),
    )

    assert len(corpus.judge_configs) == 4


# =============================================================================
# A config the store would contradict is withheld and named
# =============================================================================


def test_an_active_corpus_config_for_a_dim_the_operator_configured_is_withheld_and_named() -> None:
    storage, _ = memory_storage()
    storage.save_judge_config(make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="operators-own"))

    outcome = _seed(storage, corpus=_two_pacing_judges())

    assert outcome.conflicted == {"judge_config": [f"{DIM}/pacing-strict"]}
    assert outcome.created_keys["judge_config"] == [f"{DIM}/pacing-lenient"], "an archived config contradicts nothing"
    assert f"1 NOT WRITTEN, CONFLICTING WITH THE STORE ({DIM}/pacing-strict)" in outcome.summary()
    active = storage.load_active_judge_config(DIM, SCOPE)
    assert active is not None and active.name == "operators-own"


def test_once_the_operators_config_is_archived_the_corpus_config_seeds() -> None:
    """The positive control on the same store: the conflict was the live config, nothing else."""
    storage, _ = memory_storage()
    storage.save_judge_config(make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="operators-own", archived=True))

    outcome = _seed(storage, corpus=_two_pacing_judges())

    assert outcome.conflicted == {}
    assert outcome.created["judge_config"] == 2


# =============================================================================
# Templates meet every gate create_template applies, before anything is written
# =============================================================================


def test_a_template_authoring_admits_is_seeded_in_the_shape_admission_resolves() -> None:
    storage, _ = memory_storage()

    outcome = _seed(storage, templates=[toyhost_template()])

    assert outcome.created["eval_template"] == 1
    (stored,) = storage.query_templates(SCOPE)
    assert toyhost_template().kind_spec == {}, "the corpus states no spec, so a filled one came from admission"
    assert stored.kind_spec["graded_fields"] == ["invoice_number", "invoice_date", "total_amount", "vendor_name"]


def _refused(
    storage: EvalStorage, template: EvalTemplate, refuse_undeliverable: Callable[[EvalTemplate], None] = _admit
) -> str:
    """Seed a corpus of ``template`` beside a judge config; assert nothing at all was written; return why."""
    corpus = SeedCorpus(
        scope_id=SCOPE,
        templates=(template,),
        judge_configs=(make_judge_config(scope_id=SCOPE, rubric_dim_id=DIM, name="pacing"),),
    )
    with pytest.raises(ValidationFailedError) as refused:
        _seed(storage, corpus=corpus, refuse_undeliverable=refuse_undeliverable)
    assert storage.query_judge_configs(SCOPE) == [], "a refused corpus writes nothing, not the rest of it"
    return refused.value.message


def test_a_template_naming_world_state_the_host_cannot_supply_refuses_the_seed() -> None:
    storage, _ = memory_storage()
    typo = toyhost_template().model_copy(update={"goal_state_checks": ['fired("payment_hlod")']})

    said = _refused(storage, typo)

    assert "names no dimension this host's world declares" in said and repr(typo.name) in said
    assert storage.query_templates(SCOPE) == []


def test_a_goal_check_without_a_control_refuses_the_seed() -> None:
    storage, _ = memory_storage()
    unproven = toyhost_template().model_copy(update={"goal_check_controls": None})

    said = _refused(storage, unproven)

    assert "Unproven: 'call_count(\"extractor.emit_field\") == 4'" in said
    assert storage.query_templates(SCOPE) == []


def test_a_host_check_refusing_a_template_refuses_the_seed() -> None:
    storage, _ = memory_storage()

    def undeliverable(template: EvalTemplate) -> None:
        raise ValidationFailedError(f"kind {template.candidate_kind!r} cannot honour this")

    said = _refused(storage, toyhost_template(), refuse_undeliverable=undeliverable)

    assert "cannot honour this" in said
    assert storage.query_templates(SCOPE) == []


def test_a_template_whose_slot_is_occupied_is_still_admitted() -> None:
    """The verdict is the corpus's and the host's, not the store's: a host's CI over an empty store
    reaches the answer its boot over a seeded one would."""
    storage, _ = memory_storage()
    typo = toyhost_template().model_copy(update={"goal_state_checks": ['fired("payment_hlod")']})
    storage.save_template(typo)

    with pytest.raises(ValidationFailedError, match="names no dimension this host's world declares"):
        _seed(storage, templates=[typo])
