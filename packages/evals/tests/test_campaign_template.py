"""A campaign knows the template its runs ran, so a bar on that template's rubric is one its author can declare.

``campaign_create`` took no template and inferred none, so a bar on a template rubric dimension was refused
because "this campaign names no template that could be read" — a remedy no caller of that action could reach,
though every run it named ran one template. So a campaign takes ``template_id``, refused when the scope holds no
such template; when it names none and every run it holds ran one template, that template is its own; and a bar
that still finds no template is refused naming both ways to give it one.
"""

from __future__ import annotations

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis.campaigns import create_campaign
from threetears.evals.kernel import EvalStorage, NotFoundError, ValidationFailedError
from threetears.evals.kernel.declaration import BarOverride
from threetears.evals.schema import RubricDim
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_design
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_SCOPE = "campaign-template"
_DIM = "extraction.layout_fidelity"


def _store(templates: dict[str, str]) -> EvalStorage:
    """A store holding a template of one rubric dim, and one run per entry of ``templates`` (run id -> template id)."""
    storage = EvalStorage(InMemoryDocumentStore())
    for template_id in sorted(set(templates.values())):
        storage.save_template(
            make_template(
                id=template_id,
                scope_id=_SCOPE,
                name=f"template {template_id}",
                rubric=[RubricDim(name=_DIM, description="Fields sit where the document put them.", scale="ordinal")],
            )
        )
    for run_id, template_id in templates.items():
        storage.save_eval_run(make_eval_run(id=run_id, scope_id=_SCOPE, template_id=template_id, status="completed"))
    return storage


def _definition(run_ids: list[str], **extra: object) -> dict[str, object]:
    design = toyhost_design().model_copy(
        update={"bars": [BarOverride(measure_id=_DIM, threshold=3, direction="higher_is_better")]}
    )
    return {
        "name": "a bar on the template's rubric",
        "subject_id": "subject-1",
        "behavior": "extraction",
        "run_ids": run_ids,
        "declared_design": design.model_dump(mode="json", exclude={"declared_at", "declared_by"}),
        **extra,
    }


def _create(storage: EvalStorage, definition: dict[str, object]):  # type: ignore[no-untyped-def]
    return create_campaign(storage, definition, scope_id=_SCOPE, created_by="t", profile=toyhost_profile())


def test_runs_of_one_template_give_the_campaign_its_template_and_a_bar_on_its_rubric_is_declared() -> None:
    storage = _store({"run-a": "tpl-1", "run-b": "tpl-1"})
    campaign = _create(storage, _definition(["run-a", "run-b"]))
    assert campaign.template_id == "tpl-1"


def test_a_named_template_is_the_campaigns() -> None:
    storage = _store({"run-a": "tpl-1", "run-b": "tpl-2"})
    campaign = _create(storage, _definition(["run-a", "run-b"], template_id="tpl-2"))
    assert campaign.template_id == "tpl-2"


def test_a_named_template_the_scope_does_not_hold_is_refused_with_how_to_find_one() -> None:
    storage = _store({"run-a": "tpl-1"})
    with pytest.raises(NotFoundError, match="templates_list"):
        _create(storage, _definition(["run-a"], template_id="tpl-missing"))


def test_a_bar_with_no_template_to_read_names_both_ways_to_give_it_one() -> None:
    storage = _store({"run-a": "tpl-1", "run-b": "tpl-2"})
    with pytest.raises(ValidationFailedError) as refused:
        _create(storage, _definition(["run-a", "run-b"]))
    said = str(refused.value)
    assert "template_id" in said and "templates_list" in said and "tpl-1" in said and "tpl-2" in said


def test_the_action_takes_the_template() -> None:
    (tool, *_) = eval_catalogue().mount_all(standard_tools())
    action = tool.action("campaign_create") or next(
        t.action("campaign_create") for t in eval_catalogue().mount_all(standard_tools()) if t.action("campaign_create")
    )
    assert "template_id" in action.params.model_fields
