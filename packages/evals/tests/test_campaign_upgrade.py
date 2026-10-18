"""A campaign stored before aspects existed reads as a campaign of one aspect, by a pure dict step.

Driven through :func:`upgrade_campaign_v15` over the frozen v15 campaign and over campaigns this build writes, with
each upgraded part read back through the shape it will be stored in, so the step's output is held to the models
rather than to a copy of its own rules.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import minimal_declaration
from threetears.evals.kernel.campaign_upgrade import QUESTION_SEPARATOR, UPGRADED_CONTROL_LABEL, upgrade_campaign_v15
from threetears.evals.kernel.declaration import (
    Aspect,
    CampaignArm,
    CampaignDesign,
    CostLower,
    DecisionRule,
    Floor,
    Question,
    RateDrop,
    preregistered_questions,
)
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.schema.versioning import DURABLE_ADDRESSING_FIELDS, CoreVersionRefused, upgrade_durable_document

#: The frozen v15 campaign, as the release that shipped the corpus wrote it.
_FROZEN_V15 = Path(__file__).resolve().parent / "fixtures" / "durable_documents" / "v15" / "eval_campaign.json"

#: The fields the store reads a campaign by, which no step may touch.
_ADDRESSING = DURABLE_ADDRESSING_FIELDS["eval_campaign"]

#: A control's pre-image, and the key it digests to.
_LEVERS = {"model": SweepableValue.of("glm", display="GLM")}
_KEY = compute_variant_key(_LEVERS)


def _frozen() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(_FROZEN_V15.read_text(encoding="utf-8"))
    return loaded


def _v15(
    declared_design: CampaignDesign | None = None, *, template_id: str | None = None, run_ids: Sequence[str] = ()
) -> dict[str, Any]:
    """A stored v15 campaign: the frozen one's fields, holding ``template_id``, ``run_ids`` and this design."""
    stored = _frozen()
    stored.update(
        template_id=template_id,
        run_ids=list(run_ids),
        declared_design=None if declared_design is None else declared_design.model_dump(mode="json"),
    )
    return stored


def _stored(*questions: Question, control: bool = True, **campaign: Any) -> dict[str, Any]:
    """A v15 campaign, its design asking ``questions``, as its stored document."""
    design = minimal_declaration(control=_KEY, control_levers=_LEVERS) if control else minimal_declaration()
    design = design.model_copy(update={"questions": list(questions)})
    return _v15(design, **campaign)


def _floors(upgraded: dict[str, Any]) -> list[Floor]:
    return [Floor.model_validate(floor) for floor in upgraded["floors"]]


def _aspect(upgraded: dict[str, Any]) -> Aspect:
    (aspect,) = upgraded["aspects"]
    return Aspect.model_validate(aspect)


class TestTheFrozenCampaign:
    def test_a_behavior_holding_a_slash_labels_an_aspect_that_validates(self) -> None:
        upgraded = upgrade_campaign_v15({**_frozen(), "behavior": "queue/discipline"})

        assert _aspect(upgraded).label == "queue-discipline"
        assert upgraded["behavior"] == "queue/discipline"

    def test_it_reads_as_one_aspect_labelled_with_its_behavior(self) -> None:
        stored = _frozen()
        upgraded = upgrade_campaign_v15(stored)

        aspect = _aspect(upgraded)
        design = stored["declared_design"]
        assert aspect.label == stored["behavior"]
        assert (aspect.template_id, aspect.run_ids, aspect.stratum) == (stored["template_id"], stored["run_ids"], None)
        assert aspect.declared_at == design["declared_at"]
        assert aspect.design is not None
        moved = {"questions", "control", "control_levers", "control_identity_version", "declared_at", "declared_by"}
        assert aspect.design.model_dump(mode="json") == {
            **{key: value for key, value in design.items() if key not in moved},
            "merit_axes": None,
        }
        assert aspect.margin is None

    def test_its_control_moves_verbatim_onto_the_control_arm(self) -> None:
        stored = _frozen()
        design = stored["declared_design"]

        (arm,) = upgrade_campaign_v15(stored)["arms"]

        assert arm == {
            "label": UPGRADED_CONTROL_LABEL,
            "role": "control",
            "variant_key": design["control"],
            "levers": design["control_levers"],
            "identity_version": design["control_identity_version"],
        }
        assert CampaignArm.model_validate(arm).variant_key == design["control"]

    def test_the_campaign_declares_what_its_design_did_and_no_more(self) -> None:
        stored = _frozen()
        upgraded = upgrade_campaign_v15(stored)

        assert upgraded["question"] == ""
        assert upgraded["floors"] == []
        assert upgraded["declared_at"] == stored["declared_design"]["declared_at"]
        assert upgraded["declared_by"] == stored["declared_design"]["declared_by"]
        assert upgraded["call"] is None
        assert not {"template_id", "run_ids", "declared_design"} & set(upgraded)

    def test_every_other_field_passes_through_untouched(self) -> None:
        stored = _frozen()
        upgraded = upgrade_campaign_v15(stored)

        kept = set(stored) - {"template_id", "run_ids", "declared_design"}
        assert _ADDRESSING <= kept
        assert {key: upgraded[key] for key in kept} == {key: stored[key] for key in kept}


class TestARuleBearingCampaign:
    def test_each_clause_of_a_live_rule_is_a_floor_on_the_aspect(self) -> None:
        rule = DecisionRule(
            clauses=(RateDrop(max_points=3, reading="cites a source"), RateDrop(max_points=5), CostLower())
        )
        upgraded = upgrade_campaign_v15(_stored(Question(text="Switch to the candidate?", decision=rule)))

        assert [floor.key for floor in _floors(upgraded)] == [
            "conversation/rate_drop/cites a source",
            "conversation/rate_drop/*",
            "conversation/cost_lower/*",
        ]
        assert [floor.clause for floor in _floors(upgraded)] == list(rule.clauses)

    def test_the_margin_is_the_floor_that_reads_every_pass_rate(self) -> None:
        first = DecisionRule(clauses=(RateDrop(max_points=3, reading="cites a source"), RateDrop(max_points=5)))
        upgraded = upgrade_campaign_v15(_stored(Question(text="Is it as good?", decision=first)))

        assert _aspect(upgraded).margin == 5.0

    def test_the_margin_tightens_with_the_floor_it_is_read_off(self) -> None:
        first = DecisionRule(clauses=(RateDrop(max_points=3, reading="cites a source"), RateDrop(max_points=5)))
        second = DecisionRule(clauses=(RateDrop(max_points=2),))
        upgraded = upgrade_campaign_v15(
            _stored(Question(text="Is it as good?", decision=first), Question(text="And cheaper?", decision=second))
        )

        every_rate = next(floor for floor in _floors(upgraded) if floor.key == "conversation/rate_drop/*")
        assert every_rate.clause == RateDrop(max_points=2)
        assert _aspect(upgraded).margin == 2.0

    def test_two_clauses_on_one_reading_are_the_tighter_floor(self) -> None:
        first = DecisionRule(clauses=(RateDrop(max_points=5), CostLower()))
        second = DecisionRule(clauses=(RateDrop(max_points=2), CostLower()))
        upgraded = upgrade_campaign_v15(
            _stored(Question(text="Is it as good?", decision=first), Question(text="And cheaper?", decision=second))
        )

        assert [floor.clause for floor in _floors(upgraded)] == [RateDrop(max_points=2), CostLower()]

    def test_a_retired_or_superseded_question_s_rule_is_no_floor(self) -> None:
        retired = Question(
            text="Old?",
            decision=DecisionRule(clauses=(RateDrop(max_points=1),)),
            retired_at="2026-10-01T00:00:00+00:00",
        )
        original = Question(text="First wording?", decision=DecisionRule(clauses=(CostLower(),)))
        reworded = Question(text="Second wording?", supersedes=original.id)
        upgraded = upgrade_campaign_v15(_stored(retired, original, reworded))

        assert upgraded["floors"] == []
        assert _aspect(upgraded).margin is None
        assert upgraded["question"] == "Second wording?"


class TestAMultiQuestionCampaign:
    def test_its_live_questions_join_into_one(self) -> None:
        upgraded = upgrade_campaign_v15(_stored(Question(text="Is it kinder?"), Question(text="Is it quicker?")))

        assert upgraded["question"] == QUESTION_SEPARATOR.join(["Is it kinder?", "Is it quicker?"])

    def test_the_aspect_asks_about_the_union_of_their_axes(self) -> None:
        upgraded = upgrade_campaign_v15(
            _stored(
                Question(text="Is it kinder?", merit_axes=["quality"]),
                Question(text="Is it cheaper and as kind?", merit_axes=["cost", "quality"]),
            )
        )

        design = _aspect(upgraded).design
        assert design is not None
        assert design.merit_axes == ["quality", "cost"]

    def test_an_unscoped_question_leaves_the_aspect_unscoped(self) -> None:
        upgraded = upgrade_campaign_v15(
            _stored(Question(text="Is it kinder?", merit_axes=["quality"]), Question(text="Is it better at all?"))
        )

        design = _aspect(upgraded).design
        assert design is not None
        assert design.merit_axes == []

    def test_it_is_declared_when_its_last_question_was_asked(self) -> None:
        stored = _stored(
            Question(text="Is it quicker?", asked_at="2026-10-03T00:00:00+00:00"),
            Question(text="Is it kinder?", asked_at="2026-10-02T00:00:00+00:00"),
        )
        stored["declared_design"]["declared_at"] = "2026-10-01T00:00:00+00:00"

        upgraded = upgrade_campaign_v15(stored)

        assert upgraded["declared_at"] == "2026-10-03T00:00:00+00:00"
        assert _aspect(upgraded).declared_at == "2026-10-01T00:00:00+00:00"

    def test_a_design_re_stamped_after_its_questions_leaves_them_pre_registered(self) -> None:
        # Naming the control after runs joined re-stamps the design: its questions were still asked first.
        stored = _stored(Question(text="Is it kinder?", asked_at="2026-10-02T00:00:00+00:00"))
        stored["declared_design"]["declared_at"] = "2026-10-09T00:00:00+00:00"

        upgraded = upgrade_campaign_v15(stored)

        assert upgraded["declared_at"] == "2026-10-02T00:00:00+00:00"
        assert _aspect(upgraded).declared_at == "2026-10-09T00:00:00+00:00"

    def test_one_asking_nothing_is_declared_when_its_design_was(self) -> None:
        stored = _stored(Question(text="Old?", retired_at="2026-10-05T00:00:00+00:00"))
        stored["declared_design"]["declared_at"] = "2026-10-04T00:00:00+00:00"

        upgraded = upgrade_campaign_v15(stored)

        assert upgraded["declared_at"] == "2026-10-04T00:00:00+00:00"
        design = _aspect(upgraded).design
        assert design is not None
        assert design.merit_axes is None

    @pytest.mark.parametrize("asked_at", ["last tuesday", "2026-10-02T00:00:00", None, 7])
    def test_an_unorderable_asked_at_leaves_the_question_undated_and_the_campaign_readable(self, asked_at: Any) -> None:
        """v15 never validated a question's time, so the step reads it rather than refusing the document: which part
        was asked last cannot be said, so the question is undated, read as asked after the evidence. A refusal
        here would make a stored campaign unreadable, and with it every listing of its scope."""
        stored = _stored(Question(text="Is it kinder?"), Question(text="Is it quicker?"))
        stored["declared_design"]["questions"][0]["asked_at"] = asked_at
        design = minimal_declaration().model_copy(
            update={"questions": [Question(text="Is it kinder?", asked_at="2026-10-02T00:00:00")]}
        )

        upgraded = upgrade_campaign_v15(stored)
        read = EvalCampaign.from_dict({**stored, "schema_version": 15})

        assert (upgraded["declared_at"], _aspect(upgraded).asked_at) == (None, None)
        assert read.declared_at is None and read.aspects[0].asked_at is None
        with pytest.raises(ValueError, match="is stamped"):
            preregistered_questions(design, "2026-10-09T00:00:00+00:00")

    def test_a_question_without_an_id_or_words_is_read_as_having_none(self) -> None:
        stored = _stored(Question(text="Is it kinder?"))
        del stored["declared_design"]["questions"][0]["id"], stored["declared_design"]["questions"][0]["text"]

        assert upgrade_campaign_v15(stored)["question"] == ""


class TestWhatV15NeverValidated:
    def test_a_run_named_twice_is_held_once_in_first_named_order(self) -> None:
        stored = _v15(run_ids=["run-1", "run-2", "run-1"])

        read = EvalCampaign.from_dict({**stored, "schema_version": 15})

        assert _aspect(upgrade_campaign_v15(stored)).run_ids == ["run-1", "run-2"]
        assert read.run_ids == ["run-1", "run-2"]

    def test_a_document_the_step_cannot_read_is_refused_as_a_version_refusal_never_a_raw_error(self) -> None:
        stored = _v15()
        del stored["behavior"]

        with pytest.raises(CoreVersionRefused, match="v15 to v16 step") as refused:
            upgrade_durable_document({**stored, "schema_version": 15})
        with pytest.raises(ValidationError, match="schema_version"):
            EvalCampaign.from_dict({**stored, "schema_version": 15})

        assert isinstance(refused.value.__cause__, KeyError)


class TestACampaignWithNoRule:
    def test_it_has_no_margin_and_no_floor(self) -> None:
        upgraded = upgrade_campaign_v15(_stored(Question(text="Is it kinder?")))

        assert _aspect(upgraded).margin is None
        assert upgraded["floors"] == []

    def test_one_with_no_control_has_no_arm(self) -> None:
        assert upgrade_campaign_v15(_stored(control=False))["arms"] == []

    def test_one_with_no_design_is_one_exploratory_aspect(self) -> None:
        stored = _v15(template_id="template-1", run_ids=["run-1", "run-2"])
        upgraded = upgrade_campaign_v15(stored)

        aspect = _aspect(upgraded)
        assert (aspect.design, aspect.margin, aspect.declared_at) == (None, None, None)
        assert (aspect.template_id, aspect.run_ids) == ("template-1", ["run-1", "run-2"])
        assert (upgraded["question"], upgraded["arms"], upgraded["floors"]) == ("", [], [])
        assert (upgraded["declared_at"], upgraded["declared_by"], upgraded["call"]) == (None, "", None)


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param(_frozen(), id="frozen"),
        pytest.param(
            _stored(Question(text="Switch?", decision=DecisionRule(clauses=(RateDrop(max_points=5),)))), id="ruled"
        ),
        pytest.param(_v15(run_ids=["run-1"]), id="exploratory"),
    ],
)
class TestTheStepIsPure:
    def test_its_input_is_not_modified(self, stored: dict[str, Any]) -> None:
        before = copy.deepcopy(stored)

        upgraded = upgrade_campaign_v15(stored)
        upgraded["aspects"][0]["run_ids"].append("run-added")
        if upgraded["aspects"][0]["design"] is not None:
            upgraded["aspects"][0]["design"]["axes"].clear()

        assert stored == before

    def test_the_same_document_upgrades_to_the_same_document(self, stored: dict[str, Any]) -> None:
        assert upgrade_campaign_v15(stored) == upgrade_campaign_v15(copy.deepcopy(stored))

    def test_it_never_touches_an_addressing_field(self, stored: dict[str, Any]) -> None:
        upgraded = upgrade_campaign_v15(stored)

        assert {key: upgraded.get(key) for key in _ADDRESSING} == {key: stored.get(key) for key in _ADDRESSING}
