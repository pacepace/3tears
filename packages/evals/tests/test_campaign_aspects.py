"""A campaign asks one question of its arms, measured on aspects: each a template, its runs, a margin and a design.

Driven through the campaign family's functions over the in-memory store, the path every surface reaches. Pinned here:

- **A run joins one aspect**: routed by the template it ran, or by the aspect named; a run no aspect or several could
  take is refused naming them, and so is a run held by another aspect.
- **Adoption keeps what it adopts**: ``aspects_from`` carries each source's aspects whole, each keeping when its
  design was declared; what the create adds — a question, a floor, a margin — dates the campaign now, and with
  nothing added it is dated by its latest source.
- **An arm is one contestant stack**: its runs carry one key, and an adopted aspect's arm of the same label or role
  carries it too, or the create is refused naming every key.
- **A stored v15 campaign is upgraded on its stored read**, through the store the engine reads with, and is written
  back at the v16 shape; the store still finds it by the fields it addressed it by.
- **The campaign holds itself together**: unique aspect labels, a run in one aspect, floors on its own aspects and
  each once, one control and first.
- **Unnamed variants are read as candidates**, labelled by what tells them apart, and never stored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.aspects import aspect_view, aspect_views
from threetears.evals.analysis.bundle.schema import asked_in_time
from threetears.evals.analysis.campaigns import (
    add_runs_to_campaign,
    campaign_arms,
    create_campaign,
    remove_runs_from_campaign,
    set_campaign_control,
)
from threetears.evals.kernel import EvalCampaign, EvalStorage, IDENTITY_VERSION
from threetears.evals.kernel.declaration import Aspect, CampaignArm, CostLower, Floor, RateDrop
from threetears.evals.kernel.errors import RefusedInput, ValidationFailedError
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.schema.values import SweepableValue
from threetears.evals.schema.versioning import DURABLE_SCHEMA_VERSION
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    make_aspect_campaign,
    make_eval_run,
    make_template,
    memory_storage,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_SCOPE = "uni-1"

#: The host the campaigns here are gated against; their designs name only the candidate model.
_HOST = toyhost_profile()

#: The frozen v15 campaign, as the release that shipped the corpus stored it.
_FROZEN_V15 = Path(__file__).resolve().parent / "fixtures" / "durable_documents" / "v15" / "eval_campaign.json"

#: When the adopted campaigns here declared what they declared: before any create in this file.
_LONG_AGO = "2026-01-01T00:00:00+00:00"

#: When each template's runs ran, and when each source campaign asked its question: each before its own runs, the
#: tools campaign's after the tone campaign's runs, so dating either aspect by the other's time would read it wrong.
_RAN = {"tpl-tone": "2026-01-10T00:00:00+00:00", "tpl-tools": "2026-02-10T00:00:00+00:00"}
_ASKED = {"tpl-tone": _LONG_AGO, "tpl-tools": "2026-02-01T00:00:00+00:00"}


def _levers(model: str) -> dict[str, SweepableValue]:
    return {"model": SweepableValue.of(model, display=model.upper())}


def _key(model: str) -> str:
    return compute_variant_key(_levers(model))


def _store() -> tuple[EvalStorage, InMemoryDocumentStore]:
    """Two templates, and two runs of each: one of model ``a``, one of model ``b``, their pre-images recorded."""
    storage, store = memory_storage()
    for template_id in ("tpl-tone", "tpl-tools"):
        storage.save_template(make_template(id=template_id, scope_id=_SCOPE, name=template_id))
        for model in ("a", "b"):
            storage.save_eval_run(
                make_eval_run(
                    id=f"{template_id}-{model}",
                    scope_id=_SCOPE,
                    template_id=template_id,
                    candidate_model=model,
                    variant_levers=_levers(model),
                    identity_version=IDENTITY_VERSION,
                    created_at=_RAN[template_id],
                )
            )
    return storage, store


def _aspect(label: str, template_id: str, *run_ids: str, margin: float | None = 5.0) -> dict[str, Any]:
    return {
        "label": label,
        "template_id": template_id,
        "run_ids": list(run_ids),
        "margin": margin,
        "design": minimal_aspect_design().model_dump(mode="json", exclude={"merit_axes"}),
    }


def _create(
    storage: EvalStorage, *, aspects_from: tuple[str, ...] = (), exploratory: bool = False, **definition: Any
) -> EvalCampaign:
    fields: dict[str, Any] = {"name": "two aspects", "subject_id": "subject-1", "behavior": "assistant"}
    fields.update(definition)
    return create_campaign(
        storage,
        fields,
        scope_id=_SCOPE,
        created_by="op",
        profile=_HOST,
        exploratory=exploratory,
        aspects_from=aspects_from,
    )


def _two_aspects(storage: EvalStorage, **definition: Any) -> EvalCampaign:
    """A campaign of a tone aspect and a tools aspect, each holding its model-a run."""
    return _create(
        storage,
        question="Is b as good as a?",
        aspects=[_aspect("tone", "tpl-tone", "tpl-tone-a"), _aspect("tools", "tpl-tools", "tpl-tools-a")],
        **definition,
    )


# --- a run joins one aspect ----------------------------------------------------------------------------------------


def test_a_run_joins_the_aspect_gathered_on_its_template() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage)

    joined = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tools-b", "tpl-tone-b"])

    assert [aspect.run_ids for aspect in joined.aspects] == [
        ["tpl-tone-a", "tpl-tone-b"],
        ["tpl-tools-a", "tpl-tools-b"],
    ]
    assert storage.load_campaign(campaign.id, _SCOPE) == joined


def test_a_run_no_aspect_is_gathered_on_is_refused_naming_it_and_its_template() -> None:
    storage, _ = _store()
    storage.save_template(make_template(id="tpl-other", scope_id=_SCOPE, name="tpl-other"))
    storage.save_eval_run(make_eval_run(id="other-run", scope_id=_SCOPE, template_id="tpl-other"))
    campaign = _two_aspects(storage)

    with pytest.raises(ValidationFailedError) as refused:
        add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-b", "other-run"])

    said = str(refused.value)
    assert "'other-run' ran template 'tpl-other'" in said and "no aspect is gathered on it" in said
    assert "'tone' on tpl-tone" in said and "name the aspect" in refused.value.remedy
    assert storage.load_campaign(campaign.id, _SCOPE) == campaign, "nothing was attached"


def test_a_run_two_aspects_are_gathered_on_is_refused_as_ambiguous_and_joins_the_one_named() -> None:
    storage, _ = _store()
    campaign = _create(
        storage,
        question="Is b as good as a?",
        aspects=[_aspect("tone", "tpl-tone", "tpl-tone-a"), _aspect("tone again", "tpl-tone")],
    )

    with pytest.raises(ValidationFailedError, match="aspects 'tone', 'tone again' all are"):
        add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-b"])
    joined = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-b"], aspect="tone again")

    assert joined.aspect("tone again") is not None and joined.aspect("tone again").run_ids == ["tpl-tone-b"]  # type: ignore[union-attr]


def test_a_run_held_by_another_aspect_is_refused_and_one_held_by_its_own_is_ignored() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage)

    with pytest.raises(
        ValidationFailedError, match=r"'tpl-tone-a' \(held by aspect 'tone'\) cannot join aspect 'tools'"
    ):
        add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a"], aspect="tools")
    again = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a"])

    assert again.run_ids == campaign.run_ids


def test_a_run_leaves_the_aspect_holding_it_and_a_wrong_aspect_is_refused() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage)

    with pytest.raises(ValidationFailedError, match="aspect 'tools' of campaign"):
        remove_runs_from_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a"], aspect="tools")
    with pytest.raises(ValidationFailedError, match="has no aspect 'speed'"):
        remove_runs_from_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a"], aspect="speed")
    left = remove_runs_from_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a"])

    assert [aspect.run_ids for aspect in left.aspects] == [[], ["tpl-tools-a"]]


def test_one_aspect_naming_no_template_takes_every_run_whatever_it_ran() -> None:
    storage, _ = _store()
    campaign = _create(storage, aspects=[{"label": "everything"}], exploratory=True)

    joined = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-a", "tpl-tools-b"])

    assert joined.aspects[0].run_ids == ["tpl-tone-a", "tpl-tools-b"]


@pytest.mark.parametrize("named", [False, True], ids=["the-lone-aspect", "named"])
def test_an_aspect_naming_a_template_refuses_a_run_of_another(named: bool) -> None:
    """Its bars, floors and gate read its own template, so a run of another would be read against the wrong one."""
    storage, _ = _store()
    campaign = _create(storage, aspects=[{"label": "tone", "template_id": "tpl-tone"}], exploratory=True)

    with pytest.raises(ValidationFailedError, match="'tpl-tools-b' ran template 'tpl-tools', and aspect 'tone'"):
        add_runs_to_campaign(
            storage, campaign.id, _SCOPE, ["tpl-tone-b", "tpl-tools-b"], aspect="tone" if named else None
        )

    assert storage.load_campaign(campaign.id, _SCOPE) == campaign, "nothing was attached"


def test_each_aspect_infers_its_own_template_and_refuses_a_declared_design_with_no_margin() -> None:
    storage, _ = _store()
    inferred = _create(storage, aspects=[{"label": "tone", "run_ids": ["tpl-tone-a", "tpl-tone-b"]}], exploratory=True)

    with pytest.raises(ValidationFailedError, match="declares a design and no margin, and the campaign asks"):
        _create(storage, question="Is b as good as a?", aspects=[_aspect("tone", "tpl-tone", margin=None)])
    unasked = _create(storage, aspects=[_aspect("tone", "tpl-tone", margin=None)])

    assert inferred.aspects[0].template_id == "tpl-tone"
    assert unasked.aspects[0].margin is None, "a campaign asking nothing reads no drop, so it needs no margin"


# --- adoption --------------------------------------------------------------------------------------------------------


def _source(storage: EvalStorage, label: str, template_id: str, control: str | None = "a") -> EvalCampaign:
    """A campaign of one aspect asked its question before its runs, holding its two runs, its control model
    ``control``'s run, named when the question was asked."""
    asked = _ASKED[template_id]
    arms = (
        [
            CampaignArm(
                label="control",
                role="control",
                variant_key=_key(control),
                levers=_levers(control),
                identity_version=IDENTITY_VERSION,
                named_at=asked,
            )
        ]
        if control is not None
        else []
    )
    source = EvalCampaign(
        scope_id=_SCOPE,
        name=f"{label} alone",
        subject_id="subject-1",
        behavior=label,
        question=f"Is b as good as a at {label}?",
        arms=arms,
        aspects=[
            Aspect(
                label=label,
                template_id=template_id,
                margin=4.0,
                design=minimal_aspect_design(merit_axes=[]),
                run_ids=[f"{template_id}-a", f"{template_id}-b"],
                declared_at=asked,
                asked_at=asked,
            )
        ],
        floors=[Floor(aspect=label, clause=RateDrop(max_points=4.0))],
        declared_at=asked,
        declared_by="author",
        created_by="author",
    )
    storage.save_campaign(source)
    return source


def _confirms(campaign: EvalCampaign, label: str) -> bool:
    """Whether the aspect's question was asked before its first run, as every per-aspect reader decides it."""
    (template_id,) = {aspect.template_id for aspect in campaign.aspects if aspect.label == label}
    return asked_in_time(aspect_view(campaign, label).declared, _RAN[str(template_id)])


def _after_the_runs(stamp: str | None, label: str, campaign: EvalCampaign) -> bool:
    (template_id,) = {aspect.template_id for aspect in campaign.aspects if aspect.label == label}
    return stamp is not None and stamp > _RAN[str(template_id)]


def test_adopted_aspects_keep_their_own_times_so_each_stays_confirmatory() -> None:
    """Each source asked its question before its own runs; the tools source's time is after the tone source's runs,
    so dating the adoption by its latest source would turn the tone aspect post hoc."""
    storage, _ = _store()
    tone, tools = _source(storage, "tone", "tpl-tone"), _source(storage, "tools", "tpl-tools")

    adopted = _create(storage, aspects_from=(tone.id, tools.id))

    assert [aspect.asked_at for aspect in adopted.aspects] == [_ASKED["tpl-tone"], _ASKED["tpl-tools"]]
    assert _confirms(adopted, "tone") and _confirms(adopted, "tools")
    assert [aspect.run_ids for aspect in adopted.aspects] == [tone.run_ids, tools.run_ids]
    assert adopted.question == f"{tone.question}\n\n{tools.question}", "the sources' questions, joined"
    assert (adopted.declared_at, adopted.declared_by) == (_ASKED["tpl-tools"], "author"), "the words: the latest"
    assert storage.load_campaign(tone.id, _SCOPE) == tone, "a source is read, never written"


def test_what_is_added_at_adoption_is_dated_now_by_itself_and_the_adopted_aspects_stay_confirmatory() -> None:
    storage, _ = _store()
    tone, tools = _source(storage, "tone", "tpl-tone"), _source(storage, "tools", "tpl-tools")

    adopted = _create(
        storage,
        aspects_from=(tone.id, tools.id),
        question="Is b as good as a, across both?",
        floors=[{"aspect": "tools", "clause": {"kind": "cost_lower"}}],
        margins={"tone": 2.5},
    )

    assert _confirms(adopted, "tone") and _confirms(adopted, "tools"), "nothing added re-dates an adopted aspect"
    tools_view = aspect_view(adopted, "tools").declared
    assert tools_view is not None
    added = {floor.key: floor.declared_at for floor in tools_view.floors}
    assert added["tools/rate_drop/*"] is None, "the source's floor was declared with its question"
    assert _after_the_runs(added["tools/cost_lower/*"], "tools", adopted), "the added floor reads post hoc"
    tone_view = aspect_view(adopted, "tone").declared
    assert tone_view is not None and tone_view.margin == 2.5
    assert _after_the_runs(tone_view.margin_declared_at, "tone", adopted), "the added margin reads post hoc"
    assert _after_the_runs(tone_view.question_declared_at, "tone", adopted), "the new words read post hoc"
    assert tone_view.asked_at == _ASKED["tpl-tone"]
    assert adopted.declared_by == "op"


def test_a_margin_for_an_aspect_not_adopted_or_off_the_scale_is_refused_and_nothing_unreadable_is_stored() -> None:
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone")

    with pytest.raises(ValidationFailedError, match="'tools', which is no adopted aspect"):
        _create(storage, aspects_from=(tone.id,), margins={"tools": 2.5})
    for margin in (0, 100, 150, float("nan"), float("inf")):
        with pytest.raises(ValidationFailedError, match="margins"):
            _create(storage, aspects_from=(tone.id,), margins={"tone": margin})

    assert [campaign.id for campaign in storage.list_campaigns(_SCOPE)] == [tone.id], "the scope still lists"


def test_a_campaign_whose_document_would_not_read_back_is_never_written() -> None:
    """A copy skips validation, so the write is held to the stored read: no write can refuse a scope's listing."""
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone")
    (aspect,) = tone.aspects
    broken = tone.model_copy(update={"id": "broken", "aspects": [aspect.model_copy(update={"margin": 0.0})]})

    with pytest.raises(ValidationFailedError, match="would be stored as a document its own read refuses"):
        storage.save_campaign(broken)

    assert storage.load_campaign("broken", _SCOPE) is None
    assert [campaign.id for campaign in storage.list_campaigns(_SCOPE)] == [tone.id]


def test_two_adopted_aspects_under_one_label_are_refused_naming_both_and_one_may_be_relabelled() -> None:
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone")
    other = _source(storage, "tools", "tpl-tools").model_copy(update={"id": "tools-as-tone", "behavior": "tone"})
    (aspect,) = other.aspects
    other = other.model_copy(
        update={
            "aspects": [aspect.model_copy(update={"label": "tone"})],
            "floors": [floor.model_copy(update={"aspect": "tone"}) for floor in other.floors],
        }
    )
    storage.save_campaign(other)

    with pytest.raises(ValidationFailedError, match="two aspects would be labelled 'tone'") as refused:
        _create(storage, aspects_from=(tone.id, other.id))
    relabelled = _create(storage, aspects_from=(tone.id, {"campaign_id": other.id, "label": "tools"}))

    assert repr(tone.id) in str(refused.value) and repr(other.id) in str(refused.value)
    assert "aspects_from: [{campaign_id, label}]" in refused.value.remedy
    assert [aspect.label for aspect in relabelled.aspects] == ["tone", "tools"]
    assert [floor.key for floor in relabelled.floors] == ["tone/rate_drop/*", "tools/rate_drop/*"]
    assert relabelled.aspect("tools").asked_at == _ASKED["tpl-tools"]  # type: ignore[union-attr]


def test_an_adoption_holding_a_run_in_two_aspects_is_refused_naming_both() -> None:
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone")

    with pytest.raises(ValidationFailedError, match=rf"'tpl-tone-a' is held by campaign {tone.id!r} \(aspect 'tone'\)"):
        _create(storage, aspects_from=(tone.id,), aspects=[_aspect("again", "tpl-tone", "tpl-tone-a")])


# --- when the control was named --------------------------------------------------------------------------------------


def test_a_control_named_after_the_first_run_turns_the_aspect_s_readings_post_hoc() -> None:
    """Choosing what is read against once the evidence exists makes the reading post hoc, whatever the question."""
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone", control=None)
    assert _confirms(tone, "tone")

    controlled = set_campaign_control(storage, tone.id, _SCOPE, "tpl-tone-a", set_by="op", profile=_HOST)

    assert controlled.control is not None and _after_the_runs(controlled.control.named_at, "tone", controlled)
    assert not _confirms(controlled, "tone")
    assert controlled.aspects[0].asked_at == _ASKED["tpl-tone"], "the question's own time is kept"


def test_an_arm_named_at_adoption_is_dated_now_and_one_the_source_already_named_keeps_its_time() -> None:
    storage, _ = _store()
    tone, untold = _source(storage, "tone", "tpl-tone"), _source(storage, "tools", "tpl-tools", control=None)

    kept = _create(
        storage, aspects_from=(tone.id,), arms=[{"label": "a", "role": "control", "from_run_ids": ["tpl-tone-a"]}]
    )
    named = _create(
        storage, aspects_from=(untold.id,), arms=[{"label": "a", "role": "control", "from_run_ids": ["tpl-tools-a"]}]
    )

    assert kept.control is not None and kept.control.named_at == _ASKED["tpl-tone"] and _confirms(kept, "tone")
    assert named.control is not None and not _confirms(named, "tools"), "a control chosen after the runs"


# --- an arm is one contestant stack ----------------------------------------------------------------------------------


def test_an_arm_whose_runs_carry_two_keys_is_refused_naming_each() -> None:
    storage, _ = _store()

    with pytest.raises(ValidationFailedError) as refused:
        _create(
            storage,
            aspects=[_aspect("tone", "tpl-tone", "tpl-tone-a"), _aspect("tools", "tpl-tools", "tpl-tools-b")],
            arms=[{"label": "a", "role": "control", "from_run_ids": ["tpl-tone-a", "tpl-tools-b"]}],
        )

    said = str(refused.value)
    assert f"run 'tpl-tone-a' (aspect 'tone') carries {_key('a')}" in said
    assert f"run 'tpl-tools-b' (aspect 'tools') carries {_key('b')}" in said
    assert storage.list_campaigns(_SCOPE) == []


def test_an_arm_read_from_a_run_in_each_aspect_is_one_arm() -> None:
    storage, _ = _store()

    campaign = _two_aspects(
        storage, arms=[{"label": "a", "role": "control", "from_run_ids": ["tpl-tone-a", "tpl-tools-a"]}]
    )

    assert campaign.control is not None
    assert (campaign.control.label, campaign.control.variant_key) == ("a", _key("a"))
    assert campaign.control.levers == _levers("a")


def test_an_arm_an_adopted_aspect_knows_by_another_key_is_refused_naming_both() -> None:
    storage, _ = _store()
    tone = _source(storage, "tone", "tpl-tone", control="a")

    with pytest.raises(ValidationFailedError) as refused:
        _create(
            storage,
            aspects_from=(tone.id,),
            arms=[{"label": "b", "role": "control", "from_run_ids": ["tpl-tone-b"]}],
        )

    said = str(refused.value)
    assert _key("b") in said and _key("a") in said and repr(tone.id) in said


def test_two_adopted_campaigns_whose_controls_disagree_are_refused_naming_both_keys() -> None:
    storage, _ = _store()
    tone, tools = _source(storage, "tone", "tpl-tone", control="a"), _source(storage, "tools", "tpl-tools", control="b")

    with pytest.raises(ValidationFailedError) as refused:
        _create(storage, aspects_from=(tone.id, tools.id))

    assert _key("a") in str(refused.value) and _key("b") in str(refused.value)


# --- the stored read upgrades a v15 campaign -------------------------------------------------------------------------


def _stored_v15(store: InMemoryDocumentStore, **fields: Any) -> dict[str, Any]:
    document: dict[str, Any] = {**json.loads(_FROZEN_V15.read_text(encoding="utf-8")), **fields}
    store.upsert(document)
    return document


def test_a_stored_v15_campaign_reads_as_one_aspect_through_the_store_and_writes_back_at_v16() -> None:
    storage, store = _store()
    document = _stored_v15(store, template_id="tpl-tone", run_ids=["tpl-tone-a"])

    read = storage.load_campaign(document["id"], document["scope_id"])

    assert read is not None and read.schema_version == DURABLE_SCHEMA_VERSION
    (view,) = aspect_views(read)
    assert (view.aspect, view.template_id, view.run_ids) == (document["behavior"], "tpl-tone", ["tpl-tone-a"])
    assert view.declared is not None and view.declared.control is not None
    assert view.declared.control.variant_key == document["declared_design"]["control"]
    assert store.get(document["id"], document["scope_id"]) == document, "a read writes nothing back"

    add_runs_to_campaign(storage, read.id, read.scope_id, ["tpl-tone-b"])

    written = store.get(document["id"], document["scope_id"])
    assert written is not None and written["schema_version"] == DURABLE_SCHEMA_VERSION
    assert not {"template_id", "declared_design"} & set(written)
    assert [aspect["run_ids"] for aspect in written["aspects"]] == [["tpl-tone-a", "tpl-tone-b"]]


def test_the_store_finds_a_stored_v15_campaign_by_the_fields_it_addressed_it_by() -> None:
    storage, store = _store()
    document = _stored_v15(store)

    (listed,) = storage.list_campaigns(
        document["scope_id"], subject_id=document["subject_id"], behavior=document["behavior"], archived=False
    )

    assert listed.id == document["id"] and len(listed.aspects) == 1


def test_a_v15_campaign_adopted_into_a_campaign_of_two_keeps_its_design_s_time() -> None:
    storage, store = _store()
    document = _stored_v15(store, behavior="tone", template_id="tpl-tone", run_ids=["tpl-tone-a", "tpl-tone-b"])
    tools = _source(storage, "tools", "tpl-tools", control=None)

    adopted = _create(storage, aspects_from=(document["id"], tools.id), margins={"tone": 3.0})

    tone = adopted.aspect("tone")
    assert tone is not None and tone.declared_at == document["declared_design"]["declared_at"]
    assert tone.margin == 3.0
    assert adopted.control is not None and adopted.control.variant_key == document["declared_design"]["control"]


# --- the campaign holds itself together ------------------------------------------------------------------------------


def _arm(label: str, model: str, role: str = "candidate") -> CampaignArm:
    return CampaignArm(
        label=label, role=role, variant_key=_key(model), levers=_levers(model), identity_version=IDENTITY_VERSION
    )


@pytest.mark.parametrize(
    ("fields", "says"),
    [
        pytest.param(
            {"aspects": [Aspect(label="tone"), Aspect(label="tone")]}, "name more than one aspect", id="a-label-twice"
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone", run_ids=["r"]), Aspect(label="tools", run_ids=["r"])]},
            "held by aspects 'tone' and 'tools'",
            id="a-run-in-two-aspects",
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone", run_ids=["r", "r"])]}, "held twice by aspect 'tone'", id="a-run-twice"
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone")], "floors": [Floor(aspect="speed", clause=CostLower())]},
            "which the campaign has no aspect for",
            id="a-floor-on-no-aspect",
        ),
        pytest.param(
            {
                "aspects": [Aspect(label="tone")],
                "floors": [Floor(aspect="tone", clause=CostLower()), Floor(aspect="tone", clause=CostLower())],
            },
            "tone/cost_lower/\\* are declared more than once",
            id="a-floor-twice",
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone")], "arms": [_arm("b", "b"), _arm("a", "a", "control")]},
            "comes after a candidate",
            id="a-control-not-first",
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone")], "arms": [_arm("a", "a", "control"), _arm("b", "b", "control")]},
            "2 control arms",
            id="two-controls",
        ),
        pytest.param(
            {"aspects": [Aspect(label="tone")], "arms": [_arm("a", "a"), _arm("a", "b")]},
            "arm label",
            id="a-label-on-two-arms",
        ),
        pytest.param({"aspects": []}, "at least 1 item", id="no-aspect"),
    ],
)
def test_a_campaign_that_does_not_hold_together_is_refused(fields: dict[str, Any], says: str) -> None:
    with pytest.raises(ValidationError, match=says):
        EvalCampaign(scope_id=_SCOPE, name="c", subject_id="s", behavior="b", created_by="op", **fields)


def test_its_runs_and_whether_it_is_exploratory_are_derived_from_its_aspects() -> None:
    campaign = make_aspect_campaign(run_ids=["r1", "r2"], design=None)
    declared = campaign.model_copy(
        update={"aspects": [*campaign.aspects, Aspect(label="tools", design=minimal_aspect_design(), run_ids=["r3"])]}
    )

    assert (campaign.run_ids, campaign.exploratory) == (["r1", "r2"], True)
    assert (declared.run_ids, declared.exploratory) == (["r1", "r2", "r3"], False)
    assert declared.aspect("tools") is declared.aspects[1] and declared.aspect("speed") is None
    assert EvalCampaign.from_dict(declared.to_dict()) == declared, "the derived echo is discarded on read"


def test_a_campaign_of_two_aspects_is_two_views_and_none_is_implied() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage)

    tone, tools = aspect_views(campaign)

    assert (tone.aspect, tone.template_id, tone.run_ids) == ("tone", "tpl-tone", ["tpl-tone-a"])
    assert tools.declared is not None and tools.declared.question == campaign.question
    assert tools.declared.asked_at == campaign.declared_at and tools.declared.margin == 5.0
    assert aspect_view(campaign, "tools") == tools
    with pytest.raises(RefusedInput, match="2 aspects"):
        aspect_view(campaign)


# --- unnamed variants are candidates ---------------------------------------------------------------------------------


def test_every_unnamed_variant_among_the_runs_is_read_as_a_candidate_and_never_stored() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage, arms=[{"label": "a", "role": "control", "from_run_ids": ["tpl-tone-a"]}])
    campaign = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-b", "tpl-tools-b"])

    control, candidate = campaign_arms(storage, campaign, profile=_HOST)

    assert (control.label, control.role, control.variant_key) == ("a", "control", _key("a"))
    assert (candidate.role, candidate.variant_key, candidate.levers) == ("candidate", _key("b"), _levers("b"))
    assert candidate.label == "model=B", "labelled by the lever on which the variants differ"
    assert [arm.label for arm in campaign.arms] == ["a"], "a derived arm is never stored"


def test_a_derived_label_a_named_arm_already_has_is_told_apart() -> None:
    storage, _ = _store()
    campaign = _two_aspects(storage, arms=[{"label": "model=B", "role": "control", "from_run_ids": ["tpl-tone-a"]}])
    campaign = add_runs_to_campaign(storage, campaign.id, _SCOPE, ["tpl-tone-b"])

    _control, candidate = campaign_arms(storage, campaign, profile=_HOST)

    assert candidate.label == "model=B (2)"
