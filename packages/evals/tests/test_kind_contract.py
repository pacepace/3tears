"""A kind's contract: two Pydantic models a product writes, and everything the engine derives from them.

The toy extractor's models (``tests/fixtures/toyhost/contract.py``) are the path a product follows,
and these pin what the engine owes it for writing them and nothing else:

* the model is read once, at declaration, and a field the engine could not read as a lever is
  refused there, by name;
* a launch's overlays are validated against it before any run exists — refused by field, with
  nothing created and the admission given back — and every run records the validated model whole;
* each field is a lever every lens reads through the host's registry, on the axis its declaration
  implies, and a coordinate of the variant key — computed over what ran, so a launch naming a
  default and one naming nothing are one variant;
* the profile refuses a kind whose levers it does not register, so a knob a launch can turn and no
  analysis can see is not expressible;
* a template's ``kind_spec`` is validated where it is authored — refused by field, stored resolved —
  and again where it launches, frozen onto every run and hashed into its measurement context, and
  the kind's launcher reads it typed.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from enum import Enum
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals.contracts import EvalStorage, ValidationFailedError
from threetears.evals.contracts.host import (
    SHARED_CORE,
    SweepableRegistry,
    HostProfile,
    Interval,
    IntervalScale,
    KindContract,
    KindContractError,
    MeasureRegistry,
    NominalScale,
    Ordinal,
    freeze,
)
from threetears.evals.contracts.host.profile import ProfileRegistrationError
from threetears.evals.contracts.host.values import OrdinalScale
from threetears.evals.contracts.identity import derive_context_identity, derive_variant_identity
from threetears.evals.run import (
    LaunchGroup,
    LaunchHost,
    LaunchRequest,
    create_template,
    start_run,
    update_template,
)
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT, ExtractorOverlays, ExtractorSpec
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.memory_store import InMemoryDocumentStore


def _run_of_the_kind(**overlays: Any):
    """A toy-extractor run whose launch turned ``overlays``, frozen as a launch freezes them."""
    validated = TOY_EXTRACTOR_CONTRACT.validate_overlays(overlays)
    return make_eval_run(candidate_kind=TOY_EXTRACTOR_KIND, overlays=freeze(validated))


# --- the model, read at declaration -----------------------------------------------------------------


class _Undescribed(BaseModel):
    knob: int = 1


class _Aliased(BaseModel):
    knob: int = Field(1, alias="dial", description="a knob a launch would have to call by another name")


class _OrdinalWithoutOrder(BaseModel):
    knob: Annotated[str, Ordinal()] = Field("a", description="a string nobody declared an order for")


class _IntervalOverText(BaseModel):
    knob: Annotated[str, Interval(unit="pages")] = Field("a", description="text claiming a unit")


class _MapKeyedByNumber(BaseModel):
    knob: dict[int, str] = Field(default_factory=dict, description="entries no lever could be named after")


@pytest.mark.parametrize(
    ("model", "said"),
    [
        (_Undescribed, "has no description"),
        (_Aliased, "declares an alias"),
        (_OrdinalWithoutOrder, "marked Ordinal but is not a Literal or an Enum"),
        (_IntervalOverText, "marked Interval but is not an int or a float"),
        (_MapKeyedByNumber, "keyed by something other than str"),
    ],
    ids=["no-description", "alias", "ordinal-without-order", "interval-over-text", "map-keyed-by-number"],
)
def test_a_field_the_engine_cannot_read_as_a_lever_is_refused_at_declaration_by_name(model, said):
    with pytest.raises(KindContractError) as refused:
        KindContract("probe", overlays=model)

    assert "probe.knob" in str(refused.value) and said in str(refused.value)


class _Excluded(BaseModel):
    knob: int = Field(1, exclude=True, description="a knob the record would never see")


class _ExcludedWhenFalsy(BaseModel):
    knob: int = Field(1, exclude_if=lambda value: not value, description="a knob the record sometimes drops")


class _Encounter(BaseModel):
    hidden_dc: int = Field(10, exclude=True)


class _NestedExclusion(BaseModel):
    encounters: list[_Encounter] = Field(default_factory=list, description="the encounters a session builds to")


@pytest.mark.parametrize(
    ("model", "path"),
    [(_Excluded, "probe.knob"), (_ExcludedWhenFalsy, "probe.knob"), (_NestedExclusion, "probe.encounters.hidden_dc")],
    ids=["excluded", "excluded-conditionally", "excluded-inside-a-nested-model"],
)
@pytest.mark.parametrize("as_spec", [False, True], ids=["overlay", "spec"])
def test_a_field_the_record_would_drop_is_refused_at_declaration_by_name(model, path, as_spec):
    """A field serialization leaves out is one the frozen record cannot see, so it would merge runs.

    The launcher acts on the validated model, which still holds the field; the run records the model's
    JSON form, which does not. Two launches at different values would then record one set of overlays
    (or one spec) and share a key while running differently — a wrong merge nothing downstream undoes.
    """
    with pytest.raises(KindContractError) as refused:
        KindContract("probe", spec=model) if as_spec else KindContract("probe", overlays=model)

    assert path in str(refused.value) and "left out of the frozen record" in str(refused.value)


def test_a_set_freezes_in_one_order_whatever_order_it_was_built_in():
    """A set has no order, so its record must not have one: one configuration is one key in every process.

    Built from two orders whose sets iterate differently (8 and 0 share a hash slot, so insertion
    order decides iteration) — the in-process stand-in for two processes with different hash seeds.
    """

    class Tables(BaseModel):
        open_tables: set[int] = Field(default_factory=set, description="the tables a session may seat")
        rules_by_table: dict[str, frozenset[int]] = Field(default_factory=dict, description="each table's rules")

    contract = KindContract("tables", overlays=Tables)
    one = freeze(contract.validate_overlays({"open_tables": [8, 0], "rules_by_table": {"a": [8, 0]}}))
    other = freeze(contract.validate_overlays({"open_tables": [0, 8], "rules_by_table": {"a": [0, 8]}}))

    assert one == other == {"open_tables": [0, 8], "rules_by_table": {"a": [0, 8]}}


def test_a_blank_kind_or_prefix_is_refused():
    with pytest.raises(KindContractError, match="needs the kind's name"):
        KindContract(" ", overlays=ExtractorOverlays)
    with pytest.raises(KindContractError, match="lever prefix"):
        KindContract(TOY_EXTRACTOR_KIND, overlays=ExtractorOverlays, prefix="")


def test_every_field_is_a_lever_under_the_prefix_and_a_map_adds_its_family():
    assert TOY_EXTRACTOR_CONTRACT.lever_names == (
        "extractor.prompt_style",
        "extractor.page_limit",
        "extractor.instructions",
        "extractor.field_aliases",
        "extractor.field_aliases.*",
    )
    assert KindContract("gm", overlays=ExtractorOverlays).lever_names[0] == "gm.prompt_style", (
        "the kind's name is the prefix when none is given"
    )


# --- validation at launch -----------------------------------------------------------------------------


def test_an_overlay_the_model_refuses_is_refused_naming_each_field():
    with pytest.raises(ValidationFailedError) as refused:
        TOY_EXTRACTOR_CONTRACT.validate_overlays({"prompt_style": "shouty", "page_limit": 0})

    assert "prompt_style" in refused.value.message and "page_limit" in refused.value.message
    assert [error["loc"] for error in refused.value.details["errors"]] == [("prompt_style",), ("page_limit",)]


def test_an_overlay_the_model_does_not_have_is_refused_naming_it_and_the_ones_it_has():
    with pytest.raises(ValidationFailedError, match=r"has no overlay page_limt; its overlays are field_aliases"):
        TOY_EXTRACTOR_CONTRACT.validate_overlays({"page_limt": 3})


def test_a_kind_with_no_overlay_model_refuses_any_overlay_and_admits_none():
    bare = KindContract("bare")

    with pytest.raises(ValidationFailedError, match="declares no overlays, so a launch cannot turn knob"):
        bare.validate_overlays({"knob": 1})
    assert bare.validate_overlays({}) is None and bare.validate_overlays(None) is None
    assert bare.lever_names == ()


def test_the_validated_model_freezes_whole_so_naming_a_default_and_naming_nothing_record_alike():
    named = freeze(TOY_EXTRACTOR_CONTRACT.validate_overlays({"prompt_style": "standard"}))
    silent = freeze(TOY_EXTRACTOR_CONTRACT.validate_overlays(None))

    assert named == silent == {"prompt_style": "standard", "page_limit": 10, "instructions": "", "field_aliases": {}}


# --- levers and levels --------------------------------------------------------------------------------


def test_each_field_sits_on_the_axis_its_declaration_implies():
    levels = TOY_EXTRACTOR_CONTRACT.levels(
        _run_of_the_kind(prompt_style="verbose", page_limit=3, instructions="read the footer first")
    )

    assert levels["extractor.prompt_style"].scale == OrdinalScale(rank=2), "ranked in declaration order"
    assert levels["extractor.page_limit"].scale == IntervalScale(value=3.0, unit="pages")
    assert levels["extractor.page_limit"].display == "3pages"
    assert levels["extractor.instructions"].scale == NominalScale()
    assert levels["extractor.instructions"].display == "read the footer first"
    assert set(levels) == set(TOY_EXTRACTOR_CONTRACT.lever_names) - {"extractor.field_aliases.*"}, (
        "every fixed lever has a level; the family's container carries no coordinate"
    )


def test_an_enum_marked_ordinal_ranks_in_declaration_order_and_an_unmarked_one_is_nominal():
    class Tier(Enum):
        LOW = "low"
        HIGH = "high"

    class Tiers(BaseModel):
        ranked: Annotated[Tier, Ordinal()] = Field(Tier.LOW, description="a ranked tier")
        unranked: Tier = Field(Tier.LOW, description="a tier nobody said was ordered")

    contract = KindContract("tiers", overlays=Tiers)
    run = make_eval_run(
        candidate_kind="tiers",
        overlays=freeze(contract.validate_overlays({"ranked": "high", "unranked": "high"})),
    )

    levels = contract.levels(run)

    assert levels["tiers.ranked"].scale == OrdinalScale(rank=1)
    assert levels["tiers.unranked"].scale == NominalScale()


def test_a_long_text_level_is_joined_by_content_and_shown_cut():
    first = TOY_EXTRACTOR_CONTRACT.levels(_run_of_the_kind(instructions="x" * 200 + "a"))["extractor.instructions"]
    second = TOY_EXTRACTOR_CONTRACT.levels(_run_of_the_kind(instructions="x" * 200 + "b"))["extractor.instructions"]

    assert first.display == second.display and len(first.display) == 60
    assert first.content_hash != second.content_hash, "two texts are two levels however alike they render"


def test_a_run_of_another_kind_sits_at_one_shared_level_and_reads_as_no_value():
    elsewhere = make_eval_run(candidate_kind="another-kind")
    levels = TOY_EXTRACTOR_CONTRACT.levels(elsewhere)

    assert {level.display for level in levels.values()} == {f"(not a {TOY_EXTRACTOR_KIND} run)"}
    assert toyhost_profile().sweepables.read_all(elsewhere)["extractor.prompt_style"] is None


def test_the_registry_reads_each_field_and_each_entry_of_the_map_as_its_own_lever():
    run = _run_of_the_kind(prompt_style="terse", field_aliases={"vendor_name": "supplier", "total_amount": "sum"})

    resolved = toyhost_profile().sweepables.resolve_levers(run)

    assert resolved.values["extractor.prompt_style"] == "terse"
    assert resolved.values["extractor.field_aliases.vendor_name"] == "supplier"
    assert resolved.overlaid == {"extractor.field_aliases.vendor_name", "extractor.field_aliases.total_amount"}
    assert toyhost_profile().sweepables.read_residual(
        "extractor.field_aliases", run, [], frozenset({"extractor.field_aliases.vendor_name"})
    ) == {"total_amount": "sum"}


def test_a_campaign_may_name_an_entry_of_the_map_as_its_axis_and_never_the_family_itself():
    sweepables = toyhost_profile().sweepables

    assert sweepables.refuse_as_axis("extractor.field_aliases.vendor_name") is None
    assert sweepables.refuse_as_axis("extractor.field_aliases") is None, "the whole map is a lever of its own"
    assert sweepables.refuse_as_axis("extractor.field_aliases.*") is not None


# --- identity -----------------------------------------------------------------------------------------


def _variant_key(run) -> str:
    return derive_variant_identity(run=run, profile=toyhost_profile()).variant_key


def test_the_variant_key_is_computed_over_what_ran_not_over_what_the_launch_named():
    assert _variant_key(_run_of_the_kind(page_limit=10)) == _variant_key(_run_of_the_kind()), (
        "a launch restating a default ran the same knob at the same level"
    )
    assert _variant_key(_run_of_the_kind(page_limit=11)) != _variant_key(_run_of_the_kind())
    assert _variant_key(_run_of_the_kind(field_aliases={"vendor_name": "supplier"})) != _variant_key(
        _run_of_the_kind(field_aliases={"vendor_name": "seller"})
    ), "the map's entries reach the key through the map's own lever"


# --- the profile ---------------------------------------------------------------------------------------


def test_a_profile_registers_its_kinds_levers_itself_and_refuses_a_second_contract_for_a_kind():
    """Named once, on ``kinds``: every lens reads the contract's levers off the profile's registry."""
    profile = HostProfile(
        host_id="registered", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()), kinds=(TOY_EXTRACTOR_CONTRACT,)
    )

    for lever in TOY_EXTRACTOR_CONTRACT.lever_names:
        declared = profile.sweepables.get(lever)
        assert declared is not None and declared.role == "lever", f"{lever} is not on the registry lenses read"
    assert profile.host_sweepables.get("extractor.prompt_style") is None, "the host registered nothing by hand"
    assert profile.kind_contract(TOY_EXTRACTOR_KIND) is TOY_EXTRACTOR_CONTRACT
    assert profile.kind_contract("another-kind") == KindContract("another-kind"), (
        "a kind with no contract is asked an empty one, which refuses anything it is handed"
    )

    with pytest.raises(ProfileRegistrationError, match="more than one contract"):
        replace(profile, kinds=(TOY_EXTRACTOR_CONTRACT, TOY_EXTRACTOR_CONTRACT))


def test_a_profile_refuses_a_kinds_lever_the_host_registered_by_hand():
    """A second registration is a second place to name the kind — refused, naming the levers."""
    with pytest.raises(ProfileRegistrationError) as refused:
        HostProfile(
            host_id="twice",
            host_sweepables=SHARED_CORE.extend(TOY_EXTRACTOR_CONTRACT.sweepables),
            measures=MeasureRegistry(()),
            kinds=(TOY_EXTRACTOR_CONTRACT,),
        )

    assert "extractor.prompt_style" in str(refused.value) and "name the kind once" in str(refused.value)


class _Table(BaseModel):
    house_rules: dict[str, str] = Field(default_factory=dict, description="the house rules the table plays by")


@pytest.mark.parametrize(
    ("first", "second"),
    [("gm", "gm"), ("gm", "gm.house_rules"), ("gm.house_rules", "gm")],
    ids=["equal", "second-inside-first", "first-inside-second"],
)
def test_a_profile_refuses_two_contracts_whose_lever_prefixes_overlap(first, second):
    """Overlapping prefixes can name one lever twice, and the later kind's level would overwrite the earlier's.

    ``gm`` and ``gm.house_rules`` overlap even with different fields: the first kind's open family
    names ``gm.house_rules.<key>``, which is every lever the second kind could name.
    """
    with pytest.raises(ProfileRegistrationError, match="prefixes overlap"):
        HostProfile(
            host_id="overlap",
            host_sweepables=SHARED_CORE,
            measures=MeasureRegistry(()),
            kinds=(
                KindContract("gm", overlays=_Table, prefix=first),
                KindContract("scout", overlays=_Table, prefix=second),
            ),
        )


def test_two_contracts_whose_prefixes_only_share_letters_are_admitted():
    """The acceptance case on the same fixture: ``gm`` and ``gmx`` cannot name one lever."""
    profile = HostProfile(
        host_id="apart",
        host_sweepables=SHARED_CORE,
        measures=MeasureRegistry(()),
        kinds=(KindContract("gm", overlays=_Table), KindContract("gmx", overlays=_Table)),
    )

    assert {"gm.house_rules", "gmx.house_rules"} <= set(profile.sweepables.names)


def test_a_profile_refuses_a_registry_without_the_levers_the_engine_resolves():
    with pytest.raises(ProfileRegistrationError, match="model, candidate_kind"):
        HostProfile(host_id="coreless", host_sweepables=SweepableRegistry(()), measures=MeasureRegistry(()))


# --- kinds in the variant key ----------------------------------------------------------------------------


def _two_kind_host() -> HostProfile:
    """A host with a contracted router kind and an uncontracted game master, neither turning any knob."""
    return HostProfile(
        host_id="two-kinds",
        host_sweepables=SHARED_CORE,
        measures=MeasureRegistry(()),
        kinds=(KindContract("router"),),
    )


def test_runs_of_two_kinds_at_one_model_with_no_overlays_are_two_variants():
    """A router and a game master on one model are two contestants, so their observations never pool.

    The variant key is what cells group on, and a campaign may hold runs of any template — so a key
    with no kind in it pooled two kinds into one arm whenever nothing else told them apart.
    """
    profile = _two_kind_host()
    router = make_eval_run(candidate_kind="router", candidate_model="m")
    game_master = make_eval_run(candidate_kind="gm", candidate_model="m")

    router_key = derive_variant_identity(run=router, profile=profile).variant_key
    game_master_key = derive_variant_identity(run=game_master, profile=profile).variant_key

    assert router_key != game_master_key
    assert router_key == derive_variant_identity(run=router.model_copy(), profile=profile).variant_key, (
        "two runs of one kind at one model are still one variant"
    )


def test_a_run_of_another_kind_never_shares_a_level_with_a_none_overlay():
    """The "not a run of this kind" level hashes apart from every value, ``None`` included.

    Rendered differently and hashed alike, a pivot over the lever joined a run of another kind with
    a run of this one whose field was ``None`` — one level, two meanings.
    """

    class Seating(BaseModel):
        table: str | None = Field(None, description="the table a session is seated at, if any")

    contract = KindContract("gm", overlays=Seating)
    unseated = make_eval_run(candidate_kind="gm", overlays=freeze(contract.validate_overlays({})))
    elsewhere = make_eval_run(candidate_kind="router")

    assert contract.levels(unseated)["gm.table"].content_hash != contract.levels(elsewhere)["gm.table"].content_hash


# --- the launch -----------------------------------------------------------------------------------------


def _launching(max_admitted_runs: int = TOYHOST_LAUNCH_SETTINGS.max_admitted_runs) -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(toyhost_template())
    settings = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_admitted_runs": max_admitted_runs})
    host, _client = toyhost_launch_host(storage=storage, settings=lambda: settings)
    return host, storage


async def _launch(host: LaunchHost, **overlays: Any):
    runs = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        overlays=overlays,
    )
    async with asyncio.timeout(10):
        while any(host.job_manager.is_active(run.id) for run in runs):
            await asyncio.sleep(0.01)
    return runs


async def test_a_launch_records_the_validated_overlays_and_stamps_them_into_the_variant():
    host, storage = _launching()

    (run,) = await _launch(host, prompt_style="verbose", field_aliases={"vendor_name": "supplier"})

    stored = storage.load_eval_run(run.id, run.scope_id)
    assert stored is not None and stored.overlays == {
        "prompt_style": "verbose",
        "page_limit": 10,
        "instructions": "",
        "field_aliases": {"vendor_name": "supplier"},
    }
    levers = stored.variant_levers
    assert levers is not None
    assert levers["extractor.prompt_style"].display == "verbose"


async def test_a_refused_overlay_creates_no_run_and_gives_its_admission_back():
    """Refused at the dispatch, before any arm is prepared — and the launch that follows is admitted.

    The ceiling admits one run: if the refused launch had kept its reservation, or left a run behind
    holding a slot, the second launch would be refused for room instead of running.
    """
    host, storage = _launching(max_admitted_runs=1)

    with pytest.raises(ValidationFailedError, match="page_limit"):
        await _launch(host, page_limit=0)

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    assert host.job_manager.admitted_count == 0
    (run,) = await _launch(host, page_limit=2)
    assert run.overlays["page_limit"] == 2


async def test_a_kind_whose_contract_declares_no_overlays_refuses_any_overlay_at_launch():
    host, storage = _launching()
    spec_only = KindContract(TOY_EXTRACTOR_KIND, spec=ExtractorSpec)
    uncontracted = replace(
        host, eval_host=replace(host.eval_host, profile=replace(host.eval_host.profile, kinds=(spec_only,)))
    )

    with pytest.raises(ValidationFailedError, match="declares no overlays, so a launch cannot turn prompt_style"):
        await _launch(uncontracted, prompt_style="terse")

    assert storage.query_eval_runs(TOYHOST_SCOPE) == []
    (run,) = await _launch(uncontracted)
    assert run.overlays == {}, "a kind with no overlay model records none"


def test_a_launcher_reads_the_overlays_as_its_kinds_own_model():
    """``overlays_as`` hands back the typed model, and refuses a model the kind does not declare."""
    validated = TOY_EXTRACTOR_CONTRACT.validate_overlays({"page_limit": 4})
    request = LaunchRequest(
        template=toyhost_template(),
        kind=TOY_EXTRACTOR_KIND,
        subject_id="s",
        candidate_model=None,
        k_runs=1,
        scope_id=TOYHOST_SCOPE,
        n_variations=0,
        judge_model=None,
        judge_config_ids=None,
        simulator_model=None,
        cassette_mode="off",
        cassette_corpus_id=None,
        overlays=validated,
        kind_spec=None,
        max_cost_usd=None,
        max_metered_calls=None,
        launch_group=LaunchGroup(candidate_models=[]),
    )

    assert request.overlays_as(ExtractorOverlays).page_limit == 4

    class Other(BaseModel):
        pass

    with pytest.raises(TypeError, match="overlays are ExtractorOverlays, not Other"):
        request.overlays_as(Other)
    with pytest.raises(TypeError, match="kind spec are NoneType, not ExtractorSpec"):
        request.kind_spec_as(ExtractorSpec)


def test_a_literal_level_outside_its_declaration_cannot_be_ranked():
    """A frozen value the model's ``Literal`` does not list reads nominal rather than being given a rank."""

    class Ladder(BaseModel):
        rung: Annotated[Literal["low", "high"], Ordinal()] = Field("low", description="a two-rung ladder")

    contract = KindContract("ladder", overlays=Ladder)
    run = make_eval_run(candidate_kind="ladder", overlays={"rung": "middle"})

    assert contract.levels(run)["ladder.rung"].scale == NominalScale()


# --- the template's kind spec --------------------------------------------------------------------------


class _AliasedSpec(BaseModel):
    labels: list[str] = Field(alias="classes")


def test_a_spec_field_with_an_alias_is_refused_at_declaration_by_name():
    with pytest.raises(KindContractError, match=r"router\.labels: declares an alias"):
        KindContract("router", spec=_AliasedSpec)


@pytest.mark.parametrize(
    ("kind_spec", "named"),
    [
        ({"graded_fields": []}, "graded_fields: List should have at least 1 item"),
        ({"graded_fields": ["tip_amount"]}, "graded_fields.0: Input should be"),
        ({"graded": ["total_amount"]}, "has no kind_spec field graded; its kind_spec fields are graded_fields"),
    ],
    ids=["empty", "a-field-the-spec-does-not-grade", "a-name-the-spec-does-not-have"],
)
def test_a_spec_the_kinds_model_refuses_is_refused_naming_the_field(kind_spec, named):
    with pytest.raises(ValidationFailedError) as refused:
        TOY_EXTRACTOR_CONTRACT.validate_spec(kind_spec)

    assert named in refused.value.message


def test_a_kind_with_no_spec_model_refuses_any_spec_and_admits_none():
    with pytest.raises(ValidationFailedError, match="declares no kind_spec fields, so a template cannot state labels"):
        KindContract("bare").validate_spec({"labels": ["a"]})
    assert KindContract("bare").validate_spec({}) is None


def _authoring_checks() -> dict[str, Any]:
    """The host checks ``create_template`` asks, each admitting — the spec is what is under test."""
    return {
        "require_known_tools_allowed": lambda _tools: None,
        "refuse_undeclared_world_seed": lambda _template: None,
        "refuse_undeliverable_template": lambda _template: None,
    }


def _definition(**kind_spec: Any) -> dict[str, Any]:
    return {
        "name": "grade two fields",
        "intent": "extract and be graded on a subset of the invoice",
        "candidate_kind": TOY_EXTRACTOR_KIND,
        "kind_spec": kind_spec,
    }


def test_authoring_refuses_a_spec_the_kinds_model_refuses_naming_the_template_and_the_field():
    host = toyhost_host()

    with pytest.raises(ValidationFailedError) as refused:
        create_template(host, _definition(graded_fields=[]), scope_id=TOYHOST_SCOPE, **_authoring_checks())

    assert "template 'grade two fields'" in refused.value.message and "graded_fields" in refused.value.message
    assert host.storage.query_templates(TOYHOST_SCOPE) == [], "a refused template is not stored"


def test_authoring_stores_the_spec_the_model_resolved_and_an_omitted_spec_resolves_to_the_defaults():
    host = toyhost_host()

    stated = create_template(
        host, _definition(graded_fields=["total_amount"]), scope_id=TOYHOST_SCOPE, **_authoring_checks()
    )
    silent = create_template(
        host, {**_definition(), "name": "grade every field"}, scope_id=TOYHOST_SCOPE, **_authoring_checks()
    )

    assert host.storage.load_template(stated.id, TOYHOST_SCOPE).kind_spec == {"graded_fields": ["total_amount"]}
    assert silent.kind_spec == {"graded_fields": list(ExtractorSpec().graded_fields)}


def test_an_update_writing_the_spec_is_refused_and_one_writing_another_field_is_not():
    """Scoped to the writes that touch the spec, like every authoring guard over stored state."""
    host = toyhost_host()
    template = create_template(host, _definition(), scope_id=TOYHOST_SCOPE, **_authoring_checks())
    # A stored spec the model has since outgrown — written past authoring, as a model change would leave it.
    host.storage.save_template(template.model_copy(update={"kind_spec": {"graded_fields": ["tip_amount"]}}))

    with pytest.raises(ValidationFailedError, match="graded_fields"):
        update_template(host, template.id, TOYHOST_SCOPE, {"kind_spec": {"graded_fields": []}}, **_authoring_checks())
    renamed = update_template(
        host, template.id, TOYHOST_SCOPE, {"description": "still editable"}, **_authoring_checks()
    )
    assert renamed.description == "still editable"
    repaired = update_template(
        host, template.id, TOYHOST_SCOPE, {"kind_spec": {"graded_fields": ["vendor_name"]}}, **_authoring_checks()
    )
    assert repaired.kind_spec == {"graded_fields": ["vendor_name"]}


async def test_a_launch_freezes_the_spec_onto_its_runs_and_the_kind_grades_what_it_states():
    host, storage = _launching()
    storage.save_template(toyhost_template().model_copy(update={"kind_spec": {"graded_fields": ["invoice_number"]}}))

    (run,) = await _launch(host)

    stored = storage.load_eval_run(run.id, run.scope_id)
    assert stored is not None and stored.kind_spec == {"graded_fields": ["invoice_number"]}
    results = storage.query_eval_results_by_run(run.id, run.scope_id)
    assert results and all("of 1 graded fields exact" in result.goal_state_outcomes[0].detail for result in results), (
        "the launcher hands the kind the spec it validated, so one field is graded"
    )


async def test_a_launch_refuses_a_stored_spec_the_kinds_model_now_refuses_and_creates_no_run():
    host, storage = _launching()
    storage.save_template(toyhost_template().model_copy(update={"kind_spec": {"graded_fields": ["tip_amount"]}}))

    with pytest.raises(ValidationFailedError) as refused:
        await _launch(host)

    assert "graded_fields" in refused.value.message and toyhost_template().name in refused.value.message
    assert storage.query_eval_runs(TOYHOST_SCOPE) == []


def test_the_frozen_spec_is_part_of_the_runs_measurement_context():
    """Two runs alike but for what their templates stated are two conditions; the same spec is one."""
    profile = toyhost_profile()

    def context(**kind_spec: Any) -> str:
        validated = TOY_EXTRACTOR_CONTRACT.validate_spec(kind_spec)
        run = make_eval_run(candidate_kind=TOY_EXTRACTOR_KIND, kind_spec=freeze(validated))
        return derive_context_identity(run, profile).context_key

    assert context(graded_fields=["total_amount"]) != context(graded_fields=["vendor_name"])
    assert context() == context(graded_fields=list(ExtractorSpec().graded_fields)), (
        "a spec stating the default and one leaving it out ran the same condition"
    )
