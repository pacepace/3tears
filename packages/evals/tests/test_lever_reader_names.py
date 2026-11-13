"""Every lever carries a declared, stored reader name, and every surface names a lever from it (owner, 2026-11-11).

A host declares what a reader calls each lever (``Sweepable.reader_name``, a kind field's ``title``), and a lever
declared without one is refused where it is declared, as a host measure without a ``reader_name`` is. A campaign stores
the name of each axis it declares (``SweptAxis.reader_name``): the author's, else the host's, and the quick path's is
the factor its author wrote. Pages read the stored name through the one source
(:func:`~threetears.evals.analysis.style.lever_reader_name` over a bundle's ``lever_names``), so a page rebuilt from the
store names a lever as the page built in process did, even under a host that no longer declares it.

The name is presentation, never identity: it enters no variant key, no measurement context, no host declarations
digest and no fingerprint, so renaming a lever splits no history (``test_declared_names.py`` pins every identity of a
campaign that names its lever).

Mutations that turn this file red: the registry admitting a lever with no reader name, or a family with no member
namer; a create storing an axis unnamed where its host names the lever, or overwriting an authored name; the quick path
filing the engine's key (``callable.prompt``) as the name; a bundle naming a lever from its id where the store names it;
a reader name reaching a variant key, a context key or the host declarations digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.analysis import host_declarations_digest
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.service import campaign_report
from threetears.evals.analysis.campaign_title import declared_title, title_arm_name
from threetears.evals.kernel.declaration import (
    AspectDesign,
    CampaignArm,
    ControlDeclaration,
    SweptAxis,
    name_declared_axes,
)
from threetears.evals.kernel.host import RegistrationError, Sweepable, SweepableRegistry
from threetears.evals.kernel.identity import derive_context_identity, derive_variant_identity
from threetears.evals.quick.one_call import callable_host
from threetears.evals.analysis.style import lever_reader_name
from threetears.evals.schema import SweepableValue
from threetears.evals.schema.campaign_upgrade import upgrade_campaign_v16
from threetears.evals.schema.lever_words import lever_id_words
from threetears.evals.storage import SqliteDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop

SCOPE = "lever-names"

CASES = [{"text": f"case {n}", "label": "billing" if n % 2 else "bug"} for n in range(8)]


async def billing(case: Mapping[str, Any]) -> str:
    return "billing"


async def labelled(case: Mapping[str, Any]) -> str:
    return str(case["label"])


#: A factorial over a model and a prompt, the control first: what an eval's author writes.
ARMS = {
    "model=a, prompt=v1": billing,
    "model=a, prompt=v2": labelled,
    "model=b, prompt=v1": labelled,
    "model=b, prompt=v2": labelled,
}


def _lever(**fields: Any) -> Sweepable:
    return Sweepable(
        **{"name": "depth", "role": "lever", "read": lambda _run, _results: 1, "reader_prose": "x"} | fields
    )


# --- a host declares the name, or is refused -------------------------------------------------------------------------


def test_a_host_lever_with_no_reader_name_is_refused_naming_it_and_how_to_declare_it() -> None:
    with pytest.raises(RegistrationError) as refused:
        SweepableRegistry([_lever()])

    said = str(refused.value)
    assert "lever depth has no reader_name" in said
    assert "an arm's name, a setting or a factor would print its id" in said
    assert "declare Sweepable(reader_name=...) with the words a reader calls it" in said


@pytest.mark.parametrize(
    ("fields", "said"),
    [
        ({"reader_name": " "}, "depth declares a blank reader_name"),
        (
            {"open_family": "any entry", "owns_member": lambda name: name.startswith("depth.")},
            "lever family depth has no member_reader_name",
        ),
        ({"reader_name": "depth", "member_reader_name": str}, "depth declares member_reader_name and is not an open"),
    ],
    ids=["blank", "family-without-a-namer", "namer-on-a-fixed-lever"],
)
def test_a_name_no_page_could_print_is_refused_where_it_is_declared(fields: dict[str, Any], said: str) -> None:
    with pytest.raises(RegistrationError, match=said):
        SweepableRegistry([_lever(**fields)])


def test_an_apparatus_input_needs_no_reader_name() -> None:
    registry = SweepableRegistry(
        [_lever(name="bench", role="apparatus", confounds="a different bench measured it", reader_name=None)]
    )
    assert registry.reader_name_of("bench") is None


def test_the_registry_names_a_fixed_lever_a_kind_field_and_a_family_member_as_declared() -> None:
    registry = toyhost_profile(tunable_retrieval=True).sweepables

    assert registry.reader_name_of("model") == "model"
    assert registry.reader_name_of("chunk_tokens") == "chunk tokens"
    # A kind's field by its title (the toy host's say `extractor`), never by its id; a map's entry by the title and
    # the key as the launch wrote it; a family's member as the family names it.
    assert registry.reader_name_of("extractor.prompt_style") == "extractor prompt style"
    assert registry.reader_name_of("extractor.field_aliases.total") == "extractor field aliases total"
    assert registry.reader_name_of("retrieval.rerank_depth") == "rerank depth"
    assert registry.reader_name_of("no_such_lever") is None


@pytest.mark.parametrize("given", [" ", "", None], ids=["blank", "empty", "no-text"])
def test_a_family_naming_a_member_with_no_words_is_refused_where_it_is_asked_naming_both(given: Any) -> None:
    """A family's member_reader_name is a callable, so what it returns is seen only when a member is named: a blank
    stored the axis unnamed, and every page printed the member's id. It is refused there, naming family and member."""
    registry = SweepableRegistry(
        [
            _lever(
                name="retrieval",
                open_family="any retrieval setting",
                owns_member=lambda name: name.startswith("retrieval."),
                member_reader_name=lambda _name: given,
            )
        ]
    )
    with pytest.raises(RegistrationError, match="lever family retrieval's member_reader_name named its member "):
        registry.reader_name_of("retrieval.rerank_depth")


# --- a campaign stores the name --------------------------------------------------------------------------------------


def _design(*axes: SweptAxis) -> AspectDesign:
    return AspectDesign(axes=list(axes), held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"))


def _axis(axis_id: str, reader_name: str | None = None) -> SweptAxis:
    return SweptAxis(axis_id=axis_id, reader_name=reader_name, values=[SweepableValue.of("v", display="v")])


def test_a_declared_axis_takes_its_hosts_name_and_keeps_its_authors() -> None:
    named = name_declared_axes(
        _design(_axis("extractor.prompt_style"), _axis("chunk_tokens", "Chunk size"), _axis("no_such_lever")),
        toyhost_profile(),
    )

    assert [axis.reader_name for axis in named.axes] == ["extractor prompt style", "Chunk size", None]


def test_a_blank_authored_name_is_refused() -> None:
    with pytest.raises(ValueError, match="reader_name"):
        _axis("model", "  ")


async def test_the_quick_path_stores_each_factor_as_its_axis_name() -> None:
    result = await Eval(CASES, ARMS, ["label"], k=1, name="model and prompt").run(store=":memory:", scope=SCOPE)
    campaign = off_loop(result.host.storage.load_campaign, result.campaign_id, SCOPE)

    assert campaign is not None and campaign.aspects[0].design is not None
    assert [(axis.axis_id, axis.reader_name) for axis in campaign.aspects[0].design.axes] == [
        ("model", "model"),
        ("callable.prompt", "prompt"),
    ]


# --- every surface names a lever from the store ----------------------------------------------------------------------


async def test_a_page_rebuilt_from_the_store_names_each_lever_as_the_page_built_in_process(tmp_path: Path) -> None:
    """The quick path's prompt lever is declared by the eval that ran it; a host reopening the store declares none, so
    the page it rebuilds names the lever from the campaign's stored axis, as the in-process page did."""
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as store:
        result = await Eval(CASES, ARMS, ["label"], k=1, name="model and prompt").run(store=store, scope=SCOPE)
        in_process = result.page
    with SqliteDocumentStore(path) as reopened:
        host = callable_host(store=reopened)
        assert host.profile.sweepables.reader_name_of("callable.prompt") is None, "the reopening host declares none"
        rebuilt = off_loop(campaign_report, host, result.campaign_id, SCOPE)

    # Every lever the arms ran at is named alike; the other callable kind's lever, which no arm ran at (each arm sits
    # at its "not this kind" level, which no page prints), is named only by the host that declares it.
    ran = ("model", "callable.prompt")
    named = [{lever: fold.report.bundle.lever_names[lever] for lever in ran} for fold in in_process.aspects]
    assert named == [{lever: fold.report.bundle.lever_names[lever] for lever in ran} for fold in rebuilt.aspects]
    assert named == [{"model": "model", "callable.prompt": "prompt"}], "named as the eval's author wrote the factor"
    # Each page names the lever by the stored name, never by the engine's key read as words. The two pages are not
    # otherwise one text: a host that does not declare the lever reads its coverage from the runs alone, and its remedy
    # quotes the key a host would declare.
    for page in (campaign_report_markdown(in_process), campaign_report_markdown(rebuilt)):
        assert "changing the prompt" in page
        assert "callable prompt" not in page
    assert "callable" not in campaign_report_markdown(in_process)


def test_a_title_and_a_listing_spell_a_code_made_arm_with_its_levers_stored_names() -> None:
    """A title, a listing and its "Adopt ..." line named an arm code spelled from several levers by their ids read
    as words (``Callable prompt v2, model b``) while the page spelled it with their stored names."""
    # Built unchecked: the name is read off the label and the levers alone, and a real arm's key needs a run to address.
    arm = CampaignArm.model_construct(
        label="callable.prompt=v2, model=b",
        role="candidate",
        variant_key="b2" * 32,
        levers={"callable.prompt": SweepableValue.of("v2", display="v2"), "model": SweepableValue.of("b", display="b")},
        short_name=None,
    )
    declared = {"callable.prompt": "prompt", "model": "model"}

    assert title_arm_name(arm, given=(), declared=declared) == "Prompt v2, model b"
    assert title_arm_name(arm, given=()) == "Callable prompt v2, model b", "with no stored names, the ids as words"
    assert declared_title("grid", [arm], [("label", None)], given=(), declared=declared).startswith(
        "Prompt v2, model b"
    )


@pytest.mark.parametrize("lever", ["max_output_tokens", "extractor.prompt_style", "callable.prompt", "model"])
def test_one_rule_reads_a_lever_s_id_as_words_for_a_page_and_for_the_v16_step(lever: str) -> None:
    """Two rules turned a lever's id into words, a page's and the v16 step's, each claiming to be the one place. One
    function owns it now; the step only drops the quick path's kind namespace before it."""
    assert lever_reader_name(lever) == lever_id_words(lever)
    stored = {"aspects": [{"design": {"axes": [{"axis_id": lever}]}}]}
    (axis,) = upgrade_campaign_v16(stored)["aspects"][0]["design"]["axes"]
    assert axis["reader_name"] == lever_id_words(lever.removeprefix("callable."))


# --- the name is no identity -----------------------------------------------------------------------------------------


def test_a_host_renaming_its_levers_moves_no_variant_key_context_key_or_declarations_digest() -> None:
    profile = toyhost_profile()
    host = profile.host_sweepables
    renamed = replace(
        profile,
        host_sweepables=SweepableRegistry(
            [
                replace(declared, reader_name=f"renamed {declared.name}") if declared.role == "lever" else declared
                for declared in host.declarations
            ],
            roles=host.roles,
        ),
    )
    campaign, storage = toyhost_campaign(profile=profile)
    runs = storage.load_eval_runs(campaign.run_ids, campaign.scope_id)

    assert renamed.sweepables.reader_name_of("chunk_tokens") == "renamed chunk_tokens"
    assert host_declarations_digest(renamed) == host_declarations_digest(profile)
    for run in runs:
        assert derive_variant_identity(run=run, profile=renamed) == derive_variant_identity(run=run, profile=profile)
        assert derive_context_identity(run, renamed) == derive_context_identity(run, profile)
