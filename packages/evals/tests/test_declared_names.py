"""Names are declared, never derived: an aspect's short name and a goal check's display name.

An aspect declares a ``short_name`` (``Aspect.short_name``) and a template a display name per goal check
(``EvalTemplate.goal_check_names``). A page prints the declared name, and where none is declared it prints the full
label or the full expression, unchanged and never shortened. Neither name is part of anything a run, a cell, a pool
or a pairing keys on, nor of any fingerprint an analysis or a reporter case is checked against: naming an aspect or a
check changes no reading and stales nothing.

**Runs from before names existed stay poolable.** :data:`PINNED` holds every identity and digest of one fixed
template and campaign as the tree at ``00107a5b7`` computed them (the commit names could first be declared on), byte
for byte. The same fixture without names, and with every name declared, computes them alike today. The stored
documents of the fixture without names are pinned too (:data:`PINNED_DOCUMENTS`): a template and a campaign that
name nothing are written exactly as before.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import assemble_context_bundle, build_campaign_report, build_code_only_report
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, evidence_digest
from threetears.evals.analysis.campaign_bundle import (
    build_campaign_bundle,
    build_campaign_user_message,
    campaign_evidence_digest,
)
from threetears.evals.analysis.campaign_readings import campaign_readings
from threetears.evals.analysis.campaigns import create_campaign, update_campaign
from threetears.evals.analysis.cells import CELL_MODEL_VERSION
from threetears.evals.analysis.report.campaign.names import AspectNames, aspect_name, legend, measure_name_of
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.declaration import Aspect, CampaignArm
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.identity import (
    IDENTITY_VERSION,
    case_set_fingerprint,
    compute_content_hash,
    derive_context_identity,
    derive_variant_identity,
)
from threetears.evals.kernel.metrics import goal_check_measure
from threetears.evals.run.definition_digests import measured_template_digest, template_digest
from threetears.evals.schema.models import EvalTemplate, GoalStateOutcome, RoleUsage
from threetears.evals.schema.versioning import DURABLE_SCHEMA_MINOR, DURABLE_SCHEMA_VERSION
from packages.evals.tests.campaign_support import ASKED_AT, ASSEMBLED_AT, QUESTION
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_aspect_campaign,
    make_eval_result,
    make_eval_run,
    make_template,
    minimal_aspect_design,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture

#: The fixture's two goal checks, each long enough that a page shortening one would show.
CHECKS = ("state.shop.cart.length >= 1", "state.chat.messages.length >= 2")
#: Its two aspects' labels, the second a near-collision of the first, as real campaigns' are.
LABELS = ("Queue discipline under a full queue", "Queue discipline under an empty queue")
#: The names the fixture declares when it declares any.
SHORT_NAMES = {LABELS[0]: "Full queue", LABELS[1]: "Empty queue"}
CHECK_NAMES = {CHECKS[0]: "Cart filled", CHECKS[1]: "Replied twice"}
ARMS = ("production", "candidate")
CASES = 6
STAMP = "2026-10-01T00:00:00+00:00"


def _template(*, named: bool) -> EvalTemplate:
    """The fixture's one template, on fixed ids and times, naming its checks where ``named``."""
    return make_template(
        id="tpl-1",
        goal_state_checks=list(CHECKS),
        created_at=STAMP,
        updated_at=STAMP,
        **({"goal_check_names": dict(CHECK_NAMES)} if named else {}),
    )


def _fixture(*, named: bool) -> dict[str, Any]:
    """The template, the runs, every aspect's bundle and the campaign bundle over them, each built from fixed ids."""
    template = _template(named=named)
    arms = [
        CampaignArm(
            label=label,
            role="control" if index == 0 else "candidate",
            variant_key=fixture_variant_key(label),
            **({"named_at": ASKED_AT} if index == 0 else {}),
        )
        for index, label in enumerate(ARMS)
    ]
    held: list[EvalCampaign] = []
    bundles: dict[str, AnalysisContextBundle] = {}
    every_run = []
    for position, label in enumerate(LABELS):
        runs = [
            make_eval_run(
                id=f"run-{position}-{arm}",
                status="completed",
                candidate_model=arm,
                test_case_ids=[f"tc-{index}" for index in range(CASES)],
                created_at=STAMP,
            )
            for arm in ARMS
        ]
        every_run += runs
        results = {
            run.id: [
                make_eval_result(
                    id=f"{run.id}-{index}",
                    eval_run_id=run.id,
                    scope_id=run.scope_id,
                    model=run.candidate_model,
                    test_case_id=f"tc-{index}",
                    scored_at=STAMP,
                    goal_state_outcomes=[
                        GoalStateOutcome(expression=CHECKS[0], passed=index < 3 + arm_index + position),
                        GoalStateOutcome(expression=CHECKS[1], passed=index != 0),
                    ],
                    cost_usd=0.002 * (arm_index + 1),
                    usage=[
                        RoleUsage(
                            role="candidate",
                            model=run.candidate_model,
                            cost_usd=0.002 * (arm_index + 1),
                            call_count=1,
                            served_model=run.candidate_model,
                        )
                    ],
                )
                for index in range(CASES)
            ]
            for arm_index, run in enumerate(runs)
        }
        one = make_aspect_campaign(
            id="campaign-names",
            label=label,
            template_id=template.id,
            run_ids=[run.id for run in runs],
            design=minimal_aspect_design(merit_axes=[]),
            question=QUESTION,
            asked_at=ASKED_AT,
            control=arms[0],
            margin=10.0,
            arms=arms,
            scope_id=runs[0].scope_id,
            name="names",
            subject_id=runs[0].subject_snapshot.subject_id,
            subject_kind="s",
            created_at=ASKED_AT,
        )
        if named:
            (aspect,) = one.aspects
            one = one.model_copy(update={"aspects": [aspect.model_copy(update={"short_name": SHORT_NAMES[label]})]})
        held.append(one)
        bundles[label] = assemble_context_bundle(
            one, storage=ToyhostStorage(runs, results, templates=[template]), profile=toyhost_profile()
        )
    campaign = held[0].model_copy(update={"aspects": [aspect for one in held for aspect in one.aspects]})
    readings = campaign_readings(campaign, bundles, arms=arms)
    return {
        "template": template,
        "runs": every_run,
        "campaign": campaign,
        "bundles": bundles,
        "readings": readings,
        "arms": arms,
        "campaign_bundle": build_campaign_bundle(
            campaign, [bundles[label] for label in LABELS], arms=arms, readings=readings
        ),
    }


def _sha(document: Any) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identities(*, named: bool) -> dict[str, str]:
    """Everything a run, a cell, a pool, a pairing, an analysis or a reporter case keys on, for the fixture."""
    built = _fixture(named=named)
    profile = toyhost_profile()
    template: EvalTemplate = built["template"]
    found: dict[str, str] = {
        "IDENTITY_VERSION": str(IDENTITY_VERSION),
        "CELL_MODEL_VERSION": str(CELL_MODEL_VERSION),
        "bundle schema_version": str(AnalysisContextBundle.model_fields["schema_version"].default),
        "durable version": f"{DURABLE_SCHEMA_VERSION}.{DURABLE_SCHEMA_MINOR}",
        "measured_template_digest": measured_template_digest(template),
        "template_digest": template_digest(template),
        "case content hash": str(compute_content_hash({"tone": "casual"})),
    }
    for run in built["runs"]:
        found[f"{run.id} context_key"] = derive_context_identity(run, profile).context_key
        found[f"{run.id} variant_key"] = derive_variant_identity(run=run, profile=profile).variant_key
        found[f"{run.id} case_set_fingerprint"] = case_set_fingerprint(run)
    for label, bundle in built["bundles"].items():
        found[f"{label} fingerprint"] = bundle.fingerprint()
        found[f"{label} evidence_digest"] = evidence_digest(bundle.to_dict())
        found[f"{label} cells"] = ",".join(
            sorted(f"{cell.variant_key}@{cell.apparatus_class_id}" for cell in bundle.cells)
        )
        found[f"{label} measure_catalog"] = _sha(
            {name: d.model_dump(mode="json") for name, d in bundle.measure_catalog.items()}
        )
    campaign_bundle = built["campaign_bundle"]
    found["readings"] = _sha(built["readings"].model_dump(mode="json"))
    found["campaign fingerprint"] = campaign_bundle.fingerprint()
    found["campaign_evidence_digest"] = campaign_evidence_digest(campaign_bundle.to_dict())
    found["writer message"] = hashlib.sha256(build_campaign_user_message(campaign_bundle).encode()).hexdigest()
    return found


def documents() -> dict[str, str]:
    """The stored documents of the fixture as written, named nothing: the template, the campaign and each bundle."""
    built = _fixture(named=False)
    return {
        "template": _sha(built["template"].to_dict()),
        "campaign": _sha(built["campaign"].to_dict()),
        # Less the build that assembled it, which is no part of the bundle's shape.
        **{
            f"{label} bundle": _sha({k: v for k, v in bundle.to_dict().items() if k != "package_version"})
            for label, bundle in built["bundles"].items()
        },
    }


#: Every identity and digest of the fixture, as the tree at ``00107a5b7`` computed them (:func:`identities`).
PINNED: dict[str, str] = {
    "CELL_MODEL_VERSION": "14",
    "IDENTITY_VERSION": "25",
    "Queue discipline under a full queue cells": (
        "b5891590ed94b7c2b7425372664a24f7f2c4d80e77a9e001ece5e0a7544a0ed9@6aa21c8030ab220f21fa1422945cfff2cfa"
        "7e62ca5159aa595e5b93112d321c9,ed8b4683c764e0d9a024892092fe2127836af2584ab8aff56143c7c791cc867c@6aa21"
        "c8030ab220f21fa1422945cfff2cfa7e62ca5159aa595e5b93112d321c9"
    ),
    "Queue discipline under a full queue evidence_digest": (
        "v2:33bebed6922837b7b39074eeeac1f2dbca69d78d4e58e9a0570c5b48e798426e"
    ),
    "Queue discipline under a full queue fingerprint": (
        "v2:33bebed6922837b7b39074eeeac1f2dbca69d78d4e58e9a0570c5b48e798426e"
    ),
    "Queue discipline under a full queue measure_catalog": (
        "be7db1240bcd8149ba17c91f78963c8632823d88d0f99674952fa7eb408d3141"
    ),
    "Queue discipline under an empty queue cells": (
        "b5891590ed94b7c2b7425372664a24f7f2c4d80e77a9e001ece5e0a7544a0ed9@6aa21c8030ab220f21fa1422945cfff2cfa"
        "7e62ca5159aa595e5b93112d321c9,ed8b4683c764e0d9a024892092fe2127836af2584ab8aff56143c7c791cc867c@6aa21"
        "c8030ab220f21fa1422945cfff2cfa7e62ca5159aa595e5b93112d321c9"
    ),
    "Queue discipline under an empty queue evidence_digest": (
        "v2:d49f240f47bd76d1a7b1dfcb2e96a3523ea1203329e9f729ba5292f508dc8a1a"
    ),
    "Queue discipline under an empty queue fingerprint": (
        "v2:d49f240f47bd76d1a7b1dfcb2e96a3523ea1203329e9f729ba5292f508dc8a1a"
    ),
    "Queue discipline under an empty queue measure_catalog": (
        "be7db1240bcd8149ba17c91f78963c8632823d88d0f99674952fa7eb408d3141"
    ),
    "bundle schema_version": "64",
    "campaign fingerprint": "v2:1896767143a9454239aebb4a33710a0d4167bd4fced79c5fd236c217a8f1834b",
    "campaign_evidence_digest": "v2:1896767143a9454239aebb4a33710a0d4167bd4fced79c5fd236c217a8f1834b",
    "case content hash": "2427afda4a091bccfdeffbdc0ef058b06862b1c310f68df406edf30604604ae2",
    "durable version": "16.2",
    "measured_template_digest": "16e80e0cbd36ae4d39ca180c5f08bcb9d165afaab0e7e014ff42464f11e4f3df",
    "readings": "29f0e8ce68005f27cfc8ff7c76ef3623a174125af4a46b8e587bb7af7d88ebc9",
    "run-0-candidate case_set_fingerprint": "89df90840e5dc430494c9d36ab9ae0dff05c199d54ddf7027f3f339c44bcfe56",
    "run-0-candidate context_key": "bdcd277ef947176d0cf5a586ccaf20fead3df0127fe4408595e4b90b8e8b49c0",
    "run-0-candidate variant_key": "8555bc22f5cbb0bf7c72b3c21f66e8876da90c33fedeb9ffd9d7b4571164c0f3",
    "run-0-production case_set_fingerprint": "89df90840e5dc430494c9d36ab9ae0dff05c199d54ddf7027f3f339c44bcfe56",
    "run-0-production context_key": "bdcd277ef947176d0cf5a586ccaf20fead3df0127fe4408595e4b90b8e8b49c0",
    "run-0-production variant_key": "948034e5bf867d07c3ba1cafab809575218262d649cf00973135e0be56068b6d",
    "run-1-candidate case_set_fingerprint": "89df90840e5dc430494c9d36ab9ae0dff05c199d54ddf7027f3f339c44bcfe56",
    "run-1-candidate context_key": "bdcd277ef947176d0cf5a586ccaf20fead3df0127fe4408595e4b90b8e8b49c0",
    "run-1-candidate variant_key": "8555bc22f5cbb0bf7c72b3c21f66e8876da90c33fedeb9ffd9d7b4571164c0f3",
    "run-1-production case_set_fingerprint": "89df90840e5dc430494c9d36ab9ae0dff05c199d54ddf7027f3f339c44bcfe56",
    "run-1-production context_key": "bdcd277ef947176d0cf5a586ccaf20fead3df0127fe4408595e4b90b8e8b49c0",
    "run-1-production variant_key": "948034e5bf867d07c3ba1cafab809575218262d649cf00973135e0be56068b6d",
    "template_digest": "2508d494320bd907e2d0bfeab1a6b53b37ccbcf645462cd754d319b9852f4553",
    "writer message": "132c698dc3c7b1c3764cd8e1ea59f230ae0353243aa7bd4d094eda7267396c0b",
}

#: The fixture's stored documents naming nothing, as the tree at ``00107a5b7`` wrote them (:func:`documents`).
PINNED_DOCUMENTS: dict[str, str] = {
    "Queue discipline under a full queue bundle": "299bc62becdd7b685e6ff99a22b9d53c5d6fb935e9f124bc327f79d0714335ef",
    "Queue discipline under an empty queue bundle": "d15688c9ff1aac1a11302e5ff35368cdeedf8f6f5a697bea95deae05f667ca4d",
    "campaign": "6c2b383b36fc2664a2457021aac7d61d330ed17119760d15694105bb2a99c215",
    "template": "ebdd07699d7ce094a952d8f66505439368260a790589e08f69ae10ad3c10c755",
}


# --- runs from before names existed stay poolable -----------------------------------------------------------------


class TestEveryIdentityIsAsBeforeNamesExisted:
    """No identity, key, digest or fingerprint moves, whether the fixture names nothing or names everything."""

    def test_a_template_and_campaign_naming_nothing_key_as_before(self) -> None:
        assert identities(named=False) == PINNED

    def test_naming_every_aspect_and_check_moves_no_identity(self) -> None:
        assert identities(named=True) == PINNED

    def test_a_template_campaign_and_bundle_naming_nothing_are_written_as_before(self) -> None:
        assert documents() == PINNED_DOCUMENTS

    def test_the_names_are_carried_though_no_digest_reads_them(self) -> None:
        built = _fixture(named=True)
        assert built["template"].to_dict()["goal_check_names"] == CHECK_NAMES
        assert [aspect.short_name for aspect in built["campaign"].aspects] == [SHORT_NAMES[label] for label in LABELS]
        for label, bundle in built["bundles"].items():
            assert bundle.to_dict()["short_name"] == SHORT_NAMES[label]
            assert bundle.goal_check_names == CHECK_NAMES


# --- an aspect's short name ---------------------------------------------------------------------------------------


class TestAnAspectsShortName:
    def test_one_of_at_most_the_cap_in_characters_is_kept_in_any_script(self) -> None:
        arabic = "ا" * 32
        assert Aspect(label="x", short_name=arabic).short_name == arabic
        assert Aspect(label="x", short_name="b" * 32).short_name == "b" * 32

    @pytest.mark.parametrize("name", ["b" * 33, "  ", ""], ids=["over the cap", "blank", "empty"])
    def test_an_over_long_or_blank_one_is_refused(self, name: str) -> None:
        with pytest.raises(ValidationError, match="aspect 'x'"):
            Aspect(label="x", short_name=name)

    def test_none_is_written_as_no_key_at_all(self) -> None:
        assert "short_name" not in Aspect(label="x").model_dump()

    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(("Queue", "Queue"), id="two alike"),
            pytest.param(("Empty", None), id="one printing the other's label"),
        ],
    )
    def test_two_aspects_printed_alike_are_refused(self, names: tuple[str | None, str | None]) -> None:
        campaign = _fixture(named=False)["campaign"]
        aspects = [
            aspect.model_copy(update={"label": label, "short_name": name})
            for aspect, label, name in zip(campaign.aspects, ("Full", "Empty"), names, strict=True)
        ]
        with pytest.raises(ValidationError, match="would print for more than one aspect"):
            EvalCampaign.model_validate({**campaign.model_dump(), "aspects": [a.model_dump() for a in aspects]})


class TestAShortNameAtDeclareAndAdopt:
    """Declared on create, edited on a campaign whose declaration is frozen, carried by adoption, refused when bad."""

    def test_it_is_edited_on_a_frozen_campaign_and_stales_no_analysis(self) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        assert fixture.campaign.run_ids, "the corpus campaign holds its runs, so its declaration is frozen"
        before = assemble_context_bundle(fixture.campaign, storage=host.storage, profile=host.profile)
        (aspect,) = fixture.campaign.aspects

        named = update_campaign(
            host.storage,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            {"short_names": {aspect.label: "Invoices"}},
            updated_by="op",
            profile=host.profile,
        )

        assert [each.short_name for each in named.aspects] == ["Invoices"]
        assert named.declared_at == fixture.campaign.declared_at, "naming is no declaration: nothing is re-dated"
        after = assemble_context_bundle(named, storage=host.storage, profile=host.profile)
        assert aspect_name(after) == "Invoices"
        assert after.fingerprint() == before.fingerprint()
        cleared = update_campaign(
            host.storage,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            {"short_names": {aspect.label: None}},
            updated_by="op",
            profile=host.profile,
        )
        assert cleared.aspects[0].short_name is None
        assert cleared.to_dict()["aspects"] == fixture.campaign.to_dict()["aspects"], "cleared, it is stored as before"

    @pytest.mark.parametrize(
        ("short_names", "refusal"),
        [
            pytest.param({"no such aspect": "x"}, "which the campaign has no aspect for", id="unknown label"),
            pytest.param({"extract_invoice_fields": "b" * 33}, "at most 32", id="over the cap"),
            pytest.param(["extract_invoice_fields"], "maps an aspect's label", id="not a map"),
        ],
    )
    def test_a_bad_one_is_refused_on_update(self, short_names: object, refusal: str) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        with pytest.raises(ValidationFailedError, match=refusal):
            update_campaign(
                host.storage,
                fixture.campaign.id,
                TOYHOST_SCOPE,
                {"short_names": short_names},
                updated_by="op",
                profile=host.profile,
            )

    def test_adoption_carries_it(self) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        (aspect,) = fixture.campaign.aspects
        update_campaign(
            host.storage,
            fixture.campaign.id,
            TOYHOST_SCOPE,
            {"short_names": {aspect.label: "Invoices"}},
            updated_by="op",
            profile=host.profile,
        )

        adopted = create_campaign(
            host.storage,
            {"name": "adopted", "subject_id": fixture.campaign.subject_id, "behavior": fixture.campaign.behavior},
            scope_id=TOYHOST_SCOPE,
            created_by="op",
            profile=host.profile,
            aspects_from=[fixture.campaign.id],
        )

        assert [each.short_name for each in adopted.aspects] == ["Invoices"]

    def test_a_create_declares_it_and_refuses_an_over_long_one(self) -> None:
        fixture = ops_fixture()
        host = fixture.host.eval_host
        definition = {
            "name": "named",
            "subject_id": fixture.campaign.subject_id,
            "behavior": fixture.campaign.behavior,
            "aspects": [{"label": "one", "short_name": "First"}, {"label": "two", "short_name": "b" * 33}],
        }
        with pytest.raises(ValidationFailedError, match="at most 32"):
            create_campaign(
                host.storage,
                definition,
                scope_id=TOYHOST_SCOPE,
                created_by="op",
                profile=host.profile,
                exploratory=True,
            )
        definition["aspects"] = [{"label": "one", "short_name": "First"}, {"label": "two"}]
        created = create_campaign(
            host.storage, definition, scope_id=TOYHOST_SCOPE, created_by="op", profile=host.profile, exploratory=True
        )
        assert [each.short_name for each in created.aspects] == ["First", None]


# --- a goal check's display name ----------------------------------------------------------------------------------


class TestAGoalChecksDisplayName:
    def test_it_is_kept_for_a_check_the_template_declares(self) -> None:
        assert _template(named=True).goal_check_names == CHECK_NAMES

    @pytest.mark.parametrize(
        ("names", "refusal"),
        [
            pytest.param({"state.other >= 1": "Other"}, "is not a goal check", id="not its check"),
            pytest.param({CHECKS[0]: "Same", CHECKS[1]: "Same"}, "to more than one check", id="one name twice"),
            pytest.param({CHECKS[0]: "b" * 33}, "at most 32", id="over the cap"),
            pytest.param({CHECKS[0]: " "}, "is blank", id="blank"),
        ],
    )
    def test_a_bad_one_is_refused(self, names: dict[str, str], refusal: str) -> None:
        with pytest.raises(ValidationError, match=refusal):
            make_template(goal_state_checks=list(CHECKS), goal_check_names=names)

    def test_naming_a_check_changes_neither_template_digest(self) -> None:
        unnamed, named = _template(named=False), _template(named=True)
        assert measured_template_digest(named) == measured_template_digest(unnamed)
        assert template_digest(named) == template_digest(unnamed)


# --- what the page prints -----------------------------------------------------------------------------------------


def _page(*, named: bool) -> Any:
    built = _fixture(named=named)
    reports = {
        label: build_code_only_report(
            bundle, assembled_at=ASSEMBLED_AT, campaign_name="names", arm_labels=built["readings"].arm_labels()
        )
        for label, bundle in built["bundles"].items()
    }
    return built, build_campaign_report(built["campaign"], built["readings"], reports, None)


def _cells(page: Any) -> list[str]:
    """Every table cell's value on the campaign's own page, its aspect folds left out."""
    return [
        str(cell.value) for block in page.blocks if block.kind == "table" for row in block.rows for cell in row.values()
    ]


class TestThePagePrintsDeclaredNames:
    def test_every_table_cell_names_an_aspect_by_its_short_name(self) -> None:
        _, page = _page(named=True)
        cells = _cells(page)
        assert any(cell in SHORT_NAMES.values() for cell in cells), "the page names its aspects in its tables"
        assert not [cell for cell in cells if any(label in cell for label in LABELS)]

    def test_the_full_label_stays_in_each_fold_header(self) -> None:
        _, page = _page(named=True)
        assert [fold.aspect for fold in page.aspects] == list(LABELS)
        markdown = campaign_report_markdown(page)
        assert all(label in markdown for label in LABELS)

    def test_with_no_name_every_cell_prints_the_full_label_unchanged(self) -> None:
        _, page = _page(named=False)
        cells = _cells(page)
        assert any(cell in LABELS for cell in cells)
        assert not [cell for cell in cells if any(name in cell for name in SHORT_NAMES.values())]

    def test_the_legend_pairs_each_short_name_with_its_label(self) -> None:
        named, _ = _page(named=True)
        unnamed, _ = _page(named=False)
        names = AspectNames.of(named["bundles"][label] for label in LABELS)
        assert legend(names) == [(SHORT_NAMES[label], label) for label in LABELS]
        assert legend(AspectNames.of(unnamed["bundles"][label] for label in LABELS)) == []


class TestAnUndeclaredNameFallsBackWhole:
    def test_an_aspect_with_no_short_name_is_its_label_never_shortened(self) -> None:
        unnamed = _fixture(named=False)["bundles"][LABELS[0]]
        long = unnamed.model_copy(update={"aspect": "An aspect label far longer than any short name could be"})
        assert aspect_name(long) == "An aspect label far longer than any short name could be"
        assert AspectNames.of([long])("an aspect no evidence was given for") == "an aspect no evidence was given for"

    def test_a_check_with_no_display_name_is_its_whole_expression_as_code(self) -> None:
        unnamed = _fixture(named=False)["bundles"][LABELS[0]]
        named = _fixture(named=True)["bundles"][LABELS[0]]
        measure = goal_check_measure(CHECKS[0])
        assert measure_name_of(unnamed, measure) == f"`{CHECKS[0]}`"
        assert measure_name_of(named, measure) == CHECK_NAMES[CHECKS[0]]

    def test_measure_name_prints_a_declared_name_and_never_the_rate_name(self) -> None:
        measure = goal_check_measure(CHECKS[0])
        assert measure_name(measure, "Cart filled") == "Cart filled"
        assert measure_name(measure, f"Pass rate of {CHECKS[0]}") == f"`{CHECKS[0]}`"
        assert measure_name(measure) == f"`{CHECKS[0]}`"
