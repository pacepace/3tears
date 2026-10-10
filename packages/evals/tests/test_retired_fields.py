"""Fields retired within schema v8 read on a stored read and nowhere else.

A field that carried nothing, or a confusable name, is retired without dropping every stored document
(``__retired_fields__``, :mod:`threetears.evals.contracts.base`). These pin, per retirement, that a document
stored before it still loads through storage, reading as the same document minus what it never meant, and
that a construction naming the old key is refused, saying what became of it.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.reporter_kind import ReporterCase, rebuild_bundle
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.declaration import CampaignDesign
from threetears.evals.contracts.hashing import canonical_digest
from packages.evals.tests.factories import make_analysis, make_campaign, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle, toyhost_design


@pytest.mark.parametrize("status", ["open", "closed"])
def test_a_stored_campaign_carrying_a_status_still_loads(status: str) -> None:
    """``status`` had no writer and nothing enforced it; a stored campaign reads with it discarded."""
    storage, store = memory_storage()
    campaign = make_campaign(run_ids=["run-1"])
    stored = campaign.to_dict()
    stored["status"] = status
    store.upsert(stored)

    loaded = storage.load_campaign(campaign.id, campaign.scope_id)

    assert loaded == campaign
    assert storage.list_campaigns(campaign.scope_id) == [campaign]
    assert "status" not in loaded.to_dict()


def test_a_campaign_has_no_status_to_set() -> None:
    assert "status" not in EvalCampaign.model_fields
    with pytest.raises(ValidationError, match="`status` is not a field of EvalCampaign: it was removed"):
        make_campaign(status="closed")


# --- the declaration's `controls` is `held_fixed` -----------------------------------------------------


def _with_old_names(design: dict[str, Any]) -> dict[str, Any]:
    """A stored declaration as it was written before the rename."""
    old = dict(design)
    old["controls"] = old.pop("held_fixed")
    return old


def test_a_stored_campaign_declaring_controls_reads_them_as_held_fixed() -> None:
    storage, store = memory_storage()
    campaign = make_campaign(declared_design=toyhost_design())
    stored = campaign.to_dict()
    stored["declared_design"] = _with_old_names(stored["declared_design"])
    store.upsert(stored)

    loaded = storage.load_campaign(campaign.id, campaign.scope_id)

    assert loaded == campaign
    assert loaded.declared_design is not None
    assert loaded.declared_design.held_fixed == toyhost_design().held_fixed
    assert "controls" not in loaded.to_dict()["declared_design"], "the old name is never written back"


def test_a_stored_analysis_whose_design_snapshot_declares_controls_still_loads() -> None:
    storage, store = memory_storage()
    analysis = make_analysis(design_snapshot=toyhost_design())
    stored = analysis.to_dict()
    stored["design_snapshot"] = _with_old_names(stored["design_snapshot"])
    store.upsert(stored)

    assert storage.load_analysis(analysis.id, analysis.scope_id) == analysis


def test_a_declaration_written_today_under_the_old_name_is_refused_naming_the_new_one() -> None:
    """A caller authoring a design now spells it `held_fixed`; `controls` is refused, saying so."""
    old = _with_old_names(toyhost_design().model_dump(mode="json"))
    with pytest.raises(
        ValidationError, match="`controls` is not a field of CampaignDesign: it was renamed `held_fixed`"
    ):
        CampaignDesign(**old)
    with pytest.raises(ValidationError, match="carries both `controls` and `held_fixed`"):
        CampaignDesign.from_dict({**old, "held_fixed": old["controls"]})


def test_a_reporter_case_frozen_before_the_rename_rebuilds_with_every_value() -> None:
    """A frozen bundle is evidence its labels were written against: the rename moves its values, losing none."""
    bundle = toyhost_bundle()
    frozen = bundle.to_dict()
    frozen["controls_reading"] = frozen.pop("held_fixed_reading")
    frozen["declared_design"] = _with_old_names(frozen["declared_design"])
    case = ReporterCase(
        bundle=frozen, bundle_fingerprint=canonical_digest(frozen), bundle_assembled_at="2026-01-01T00:00:00+00:00"
    )

    rebuilt = rebuild_bundle(case)

    assert rebuilt.held_fixed_reading == bundle.held_fixed_reading
    assert rebuilt.declared_design == bundle.declared_design
    assert rebuilt.to_dict() == bundle.to_dict()
