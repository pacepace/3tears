"""Fields retired within schema v8 read on a stored read and nowhere else.

A field that carried nothing, or a confusable name, is retired without dropping every stored document
(``__retired_fields__``, :mod:`threetears.evals.contracts.base`). These pin, per retirement, that a document
stored before it still loads through storage, reading as the same document minus what it never meant, and
that a construction naming the old key is refused, saying what became of it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.campaign import EvalCampaign
from packages.evals.tests.factories import make_campaign, memory_storage


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
