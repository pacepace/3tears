"""A sweep's help names a way to declare its campaign's design that the freeze rule allows.

A sweep's campaign is exploratory and holds its runs, so its declaration is frozen: ``campaign_update`` refuses a
design there. The help once told the caller to declare one that way. It now names ``campaign_create`` over the sweep's
runs, and this reads that help against the rule: the call it names succeeds, and the one it must not name is refused.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.campaigns import create_campaign, update_campaign
from threetears.evals.kernel.errors import ConflictError
from threetears.evals.ops import sweeps
from threetears.evals.ops.sweeps import SweepSettings
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, ops_fixture


def _help_texts() -> list[str]:
    """Every place a sweep's help speaks of declaring a design after the runs."""
    field = SweepSettings.model_fields["campaign_name"].description or ""
    module = sweeps.__doc__ or ""
    section = module.split("**Its campaign is exploratory.**", 1)[1].split("**When it stops early.**", 1)[0]
    return [field, section]


def _exploratory_sweep_campaign(fixture: OpsFixture) -> Any:
    """The corpus campaign as a sweep leaves its own: exploratory, holding its runs."""
    (aspect,) = fixture.campaign.aspects
    swept = fixture.campaign.model_copy(
        update={"aspects": [aspect.model_copy(update={"design": None})], "arms": [], "floors": [], "question": ""}
    )
    fixture.host.eval_host.storage.save_campaign(swept)
    return swept


def test_the_help_does_not_send_the_caller_to_a_call_the_freeze_refuses() -> None:
    fixture = ops_fixture()
    swept = _exploratory_sweep_campaign(fixture)
    host = fixture.host.eval_host
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None

    with pytest.raises(ConflictError, match="what it declared is frozen"):
        update_campaign(
            host.storage,
            swept.id,
            TOYHOST_SCOPE,
            {"aspects": [{"label": aspect.label, "design": aspect.design.model_dump(mode="json")}]},
            updated_by="op",
            profile=host.profile,
        )

    for text in _help_texts():
        assert "campaign_create" in text
        assert "run_ids" in text and "from_run_ids" in text
        assert "afterwards with campaign_update" not in text and "with campaign_update." not in text


def test_the_call_the_help_names_declares_the_swept_runs_a_design() -> None:
    fixture = ops_fixture()
    swept = _exploratory_sweep_campaign(fixture)
    host = fixture.host.eval_host
    (aspect,) = fixture.campaign.aspects
    assert aspect.design is not None and len(swept.run_ids) >= 2

    declared = create_campaign(
        host.storage,
        {
            "name": "the sweep, declared",
            "subject_id": swept.subject_id,
            "behavior": swept.behavior,
            "question": "does the second arm do better?",
            "arms": [{"label": "control", "role": "control", "from_run_ids": [swept.run_ids[0]]}],
            "aspects": [
                {
                    "label": aspect.label,
                    "run_ids": list(swept.run_ids),
                    "margin": 5.0,
                    "design": aspect.design.model_dump(mode="json"),
                }
            ],
        },
        scope_id=TOYHOST_SCOPE,
        created_by="op",
        profile=host.profile,
    )

    assert declared.id != swept.id
    assert set(declared.run_ids) == set(swept.run_ids)
    assert declared.question == "does the second arm do better?"
