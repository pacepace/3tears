"""A campaign lives in one scope, and holds only runs of that scope.

Every read of a campaign's members is a read within its scope, so a member in another scope is a
run the campaign claims and no analysis of it can see. Membership outside the scope is therefore
refused at both doors that grant it — creation and ``add_runs_to_campaign`` — and each refusal is
asserted beside its acceptance on the same fixture, so a check that refused everything could not
pass for one that refuses the right thing.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import campaigns
from threetears.evals.contracts.errors import ValidationFailedError

from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.factories import memory_storage

#: The host the campaigns here are declared against; their designs name nothing of its vocabulary.
_HOST = toyhost_profile()

_HOME = "scope-a"
_ELSEWHERE = "scope-b"


def _storage_with_a_run_in_each_scope():
    storage, store = memory_storage()
    here = make_eval_run(id="run-here", scope_id=_HOME)
    there = make_eval_run(id="run-there", scope_id=_ELSEWHERE)
    storage.save_eval_run(here)
    storage.save_eval_run(there)
    return storage, store


def _definition(*run_ids: str) -> dict[str, object]:
    return {"name": "c", "subject_id": "subject-1", "behavior": "conversation", "run_ids": list(run_ids)}


class TestCreation:
    def test_a_campaign_over_runs_of_its_scope_is_created_in_that_scope(self) -> None:
        storage, _ = _storage_with_a_run_in_each_scope()

        campaign = campaigns.create_campaign(
            storage, _definition("run-here"), scope_id=_HOME, created_by="op", profile=_HOST
        )

        assert campaign.scope_id == _HOME
        assert storage.load_campaign(campaign.id, _HOME) == campaign
        assert storage.load_campaign(campaign.id, _ELSEWHERE) is None

    def test_a_run_of_another_scope_is_refused_and_nothing_is_written(self) -> None:
        storage, store = _storage_with_a_run_in_each_scope()

        with pytest.raises(ValidationFailedError, match="'run-there'.*scope 'scope-a'"):
            campaigns.create_campaign(
                storage, _definition("run-here", "run-there"), scope_id=_HOME, created_by="op", profile=_HOST
            )

        assert storage.list_campaigns(_HOME) == []
        assert {doc["doc_type"] for doc in store.documents.values()} == {"eval_run"}

    def test_the_definition_cannot_choose_its_own_scope(self) -> None:
        """``scope_id`` is server-owned: the surface names it, as it names ``created_by``."""
        storage, _ = _storage_with_a_run_in_each_scope()

        campaign = campaigns.create_campaign(
            storage, {**_definition("run-here"), "scope_id": _ELSEWHERE}, scope_id=_HOME, created_by="op", profile=_HOST
        )

        assert campaign.scope_id == _HOME


class TestAttachment:
    def test_a_run_of_the_campaign_s_scope_is_attached(self) -> None:
        storage, _ = _storage_with_a_run_in_each_scope()
        campaign = campaigns.create_campaign(storage, _definition(), scope_id=_HOME, created_by="op", profile=_HOST)

        updated = campaigns.add_runs_to_campaign(storage, campaign.id, _HOME, ["run-here"])

        assert updated.run_ids == ["run-here"]
        assert storage.load_campaign(campaign.id, _HOME).run_ids == ["run-here"]

    def test_a_run_of_another_scope_is_refused_and_the_campaign_is_left_as_it_was(self) -> None:
        storage, _ = _storage_with_a_run_in_each_scope()
        campaign = campaigns.create_campaign(
            storage, _definition("run-here"), scope_id=_HOME, created_by="op", profile=_HOST
        )

        with pytest.raises(ValidationFailedError, match="'run-there'.*scope 'scope-a'"):
            campaigns.add_runs_to_campaign(storage, campaign.id, _HOME, ["run-there"])

        assert storage.load_campaign(campaign.id, _HOME).run_ids == ["run-here"]

    def test_re_attaching_a_member_is_not_refused(self) -> None:
        """An id already held needs no re-check: it was admitted under the same rule."""
        storage, store = _storage_with_a_run_in_each_scope()
        campaign = campaigns.create_campaign(
            storage, _definition("run-here"), scope_id=_HOME, created_by="op", profile=_HOST
        )
        del store.documents[(_HOME, "run-here")]

        assert campaigns.add_runs_to_campaign(storage, campaign.id, _HOME, ["run-here"]).run_ids == ["run-here"]


def test_the_view_reads_members_in_the_campaign_s_scope() -> None:
    storage, store = _storage_with_a_run_in_each_scope()
    campaign = campaigns.create_campaign(
        storage, _definition("run-here"), scope_id=_HOME, created_by="op", profile=_HOST
    )
    del store.documents[(_HOME, "run-here")]

    view = campaigns.get_campaign_view(storage, campaign.id, _HOME)

    assert view.unresolved_run_ids == ["run-here"]
    assert view.window is None
