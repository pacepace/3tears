"""The curation family answers without an ``EvalService`` — the property that makes it a seam.

The host's own suite owns the behaviour: what archive protects, what delete cascades to,
what a partial failure reports. It exercises all of it through the host's service, which is
right — that is how a host's REST and MCP surfaces reach these functions.

This file pins something the behaviour suite structurally cannot: that
:mod:`threetears.evals.run.curation` is reachable *without* constructing a service, over
nothing but a storage double. That is the difference between an extraction and a
file split, and it is invisible to every test that goes through the service —
which would keep passing if the family were folded straight back into the class.
Without a test that never mentions ``EvalService``, "the family is exercisable on
its own" is a claim about the code's shape rather than a demonstrated property.

Detaching a run from a campaign is the campaign family's
(:func:`threetears.evals.analysis.campaigns.remove_runs_from_campaign`), and its seam is pinned here
beside the delete cascade's own detach, because the two edit the same membership under the same lock.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.evals.analysis import campaigns
from threetears.evals.contracts.errors import NotFoundError, StorageError, ValidationFailedError
from threetears.evals.run import curation
from threetears.evals.contracts.storage import EvalStorage
from packages.evals.tests.factories import minimal_declaration
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


SCOPE = "uni-1"


def _storage_double() -> MagicMock:
    """A storage stand-in that succeeds at everything.

    ``spec=EvalStorage`` so a call to a method storage does not have fails here
    rather than passing silently against a permissive mock.
    """
    storage = MagicMock(spec=EvalStorage)
    storage.query_eval_results_by_run.return_value = []
    storage.list_campaigns.return_value = []
    storage.delete_eval_result.return_value = True
    storage.delete_eval_run.return_value = True
    storage.delete_analysis.return_value = True
    storage.delete_insight.return_value = True
    storage.save_campaign.return_value = True
    return storage


def _run(run_id: str = "doomed", *, status: str = "completed") -> Any:
    """A run stand-in carrying only what the delete path reads."""
    run = MagicMock()
    run.id = run_id
    run.status = status
    return run


# ---------------------------------------------------------------------------
# The confirmation contract — a pure function, no storage of any kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confirm", [None, "", "yes", "DOOMED", "doome"])
def test_confirmation_refuses_anything_that_is_not_the_id(confirm):
    with pytest.raises(ValidationFailedError) as excinfo:
        curation.require_delete_confirmation("run", "doomed", confirm)
    # The refusal names the id to echo, so the caller can act on it without
    # going and reading the source of the check.
    assert "confirm='doomed'" in str(excinfo.value.message)


def test_confirmation_accepts_the_echoed_id_ignoring_surrounding_whitespace():
    curation.require_delete_confirmation("run", "doomed", "  doomed  ")


def test_confirmation_defaults_to_offering_the_archive_alternative():
    """The default is what every archivable eval object should keep saying."""
    with pytest.raises(ValidationFailedError) as excinfo:
        curation.require_delete_confirmation("run", "doomed", None)
    assert "archive it instead to exclude it from cohorts without destroying it" in str(excinfo.value.message)


def test_confirmation_names_the_cascade_when_one_is_supplied():
    """A caller told only the id is weighing the wrong object."""
    with pytest.raises(ValidationFailedError) as excinfo:
        curation.require_delete_confirmation("case", "doomed", None, cascade="and its 4 stored eval results")
    assert "deleting case 'doomed' and its 4 stored eval results is unrecoverable" in str(excinfo.value.message)


def test_the_archivable_set_named_by_the_default_alternative_matches_the_models():
    """The default refusal says "archive it instead"; these are the objects that can.

    A canary rather than a comment, because a comment naming a set goes stale the first
    time the set changes — and this one had. It claimed the property was "true of runs,
    results, analyses and insights", which was wrong about two of the four, and that
    wrongness reached the live refusal text for both: a caller refused a result delete or
    an insight delete was told to archive an object with no ``archived`` field and no
    writer for one.

    If a model here gains or loses ``archived``, this fails — and the fix is to move it
    between the two groups AND revisit whichever ``delete_*`` sends its callers to the
    default alternative.
    """
    from threetears.evals.contracts.campaign import EvalAnalysis, EvalCampaign, EvalInsight
    from threetears.evals.contracts.models import EvalResult, EvalRun

    archivable = {EvalRun, EvalAnalysis, EvalCampaign}
    not_archivable = {EvalResult, EvalInsight}

    assert all("archived" in model.model_fields for model in archivable)
    assert not any("archived" in model.model_fields for model in not_archivable)


def test_a_refused_result_delete_is_pointed_at_the_run_archive_not_a_result_one():
    """There is no per-result archive; the reversible move exists one level up."""
    storage = _storage_double()
    result = MagicMock()
    result.id = "res-1"
    result.eval_run_id = "run-1"

    with pytest.raises(ValidationFailedError) as excinfo:
        curation.delete_result(storage, result, SCOPE, confirm=None)

    message = str(excinfo.value.message)
    assert "archive the whole run instead" in message
    assert "there is no per-result archive" in message


def test_a_refused_insight_delete_is_not_offered_an_archive_that_does_not_exist():
    """An insight has no archive, and the refusal says why rather than implying a gap.

    The ANALYSIS shown false is archived, and the insights it minted are deleted. So delete
    is the answer here, and the message must not send a caller looking for a surface that
    does not exist.
    """
    storage = _storage_double()
    storage.load_insight.return_value = MagicMock(id="ins-1")

    with pytest.raises(ValidationFailedError) as excinfo:
        curation.delete_insight(storage, "ins-1", SCOPE, confirm=None)

    message = str(excinfo.value.message)
    assert "an insight has no archive" in message
    assert "archive it instead to exclude it from cohorts" not in message


def test_confirmation_takes_a_replacement_alternative_for_an_unarchivable_object():
    """Pointing a refused caller at a surface the object does not have is worse than silence."""
    with pytest.raises(ValidationFailedError) as excinfo:
        curation.require_delete_confirmation("case", "doomed", None, alternative="edit it in place")
    message = str(excinfo.value.message)
    assert "or edit it in place." in message
    assert "archive it instead" not in message


# ---------------------------------------------------------------------------
# The family, driven with no service in sight
# ---------------------------------------------------------------------------


def test_delete_run_destroys_results_before_the_run_document():
    """Order matters: a run that outlived its results is recoverable state; results
    that outlived their run are unreadable rows that still enter every aggregate.
    """
    storage = _storage_double()
    result = MagicMock()
    result.id = "res-1"
    storage.query_eval_results_by_run.return_value = [result]
    calls: list[str] = []
    storage.delete_eval_result.side_effect = lambda *a, **k: calls.append("result") or True
    storage.delete_eval_run.side_effect = lambda *a, **k: calls.append("run") or True

    removed = curation.delete_run(storage, _run(), SCOPE, confirm="doomed")

    assert calls == ["result", "run"]
    assert removed == {"run_id": "doomed", "results_deleted": 1, "campaigns_detached": []}


def test_delete_run_refuses_a_live_run_and_destroys_nothing():
    storage = _storage_double()

    with pytest.raises(ValidationFailedError):
        curation.delete_run(storage, _run(status="running"), SCOPE, confirm="doomed")

    storage.delete_eval_result.assert_not_called()
    storage.delete_eval_run.assert_not_called()


def test_a_failed_result_delete_leaves_the_run_document_alone():
    storage = _storage_double()
    result = MagicMock()
    result.id = "res-1"
    storage.query_eval_results_by_run.return_value = [result]
    storage.delete_eval_result.return_value = False

    with pytest.raises(StorageError):
        curation.delete_run(storage, _run(), SCOPE, confirm="doomed")

    storage.delete_eval_run.assert_not_called()


def test_a_refused_result_delete_names_its_cause_and_leaves_the_run_alone():
    """#650: a result kept because its trace survived is a failed cascade, not a clean one."""
    storage = _storage_double()
    kept, gone = MagicMock(), MagicMock()
    kept.id, gone.id = "res-kept", "res-gone"
    storage.query_eval_results_by_run.return_value = [kept, gone]

    def _delete(result_id: str, scope_id: str) -> bool:
        if result_id == "res-kept":
            raise StorageError("result 'res-kept' left intact: its trace 'res-kept:trace' could not be deleted")
        return True

    storage.delete_eval_result.side_effect = _delete

    with pytest.raises(StorageError, match=r"1 of 2 result\(s\).*res-kept.*trace 'res-kept:trace'"):
        curation.delete_run(storage, _run(), SCOPE, confirm="doomed")

    assert storage.delete_eval_result.call_count == 2
    storage.delete_eval_run.assert_not_called()


def test_delete_insight_loads_its_own_target_and_raises_when_absent():
    """The one delete with no service read helper in front of it, so the
    not-found check lives here rather than at the call site.
    """
    storage = _storage_double()
    storage.load_insight.return_value = None

    with pytest.raises(NotFoundError):
        curation.delete_insight(storage, "nope", SCOPE, confirm="nope")

    storage.delete_insight.assert_not_called()


def test_detaching_an_unattached_run_is_refused_before_any_write():
    storage = _storage_double()
    campaign = MagicMock()
    campaign.run_ids = ["r1"]
    storage.load_campaign.return_value = campaign

    with pytest.raises(ValidationFailedError):
        campaigns.remove_runs_from_campaign(storage, "camp-1", SCOPE, ["r1", "never-attached"])

    storage.save_campaign.assert_not_called()


def test_detaching_the_run_a_control_was_declared_from_is_no_longer_refused():
    """The guard this path used to carry is GONE, and its absence is the deliverable.

    While the control was a run id, detaching that run left the campaign pointing at a
    non-member, so the detach refused and made the operator clear the designation first. The
    declaration names a VARIANT: detaching a run cannot invalidate one, because another
    observation may carry the identical contestant stack — and where none does, the bundle
    reports the control as unobserved rather than the campaign silently reading as an
    undesignated bag of runs. That question is also only answerable where the observations
    are read, which a detach has no reason to pay for.
    """
    declaration = minimal_declaration(control="a" * 64)
    storage = _storage_double()
    campaign = MagicMock()
    campaign.run_ids = ["ctl", "cell"]
    campaign.declared_design = declaration
    storage.load_campaign.return_value = campaign
    storage.save_campaign.return_value = True

    campaigns.remove_runs_from_campaign(storage, "camp-1", SCOPE, ["ctl"])

    assert campaign.run_ids == ["cell"]
    assert campaign.declared_design is declaration, "detaching a run must not edit the declaration"


def test_detaching_a_run_leaves_the_declaration_untouched():
    """Membership and the declaration are two facts, and this path writes only the first."""
    declaration = minimal_declaration(control="b" * 64)
    storage = _storage_double()
    campaign = MagicMock()
    campaign.run_ids = ["ctl", "cell", "stray"]
    campaign.declared_design = declaration
    storage.load_campaign.return_value = campaign
    storage.save_campaign.return_value = True

    campaigns.remove_runs_from_campaign(storage, "camp-1", SCOPE, ["stray"])

    assert campaign.run_ids == ["ctl", "cell"]
    assert campaign.declared_design.control == "b" * 64


def test_deleting_a_run_detaches_membership_and_leaves_the_declared_control_alone():
    """The cascade's control-clearing branch is gone for the same reason the detach guard is.

    It existed so destroying a run could not leave a designation pointing at an id that no
    longer resolves — a dangling reference reading downstream as a control the analysis merely
    failed to load. A variant key is not a pointer at a run, so nothing dangles, and clearing
    the declaration here would DESTROY operator input on the strength of one run's deletion.
    """
    declaration = minimal_declaration(control="c" * 64)
    storage = _storage_double()
    campaign = MagicMock()
    campaign.id = "camp-1"
    campaign.run_ids = ["ctl", "cell"]
    campaign.declared_design = declaration
    storage.list_campaigns.return_value = [campaign]
    storage.save_campaign.return_value = True

    removed = curation.delete_run(storage, _run("ctl"), SCOPE, confirm="ctl")

    assert removed["campaigns_detached"] == ["camp-1"]
    # Only the run's own scope is scanned: a campaign holds runs of its own scope alone.
    storage.list_campaigns.assert_called_once_with(SCOPE)

    assert campaign.run_ids == ["cell"]
    assert campaign.declared_design.control == "c" * 64


def test_archiving_a_run_already_in_the_target_state_writes_nothing():
    storage = _storage_double()
    current = MagicMock()
    current.archived = True
    storage.load_eval_runs.return_value = [current]

    # The run is loaded as a listing, with the installed host's listing elisions applied.
    assert curation.set_run_archived(storage, "r1", SCOPE, archived=True, profile=toyhost_profile()) is current
    storage.set_eval_run_archived.assert_not_called()
    storage.save_eval_run.assert_not_called()
