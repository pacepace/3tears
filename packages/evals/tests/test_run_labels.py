"""A run can be labelled, listed by its label and cleared, and no key or fingerprint moves (#648).

A label names a run — a baseline, a production snapshot — so an operator can refer to it, and list it, by
something other than its id. It is curation, like archiving: no variant key, context key or bundle fingerprint is
computed over it. Everything here goes through the toy host's operations host, by the ``run_label`` and
``runs_list`` actions an agent calls, and reads the stored run back through the host's storage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.kernel import resolve_variant_identity
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.ops import RunLine, RunListing
from threetears.evals.run.run_document import normalized_run_labels
from threetears.evals.schema import EvalRun
from packages.evals.tests.ops_support import CALLER, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop


#: The frozen core corpus, one directory per core version (``test_core_schema_pins.py`` pins it).
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "core_documents"


@pytest.fixture
def tools() -> dict[str, Any]:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}


async def _call(tools: dict[str, Any], fixture: OpsFixture, arguments: dict[str, Any]) -> Any:
    outcome = await tools["evals"].call(arguments, host=fixture.host, caller=CALLER)
    assert not outcome.is_error, outcome.text
    return outcome


def _stored(fixture: OpsFixture, run_id: str) -> EvalRun:
    run, _ = fixture.host.eval_host.storage.load_eval_run_with_etag(run_id, CALLER.scope_id)
    assert run is not None
    return run


def _keys(fixture: OpsFixture, run_id: str) -> tuple[Any, ...]:
    """Everything a label must never move: the run's identity, and the campaign's bundle fingerprint."""
    eval_host = fixture.host.eval_host
    run = _stored(fixture, run_id)
    identity = resolve_variant_identity(run=run, profile=eval_host.profile)
    bundle = assemble_context_bundle(fixture.campaign, storage=eval_host.storage, profile=eval_host.profile)
    return (identity.variant_key, run.context_key, run.context_components, run.variant_levers, bundle.fingerprint())


class TestARunIsLabelledListedAndCleared:
    async def test_a_label_is_set_listed_by_and_cleared_and_nothing_moves(self, tools: dict[str, Any]) -> None:
        fixture = off_loop(ops_fixture, generation=False)
        run_id, other_id = fixture.campaign.run_ids[:2]
        before = off_loop(_keys, fixture, run_id)

        labelled = await _call(
            tools, fixture, {"action": "run_label", "run_id": run_id, "run_labels": [" production ", "baseline"]}
        )
        assert RunLine.model_validate(labelled.structured).labels == ["baseline", "production"]
        assert labelled.text.endswith(", labels baseline, production")
        assert off_loop(_stored, fixture, run_id).labels == ["baseline", "production"]

        listed = await _call(tools, fixture, {"action": "runs_list", "label": "baseline"})
        listing = RunListing.model_validate(listed.structured)
        assert [line.id for line in listing.runs] == [run_id]
        assert listing.label == "baseline"
        assert "labelled baseline" in listed.text
        everything = RunListing.model_validate((await _call(tools, fixture, {"action": "runs_list"})).structured)
        assert {run_id, other_id} <= {line.id for line in everything.runs}, "no label lists every run"

        assert off_loop(_keys, fixture, run_id) == before, "a label moves no key and no fingerprint"

        cleared = await _call(tools, fixture, {"action": "run_label", "run_id": run_id, "run_labels": []})
        assert RunLine.model_validate(cleared.structured).labels == []
        assert off_loop(_stored, fixture, run_id).labels == []
        relisted = RunListing.model_validate(
            (await _call(tools, fixture, {"action": "runs_list", "label": "baseline"})).structured
        )
        assert relisted.runs == []
        assert off_loop(_keys, fixture, run_id) == before

    async def test_setting_the_labels_a_run_carries_writes_nothing(self, tools: dict[str, Any]) -> None:
        fixture = off_loop(ops_fixture, generation=False)
        run_id = fixture.campaign.run_ids[0]
        storage = fixture.host.eval_host.storage
        await _call(tools, fixture, {"action": "run_label", "run_id": run_id, "run_labels": ["baseline"]})
        _, etag = off_loop(storage.load_eval_run_with_etag, run_id, CALLER.scope_id)
        await _call(tools, fixture, {"action": "run_label", "run_id": run_id, "run_labels": ["baseline", "baseline"]})
        assert off_loop(storage.load_eval_run_with_etag, run_id, CALLER.scope_id)[1] == etag

    async def test_an_archived_labelled_run_is_listed_by_its_label_only_with_the_archived(
        self, tools: dict[str, Any]
    ) -> None:
        fixture = off_loop(ops_fixture, generation=False)
        run_id = fixture.campaign.run_ids[0]
        await _call(tools, fixture, {"action": "run_label", "run_id": run_id, "run_labels": ["baseline"]})
        await _call(tools, fixture, {"action": "run_archive", "run_id": run_id, "archived": True})
        hidden = await _call(tools, fixture, {"action": "runs_list", "label": "baseline"})
        assert RunListing.model_validate(hidden.structured).runs == []
        shown = await _call(tools, fixture, {"action": "runs_list", "label": "baseline", "include_archived": True})
        assert [line.id for line in RunListing.model_validate(shown.structured).runs] == [run_id]

    async def test_an_unknown_run_is_refused_as_not_found(self, tools: dict[str, Any]) -> None:
        fixture = off_loop(ops_fixture, generation=False)
        outcome = await tools["evals"].call(
            {"action": "run_label", "run_id": "no-such-run", "run_labels": ["baseline"]},
            host=fixture.host,
            caller=CALLER,
        )
        assert outcome.is_error
        assert "not found" in outcome.text


class TestALabelIsAShortName:
    def test_labels_are_stripped_deduplicated_and_sorted(self) -> None:
        assert normalized_run_labels(["b", " a", "b ", "a"]) == ["a", "b"]

    @pytest.mark.parametrize(("labels", "fault"), [(["  "], "is blank"), (["x" * 65], "longer than 64")])
    def test_a_label_that_cannot_name_a_run_is_refused(self, labels: list[str], fault: str) -> None:
        with pytest.raises(ValidationFailedError, match=fault):
            normalized_run_labels(labels)

    def test_more_than_sixteen_labels_are_refused(self) -> None:
        with pytest.raises(ValidationFailedError, match="at most 16"):
            normalized_run_labels([f"label-{index}" for index in range(17)])

    @pytest.mark.parametrize("labels", [["b", "a"], ["a", "a"], [""], ["x" * 65]])
    def test_a_stored_run_holds_short_labels_sorted_and_once_each(self, labels: list[str]) -> None:
        document = json.loads((FIXTURES / "v12" / "eval_run.json").read_text())
        document["labels"] = labels
        with pytest.raises(ValueError, match="run label"):
            EvalRun.from_dict(document)


def test_a_run_stored_before_labels_reads_with_none() -> None:
    """The v11 step is the identity: a run no one could have labelled reads with no labels."""
    document = json.loads((FIXTURES / "v11" / "eval_run.json").read_text())
    assert "labels" not in document
    assert EvalRun.from_dict(document).labels == []
