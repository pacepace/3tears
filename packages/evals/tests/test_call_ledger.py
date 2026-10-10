"""The call ledger: what a candidate did, in order, apart from the world it left behind."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from threetears.evals.schema import CallLedger, RecordedCall
from packages.evals.tests.factories import make_eval_result, make_eval_trace, memory_storage


def test_calls_are_recorded_in_order_across_tools():
    ledger = CallLedger()
    ledger.record("shop", "search", {"query": "kitchen"})
    ledger.record("chat", "send_message", {"content": "found it"})
    ledger.record("shop", "add_item", {"item_ref": "r1"})

    assert [(call.tool, call.action) for call in ledger.calls] == [
        ("shop", "search"),
        ("chat", "send_message"),
        ("shop", "add_item"),
    ]


def test_a_call_with_no_parameters_records_an_empty_mapping():
    ledger = CallLedger()
    ledger.record("shop", "checkout")

    assert ledger.calls == [RecordedCall(tool="shop", action="checkout", params={})]


def test_recorded_parameters_are_insulated_from_the_caller():
    """A caller mutating what it passed after the call cannot rewrite what the candidate asked for."""
    params = {"query": "kitchen", "filters": ["new"]}
    ledger = CallLedger()
    ledger.record("shop", "search", params)
    params["query"] = "outdoor"
    params["filters"].append("sale")

    assert ledger.calls[0].params == {"query": "kitchen", "filters": ["new"]}


def test_a_ledger_round_trips_through_json():
    ledger = CallLedger()
    ledger.record("shop", "add_item", {"item_ref": "r1", "tags": ["kitchen"]})

    assert CallLedger.model_validate(json.loads(ledger.model_dump_json())) == ledger


@pytest.mark.parametrize(
    "stored",
    [
        {"calls": [], "namespaces": {}},
        {"calls": [{"tool": "shop", "action": "search", "params": {}, "result": "ok"}]},
        {"calls": [{"tool": "", "action": "search"}]},
    ],
    ids=["unknown ledger key", "unknown call key", "empty tool"],
)
def test_a_stored_ledger_is_read_as_strictly_as_every_stored_eval_shape(stored):
    with pytest.raises(ValidationError):
        CallLedger.model_validate(stored)


def test_a_trace_carrying_only_a_ledger_is_stored():
    """A kind whose candidate produced no output still made calls, and a re-check reads them back."""
    storage, _ = memory_storage()
    ledger = CallLedger()
    ledger.record("shop", "search", {"query": "linen"})
    result = make_eval_result()
    storage.save_eval_result(result, make_eval_trace(result_id=result.id, trace=[], call_ledger=ledger))

    stored = storage.load_eval_result(result.id, result.scope_id)
    trace = storage.load_eval_trace(result.id, result.scope_id)
    assert stored is not None and stored.has_trace is True
    assert trace is not None and trace.call_ledger == ledger
