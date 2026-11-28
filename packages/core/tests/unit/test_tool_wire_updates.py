"""a running call's updates on the wire: the request for them, and one update, read tolerantly and refused when broken."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError
from threetears.core.tool_wire import UPDATE_LOG_LEVELS, CallRequest, CallUpdate, CallUpdates


class TestTheRequestForUpdates:
    def test_it_asks_for_something_only_with_a_token_or_a_level(self) -> None:
        assert CallUpdates(subject="s").wants_any() is False
        assert CallUpdates(subject="s", progress_token=0).wants_any() is True
        assert CallUpdates(subject="s", log_level="debug").wants_any() is True

    def test_a_call_carries_it_and_one_without_it_is_unchanged(self) -> None:
        request = CallRequest.model_validate(
            {"tool_name": "t", "tool_version": "1", "arguments": {}, "updates": {"subject": "s", "progress_token": "p"}}
        )
        assert request.updates == CallUpdates(subject="s", progress_token="p")
        assert CallRequest(tool_name="t", tool_version="1", arguments={}).updates is None

    def test_a_newer_senders_field_is_ignored(self) -> None:
        updates = CallUpdates.model_validate({"subject": "s", "log_level": "info", "from_the_future": 1})
        assert updates.log_level == "info"

    def test_a_level_off_the_scale_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            CallUpdates.model_validate({"subject": "s", "log_level": "verbose"})


class TestOneUpdate:
    def test_a_progress_update_round_trips(self) -> None:
        update = CallUpdate(seq=1, kind="progress", progress=0.5, total=1.0, message="half")
        assert CallUpdate.model_validate_json(update.model_dump_json()) == update

    def test_a_log_update_carries_any_json_value(self) -> None:
        update = CallUpdate(seq=2, kind="log", level="warning", data={"rows": [1, 2]}, logger="build")
        assert json.loads(update.model_dump_json())["data"] == {"rows": [1, 2]}

    @pytest.mark.parametrize(
        "body",
        [
            {"seq": 0, "kind": "progress", "progress": 1},
            {"seq": 1, "kind": "progress"},
            {"seq": 1, "kind": "progress", "progress": math.inf},
            {"seq": 1, "kind": "progress", "progress": 1, "total": math.nan},
            {"seq": 1, "kind": "log"},
            {"seq": 1, "kind": "a newer kind"},
        ],
        ids=["seq-zero", "no-progress", "infinite", "total-nan", "log-no-level", "unknown-kind"],
    )
    def test_an_update_missing_what_its_kind_needs_is_refused(self, body: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            CallUpdate.model_validate(body)

    def test_the_levels_run_least_severe_first(self) -> None:
        assert UPDATE_LOG_LEVELS[0] == "debug" and UPDATE_LOG_LEVELS[-1] == "emergency"
        assert len(set(UPDATE_LOG_LEVELS)) == 8
