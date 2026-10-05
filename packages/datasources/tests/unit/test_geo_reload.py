"""the tool pod -> hub contract for reporting reloaded geography shapes.

every client test drives the real decode: a reply JSON-encoded as the hub sends it,
handed back from a scripted ``request_raw``.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from uuid import uuid7

import pytest
from pydantic import ValidationError

from threetears.datasources.geo_reload import (
    GEO_RELOAD_ERROR_CODES,
    MAX_RELOADED_LAYERS,
    GeoLayersReloadedRequest,
    GeoReloadRefusedError,
    GeoReloadUnavailableError,
    LayerVersions,
    generations_to_delete,
    report_geo_layers_reloaded,
)
from threetears.nats import RequestTimeoutError, Subject, set_default_namespace

_TOKEN = "hub-minted-identity-token"
_LAYER = "us_county_census2022"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """bind the subject namespace a connected client would have set."""
    set_default_namespace("3tears")


class _ScriptedRequests:
    """answers ``request_raw`` with scripted replies, recording each request it was sent."""

    def __init__(self, *replies: dict[str, Any] | Exception, echo: bool = True) -> None:
        self._replies = list(replies)
        self._echo = echo
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta) -> bytes:
        del timeout
        self.sent.append((subject.path, json.loads(payload)))
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if self._echo:
            reply = {"correlation_id": self.sent[-1][1]["correlation_id"], **reply}
        return json.dumps(reply).encode()


async def _report(nc: _ScriptedRequests, generations: dict[str, int] | None = None) -> dict[str, LayerVersions]:
    return await report_geo_layers_reloaded(
        nc,  # type: ignore[arg-type]
        identity_token=_TOKEN,
        generations=generations or {_LAYER: 2},
    )


class TestRequest:
    async def test_the_report_goes_to_the_hub_subject_with_the_token_and_generations(self) -> None:
        nc = _ScriptedRequests({"success": True, "versions": {_LAYER: 2}, "previous_versions": {_LAYER: 1}})
        assert await _report(nc) == {_LAYER: LayerVersions(version=2, previous=1)}
        subject, body = nc.sent[0]
        assert subject == "3tears.hub.geo.layers.reloaded"
        assert body["identity_token"] == _TOKEN
        assert body["generations"] == {_LAYER: 2}

    def test_the_token_is_redacted_off_the_wire(self) -> None:
        request = GeoLayersReloadedRequest(identity_token=_TOKEN, correlation_id=uuid7(), generations={_LAYER: 1})
        assert _TOKEN not in repr(request)
        assert _TOKEN in request.model_dump_json()

    @pytest.mark.parametrize("generations", [{}, {_LAYER: 0}, {"": 2}])
    async def test_a_report_out_of_bounds_is_refused_as_invalid_and_sends_nothing(
        self, generations: dict[str, int]
    ) -> None:
        nc = _ScriptedRequests()
        with pytest.raises(GeoReloadRefusedError) as caught:
            await report_geo_layers_reloaded(nc, identity_token=_TOKEN, generations=generations)  # type: ignore[arg-type]
        assert caught.value.error_code == "INVALID_REQUEST"
        assert nc.sent == []

    def test_a_generation_below_one_is_refused_by_the_model(self) -> None:
        with pytest.raises(ValidationError):
            GeoLayersReloadedRequest(identity_token=_TOKEN, correlation_id=uuid7(), generations={_LAYER: 0})

    def test_too_many_layers_are_refused_by_the_model(self) -> None:
        many = {f"layer_{i}": 1 for i in range(MAX_RELOADED_LAYERS + 1)}
        with pytest.raises(ValidationError):
            GeoLayersReloadedRequest(identity_token=_TOKEN, correlation_id=uuid7(), generations=many)

    async def test_no_token_sends_nothing(self) -> None:
        nc = _ScriptedRequests()
        with pytest.raises(GeoReloadUnavailableError):
            await report_geo_layers_reloaded(nc, identity_token="", generations={_LAYER: 1})  # type: ignore[arg-type]
        assert nc.sent == []


class TestReply:
    async def test_a_refusal_carries_its_code_and_the_current_versions(self) -> None:
        nc = _ScriptedRequests(
            {"success": False, "error_code": "GENERATION_BEHIND", "error_message": "behind", "versions": {_LAYER: 5}}
        )
        with pytest.raises(GeoReloadRefusedError) as caught:
            await _report(nc)
        assert caught.value.error_code == "GENERATION_BEHIND"
        assert caught.value.versions == {_LAYER: 5}

    async def test_reload_failed_is_retryable(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "RELOAD_FAILED", "error_message": "db"})
        with pytest.raises(GeoReloadUnavailableError):
            await _report(nc)

    async def test_an_unknown_code_is_a_refusal(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "SOMETHING_NEW", "error_message": "x"})
        with pytest.raises(GeoReloadRefusedError):
            await _report(nc)

    async def test_a_refusal_without_a_correlation_id_answers_this_request(self) -> None:
        nc = _ScriptedRequests({"success": False, "error_code": "INVALID_REQUEST", "error_message": "x"}, echo=False)
        with pytest.raises(GeoReloadRefusedError):
            await _report(nc)

    async def test_a_reply_to_another_request_is_not_an_answer(self) -> None:
        stray = {
            "success": True,
            "correlation_id": str(uuid7()),
            "versions": {_LAYER: 2},
            "previous_versions": {_LAYER: 1},
        }
        nc = _ScriptedRequests(stray, echo=False)
        with pytest.raises(GeoReloadUnavailableError):
            await _report(nc)

    async def test_a_success_that_does_not_carry_the_generation_is_not_trusted(self) -> None:
        nc = _ScriptedRequests({"success": True, "versions": {_LAYER: 1}, "previous_versions": {_LAYER: None}})
        with pytest.raises(GeoReloadUnavailableError):
            await _report(nc)

    async def test_a_success_that_does_not_carry_the_previous_version_is_not_trusted(self) -> None:
        # without it the pod cannot tell which older generation clients still read
        nc = _ScriptedRequests({"success": True, "versions": {_LAYER: 2}})
        with pytest.raises(GeoReloadUnavailableError, match="previous"):
            await _report(nc)

    async def test_a_first_move_has_no_previous_version(self) -> None:
        nc = _ScriptedRequests({"success": True, "versions": {_LAYER: 1}, "previous_versions": {_LAYER: None}})
        assert await _report(nc, {_LAYER: 1}) == {_LAYER: LayerVersions(version=1, previous=None)}

    async def test_a_timeout_is_retryable(self) -> None:
        nc = _ScriptedRequests(RequestTimeoutError("no answer"))
        with pytest.raises(GeoReloadUnavailableError):
            await _report(nc)

    async def test_a_reply_that_does_not_decode_is_retryable(self) -> None:
        nc = _ScriptedRequests({"success": "maybe"})
        with pytest.raises(GeoReloadUnavailableError):
            await _report(nc)


def test_the_vocabulary_names_every_refusal_the_contract_describes() -> None:
    assert {
        "INVALID_REQUEST",
        "IDENTITY_REFUSED",
        "LAYER_NOT_REGISTERED",
        "LAYER_NOT_OWNED",
        "GENERATION_BEHIND",
        "GENERATION_OUT_OF_RANGE",
        "RELOAD_FAILED",
    } == GEO_RELOAD_ERROR_CODES


class TestRetention:
    """a pod keeps the two generations the hub serves -- the reported one and the recorded previous one."""

    async def test_after_a_failed_load_the_previous_version_is_the_one_replaced_not_one_below(self) -> None:
        # the hub was at 3; a load of 6 failed part way and was never reported; 7 is reported over 3
        nc = _ScriptedRequests({"success": True, "versions": {_LAYER: 7}, "previous_versions": {_LAYER: 3}})
        (reloaded,) = (await _report(nc, {_LAYER: 7})).values()
        assert reloaded == LayerVersions(version=7, previous=3)
        assert generations_to_delete({2, 3, 6, 7}, version=reloaded.version, previous=reloaded.previous) == {2, 6}

    def test_a_first_move_keeps_only_the_reported_generation(self) -> None:
        assert generations_to_delete({1, 2}, version=2, previous=None) == {1}

    def test_nothing_the_hub_serves_is_deleted(self) -> None:
        assert generations_to_delete({3, 7}, version=7, previous=3) == frozenset()
