"""a hub door request without a proof carries no ``identity_proof`` key, so a hub that predates it reads it.

A caller holding no key the token binds sends the token alone. A hub from before the door proofs
(0.66.0 to 0.68.0) may read the request with every unknown field refused, so a body carrying
``"identity_proof": null`` would be refused there. Each request is checked against a model built
the way such a hub's is: its fields as they were, and ``extra="forbid"``.
"""

from __future__ import annotations

import json

from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel, ConfigDict

from threetears.agent.tools.engagement_resolver import EngagementScopeRequestModel
from threetears.agent.tools.namespace_discovery_client import NamespaceDiscoveryRequest, NamespaceTypeFilter
from threetears.agent.tools.object_resolver import ObjectResolveRequestModel


class _Release066EngagementScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_token: str
    correlation_id: UUID
    engagement_id: UUID | None = None


class _Release066NamespaceDiscovery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    identity_token: str | None = None
    user_identity_token: str | None = None
    namespace_type: NamespaceTypeFilter | None = None


class _Release066ObjectResolve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity_token: str
    correlation_id: UUID
    object_id: UUID


_CASES: list[tuple[BaseModel, type[BaseModel]]] = [
    (EngagementScopeRequestModel(identity_token="t", correlation_id=uuid7()), _Release066EngagementScope),
    (NamespaceDiscoveryRequest(correlation_id=uuid7(), identity_token="t"), _Release066NamespaceDiscovery),
    (
        ObjectResolveRequestModel(identity_token="t", correlation_id=uuid7(), object_id=uuid7()),
        _Release066ObjectResolve,
    ),
]


@pytest.mark.parametrize(("request_model", "older_hub"), _CASES, ids=lambda case: type(case).__name__)
def test_no_proof_means_no_key_on_the_wire(request_model: BaseModel, older_hub: type[BaseModel]) -> None:
    wire = json.loads(request_model.model_dump_json())
    assert "identity_proof" not in wire
    older_hub.model_validate(wire)


@pytest.mark.parametrize(("request_model", "older_hub"), _CASES, ids=lambda case: type(case).__name__)
def test_a_proof_is_still_sent(request_model: BaseModel, older_hub: type[BaseModel]) -> None:
    proven = request_model.model_copy(update={"identity_proof": "proof"})
    assert json.loads(proven.model_dump_json())["identity_proof"] == "proof"
