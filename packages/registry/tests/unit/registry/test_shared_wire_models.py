"""the registry's names for the probe ack, the discovery request and the call reply are core's classes.

a pod sends each of these and the registry parses it, so both ends read one class from
:mod:`threetears.core.tool_wire`. the registry still publishes them under the names it always
has; a copy under that name would be a second class, which an ``isinstance`` against the pod's
or a nested field typed with it would refuse.
"""

from __future__ import annotations

import pytest

import threetears.core.tool_wire as tool_wire
import threetears.registry.discovery as discovery
import threetears.registry.proxy as proxy
import threetears.registry.registration as registration

#: (registry module, its name, core's name) for each shared class.
_SHARED = [
    (registration, "ProbeResponse", "ProbeAck"),
    (discovery, "DiscoverRequest", "DiscoveryProbeRequest"),
    (discovery, "DiscoverToolEntry", "DiscoveryProbeToolEntry"),
    (proxy, "ProxyCallResponse", "CallResponse"),
]

_IDS = [f"{module.__name__}.{name}" for module, name, _ in _SHARED]


@pytest.mark.parametrize(("module", "name", "core_name"), _SHARED, ids=_IDS)
def test_the_registry_name_is_cores_object(module: object, name: str, core_name: str) -> None:
    assert getattr(module, name) is getattr(tool_wire, core_name)


@pytest.mark.parametrize(("module", "name", "core_name"), _SHARED, ids=_IDS)
def test_the_registry_module_still_publishes_the_name(module: object, name: str, core_name: str) -> None:
    assert name in getattr(module, "__all__")


def test_a_pod_built_discovery_request_is_what_the_registry_parses() -> None:
    entry = tool_wire.DiscoveryProbeToolEntry(name="t", version="1.0")
    request = discovery.DiscoverRequest(agent_id="a", tool_manifest=[entry])
    assert isinstance(request, tool_wire.DiscoveryProbeRequest)
    assert request.tool_manifest[0] is entry
    assert "pod_id" not in request.model_dump(mode="json")
