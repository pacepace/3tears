"""agent-tools' old import paths name core's tool-wire objects, not copies of them.

the wire protocol moved into :mod:`threetears.core.tool_wire`; ``server``,
``context_envelope`` and ``http_operation`` re-export it. a copy under the old
name would be a second class: a message validated by one would fail an
``isinstance`` against the other, and a nested field typed with one would
refuse an instance of the other.
"""

from __future__ import annotations

import pytest

import threetears.agent.tools.context_envelope as context_envelope
import threetears.agent.tools.http_operation as http_operation
import threetears.agent.tools.server as server
import threetears.core.tool_wire as tool_wire
from threetears.agent.tools.context_envelope import CallContext as OldCallContext
from threetears.agent.tools.http_operation import RestAffordance as OldRestAffordance
from threetears.agent.tools.server import CallRequest as OldCallRequest
from threetears.agent.tools.server import ToolManifestEntry as OldToolManifestEntry
from threetears.core.tool_wire import CallContext, CallRequest, RestAffordance, ToolManifestEntry

#: each old module, onto the names it re-exports from core.
_REEXPORTS = {
    context_envelope: ("CallContext", "bind_log_context"),
    http_operation: (
        "CACHEABLE_METHODS",
        "MAX_POINTER_AGE_SECONDS",
        "QUERY_METHODS",
        "HttpMethod",
        "ParameterBinding",
        "PathTemplateBinding",
        "RestAffordance",
        "RestAffordanceError",
    ),
    server: (
        "CallAccepted",
        "CallRequest",
        "CallResponse",
        "HeartbeatMessage",
        "RefusedTool",
        "RegistrationManifest",
        "RegistrationResponse",
        "ToolManifestEntry",
    ),
}

_CASES = [(module, name) for module, names in _REEXPORTS.items() for name in names]


@pytest.mark.parametrize(("module", "name"), _CASES, ids=[f"{m.__name__}.{n}" for m, n in _CASES])
def test_the_old_path_is_cores_object(module: object, name: str) -> None:
    assert getattr(module, name) is getattr(tool_wire, name)


@pytest.mark.parametrize(("module", "name"), _CASES, ids=[f"{m.__name__}.{n}" for m, n in _CASES])
def test_the_old_module_still_publishes_the_name(module: object, name: str) -> None:
    assert name in getattr(module, "__all__")


def test_an_old_path_instance_validates_into_a_core_model() -> None:
    context = OldCallContext(trace={"a": "1"})
    request = CallRequest(tool_name="t", tool_version="1.0", arguments={}, context=context)
    assert isinstance(request, OldCallRequest)
    assert request.context is context
    assert isinstance(request.context, CallContext)


def test_an_old_path_declaration_rides_a_core_manifest_entry() -> None:
    declaration = OldRestAffordance(method="GET", path_template="/x")
    entry = ToolManifestEntry(name="t", version="1.0", description="", input_schema={}, face_rest=declaration)
    assert isinstance(entry, OldToolManifestEntry)
    assert isinstance(entry.face_rest, RestAffordance)
