"""the tool-call wire protocol is core's own surface, and costs only core to read.

a tool pod and the registry both parse these messages, so they are defined in
``threetears.core.tool_wire`` and nowhere else, and importing them loads neither
end's implementation: not agent-tools, not the registry, and none of core's
optional extras.
"""

from __future__ import annotations

import subprocess
import sys

import threetears.core.tool_wire as tool_wire

#: every name the package publishes, onto the module that defines it.
_EXPECTED: dict[str, str] = {
    "CallContext": "threetears.core.tool_wire.context",
    "bind_log_context": "threetears.core.tool_wire.context",
    "CACHEABLE_METHODS": "threetears.core.tool_wire.rest",
    "MAX_POINTER_AGE_SECONDS": "threetears.core.tool_wire.rest",
    "QUERY_METHODS": "threetears.core.tool_wire.rest",
    "HttpMethod": "threetears.core.tool_wire.rest",
    "ParameterBinding": "threetears.core.tool_wire.rest",
    "PathTemplateBinding": "threetears.core.tool_wire.rest",
    "RestAffordance": "threetears.core.tool_wire.rest",
    "RestAffordanceError": "threetears.core.tool_wire.rest",
    "CallAccepted": "threetears.core.tool_wire.messages",
    "CallRequest": "threetears.core.tool_wire.messages",
    "CallResponse": "threetears.core.tool_wire.messages",
    "HeartbeatMessage": "threetears.core.tool_wire.messages",
    "RefusedTool": "threetears.core.tool_wire.messages",
    "RegistrationManifest": "threetears.core.tool_wire.messages",
    "RegistrationResponse": "threetears.core.tool_wire.messages",
    "ToolManifestEntry": "threetears.core.tool_wire.messages",
}

#: what importing the protocol must not load: either end, or an extra core makes optional.
_PROBE = """
import sys
import threetears.core.tool_wire
loaded = sorted(
    name
    for name in sys.modules
    if name.startswith(("threetears.agent", "threetears.registry"))
    or name.split(".")[0] in ("httpx", "cryptography", "jwt", "langchain", "langgraph", "pgvector")
)
assert not loaded, loaded
print("imported")
"""


def test_the_package_publishes_exactly_the_protocol() -> None:
    assert sorted(tool_wire.__all__) == sorted(_EXPECTED)


def test_each_name_is_resolved_from_its_defining_module() -> None:
    for name, module_name in _EXPECTED.items():
        defining = sys.modules[module_name]
        assert getattr(tool_wire, name) is getattr(defining, name), name


def test_each_class_is_defined_in_core() -> None:
    for name, module_name in _EXPECTED.items():
        value = getattr(tool_wire, name)
        if isinstance(value, type) or callable(value):
            assert value.__module__ == module_name, name


def test_importing_the_protocol_loads_neither_end_nor_an_extra() -> None:
    result = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
