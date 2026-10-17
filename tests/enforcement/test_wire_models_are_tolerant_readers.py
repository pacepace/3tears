"""enforcement: a wire model read across a deployment boundary is a tolerant reader, and only those are.

framework components and their consumers upgrade on their own schedules (expand / migrate / contract,
owner ruling 2026-10-07). a model a receiver validates from another deployable's bytes must therefore
ignore a field a newer sender adds instead of refusing the whole message, while still refusing the
fields that would steer authority. :class:`threetears.core.wire.TolerantWireModel` is that reader.

the inventory below is the set of 3tears models a separately deployed receiver reads. this gate fails
when an inventoried model is not a tolerant reader, and when a model outside the inventory becomes one:
config and internal models stay strict, and loosening one is a decision this file has to record.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from threetears.enforcement.common import parse_python_path
from threetears.core.wire import TolerantWireModel

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: fully qualified name onto who reads it from whom.
_WIRE_MODELS: dict[str, str] = {
    "threetears.registry.proxy.ProxyCallRequest": "the registry, from an agent",
    "threetears.agent.tools.server.CallRequest": "a tool pod, from the registry",
    "threetears.datasources.query_client.DatasourceQueryRequest": "the hub, from an agent or tool pod",
    "threetears.datasources.query_client.RelationFingerprintRequest": "the hub, from a tool pod",
    "threetears.datasources.query_client.RelationFingerprintResult": "a tool pod, from the hub",
    "threetears.datasources.geo_reload.GeoLayersReloadedRequest": "the hub, from a tool pod",
    "threetears.agent.audit.envelope.AuditEvent": "the hub's audit consumer, from every producer",
    "threetears.agent.audit.erasure.AuditAnonymizeRequest": "the hub, from an agent or tool pod",
    "threetears.core.collections.key_purge.CollectionKeysPurgeRequest": "the hub, from a tool pod",
    "threetears.datasources.query_client.DatasourceExportRequest": "the hub, from a tool pod",
    "threetears.datasources.query_client.DatasourceExportDeleteRequest": "the hub, from a tool pod",
    "threetears.datasources.query_client.DatasourceExportListRequest": "the hub, from a tool pod",
}

_SOURCE_ROOTS = tuple(
    sorted(
        path
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)


def _tolerant_subclasses_in_source() -> set[str]:
    """
    find every class in source whose bases name TolerantWireModel.

    :return: fully qualified class names
    :rtype: set[str]
    """
    found: set[str] = set()
    for root in _SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            module = ".".join(path.relative_to(root).with_suffix("").parts)
            tree = parse_python_path(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    names = {
                        base.id if isinstance(base, ast.Name) else getattr(base, "attr", "") for base in node.bases
                    }
                    if "TolerantWireModel" in names:
                        found.add(f"{module}.{node.name}")
    return found


def test_every_inventoried_wire_model_is_a_tolerant_reader() -> None:
    """
    each model a separately deployed receiver reads ignores unknown fields.

    :return: nothing
    :rtype: None
    """
    strict: list[str] = []
    for fqname in _WIRE_MODELS:
        module_name, _, class_name = fqname.rpartition(".")
        model = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(model, TolerantWireModel) or model.model_config.get("extra") != "ignore":
            strict.append(fqname)
    assert not strict, f"wire models that would refuse a newer sender's field: {strict}"


def test_no_model_outside_the_inventory_is_loosened() -> None:
    """
    a tolerant reader nobody inventoried is a config or internal model loosened without a decision.

    :return: nothing
    :rtype: None
    """
    found = _tolerant_subclasses_in_source()
    assert found, "the walk found no tolerant reader at all; it is not seeing the source"
    unlisted = sorted(found - set(_WIRE_MODELS))
    assert not unlisted, f"tolerant readers missing from the wire inventory: {unlisted}"


def test_a_tolerant_reader_still_refuses_a_named_field() -> None:
    """
    the gate's subject can fail where it matters: the datasource request refuses a body principal.

    :return: nothing
    :rtype: None
    """
    from pydantic import ValidationError

    from threetears.datasources.query_client import DatasourceQueryRequest

    try:
        DatasourceQueryRequest.model_validate(
            {
                "correlation_id": "0190f2a0-0000-7000-8000-000000000000",
                "identity_token": "t",
                "query": "select 1",
                "user_id": "someone-else",
            }
        )
    except ValidationError as exc:
        assert "may not be sent" in str(exc)
    else:
        raise AssertionError("a body user_id reached the datasource door")
