"""
enforcement: the pod's discovery result is a subset of the registry's, field for field.

A pod polling for readiness parses the registry's discovery reply with its own models,
:class:`~threetears.agent.tools.server.DiscoveryProbeResponse` and
:class:`~threetears.agent.tools.server.DiscoveryProbeResultEntry`, which keep only the fields
readiness polling needs. The registry sends its full models,
:class:`~threetears.registry.discovery.DiscoverResponse` and
:class:`~threetears.registry.discovery.DiscoverResultEntry`. The two are separate classes on
purpose (folding them would grow the pod's surface), so nothing structural keeps them in step: a
shared field renamed, retyped or given another default on one side would parse as a missing or
wrong value on the other, and a pod would wait for a status it can never read.

**What it holds.** Every field of each pod model exists in the registry model with the same name,
wire aliases, annotation, default and requiredness, and both models name their fields on the wire the
same way. The response's ``tools`` is compared by its container and its element types, which are the
entry pair. The owner chose this test over a base class (2026-10-25), which would hold the same at the
cost of reordering the registry's serialized fields; the base class is weighed in pacepace/3tears#960.
"""

from __future__ import annotations

from typing import get_args, get_origin

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from threetears.agent.tools.server import DiscoveryProbeResponse, DiscoveryProbeResultEntry
from threetears.registry.discovery import DiscoverResponse, DiscoverResultEntry

#: (pod model, registry model) for each pair the pod reads a subset of.
_PAIRS: list[tuple[type[BaseModel], type[BaseModel]]] = [
    (DiscoveryProbeResultEntry, DiscoverResultEntry),
    (DiscoveryProbeResponse, DiscoverResponse),
]

#: a field whose annotation names the other side's model, compared by the pair it names instead.
_PAIRED_ANNOTATIONS = {(DiscoveryProbeResponse, "tools"): (DiscoveryProbeResultEntry, DiscoverResultEntry)}


def _default(field: FieldInfo) -> object:
    """the default a field takes, from a value or either kind of factory; ``PydanticUndefined`` when it has none."""
    return field.get_default(call_default_factory=True, validated_data={})


def _wire_names(field: FieldInfo) -> tuple[object, object, object]:
    """the names a field is read and written under on the wire, where they differ from its own."""
    return (field.alias, field.validation_alias, field.serialization_alias)


@pytest.mark.parametrize(("pod", "registry"), _PAIRS, ids=lambda model: model.__name__)
def test_every_pod_field_is_the_registrys_field(pod: type[BaseModel], registry: type[BaseModel]) -> None:
    assert pod.model_fields, f"{pod.__name__} declares no fields: the comparison would pass over nothing"
    assert pod.model_config.get("alias_generator") == registry.model_config.get("alias_generator"), (
        f"{pod.__name__} and {registry.__name__} name their fields on the wire differently"
    )
    for name, field in pod.model_fields.items():
        assert name in registry.model_fields, f"{pod.__name__}.{name} has no field of that name on {registry.__name__}"
        theirs = registry.model_fields[name]
        if (pod, name) in _PAIRED_ANNOTATIONS:
            ours_entry, theirs_entry = _PAIRED_ANNOTATIONS[pod, name]
            assert get_origin(field.annotation) is get_origin(theirs.annotation), (
                f"{pod.__name__}.{name} and {registry.__name__}.{name} hold their entries in different containers"
            )
            assert (get_args(field.annotation), get_args(theirs.annotation)) == ((ours_entry,), (theirs_entry,)), (
                f"{pod.__name__}.{name} holds {get_args(field.annotation)}, {registry.__name__}.{name} holds "
                f"{get_args(theirs.annotation)}"
            )
        else:
            assert field.annotation == theirs.annotation, (
                f"{pod.__name__}.{name} is {field.annotation}, {registry.__name__}.{name} is {theirs.annotation}"
            )
        assert _default(field) == _default(theirs), (
            f"{pod.__name__}.{name} and {registry.__name__}.{name} default apart"
        )
        assert field.is_required() == theirs.is_required(), f"{name} is required on one side only"
        assert _wire_names(field) == _wire_names(theirs), f"{name} goes by another name on the wire on one side"


def test_the_registry_entry_carries_more_than_the_pods() -> None:
    """the pair is a real subset: were it whole, the two models should be one class instead."""
    assert set(DiscoveryProbeResultEntry.model_fields) < set(DiscoverResultEntry.model_fields)
