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
annotation and default. The response's ``tools`` is compared by its element types, which are the
entry pair. A base class would hold the same (backlogged), at the cost of reordering the registry's
serialized fields.
"""

from __future__ import annotations

from typing import get_args

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
    """the default a field takes, from a value or a factory; ``...`` when it has none."""
    if field.default_factory is not None:
        return field.default_factory()  # type: ignore[call-arg]
    return field.default


@pytest.mark.parametrize(("pod", "registry"), _PAIRS, ids=lambda model: model.__name__)
def test_every_pod_field_is_the_registrys_field(pod: type[BaseModel], registry: type[BaseModel]) -> None:
    assert pod.model_fields, f"{pod.__name__} declares no fields: the comparison would pass over nothing"
    for name, field in pod.model_fields.items():
        assert name in registry.model_fields, f"{pod.__name__}.{name} has no field of that name on {registry.__name__}"
        theirs = registry.model_fields[name]
        if (pod, name) in _PAIRED_ANNOTATIONS:
            ours_args, theirs_args = get_args(field.annotation), get_args(theirs.annotation)
            assert (ours_args, theirs_args) == (
                (_PAIRED_ANNOTATIONS[pod, name][0],),
                (_PAIRED_ANNOTATIONS[pod, name][1],),
            )
        else:
            assert field.annotation == theirs.annotation, (
                f"{pod.__name__}.{name} is {field.annotation}, {registry.__name__}.{name} is {theirs.annotation}"
            )
        assert _default(field) == _default(theirs), (
            f"{pod.__name__}.{name} and {registry.__name__}.{name} default apart"
        )
        assert field.is_required() == theirs.is_required(), f"{name} is required on one side only"


def test_the_registry_entry_carries_more_than_the_pods() -> None:
    """the pair is a real subset: were it whole, the two models should be one class instead."""
    assert set(DiscoveryProbeResultEntry.model_fields) < set(DiscoverResultEntry.model_fields)
