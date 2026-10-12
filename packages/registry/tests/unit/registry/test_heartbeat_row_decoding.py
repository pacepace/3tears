"""a heartbeat row's ``tools`` field decodes, or the row is refused as corrupt.

the field is stored as JSON text by some writers. text that did not parse used to decode as
``[]``: a pod read as owning no tools, which quietly drops it out of the catalog.
"""

from __future__ import annotations

import json

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend


def _collection() -> HeartbeatCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=create_registry_l1_backend())
    return HeartbeatCollection(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))


def test_tools_stored_as_json_text_decode_to_the_list() -> None:
    row = _collection().deserialize(json.dumps({"pod_id": "p1", "tools": json.dumps(["a", "b"])}).encode())
    assert row["tools"] == ["a", "b"]


def test_tools_stored_as_text_that_is_not_json_refuse_the_row_by_pod() -> None:
    payload = json.dumps({"pod_id": "p1", "tools": "not json"}).encode()
    with pytest.raises(ValueError, match="heartbeat row for pod 'p1' carries a tools field that is not JSON"):
        _collection().deserialize(payload)
