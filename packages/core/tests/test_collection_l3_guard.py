"""``BaseCollection.required_l3_pool`` -- the guard 16 raw-SQL call sites now depend on.

`l3_pool` is legitimately ``None``: a collection can run on L1+L2 alone, and
``BaseCollection``'s own docstring instructs callers to guard rather than assume. In
practice the guard was routinely skipped, so `await self.l3_pool.fetch(...)` raised
``AttributeError: 'NoneType' object has no attribute 'fetch'`` from inside a query method,
saying nothing about the real mistake.

`required_l3_pool` replaced sixteen of those unguarded dereferences across
`threetears.conversations` and `threetears.agent.workspace`. Its whole value is the failure
it produces, so that failure is what these tests pin -- a guard whose error path is never
executed is indistinguishable from one that does not work.
"""

from __future__ import annotations

from typing import Any

import pytest
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


class _Thing(BaseEntity):
    """Minimal entity; this suite is about the collection's L3 handle, not entity behaviour."""

    primary_key_field: str = "id"


class _ThingCollection(BaseCollection[_Thing]):
    """A collection with no L3 configured, which is the state under test.

    The store methods are the abstract surface every concrete collection implements; none
    of them is reached here, because `required_l3_pool` is consulted BEFORE any query runs.
    They exist so the class is instantiable.
    """

    primary_key_column = "id"

    @property
    def table_name(self) -> str:
        return "things"

    @property
    def entity_class(self) -> type[_Thing]:
        return _Thing

    async def fetch_from_store(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def save_to_store(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def delete_from_store(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def serialize(self, data: dict[str, Any]) -> bytes:
        raise NotImplementedError

    def deserialize(self, data: bytes) -> dict[str, Any]:
        raise NotImplementedError


@pytest.fixture
def collection_without_l3() -> _ThingCollection:
    """No ``l3_pool`` on the registry -- the documented, supported L1/L2-only shape."""
    return _ThingCollection(CollectionRegistry(), DefaultCoreConfig(), nats_client=None)


def test_it_raises_rather_than_returning_none(collection_without_l3: _ThingCollection) -> None:
    """The point of the guard: fail here, not with an AttributeError inside a query."""
    assert collection_without_l3.l3_pool is None, "fixture must have no L3 for this to mean anything"

    with pytest.raises(RuntimeError) as excinfo:
        _ = collection_without_l3.required_l3_pool

    message = str(excinfo.value)
    # The message has to name the mistake, or it is no better than the AttributeError it
    # replaced. All three parts are load-bearing for whoever reads it in a traceback.
    assert "_ThingCollection" in message, "must name the collection that is misconfigured"
    assert "things" in message, "must name the table, which is how it is wired"
    assert "L3" in message


def test_it_returns_the_handle_when_one_is_configured() -> None:
    """The happy path must be a plain pass-through, not a copy or a wrapper.

    Sixteen call sites replaced ``self.l3_pool`` with this property, so anything other than
    the identical object would change what those queries run against. Asserted against
    ``l3_pool`` rather than against the object handed to ``configure``: the registry
    resolves a raw pool into an L3 backend on the way in, and the contract here is
    pass-through of whatever it resolved, not of what the caller passed.
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=object())
    collection = _ThingCollection(registry, DefaultCoreConfig(), nats_client=None)

    assert collection.l3_pool is not None, "fixture must have an L3 for this to mean anything"
    assert collection.required_l3_pool is collection.l3_pool
