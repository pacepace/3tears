"""a pod asks the hub's broker what it can do: the capability names on an ``l3.query`` reply, none from an older one."""

from __future__ import annotations

from threetears.core.backends.broker_generation import GENERATION_BY_TRIGGER, HUB_CAPABILITIES_FIELD

from .scripted_broker import scripted_proxy


async def test_the_capabilities_a_reply_names_are_the_hubs() -> None:
    proxy = scripted_proxy({"success": True, "rows": [{"one": 1}], HUB_CAPABILITIES_FIELD: [GENERATION_BY_TRIGGER]})
    assert await proxy.hub_capabilities() == frozenset({GENERATION_BY_TRIGGER})


async def test_a_hub_built_before_the_field_announces_none() -> None:
    proxy = scripted_proxy({"success": True, "rows": [{"one": 1}]})
    assert await proxy.hub_capabilities() == frozenset()
