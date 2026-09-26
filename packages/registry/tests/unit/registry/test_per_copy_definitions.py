"""unit -- every copy of a tool keeps its OWN definition, and one function picks what a caller sees.

The catalog used to hold one description, one schema and one confirmation flag per
``name@version``, and every registration overwrote them. Whichever pod announced last defined
the tool for every caller -- including a pod that had no business defining it, and including
turning a human-approval gate off for everybody. Each pod's copy now carries the definitions
that pod announced, and :meth:`CatalogEntry.select_copies` is the one place that decides which
copies a caller can see, which definition it is shown, and which copies it may be routed to.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from threetears.nats import Subjects
from threetears.registry.catalog import (
    CatalogEntry,
    CopyStatus,
    ToolCatalog,
    ToolDefinition,
    ToolEndpoint,
)

from ._copies import definition, endpoint, entry

__all__: list[str] = []

_AGENT_A = UUID("01948a00-aaaa-7000-8000-00000000000a")
_AGENT_B = UUID("01948a00-aaaa-7000-8000-00000000000b")
_TTL = timedelta(seconds=45)
_FULL = "threetears.calculator@1.0.0"


def _inproc(agent_id: UUID) -> str:
    """the pod-id of ``agent_id``'s in-process tool server.

    :param agent_id: the owning agent
    :ptype agent_id: UUID
    :return: the ``{agent_id}.{instance}`` composite
    :rtype: str
    """
    return Subjects.agent_inprocess_pod_id(agent_id, "inst-1")


def _kv() -> AsyncMock:
    """a KV double that records puts and starts empty.

    :return: the double
    :rtype: AsyncMock
    """
    kv = AsyncMock()
    kv.keys = AsyncMock(return_value=[])
    kv.put = AsyncMock()
    kv.delete = AsyncMock()
    return kv


class TestToolDefinition:
    """the definition value and its two digests."""

    def test_the_schema_digest_ignores_key_order(self) -> None:
        """two spellings of one schema are one schema."""
        one = definition(input_schema={"type": "object", "properties": {"a": {"type": "string"}}})
        two = definition(input_schema={"properties": {"a": {"type": "string"}}, "type": "object"})
        assert one.schema_digest == two.schema_digest

    def test_the_schema_digest_changes_with_the_schema(self) -> None:
        """the paired change: an added property is a different schema."""
        one = definition(input_schema={"type": "object", "properties": {}})
        two = definition(input_schema={"type": "object", "properties": {"a": {"type": "string"}}})
        assert one.schema_digest != two.schema_digest

    def test_the_definition_digest_covers_every_field(self) -> None:
        """a description, a timeout or a confirmation flag each make a different definition."""
        base = definition("calc")
        variants = [
            definition("calc v2"),
            definition("calc", timeout_seconds=9.0),
            definition("calc", requires_confirmation=True),
            definition("calc", output_schema={"type": "string"}),
            definition("calc", input_schema={"type": "object", "properties": {"x": {}}}),
        ]
        digests = {base.digest, *(variant.digest for variant in variants)}
        assert len(digests) == len(variants) + 1
        # the schema digest moves only with the input schema
        assert {variant.schema_digest for variant in variants[:4]} == {base.schema_digest}

    def test_a_definition_round_trips_through_its_dict(self) -> None:
        """what is persisted comes back as the same value."""
        original = definition("calc", timeout_seconds=3.5, requires_confirmation=True, output_schema={"a": 1})
        assert ToolDefinition.from_dict(json.loads(json.dumps(original.to_dict()))) == original


class TestRegisterMergesPerCopy:
    """a registration changes only the registering pod's copy."""

    @pytest.mark.asyncio
    async def test_two_pods_keep_their_own_descriptions(self) -> None:
        """the second pod's announcement no longer rewrites the first pod's copy."""
        catalog = ToolCatalog()
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("A")))
        )
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-B", tool_definition=definition("B")))
        )

        held = catalog.get(_FULL)
        assert held is not None
        now = datetime.now(UTC)
        a_copy = held.get_endpoint("pod-A")
        b_copy = held.get_endpoint("pod-B")
        assert a_copy is not None and b_copy is not None
        assert [d.definition.description for d in a_copy.live_definitions(now, _TTL)] == ["A"]
        assert [d.definition.description for d in b_copy.live_definitions(now, _TTL)] == ["B"]

    @pytest.mark.asyncio
    async def test_the_entry_carries_no_definition_of_its_own(self) -> None:
        """there is no entry-level field left for a registration to overwrite."""
        held = entry("threetears.calculator", "1.0.0", endpoint("pod-A"))
        for gone in ("description", "input_schema", "output_schema", "timeout_seconds", "requires_confirmation"):
            assert not hasattr(held, gone), gone

    @pytest.mark.asyncio
    async def test_a_reannounced_definition_keeps_when_it_was_first_seen(self) -> None:
        """a heartbeat re-announcement refreshes liveness and nothing else."""
        catalog = ToolCatalog()
        early = datetime.now(UTC) - timedelta(seconds=30)
        calc = definition("calc")
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=calc, first=early))
        )
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=calc)))

        held = catalog.get(_FULL)
        assert held is not None
        copy = held.get_endpoint("pod-A")
        assert copy is not None
        (only,) = copy.definitions.values()
        assert only.first_announced == early
        assert only.last_announced > early

    @pytest.mark.asyncio
    async def test_one_pod_may_carry_two_live_definitions(self) -> None:
        """replicas share a pod id mid-rollout, so one copy can announce two definitions."""
        catalog = ToolCatalog()
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("v1")))
        )
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("v2")))
        )

        held = catalog.get(_FULL)
        assert held is not None
        copy = held.get_endpoint("pod-A")
        assert copy is not None
        assert sorted(d.definition.description for d in copy.live_definitions(datetime.now(UTC), _TTL)) == [
            "v1",
            "v2",
        ]

    @pytest.mark.asyncio
    async def test_a_stale_definition_is_pruned_on_its_own_copys_registration(self) -> None:
        """older than the TTL and re-announced by nobody: dropped when that pod next registers."""
        catalog = ToolCatalog()
        stale = datetime.now(UTC) - timedelta(seconds=600)
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("old")))
        )
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-B", tool_definition=definition("b")))
        )
        held = catalog.get(_FULL)
        assert held is not None
        # both pods then go quiet on these definitions: each ages past the TTL, re-announced by nobody
        for pod_id in ("pod-A", "pod-B"):
            quiet = held.get_endpoint(pod_id)
            assert quiet is not None
            for announcement in quiet.definitions.values():
                announcement.last_announced = stale

        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("new")))
        )

        a_copy = held.get_endpoint("pod-A")
        b_copy = held.get_endpoint("pod-B")
        assert a_copy is not None and b_copy is not None
        assert [d.definition.description for d in a_copy.definitions.values()] == ["new"]
        # another pod's stale definition is that pod's to refresh, not this registration's to drop
        assert [d.definition.description for d in b_copy.definitions.values()] == ["b"]


class TestSelectCopies:
    """the one selection function discovery and the proxy both ask."""

    def test_an_unannounced_or_unready_copy_is_not_visible(self) -> None:
        """visible means available AND holding a live definition."""
        stale = datetime.now(UTC) - timedelta(seconds=600)
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-ready"),
            endpoint("pod-pending", "pending"),
            endpoint("pod-down", "unavailable"),
            endpoint("pod-stale", first=stale),
            ToolEndpoint(pod_id="pod-bare", status="available"),
        )
        selection = held.select_copies(None, ttl=_TTL)
        assert [ep.pod_id for ep in selection.visible] == ["pod-ready"]
        assert [ep.pod_id for ep in selection.routable] == ["pod-ready"]

    def test_confirmation_is_required_when_any_visible_copy_requires_it(self) -> None:
        """one ungated copy cannot turn the gate off for a caller who can see a gated one."""
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-gated", tool_definition=definition("gated", requires_confirmation=True)),
            endpoint("pod-open", tool_definition=definition("open")),
        )
        assert held.select_copies(None, ttl=_TTL).requires_confirmation is True

    def test_confirmation_is_not_required_when_no_visible_copy_requires_it(self) -> None:
        """the paired twin: the OR is over copies, not a constant."""
        held = entry("threetears.calculator", "1.0.0", endpoint("pod-open", tool_definition=definition("open")))
        assert held.select_copies(None, ttl=_TTL).requires_confirmation is False

    def test_a_gated_copy_the_caller_cannot_see_does_not_gate_it(self) -> None:
        """another agent's in-process copy is invisible to this caller, gate and all."""
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint(_inproc(_AGENT_B), tool_definition=definition("b only", requires_confirmation=True)),
            endpoint("pod-shared", tool_definition=definition("shared")),
        )
        selection = held.select_copies(_AGENT_A, ttl=_TTL)
        assert selection.requires_confirmation is False
        assert [ep.pod_id for ep in selection.visible] == ["pod-shared"]

    def test_an_agent_sees_its_own_in_process_copy_first(self) -> None:
        """the caller's own copies form the tier when it has any available.

        The agent's own copy is the OLDER announcement and shares the shared copy's schema, so
        neither recency nor schema could be what picks it: only the tier does.
        """
        now = datetime.now(UTC)
        own = definition("A ONLY")
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint(_inproc(_AGENT_A), tool_definition=own, first=now - timedelta(seconds=20)),
            endpoint("pod-shared", tool_definition=definition("shared"), first=now - timedelta(seconds=5)),
        )
        mine = held.select_copies(_AGENT_A, now=now, ttl=_TTL)
        assert mine.own_tier is True
        assert mine.shown == own
        assert [ep.pod_id for ep in mine.routable] == [_inproc(_AGENT_A)]

        theirs = held.select_copies(_AGENT_B, now=now, ttl=_TTL)
        assert theirs.own_tier is False
        assert theirs.shown is not None and theirs.shown.description == "shared"
        assert [ep.pod_id for ep in theirs.routable] == ["pod-shared"]

    def test_the_most_recently_first_announced_definition_is_shown(self) -> None:
        """two serve-everyone copies differ; the newer announcement is what a caller reads."""
        now = datetime.now(UTC)
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-old", tool_definition=definition("old"), first=now - timedelta(seconds=20)),
            endpoint("pod-new", tool_definition=definition("new"), first=now - timedelta(seconds=5)),
        )
        selection = held.select_copies(None, now=now, ttl=_TTL)
        assert selection.shown is not None
        assert selection.shown.description == "new"

    def test_a_tie_on_first_announced_is_broken_by_digest(self) -> None:
        """same instant, two definitions: the lexically smaller digest wins, in either order."""
        now = datetime.now(UTC)
        left, right = definition("left"), definition("right")
        expected = min((left, right), key=lambda d: d.digest)
        for first, second in ((left, right), (right, left)):
            held = entry(
                "threetears.calculator",
                "1.0.0",
                endpoint("pod-1", tool_definition=first, first=now),
                endpoint("pod-2", tool_definition=second, first=now),
            )
            assert held.select_copies(None, now=now, ttl=_TTL).shown == expected

    def test_only_copies_announcing_the_shown_schema_are_routable(self) -> None:
        """a caller is routed only where the schema it was shown is served."""
        now = datetime.now(UTC)
        wide = definition("wide", input_schema={"type": "object", "properties": {"extra": {}}})
        narrow = definition("narrow")
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-narrow", tool_definition=narrow, first=now - timedelta(seconds=20)),
            endpoint("pod-wide", tool_definition=wide, first=now - timedelta(seconds=5)),
        )
        selection = held.select_copies(None, now=now, ttl=_TTL)
        assert selection.shown == wide
        assert [ep.pod_id for ep in selection.routable] == ["pod-wide"]
        assert set(selection.routed_definitions) == {"pod-wide"}

    def test_a_named_schema_routes_only_to_copies_serving_it(self) -> None:
        """the proxy passes the digest its caller was shown; only matching copies qualify."""
        now = datetime.now(UTC)
        wide = definition("wide", input_schema={"type": "object", "properties": {"extra": {}}})
        narrow = definition("narrow")
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-narrow", tool_definition=narrow, first=now - timedelta(seconds=20)),
            endpoint("pod-wide", tool_definition=wide, first=now - timedelta(seconds=5)),
        )
        selection = held.select_copies(None, narrow.schema_digest, now=now, ttl=_TTL)
        assert [ep.pod_id for ep in selection.routable] == ["pod-narrow"]
        assert selection.routed_definitions["pod-narrow"] == narrow
        assert selection.definition_changed is False

    def test_a_schema_nobody_serves_any_more_is_a_changed_definition(self) -> None:
        """copies are visible, none serves the schema named: the caller must re-discover."""
        held = entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("a")))
        selection = held.select_copies(None, "0" * 64, ttl=_TTL)
        assert selection.routable == ()
        assert selection.definition_changed is True

    def test_nothing_visible_is_not_a_changed_definition(self) -> None:
        """no copy at all is unavailability, not a stale view."""
        held = entry("threetears.calculator", "1.0.0", endpoint("pod-A", "pending"))
        selection = held.select_copies(None, "0" * 64, ttl=_TTL)
        assert selection.definition_changed is False

    def test_available_to_agrees_with_the_selection(self) -> None:
        """a copy with no live definition does not make a tool available to anybody."""
        bare = entry("threetears.calculator", "1.0.0", ToolEndpoint(pod_id="pod-bare", status="available"))
        assert bare.available_to(None) is False
        announced_copy = entry("threetears.calculator", "1.0.0", endpoint("pod-A"))
        assert announced_copy.available_to(None) is True


class TestCopyStatus:
    """the answer a polling pod reads about its OWN copy."""

    def test_each_state_of_a_pods_own_copy(self) -> None:
        """available needs a live definition as well as a confirmed probe."""
        stale = datetime.now(UTC) - timedelta(seconds=600)
        held = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-ready"),
            endpoint("pod-pending", "pending"),
            endpoint("pod-stale", first=stale),
        )
        assert held.copy_status("pod-ready", ttl=_TTL) is CopyStatus.AVAILABLE
        assert held.copy_status("pod-pending", ttl=_TTL) is CopyStatus.PENDING
        assert held.copy_status("pod-stale", ttl=_TTL) is CopyStatus.UNAVAILABLE
        assert held.copy_status("pod-nobody", ttl=_TTL) is CopyStatus.ABSENT


class TestRemovingOneCopy:
    """a refused registration withdraws that pod's copy and nobody else's."""

    @pytest.mark.asyncio
    async def test_remove_copy_leaves_the_other_copies(self) -> None:
        """one copy gone, the incumbent untouched, the change persisted."""
        catalog = ToolCatalog()
        kv = _kv()
        await catalog.load_from_kv(kv)
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A"), endpoint("pod-B")))
        kv.put.reset_mock()

        assert await catalog.remove_copy(_FULL, "pod-A") is True

        held = catalog.get(_FULL)
        assert held is not None
        assert [ep.pod_id for ep in held.endpoints] == ["pod-B"]
        kv.put.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_removing_the_last_copy_removes_the_tool(self) -> None:
        """an entry nobody serves is not left behind."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A")))
        assert await catalog.remove_copy(_FULL, "pod-A") is True
        assert catalog.get(_FULL) is None

    @pytest.mark.asyncio
    async def test_removing_a_copy_that_is_not_there_changes_nothing(self) -> None:
        """a pod that never held a copy has nothing to withdraw."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A")))
        assert await catalog.remove_copy(_FULL, "pod-Z") is False
        assert await catalog.remove_copy("nothing@1.0", "pod-A") is False

    @pytest.mark.asyncio
    async def test_a_verified_copy_is_found_by_pod(self) -> None:
        """the registry can ask whether a pod id has ever registered as a verified publisher."""
        catalog = ToolCatalog()
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-A", verified_publisher=True)))
        await catalog.register(entry("threetears.calculator", "1.0.0", endpoint("pod-B")))
        assert catalog.pod_has_verified_copy("pod-A") is True
        assert catalog.pod_has_verified_copy("pod-B") is False


class TestDefinitionsSurviveTheProbeAndThePersistence:
    """the two places a copy is rebuilt rather than mutated."""

    @pytest.mark.asyncio
    async def test_mark_ready_persists_each_copys_definitions(self) -> None:
        """the KV projection written before the in-memory flip carries the definitions."""
        catalog = ToolCatalog()
        kv = _kv()
        await catalog.load_from_kv(kv)
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint("pod-A", "pending", tool_definition=definition("calc")))
        )
        kv.put.reset_mock()

        await catalog.mark_ready("pod-A")

        payload = json.loads(kv.put.call_args[0][1].decode("utf-8"))
        (persisted,) = payload["endpoints"]
        assert persisted["status"] == "available"
        assert [d["definition"]["description"] for d in persisted["definitions"]] == ["calc"]

    def test_a_persisted_entry_names_its_shape_and_holds_no_entry_definition(self) -> None:
        """new writes say which shape they are, so a reader never has to guess."""
        data = entry("threetears.calculator", "1.0.0", endpoint("pod-A", verified_publisher=True)).to_dict()
        assert data["shape"] == 2
        for gone in ("description", "input_schema", "output_schema", "timeout_seconds", "requires_confirmation"):
            assert gone not in data, gone
        assert data["endpoints"][0]["verified_publisher"] is True

    def test_a_current_entry_round_trips(self) -> None:
        """definitions, their announcement times and the publisher mark all come back."""
        original = entry(
            "threetears.calculator",
            "1.0.0",
            endpoint("pod-A", tool_definition=definition("calc", requires_confirmation=True), verified_publisher=True),
        )
        restored = CatalogEntry.from_dict(json.loads(json.dumps(original.to_dict())))
        copy = restored.get_endpoint("pod-A")
        original_copy = original.get_endpoint("pod-A")
        assert copy is not None and original_copy is not None
        assert copy.definitions == original_copy.definitions
        assert copy.verified_publisher is True

    def test_from_dict_refuses_an_entry_that_names_no_shape(self) -> None:
        """the old shape reaches the catalog only through the one translation point."""
        legacy = {
            "tool_name": "threetears.calculator",
            "tool_version": "1.0.0",
            "full_name": _FULL,
            "description": "whoever wrote last",
            "input_schema": {"type": "object"},
            "endpoints": [],
            "date_registered": datetime.now(UTC).isoformat(),
        }
        with pytest.raises(ValueError, match="shape"):
            CatalogEntry.from_dict(legacy)

    @pytest.mark.asyncio
    async def test_load_from_kv_translates_the_old_shape_and_drops_its_definition(self) -> None:
        """an entry-level definition of unknown provenance is not handed to any copy.

        It may be a stray's overwrite, so the copy comes back with NO definition and is shown
        to nobody until its pod announces again on its next heartbeat.
        """
        legacy = {
            "tool_name": "threetears.calculator",
            "tool_version": "1.0.0",
            "full_name": _FULL,
            "description": "STRAY",
            "input_schema": {"type": "object"},
            "output_schema": None,
            "timeout_seconds": 5.0,
            "requires_confirmation": False,
            "endpoints": [
                {"pod_id": "pod-A", "status": "available", "date_last_heartbeat": datetime.now(UTC).isoformat()}
            ],
            "date_registered": datetime.now(UTC).isoformat(),
        }
        kv = AsyncMock()
        kv.keys = AsyncMock(return_value=["threetears_calculator_AT_1_0_0"])
        stored = MagicMock()
        stored.value = json.dumps(legacy).encode("utf-8")
        kv.get = AsyncMock(return_value=stored)
        catalog = ToolCatalog()

        await catalog.load_from_kv(kv)

        held = catalog.get(_FULL)
        assert held is not None
        copy = held.get_endpoint("pod-A")
        assert copy is not None
        assert copy.definitions == {}
        assert copy.status == "unavailable"
        assert held.select_copies(None).visible == ()

    @pytest.mark.asyncio
    async def test_load_from_kv_keeps_a_current_entrys_definitions(self) -> None:
        """the translation leaves the current shape alone; only liveness is reset."""
        current = entry("threetears.calculator", "1.0.0", endpoint("pod-A", tool_definition=definition("calc")))
        kv = AsyncMock()
        kv.keys = AsyncMock(return_value=["k"])
        stored = MagicMock()
        stored.value = json.dumps(current.to_dict()).encode("utf-8")
        kv.get = AsyncMock(return_value=stored)
        catalog = ToolCatalog()

        await catalog.load_from_kv(kv)

        held = catalog.get(_FULL)
        assert held is not None
        copy = held.get_endpoint("pod-A")
        assert copy is not None
        assert [d.definition.description for d in copy.definitions.values()] == ["calc"]
        assert copy.status == "unavailable"


class TestDefinitionTtlConfig:
    """the window a definition stays live without being re-announced."""

    def test_default_is_three_heartbeats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """45 seconds at the 15-second default heartbeat."""
        from threetears.registry.config import get_definition_ttl

        monkeypatch.delenv("THREETEARS_REGISTRY_DEFINITION_TTL", raising=False)
        assert get_definition_ttl() == 45.0

    def test_env_overrides_and_nonsense_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a positive number is taken; zero, negative and text fall back to the default."""
        from threetears.registry.config import get_definition_ttl

        monkeypatch.setenv("THREETEARS_REGISTRY_DEFINITION_TTL", "90")
        assert get_definition_ttl() == 90.0
        for bad in ("0", "-5", "soon"):
            monkeypatch.setenv("THREETEARS_REGISTRY_DEFINITION_TTL", bad)
            assert get_definition_ttl() == 45.0
