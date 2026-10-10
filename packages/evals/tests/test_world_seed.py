"""The shared seed walk — every refusal it makes, driven through the walk itself rather than through a host.

The walk lives once, in the engine. Were it copied by hand into each host, each host's tests would
prove its own copy and nothing would prove the rules once. These run over the toy host's vocabulary, the one the
engine has never been taught, and over a dotted-name registry, so a refusal that leaned on either host's naming
would fail one or the other. Each refusal is pinned by the kind it reports and by the path its message names,
because the kind is what a host translates and the path is what an author edits.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel.host import SeedRefused, SeedWrite, WorldDimension, WorldRegistry, check_seed
from packages.evals.tests.fixtures.toyhost.world import toyhost_world


#: The toy host's two carriers, as its kind attaches them.
_ATTACHED = ("page_reader", "console")

#: A seed the toy world can take in full: one document dimension and one triggered operator one.
_SOUND = {
    "page_reader": {"document_language": "de", "scan_quality": "faint"},
    "console": {"operator_corrections": ["x"]},
}


class TestASoundSeed:
    """What the walk hands back, and what it does not do."""

    def test_every_value_comes_back_as_a_write_through_its_seed_handle_in_seed_order(self) -> None:
        registry, _state = toyhost_world()

        writes = check_seed(registry, _SOUND, attached=_ATTACHED)

        assert writes == (
            SeedWrite("page_reader", "document_language", "toy.seed_language", "de"),
            SeedWrite("page_reader", "scan_quality", "toy.seed_scan_quality", "faint"),
            SeedWrite("console", "operator_corrections", "toy.arm_operator_corrections", ["x"]),
        )

    def test_the_walk_writes_nothing_so_a_refusal_cannot_leave_a_world_half_seeded(self) -> None:
        """The caller applies the writes; the walk only checks them, even when every one of them is sound."""
        registry, state = toyhost_world()

        check_seed(registry, _SOUND, attached=_ATTACHED)

        assert (state.document_language, state.scan_quality) == ("en", "clean")


class TestEachRefusal:
    """One case per question the walk asks, each on an otherwise-sound seed so the refusal is that question's."""

    @pytest.mark.parametrize(
        ("namespaces", "kind", "name", "named"),
        [
            ({"page_reader": ["de"]}, "malformed", None, "'page_reader'"),
            ({"page_reader": {"documnet_language": "de"}}, "undeclared", "documnet_language", "'documnet_language'"),
            ({"console": {"document_language": "de"}}, "misplaced", "document_language", "supplied by 'page_reader'"),
            ({"console": {"ingest_backlog": 3}}, "unseedable", "ingest_backlog", "'ingest_backlog'"),
            (
                {"page_reader": {"document_language": "nl"}},
                "nonconforming",
                "document_language",
                "document_language is 'nl'",
            ),
            ({"page_reader": {"document_language": "de"}}, "unattached", "document_language", "'page_reader'"),
        ],
        ids=["malformed", "undeclared", "misplaced", "unseedable", "nonconforming", "unattached"],
    )
    def test_the_refusal_names_its_kind_and_its_path(
        self, namespaces: dict[str, Any], kind: str, name: str | None, named: str
    ) -> None:
        registry, _state = toyhost_world()
        attached = ("console",) if kind == "unattached" else _ATTACHED

        with pytest.raises(SeedRefused) as refused:
            check_seed(registry, namespaces, attached=attached)

        assert refused.value.kind == kind
        assert refused.value.name == name
        assert named in str(refused.value), str(refused.value)

    def test_a_nonconforming_value_names_every_path_it_fails_at(self) -> None:
        """An author fixing the first would otherwise meet the next one a run later."""
        registry, _state = toyhost_world()

        with pytest.raises(SeedRefused) as refused:
            check_seed(registry, {"console": {"operator_corrections": ["reprice", 3, 4]}}, attached=_ATTACHED)

        assert refused.value.violations == (
            "operator_corrections[1] is int 3, not string",
            "operator_corrections[2] is int 4, not string",
        )

    def test_an_unattached_refusal_says_what_is_attached(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(SeedRefused) as refused:
            check_seed(registry, {"console": {"operator_corrections": []}}, attached=("page_reader",))

        assert refused.value.attached == ("page_reader",)
        assert "attached: page_reader" in str(refused.value)

    def test_a_refusal_is_a_value_error_so_a_host_with_no_wording_of_its_own_can_catch_it_as_one(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(ValueError, match="documnet_language"):
            check_seed(registry, {"page_reader": {"documnet_language": "de"}})


class TestTheRegistryIsAskedBeforeTheSubject:
    """A template defect is reported as one, not as a carrier this subject happens not to hold."""

    def test_an_undeclared_key_under_an_unattached_carrier_is_refused_as_undeclared(self) -> None:
        registry, _state = toyhost_world()

        with pytest.raises(SeedRefused) as refused:
            check_seed(registry, {"page_reader": {"documnet_language": "de"}}, attached=("console",))

        assert refused.value.kind == "undeclared"

    def test_a_later_namespace_s_registry_refusal_outranks_an_earlier_one_s_attachment(self) -> None:
        registry, _state = toyhost_world()
        seed = {"page_reader": {"document_language": "de"}, "console": {"ingest_backlog": 3}}

        with pytest.raises(SeedRefused) as refused:
            check_seed(registry, seed, attached=("console",))

        assert refused.value.kind == "unseedable"

    def test_with_no_subject_the_attachment_is_not_asked(self) -> None:
        """At authoring no subject exists, so the registry's questions are the only ones there are."""
        registry, _state = toyhost_world()

        assert len(check_seed(registry, {"page_reader": {"document_language": "de"}})) == 1


def _dotted(namespace: str, key: str) -> str:
    return f"{namespace}.{key}"


def _dotted_registry(address=_dotted) -> WorldRegistry:
    """A host whose dimension names compose the carrier in (``shelf.stock``).

    Args:
        address: The addressing it declares; the composition by default.

    Returns:
        The registry.
    """
    return WorldRegistry(
        [
            WorldDimension(
                name="shelf.stock",
                carrier="shelf",
                schema={"type": "integer", "minimum": 0},
                matters="restocking scenarios presume a depleted shelf and prove nothing against a full one",
                seed="h.seed",
                read="h.read",
                perceived_by=("aisle",),
            ),
            WorldDimension(
                name="till.float",
                carrier="till",
                schema={"type": "integer"},
                matters="a change-giving scenario presumes the till holds a float to give change from",
                seed="h.seed_float",
                read="h.read_float",
                perceived_by=("aisle",),
            ),
        ],
        bindings={
            "h.seed": lambda value: None,
            "h.read": lambda: 0,
            "h.seed_float": lambda value: None,
            "h.read_float": lambda: 0,
            "h.view": lambda **_: {},
        },
        subject_view="h.view",
        address=address,
    )


class TestWhatAHostSupplies:
    """Addressing (declared on the registry) and the two skips are the host's; every rule after them is still the walk's."""

    def test_a_host_s_addressing_names_the_dimension(self) -> None:
        writes = check_seed(_dotted_registry(), {"shelf": {"stock": 3}})

        assert writes == (SeedWrite("shelf", "shelf.stock", "h.seed", 3),)

    def test_a_host_s_addressing_cannot_carry_a_value_past_the_carrier_check(self) -> None:
        """An address that composes a name another carrier supplies is still refused as misplaced."""
        with pytest.raises(SeedRefused) as refused:
            check_seed(_dotted_registry(address=lambda _namespace, key: f"till.{key}"), {"shelf": {"float": 3}})

        assert refused.value.kind == "misplaced"
        assert "supplied by 'till'" in str(refused.value)

    def test_a_key_the_run_writes_itself_is_refused_like_any_undeclared_key(self) -> None:
        """No pass-over: the walk once skipped a run-owned ledger key, and a skipped key is an unchecked one.

        ``check_seed`` takes no skip arguments any more, so the one shape that needed them — a key no
        dimension declares, sitting beside one that does — is refused, naming the undeclared key.
        """
        with pytest.raises(SeedRefused) as refused:
            check_seed(_dotted_registry(), {"shelf": {"stock": 3, "ledger": []}})

        assert refused.value.kind == "undeclared"
        assert refused.value.name == "shelf.ledger"
