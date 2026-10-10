"""The run-time algebra: what one run DID with the world, and what that forbids downstream.

Registration answers what a host *could* do; this answers what a run did, and both halves of
it are derived from the run's own record. A declared per-run mode would be a claim that can
disagree with the run, and it could not express the ordinary case — a run that seeds some
dimensions and witnesses others.

Two things are proved here, on the TOY host. The engine must compute this knowing nothing, so a
fixture whose every dimension hangs off one carrier and whose placements never move would let host
coupling pass unnoticed:

1. **The algebra.** One dimension lands in different quadrants across two runs of one host —
   the case no static map or mode flag can express.
2. **The refusal.** A knob registered as both a sweepable and a world dimension is refused,
   unconditionally, where both registries are in hand.

How a particular host derives its runs' placements, and how its operator surfaces disclose a
witnessed dimension, are that host's to prove.
"""

from __future__ import annotations

import pytest

from threetears.evals.kernel.host.profile import HostProfile, ProfileRegistrationError
from threetears.evals.kernel.host.sweepables import Sweepable, SweepableRegistry
from threetears.evals.kernel.host.world import WorldDimension, WorldRegistrationError, WorldRegistry
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_SWEEPABLE_REGISTRY
from packages.evals.tests.fixtures.toyhost.world import toyhost_world


#: The toy host's two carriers, spelled here so a test can attach one and not the other.
_PAGE_READER = "page_reader"
_CONSOLE = "console"


class TestThePlacementAlgebra:
    """Seeded x perceived, asked of one run, over a registry that never sees a name."""

    def test_a_seeded_and_perceived_dimension_is_representable(self):
        world, _state = toyhost_world()
        placements = world.place(seeded={"document_language"}, carriers={_PAGE_READER, _CONSOLE})
        assert placements["document_language"] == "representable"

    def test_a_seeded_dimension_no_surface_presents_is_judge_only(self):
        """The legitimate quadrant: a goal check grades what the subject was never shown."""
        world, _state = toyhost_world()
        placements = world.place(seeded={"vendor_template"}, carriers={_PAGE_READER, _CONSOLE})
        assert placements["vendor_template"] == "judge_only"

    def test_a_perceived_dimension_this_run_did_not_seed_is_witnessed(self):
        world, _state = toyhost_world()
        placements = world.place(seeded=set(), carriers={_PAGE_READER, _CONSOLE})
        assert placements["ingest_backlog"] == "witnessed"

    def test_a_dimension_neither_seeded_nor_perceived_is_out_of_play(self):
        """Refused at registration, ordinary here — which is why the two vocabularies differ."""
        world, _state = toyhost_world()
        placements = world.place(seeded=set(), carriers={_PAGE_READER, _CONSOLE})
        assert placements["vendor_template"] == "out_of_play"

    def test_the_same_dimension_lands_in_different_quadrants_across_two_runs_of_one_host(self):
        """The case a declared mode cannot express, and the reason both bits are derived.

        A host that commissions one run and observes another places one dimension two ways, and
        no static map beside the registry could say which — the run has to.
        """
        world, _state = toyhost_world()
        commissioned = world.place(seeded={"document_language"}, carriers={_PAGE_READER, _CONSOLE})
        observed = world.place(seeded=set(), carriers={_PAGE_READER, _CONSOLE})
        assert commissioned["document_language"] == "representable"
        assert observed["document_language"] == "witnessed"

    def test_a_dimension_whose_carrier_is_absent_is_out_of_play_however_the_run_asked(self):
        """The carrier gates seeding as well as perception, because the seed writes through it.

        Reporting a dimension ``judge_only`` because a template named it would record an
        instantiation that did not happen — the founding incident's own shape, one layer up.
        """
        world, _state = toyhost_world()
        placements = world.place(seeded={"ingest_backlog", "operator_corrections"}, carriers={_PAGE_READER})
        assert placements["ingest_backlog"] == "out_of_play"
        assert placements["operator_corrections"] == "out_of_play"

    def test_every_declaration_appears_because_out_of_play_is_a_finding(self):
        world, _state = toyhost_world()
        placements = world.place(seeded=set(), carriers=set())
        assert sorted(placements) == sorted(world.names)
        assert set(placements.values()) == {"out_of_play"}

    def test_a_dimension_no_run_can_seed_stays_witnessed_however_the_caller_asked(self):
        """The two algebras must not be able to disagree about one dimension.

        A registration saying ``seed=None`` is what makes a dimension witnessed, and a name
        appearing in the caller's ``seeded`` set is a report of what a run ASKED to set. If the
        ask could override the registration, a run record would assert an instantiation that
        never happened while the registry went on calling the same dimension uncontrollable —
        and the disclosure, which reads both, would drop it from every line it renders.
        """
        world, _state = toyhost_world()
        placements = world.place(seeded={"ingest_backlog"}, carriers={_PAGE_READER, _CONSOLE})
        assert world.capability("ingest_backlog") == "witnessed"
        assert placements["ingest_backlog"] == "witnessed"

    def test_a_seeded_name_this_registry_never_declared_is_ignored_rather_than_refused(self):
        """One question, one answer. The seeding path already refuses an undeclared seed."""
        world, _state = toyhost_world()
        placements = world.place(seeded={"not_a_dimension"}, carriers={_PAGE_READER, _CONSOLE})
        assert "not_a_dimension" not in placements

    def test_the_algebra_does_not_read_a_name(self):
        """Rename every dimension and every placement follows its registration, not its spelling."""
        world, _state = toyhost_world()
        renamed = WorldRegistry(
            [
                WorldDimension(
                    name=f"q{index}",
                    carrier=declared.carrier,
                    schema=declared.schema,
                    matters=declared.matters,
                    seed=declared.seed,
                    read=declared.read,
                    perturb=declared.perturb,
                    evidence=declared.evidence,
                    perceived_by=declared.perceived_by,
                    when=declared.when,
                )
                for index, declared in enumerate(world.declarations)
            ],
            bindings=world.bindings,
            subject_view=world.subject_view,
            perturb_ambient=world.perturb_ambient,
        )
        original = world.place(seeded={"document_language"}, carriers={_PAGE_READER, _CONSOLE})
        under_new_names = renamed.place(seeded={"q0"}, carriers={_PAGE_READER, _CONSOLE})
        assert list(under_new_names.values()) == list(original.values())


class TestSelectingByCapability:
    """The registration-time selector the disclosure surfaces read."""

    def test_each_quadrant_selects_only_its_own(self):
        world, _state = toyhost_world()
        assert [d.name for d in world.of_capability("witnessed")] == ["ingest_backlog"]
        assert [d.name for d in world.of_capability("judge_only")] == ["vendor_template", "supervisor_signoff"]
        assert "document_language" in [d.name for d in world.of_capability("representable")]

    def test_an_empty_answer_is_an_answer(self):
        """A host every dimension of which a run can seed has no witnessed state to disclose."""
        world = WorldRegistry(
            [
                WorldDimension(
                    name="shelf_stock",
                    carrier="shelf_sensor",
                    schema={"type": "integer", "minimum": 0},
                    matters="restocking scenarios presume a depleted shelf",
                    seed="h.seed",
                    read="h.read",
                )
            ],
            bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
        )
        assert world.of_capability("witnessed") == ()


class TestTheCarrierIsRequired:
    """A dimension that names no carrier cannot be placed for any subject."""

    def test_a_blank_carrier_is_refused_at_registration(self):
        with pytest.raises(WorldRegistrationError, match="names no carrier"):
            WorldRegistry(
                [
                    WorldDimension(
                        name="shelf_stock",
                        carrier="   ",
                        schema={"type": "integer"},
                        matters="restocking scenarios presume a depleted shelf",
                        seed="h.seed",
                        read="h.read",
                    )
                ],
                bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
            )


def _profile_with(*, sweepables: SweepableRegistry | None = None) -> HostProfile:
    """A toy profile whose two registries can be made to overlap.

    Args:
        sweepables: A replacement sweepables registry, for introducing an overlap.

    Returns:
        The constructed profile.

    Raises:
        ProfileRegistrationError: The two registries contradict each other, which is what
            several tests here are asserting.
    """
    base = toyhost_profile()
    world, _state = toyhost_world()
    return HostProfile(
        host_id=base.host_id,
        host_sweepables=sweepables if sweepables is not None else base.host_sweepables,
        measures=base.measures,
        bars=base.bars,
        style=base.style,
        caveat_kinds=base.caveat_kinds,
        world=world,
        variant_levers=base.variant_levers,
        kinds=base.kinds,
    )


def _sweepables_naming(dimension: str) -> SweepableRegistry:
    """The toy host's sweepables plus one lever sharing a world dimension's name.

    Args:
        dimension: The world dimension name to collide with.

    Returns:
        The extended registry.
    """
    return TOYHOST_SWEEPABLE_REGISTRY.extend(
        [
            Sweepable(
                name=dimension,
                role="lever",
                read=lambda run, results: None,
                reader_prose="how many documents were queued behind this batch when it ran",
            )
        ]
    )


class TestTheOverlapRefusal:
    """One name in both registries is refused, unconditionally, checked where both are in hand."""

    def test_an_undeclared_overlap_is_refused_naming_the_knob(self):
        with pytest.raises(ProfileRegistrationError, match="ingest_backlog"):
            _profile_with(sweepables=_sweepables_naming("ingest_backlog"))

    def test_a_second_undeclared_overlap_is_refused_the_same_way(self):
        """Per forbidden shape, not once: a refusal proved on one name is proved on one name."""
        with pytest.raises(ProfileRegistrationError, match="document_language"):
            _profile_with(sweepables=_sweepables_naming("document_language"))

    def test_registries_sharing_no_name_register(self):
        """The acceptance case on the same fixture, so the refusal is shown to discriminate."""
        profile = _profile_with()
        assert profile.world is not None
        assert not set(profile.sweepables.names) & set(profile.world.names)

    def test_a_host_with_no_world_has_no_overlap_to_refuse(self):
        assert toyhost_profile(with_world=False).world is None
