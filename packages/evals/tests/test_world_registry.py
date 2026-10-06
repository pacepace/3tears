"""The world registry — its refusals, its capability algebra, and its one call path.

Every test here constructs through the **real** registration API. A hand-assembled registry
shape would test the reader rather than the writer, and the refusals are the whole point of this
record: a registration can be wrong on its face, and catching it at birth is cheaper than
catching it as a run that behaved oddly.

The toy host is the fixture on purpose. The engine must compute these answers over a host it has
never seen, so the quadrant tests run over invoice-extraction vocabulary rather than a conversational host's —
which is also what makes a name-branching regression visible here rather than in production.
"""

from __future__ import annotations

import pytest

from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.host.world import (
    Triggered,
    WorldDimension,
    WorldRegistrationError,
    WorldRegistry,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.world import toyhost_world


def _sound_dimension(**overrides: object) -> WorldDimension:
    """A dimension that registers cleanly, so a test can break exactly one thing about it.

    Args:
        **overrides: Fields to replace.

    Returns:
        The dimension.
    """
    fields: dict[str, object] = {
        "name": "shelf_stock",
        "schema": {"type": "integer", "minimum": 0},
        "matters": "restocking scenarios presume a depleted shelf and prove nothing against a full one",
        "carrier": "shelf_sensor",
        "seed": "h.seed",
        "read": "h.read",
    }
    fields.update(overrides)
    return WorldDimension(**fields)  # type: ignore[arg-type]


_SOUND_BINDINGS = {"h.seed": lambda value: None, "h.read": lambda: 0, "h.view": lambda **_: {}}


class TestRegistrationRefusals:
    """A registration that is wrong on its face is refused where it is written."""

    def test_a_dimension_with_no_matters_prose_is_refused(self) -> None:
        """A bare dimension name is a label — and here there is no role axis to exempt one."""
        with pytest.raises(WorldRegistrationError, match="states no reason it matters"):
            WorldRegistry([_sound_dimension(matters="   ")], bindings=_SOUND_BINDINGS)

    def test_a_dimension_that_is_neither_seedable_nor_perceivable_is_refused(self) -> None:
        """The fourth quadrant. No run can set it and no subject can see it — that is a field."""
        with pytest.raises(WorldRegistrationError, match="which is a field, not a dimension"):
            WorldRegistry([_sound_dimension(seed=None, read=None)], bindings=_SOUND_BINDINGS)

    def test_a_seed_with_no_read_is_refused(self) -> None:
        """An instantiation nobody can verify took is the defect this contract exists to catch.

        Declared at birth rather than discovered as a run that behaved as though the seed had not
        landed — which is indistinguishable, from the outside, from a subject behaving oddly.
        """
        with pytest.raises(WorldRegistrationError, match="an instantiation nobody can verify took"):
            WorldRegistry([_sound_dimension(read=None)], bindings=_SOUND_BINDINGS)

    def test_a_blank_name_is_refused(self) -> None:
        """The name is the key the whole registry is addressed by, and the one field with no rule.

        A nameless dimension can be neither presumed by a template nor named in a report — and it
        would be reported on as the empty string, which reads as an engine bug rather than a
        registration one.
        """
        with pytest.raises(WorldRegistrationError, match="declares a blank name"):
            WorldRegistry([_sound_dimension(name="   ")], bindings=_SOUND_BINDINGS)

    def test_a_perturbation_handle_beside_a_seed_handle_is_refused(self) -> None:
        """A run already sets this dimension, so a second write path is the parallel path refused.

        Conformance driven through the second path would prove a path no run takes, which is the
        whole reason the seed handle must BE the runner's own seeding path rather than a mirror
        of it.
        """
        with pytest.raises(WorldRegistrationError, match="declares a perturbation handle beside its seed handle"):
            WorldRegistry(
                [_sound_dimension(perturb="h.force")],
                bindings={**_SOUND_BINDINGS, "h.force": lambda value: None},
            )

    def test_a_human_trigger_that_names_a_fire_handle_is_refused(self) -> None:
        """Code that can fire the condition is proof the condition is not human-only.

        Left standing, the dimension would be recorded "not instantiable in an unattended run"
        while an unattended run could instantiate it — a disclosure that is exactly backwards.
        """
        with pytest.raises(WorldRegistrationError, match="triggered by a human and names fire handle"):
            WorldRegistry(
                [_sound_dimension(when=Triggered(kind="human", condition="a person acts", fire="h.fire"))],
                bindings={**_SOUND_BINDINGS, "h.fire": lambda: None},
            )

    def test_a_perturbation_handle_with_no_read_handle_is_refused(self) -> None:
        """The seed-without-read refusal by the other door, and it was missing.

        Left standing, a perturbation binding wired to nothing makes perception A/B report that
        nothing on the surface carries the dimension — sending the host to inspect a renderer that
        is fine, while no other check even runs on a witnessed dimension to name the real defect.
        """
        with pytest.raises(WorldRegistrationError, match="declares a perturbation handle but no read handle"):
            WorldRegistry(
                [_sound_dimension(seed=None, read=None, perturb="h.force", perceived_by=("aisle",))],
                bindings={**_SOUND_BINDINGS, "h.force": lambda value: None},
                subject_view="h.view",
            )

    def test_a_schema_whose_own_bounds_admit_nothing_is_refused(self) -> None:
        """Record incoherence, checkable with nothing but the schema in hand.

        Nested schemas count: an ``items`` a generator can never satisfy is the same defect one
        level down, and it would otherwise surface mid-run as an engine error.
        """
        with pytest.raises(WorldRegistrationError, match=r"schema declares minimum=5 above maximum=1"):
            WorldRegistry(
                [_sound_dimension(schema={"type": "integer", "minimum": 5, "maximum": 1})], bindings=_SOUND_BINDINGS
            )

        with pytest.raises(WorldRegistrationError, match=r"schema\.items declares minLength=4 above maxLength=2"):
            WorldRegistry(
                [
                    _sound_dimension(
                        schema={"type": "array", "items": {"type": "string", "minLength": 4, "maxLength": 2}}
                    )
                ],
                bindings=_SOUND_BINDINGS,
            )

    @pytest.mark.parametrize(
        ("schema", "says"),
        [
            (
                {"type": "object", "properties": {"count": {"type": "integer", "minimum": 5, "maximum": 1}}},
                r"schema\.count declares minimum=5 above maximum=1",
            ),
            (
                {"type": "object", "properties": {}, "additionalProperties": {"type": "string", "enum": [3]}},
                r"schema\.additionalProperties declares type='string' and its enum names 3",
            ),
            (
                {"anyOf": [{"type": "object", "properties": {}}, {"type": "integer", "minimum": 5, "maximum": 1}]},
                r"schema\.anyOf\[1\] declares minimum=5 above maximum=1",
            ),
            (
                {"anyOf": [{"anyOf": [{"type": "string", "minLength": 4, "maxLength": 2}]}]},
                r"schema\.anyOf\[0\]\.anyOf\[0\] declares minLength=4 above maxLength=2",
            ),
            ({"anyOf": []}, r"schema declares an empty anyOf, which offers no shape"),
        ],
    )
    def test_a_contradiction_anywhere_a_schema_can_sit_is_refused(self, schema: dict[str, object], says: str) -> None:
        """Every place a nested schema can sit — a property, ``additionalProperties``, an ``anyOf`` branch.

        This once descended into ``items`` alone, so a property or a shape bounding its value to nothing
        registered cleanly and surfaced mid-run as an engine error.
        """
        with pytest.raises(WorldRegistrationError, match=says):
            WorldRegistry([_sound_dimension(schema=schema)], bindings=_SOUND_BINDINGS)

    def test_a_coherent_shape_list_registers(self) -> None:
        """The acceptance half on the same keyword, so a check refusing every ``anyOf`` cannot pass the table above."""
        schema = {
            "anyOf": [
                {"type": "object", "properties": {}, "additionalProperties": False},
                {"type": "integer", "minimum": 0},
            ]
        }

        WorldRegistry([_sound_dimension(schema=schema)], bindings=_SOUND_BINDINGS)

    def test_an_enum_contradicting_its_own_declared_type_is_refused(self) -> None:
        """Where the two disagree one is a typo, and nothing downstream can tell which.

        A generator reading the enum produces values the type forbids; one reading the type
        produces values the enum forbids. Refusing it is the same move as the bound pairs — the
        record incoherent on its face — and it closes the one path where the kit still exempts
        ``type`` from its keyword audit, since ``enum`` does not honour it.
        """
        with pytest.raises(WorldRegistrationError, match=r"declares type='integer' and its enum names 'a'"):
            WorldRegistry([_sound_dimension(schema={"type": "integer", "enum": [1, "a"]})], bindings=_SOUND_BINDINGS)

        # `bool` is a subclass of `int` in Python and is not an integer in JSON Schema, so the
        # naive isinstance check would call this coherent.
        with pytest.raises(WorldRegistrationError, match="its enum names True"):
            WorldRegistry([_sound_dimension(schema={"type": "integer", "enum": [True]})], bindings=_SOUND_BINDINGS)

    def test_an_enum_is_checked_against_a_union_type_as_the_union(self) -> None:
        """Reading only a scalar ``type`` let a union escape BOTH refusals.

        The kit resolves an ``enum`` schema before it ever looks at ``type``, so its own
        union-type refusal never fires for one — registration is the only place this shape can be
        caught, and it was skipping every list-valued type.
        """
        with pytest.raises(WorldRegistrationError, match="its enum names 1"):
            WorldRegistry(
                [_sound_dimension(schema={"type": ["string", "null"], "enum": [1]})], bindings=_SOUND_BINDINGS
            )

    def test_const_is_held_to_every_rule_an_enum_is(self) -> None:
        """`const` is a one-value enum, and it survived three earlier fixes by having its own branch.

        Normalising it into the enum path rather than giving it a parallel check is what closes
        the class: every rule here reaches it, including the ones added before it did.
        """
        with pytest.raises(WorldRegistrationError, match=r"declares type='integer' and its const names 'abc'"):
            WorldRegistry([_sound_dimension(schema={"type": "integer", "const": "abc"})], bindings=_SOUND_BINDINGS)

        with pytest.raises(WorldRegistrationError, match="its const names 1"):
            WorldRegistry([_sound_dimension(schema={"type": ["string", "null"], "const": 1})], bindings=_SOUND_BINDINGS)

        for sound in ({"type": "integer", "const": 3}, {"type": "integer", "const": 3.0}, {"const": 0}):
            assert WorldRegistry([_sound_dimension(schema=sound)], bindings=_SOUND_BINDINGS).names == ("shelf_stock",)

    def test_an_enum_that_admits_nothing_is_refused(self) -> None:
        """The same category as bounds that cross: a record wrong on its face."""
        with pytest.raises(WorldRegistrationError, match="enumerates nothing"):
            WorldRegistry([_sound_dimension(schema={"enum": []})], bindings=_SOUND_BINDINGS)

    def test_an_enum_agreeing_with_its_type_registers_normally(self) -> None:
        """The ordinary shapes a host writes, which must not be collateral of the refusals above.

        A false refusal on a correct registration is worse than having no check: it is the canary
        crying wolf, and a canary that cries wolf gets switched off. ``1.0`` IS an integer in JSON
        Schema — a number with no fractional part — so a bare ``isinstance`` against ``int`` would
        reject a legal declaration. A bare enum with no declared type is fine too: nothing for it
        to contradict.
        """
        for schema in (
            {"type": "string", "enum": ["a", "b"]},
            {"type": "number", "enum": [1, 2.5]},
            {"type": "integer", "enum": [1.0]},
            {"type": "boolean", "enum": [True, False]},
            {"type": ["string", "null"], "enum": ["a", None]},
            {"enum": [1, "a"]},
        ):
            assert WorldRegistry([_sound_dimension(schema=schema)], bindings=_SOUND_BINDINGS).names == ("shelf_stock",)

    def test_a_binding_the_engine_could_not_call_is_refused_at_registration(self) -> None:
        """Resolvability and callability say nothing about arity, and arity is contract.

        A host that binds its renderer as ``render(attached)`` rather than ``render(*, surfaces)``
        passes every other refusal and then fails as a TypeError inside a run, attributed to
        whatever the run was doing — the exact failure the unresolvable-handle refusal prevents.
        """
        with pytest.raises(WorldRegistrationError, match="subject_view handle 'h.view' is bound to"):
            WorldRegistry(
                [_sound_dimension(perceived_by=("aisle",))],
                bindings={**_SOUND_BINDINGS, "h.view": lambda attached: {}},
                subject_view="h.view",
            )

        with pytest.raises(WorldRegistrationError, match="seed handle 'h.seed' is bound to"):
            WorldRegistry([_sound_dimension()], bindings={**_SOUND_BINDINGS, "h.seed": lambda: None})

    def test_a_binding_with_no_introspectable_signature_is_accepted(self) -> None:
        """A callable whose signature cannot be derived is a real binding, not a bad one.

        Compiled extension callables are the case: ``inspect.signature`` raises on them rather
        than returning something to check. Refusing what cannot be checked would make this a
        canary that cries wolf, and a canary that cries wolf gets switched off.
        """

        class Opaque:
            """Stands in for a compiled callable — signature derivation raises rather than answers."""

            @property
            def __signature__(self) -> object:
                raise ValueError("no signature found")

            def __call__(self) -> int:
                return 0

        world = WorldRegistry(
            [_sound_dimension(seed="h.seed", read="h.opaque")],
            bindings={**_SOUND_BINDINGS, "h.opaque": Opaque()},
        )

        assert world.names == ("shelf_stock",)

    def test_an_ambient_perturbation_with_no_subject_view_is_refused(self) -> None:
        """The only thing ambient perturbation proves is that the subject's view did not move.

        Without a view it is a capability nothing can ever read — the same defect as a
        ``perceived_by`` on a registry that cannot render one, from the other side.
        """
        with pytest.raises(
            WorldRegistrationError, match="perturb_ambient is declared but this registry has no subject_view"
        ):
            WorldRegistry(
                [_sound_dimension()],
                bindings={**_SOUND_BINDINGS, "h.shift": lambda: None},
                perturb_ambient="h.shift",
            )

    def test_every_optional_handle_is_resolved_at_registration_like_every_other(self) -> None:
        """The optional capabilities are handles, so a typo in one is checkable where it is written.

        Exempting them would put the newest and least-exercised bindings on the one path that
        fails inside a run instead of at registration.
        """
        for dimension, pattern in (
            (
                _sound_dimension(seed=None, read="h.read", perturb="h.typo", perceived_by=("x",)),
                "perturb handle 'h.typo'",
            ),
            (
                _sound_dimension(when=Triggered(kind="turn", condition="a turn passes", fire="h.typo")),
                "fire handle 'h.typo'",
            ),
        ):
            with pytest.raises(WorldRegistrationError, match=pattern):
                WorldRegistry([dimension], bindings=_SOUND_BINDINGS, subject_view="h.view")

        with pytest.raises(WorldRegistrationError, match="perturb_ambient names handle 'h.typo'"):
            WorldRegistry(
                [_sound_dimension()], bindings=_SOUND_BINDINGS, subject_view="h.view", perturb_ambient="h.typo"
            )

    def test_a_handle_with_no_binding_is_refused(self) -> None:
        """The registry knows its own resolution table, so a typo is checkable at registration."""
        with pytest.raises(WorldRegistrationError, match="'h.typo' is not in this registry's binding table"):
            WorldRegistry([_sound_dimension(seed="h.typo")], bindings=_SOUND_BINDINGS)

    def test_a_perceived_dimension_is_refused_when_the_registry_has_no_subject_view(self) -> None:
        """An unverifiable perception claim is the defect with a declaration wrapped around it.

        Perception A/B is what makes a ``perceived_by`` survive a refactor: delete the renderer
        and the declaration fails the next day. With no ``subject_view`` there is nothing to vary
        against, so the claim could never fail and would go on reading as supported forever.
        """
        with pytest.raises(WorldRegistrationError, match="has no subject_view to render it"):
            WorldRegistry([_sound_dimension(perceived_by=("shelf_panel",))], bindings=_SOUND_BINDINGS)

    def test_a_perceived_dimension_registers_once_a_subject_view_is_bound(self) -> None:
        """The refusal above is conditional, not a ban on declaring perception.

        Without this the previous test passes just as well against an implementation that
        refused every ``perceived_by`` outright, which is a different and wrong rule.
        """
        world = WorldRegistry(
            [_sound_dimension(perceived_by=("shelf_panel",))],
            bindings=_SOUND_BINDINGS,
            subject_view="h.view",
        )

        assert world.capability("shelf_stock") == "representable"

    def test_a_subject_view_handle_with_no_binding_is_refused(self) -> None:
        """Same rule as a dimension's handles, on the one handle the registry owns itself."""
        with pytest.raises(WorldRegistrationError, match="subject_view names handle 'h.missing'"):
            WorldRegistry([_sound_dimension()], bindings=_SOUND_BINDINGS, subject_view="h.missing")

    def test_a_triggered_dimension_with_no_condition_is_refused(self) -> None:
        """Seeding a triggered dimension arms it; with no condition, nothing could ever fire it."""
        with pytest.raises(WorldRegistrationError, match="is triggered but names no condition"):
            WorldRegistry(
                [_sound_dimension(when=Triggered(kind="turn", condition=" "))],
                bindings=_SOUND_BINDINGS,
            )

    def test_a_schema_that_is_not_a_mapping_is_refused_naming_the_dimension(self) -> None:
        """The type guard's own claim: it fails here rather than inside a generator later.

        A schema is what synthesizes a value for round-trip and what validates a seed, so a
        non-mapping reaches its first confusing failure a long way from the registration that
        caused it.
        """
        with pytest.raises(WorldRegistrationError, match="shelf_stock's schema is not a mapping"):
            _sound_dimension(schema="integer")

    def test_two_dimensions_of_one_name_are_refused(self) -> None:
        """The registry keys by name, so a duplicate would shadow rather than collide."""
        with pytest.raises(WorldRegistrationError, match="is declared twice"):
            WorldRegistry([_sound_dimension(), _sound_dimension()], bindings=_SOUND_BINDINGS)

    def test_labeled_evidence_with_no_read_handle_is_refused(self) -> None:
        """Evidence describes a read, so with none it is a claim nothing will ever render.

        The input registry refuses the identical shape — a confound reason on a declaration no
        scan speaks for — and this is that rule on the axis this registry owns.
        """
        with pytest.raises(WorldRegistrationError, match="declares labeled evidence but no read handle"):
            WorldRegistry(
                [_sound_dimension(seed=None, read=None, perceived_by=("shelf_panel",), evidence="labeled")],
                bindings=_SOUND_BINDINGS,
                subject_view="h.view",
            )

    def test_a_binding_that_is_not_callable_is_refused(self) -> None:
        """A URL in the table resolves and then fails as a bare TypeError inside a run.

        That is the state the unresolvable-handle refusal exists to prevent, reached by the other
        door — a handle that resolves to something that cannot be called.
        """
        with pytest.raises(WorldRegistrationError, match="binding 'h.read' is not callable"):
            WorldRegistry([_sound_dimension()], bindings={"h.seed": lambda value: None, "h.read": "https://tv/state"})

    def test_every_defect_in_one_set_is_reported_at_once(self) -> None:
        """A host fixing its registrations sees the whole list, not one per construction."""
        with pytest.raises(WorldRegistrationError) as caught:
            WorldRegistry(
                [
                    _sound_dimension(name="a", matters=""),
                    _sound_dimension(name="b", read=None),
                ],
                bindings=_SOUND_BINDINGS,
            )

        message = str(caught.value)
        assert "a states no reason it matters" in message
        assert "b declares a seed handle but no read handle" in message


class TestCapabilityAlgebra:
    """The registration-time quadrants, computed from shape and never from a name."""

    def test_a_seedable_perceivable_dimension_is_representable(self) -> None:
        """The precondition a scenario presumes, instantiated — the quadrant everyone designs for."""
        world, _state = toyhost_world()

        assert world.capability("scan_quality") == "representable"

    def test_a_seedable_unperceived_dimension_is_judge_only(self) -> None:
        """Legitimate: a goal check may read what the subject never saw.

        The toy host's adjudicated vendor template is exactly this — showing the extractor the
        answer would measure nothing, and the goal check still needs it.
        """
        world, _state = toyhost_world()

        assert world.capability("vendor_template") == "judge_only"

    def test_a_perceived_unseedable_dimension_is_witnessed(self) -> None:
        """The quadrant no existing eval design names, and the one the founding incident occupied.

        "We have no such concept" and "your subject sees this and your experiment does not control
        it" are different findings with different remedies. A map enumerating only seedable things
        reports the second as the first — right by accident, wrong in meaning.
        """
        world, _state = toyhost_world()

        assert world.capability("ingest_backlog") == "witnessed"

    def test_the_refused_shape_has_no_quadrant_rather_than_the_nearest_one(self) -> None:
        """``WorldDimension`` is public and constructible standalone, so the property must refuse.

        Answering ``witnessed`` for a field would make the one shape the design refuses
        indistinguishable from the one it most wants disclosed.
        """
        field_not_a_dimension = _sound_dimension(seed=None, read=None)

        with pytest.raises(WorldRegistrationError, match="has no quadrant"):
            _ = field_not_a_dimension.capability

    def test_an_undeclared_name_is_not_a_quadrant(self) -> None:
        """None and ``witnessed`` are different answers and collapsing them loses the remedy."""
        world, _state = toyhost_world()

        assert world.capability("shelf_stock") is None

    def test_a_perturbable_witnessed_dimension_is_still_witnessed(self) -> None:
        """The algebra reads ``seed`` alone, and that is the truth about what RUNS can do.

        A host with a test rig can force ``ingest_backlog``; no run can. Letting the conformance
        capability promote the quadrant would report the dangerous quadrant as representable —
        the founding incident, re-created by the machinery built to catch it.
        """
        world, _state = toyhost_world()
        declared = world.get("ingest_backlog")
        assert declared is not None

        assert declared.perturb is not None
        assert not declared.seedable
        assert world.capability("ingest_backlog") == "witnessed"

    def test_the_toy_host_supplies_every_registrable_quadrant(self) -> None:
        """A fixture that cannot produce the dangerous quadrant proves nothing about it.

        Three quadrants are registrable; the fourth is refused at registration
        (``TestRegistrationRefusals``), so it has no fixture by construction rather than by
        omission.
        """
        world, _state = toyhost_world()

        assert {declared.capability for declared in world.declarations} == {
            "representable",
            "judge_only",
            "witnessed",
        }


class TestTheEngineNeverInterpretsADimensionName:
    """Host-agnosticism, asserted rather than documented."""

    def test_the_quadrant_of_a_renamed_dimension_does_not_move(self) -> None:
        """Rename every dimension to nonsense; the algebra returns the same answers.

        This is the mechanical form of "the engine never interprets a dimension name". A branch
        on a name would show up here and nowhere else, because every other test uses names that
        happen to read sensibly.
        """
        world, _state = toyhost_world()
        renamed = WorldRegistry(
            [
                WorldDimension(
                    name=f"q{index}",
                    schema=declared.schema,
                    matters=declared.matters,
                    carrier=declared.carrier,
                    seed=declared.seed,
                    read=declared.read,
                    perceived_by=declared.perceived_by,
                )
                for index, declared in enumerate(world.declarations)
            ],
            bindings=world.bindings,
            subject_view=world.subject_view,
        )

        assert [declared.capability for declared in renamed.declarations] == [
            declared.capability for declared in world.declarations
        ]


class TestTheBindingTable:
    """One call path, whether the binding is in-process or across a service boundary."""

    async def test_a_synchronous_binding_is_called_and_returned(self) -> None:
        """The ordinary case — an in-memory world."""
        world, state = toyhost_world()

        await world.call("toy.seed_language", "de")

        assert await world.call("toy.read_language") == "de"
        assert state.document_language == "de"

    async def test_an_asynchronous_binding_is_awaited_by_the_same_call(self) -> None:
        """No second contract and no migration between the two kinds of host.

        A host whose world is a live device reached by async RPC is the consumer this contract
        exists for, so the awaitable path is exercised from the start rather than added
        when one shows up.
        """
        world, state = toyhost_world()
        state.ingest_backlog = 12

        assert await world.call("toy.read_ingest_backlog") == 12

    async def test_calling_an_unbound_handle_names_the_handle(self) -> None:
        """Registration refuses every handle it can see, so reaching this means an outside caller."""
        world, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="no binding resolves handle 'toy.nope'"):
            await world.call("toy.nope")

    async def test_a_binding_that_raises_keeps_its_type_and_gains_the_handle(self) -> None:
        """Wrapping would erase the type a runner branches on; a bare raise loses the attribution.

        ``ApparatusError`` is an eval concept a host raises from exactly these paths, and the
        difference between a broken rig and a legitimate in-world failure is what the runner
        reads. So the exception passes through and picks up a note instead.
        """

        def explode() -> None:
            raise KeyError("no such document")

        world = WorldRegistry(
            [_sound_dimension(seed="h.seed", read="h.boom")], bindings={**_SOUND_BINDINGS, "h.boom": explode}
        )

        with pytest.raises(KeyError) as caught:
            await world.call("h.boom")

        assert "raised by world handle 'h.boom'" in caught.value.__notes__

    async def test_the_binding_table_property_hands_back_a_copy(self) -> None:
        """Documented as a copy, and now pinned as one.

        A caller composing a derived registry mutates what it is handed; if that reached the
        original's table, one host's composition would rewrite another host's resolution — the
        cross-host leak ``extend`` exists to prevent, through a different door.
        """
        world, _state = toyhost_world()

        world.bindings["toy.read_language"] = lambda: "leaked"

        assert await world.call("toy.read_language") == "en"


class TestComposition:
    """Extend-not-edit, so one host's vocabulary can never reach another's."""

    def test_extend_returns_a_new_registry_and_leaves_the_original_alone(self) -> None:
        """The shared-core rule the input registry set: a host adds to a copy."""
        world, _state = toyhost_world()

        extended = world.extend(
            [_sound_dimension()],
            bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
        )

        assert "shelf_stock" in extended.names
        assert "shelf_stock" not in world.names

    def test_extending_keeps_the_addressing_every_inherited_dimension_is_named_by(self) -> None:
        """A derived registry that lost its host's addressing would read every inherited seed key as another name."""
        base = WorldRegistry(
            [_sound_dimension(name="shelf.stock", carrier="shelf")],
            bindings=_SOUND_BINDINGS,
            address=lambda carrier, key: f"{carrier}.{key}",
        )

        extended = base.extend([_sound_dimension(name="shelf.float", carrier="shelf")])

        assert extended.address("shelf", "float") == "shelf.float"
        assert extended.held_at("shelf.float", {"shelf": {"stock": 1, "float": 2}}) == "float"

    def test_extending_cannot_rebind_a_handle_the_registry_already_binds(self) -> None:
        """The duplicate-name refusal one level down, where re-validation would not catch it.

        A rebound handle still resolves, so the combined registry validates — and conformance
        then seeds and reads through the shadowing binding, reporting PASSED for a dimension
        whose real seeding path nothing exercised.
        """
        world, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="would rebind 'toy.read_language'"):
            world.extend((), bindings={"toy.read_language": lambda: "xx"})

    def test_extending_may_re_supply_the_identical_binding(self) -> None:
        """Nothing is displaced, so nothing can go unexercised — and a host composing several
        carriers over one shared handle would otherwise have to remember which contributed it
        first. The rule is against SHADOWING, and re-supplying the same callable shadows nothing.
        """
        world, _state = toyhost_world()
        same = world.bindings["toy.read_language"]

        extended = world.extend((), bindings={"toy.read_language": same})

        assert extended.names == world.names

    def test_extending_carries_and_protects_the_ambient_perturbation_handle(self) -> None:
        """The optional capability composes under the same rule as the subject view."""
        world, _state = toyhost_world()

        assert world.extend(()).perturb_ambient == "toy.perturb_processing_shift"
        with pytest.raises(WorldRegistrationError, match="would replace perturb_ambient"):
            world.extend((), bindings={"toy.other_shift": lambda: None}, perturb_ambient="toy.other_shift")

    def test_extending_cannot_replace_an_existing_subject_view(self) -> None:
        """Every inherited perception claim was proven against the first one."""
        world, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="would replace subject_view"):
            world.extend((), bindings={"toy.other_view": lambda **_: {}}, subject_view="toy.other_view")

    def test_extending_with_an_unsound_declaration_is_refused(self) -> None:
        """Validation runs over the COMBINED set, not just the addition."""
        world, _state = toyhost_world()

        with pytest.raises(WorldRegistrationError, match="is declared twice"):
            world.extend([_sound_dimension(name="scan_quality", seed=None, read=None, perceived_by=("x",))])


class TestTheBaseWorldAndCoherence:
    """A host whose dimensions are coupled names the world checks compose over, and which worlds it holds.

    Both are refused where they are written when a check could not use them: a base value no check could
    put there is a check that silently starts from somewhere else, and an unbound coherence handle is a
    coupling the kit would never be told about.
    """

    _BOUND = {**_SOUND_BINDINGS, "h.holds": lambda world: []}

    def test_a_sound_base_world_and_coherence_handle_register_and_are_read_back(self) -> None:
        registry = WorldRegistry(
            [_sound_dimension()], bindings=self._BOUND, base_world={"shelf_stock": 3}, coherence="h.holds"
        )

        assert registry.base_world == {"shelf_stock": 3}
        assert registry.coherence == "h.holds"

    def test_the_base_world_read_back_is_a_copy(self) -> None:
        registry = WorldRegistry([_sound_dimension()], bindings=self._BOUND, base_world={"shelf_stock": 3})

        registry.base_world["shelf_stock"] = 9  # type: ignore[index]

        assert registry.base_world == {"shelf_stock": 3}

    @pytest.mark.parametrize(
        ("dimension", "base", "refusal"),
        [
            (_sound_dimension(), {"aisle_count": 1}, "does not declare"),
            (
                _sound_dimension(seed=None, read=None, perceived_by=("aisle",)),
                {"shelf_stock": 1},
                "which nothing here can set",
            ),
            (
                _sound_dimension(when=Triggered(kind="turn", condition="a delivery lands", fire="h.read")),
                {"shelf_stock": 1},
                "which is triggered",
            ),
            (_sound_dimension(), {"shelf_stock": -1}, "below the minimum 0"),
            (_sound_dimension(), {"shelf_stock": "three"}, "not integer"),
        ],
        ids=["undeclared", "not-settable", "triggered", "out-of-bounds", "wrong-type"],
    )
    def test_a_base_world_no_check_could_start_from_is_refused(
        self, dimension: WorldDimension, base: dict[str, object], refusal: str
    ) -> None:
        with pytest.raises(WorldRegistrationError, match=refusal):
            WorldRegistry([dimension], bindings=self._BOUND, subject_view="h.view", base_world=base)

    def test_a_coherence_handle_nothing_binds_is_refused(self) -> None:
        with pytest.raises(WorldRegistrationError, match="coherence names handle 'h.nothing'"):
            WorldRegistry([_sound_dimension()], bindings=self._BOUND, coherence="h.nothing")

    def test_a_coherence_handle_the_engine_cannot_call_with_a_world_is_refused(self) -> None:
        with pytest.raises(WorldRegistrationError, match="coherence handle 'h.blind'"):
            WorldRegistry([_sound_dimension()], bindings={**self._BOUND, "h.blind": lambda: None}, coherence="h.blind")

    def test_extending_carries_both_and_refuses_to_change_either(self) -> None:
        """Every check already proved composed over the first base value and asked the first handle."""
        registry = WorldRegistry(
            [_sound_dimension()], bindings=self._BOUND, base_world={"shelf_stock": 3}, coherence="h.holds"
        )

        extended = registry.extend((), base_world={"shelf_stock": 3})
        assert (extended.base_world, extended.coherence) == ({"shelf_stock": 3}, "h.holds")
        with pytest.raises(WorldRegistrationError, match="would change the base world's 'shelf_stock'"):
            registry.extend((), base_world={"shelf_stock": 4})
        with pytest.raises(WorldRegistrationError, match="would replace coherence"):
            registry.extend((), bindings={"h.other": lambda world: []}, coherence="h.other")


class TestRepresentabilityOnTheProfile:
    """What the authoring gate will read, before any run exists."""

    def test_a_seedable_dimension_is_covered(self) -> None:
        """The map IS the registry, so it cannot drift from the thing it describes."""
        assert toyhost_profile().representable("scan_quality").state == "covered"

    def test_a_witnessed_dimension_is_uncovered_and_says_why_it_is_not_settable(self) -> None:
        """The founding incident, inverted.

        A witnessed dimension is a real declaration and still not something a scenario may
        presume it *set*. Reporting it covered because it appears in the registry would repeat
        the original mistake with a declaration wrapped around it — so the reason distinguishes
        "nobody declared it" from "declared, and no run controls it", because the remedies are
        different.
        """
        answer = toyhost_profile().representable("ingest_backlog")

        assert answer.state == "uncovered"
        assert "ingest_backlog" in answer.reason
        assert "no run can seed it" in answer.reason

    def test_an_undeclared_dimension_is_uncovered_naming_the_dimension(self) -> None:
        """The unit is a precondition, never an area — an author told "this area is unevaluable"
        abandons a probe another mechanism could have answered."""
        answer = toyhost_profile().representable("shelf_stock")

        assert answer.state == "uncovered"
        assert "shelf_stock" in answer.reason

    def test_a_host_with_an_empty_world_is_not_a_host_with_no_world(self) -> None:
        """Registering nothing says "this host seeds nothing", which a run record can expose.

        Having no world at all says the question does not apply. Collapsing the two would turn a
        checkable claim back into a shrug.
        """
        seedless = HostProfile(
            host_id="seedless",
            host_sweepables=toyhost_profile().sweepables,
            measures=toyhost_profile().measures,
            world=WorldRegistry(),
        )

        assert seedless.representable("scan_quality").state == "uncovered"
        assert toyhost_profile(with_world=False).representable("scan_quality").state == "inapplicable"

    def test_a_goal_check_s_state_path_names_nothing_on_a_host_with_no_world(self) -> None:
        """The goal language roots ``state`` at declared dimensions only, so no world means nothing to read.

        Representability stays ``inapplicable`` — the question of seeding does not apply — but a
        check that READS a state path on such a host can never read anything, which is a defect
        the authoring gate refuses rather than a question it waives.
        """
        answer = toyhost_profile(with_world=False).addressable("scan_quality")

        assert answer.state == "uncovered"
        assert "declares none" in answer.reason
        assert toyhost_profile().addressable("scan_quality").state == "covered"

    def test_a_goal_check_over_a_dimension_nothing_reads_back_is_refused(self) -> None:
        """A perceived, unseedable dimension may declare no ``read``; the end state is read through ``read``
        alone, so no cell ever holds it and a check over it could never be established.

        The readable sibling on the same registry is covered, so the refusal is the missing handle and
        not the dimension's quadrant.
        """
        profile = HostProfile(
            host_id="glimpsed",
            host_sweepables=toyhost_profile().sweepables,
            measures=toyhost_profile().measures,
            world=WorldRegistry(
                [
                    WorldDimension(
                        name="tv_channel",
                        carrier="tv",
                        schema={"type": "string"},
                        matters="what the television shows the subject",
                        perceived_by=("screen",),
                    ),
                    WorldDimension(
                        name="tv_volume",
                        carrier="tv",
                        schema={"type": "integer"},
                        matters="how loud the television is",
                        read="tv.volume",
                        perceived_by=("screen",),
                    ),
                ],
                bindings={"tv.volume": lambda: 3, "tv.view": lambda surfaces=None: {"screen": ""}},
                subject_view="tv.view",
            ),
        )

        refused = profile.addressable("tv_channel")

        assert refused.state == "uncovered"
        assert "declares no read handle" in refused.reason
        assert profile.addressable("tv_volume").state == "covered"


class TestTheWorldIsRealRatherThanAShape:
    """Seeding moves the object the subject view renders from, on one path."""

    async def test_seeding_changes_what_the_subject_perceives(self) -> None:
        """Perception A/B in its simplest form.

        If the two were separate paths — a seeder writing one store and a renderer reading
        another — this passes only by coincidence, and the coincidence is what the contract is
        built to remove.
        """
        world, state = toyhost_world()

        await world.call("toy.seed_language", "de")
        rendered = await world.call(world.subject_view or "")

        assert "language=de" in rendered["document_header"]
        assert state.document_language == "de"

    async def test_a_judge_only_dimension_reaches_no_surface(self) -> None:
        """The declaration and the rendering agree because the rendering is the same object."""
        world, _state = toyhost_world()

        await world.call("toy.seed_vendor_template", "globex-2021")
        rendered = await world.call(world.subject_view or "")

        assert await world.call("toy.read_vendor_template") == "globex-2021"
        assert not any("globex-2021" in text for text in rendered.values())

    async def test_a_subject_without_a_surface_perceives_nothing_carried_by_it(self) -> None:
        """Perception is per-subject: what a subject faces depends on what it has attached."""
        world, state = toyhost_world()
        state.ingest_backlog = 7

        rendered = state.render_subject_view(surfaces=("document_header",))

        assert "operator_context" not in rendered
