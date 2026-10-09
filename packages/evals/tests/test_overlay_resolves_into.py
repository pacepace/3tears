"""One overlay knob and the host surface it is written into read as one lever, not two that always move together.

The defect, from a host's shakedown: a kind's overlay field (``reasoning_effort``) and the host's own lever for
what that knob resolved into (``llm_parameters``, a content hash of the resolved model parameters) move in
lockstep, so every effort sweep reported both as moved and each as confounded by the other — the knob could never
be read on its own. The same pairing holds for every overlay whose value the host also records resolved.

``ResolvesInto(lever)`` on the overlay field names the surface. Within the runs a lens compares, the surface folds
into the knob wherever it is constant within each of the knob's levels — it moved only where the knob did. Where it
differs between runs that held the knob at one level, something else wrote into it, and it stays a lever and a
confound of its own; a run that did not record the surface folds nothing. A map field's marker rides on the map's
own lever, whose level is exactly the members a launch set.

Every campaign here is the toy host's, with its extractor kind re-contracted to carry the knob, and is read through
:func:`~threetears.evals.analysis.assemble_context_bundle` — the entry point every lens is reached through.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals.analysis import AnalysisContextBundle, Confound, LeverCoverageInput, assemble_context_bundle
from threetears.evals.contracts import (
    CampaignDesign,
    ControlDeclaration,
    EvalCampaign,
    EvalRun,
    SweptAxis,
    resolve_variant_identity,
)
from threetears.evals.contracts.host import (
    SHARED_CORE,
    ActsOn,
    HostProfile,
    KindContract,
    KindContractError,
    Ordinal,
    RegistrationError,
    ResolvesInto,
    Sweepable,
    SweepableValue,
    freeze,
)
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_SWEEPABLE_REGISTRY
from packages.evals.tests.fixtures.toyhost.variant import variant_levers

if TYPE_CHECKING:
    from threetears.evals.contracts import EvalResult

#: The host's lever for the resolved model parameters, and the overlay knob written into it.
_PARAMS = "llm_parameters"
_EFFORT = "extractor.reasoning_effort"

#: The host's lever for the resolved tool configuration, and the overlay map written into it.
_TOOLS = "resolved_tool_configs"
_TOOL_MAP = "extractor.tool_configs"
_TOOL_MEMBER = "extractor.tool_configs.search"

#: Where the toy host keeps its own keys on a run.
_NAMESPACE = "toyhost"

#: A fixed namespace for the ids this module derives, so a run's id is a function of what it is.
_IDS = uuid.UUID("0b8f6a3e-5c21-4d7a-9e1f-2a6c4b8d0e13")


def _recorded(key: str) -> Any:
    """A reader of the content hash of one toy-host key, ``None`` where the run recorded none."""

    def read(run: EvalRun, _results: Sequence[EvalResult]) -> str | None:
        value = (run.host_payload or {}).get(_NAMESPACE, {}).get(key)
        return None if value is None else SweepableValue.of(value, display=key).content_hash

    return read


#: The host's two resolved surfaces: fixed levers of its own, levelled by content.
_SURFACES = (
    Sweepable(
        name=_PARAMS,
        role="lever",
        read=_recorded(_PARAMS),
        reader_prose="the model parameters the extractor resolved, after any overlaid knob",
    ),
    Sweepable(
        name=_TOOLS,
        role="lever",
        read=_recorded(_TOOLS),
        reader_prose="the tool configuration the extractor resolved, after any overlaid entry",
    ),
)


def _variant_levers(run: EvalRun) -> dict[str, SweepableValue]:
    """The toy host's own levels, plus each resolved surface's — the coordinates the host writes."""
    payload = (run.host_payload or {}).get(_NAMESPACE, {})
    return {
        **variant_levers(run),
        **{name: SweepableValue.of(payload.get(name), display=name) for name in (_PARAMS, _TOOLS)},
    }


class _Overlays(BaseModel):
    """The extractor's knobs, each written into a surface the host records resolved."""

    reasoning_effort: Annotated[Literal["low", "high"], Ordinal(), ResolvesInto(_PARAMS)] = Field(
        "low", description="how hard the extractor is told to reason"
    )
    tool_configs: Annotated[dict[str, str], ResolvesInto(_TOOLS)] = Field(
        default_factory=dict, description="the per-tool settings the launch overlaid"
    )


def _contract(overlays: type[BaseModel] = _Overlays) -> KindContract:
    """The toy extractor kind, re-contracted to carry ``overlays``."""
    return KindContract(TOY_EXTRACTOR_KIND, overlays=overlays, prefix="extractor", seats=TOY_EXTRACTOR_CONTRACT.seats)


def _profile(overlays: type[BaseModel] = _Overlays, *, surfaces: Sequence[Sweepable] = _SURFACES) -> HostProfile:
    """The toy host, registering the resolved surfaces and its extractor kind carrying ``overlays``."""
    return replace(
        toyhost_profile(),
        host_sweepables=TOYHOST_SWEEPABLE_REGISTRY.extend(surfaces),
        variant_levers=_variant_levers,
        kinds=(_contract(overlays),),
    )


def _batch(
    effort: str, params: dict[str, Any] | None, *, tools: dict[str, str] | None = None, resolved_tools: Any = "default"
) -> EvalRun:
    """One toy batch at a reasoning effort, recording the parameters it resolved (``None`` records none)."""
    recorded: dict[str, Any] = {_TOOLS: resolved_tools}
    if params is not None:
        recorded[_PARAMS] = params
    batch = toyhost_batch(
        chunk_tokens=512,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        **recorded,
    )
    overlays = freeze(_contract().validate_overlays({"reasoning_effort": effort, "tool_configs": tools or {}}))
    salt = f"{effort}|{params}|{tools}|{resolved_tools}"
    return batch.model_copy(update={"id": str(uuid.uuid5(_IDS, salt)), "overlays": overlays})


def _bundle(batches: Sequence[EvalRun], *, control: EvalRun | None = None) -> AnalysisContextBundle:
    """Assemble a campaign over ``batches``, with ``control`` declared as its control arm when given."""
    profile = _profile()
    results = {
        batch.id: toyhost_measurements(
            batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.7 + 0.05 * index
        )
        for index, batch in enumerate(batches)
    }
    design = None
    if control is not None:
        design = CampaignDesign(
            axes=[
                SweptAxis(
                    axis_id=_EFFORT,
                    values=[SweepableValue.of(level, display=level) for level in ("low", "high")],
                    rationale="how hard the extractor reasons",
                )
            ],
            control=resolve_variant_identity(run=control, profile=profile).variant_key,
            controls=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
            declared_at="2026-03-14T09:30:00+00:00",
        )
    campaign = EvalCampaign(
        id="7c1d2e3f-4a5b-4c6d-8e7f-9a0b1c2d3e4f",
        scope_id=TOYHOST_SCOPE,
        name="effort",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        declared_design=design,
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(batches, results), profile=profile)


def _row(bundle: AnalysisContextBundle, lever: str) -> LeverCoverageInput:
    (row,) = [entry for entry in bundle.coverage if entry.name == lever]
    return row


def _confound(row: LeverCoverageInput, dimension: str) -> Confound | None:
    return next((confound for confound in row.confounded_by if confound.dimension == dimension), None)


_LOW = {"max_output_tokens": 4096, "reasoning_effort": "low"}
_HIGH = {"max_output_tokens": 4096, "reasoning_effort": "high"}
#: The high setting resolved with a different token cap — something besides the effort knob wrote into it.
_HIGH_RECAPPED = {"max_output_tokens": 8192, "reasoning_effort": "high"}


class TestASurfaceThatMovedOnlyWithItsKnobIsOneLever:
    """The effort sweep the defect was found on: the parameters move 1:1 with the knob."""

    def _sweep(self) -> AnalysisContextBundle:
        control = _batch("low", _LOW)
        return _bundle([control, _batch("high", _HIGH)], control=control)

    def test_the_surface_earns_no_coverage_row_of_its_own(self) -> None:
        bundle = self._sweep()
        assert _EFFORT in {row.name for row in bundle.coverage}
        assert _PARAMS not in {row.name for row in bundle.coverage}

    def test_the_knob_is_not_confounded_by_its_own_surface(self) -> None:
        bundle = self._sweep()
        assert _confound(_row(bundle, _EFFORT), _PARAMS) is None
        assert _PARAMS not in bundle.confound_catalog

    def test_the_contrast_moved_one_lever(self) -> None:
        design = self._sweep().design
        assert [set(arm.moved) for arm in design.contrasts] == [{_EFFORT}]
        assert design.shape == "one_factor_at_a_time"

    def test_each_arm_is_named_by_the_knob_and_keeps_the_surface_in_its_key(self) -> None:
        entries = self._sweep().variant_index
        assert len(entries) == 2
        for entry in entries:
            assert _PARAMS in entry.folded
            assert _PARAMS in entry.levers, "the key is digested from the surface, so it stays in the pre-image"
            assert _PARAMS not in entry.named_levers
            assert _EFFORT in entry.named_levers
            assert entry.swept == {}, "the knob names the arm from `levers`; nothing is added beside it"

    def test_a_repeat_of_one_level_with_the_same_surface_still_folds(self) -> None:
        """Two runs at one level that resolved alike are the functional dependency holding, not breaking."""
        control = _batch("low", _LOW)
        repeat = control.model_copy(update={"id": str(uuid.uuid5(_IDS, "repeat"))})
        bundle = _bundle([control, repeat, _batch("high", _HIGH)])
        assert _PARAMS not in {row.name for row in bundle.coverage}
        assert _confound(_row(bundle, _EFFORT), _PARAMS) is None


class TestASurfaceThatMovedWithoutItsKnobStaysAConfound:
    """The same sweep, with the high setting run twice and resolving to two different parameter sets."""

    def _sweep(self) -> AnalysisContextBundle:
        control = _batch("low", _LOW)
        return _bundle([control, _batch("high", _HIGH), _batch("high", _HIGH_RECAPPED)], control=control)

    def test_the_knob_s_row_names_the_surface_as_a_confound(self) -> None:
        bundle = self._sweep()
        confound = _confound(_row(bundle, _EFFORT), _PARAMS)
        assert confound is not None
        assert (confound.kind, confound.status) == ("swept_lever", "varied")

    def test_the_catalog_says_the_surface_moved_where_the_knob_held(self) -> None:
        reason = self._sweep().confound_catalog[_PARAMS]
        assert f"the resolved surface that {_EFFORT} is written into" in reason
        assert f"held {_EFFORT} at one level" in reason
        assert "the model parameters the extractor resolved" in reason

    def test_the_surface_keeps_its_own_row(self) -> None:
        assert _PARAMS in {row.name for row in self._sweep().coverage}

    def test_no_arm_is_named_as_if_the_surface_had_folded(self) -> None:
        assert all(_PARAMS not in entry.folded for entry in self._sweep().variant_index)


class TestASurfaceARunDidNotRecordFoldsNothing:
    """A run that predates the host's capture of the surface cannot show agreement."""

    def _sweep(self) -> AnalysisContextBundle:
        control = _batch("low", None)
        return _bundle([control, _batch("high", _HIGH)], control=control)

    def test_the_surface_is_an_undecided_confound_on_the_knob(self) -> None:
        confound = _confound(_row(self._sweep(), _EFFORT), _PARAMS)
        assert confound is not None
        assert (confound.kind, confound.status) == ("swept_lever", "undecided")

    def test_the_contrast_moved_both(self) -> None:
        design = self._sweep().design
        assert [set(arm.moved) for arm in design.contrasts] == [{_EFFORT, _PARAMS}]
        assert design.shape == "multi_factor"

    def test_no_arm_is_named_as_if_the_surface_had_folded(self) -> None:
        assert all(_PARAMS not in entry.folded for entry in self._sweep().variant_index)


class TestAMapKnobFoldsItsSurfaceThroughItsMembers:
    """A map overlay: the member a launch set, the map's own lever, and the host surface it is written into."""

    def _sweep(self, *, drifted: bool = False) -> AnalysisContextBundle:
        control = _batch("low", _LOW, resolved_tools={"search": "shallow"})
        deep = _batch("low", _LOW, tools={"search": "deep"}, resolved_tools={"search": "deep"})
        batches = [control, deep]
        if drifted:
            # The control's map again, resolving to another configuration: something else wrote into it.
            batches.append(_batch("low", _LOW, resolved_tools={"search": "shallow", "timeout_s": "30"}))
        return _bundle(batches)

    def test_only_the_member_is_a_lever(self) -> None:
        bundle = self._sweep()
        rows = {row.name for row in bundle.coverage}
        assert _TOOL_MEMBER in rows
        assert {_TOOL_MAP, _TOOLS}.isdisjoint(rows)
        assert {confound.dimension for confound in _row(bundle, _TOOL_MEMBER).confounded_by}.isdisjoint(
            {_TOOL_MAP, _TOOLS}
        )

    def test_each_arm_is_named_by_the_member(self) -> None:
        for entry in self._sweep().variant_index:
            assert {_TOOL_MAP, _TOOLS} <= set(entry.folded)
            assert {_TOOL_MAP, _TOOLS}.isdisjoint(entry.named_levers)

    def test_a_surface_that_moved_while_the_map_held_stays_a_confound(self) -> None:
        bundle = self._sweep(drifted=True)
        confound = _confound(_row(bundle, _TOOL_MEMBER), _TOOLS)
        assert confound is not None
        assert confound.status == "varied"
        assert _TOOLS in {row.name for row in bundle.coverage}


# --- registration -----------------------------------------------------------------------------------


def _knob(marker: object) -> type[BaseModel]:
    """An overlay model whose one knob carries ``marker``."""

    class _One(BaseModel):
        reasoning_effort: Annotated[Literal["low", "high"], marker] = Field("low", description="how hard to reason")

    return _One


class TestAResolvesIntoNoLensCouldFoldIsRefusedWhereItIsDeclared:
    def test_the_marker_reaches_the_knob_s_declaration(self) -> None:
        declared = _profile().sweepables.get(_EFFORT)
        assert declared is not None
        assert declared.resolves_into == _PARAMS
        assert _profile().sweepables.resolution_surfaces[_PARAMS].name == _EFFORT

    def test_it_combines_with_the_other_markers(self) -> None:
        class _Marked(BaseModel):
            reasoning_effort: Annotated[
                Literal["low", "high"], Ordinal(), ActsOn("reasoning_ratio"), ResolvesInto(_PARAMS)
            ] = Field("low", description="how hard to reason")

        declared = _profile(_Marked).sweepables.get(_EFFORT)
        assert declared is not None
        assert (declared.acts_on, declared.resolves_into) == ("reasoning_ratio", _PARAMS)

    def test_it_moves_no_variant_key(self) -> None:
        batch = _batch("high", _HIGH)
        unmarked = _profile(_knob(Ordinal()))
        marked = _profile(_knob(ResolvesInto(_PARAMS)))
        assert (
            resolve_variant_identity(run=batch, profile=marked).variant_key
            == resolve_variant_identity(run=batch, profile=unmarked).variant_key
        )

    def test_an_undeclared_surface_is_refused(self) -> None:
        with pytest.raises(RegistrationError, match="'no_such_lever', which is not declared"):
            _profile(_knob(ResolvesInto("no_such_lever")))

    def test_a_surface_that_is_not_a_lever_is_refused(self) -> None:
        with pytest.raises(RegistrationError, match="'ocr_engine_version', which is not a fixed lever"):
            _profile(_knob(ResolvesInto("ocr_engine_version")))

    def test_an_open_family_as_the_surface_is_refused(self) -> None:
        family = Sweepable(
            name="tool_overrides",
            role="lever",
            read=lambda _run, _results: {},
            reader_prose="any tool setting the launch overlaid",
            open_family="a launch may overlay any tool's settings",
        )
        with pytest.raises(RegistrationError, match="'tool_overrides', which is not a fixed lever"):
            _profile(_knob(ResolvesInto("tool_overrides")), surfaces=(*_SURFACES, family))

    def test_two_knobs_written_into_one_surface_are_refused(self) -> None:
        class _Two(BaseModel):
            reasoning_effort: Annotated[Literal["low", "high"], ResolvesInto(_PARAMS)] = Field(
                "low", description="how hard to reason"
            )
            max_tokens: Annotated[int, ResolvesInto(_PARAMS)] = Field(4096, description="the output token cap")

        with pytest.raises(RegistrationError, match=f"'{_PARAMS}' is named as the resolved surface of both"):
            _profile(_Two)

    def test_the_marker_twice_on_one_field_is_refused(self) -> None:
        class _Twice(BaseModel):
            reasoning_effort: Annotated[Literal["low", "high"], ResolvesInto(_PARAMS), ResolvesInto(_TOOLS)] = Field(
                "low", description="how hard to reason"
            )

        with pytest.raises(KindContractError, match="ResolvesInto more than once"):
            _contract(_Twice)


class TestAFixedLeverDeclaringASurfaceIsCheckedAtRegistration:
    """The registry rules a kind's marker lands on, stated for a host's own declarations."""

    @staticmethod
    def _lever(name: str, **declared: Any) -> Sweepable:
        return Sweepable(name=name, role="lever", read=lambda _run, _results: None, reader_prose=name, **declared)

    def test_a_fixed_lever_naming_a_fixed_lever_is_admitted(self) -> None:
        registry = SHARED_CORE.extend([self._lever("surface"), self._lever("knob", resolves_into="surface")])
        assert registry.resolution_surfaces["surface"].name == "knob"

    def test_apparatus_naming_a_surface_is_refused(self) -> None:
        rig = Sweepable(
            name="rig",
            role="apparatus",
            read=lambda _run, _results: None,
            reader_prose="the rig",
            confounds="a different rig measured it",
            resolves_into="surface",
        )
        with pytest.raises(RegistrationError, match="'rig' is a apparatus and declares resolves_into"):
            SHARED_CORE.extend([self._lever("surface"), rig])

    def test_a_residual_reader_on_a_fixed_lever_is_refused(self) -> None:
        knob = self._lever("knob", resolves_into="surface", read_residual=lambda _run, _results, _removed: {})
        with pytest.raises(RegistrationError, match="'knob' declares read_residual and is not an open family"):
            SHARED_CORE.extend([self._lever("surface"), knob])

    def test_a_knob_with_no_coordinate_of_its_own_is_refused(self) -> None:
        knob = self._lever("knob", resolves_into="surface", no_own_coordinate="carried by the surface")
        with pytest.raises(RegistrationError, match="declares resolves_into and no_own_coordinate"):
            SHARED_CORE.extend([self._lever("surface"), knob])

    def test_a_chain_or_cycle_of_fixed_levers_is_refused(self) -> None:
        with pytest.raises(RegistrationError, match="which itself resolves into"):
            SHARED_CORE.extend([self._lever("a", resolves_into="b"), self._lever("b", resolves_into="a")])

    def test_an_open_family_still_needs_its_residual_reader(self) -> None:
        family = Sweepable(
            name="knobs",
            role="lever",
            read=lambda _run, _results: {},
            reader_prose="any knob",
            open_family="a launch may overlay any knob",
            resolves_into="surface",
        )
        with pytest.raises(RegistrationError, match="one of resolves_into / read_residual without the other"):
            SHARED_CORE.extend([self._lever("surface"), family])
