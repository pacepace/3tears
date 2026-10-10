"""Whether a swept lever took effect, and whether two arms did the same thing — read off what the runs measured.

Two defects of one shape (#577, #576): a comparison's outcome numbers cannot tell a reader why nothing moved,
or whether the arms compared were held alike in what they actually did.

* **A lever that never took effect reads as a clean null.** A lever may declare the measure or covariate it acts
  on (``Sweepable.acts_on``, or ``ActsOn`` on a kind's overlay field), and each coverage row checks it with the
  engine's own separation test: ``moved``, ``inert`` (every pair of levels tested, none separated) or
  ``unchecked`` with the reason. The toy host's ``chunk_tokens`` and its kind's ``page_limit`` act on the
  engine's ``context_tokens_in`` covariate.
* **One reasoning-effort word does not hold reasoning constant across models.** Each arm's mean reasoning share
  (``reasoning_ratio``) is read onto the bundle, and a comparison across candidate models whose shares are at
  least ``REASONING_SHARE_DIVERGENCE`` apart names an ``observed_mechanism`` confound carrying the threshold and
  both values — on the model coverage row, its divergences, and each pairwise model contrast. It qualifies the
  comparison; it suppresses nothing. On any other lever the share moving is what the lever did, not a confound.

Every campaign here is the toy host's, built through its own corpus builders.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

from threetears.evals.analysis import (
    AnalysisContextBundle,
    Confound,
    LeverCoverageInput,
    MechanismCheck,
    assemble_context_bundle,
)
from threetears.evals.analysis.bundle import REASONING_SHARE_DIVERGENCE, observed_mechanism_key
from threetears.evals.contracts import (
    REASONING_RATIO_KEY,
    CampaignDesign,
    ControlDeclaration,
    EvalCampaign,
    EvalResult,
    EvalRun,
    SweptAxis,
    resolve_variant_identity,
)
from threetears.evals.contracts.metrics import MeasurePopulation, MetricDescriptor
from threetears.evals.contracts.host import (
    MeasureRegistry,
    CANDIDATE_MODEL_LEVER,
    SHARED_CORE,
    ActsOn,
    HostProfile,
    Interval,
    KindContract,
    KindContractError,
    MemberActsOn,
    ProfileRegistrationError,
    RegistrationError,
    Sweepable,
    SweepableRegistry,
    SweepableValue,
    freeze,
)
from threetears.evals.contracts.host.sweepables import CORE_ROLES, CORE_SWEEPABLES
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT, ExtractorOverlays
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import (
    TOYHOST_EXTRACTION_FAMILY,
    TOYHOST_MEASURES,
    toyhost_profile,
)
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_ROLES, TOYHOST_SWEEPABLES

_CONTEXT = "context_tokens_in"
_PAGE_LIMIT = "extractor.page_limit"

#: Models the toy extractor could run on. No name means anything to the engine.
_MODEL_A = "extractor-alpha"
_MODEL_B = "extractor-beta"
_MODEL_C = "extractor-gamma"

#: Covariates one observation recorded, given its document and repeat.
Covariates = Callable[[str, int], Mapping[str, float]]


def _constant(values: Mapping[str, float]) -> Covariates:
    """The same covariates on every observation."""
    return lambda _document, _repeat: values


def _per_document(name: str, base: float, step: float) -> Covariates:
    """``name`` at ``base`` plus ``step`` per document index — spread within a level, so a test has noise to read."""
    return lambda document, _repeat: {name: base + step * TOYHOST_DOCUMENTS.index(document)}


@dataclass(frozen=True)
class _Arm:
    """One batch of a test campaign and what its observations recorded."""

    batch: EvalRun
    covariates: Covariates | None = None
    total_ms: float = 900.0
    llm_ms: float | None = None
    documents: tuple[str, ...] = TOYHOST_DOCUMENTS
    keep: Callable[[str, int], bool] | None = None
    alter: Callable[[EvalResult], EvalResult] | None = None


def _profile(*, acts_on: str | None = _CONTEXT, kinds: tuple[KindContract, ...] | None = None) -> HostProfile:
    """The toy host's profile with ``chunk_tokens`` declaring ``acts_on`` (``None`` declares no mechanism)."""
    declarations = tuple(
        replace(declared, acts_on=acts_on) if declared.name == "chunk_tokens" else declared
        for declared in TOYHOST_SWEEPABLES
    )
    base = toyhost_profile()
    return replace(
        base,
        host_sweepables=SHARED_CORE.extend(declarations, roles=TOYHOST_ROLES),
        kinds=base.kinds if kinds is None else kinds,
    )


def _renamed(batch: EvalRun, salt: str, **update: object) -> EvalRun:
    """``batch`` under a fresh id derived from ``salt``, with ``update`` applied."""
    return batch.model_copy(update={"id": str(uuid.uuid5(uuid.UUID(batch.id), salt)), **update})


def _chunk_batch(chunk_tokens: int) -> EvalRun:
    """One toy batch at a chunk width, its rig held at the fixture's defaults."""
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )


def _model_batch(model: str) -> EvalRun:
    """One toy batch on ``model``, every toy-host setting held — the model is the only lever that moves."""
    return _renamed(_chunk_batch(512), model, candidate_model=model)


def _page_batch(pages: int) -> EvalRun:
    """One toy batch whose launch overlaid the kind's ``page_limit`` — the kind's lever is the only one that moves."""
    overlays = freeze(TOY_EXTRACTOR_CONTRACT.validate_overlays({"page_limit": pages}))
    return _renamed(_chunk_batch(512), f"pages-{pages}", overlays=overlays)


def _observations(arm: _Arm, profile: HostProfile) -> list[EvalResult]:
    """The arm's observations, over its documents, each carrying its covariates and latency."""
    results = toyhost_measurements(arm.batch, profile=profile, cost_usd=0.02, total_ms=arm.total_ms, field_accuracy=0.8)
    kept = []
    for result in results:
        if result.test_case_id not in arm.documents:
            continue
        if arm.keep is not None and not arm.keep(result.test_case_id, result.k_iteration):
            continue
        update: dict[str, object] = {}
        if arm.covariates is not None:
            update["covariates"] = dict(arm.covariates(result.test_case_id, result.k_iteration))
        if arm.llm_ms is not None and result.latency is not None:
            update["latency"] = result.latency.model_copy(update={"llm_ms": arm.llm_ms})
        altered = result.model_copy(update=update)
        kept.append(arm.alter(altered) if arm.alter is not None else altered)
    return kept


def _bundle(
    arms: Sequence[_Arm], *, profile: HostProfile | None = None, design: CampaignDesign | None = None
) -> AnalysisContextBundle:
    """Assemble a campaign over ``arms``."""
    host = profile if profile is not None else _profile()
    storage = ToyhostStorage([arm.batch for arm in arms], {arm.batch.id: _observations(arm, host) for arm in arms})
    campaign = EvalCampaign(
        id="5f0c3b9e-7d61-4a2b-9c84-3e1f6a2d7b90",
        scope_id=TOYHOST_SCOPE,
        name="mechanism",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[arm.batch.id for arm in arms],
        declared_design=design,
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=storage, profile=host)


def _row(bundle: AnalysisContextBundle, lever: str) -> LeverCoverageInput:
    (row,) = [entry for entry in bundle.coverage if entry.name == lever]
    return row


def _chunk_sweep(
    narrow: Covariates | None, wide: Covariates | None, *, profile: HostProfile | None = None
) -> AnalysisContextBundle:
    """A chunk-width sweep at 256 and 1024 tokens, each batch recording the given covariates."""
    return _bundle([_Arm(_chunk_batch(256), narrow), _Arm(_chunk_batch(1024), wide)], profile=profile)


def _observed(confounds: Sequence[Confound]) -> list[Confound]:
    return [confound for confound in confounds if confound.kind == "observed_mechanism"]


class TestALeverIsCheckedAgainstTheMechanismItDeclares:
    """#577: a lever whose declared mechanism did not measurably move gives no evidence it took effect."""

    def test_the_declared_campaign_s_uniform_shift_needs_a_range_to_be_called(self) -> None:
        """The toy host's context shifts by one amount on every document, and a token count declares no range: no
        test of the mean can call that, so the check says so and names the remedy, never ``moved`` (#597)."""
        mechanism = _row(toyhost_bundle(), "chunk_tokens").mechanism
        assert mechanism == MechanismCheck(
            state="unchecked",
            measure=_CONTEXT,
            level_means={"256": 2400.0, "1024": 7600.0},
            level_n={"256": 12, "1024": 12},
            reason="uniform_move_needs_range",
        )

    def test_identical_constants_are_the_degenerate_inert(self) -> None:
        mechanism = _row(
            _chunk_sweep(_constant({_CONTEXT: 3000}), _constant({_CONTEXT: 3000})), "chunk_tokens"
        ).mechanism
        assert (mechanism.state, mechanism.reason) == ("inert", None)
        assert mechanism.level_means == {"256": 3000.0, "1024": 3000.0}

    def test_constants_written_alike_are_alike(self) -> None:
        # Three observations of 0.1 sum to 0.30000000000000004 in floats; the levels still do not separate.
        bundle = _chunk_sweep(_constant({_CONTEXT: 0.1}), _constant({_CONTEXT: 0.1}))
        assert _row(bundle, "chunk_tokens").mechanism.state == "inert"

    def test_a_constant_read_at_unequal_repeats_is_inert(self) -> None:
        # Three 0.1s average to 0.10000000000000002 in floats; one 0.1 is 0.1. The case means are exact.
        bundle = _bundle(
            [
                _Arm(_chunk_batch(256), _constant({_CONTEXT: 0.1})),
                _Arm(_chunk_batch(1024), _constant({_CONTEXT: 0.1}), keep=lambda _document, repeat: repeat == 1),
            ]
        )
        assert _row(bundle, "chunk_tokens").mechanism.state == "inert"

    def test_noise_within_overlapping_levels_is_inert_not_moved(self) -> None:
        # Every case differs, and the wide level sits a hair above the narrow one on each: overlapping, never apart.
        narrow = _per_document(_CONTEXT, 3000.0, 40.0)

        def wide(document: str, _repeat: int) -> Mapping[str, float]:
            index = TOYHOST_DOCUMENTS.index(document)
            return {_CONTEXT: 3000.0 + 40.0 * index + (31.0 if index % 2 else -29.0)}

        mechanism = _row(_chunk_sweep(narrow, wide), "chunk_tokens").mechanism
        assert mechanism.level_means["256"] != mechanism.level_means["1024"], (
            "the means must differ, or this is the constant case"
        )
        assert mechanism.state == "inert"

    def test_clearly_separated_levels_are_moved(self) -> None:
        # Wider chunks carry more on every document, by an amount that varies with the document: spread to test.
        mechanism = _row(
            _chunk_sweep(_per_document(_CONTEXT, 3000.0, 40.0), _per_document(_CONTEXT, 9000.0, 95.0)), "chunk_tokens"
        ).mechanism
        assert mechanism.state == "moved"
        assert mechanism.level_n == {"256": 12, "1024": 12}

    def test_a_constant_gap_with_no_spread_and_no_range_is_never_moved(self) -> None:
        mechanism = _row(
            _chunk_sweep(_constant({_CONTEXT: 3000}), _constant({_CONTEXT: 3001})), "chunk_tokens"
        ).mechanism
        assert (mechanism.state, mechanism.reason) == ("unchecked", "uniform_move_needs_range")

    @pytest.mark.parametrize("n_cases", [5, 6, 12])
    def test_a_uniform_shift_with_no_range_is_unchecked_at_any_n(self, n_cases: int) -> None:
        """Every case shifted alike: the exact sign-flip test called six cases ``moved``, but it tests symmetry, not
        the mean, and a token count declares no range for the bounded test to read (#597)."""
        documents = TOYHOST_DOCUMENTS[:n_cases]
        bundle = _bundle(
            [
                _Arm(_chunk_batch(256), _per_document(_CONTEXT, 3000.0, 40.0), documents=documents),
                _Arm(_chunk_batch(1024), _per_document(_CONTEXT, 3001.0, 40.0), documents=documents),
            ]
        )
        mechanism = _row(bundle, "chunk_tokens").mechanism
        assert (mechanism.state, mechanism.reason) == ("unchecked", "uniform_move_needs_range")

    @pytest.mark.parametrize(("n_cases", "state"), [(4, "inert"), (12, "moved")])
    def test_a_uniform_shift_on_a_declared_range_is_read_by_the_bounded_test(self, n_cases: int, state: str) -> None:
        """On a declared range the bounded test reads a shift of one amount: four documents moved +0.75 on a 0-1 rate
        cannot show the mean moved, twelve can (#597). The same shift with no range is never called."""
        documents = TOYHOST_DOCUMENTS[:n_cases]

        def hit_rate(shift: float) -> Callable[[EvalResult], EvalResult]:
            def alter(result: EvalResult) -> EvalResult:
                index = TOYHOST_DOCUMENTS.index(result.test_case_id)
                return result.model_copy(
                    # Rounded so each value is the decimal it reads as: the shift is exactly one amount.
                    update={"host_measures": {**result.host_measures, _HIT_RATE: round(0.1 + 0.01 * index + shift, 6)}}
                )

            return alter

        arms = [
            _Arm(_chunk_batch(256), alter=hit_rate(0.0), documents=documents),
            _Arm(_chunk_batch(1024), alter=hit_rate(0.75), documents=documents),
        ]
        mechanism = _row(_bundle(arms, profile=_hit_rate_profile((0.0, 1.0))), "chunk_tokens").mechanism
        assert (mechanism.state, mechanism.measure) == (state, _HIT_RATE)
        unranged = _row(_bundle(arms, profile=_hit_rate_profile(None)), "chunk_tokens").mechanism
        assert (unranged.state, unranged.reason) == ("unchecked", "uniform_move_needs_range")

    def test_one_case_a_level_is_too_few_to_test(self) -> None:
        bundle = _bundle(
            [
                _Arm(_chunk_batch(256), _per_document(_CONTEXT, 3000.0, 40.0), documents=TOYHOST_DOCUMENTS[:1]),
                _Arm(_chunk_batch(1024), _per_document(_CONTEXT, 3000.0, 40.0), documents=TOYHOST_DOCUMENTS[1:2]),
            ]
        )
        mechanism = _row(bundle, "chunk_tokens").mechanism
        assert (mechanism.state, mechanism.reason) == ("unchecked", "too_few_observations")
        assert mechanism.level_n == {"256": 1, "1024": 1}

    def test_two_levels_agreeing_beside_an_unobserved_one_is_unchecked(self) -> None:
        bundle = _bundle(
            [
                _Arm(_chunk_batch(256), _constant({_CONTEXT: 3000})),
                _Arm(_chunk_batch(512), None),
                _Arm(_chunk_batch(1024), _constant({_CONTEXT: 3000})),
            ]
        )
        mechanism = _row(bundle, "chunk_tokens").mechanism
        assert (mechanism.state, mechanism.reason) == ("unchecked", "levels_unobserved")
        assert mechanism.unobserved_levels == ["512"]

    def test_a_mechanism_observed_at_one_level_only_is_unchecked(self) -> None:
        mechanism = _row(_chunk_sweep(_constant({_CONTEXT: 3000}), None), "chunk_tokens").mechanism
        assert (mechanism.state, mechanism.reason) == ("unchecked", "levels_unobserved")
        assert (mechanism.level_means, mechanism.level_n) == ({"256": 3000.0}, {"256": 12})
        assert mechanism.unobserved_levels == ["1024"]

    def test_a_lever_declaring_no_mechanism_is_unchecked_not_effective(self) -> None:
        bundle = _chunk_sweep(_constant({_CONTEXT: 3000}), _constant({_CONTEXT: 9000}), profile=_profile(acts_on=None))
        assert _row(bundle, "chunk_tokens").mechanism == MechanismCheck(state="unchecked", reason="not_declared")

    def test_the_candidate_model_lever_declares_no_mechanism(self) -> None:
        bundle = _bundle([_Arm(_model_batch(_MODEL_A)), _Arm(_model_batch(_MODEL_B))])
        assert _row(bundle, "model").mechanism.reason == "not_declared"


#: A host-declared, per-result latency measure a lever can act on. Declaring ``scored`` or nothing, a turn's
#: time is read as ``delivered``, so a call the model refused straight away is no observation of it.
_RETRIEVE_MS = "retrieve_ms"


def _retrieve_profile(population: MeasurePopulation | None) -> HostProfile:
    """The toy profile, with ``chunk_tokens`` acting on a host latency measure declaring ``population``."""
    descriptor = MetricDescriptor(
        name=_RETRIEVE_MS,
        reader_name="Retrieval time",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        description="Wall-clock the retriever took for one document.",
        higher_is_better=False,
        unit="ms",
        merit_axis="latency",
        population=population,
    )
    declarations = tuple(
        replace(declared, acts_on=_RETRIEVE_MS) if declared.name == "chunk_tokens" else declared
        for declared in TOYHOST_SWEEPABLES
    )
    return replace(
        toyhost_profile(),
        host_sweepables=SHARED_CORE.extend(declarations, roles=TOYHOST_ROLES),
        measures=MeasureRegistry([*TOYHOST_MEASURES, descriptor], families=(TOYHOST_EXTRACTION_FAMILY,)),
    )


#: A host rate the chunk width acts on in :func:`_hit_rate_profile`.
_HIT_RATE = "retrieval_hit_rate"


def _hit_rate_profile(value_range: tuple[float, float] | None) -> HostProfile:
    """The toy profile, with ``chunk_tokens`` acting on a host rate that declares ``value_range``."""
    descriptor = MetricDescriptor(
        name=_HIT_RATE,
        reader_name="Retrieval hit rate",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="subsystem",
        description="Share of the document's fields the retriever surfaced.",
        higher_is_better=True,
        value_range=value_range,
    )
    declarations = tuple(
        replace(declared, acts_on=_HIT_RATE) if declared.name == "chunk_tokens" else declared
        for declared in TOYHOST_SWEEPABLES
    )
    return replace(
        toyhost_profile(),
        host_sweepables=SHARED_CORE.extend(declarations, roles=TOYHOST_ROLES),
        measures=MeasureRegistry([*TOYHOST_MEASURES, descriptor], families=(TOYHOST_EXTRACTION_FAMILY,)),
    )


def _retrieved(*, refuse: tuple[str, ...] = ()) -> Callable[[EvalResult], EvalResult]:
    """Each observation's retrieve time: 1 s for a turn taken, 50 ms for a call refused on a ``refuse`` document."""

    def alter(result: EvalResult) -> EvalResult:
        refused = result.test_case_id in refuse
        update: dict[str, object] = {
            "host_measures": {**result.host_measures, _RETRIEVE_MS: 50.0 if refused else 1000.0}
        }
        if refused:
            update["candidate_error"] = "the provider refused the request"
        return result.model_copy(update=update)

    return alter


class TestAMechanismOnATurnsTimeIsReadOverTurnsTaken:
    """A lever acting on a latency measure is checked over the turns its levels took, as the cells read it."""

    @pytest.mark.parametrize("population", ["scored", None])
    def test_a_refused_call_s_round_trip_does_not_move_the_mechanism(
        self, population: MeasurePopulation | None
    ) -> None:
        refused = TOYHOST_DOCUMENTS[: len(TOYHOST_DOCUMENTS) // 2]
        bundle = _bundle(
            [
                _Arm(_chunk_batch(256), alter=_retrieved(refuse=refused)),
                _Arm(_chunk_batch(1024), alter=_retrieved()),
            ],
            profile=_retrieve_profile(population),
        )
        mechanism = _row(bundle, "chunk_tokens").mechanism
        # Read over every result, the refusals' 50 ms would pull the narrow level's mean far below 1 s.
        assert mechanism.level_means == {"256": 1000.0, "1024": 1000.0}
        assert mechanism.state == "inert"


class TestAKindOverlayLeverDeclaresItsMechanismToo:
    """A knob a launch turns is a lever like any other, and names what it acts on with ``ActsOn``."""

    def test_the_toy_kind_s_page_limit_is_declared(self) -> None:
        declared = toyhost_profile().sweepables.get(_PAGE_LIMIT)
        assert declared is not None
        assert declared.acts_on == _CONTEXT

    def test_its_coverage_row_is_checked(self) -> None:
        bundle = _bundle(
            [
                _Arm(_page_batch(5), _per_document(_CONTEXT, 2000.0, 40.0)),
                _Arm(_page_batch(20), _per_document(_CONTEXT, 8000.0, 95.0)),
            ]
        )
        mechanism = _row(bundle, _PAGE_LIMIT).mechanism
        assert (mechanism.state, mechanism.measure) == ("moved", _CONTEXT)

    def test_it_moves_no_variant_key(self) -> None:
        class _UndeclaredOverlays(ExtractorOverlays):
            page_limit: Annotated[int, Interval(unit="pages")] = Field(
                10, ge=1, le=50, description="how many pages of a document the extractor reads"
            )

        undeclared = _profile(kinds=(replace(TOY_EXTRACTOR_CONTRACT, overlays=_UndeclaredOverlays),))
        assert undeclared.sweepables.get(_PAGE_LIMIT).acts_on is None  # type: ignore[union-attr]
        batch = _page_batch(20)
        assert (
            resolve_variant_identity(run=batch, profile=undeclared).variant_key
            == resolve_variant_identity(run=batch, profile=_profile()).variant_key
        )

    def test_an_unknown_name_is_refused_at_the_profile(self) -> None:
        class _Misnamed(BaseModel):
            pages: Annotated[int, ActsOn("no_such_measure")] = Field(10, description="pages read")

        with pytest.raises(ProfileRegistrationError, match="no_such_measure"):
            _profile(kinds=(KindContract(TOY_EXTRACTOR_KIND, overlays=_Misnamed),))

    def test_a_map_field_is_refused_where_it_is_declared(self) -> None:
        class _Mapped(BaseModel):
            aliases: Annotated[dict[str, str], ActsOn(_CONTEXT)] = Field(default_factory=dict, description="aliases")

        with pytest.raises(KindContractError, match="ActsOn"):
            KindContract("mapped", overlays=_Mapped)


class _CappedOverlays(ExtractorOverlays):
    """The toy extractor's knobs plus an open map of caps, one of whose entries names its mechanism (#585)."""

    caps: Annotated[dict[str, int], MemberActsOn({"max_pages": _CONTEXT})] = Field(
        default_factory=dict, description="per-tool call caps the extractor honours, by tool"
    )


_MAX_PAGES = "extractor.caps.max_pages"
_RETRIES = "extractor.caps.retries"


def _capped_profile() -> HostProfile:
    return _profile(kinds=(replace(TOY_EXTRACTOR_CONTRACT, overlays=_CappedOverlays),))


def _capped_batch(**caps: int) -> EvalRun:
    """One toy batch whose launch set only these entries of the ``caps`` map."""
    overlays = freeze(_CappedOverlays.model_validate({"caps": caps}))
    salt = "caps-" + "-".join(f"{key}{value}" for key, value in sorted(caps.items()))
    return _renamed(_chunk_batch(512), salt, overlays=overlays)


class TestAMemberOfAnOpenMapDeclaresItsMechanism:
    """#585: a map's entries are distinct knobs, so each names what it acts on — and only the named one is checked."""

    def test_the_named_member_carries_the_mechanism_and_no_other_does(self) -> None:
        registry = _capped_profile().sweepables
        assert registry.acts_on(_MAX_PAGES) == _CONTEXT
        assert registry.acts_on(_RETRIES) is None
        assert registry.acts_on("extractor.caps") is None

    def test_a_sweep_of_the_named_member_is_moved(self) -> None:
        bundle = _bundle(
            [
                _Arm(_capped_batch(max_pages=5), _per_document(_CONTEXT, 2000.0, 40.0)),
                # A gap that varies by document: one amount on every document is never called with no range (#597).
                _Arm(_capped_batch(max_pages=20), _per_document(_CONTEXT, 8000.0, 95.0)),
            ],
            profile=_capped_profile(),
        )
        mechanism = _row(bundle, _MAX_PAGES).mechanism
        assert (mechanism.state, mechanism.measure) == ("moved", _CONTEXT)
        assert mechanism.level_means == {"5": pytest.approx(2220.0), "20": pytest.approx(8522.5)}

    def test_a_cap_the_candidate_never_reaches_is_inert(self) -> None:
        bundle = _bundle(
            [
                _Arm(_capped_batch(max_pages=30), _constant({_CONTEXT: 3000})),
                _Arm(_capped_batch(max_pages=40), _constant({_CONTEXT: 3000})),
            ],
            profile=_capped_profile(),
        )
        assert _row(bundle, _MAX_PAGES).mechanism.state == "inert"

    def test_a_sweep_of_an_unnamed_member_still_reads_not_declared(self) -> None:
        bundle = _bundle(
            [
                _Arm(_capped_batch(retries=1), _per_document(_CONTEXT, 2000.0, 40.0)),
                _Arm(_capped_batch(retries=4), _per_document(_CONTEXT, 8000.0, 40.0)),
            ],
            profile=_capped_profile(),
        )
        assert _row(bundle, _RETRIES).mechanism == MechanismCheck(state="unchecked", reason="not_declared")

    def test_it_moves_no_variant_key(self) -> None:
        class _Undeclared(ExtractorOverlays):
            caps: dict[str, int] = Field(default_factory=dict, description="per-tool call caps the extractor honours")

        undeclared = _profile(kinds=(replace(TOY_EXTRACTOR_CONTRACT, overlays=_Undeclared),))
        batch = _capped_batch(max_pages=20)
        assert (
            resolve_variant_identity(run=batch, profile=undeclared).variant_key
            == resolve_variant_identity(run=batch, profile=_capped_profile()).variant_key
        )

    @pytest.mark.parametrize("measure", ["no_such_measure", "execution_mode", "reasoning_tokens"])
    def test_a_member_mechanism_meets_the_plain_field_s_refusals(self, measure: str) -> None:
        class _Bad(BaseModel):
            caps: Annotated[dict[str, int], MemberActsOn({"max_pages": measure})] = Field(
                default_factory=dict, description="caps"
            )

        with pytest.raises(ProfileRegistrationError, match=measure):
            _profile(kinds=(KindContract(TOY_EXTRACTOR_KIND, overlays=_Bad),))

    def test_it_is_refused_on_a_field_that_is_not_a_map(self) -> None:
        class _Scalar(BaseModel):
            pages: Annotated[int, MemberActsOn({"x": _CONTEXT})] = Field(10, description="pages read")

        with pytest.raises(KindContractError, match="MemberActsOn but is not a map"):
            KindContract("scalar", overlays=_Scalar)

    def test_a_blank_entry_is_refused(self) -> None:
        class _Blank(BaseModel):
            caps: Annotated[dict[str, int], MemberActsOn({"max_pages": " "})] = Field(
                default_factory=dict, description="caps"
            )

        with pytest.raises(KindContractError, match="blank entry"):
            KindContract("blank", overlays=_Blank)

    def test_a_registry_family_refuses_a_member_it_does_not_own(self) -> None:
        with pytest.raises(RegistrationError, match="no member it recognises"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="knobs.*",
                        role="lever",
                        read=lambda _r, _s: {},
                        reader_prose="any knob",
                        open_family="a launch may overlay any knob",
                        owns_member=lambda name: name.startswith("knobs."),
                        member_acts_on={"other.cap": _CONTEXT},
                    )
                ]
            )

    def test_member_acts_on_is_refused_off_an_open_family(self) -> None:
        with pytest.raises(RegistrationError, match="member_acts_on and is not an open family"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="knob",
                        role="lever",
                        read=lambda _r, _s: 1,
                        reader_prose="a knob",
                        member_acts_on={"knob.x": _CONTEXT},
                    )
                ]
            )


class TestAnUnresolvableMechanismIsRefusedWhereItIsDeclared:
    def test_a_name_no_registry_declares_is_refused_at_the_profile(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="no_such_measure"):
            _profile(acts_on="no_such_measure")

    def test_a_non_numeric_measure_is_refused_at_the_profile(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="execution_mode"):
            _profile(acts_on="execution_mode")

    def test_a_per_row_measure_is_refused_with_what_to_declare_instead(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="reasoning_tokens") as refused:
            _profile(acts_on="reasoning_tokens")
        assert REASONING_RATIO_KEY in str(refused.value)

    def test_a_run_level_statistic_is_refused_at_the_profile(self) -> None:
        with pytest.raises(ProfileRegistrationError, match="p95_total_ms"):
            _profile(acts_on="p95_total_ms")

    @pytest.mark.parametrize("name", ["fields_stripped", REASONING_RATIO_KEY, "total_ms", "orchestration_ms"])
    def test_a_per_result_measure_is_accepted(self, name: str) -> None:
        declared = _profile(acts_on=name).sweepables.get("chunk_tokens")
        assert declared is not None
        assert declared.acts_on == name

    def test_a_blank_name_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="blank acts_on"):
            SHARED_CORE.extend(
                [Sweepable(name="knob", role="lever", read=lambda _r, _s: 1, reader_prose="a knob", acts_on=" ")]
            )

    def test_an_apparatus_input_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="only a swept lever"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="rig",
                        role="apparatus",
                        read=lambda _r, _s: 1,
                        reader_prose="the rig",
                        confounds="a different rig measured it",
                        acts_on=_CONTEXT,
                    )
                ]
            )

    def test_an_open_family_is_refused_at_registration(self) -> None:
        with pytest.raises(RegistrationError, match="open family and declares acts_on"):
            SHARED_CORE.extend(
                [
                    Sweepable(
                        name="knobs",
                        role="lever",
                        read=lambda _r, _s: {},
                        reader_prose="any knob",
                        open_family="a launch may overlay any knob",
                        acts_on=_CONTEXT,
                    )
                ]
            )


class TestAMechanismDeclarationIsNoIdentityInput:
    """What to check is not what a run ran under: declaring it moves no variant key."""

    def test_variant_keys_are_the_same_with_and_without_the_declaration(self) -> None:
        batch = _chunk_batch(256)
        declared = resolve_variant_identity(run=batch, profile=_profile())
        undeclared = resolve_variant_identity(run=batch, profile=_profile(acts_on=None))
        assert declared.variant_key == undeclared.variant_key

    def test_the_bundle_pools_the_same_arms_with_and_without_it(self) -> None:
        with_it = _chunk_sweep(_constant({_CONTEXT: 3000}), _constant({_CONTEXT: 9000}))
        without = _chunk_sweep(_constant({_CONTEXT: 3000}), _constant({_CONTEXT: 9000}), profile=_profile(acts_on=None))
        assert [entry.variant_key for entry in with_it.variant_index] == [
            entry.variant_key for entry in without.variant_index
        ]
        assert [(cell.variant_key, cell.apparatus_class_id) for cell in with_it.cells] == [
            (cell.variant_key, cell.apparatus_class_id) for cell in without.cells
        ]


class TestTheMechanismCheckStatesOnlyWhatItsEvidenceDecides:
    @pytest.mark.parametrize(
        "fields",
        [
            {"state": "unchecked"},
            {
                "state": "moved",
                "measure": _CONTEXT,
                "level_means": {"a": 1.0, "b": 2.0},
                "level_n": {"a": 2, "b": 2},
                "reason": "not_swept",
            },
            {"state": "moved", "measure": None, "level_means": {"a": 1.0, "b": 2.0}, "level_n": {"a": 2, "b": 2}},
            {"state": "unchecked", "measure": None, "reason": "levels_unobserved"},
            {"state": "moved", "measure": _CONTEXT, "level_means": {"a": 1.0}, "level_n": {"a": 2}},
            {
                "state": "inert",
                "measure": _CONTEXT,
                "level_means": {"a": 1.0, "b": 1.0},
                "level_n": {"a": 2, "b": 2},
                "unobserved_levels": ["c"],
            },
            {"state": "moved", "measure": _CONTEXT, "level_means": {"a": 1.0, "b": 2.0}, "level_n": {"a": 2}},
        ],
    )
    def test_a_contradictory_check_is_refused(self, fields: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            MechanismCheck.model_validate(fields)


def _model_comparison(
    *shares: float | None, design: CampaignDesign | None = None, profile: HostProfile | None = None
) -> AnalysisContextBundle:
    models = (_MODEL_A, _MODEL_B, _MODEL_C)[: len(shares)]
    return _bundle(
        [
            _Arm(_model_batch(model), None if share is None else _constant({REASONING_RATIO_KEY: share}))
            for model, share in zip(models, shares, strict=True)
        ],
        design=design,
        profile=profile,
    )


class TestAModelComparisonAtOneEffortDisclosesADivergentReasoningShare:
    """#576: two model arms at one reasoning-effort setting can reason very differently."""

    def test_very_different_shares_name_the_confound_with_threshold_and_values(self) -> None:
        bundle = _model_comparison(0.71, 0.30)
        assert _observed(_row(bundle, "model").confounded_by) == [
            Confound(
                dimension=observed_mechanism_key(REASONING_RATIO_KEY),
                kind="observed_mechanism",
                level_values={_MODEL_A: 0.71, _MODEL_B: 0.30},
                threshold=REASONING_SHARE_DIVERGENCE,
            )
        ]
        assert REASONING_SHARE_DIVERGENCE == 0.20
        assert "reasoning" in bundle.confound_catalog[observed_mechanism_key(REASONING_RATIO_KEY)]

    def test_the_contrast_is_qualified_not_suppressed(self) -> None:
        row = _row(_model_comparison(0.71, 0.30), "model")
        assert row.levels == sorted([_MODEL_A, _MODEL_B])
        assert row.n == 72

    def test_shares_exactly_the_threshold_apart_are_named(self) -> None:
        # 0.7 - 0.5 is 0.19999999999999996 in floats; as written it is 0.2, which is "at least 0.2".
        assert len(_observed(_row(_model_comparison(0.7, 0.5), "model").confounded_by)) == 1

    def test_matched_shares_name_no_confound(self) -> None:
        assert _observed(_row(_model_comparison(0.40, 0.45), "model").confounded_by) == []

    def test_each_arm_carries_its_share(self) -> None:
        bundle = _model_comparison(0.71, 0.30)
        assert {reading.mean for reading in bundle.arm_mechanisms} == {0.71, 0.30}
        assert all(
            (reading.covariate, reading.n_measured, reading.n_results) == (REASONING_RATIO_KEY, 36, 36)
            for reading in bundle.arm_mechanisms
        )

    def test_a_share_never_measured_names_no_confound_and_says_it_is_unmeasured(self) -> None:
        bundle = _model_comparison(None, None)
        assert _observed(_row(bundle, "model").confounded_by) == []
        assert [(r.mean, r.n_measured, r.n_results) for r in bundle.arm_mechanisms] == [(None, 0, 36), (None, 0, 36)]

    def test_one_arm_unmeasured_is_compared_with_nothing(self) -> None:
        bundle = _model_comparison(0.95, None)
        assert _observed(_row(bundle, "model").confounded_by) == []
        assert sorted((r.mean is None, r.n_measured) for r in bundle.arm_mechanisms) == [(False, 36), (True, 0)]

    def test_the_declared_campaign_measured_no_share(self) -> None:
        campaign, _storage = toyhost_campaign()
        bundle = toyhost_bundle()
        assert len(bundle.arm_mechanisms) == len(campaign.run_ids)
        assert all(reading.mean is None for reading in bundle.arm_mechanisms)


def _model_declares_reasoning_profile() -> HostProfile:
    """The toy profile, its registry re-declaring the core's model lever as acting on the reasoning share."""
    core = tuple(
        replace(declared, acts_on=REASONING_RATIO_KEY) if declared.name == CANDIDATE_MODEL_LEVER else declared
        for declared in CORE_SWEEPABLES
    )
    registry = SweepableRegistry(core, roles=CORE_ROLES).extend(TOYHOST_SWEEPABLES, roles=TOYHOST_ROLES)
    return replace(toyhost_profile(), host_sweepables=registry)


class TestALeverSMediatedEffectIsNotItsConfound:
    """On any lever but the model, the share moving is what the lever did — never a rival explanation of it."""

    def test_a_lever_acting_on_the_share_moves_it_and_is_not_confounded_by_it(self) -> None:
        bundle = _chunk_sweep(
            _per_document(REASONING_RATIO_KEY, 0.20, 0.01),
            _per_document(REASONING_RATIO_KEY, 0.70, 0.01),
            profile=_profile(acts_on=REASONING_RATIO_KEY),
        )
        row = _row(bundle, "chunk_tokens")
        assert row.mechanism.state == "moved"
        assert _observed(row.confounded_by) == []

    def test_a_model_lever_declaring_the_share_as_its_mechanism_is_not_confounded_by_it(self) -> None:
        profile = _model_declares_reasoning_profile()
        bundle = _bundle(
            [
                _Arm(_model_batch(_MODEL_A), _per_document(REASONING_RATIO_KEY, 0.20, 0.01)),
                _Arm(_model_batch(_MODEL_B), _per_document(REASONING_RATIO_KEY, 0.70, 0.01)),
            ],
            profile=profile,
        )
        row = _row(bundle, CANDIDATE_MODEL_LEVER)
        assert row.mechanism.state == "moved"
        assert _observed(row.confounded_by) == []

    def test_the_pairwise_contrasts_follow_the_same_rule(self) -> None:
        bundle = _model_comparison(
            0.50, 0.40, 0.25, design=_three_model_design(), profile=_model_declares_reasoning_profile()
        )
        comparisons = [c for family in bundle.multiple_comparisons.families for c in family.comparisons]
        assert comparisons, "the fixture must produce comparisons, or this asserts nothing"
        assert len(bundle.design.contrasts) == 2
        assert all(arm.mechanism_confounds == [] for arm in bundle.design.contrasts)
        assert all(comparison.mechanism_confounds == [] for comparison in comparisons)

    def test_a_lever_declaring_no_mechanism_is_not_confounded_by_the_share_either(self) -> None:
        bundle = _chunk_sweep(_constant({REASONING_RATIO_KEY: 0.2}), _constant({REASONING_RATIO_KEY: 0.7}))
        assert _observed(_row(bundle, "chunk_tokens").confounded_by) == []


def _three_model_design() -> CampaignDesign:
    control = resolve_variant_identity(run=_model_batch(_MODEL_A), profile=_profile()).variant_key
    return CampaignDesign(
        axes=[
            SweptAxis(
                axis_id="model",
                values=[SweepableValue.of(model, display=model) for model in (_MODEL_A, _MODEL_B, _MODEL_C)],
                rationale="which extractor model",
            )
        ],
        control=control,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="witnessed"),
        declared_at="2026-03-14T09:30:00+00:00",
    )


class TestEachPairwiseModelContrastNamesItsOwnDivergence:
    """With three models, the qualification sits on the pair that diverged, not only on the row."""

    def _bundle(self) -> AnalysisContextBundle:
        # A-B 0.10 apart, A-C 0.25 apart, B-C 0.15 apart: only the control against C diverges.
        return _model_comparison(0.50, 0.40, 0.25, design=_three_model_design())

    def test_the_design_contrasts_name_only_the_diverged_pair(self) -> None:
        design = self._bundle().design
        assert design.control_arm is not None
        by_model = {
            next(iter(arm.moved.values())): [confound.level_values for confound in arm.mechanism_confounds]
            for arm in design.contrasts
        }
        assert by_model == {_MODEL_B: [], _MODEL_C: [{_MODEL_A: 0.50, _MODEL_C: 0.25}]}

    def test_every_family_comparison_of_the_diverged_pair_names_it(self) -> None:
        bundle = self._bundle()
        variant_c = resolve_variant_identity(run=_model_batch(_MODEL_C), profile=_profile()).variant_key
        comparisons = [c for family in bundle.multiple_comparisons.families for c in family.comparisons]
        assert comparisons, "the fixture must produce comparisons, or this asserts nothing"
        for comparison in comparisons:
            expected = [{_MODEL_A: 0.50, _MODEL_C: 0.25}] if comparison.contrast.variant_key == variant_c else []
            assert [confound.level_values for confound in comparison.mechanism_confounds] == expected
        assert {c.contrast.variant_key for c in comparisons if c.mechanism_confounds} == {variant_c}


def _divergent_model_comparison(share_a: float, share_b: float) -> AnalysisContextBundle:
    """Two model arms whose whole-run time moved while the time inside model calls held — a scope divergence."""
    return _bundle(
        [
            _Arm(_model_batch(_MODEL_A), _constant({REASONING_RATIO_KEY: share_a}), total_ms=900.0, llm_ms=500.0),
            _Arm(_model_batch(_MODEL_B), _constant({REASONING_RATIO_KEY: share_b}), total_ms=2000.0, llm_ms=500.0),
        ]
    )


class TestAScopeDivergenceNamesTheSameConfound:
    """The divergence lens compares the same two levels, so it must name what the coverage row names."""

    def test_a_divergent_share_is_named_on_the_divergence(self) -> None:
        divergences = [d for d in _divergent_model_comparison(0.71, 0.30).scope_divergences if d.lever == "model"]
        assert divergences, "the fixture must produce a divergence, or this asserts nothing"
        for divergence in divergences:
            assert [c.level_values for c in _observed(divergence.confounded_by)] == [{_MODEL_A: 0.71, _MODEL_B: 0.30}]

    def test_a_matched_share_names_nothing_there(self) -> None:
        divergences = [d for d in _divergent_model_comparison(0.40, 0.45).scope_divergences if d.lever == "model"]
        assert divergences, "the fixture must produce a divergence, or this asserts nothing"
        assert all(_observed(d.confounded_by) == [] for d in divergences)


class TestAnObservedMechanismConfoundCarriesItsEvidence:
    def test_one_without_a_threshold_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="observed:x", kind="observed_mechanism", level_values={"a": 0.1, "b": 0.9})

    def test_one_with_a_single_level_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="observed:x", kind="observed_mechanism", level_values={"a": 0.1}, threshold=0.2)

    def test_an_undecided_one_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            Confound(
                dimension="observed:x",
                kind="observed_mechanism",
                status="undecided",
                level_values={"a": 0.1, "b": 0.9},
                threshold=0.2,
            )

    @pytest.mark.parametrize("kind", ["swept_lever", "apparatus"])
    def test_another_kind_carries_no_evidence_fields(self, kind: str) -> None:
        with pytest.raises(ValidationError):
            Confound(dimension="x", kind=kind, threshold=0.2)  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            Confound(dimension="x", kind=kind, level_values={"a": 1.0, "b": 2.0})  # type: ignore[arg-type]


class TestALevelSValueIsDerivedOnce:
    """A level's value is the mean of its per-case means, wherever the bundle states it."""

    @staticmethod
    def _arms() -> list[_Arm]:
        # Model A's share climbs with the document, and its first six documents ran once, the rest three times:
        # the mean of its per-case means (0.475) is not the mean over its results.
        return [
            _Arm(
                _model_batch(_MODEL_A),
                _per_document(REASONING_RATIO_KEY, 0.20, 0.05),
                keep=lambda document, repeat: TOYHOST_DOCUMENTS.index(document) >= 6 or repeat == 1,
            ),
            _Arm(_model_batch(_MODEL_B), _constant({REASONING_RATIO_KEY: 0.90})),
        ]

    def test_the_confound_and_the_arm_reading_state_the_same_value(self) -> None:
        bundle = _bundle(self._arms())
        (confound,) = _observed(_row(bundle, CANDIDATE_MODEL_LEVER).confounded_by)
        variant_a = resolve_variant_identity(run=_model_batch(_MODEL_A), profile=_profile()).variant_key
        (reading,) = [r for r in bundle.arm_mechanisms if r.variant_key == variant_a]
        over_results = (sum(0.20 + 0.05 * i for i in range(6)) + 3 * sum(0.20 + 0.05 * i for i in range(6, 12))) / 24
        assert confound.level_values[_MODEL_A] == reading.mean
        assert reading.mean == pytest.approx(0.475)
        assert over_results != pytest.approx(0.475), "the repeats must be unequal enough to tell the two apart"

    def test_the_mechanism_check_states_it_too(self) -> None:
        declared = _bundle(self._arms(), profile=_model_declares_reasoning_profile())
        undeclared = _bundle(self._arms())
        (confound,) = _observed(_row(undeclared, CANDIDATE_MODEL_LEVER).confounded_by)
        level_mean = _row(declared, CANDIDATE_MODEL_LEVER).mechanism.level_means[_MODEL_A]
        assert level_mean == confound.level_values[_MODEL_A]
        assert level_mean == pytest.approx(0.475)
