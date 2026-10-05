"""A kind that does not HAVE an apparatus dimension is not a kind that failed to record one — per kind, not per host.

The shared core declares apparatus dimensions on the terms every LLM product has them: a judge model and
how it was asked, a simulated user and how it was asked, a spend ceiling. For a kind scored by a model, a
blank judge model means the judge is *unrecoverable*, which is exactly what should block a comparison.
For a kind graded by code it is not an absence at all, and reading it as one gave every such kind
``undecided`` apparatus confounds in every bundle.

**Declared per kind, because one host grades two ways.** A host-wide declaration was true of one kind
and false of the other: with ``judge_model`` declared away for the whole host, a judged run whose judge
was genuinely unrecoverable lost its ``undecided`` confound silently. Each kind contract names the seats
in the rig it fills (:attr:`~threetears.evals.contracts.host.kinds.KindContract.seats`), and a dimension
is omitted only for a cohort in which no kind seats it.

**Declared as an allow-list, so a new core dimension cannot make a host ``undecided``.** A kind names
what it HAS; a dimension added to the rig later — by the engine or by the host — is inapplicable to every
kind that has not claimed it. ``test_a_dimension_added_to_the_rig_later_makes_no_kind_undecided`` stages
exactly that, and its inverse.

**The surface asserted is the analysis bundle's apparatus levels**, which also covers the cell partition,
since ``apparatus_class_of`` receives ``set(apparatus_levels)``.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import pytest

from threetears.evals.contracts.host import SHARED_CORE, KindContract, RolePins, Sweepable, SweepableRegistry
from threetears.evals.contracts.host.sweepables import NO_JUDGE_CONFIGS
from threetears.evals.contracts.host.profile import HostProfile, ProfileRegistrationError
from packages.evals.tests.fixtures.toyhost.contract import TOY_EXTRACTOR_CONTRACT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.sweepables import TOYHOST_SWEEPABLE_REGISTRY

#: The apparatus a model-scored, simulator-driven kind would have and the toy extractor does not: the
#: core's model-judge axes and its simulator axes. ``judge_config_ids`` is not among them: the extractor's
#: reviewer-scored dimension records it as a level, and the contract seats it.
_UNSEATED_BY_THE_EXTRACTOR = {
    "judge_model",
    "judge_request_settings",
    "judge_dim_divergence",
    "simulator_model",
    "simulator_request_settings",
}

#: A second kind on the toy host, scored by a model judge: it seats the whole judge role.
_JUDGED_KIND = "toy-judged-extractor"
_JUDGED_CONTRACT = KindContract(
    _JUDGED_KIND,
    prefix="judged",
    seats=frozenset({"judge", "adjudicator", "ocr_engine_version", "max_cost_usd"}),
)


def _toyhost_bundle(profile: HostProfile):
    """The assembled toy-host bundle under ``profile``."""
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    return toyhost_bundle(profile=profile)


def _with_kinds(*kinds: KindContract) -> HostProfile:
    return replace(toyhost_profile(), kinds=kinds)


# --- the seats are validated where the contracts and the registry are both in hand -------------


def test_a_seat_naming_nothing_the_host_declares_is_refused() -> None:
    with pytest.raises(ProfileRegistrationError, match="seats 'judgee', which is nothing this host declares"):
        _with_kinds(replace(TOY_EXTRACTOR_CONTRACT, seats=TOY_EXTRACTOR_CONTRACT.seats | {"judgee"}))


def test_a_seat_naming_a_lever_or_a_label_is_refused() -> None:
    """Only apparatus is scanned for confounds, so seating a lever or a label says nothing."""
    with pytest.raises(ProfileRegistrationError, match="seats 'chunk_tokens', which is a lever"):
        _with_kinds(replace(TOY_EXTRACTOR_CONTRACT, seats=TOY_EXTRACTOR_CONTRACT.seats | {"chunk_tokens"}))
    with pytest.raises(ProfileRegistrationError, match="seats 'batch_label', which is a label"):
        _with_kinds(replace(TOY_EXTRACTOR_CONTRACT, seats=TOY_EXTRACTOR_CONTRACT.seats | {"batch_label"}))


def test_leaving_a_dimension_whose_blank_is_a_real_level_unseated_is_refused() -> None:
    """``reviewer_pool`` reads a blank as a recorded level, so unseating it could only ever warn."""
    with pytest.raises(ProfileRegistrationError, match="leaves 'reviewer_pool' unseated"):
        _with_kinds(replace(TOY_EXTRACTOR_CONTRACT, seats=TOY_EXTRACTOR_CONTRACT.seats - {"adjudicator"}))


def test_every_defect_in_one_declaration_is_reported_at_once() -> None:
    with pytest.raises(ProfileRegistrationError) as caught:
        _with_kinds(replace(TOY_EXTRACTOR_CONTRACT, seats=frozenset({"judgee", "chunk_tokens"})))
    message = str(caught.value)
    assert "'judgee', which is nothing this host declares" in message
    assert "'chunk_tokens', which is a lever" in message
    assert "leaves 'reviewer_pool' unseated" in message


def test_a_role_seats_every_pin_it_holds() -> None:
    """Seating ``judge`` seats the core's judge axes and the host's grader, which the toy nominated into it."""
    profile = _with_kinds(TOY_EXTRACTOR_CONTRACT, _JUDGED_CONTRACT)
    for pin in ("judge_model", "judge_config_ids", "grader_version"):
        assert not profile.omits_apparatus(pin, [(_JUDGED_KIND, None)]), pin


def test_asking_whether_to_omit_without_any_run_is_refused() -> None:
    """The check IS the kinds and values that ran — answering without them is the omission it replaces."""
    with pytest.raises(ValueError, match="called with no runs"):
        toyhost_profile().omits_apparatus("judge_model", [])


# --- per kind: one store, two grading regimes ---------------------------------------------------


def test_a_judged_run_keeps_its_undecided_judge_while_a_code_graded_cohort_omits_it() -> None:
    """The case a host-wide declaration got wrong, on one profile holding both kinds.

    A cohort of extractor runs alone omits the judge model; the same blank with one judged run in the
    cohort is kept, so the judged run's unrecoverable judge still reads as undecided.
    """
    profile = _with_kinds(TOY_EXTRACTOR_CONTRACT, _JUDGED_CONTRACT)

    assert profile.omits_apparatus("judge_model", [(TOY_EXTRACTOR_KIND, None), (TOY_EXTRACTOR_KIND, None)])
    assert not profile.omits_apparatus("judge_model", [(TOY_EXTRACTOR_KIND, None), (_JUDGED_KIND, None)])
    assert not profile.omits_apparatus("judge_model", [(_JUDGED_KIND, None)])


def test_a_kind_with_no_contract_is_held_to_every_seat() -> None:
    """Silence is conservative: a kind that has declared nothing has not said it lacks a judge."""
    assert not toyhost_profile().omits_apparatus("judge_model", [("a-kind-nobody-contracted", None)])


def test_a_kind_recording_a_level_it_does_not_seat_is_reported_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """A recorded level refutes the seat declaration: the dimension is reported and the contradiction logged.

    Every run's value goes in: the one blank arm agrees with the declaration, the arm that recorded a
    judge does not, and a caller passing only the first would drop a difference that is really there.
    """
    profile = toyhost_profile()
    blank_only = profile.omits_apparatus("judge_model", [(TOY_EXTRACTOR_KIND, None)])
    with caplog.at_level(logging.WARNING):
        with_a_level = profile.omits_apparatus(
            "judge_model", [(TOY_EXTRACTOR_KIND, None), (TOY_EXTRACTOR_KIND, ["vendor/judge-1"])]
        )

    assert blank_only, "nothing contradicts the seats when every run read blank"
    assert not with_a_level, "a run that recorded a judge refutes the seats, and the dimension must be reported"
    assert "does not seat apparatus dimension 'judge_model'" in caplog.text


# --- surface: the bundle, and with it the cell partition -----------------------------------------


def test_the_toy_hosts_bundle_names_no_apparatus_confound_it_did_not_observe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No confound, and no contradiction: every level the extractor's runs record is one it seats."""
    with caplog.at_level(logging.WARNING):
        bundle = _toyhost_bundle(toyhost_profile())

    assert "does not seat apparatus dimension" not in caplog.text, caplog.text
    assert bundle.apparatus_confounds == [], (
        "the toy host's clean campaign reports an apparatus confound nobody observed: "
        f"{[(c.dimension, c.status) for c in bundle.apparatus_confounds]}"
    )


def test_the_cell_partition_carries_no_dimension_the_kind_does_not_seat() -> None:
    """``unknown_dimensions`` is digested into ``apparatus_class_id``, so a fabricated unknown stops cells pooling."""
    bundle = _toyhost_bundle(toyhost_profile())

    unknown = {dimension for cell in bundle.cells for dimension in cell.unknown_dimensions}
    assert not unknown & _UNSEATED_BY_THE_EXTRACTOR, f"the cell partition still carries {sorted(unknown)}"


def test_holding_the_kind_to_every_seat_brings_the_unseated_confounds_back() -> None:
    """The acceptance case and the refusal case on ONE fixture, so the guard cannot be inverted.

    Every assertion above is satisfied by a tree that reports no undecidables at all. This one fails if
    the seats stopped being what does the work.
    """
    bundle = _toyhost_bundle(toyhost_profile(every_seat=True))

    reported = {confound.dimension for confound in bundle.apparatus_confounds if confound.status == "undecided"}
    assert reported == _UNSEATED_BY_THE_EXTRACTOR, (
        "held to every seat, the bundle should report exactly the dimensions the extractor does not seat as "
        f"undecided, and instead reported {sorted(reported)}"
    )


# --- a dimension added to the rig later -----------------------------------------------------------


def _a_rig_that_grew() -> SweepableRegistry:
    """The toy registry plus two apparatus dimensions it did not have when the extractor declared its seats.

    One joins the judge role, the way a new core judge axis would; one belongs to no role at all, the
    way a new core rig input would. Both are blank on every run, as a new dimension is on every run that
    predates it.
    """
    return TOYHOST_SWEEPABLE_REGISTRY.extend(
        (
            Sweepable(
                name="judge_rubric_digest",
                role="apparatus",
                read=lambda _run, _results: None,
                reader_prose="the digest of the rubric the judge was handed",
                confounds="a different rubric was handed to the judge",
                indeterminate_when_blank=True,
            ),
            Sweepable(
                name="cassette_corpus",
                role="apparatus",
                read=lambda _run, _results: None,
                reader_prose="the recorded corpus the run replayed",
                confounds="a different corpus was replayed",
                indeterminate_when_blank=True,
            ),
        ),
        roles=(RolePins(name="judge", pins=("judge_rubric_digest",)),),
    )


def test_a_dimension_added_to_the_rig_later_makes_no_kind_undecided() -> None:
    """The allow-list's whole point: the extractor never claimed either new dimension, so neither applies to it."""
    profile = replace(toyhost_profile(), host_sweepables=_a_rig_that_grew())

    bundle = _toyhost_bundle(profile)

    assert bundle.apparatus_confounds == [], (
        "a dimension added after the kind declared its seats made the bundle undecided: "
        f"{[(c.dimension, c.status) for c in bundle.apparatus_confounds]}"
    )


def test_the_same_growth_reaches_a_kind_that_declared_no_seats() -> None:
    """The inverse on the same fixture: a kind held to every seat does get the new dimensions, as undecided."""
    profile = replace(toyhost_profile(every_seat=True), host_sweepables=_a_rig_that_grew())

    bundle = _toyhost_bundle(profile)

    reported = {confound.dimension for confound in bundle.apparatus_confounds if confound.status == "undecided"}
    assert {"judge_rubric_digest", "cassette_corpus"} <= reported


# --- a judge-config set nothing was scored with is a blank, not a level ------------------------------------


def test_results_nobody_scored_record_no_judge_config_level() -> None:
    """The reader's sentinel says "scored, and by no versioned config"; results nothing scored say nothing.

    Were it the sentinel, a kind that seats no judge would record a level on every run, contradicting its
    own seats on every read, and the dimension could never be omitted.
    """
    from packages.evals.tests.factories import make_eval_result, make_eval_run

    reader = SHARED_CORE.get("judge_config_ids")
    assert reader is not None
    unscored = make_eval_result(rubric_scores=[])
    scored = make_eval_result()

    assert reader.read(make_eval_run(), [unscored]) == []
    assert reader.read(make_eval_run(), [scored]) == NO_JUDGE_CONFIGS
