"""A host that does not HAVE an apparatus dimension is not a host that failed to record one.

The shared core declares seven apparatus dimensions on the terms every LLM product has them, and
six read core ``EvalRun`` fields that are ``indeterminate_when_blank``. For a host that judges and
simulates with models, a blank there means the judge or the simulator is *unrecoverable*, which is
exactly what should block a comparison. For a host that grades with code and simulates nobody it is
not an absence at all — there is no such thing to record — and reading it as one gave every such
host four ``undecided`` apparatus confounds in every bundle, enough for a real model to decline to
answer the campaign's declared question.

**The surface asserted here is the analysis bundle's apparatus levels**, which also covers the cell
partition, since ``apparatus_class_of`` receives ``set(apparatus_levels)`` and so needs no rule of
its own. A host that renders a compare table's role axes or bisects a pair of runs reports an
undecidable on those surfaces too, and owes each of them the same assertion where its fakes live —
a rule enforced in the branch you happened to be reading is not enforced.

**This file is not evidence that the class is closed.** Further surfaces assert a false sentence
about a host with no model judge, and they hang off one predicate — ``derive_context_identity``'s
``has_roles``, from which ``ContextIdentity.partial`` drives the report's ``context_incomplete``
badge. Changing it moves a HASH, so it is a separate change rather than a gap nobody recorded.

**A test here must FAIL when the fix is wrong in the other direction**, and that is the half worth
checking by hand: a change that simply stopped reporting undecidables would pass every assertion
about the toy host. ``test_removing_the_declaration_brings_the_fabricated_confounds_back`` is the
one that catches it.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.contracts.host.profile import Coverage, HostProfile, ProfileRegistrationError
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_INAPPLICABLE_APPARATUS, toyhost_profile


def _toyhost_bundle(profile: HostProfile):
    """The assembled toy-host bundle, under ``profile`` — the toy host's own or a variant of it.

    Args:
        profile: The profile to assemble under; the assembler resolves apparatus identity through it.

    Returns:
        The bundle.
    """
    from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle

    return toyhost_bundle(profile=profile)


# --- the declaration is validated where the map and the registry are both in hand -------------


def test_declaring_a_dimension_this_host_never_registered_is_refused() -> None:
    """An entry nothing would read, which would also outlive a rename of whatever it was for."""
    with pytest.raises(ProfileRegistrationError, match="never declares it as a sweepable"):
        replace(
            toyhost_profile(),
            apparatus_applicability={"a_dimension_nobody_declared": Coverage("inapplicable", "because")},
        )


def test_declaring_a_lever_or_a_label_inapplicable_is_refused() -> None:
    """Only apparatus is scanned for confounds, so exempting anything else suppresses nothing."""
    with pytest.raises(ProfileRegistrationError, match="declares it as a lever"):
        replace(toyhost_profile(), apparatus_applicability={"chunk_tokens": Coverage("inapplicable", "because")})
    with pytest.raises(ProfileRegistrationError, match="declares it as a label"):
        replace(toyhost_profile(), apparatus_applicability={"batch_label": Coverage("inapplicable", "because")})


def test_declaring_a_state_other_than_inapplicable_is_refused() -> None:
    """The engine derives the other two from the reader, so accepting either would be a no-op a host believed in."""
    for state in ("covered", "uncovered"):
        with pytest.raises(ProfileRegistrationError, match="inapplicable is the only state"):
            replace(toyhost_profile(), apparatus_applicability={"ocr_engine_version": Coverage(state, "because")})


def test_declaring_an_inapplicability_with_no_reason_is_refused() -> None:
    """The dimension drops out of every scan; a blank reason leaves nothing saying why."""
    with pytest.raises(ProfileRegistrationError, match="with no reason"):
        replace(toyhost_profile(), apparatus_applicability={"ocr_engine_version": Coverage("inapplicable", "   ")})


def test_declaring_a_dimension_whose_blank_is_a_real_level_is_refused() -> None:
    """Without ``indeterminate_when_blank`` the entry is a silent no-op that logs on every call.

    `omits_apparatus` asks `is_indeterminate`, which answers False for every value when the flag is
    off — a blank there is a recorded level. So such a declaration would never omit anything AND
    would report a contradiction about a value nobody set. `reviewer_pool` is exactly that shape:
    a toy-host apparatus dimension that declares no flag, because a pool nobody named is a real
    state of the batch.
    """
    with pytest.raises(ProfileRegistrationError, match="does not carry\\s+indeterminate_when_blank"):
        replace(toyhost_profile(), apparatus_applicability={"reviewer_pool": Coverage("inapplicable", "because")})


def test_asking_whether_to_omit_without_any_value_is_refused() -> None:
    """The check IS the values — answering without them is the unvalidated omission it replaces.

    Available by simply forgetting the argument, which would read as a passing call and silently
    restore the behaviour the value-aware check exists to end.
    """
    profile = toyhost_profile()
    with pytest.raises(ValueError, match="called with no values"):
        profile.omits_apparatus("judge_model")


def test_every_defect_in_one_declaration_is_reported_at_once() -> None:
    """A host fixing one sees the rest, which is the rule the sibling refusals in this profile follow."""
    with pytest.raises(ProfileRegistrationError) as caught:
        replace(
            toyhost_profile(),
            apparatus_applicability={
                "chunk_tokens": Coverage("inapplicable", "because"),
                "reviewer_pool": Coverage("covered", ""),
            },
        )
    message = str(caught.value)
    assert "declares it as a lever" in message
    assert "inapplicable is the only state" in message
    assert "with no reason" in message


# --- surface 1: the bundle, and with it the cell partition ------------------------------------


def test_the_toy_hosts_bundle_names_no_apparatus_confound_it_did_not_observe() -> None:
    """The fabricated confounds, asserted where a generator reads them.

    Three dimensions, not four: the fourth was ``max_cost_usd``, which the fixture now RECORDS
    rather than declaring away, so the two halves of this change each remove part of the problem.
    """
    profile = toyhost_profile()
    bundle = _toyhost_bundle(profile)

    reported = {confound.dimension for confound in bundle.apparatus_confounds}
    assert not reported & set(TOYHOST_INAPPLICABLE_APPARATUS), (
        "a dimension this host declared it does not have was reported as a confound: "
        f"{sorted(reported & set(TOYHOST_INAPPLICABLE_APPARATUS))}"
    )
    assert bundle.apparatus_confounds == [], (
        "the toy host's clean campaign reports an apparatus confound nobody observed: "
        f"{[(c.dimension, c.status) for c in bundle.apparatus_confounds]}"
    )


def test_the_cell_partition_stops_carrying_the_fabricated_unknowns() -> None:
    """The half that is not prose: ``unknown_dimensions`` is digested into ``apparatus_class_id``.

    ``merge_obstacle`` returns ``apparatus_unknown`` for any dimension unknown on either side, so
    the fabricated unknowns did not merely hedge a memo — they stopped cells from pooling, which is
    a pooling the host had declared through its subject key.
    """
    profile = toyhost_profile()
    bundle = _toyhost_bundle(profile)

    unknown = {dimension for cell in bundle.cells for dimension in cell.unknown_dimensions}
    assert not unknown & set(TOYHOST_INAPPLICABLE_APPARATUS), (
        f"the cell partition still carries {sorted(unknown & set(TOYHOST_INAPPLICABLE_APPARATUS))}, "
        "so cells that share a rig will refuse to merge"
    )


def test_removing_the_declaration_brings_the_fabricated_confounds_back() -> None:
    """The acceptance case and the refusal case on ONE fixture, so the guard cannot be inverted.

    Every assertion above is satisfied by a tree that reports no undecidables at all. This one
    fails if the declaration stopped being what does the work.
    """
    profile = replace(toyhost_profile(), apparatus_applicability={})
    undeclared = _toyhost_bundle(profile)

    reported = {confound.dimension for confound in undeclared.apparatus_confounds if confound.status == "undecided"}
    assert reported == set(TOYHOST_INAPPLICABLE_APPARATUS), (
        "with the declaration removed the bundle should fabricate exactly the confounds the "
        f"declaration exists to prevent, and instead reported {sorted(reported)}"
    )


# --- a declaration the runs refute loses, loudly ----------------------------------------------


def test_one_arm_recording_a_level_is_enough_to_refute_it() -> None:
    """Every arm's value goes in, not the first: omitting on the strength of the agreeing arm is the bug.

    ``UNRECORDED_APPARATUS`` has one arm carrying ``ocr_engine_version`` and one that never did. A
    caller passing only the blank arm would omit the dimension and lose a difference that is
    genuinely there on the other side.
    """
    claims_no_ocr = replace(
        toyhost_profile(),
        apparatus_applicability={"ocr_engine_version": Coverage("inapplicable", "no page is ever scanned, allegedly")},
    )
    profile = claims_no_ocr
    blank_only = profile.omits_apparatus("ocr_engine_version", None)
    with_a_level = profile.omits_apparatus("ocr_engine_version", None, "tess-5.3.1")

    assert blank_only, "nothing contradicts the declaration when every arm read blank"
    assert not with_a_level, "an arm that recorded a build refutes it, and the dimension must be reported"
