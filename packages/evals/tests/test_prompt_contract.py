"""A stored proposer prompt written for another draft shape is refused before the call, naming the prompt to reset.

A host met this: its stored ``eval_proposer`` and ``eval_boundary_gen`` were seeded before ``RubricDim.scale`` became
required, and the store is master once seeded, so every ``rubric_propose`` paid for its call and then failed draft
validation on ``scale``, a parse error that named a field and not the stale prompt. Each shipped proposer default
now ends with the output contract its draft is read by (``eval-output-contract: rubric_proposal <fingerprint>``), and
``plan_draft`` reads the prompt it is handed against it:

- a copy that states no contract (every copy seeded before contracts were stated) or another one is refused with a
  :class:`PromptContractError` naming the prompt and the reset, before anything is priced;
- the shipped default, which is what a reset writes back, passes;
- an operator's edit that keeps the current line passes: the line is the operator's statement that the edit was
  written for this draft shape, and the strict response format and the draft's validation still stand behind it.
  An edit carrying an older line, or none, is refused like a stale copy: an edit can go stale too.

``prompt_drift`` reports the same reading, so a host can show it before a call is refused.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import pytest

from threetears.evals.gen import (
    EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT,
    EVAL_PROPOSER_TEMPLATE_DEFAULT,
    RUBRIC_PROPOSAL_CONTRACT,
    default_system_prompt,
    plan_draft,
)
from threetears.evals.schema.models import RubricAxis
from threetears.evals.kernel import (
    CONTRACT_MARKER,
    SEED_PROMPTS,
    ConflictError,
    OutputContract,
    PromptContractError,
    SeedTemplate,
    contract_state,
    declared_contracts,
    prompt_drift,
    require_output_contract,
    seed_digest,
    template_text,
)

_RESET = "call prompt_reset for the slot"

#: The fingerprint the proposal contract has in this build. A change that moves it refuses every stored proposer
#: prompt in every deployment until each is reset, so it is made on purpose: update this pin and write the reset
#: into the release's migration notes.
_PINNED_FINGERPRINT = "3f1e702e51f92b0e"

_SEEDS: dict[RubricAxis, SeedTemplate] = {
    "capability": EVAL_PROPOSER_TEMPLATE_DEFAULT,
    "boundary": EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT,
}
_PROMPT_TYPE: dict[RubricAxis, str] = {"capability": "eval_proposer", "boundary": "eval_boundary_gen"}
_AXES = list(_SEEDS)


def _entry(prompt_type: str) -> Any:
    return next(entry for entry in SEED_PROMPTS if entry.prompt_type == prompt_type)


def _with_text(template: SeedTemplate, text: str) -> SeedTemplate:
    first, *rest = template.sections
    return dataclasses.replace(template, sections=(dataclasses.replace(first, content_template=text), *rest))


def _without_line(template: SeedTemplate) -> SeedTemplate:
    """The default as every store seeded before contracts were stated holds it: the same text, no contract line."""
    text = template.sections[0].content_template
    return _with_text(template, text.replace(RUBRIC_PROPOSAL_CONTRACT.line, "").rstrip() + "\n")


def _schema() -> dict[str, Any]:
    format_ = plan_draft(
        reset=_RESET, axis="capability", system_prompt=RUBRIC_PROPOSAL_CONTRACT.line, subject_feed="F", catalog_feed="C"
    ).response_format
    assert format_ is not None
    return copy.deepcopy(format_["json_schema"]["schema"])


def _before_scale_was_required() -> OutputContract:
    """The contract as it read before ``RubricDim.scale`` was required: what a stale stored copy was written for."""
    schema = _schema()
    dim = schema["$defs"]["RubricDim"]
    dim["required"] = [name for name in dim["required"] if name != "scale"]
    del dim["properties"]["scale"]
    return OutputContract.of("rubric_proposal", schema, revision=1)


def _plan(axis: RubricAxis, template: SeedTemplate) -> Any:
    return plan_draft(
        reset=_RESET, axis=axis, system_prompt=template_text(template.sections), subject_feed="F", catalog_feed="C"
    )


# --- the contract and its line --------------------------------------------------------------------


def test_the_proposal_contract_is_pinned_so_a_change_that_refuses_every_stored_prompt_is_deliberate() -> None:
    assert RUBRIC_PROPOSAL_CONTRACT.name == "rubric_proposal"
    assert RUBRIC_PROPOSAL_CONTRACT.fingerprint == _PINNED_FINGERPRINT, (
        "the proposal contract moved: every stored eval_proposer and eval_boundary_gen copy is refused until reset. "
        "If that is intended, update _PINNED_FINGERPRINT and name the reset in the release's migration notes"
    )
    assert RUBRIC_PROPOSAL_CONTRACT.line == f"{CONTRACT_MARKER} rubric_proposal {_PINNED_FINGERPRINT}"


def test_rewording_a_fields_documentation_does_not_move_the_fingerprint() -> None:
    schema = _schema()
    schema["$defs"]["RubricDim"]["properties"]["scale"]["description"] = "reworded for the next developer"
    schema["$defs"]["RubricDim"]["title"] = "Renamed"
    assert OutputContract.of("rubric_proposal", schema, revision=1) == RUBRIC_PROPOSAL_CONTRACT


def test_a_property_named_like_an_annotation_still_counts() -> None:
    """``description`` is a draft field as well as a keyword; a field map's keys are names, never dropped."""
    schema = _schema()
    del schema["$defs"]["RubricDim"]["properties"]["description"]
    assert OutputContract.of("rubric_proposal", schema, revision=1) != RUBRIC_PROPOSAL_CONTRACT


def test_a_structural_change_or_a_revision_moves_the_fingerprint() -> None:
    assert _before_scale_was_required() != RUBRIC_PROPOSAL_CONTRACT
    assert OutputContract.of("rubric_proposal", _schema(), revision=2) != RUBRIC_PROPOSAL_CONTRACT


@pytest.mark.parametrize("axis", _AXES)
def test_each_shipped_proposer_default_ends_with_the_current_contract_line(axis: RubricAxis) -> None:
    text = template_text(_SEEDS[axis].sections)
    assert text.rstrip().endswith(RUBRIC_PROPOSAL_CONTRACT.line)
    assert declared_contracts(text) == (RUBRIC_PROPOSAL_CONTRACT,)
    assert contract_state(text, RUBRIC_PROPOSAL_CONTRACT) == "current"


# --- refused on use ---------------------------------------------------------------------------------


@pytest.mark.parametrize("axis", _AXES)
def test_a_stored_copy_seeded_before_contracts_were_stated_is_refused_naming_the_prompt_and_the_reset(
    axis: RubricAxis,
) -> None:
    with pytest.raises(PromptContractError) as refused:
        _plan(axis, _without_line(_SEEDS[axis]))

    error = refused.value
    assert isinstance(error, ConflictError) and error.status_code == 409
    assert (error.prompt_type, error.state, error.expected, error.declared) == (
        _PROMPT_TYPE[axis],
        "undeclared",
        RUBRIC_PROPOSAL_CONTRACT,
        (),
    )
    assert f"'{_PROMPT_TYPE[axis]}'" in error.message and "nothing was paid for" in error.message
    assert f"reset the stored '{_PROMPT_TYPE[axis]}' prompt" in error.remedy and _RESET in error.remedy
    assert RUBRIC_PROPOSAL_CONTRACT.line in error.remedy


@pytest.mark.parametrize("axis", _AXES)
def test_a_stored_copy_written_for_an_older_draft_shape_is_refused_as_outdated(axis: RubricAxis) -> None:
    older = _before_scale_was_required()
    stored = _with_text(
        _SEEDS[axis],
        _SEEDS[axis].sections[0].content_template.replace(RUBRIC_PROPOSAL_CONTRACT.line, older.line),
    )

    with pytest.raises(PromptContractError) as refused:
        _plan(axis, stored)

    assert (refused.value.state, refused.value.declared) == ("outdated", (older,))
    assert older.fingerprint in refused.value.message and RUBRIC_PROPOSAL_CONTRACT.fingerprint in refused.value.message


@pytest.mark.parametrize("axis", _AXES)
def test_a_reset_copy_the_shipped_default_passes(axis: RubricAxis) -> None:
    stored = SeedTemplate.from_dict(_SEEDS[axis].to_dict())
    call = _plan(axis, stored)
    assert call.system == template_text(stored.sections)


@pytest.mark.parametrize("axis", _AXES)
def test_an_operators_edit_that_keeps_the_current_line_passes(axis: RubricAxis) -> None:
    tuned = _with_text(
        _SEEDS[axis], f"The operator's own wording, written for this shape.\n\n{RUBRIC_PROPOSAL_CONTRACT.line}"
    )
    assert _plan(axis, tuned).system.startswith("The operator's own wording")


def test_an_operators_edit_that_dropped_the_line_is_refused_like_a_stale_copy() -> None:
    with pytest.raises(PromptContractError, match="states no output contract"):
        _plan("capability", _with_text(EVAL_PROPOSER_TEMPLATE_DEFAULT, "The operator's own wording."))


def test_a_copy_declaring_the_current_contract_beside_another_is_refused() -> None:
    """Two lines cannot both be what the copy was written for; an edit that appended the new one is not current."""
    text = f"{_before_scale_was_required().line}\nrules\n{RUBRIC_PROPOSAL_CONTRACT.line}"
    assert contract_state(text, RUBRIC_PROPOSAL_CONTRACT) == "outdated"
    with pytest.raises(PromptContractError):
        require_output_contract("eval_proposer", text, RUBRIC_PROPOSAL_CONTRACT, reset=_RESET)


def test_a_prompt_with_no_output_contract_is_never_refused() -> None:
    assert contract_state("anything at all", None) == "unbound"


# --- the same reading, before any call -----------------------------------------------------------


def test_a_stale_copy_from_before_contracts_reads_refused_with_a_remedy_naming_the_line() -> None:
    shipped = EVAL_PROPOSER_TEMPLATE_DEFAULT
    older = _without_line(shipped)
    drift = prompt_drift(
        _entry("eval_proposer"), shipped=shipped, stored=older, reset=_RESET, seeded_digest=seed_digest(older)
    )

    assert (drift.state, drift.contract, drift.refused) == ("stale", "undeclared", True)
    assert drift.expected_contract == RUBRIC_PROPOSAL_CONTRACT.line
    assert _RESET in drift.remedy and RUBRIC_PROPOSAL_CONTRACT.line in drift.remedy


def test_a_customized_copy_written_for_an_older_shape_is_kept_but_reads_refused() -> None:
    """Customized is not drift, but an edit written for a draft shape the engine no longer reads is still refused."""
    shipped = EVAL_PROPOSER_TEMPLATE_DEFAULT
    tuned = _with_text(shipped, f"the operator's wording\n{_before_scale_was_required().line}")
    drift = prompt_drift(
        _entry("eval_proposer"), shipped=shipped, stored=tuned, reset=_RESET, seeded_digest=None, customized=True
    )

    assert (drift.state, drift.stale, drift.contract, drift.refused) == ("customized", False, "outdated", True)
    assert "refused" in drift.remedy and RUBRIC_PROPOSAL_CONTRACT.line in drift.remedy


def test_a_customized_copy_written_for_the_current_shape_is_not_refused_and_names_no_remedy() -> None:
    shipped = EVAL_PROPOSER_TEMPLATE_DEFAULT
    tuned = _with_text(shipped, f"the operator's wording\n{RUBRIC_PROPOSAL_CONTRACT.line}")
    drift = prompt_drift(
        _entry("eval_proposer"), shipped=shipped, stored=tuned, reset=_RESET, seeded_digest=None, customized=True
    )
    assert (drift.state, drift.contract, drift.refused, drift.remedy) == ("customized", "current", False, "")


def test_the_shipped_default_reads_current_on_both_counts() -> None:
    shipped = EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT
    drift = prompt_drift(_entry("eval_boundary_gen"), shipped=shipped, stored=shipped, reset=_RESET, seeded_digest=None)
    assert (drift.state, drift.contract, drift.refused) == ("current", "current", False)


def test_the_analysis_generators_prompt_binds_no_contract() -> None:
    """Its reply shape is sent as a schema and its grammar appended from code, so its text states no shape."""
    entry = _entry("eval_analysis_gen")
    drift = prompt_drift(entry, shipped="new text", stored="old text", reset=_RESET, seeded_digest=None)
    assert (drift.contract, drift.refused, drift.expected_contract) == ("unbound", False, "")


def test_a_shipped_default_declaring_two_contracts_is_refused_as_a_defect() -> None:
    shipped = _with_text(
        EVAL_PROPOSER_TEMPLATE_DEFAULT, f"{RUBRIC_PROPOSAL_CONTRACT.line}\n{_before_scale_was_required().line}"
    )
    with pytest.raises(ValueError, match="2 output contracts"):
        prompt_drift(_entry("eval_proposer"), shipped=shipped, stored=shipped, reset=_RESET, seeded_digest=None)


@pytest.mark.parametrize(
    ("axis", "default"),
    [("capability", EVAL_PROPOSER_TEMPLATE_DEFAULT), ("boundary", EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT)],
)
def test_a_host_handing_no_prompt_sends_the_shipped_default(axis: RubricAxis, default: SeedTemplate) -> None:
    """A host that keeps no prompt registry passes nothing: the call runs the axis's default, contract line and all."""
    call = plan_draft(axis=axis, subject_feed="F", catalog_feed="C")
    assert call.system == template_text(default.sections) == default_system_prompt(axis)
    assert RUBRIC_PROPOSAL_CONTRACT.line in call.system


def test_an_override_naming_no_reset_is_refused() -> None:
    """A stale override's refusal names the host's reset, so an override that names none is refused for that."""
    with pytest.raises(ValueError, match="needs reset="):
        plan_draft(
            axis="capability",
            system_prompt=template_text(EVAL_PROPOSER_TEMPLATE_DEFAULT.sections),
            subject_feed="F",
            catalog_feed="C",
        )


def test_a_blank_reset_is_refused_before_the_prompt_is_read() -> None:
    """A refusal's remedy would name no way to reset the slot, so the plan is refused for the blank reset itself."""
    with pytest.raises(ValueError, match="reset is blank"):
        plan_draft(
            axis="capability",
            system_prompt=template_text(EVAL_PROPOSER_TEMPLATE_DEFAULT.sections),
            subject_feed="F",
            catalog_feed="C",
            reset=" ",
        )
