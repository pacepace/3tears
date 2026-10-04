"""What a run records about its own apparatus, over a run document with no host vocabulary.

``EvalRun`` reads strictly — a key it does not declare is refused — so a field the run is meant to
carry is asserted to SURVIVE the read, beside a control showing an undeclared key does not. The subject and every value here are invented for this file.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.models import ClientRequestSettings, EvalRun


_SUBJECT = SubjectSnapshot(subject_id="summarizer-3", subject_label="Summarizer, config 3", state=None)


def _run_document(**fields: Any) -> dict[str, Any]:
    """A stored run document as the store hands it back — JSON, not a model."""
    run = EvalRun(
        id="run-1",
        scope_id="scope-a",
        subject_snapshot=_SUBJECT,
        candidate_model="writer-a",
        test_case_ids=["c-1"],
        candidate_kind="test-kind",
        k_runs=1,
        rubric_scales={},
    )
    return {**run.model_dump(mode="json"), **fields}


def test_an_undeclared_key_is_refused_by_the_read() -> None:
    """The control: without a declared field, a stamped value would be refused exactly like this."""
    with pytest.raises(ValidationError, match="not_a_run_field"):
        EvalRun.from_dict(_run_document(not_a_run_field="x"))


class TestTheRunRecordsItsCandidateKind:
    def test_the_kind_survives_the_read_and_a_round_trip(self) -> None:
        run = EvalRun.model_validate(_run_document(candidate_kind="document-summarizer"))
        assert run.candidate_kind == "document-summarizer"
        assert EvalRun.model_validate(run.model_dump(mode="json")).candidate_kind == "document-summarizer"

    def test_a_run_that_names_no_kind_is_refused(self) -> None:
        """A run states which kind it measured; one that does not cannot be read against any kind's contract."""
        document = _run_document()
        document.pop("candidate_kind", None)
        with pytest.raises(ValidationError, match="candidate_kind"):
            EvalRun.model_validate(document)


class TestTheRunRecordsHowItsApparatusWasAsked:
    def test_both_roles_settings_survive_the_read_and_a_round_trip(self) -> None:
        judge = {"max_tokens": 10240, "reasoning_max_tokens": 8192}
        simulator = {"max_tokens": 4096, "reasoning_max_tokens": None}
        run = EvalRun.model_validate(_run_document(judge_request_settings=judge, simulator_request_settings=simulator))
        assert run.judge_request_settings == ClientRequestSettings(max_tokens=10240, reasoning_max_tokens=8192)
        assert run.simulator_request_settings == ClientRequestSettings(max_tokens=4096)
        again = EvalRun.model_validate(run.model_dump(mode="json"))
        assert (again.judge_request_settings, again.simulator_request_settings) == (
            run.judge_request_settings,
            run.simulator_request_settings,
        )

    def test_a_run_stored_before_the_stamp_reads_as_unrecorded(self) -> None:
        document = _run_document()
        document.pop("judge_request_settings", None)
        document.pop("simulator_request_settings", None)
        run = EvalRun.model_validate(document)
        assert run.judge_request_settings is None and run.simulator_request_settings is None

    def test_no_reasoning_parameter_is_a_recorded_level(self) -> None:
        assert ClientRequestSettings(max_tokens=4096).reasoning_max_tokens is None

    @pytest.mark.parametrize(
        "fields", [{"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 4096, "reasoning_max_tokens": 0}]
    )
    def test_a_non_positive_setting_is_refused(self, fields) -> None:
        with pytest.raises(ValidationError):
            ClientRequestSettings(**fields)
