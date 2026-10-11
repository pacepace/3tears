"""A job record's ``faults`` (core v13): what went wrong at its lease's boundary, kept beside its ending.

The field is the same on every job record kind. These hold its shape (capped in count and length, and an entry
added past the caps kept inside them), that a record stored at v12 reads with none, and that a holder's own write of
its record keeps the faults written on the stored one. That a holder whose renewals fail writes one, and that a loss
keeps it, is in ``test_boundary_errors_are_recorded.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from threetears.evals.kernel import SWEEP_JOB, EvalStorage, EvalSweep, SweepArmRecord
from threetears.evals.kernel.job_leases import write_held_record
from threetears.evals.schema import JOB_FAULT_MAX_CHARS, JOB_FAULTS_MAX, with_fault
from threetears.evals.schema.versioning import CORE_SCHEMA_VERSION
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.stored_models import doc_type_of, stored_models

#: The frozen core corpus, one directory per core version (``test_core_schema_pins.py`` pins it).
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "core_documents"

#: Every job record kind: each stored model that declares the lease fields, faults among them.
_JOB_RECORDS = {doc_type_of(model): model for model in stored_models() if "lease" in model.model_fields}


def test_every_job_record_kind_keeps_faults() -> None:
    assert sorted(_JOB_RECORDS) == [
        "eval_action_job",
        "eval_analysis_attempt",
        "eval_judge_temperature_check",
        "eval_launch",
        "eval_run",
        "eval_second_judge_pass",
        "eval_sweep",
    ]
    assert all("faults" in model.model_fields for model in _JOB_RECORDS.values())


@pytest.mark.parametrize("doc_type", sorted(_JOB_RECORDS))
def test_a_job_record_stored_at_v12_reads_with_no_faults(doc_type: str) -> None:
    """The v12 step is the identity: a record stored before faults existed reads with none recorded."""
    document = json.loads((FIXTURES / "v12" / f"{doc_type}.json").read_text(encoding="utf-8"))
    assert document["schema_version"] == 12 and "faults" not in document

    read = _JOB_RECORDS[doc_type].from_dict(document)

    assert read.schema_version == CORE_SCHEMA_VERSION  # type: ignore[attr-defined]
    assert read.faults == []  # type: ignore[attr-defined]


@pytest.mark.parametrize("doc_type", sorted(_JOB_RECORDS))
def test_a_job_record_stored_at_v13_reads_its_faults_back(doc_type: str) -> None:
    document = json.loads((FIXTURES / "v13" / f"{doc_type}.json").read_text(encoding="utf-8"))

    read = _JOB_RECORDS[doc_type].from_dict(document)

    assert read.faults == document["faults"] and len(read.faults) == 1  # type: ignore[attr-defined]


def _sweep(**updates: object) -> EvalSweep:
    sweep = EvalSweep(
        scope_id="scope-1",
        campaign_id="campaign-1",
        template_id="tpl-1",
        subject_id="subject-1",
        arms=[SweepArmRecord(label="a", model="m1")],
        max_concurrent_arms=1,
    )
    return sweep.model_copy(update=updates)


@pytest.mark.parametrize(
    "faults",
    [[""], ["x" * (JOB_FAULT_MAX_CHARS + 1)], ["a fault"] * (JOB_FAULTS_MAX + 1)],
    ids=["blank", "too long", "too many"],
)
def test_a_stored_record_refuses_faults_past_their_caps(faults: list[str]) -> None:
    document = _sweep().to_dict() | {"faults": faults}
    with pytest.raises(ValueError, match="faults"):
        EvalSweep.from_dict(document)


def test_a_fault_added_past_the_caps_is_kept_inside_them() -> None:
    full = [f"fault {n}" for n in range(JOB_FAULTS_MAX)]

    kept = with_fault(full, "x" * (JOB_FAULT_MAX_CHARS + 50))

    assert len(kept) == JOB_FAULTS_MAX and kept[:-1] == full[:-1], "the earliest are kept"
    assert len(kept[-1]) == JOB_FAULT_MAX_CHARS and kept[-1].endswith("…"), "the newest takes the last place, cut"
    assert full[-1] == f"fault {JOB_FAULTS_MAX - 1}", "the stored list is not changed"
    assert EvalSweep.from_dict(_sweep().to_dict() | {"faults": kept}).faults == kept
    with pytest.raises(ValueError, match="blank"):
        with_fault([], "  ")


def test_a_holder_s_own_write_keeps_the_faults_written_on_its_record() -> None:
    """The holder's copy predates the fault; its write carries the stored faults, as it carries the stored lease."""
    storage = EvalStorage(InMemoryDocumentStore())
    sweep = _sweep()
    storage.save_sweep(sweep.model_copy(update={"faults": ["job lease: written by another writer"]}))

    written = write_held_record(storage, SWEEP_JOB, sweep.model_copy(update={"outcome": "completed"}))

    stored, _ = storage.load_job_record(EvalSweep, sweep.id, sweep.scope_id)
    assert stored is not None and stored.outcome == "completed"
    assert stored.faults == written.faults == ["job lease: written by another writer"]
