"""Every rendered surface names a measure, a question and an arm level in a reader's words (#627, #581).

A measure's key (``candidate_output_tokens_per_s``) is what the store files a number under; a reader needs
"Output speed". A declared question's id is a uuid; a reader needs the question. A prompt-sweep level's host
display is a fingerprint; a reader needs the name the campaign gave it. Each test here fails when its fix is
reverted: the key, the id or the host's display comes back.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from threetears.evals.analysis import (
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
)
from threetears.evals.analysis.arms import arm_names, writer_arms
from threetears.evals.kernel.declaration import SweptAxis
from threetears.evals.kernel.host.measures import MeasureRegistrationError, MeasureRegistry
from threetears.evals.schema.values import IntervalScale, SweepableValue
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, MetricDescriptor
from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_AXIS,
    TOYHOST_NARROW,
    TOYHOST_WIDE,
    toyhost_campaign,
)
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.aspect_support import with_aspect


# --- #627: a measure has a reader-facing name, and every surface heads it by that -------------------------------


def test_a_host_measure_with_no_reader_name_is_refused_at_registration() -> None:
    unnamed = next(d for d in TOYHOST_MEASURES if d.name == "field_accuracy").model_copy(update={"reader_name": None})
    with pytest.raises(MeasureRegistrationError, match="field_accuracy has no reader_name"):
        MeasureRegistry([unnamed])


def test_two_measures_a_reader_could_not_tell_apart_are_refused() -> None:
    accuracy = next(d for d in TOYHOST_MEASURES if d.name == "field_accuracy")
    twin = accuracy.model_copy(update={"name": "field_accuracy_v2", "reader_name": "FIELD ACCURACY"})
    with pytest.raises(MeasureRegistrationError, match="share the reader_name"):
        MeasureRegistry([accuracy, twin])
    # Nor may a host measure take a core measure's words: its column would read as the core's.
    taken = accuracy.model_copy(update={"reader_name": METRIC_DESCRIPTORS["total_ms"].reader_name})
    with pytest.raises(MeasureRegistrationError, match="total_ms"):
        MeasureRegistry([taken])


def test_every_core_measure_and_every_one_described_by_construction_has_a_reader_name() -> None:
    assert all(d.reader_name for d in METRIC_DESCRIPTORS.values())
    names = [d.reader_name.casefold() for d in METRIC_DESCRIPTORS.values() if d.reader_name]
    assert len(names) == len(set(names))
    assert METRIC_DESCRIPTORS["candidate_output_tokens_per_s"].reader_name == "Output speed"


def test_a_descriptor_stored_before_the_field_reads_with_none_and_is_headed_by_its_key() -> None:
    stored = {k: v for k, v in METRIC_DESCRIPTORS["total_ms"].model_dump(mode="json").items() if k != "reader_name"}
    assert MetricDescriptor.model_validate_json(json.dumps(stored)).reader_name is None


# --- #581: an arm level the campaign named is called by that name everywhere an arm is labelled ----------------


def _named_design(names: list[tuple[int, str]]) -> Any:
    """The toy design, its swept levels declared under ``names`` — the same content, so the same hashes."""
    campaign, storage = toyhost_campaign()
    axis = SweptAxis(
        axis_id=TOYHOST_AXIS,
        values=[
            SweepableValue.of(level, display=name, scale=IntervalScale(value=level, unit="tok"), keep_raw=True)
            for level, name in names
        ],
    )
    return with_aspect(campaign, design={"axes": [axis]}), storage


def test_a_declared_level_name_labels_the_arm_on_every_surface_and_moves_no_key() -> None:
    profile = toyhost_profile()
    plain, plain_storage = toyhost_campaign()
    before = assemble_context_bundle(plain, storage=plain_storage, profile=profile)
    campaign, storage = _named_design([(TOYHOST_NARROW, "current width"), (TOYHOST_WIDE, "proposed width")])
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)

    assert [e.variant_key for e in bundle.variant_index] == [e.variant_key for e in before.variant_index]
    assert sorted(arm_names(bundle.variant_index).values()) == [
        f"{TOYHOST_AXIS}=current width",
        f"{TOYHOST_AXIS}=proposed width",
    ]
    listed = writer_arms(bundle.variant_index)["arms"]
    assert sorted(arm["levels"][TOYHOST_AXIS] for arm in listed) == ["current width", "proposed width"]  # type: ignore[index]
    report = build_code_only_report(bundle, assembled_at="2026-10-10T00:00:00Z")
    scoreboard = next(b for b in report.blocks if isinstance(b, TableBlock) and b.name == "scoreboard")
    # The page names an arm by its level's name alone, never as ``axis=level``.
    assert {str(row["arm"]).removeprefix("▸ ") for row in scoreboard.values()} == {"current width", "proposed width"}


def test_a_level_declared_twice_takes_its_first_name_and_an_undeclared_one_keeps_the_hosts() -> None:
    profile = toyhost_profile()
    plain, plain_storage = toyhost_campaign()
    host_display = {
        level.content_hash: level.display
        for entry in assemble_context_bundle(plain, storage=plain_storage, profile=profile).variant_index
        for level in [entry.levers[TOYHOST_AXIS]]
    }
    campaign, storage = _named_design([(TOYHOST_NARROW, "first"), (TOYHOST_NARROW, "second")])
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    displays = {
        entry.levers[TOYHOST_AXIS].content_hash: entry.levers[TOYHOST_AXIS].display for entry in bundle.variant_index
    }
    narrow = SweepableValue.of(TOYHOST_NARROW, scale=IntervalScale(value=TOYHOST_NARROW, unit="tok")).content_hash
    wide = SweepableValue.of(TOYHOST_WIDE, scale=IntervalScale(value=TOYHOST_WIDE, unit="tok")).content_hash
    assert displays[narrow] == "first"
    assert displays[wide] == host_display[wide]
