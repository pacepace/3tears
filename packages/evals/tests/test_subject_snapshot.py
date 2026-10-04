"""The subject the engine reads, and the values it is made of.

Two rules are load-bearing enough to be worth a file of their own, because each of them fails
silently when it fails at all:

- a :class:`SweepableValue` is identity **and** rendering, and a value carrying only the first
  degrades every analysis downstream of it to "component 3 moved";
- a :class:`SubjectSnapshot` holds hashes, never component text, because retention that grows
  with what the candidate produced is the one axis an eval run's memory must not scale on.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.hashing import bytes_digest, canonical_digest
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.sweepables import IntervalScale, NominalScale, OrdinalScale, SweepableValue


class TestSweepableValueIdentity:
    """Content-addressing: what makes two levels the same level."""

    def test_the_same_content_in_a_different_key_order_is_the_same_level(self) -> None:
        first = SweepableValue.of({"a": 1, "b": 2}, display="v3")
        second = SweepableValue.of({"b": 2, "a": 1}, display="v3")

        assert first.content_hash == second.content_hash

    def test_a_different_display_over_the_same_content_does_not_split_the_level(self) -> None:
        """Rendering is not identity — a relabelled level is the same level."""
        assert (
            SweepableValue.of({"a": 1}, display="v3").content_hash
            == SweepableValue.of({"a": 1}, display="v4").content_hash
        )

    def test_different_content_is_a_different_level(self) -> None:
        assert (
            SweepableValue.of({"a": 1}, display="x").content_hash
            != SweepableValue.of({"a": 2}, display="x").content_hash
        )

    def test_the_hash_is_the_canonical_digest_of_the_content(self) -> None:
        """Pinned against the shared primitive, so the two layers cannot drift apart."""
        assert SweepableValue.of({"a": 1}, display="x").content_hash == canonical_digest({"a": 1})


class TestSweepableValueFromBytes:
    """A host supplies an opaque blob and no registry is present."""

    def test_a_level_computes_from_bytes_with_no_registry_in_sight(self) -> None:
        value = SweepableValue.of_bytes(b"an extraction schema nobody registered")

        assert value.content_hash == bytes_digest(b"an extraction schema nobody registered")

    def test_bytes_are_addressed_directly_rather_than_through_their_json_encoding(self) -> None:
        """Otherwise the identity is of the encoding, and two encodings of one blob would split."""
        assert SweepableValue.of_bytes(b"abc").content_hash != SweepableValue.of("abc", display="abc").content_hash

    def test_the_default_display_is_a_poor_label_and_an_honest_one(self) -> None:
        value = SweepableValue.of_bytes(b"abc")

        assert value.display == f"sha256:{value.content_hash[:12]}"

    def test_bytes_never_retain_their_content(self) -> None:
        assert SweepableValue.of_bytes(b"carried memory nobody should store").raw is None


class TestSweepableValueRendering:
    """Legibility — a value a reader can act on."""

    def test_a_nominal_level_without_a_display_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no natural rendering|needs a display"):
            SweepableValue.of({"a": 1})

    def test_an_ordinal_level_without_a_display_is_refused(self) -> None:
        """A rank orders the levels; it does not name them."""
        with pytest.raises(ValueError, match="needs a display"):
            SweepableValue.of({"a": 1}, scale=OrdinalScale(rank=2))

    def test_an_interval_level_derives_its_display_from_the_number(self) -> None:
        assert SweepableValue.of(0.4, scale=IntervalScale(value=0.4)).display == "0.4"

    def test_an_interval_level_carries_its_unit_into_the_display(self) -> None:
        assert SweepableValue.of(2000, scale=IntervalScale(value=2000, unit="ms")).display == "2000ms"

    def test_an_explicit_display_wins_over_the_derived_one(self) -> None:
        assert SweepableValue.of(0.4, display="the knee", scale=IntervalScale(value=0.4)).display == "the knee"

    def test_a_blank_display_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SweepableValue(content_hash="0" * 64, display="", scale=NominalScale())


class TestSweepableValueStrictRead:
    """A stored type reads as strictly as it is built: a field this build does not declare is refused."""

    def test_a_field_this_build_does_not_declare_is_refused(self):
        stored = {**SweepableValue.of({"a": 1}, display="v3").model_dump(mode="json"), "a_field_from_next_year": 1}

        with pytest.raises(ValidationError, match="a_field_from_next_year"):
            SweepableValue.model_validate(stored)

    def test_an_unknown_scale_kind_is_refused(self):
        with pytest.raises(ValidationError):
            SweepableValue.model_validate({"content_hash": "0" * 64, "display": "x", "scale": {"kind": "logarithmic"}})

    def test_a_subject_snapshot_with_an_undeclared_field_is_refused(self):
        stored = {
            **SubjectSnapshot(subject_id="s-1", subject_label="S", state={}).model_dump(mode="json"),
            "notes": "x",
        }

        with pytest.raises(ValidationError, match="notes"):
            SubjectSnapshot.model_validate(stored)


class TestSweepableValueScale:
    """Three scales, because rank without spacing draws a false chart."""

    def test_the_default_scale_is_nominal(self) -> None:
        """The only honest default for a value the caller said nothing about."""
        assert SweepableValue.of({"a": 1}, display="v3").scale.kind == "nominal"

    def test_an_interval_level_keeps_the_number_rather_than_a_rank(self) -> None:
        """0.1 / 0.4 / 0.85 must not render equally spaced — that puts the knee in the wrong place."""
        levels = [SweepableValue.of(v, scale=IntervalScale(value=v)) for v in (0.1, 0.4, 0.85)]

        assert [level.scale.value for level in levels] == [0.1, 0.4, 0.85]

    def test_the_scale_union_round_trips_through_json(self) -> None:
        original = SweepableValue.of(15, scale=IntervalScale(value=15, unit="items"))

        assert SweepableValue.model_validate(original.model_dump(mode="json")) == original


class TestSweepableValueRetention:
    """``raw`` is for a human debugging a key, and is nothing a consumer may compute from."""

    def test_content_is_dropped_by_default(self) -> None:
        assert SweepableValue.of({"recent_memory": ["turn 1", "turn 2"]}, display="Bob's memory").raw is None

    def test_content_is_retained_only_when_the_caller_asks(self) -> None:
        assert SweepableValue.of({"a": 1}, display="x", keep_raw=True).raw == {"a": 1}


class TestSubjectKey:
    """The subject key declares the pooling boundary, which pooling runs at all depends on."""

    def test_a_blank_subject_id_is_unrepresentable(self) -> None:
        # Every other required field supplied, so the refusal is about THIS one. Without that the
        # test would keep passing off any later required field's absence.
        with pytest.raises(ValidationError, match="subject_id"):
            SubjectSnapshot(subject_id="", subject_label="Bob", state=None)

    def test_a_blank_subject_label_is_unrepresentable(self) -> None:
        with pytest.raises(ValidationError, match="subject_label"):
            SubjectSnapshot(subject_id="ent-1", subject_label="", state=None)

    def test_key_and_label_are_separate_fields_even_when_a_host_has_one_string(self) -> None:
        """Passing the name into both is allowed; what it must not do is go unrecorded."""
        snapshot = SubjectSnapshot(subject_id="Bob", subject_label="Bob", state=None)

        assert snapshot.subject_id == snapshot.subject_label == "Bob"


class TestSubjectSnapshotRetention:
    """An eval run's memory is bounded by its matrix, never by what it observed."""

    def test_a_component_is_reachable_by_hash_and_not_by_content(self) -> None:
        memory = ["a long transcript the candidate produced", "and another"]
        snapshot = SubjectSnapshot(
            subject_id="ent-1",
            subject_label="Bob",
            components={"recent_memory": SweepableValue.of(memory, display="20 turns")},
            state=None,
        )

        component = snapshot.components["recent_memory"]
        assert component.content_hash == canonical_digest(memory)
        assert component.raw is None
        assert "transcript" not in snapshot.model_dump_json()

    def test_component_hashes_is_exactly_what_a_key_predicate_reads(self) -> None:
        snapshot = SubjectSnapshot(
            subject_id="ent-1",
            subject_label="Bob",
            components={
                "backstory": SweepableValue.of("who Bob is", display="backstory"),
                "style": SweepableValue.of("how Bob speaks", display="style"),
            },
            state=None,
        )

        assert snapshot.component_hashes() == {
            "backstory": canonical_digest("who Bob is"),
            "style": canonical_digest("how Bob speaks"),
        }


class TestSubjectSnapshotShape:
    """Labels identify; they never determine."""

    def test_labels_are_plain_strings_and_carry_no_identity(self) -> None:
        snapshot = SubjectSnapshot(subject_id="ent-1", subject_label="Bob", labels={"subject_name": "Bob"}, state=None)

        assert snapshot.labels == {"subject_name": "Bob"}

    def test_state_is_required_so_a_writer_cannot_skip_the_recording_versus_absence_choice(self) -> None:
        """Requiredness is load-bearing prose everywhere this field's None is described, and nothing else pins it.

        ``{}`` records that this subject carries nothing outside its variant components; ``None``
        records that the host wired no state reader. Collapsing them pools a subject carrying a
        year of accumulated state with a stripped probe, so a writer has to state which it means
        rather than inherit a default.

        A later ``= None`` would silently unmake that and redden nothing, letting a writer that
        forgot the field produce the absence by omission.
        """
        assert SubjectSnapshot.model_fields["state"].is_required()

        with pytest.raises(ValidationError) as excinfo:
            SubjectSnapshot.model_validate({"subject_id": "e1", "subject_label": "Bob", "components": {}, "labels": {}})
        assert [e["type"] for e in excinfo.value.errors()] == ["missing"]
        assert [e["loc"] for e in excinfo.value.errors()] == [("state",)]

    def test_a_snapshot_round_trips_through_json(self) -> None:
        original = SubjectSnapshot(
            subject_id="ent-1",
            subject_label="Bob",
            components={"style": SweepableValue.of("how Bob speaks", display="style")},
            state={"carried_memory": SweepableValue.of(["turn 1"], display="1 entry")},
            labels={"subject_name": "Bob"},
        )

        assert SubjectSnapshot.model_validate(original.model_dump(mode="json")) == original
