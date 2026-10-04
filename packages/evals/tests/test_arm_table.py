"""The arm table, joined rather than authored — and the coordinate table that makes it possible.

The derivation has two halves and each fails differently. The JOIN is here: an adopted
decision names the winner by its cells, a rejected one names an exclusion, the campaign's declared
control names the incumbent, and everything else the campaign observed is an arm nobody reached a
verdict on. The COORDINATE table is the half that is easy to assume away — a decision cites
cells, a cell and a control are keyed by a variant digest, and without an index every row would be
a digest and the incumbent row could not name what it ran.

The campaign shape throughout is a typical model sweep: one axis, four candidate models, the
incumbent declared as the control.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.arms import arm_table
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.contracts.authored import NO_CHART, AuthoredAnalysis, EvidenceRef
from threetears.evals.contracts.campaign import (
    EvalAnalysis,
    FindingResolution,
    GenerationProvenance,
    VariantIndexEntry,
)
from threetears.evals.contracts.declaration import CampaignDesign, ControlDeclaration, SweptAxis
from threetears.evals.contracts.host.values import SweepableValue
from threetears.evals.contracts.identity import compute_variant_key
from threetears.evals.contracts.surface import DecisionSurface


_AXIS = "candidate_model"
_WINNER = "model-b-rev2"
_DEARER = "model-c-large"
_FASTER = "model-d-mini"
_INCUMBENT = "model-b"
_APPARATUS = "a" * 64


def _level(model: str) -> SweepableValue:
    """One level of the candidate-model axis, addressed the way a host resolves it."""
    return SweepableValue.of(model, display=model)


def _entry(model: str) -> VariantIndexEntry:
    """The index entry for the arm that ran ``model`` — key and levels, as generation freezes them."""
    levers = {_AXIS: _level(model)}
    return VariantIndexEntry(variant_key=compute_variant_key(levers), levers=levers)


def _key(model: str) -> str:
    """The variant key for the arm that ran ``model``."""
    return compute_variant_key({_AXIS: _level(model)})


def _cell(model: str) -> str:
    """The cell the arm that ran ``model`` was measured at, under the fixture's one rig."""
    return cell_ref(_key(model), _APPARATUS)


def _finding(title: str) -> dict[str, Any]:
    """A schema-valid authored finding with no readings of its own."""
    return {
        "title": title,
        "body": "rev2 holds tone at k=3.",
        "confidence": "high",
        "axes": [_AXIS],
        "evidence": [],
        "chart": {"type": NO_CHART, "cells": [], "measures": [], "axis": "", "note": "", "caption": ""},
        "caveats": [],
        "invalidates": [],
        "durable": "",
    }


def _decision(disposition: str, cells: list[str], rests_on: list[int]) -> dict[str, Any]:
    """A decision with ``disposition`` naming ``cells``, resting on the findings at ``rests_on``."""
    return {
        "proposal": "Move to the newer build.",
        "disposition": disposition,
        "cells": cells,
        "confidence": "high",
        "rests_on": rests_on,
        "revisit_when": "a k=5 pass" if disposition == "deferred" else "",
    }


def _worked_decisions() -> list[dict[str, Any]]:
    """The verdicts: ``_WINNER`` adopted on finding 0, the dearer and faster arms rejected on finding 1."""
    return [
        _decision("adopted", [_cell(_WINNER)], [0]),
        _decision("rejected", [_cell(_DEARER), _cell(_FASTER)], [1]),
    ]


def _document(decisions: list[dict[str, Any]] | None = None, findings: int = 2) -> AuthoredAnalysis:
    """The authored document: ``findings`` findings and ``decisions`` (the worked example's by default)."""
    return AuthoredAnalysis.model_validate(
        {
            "headline": "Move to the newer build.",
            "summary": "",
            "findings": [_finding(f"finding {position}") for position in range(findings)],
            "decisions": _worked_decisions() if decisions is None else decisions,
            "questions": [],
            "next": [],
        }
    )


def _row(model_or_key: str, measure_id: str, value: float, *, raw: bool = False) -> dict[str, Any]:
    """One resolved evidence row, at ``model_or_key``'s cell — or, with ``raw``, at that literal reference."""
    ref = model_or_key if raw else _cell(model_or_key)
    return {
        "cell_ref": ref,
        "measure_id": measure_id,
        "reading": "measure",
        "value": value,
        "n": 6,
        "dispersion": "p95",
    }


def _resolutions(*per_finding: list[dict[str, Any]]) -> list[FindingResolution]:
    """One resolution per finding, carrying the given resolved rows in order."""
    return [FindingResolution.model_validate({"evidence": rows}) for rows in per_finding]


def _design(control: str | None) -> CampaignDesign:
    """The declaration, sweeping the candidate-model axis against ``control``."""
    return CampaignDesign(
        axes=[SweptAxis(axis_id=_AXIS, values=[_level(model) for model in (_WINNER, _DEARER, _FASTER, _INCUMBENT)])],
        control=control,
        controls=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
    )


def _analysis(**overrides: Any) -> EvalAnalysis:
    """The campaign-shaped analysis: four indexed arms, the incumbent declared as the control."""
    defaults: dict[str, Any] = {
        "campaign_id": "campaign-1",
        "subject_id": "ent-maple",
        "subject_kind": "agent",
        "behavior": "conversation",
        "generation": GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="v1",
            generator_model="anthropic/claude-opus",
            bundle_fingerprint="sha256:abc",
            generated_at="2026-08-23T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        "document": _document(),
        "design_snapshot": _design(_key(_INCUMBENT)),
        "variant_index": [_entry(model) for model in (_WINNER, _DEARER, _FASTER, _INCUMBENT)],
        "decision_surface": DecisionSurface(),
    }
    if "resolutions" in overrides and "document" not in overrides:
        # Each finding names exactly the readings its resolution resolved, as a generated analysis does.
        document = defaults["document"].model_copy(deep=True)
        for finding, resolution in zip(document.findings, overrides["resolutions"], strict=True):
            finding.evidence = [
                EvidenceRef(cell=row.cell_ref, measure_id=row.measure_id, reading=row.reading)
                for row in resolution.evidence
            ]
        defaults["document"] = document
    return EvalAnalysis(**{"scope_id": "uni-1", **defaults, **overrides})


def _by_model(table) -> dict[str, str]:
    """Each arm's status, keyed by the level a reader would recognise it as."""
    return {row.levels[0].display: row.status for row in table.rows if row.levels}


class TestTheJoinProducesAllFourRowKinds:
    """The derivation, including the row no authored memo shape had a home for."""

    def test_every_row_kind_appears_over_the_worked_example(self) -> None:
        """The arm table: a winner, two exclusions, and the incumbent it replaced."""
        table = arm_table(_analysis())

        assert _by_model(table) == {
            _WINNER: "winner",
            _DEARER: "ruled_out",
            _FASTER: "ruled_out",
            _INCUMBENT: "replaced_incumbent",
        }

    @pytest.mark.parametrize(
        ("disposition", "arm_status", "control_status"),
        [
            pytest.param("adopted", "winner", "replaced_incumbent", id="adopted-wins-and-replaces-the-control"),
            pytest.param("rejected", "ruled_out", "unresolved", id="rejected-rules-out-and-replaces-nothing"),
            pytest.param("deferred", "unresolved", "unresolved", id="deferred-is-no-verdict"),
        ],
    )
    def test_the_disposition_naming_an_arms_cell_is_what_places_it(
        self, disposition: str, arm_status: str, control_status: str
    ) -> None:
        """One fixture, one decision naming one arm, only its disposition moved — so the rule is pinned both ways.

        An inverted rule (adopted read as ruled out) goes red on the first two cases together, and
        a rule reading any decision as a verdict goes red on the third. The control beside it is
        replaced only in the case where some arm WON: a rejection beats nobody.
        """
        document = _document([_decision(disposition, [_cell(_FASTER)], [1])])

        statuses = _by_model(arm_table(_analysis(document=document)))

        assert statuses == {
            _FASTER: arm_status,
            _INCUMBENT: control_status,
            _WINNER: "unresolved",
            _DEARER: "unresolved",
        }

    def test_the_incumbent_row_names_its_level_rather_than_a_digest(self) -> None:
        """The acceptance criterion, and the reason the index is persisted at all.

        The control is a variant KEY. Without the coordinate table there is nothing to render
        but that digest, and a reader cannot tell which model was replaced.
        """
        row = next(r for r in arm_table(_analysis()).rows if r.is_control)

        assert row.levels[0].display == _INCUMBENT
        assert row.status == "replaced_incumbent"

    def test_an_arm_no_decision_names_is_unresolved_rather_than_excluded(self) -> None:
        """Reaching no verdict is a state; rendering it as ruled out invents a result."""
        document = _document([_decision("adopted", [_cell(_WINNER)], [0])])

        assert _by_model(arm_table(_analysis(document=document)))[_FASTER] == "unresolved"

    def test_a_control_nothing_beat_is_not_reported_as_replaced(self) -> None:
        """ "Replaced" is a claim about another arm, so with no winner there is nobody to have replaced it."""
        table = arm_table(_analysis(document=_document([])))
        row = next(r for r in table.rows if r.is_control)

        assert row.status == "unresolved"
        assert row.is_control, "the control is still the control — the status says nothing happened to it"

    def test_a_winning_control_reads_as_the_winner_and_still_says_it_is_the_control(self) -> None:
        """An incumbent that held is one fact a reader must not have to reconstruct from two."""
        table = arm_table(_analysis(design_snapshot=_design(_key(_WINNER))))
        row = next(r for r in table.rows if r.is_control)

        assert (row.status, row.is_control) == ("winner", True)

    def test_the_rows_are_ordered_answer_first(self) -> None:
        """A reader wants the verdict, then what it beat, then what it replaced."""
        statuses = [row.status for row in arm_table(_analysis()).rows]

        assert statuses == ["winner", "ruled_out", "ruled_out", "replaced_incumbent"]

    def test_a_declared_control_the_index_does_not_hold_still_gets_a_row(self) -> None:
        """Dropping the row would lose the incumbent entirely; an unavailable level is a state to render."""
        table = arm_table(_analysis(variant_index=[_entry(model) for model in (_WINNER, _DEARER, _FASTER)]))
        row = next(r for r in table.rows if r.is_control)

        assert row.levels == [], "the index does not hold it, and inventing levels would mislabel the arm"
        assert row.variant_key == _key(_INCUMBENT)
        assert row.status == "replaced_incumbent"

    def test_an_analysis_with_no_index_and_no_control_derives_no_rows(self) -> None:
        """One generated before the coordinate table existed says nothing rather than guessing."""
        table = arm_table(_analysis(variant_index=[], design_snapshot=None))

        assert table.rows == []

    def test_an_empty_table_still_discloses_the_cells_its_decisions_name(self) -> None:
        """No rows is when a verdict is MOST likely to have stranded, so the disclosure must survive it.

        A renderer that returns its empty-table sentence INSTEAD of the disclosures puts the loudest
        warning behind the condition that produces it.
        """
        table = arm_table(_analysis(variant_index=[], design_snapshot=None))

        assert table.rows == []
        assert table.unplaced_decision_cells == sorted([_cell(_WINNER), _cell(_DEARER), _cell(_FASTER)]), (
            "with nothing indexed every cell a decision names strands, and that is the fact both renders owe"
        )


class TestEvidencePlacesOntoTheArms:
    """The numbers come from the resolved evidence rows, placed on the arm their cell belongs to."""

    def test_a_cell_keyed_row_places_through_the_variant_index(self) -> None:
        """The placement the index exists for: a cell coordinate resolved onto its arm."""
        table = arm_table(_analysis(resolutions=_resolutions([], [_row(_INCUMBENT, "p95_s", 81.0)])))
        row = next(r for r in table.rows if r.is_control)

        assert [m.measure_id for m in row.measurements] == ["p95_s"]
        assert row.measurements[0].finding_id == "1", "a number keeps the finding it was recorded under, by position"
        assert [m.measure_id for r in table.rows if not r.is_control for m in r.measurements] == [], (
            "a row at one arm's cell must not be attributed to another"
        )

    def test_measurements_keep_the_findings_order_and_then_each_findings_own(self) -> None:
        """The generator's ordering, not a re-sort — and each row still names the finding it came from."""
        resolutions = _resolutions(
            [_row(_WINNER, "tone_min", 5.0), _row(_WINNER, "p95_s", 40.0)],
            [_row(_WINNER, "cost_usd", 0.1)],
        )
        row = next(r for r in arm_table(_analysis(resolutions=resolutions)).rows if r.status == "winner")

        assert [(m.measure_id, m.finding_id) for m in row.measurements] == [
            ("tone_min", "0"),
            ("p95_s", "0"),
            ("cost_usd", "1"),
        ]

    def test_a_cell_keyed_row_places_on_an_incumbent_the_index_does_not_hold(self) -> None:
        """The control gets a row whether or not the index holds it, so its numbers belong on it.

        Keying the placement off the INDEX rather than off the table's rows sent exactly these
        measurements to `unplaced_coordinates` — beside the row they were measured at.
        """
        table = arm_table(
            _analysis(
                resolutions=_resolutions([], [_row(_INCUMBENT, "p95_s", 81.0)]),
                variant_index=[_entry(model) for model in (_WINNER, _DEARER, _FASTER)],
            )
        )
        row = next(r for r in table.rows if r.is_control)

        assert [m.measure_id for m in row.measurements] == ["p95_s"]
        assert table.unplaced_coordinates == []

    def test_a_measurement_at_no_known_arm_is_reported_rather_than_dropped(self) -> None:
        """Silence would render as though the evidence had been weighed."""
        stray = cell_ref("f" * 64, _APPARATUS)
        table = arm_table(_analysis(resolutions=_resolutions([], [_row(stray, "p95_s", 9.2, raw=True)])))

        assert table.unplaced_coordinates == [stray]
        assert all(not row.measurements for row in table.rows)

    def test_a_reference_in_no_recognisable_shape_is_unplaced_rather_than_guessed_at(self) -> None:
        """Reading a malformed ref as a bare variant key would put a number on the wrong arm."""
        table = arm_table(_analysis(resolutions=_resolutions([], [_row(_key(_INCUMBENT), "p95_s", 9.2, raw=True)])))

        assert table.unplaced_coordinates == [_key(_INCUMBENT)]
        assert all(not row.measurements for row in table.rows)

    def test_the_why_column_names_the_findings_the_placing_decision_rests_on(self) -> None:
        """A status with no traceable claim behind it is the copy the memo is forbidden from authoring."""
        document = _document(
            [
                _decision("adopted", [_cell(_WINNER)], [0, 2]),
                _decision("rejected", [_cell(_DEARER)], [1]),
                _decision("deferred", [_cell(_FASTER)], [2]),
            ],
            findings=3,
        )
        rows = {row.levels[0].display: row.finding_ids for row in arm_table(_analysis(document=document)).rows}

        assert rows[_WINNER] == ["0", "2"]
        assert rows[_DEARER] == ["1"]
        assert rows[_FASTER] == [], "a deferred decision is not a verdict, so it places nothing and explains nothing"
        assert rows[_INCUMBENT] == [], "the incumbent is placed by the DESIGN, and no decision claims it"


class TestAVerdictTheJoinCannotPlaceIsSaidOutLoud:
    """The blocking failure: a decision naming a winner the table places nowhere reads as silence."""

    def test_a_decision_naming_a_cell_no_arm_holds_is_named(self) -> None:
        """Every row derives `unresolved` under a decision adopting a winner — say so, or it reads true."""
        stray = cell_ref("f" * 64, _APPARATUS)
        table = arm_table(_analysis(document=_document([_decision("adopted", [stray], [0])])))

        assert table.unplaced_decision_cells == [stray]
        assert {row.status for row in table.rows} == {"unresolved"}, (
            "the join placed nothing, which is precisely the state the disclosure above exists to mark"
        )

    @pytest.mark.parametrize("disposition", ["rejected", "deferred"])
    def test_every_disposition_that_names_a_stray_cell_is_named(self, disposition: str) -> None:
        """The disclosure is about the cell, not the verdict: a stranded rejection is evidence nobody sees either."""
        stray = cell_ref("f" * 64, _APPARATUS)
        document = _document([*_worked_decisions(), _decision(disposition, [stray], [1])])

        assert arm_table(_analysis(document=document)).unplaced_decision_cells == [stray]

    def test_a_decision_the_join_places_strands_nothing(self) -> None:
        """The ordinary case, asserted so the disclosure above cannot pass by always firing."""
        table = arm_table(_analysis())

        assert table.unplaced_decision_cells == []
        assert table.unplaced_coordinates == []


class TestTheIndexEntryVerifiesItself:
    """The property the shape was chosen for: a wrong entry fails loudly instead of mislabelling an arm."""

    def test_an_entry_whose_levels_recompute_its_key_constructs(self) -> None:
        """The ordinary case, asserted so the refusal below is not passing for the wrong reason."""
        entry = _entry(_WINNER)

        assert compute_variant_key(entry.levers) == entry.variant_key

    def test_an_entry_whose_levels_belong_to_another_arm_is_refused(self) -> None:
        """Mislabelling is invisible downstream — a table naming the wrong winner reads like a right one."""
        with pytest.raises(ValidationError, match="recompute its own key|digest to"):
            VariantIndexEntry(variant_key=_key(_INCUMBENT), levers={_AXIS: _level(_WINNER)})

    def test_an_entry_declaring_its_levels_unavailable_constructs_with_no_levers(self) -> None:
        """The one door past the recompute, and it is an admission: no levers to mislabel with."""
        entry = VariantIndexEntry(variant_key="f" * 64, levers={}, levels_unavailable="a superseded predicate")

        assert entry.levers == {}
        assert entry.levels_unavailable == "a superseded predicate"

    def test_an_entry_declaring_its_levels_unavailable_may_not_carry_levers_anyway(self) -> None:
        """The door is narrow on purpose — a half-map smuggled through it mislabels the arm.

        Without this the declaration would be a way to skip the recompute while still carrying
        levels, which is exactly the state the recompute exists to refuse.
        """
        with pytest.raises(ValidationError, match="must describe nothing|declares its levels unavailable"):
            VariantIndexEntry(
                variant_key=_key(_INCUMBENT),
                levers={_AXIS: _level(_WINNER)},
                levels_unavailable="a superseded predicate",
            )


class TestAnArmMeasuredUnderASupersededPredicate:
    """An arm whose key this build cannot reproduce still gets a row.

    The defect this class pins: such an arm pooled, counted toward every `n`, and then appeared
    in no index — so the arm table, the generator prompt and the coverage gate alike could not
    see it, and an analysis over a campaign written before an `IDENTITY_VERSION` bump rendered
    zero arms with nothing anywhere saying why. Dropping the row is the silence the shape was
    chosen to prevent, and it is the same reason the declared control gets one whether or not
    the index holds it.
    """

    def test_an_arm_with_unavailable_levels_still_gets_a_row(self) -> None:
        """The row is the whole point — an arm nobody can describe is still an arm that ran."""
        superseded = VariantIndexEntry(variant_key="e" * 64, levers={}, levels_unavailable="a superseded predicate")
        table = arm_table(_analysis(variant_index=[_entry(_WINNER), superseded], design_snapshot=None))

        assert {row.variant_key for row in table.rows} == {_key(_WINNER), "e" * 64}

    def test_the_row_carries_no_levels_rather_than_borrowed_ones(self) -> None:
        """Describing it with today's levels would label the arm with a stack it never ran."""
        superseded = VariantIndexEntry(variant_key="e" * 64, levers={}, levels_unavailable="a superseded predicate")
        table = arm_table(_analysis(variant_index=[_entry(_WINNER), superseded], design_snapshot=None))

        row = next(r for r in table.rows if r.variant_key == "e" * 64)
        assert row.levels == []

    def test_a_campaign_written_entirely_under_a_superseded_predicate_keeps_every_arm(self) -> None:
        """The shape that rendered zero arms: no run of it agrees with today's predicate.

        Measured before the fix as 4 pooled / 0 indexed / 0 rows, and disclosed by no
        `unplaced` row (nothing was stranded).
        """
        index = [
            VariantIndexEntry(variant_key=chr(97 + i) * 64, levers={}, levels_unavailable="a superseded predicate")
            for i in range(4)
        ]
        table = arm_table(_analysis(variant_index=index, design_snapshot=None))

        assert len(table.rows) == 4


class TestAnArmIsNamedByTheKnobItSwept:
    """A one-knob sweep whose knob is written into a resolved surface the key is digested from.

    The index entry carries the surface in `levers` (the key's pre-image) and the knob in `swept`,
    with the surface `folded`. The fold decides how an arm is NAMED; a decision cites a cell, which
    is keyed by variant, so where the verdict lands must not move with the fold at all.
    """

    _SURFACE = "resolved_tool_configs"
    _KNOB = "planner.max_rounds"

    def _entries(self, *, folded: bool) -> list[VariantIndexEntry]:
        entries = []
        for rounds in (2, 3):
            levers = {self._SURFACE: SweepableValue.of({"planner": {"max_rounds": rounds}}, display="1 entries")}
            entries.append(
                VariantIndexEntry(
                    variant_key=compute_variant_key(levers),
                    levers=levers,
                    swept={self._KNOB: SweepableValue.of(rounds, display=str(rounds))} if folded else {},
                    folded=[self._SURFACE] if folded else [],
                )
            )
        return entries

    def _analysis(self, entries: list[VariantIndexEntry]) -> EvalAnalysis:
        """The two arms, with a decision adopting the two-round arm's cell."""
        two_rounds = cell_ref(entries[0].variant_key, _APPARATUS)
        document = _document([_decision("adopted", [two_rounds], [0])], findings=1)
        return _analysis(document=document, design_snapshot=None, variant_index=entries)

    def test_the_folded_index_names_each_arm_by_its_knob_and_places_the_verdict(self) -> None:
        table = arm_table(self._analysis(self._entries(folded=True)))

        labels = sorted(", ".join(f"{level.axis_id}={level.display}" for level in row.levels) for row in table.rows)
        assert labels == [f"{self._KNOB}=2", f"{self._KNOB}=3"]
        assert {row.levels[0].display: row.status for row in table.rows} == {"2": "winner", "3": "unresolved"}
        assert table.unplaced_decision_cells == []

    def test_the_unfolded_index_names_every_arm_alike_and_still_places_the_verdict(self) -> None:
        table = arm_table(self._analysis(self._entries(folded=False)))

        labels = [", ".join(f"{level.axis_id}={level.display}" for level in row.levels) for row in table.rows]
        assert labels == [f"{self._SURFACE}=1 entries"] * 2
        entries = self._entries(folded=False)
        assert {row.variant_key: row.status for row in table.rows} == {
            entries[0].variant_key: "winner",
            entries[1].variant_key: "unresolved",
        }
        assert table.unplaced_decision_cells == []

    def test_an_entry_may_fold_only_a_lever_it_carries(self) -> None:
        (entry, _) = self._entries(folded=True)

        with pytest.raises(ValidationError, match="does not carry as levers"):
            VariantIndexEntry(
                variant_key=entry.variant_key, levers=entry.levers, swept=entry.swept, folded=["candidate_model"]
            )
        assert VariantIndexEntry(
            variant_key=entry.variant_key, levers=entry.levers, swept=entry.swept, folded=[self._SURFACE]
        ).named_levers == {self._KNOB: entry.swept[self._KNOB]}
