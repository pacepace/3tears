"""A template declares the display name a page prints for each stratum (``EvalTemplate.stratum_names``).

A name is at most 32 characters, never blank, one per stratum, and written only when declared, so a template naming no
stratum is stored as durable v16.4 stored it, and a v16.4 document reads. It is presentation only, beside
``goal_check_names``: in neither template digest and in no bundle fingerprint, and a ``template_update`` of it alone
leaves the template's ``updated_at``, so naming a stratum stales no analysis and cuts no run off from its criteria.

It rides ``AnalysisContextBundle.stratum_names``, written only when declared (an aspect bundle of a template naming no
stratum is shape 65 byte for byte), and prints where a stratum is named: the report input's ``Stratum.name``, so a
figure key's ``strata.<id>.name`` and a plan's stratum labels, and an aspect page's stratum columns and cases table.
Undeclared, a stratum prints as its id on an aspect page, and a figure key prints its reader name (the id with its
underscores as spaces, :func:`~threetears.evals.analysis.report.figure_keys.reader_name`), so a plan's prose never
prints an id above Full results through a key.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import make_aspect_campaign, make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.report_shapes_support import Shape, s1_change
from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, evidence_digest
from threetears.evals.analysis.campaign_bundle import CAMPAIGN_BUNDLE_SHAPE, build_campaign_bundle
from threetears.evals.analysis import TableBlock, build_code_only_report
from threetears.evals.analysis.report.build import NO_STRATUM
from threetears.evals.analysis.report.figure_keys import reader_name, render, resolve
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import StratumFacts
from threetears.evals.run.definition_digests import measured_template_digest, template_digest
from threetears.evals.schema.models import TEMPLATE_PRESENTATION_FIELDS, EvalTemplate

STAMP = "2026-10-01T00:00:00+00:00"
NAMES = {"off_lane": "Off-lane asks", "assistant_shape": "Sent a message"}


def _template(**fields: Any) -> EvalTemplate:
    return make_template(id="tpl-1", created_at=STAMP, updated_at=STAMP, **fields)


# --- the field ------------------------------------------------------------------------------------------------------


class TestAStratumsName:
    def test_it_is_kept_and_written_only_when_declared(self) -> None:
        assert _template(stratum_names=dict(NAMES)).to_dict()["stratum_names"] == NAMES
        assert "stratum_names" not in _template().to_dict()

    def test_a_name_in_any_script_at_the_cap_is_kept(self) -> None:
        assert _template(stratum_names={"a": "é" * 32}).stratum_names == {"a": "é" * 32}

    @pytest.mark.parametrize(
        ("names", "refusal"),
        [
            pytest.param({"a": "b" * 33}, "at most 32", id="over the cap"),
            pytest.param({"a": " "}, "is blank", id="blank"),
            pytest.param({" ": "Name"}, "blank stratum id", id="blank id"),
            pytest.param({"a": "Same", "b": "Same"}, "to more than one stratum", id="one name twice"),
        ],
    )
    def test_a_bad_one_is_refused(self, names: dict[str, str], refusal: str) -> None:
        with pytest.raises(ValidationError, match=refusal):
            _template(stratum_names=names)

    def test_it_is_a_presentation_field(self) -> None:
        assert "stratum_names" in TEMPLATE_PRESENTATION_FIELDS

    def test_a_v16_4_template_naming_no_stratum_reads_and_is_written_back_unchanged(self) -> None:
        document = {**_template().to_dict(), "schema_minor": 4}
        assert "stratum_names" not in document

        read = EvalTemplate.from_dict(document)

        assert read.stratum_names == {}
        assert read.to_dict()["schema_minor"] == EvalTemplate.model_fields["schema_minor"].default
        assert {k: v for k, v in read.to_dict().items() if k != "schema_minor"} == {
            k: v for k, v in document.items() if k != "schema_minor"
        }

    def test_a_named_template_round_trips(self) -> None:
        assert EvalTemplate.from_dict(_template(stratum_names=dict(NAMES)).to_dict()).stratum_names == NAMES

    def test_it_composes_with_check_names(self) -> None:
        both = _template(
            goal_state_checks=["state.a >= 1"], goal_check_names={"state.a >= 1": "A held"}, stratum_names=dict(NAMES)
        )
        assert EvalTemplate.from_dict(both.to_dict()).stratum_names == NAMES


# --- in no digest or fingerprint --------------------------------------------------------------------------------------


class TestNamingAStratumMovesNoIdentity:
    def test_it_changes_neither_template_digest(self) -> None:
        unnamed, named = _template(), _template(stratum_names=dict(NAMES))
        assert measured_template_digest(named) == measured_template_digest(unnamed)
        assert template_digest(named) == template_digest(unnamed)

    def test_changing_a_name_changes_neither_template_digest(self) -> None:
        one, other = _template(stratum_names=dict(NAMES)), _template(stratum_names={"off_lane": "Elsewhere"})
        assert template_digest(one) == template_digest(other)
        assert measured_template_digest(one) == measured_template_digest(other)

    @staticmethod
    def _assembled(templates: list[EvalTemplate], *, runs_of: int | None = None) -> AnalysisContextBundle:
        runs = [
            make_eval_run(id=f"run-{index}", status="completed", template_id=f"tpl-{index}", created_at=STAMP)
            for index in range(len(templates) if runs_of is None else runs_of)
        ]
        campaign = make_aspect_campaign(
            id="campaign-strata",
            label="mixed",
            run_ids=[run.id for run in runs],
            design=None,
            scope_id=runs[0].scope_id,
            subject_id=runs[0].subject_snapshot.subject_id,
        )
        return assemble_context_bundle(
            campaign,
            storage=ToyhostStorage(runs, {run.id: [] for run in runs}, templates=templates),
            profile=toyhost_profile(),
        )

    @staticmethod
    def _templates(*names: dict[str, str]) -> list[EvalTemplate]:
        return [
            make_template(id=f"tpl-{index}", name=f"template {index}", stratum_names=given)
            for index, given in enumerate(names)
        ]

    def test_the_bundle_fingerprint_does_not_read_the_names(self) -> None:
        unnamed = self._assembled(self._templates({}))
        named = self._assembled(self._templates(dict(NAMES)))
        renamed = self._assembled(self._templates({"off_lane": "Elsewhere"}))
        assert named.stratum_names == NAMES
        assert named.fingerprint() == unnamed.fingerprint() == renamed.fingerprint()
        assert evidence_digest(named.to_dict()) == evidence_digest(unnamed.to_dict())

    def test_a_bundle_naming_nothing_is_shape_65_with_no_key_and_naming_adds_only_that_key(self) -> None:
        unnamed = self._assembled(self._templates({})).to_dict()
        named = self._assembled(self._templates(dict(NAMES))).to_dict()
        assert unnamed["schema_version"] == 65 == AnalysisContextBundle.model_fields["schema_version"].default
        assert "stratum_names" not in unnamed
        assert named.pop("stratum_names") == NAMES
        assert named == unnamed

    def test_the_campaign_bundle_shape_does_not_move(self) -> None:
        assert CAMPAIGN_BUNDLE_SHAPE == 3

    def test_two_templates_naming_a_stratum_alike_name_it_and_apart_print_it_by_id(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert self._assembled(self._templates(dict(NAMES), dict(NAMES))).stratum_names == NAMES
        with caplog.at_level("WARNING"):
            apart = self._assembled(self._templates(dict(NAMES), {"off_lane": "Elsewhere"}))
        assert apart.stratum_names == {"assistant_shape": NAMES["assistant_shape"]}
        assert any("stratum_names_differ" in record.getMessage() for record in caplog.records)

    def test_a_name_two_strata_would_share_across_templates_is_kept_by_neither(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            bundle = self._assembled(self._templates({"a": "Same"}, {"b": "Same"}))
        assert bundle.stratum_names == {}
        assert any("stratum_names_clash" in record.getMessage() for record in caplog.records)

    def test_a_template_that_no_longer_resolves_names_nothing(self) -> None:
        assert self._assembled([], runs_of=1).stratum_names == {}


# --- names reach the report input and the page ----------------------------------------------------------------------


def _with_strata(shape: Shape, names: dict[str, str]) -> Shape:
    """``shape`` with its first aspect's cells each read again in a stratum, and the aspect declaring ``names``."""
    first = next(iter(shape.bundles))
    bundles = dict(shape.bundles)
    held = bundles[first]
    cells = [
        cell.model_copy(
            update={
                "strata": [
                    StratumFacts(
                        stratum="off_lane",
                        n_observations=cell.n_observations,
                        n_cases=cell.n_cases or 1,
                        measures=cell.measures,
                        judged=cell.judged,
                    )
                ]
            }
        )
        for cell in held.cell_measures
    ]
    bundles[first] = held.model_copy(update={"cell_measures": cells, "stratum_names": dict(names)})
    return Shape(
        name=shape.name,
        campaign=shape.campaign,
        bundles=bundles,
        reports=shape.reports,
        readings=shape.readings,
        arms=shape.arms,
        runs=shape.runs,
        results=shape.results,
    )


def _input(names: dict[str, str]) -> Any:
    shape = _with_strata(s1_change(), names)
    return report_input(
        build_campaign_bundle(shape.campaign, list(shape.bundles.values()), arms=shape.arms, readings=shape.readings)
    )


class TestADeclaredNameReachesThePage:
    def test_the_report_input_carries_it_as_the_strata_name(self) -> None:
        built = _input({"off_lane": "Off-lane asks"})
        assert [(stratum.id, stratum.name) for stratum in built.strata] == [("off_lane", "Off-lane asks")]

    def test_an_undeclared_stratum_has_no_name_and_a_figure_key_falls_back_to_its_reader_name(self) -> None:
        built = _input({})
        (stratum,) = built.strata
        assert stratum.name is None
        assert resolve("strata.off_lane.name", built).value == "off lane"
        assert reader_name("stratum", "off_lane", built) == "off lane"

    def test_a_figure_key_and_rendered_prose_print_the_declared_name(self) -> None:
        built = _input({"off_lane": "Off-lane asks"})
        assert resolve("strata.off_lane.name", built).value == "Off-lane asks"
        assert render("Worst in {strata.off_lane.name}", built, {}) == "Worst in Off-lane asks"

    def test_a_name_no_stratum_holds_reaches_nothing(self) -> None:
        assert [stratum.name for stratum in _input({"elsewhere": "Elsewhere"}).strata] == [None]

    @staticmethod
    def _headers(names: dict[str, str]) -> list[str]:
        shape = _with_strata(s1_change(), names)
        bundle = next(iter(shape.bundles.values()))
        report = build_code_only_report(bundle, assembled_at="2026-10-08T00:00:00+00:00")
        (table,) = [block for block in report.blocks if isinstance(block, TableBlock) and block.name == "strata"]
        return [column.header for column in table.columns]

    def test_an_aspects_stratum_column_is_headed_by_its_declared_name(self) -> None:
        headers = self._headers({"off_lane": "Off-lane asks"})
        assert "Off-lane asks" in headers and "off_lane" not in headers

    def test_an_undeclared_stratum_column_is_headed_by_its_id(self) -> None:
        headers = self._headers({})
        assert "off_lane" in headers and NO_STRATUM not in headers
