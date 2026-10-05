"""A measure family is open: a host declares its own, and says whether code or a judge produced the number.

The engine's six families are named constants (:data:`~threetears.evals.contracts.ENGINE_FAMILIES`). A
host whose measure is a kind of number none of them names declares a
:class:`~threetears.evals.contracts.MeasureFamily` on its measure registry, and its ``graded_by`` decides
what the engine does with it: a ``code`` family ranks and may be held to a bar, exactly as ``mechanical``
does; a ``judge`` family is described and never ranked, exactly as ``rubric`` is.

The toy host reports ``field_accuracy`` in its own family, ``extraction_grade``, end to end: through the
bundle's catalogue and cells and through its bar.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import (
    ENGINE_FAMILIES,
    MeasureFamily,
    MetricDescriptor,
    list_metrics,
)
from threetears.evals.contracts.declaration import UnreadableBarName, resolve_bar_name
from threetears.evals.contracts.host import MeasureRegistrationError, MeasureRegistry
from threetears.evals.contracts.metrics import CODE_GRADED_FAMILIES, is_code_graded
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.kind import FIELD_ACCURACY
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_EXTRACTION_FAMILY, toyhost_profile


def _descriptor(name: str, family: str) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        data_type="numeric",
        family=family,
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=f"{name}, for the test.",
        higher_is_better=True,
        value_range=(0.0, 1.0),
    )


_TABLE_FEEL = MeasureFamily(name="table_feel", graded_by="judge", description="How a table felt, as a judge read it.")


# --- the toy host reports in its own family, end to end --------------------------------------------


def test_the_toy_host_reports_a_measure_in_its_own_family() -> None:
    bundle = toyhost_bundle()

    assert bundle.measure_catalog[FIELD_ACCURACY].family == TOYHOST_EXTRACTION_FAMILY.name
    for cell in bundle.cell_measures:
        assert any(measure.name == FIELD_ACCURACY for measure in cell.measures.measures), cell.variant_key


def test_a_bar_on_a_code_graded_host_family_is_adjudicated() -> None:
    bundle = toyhost_bundle()

    (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == FIELD_ACCURACY]
    assert bar.state == "adjudicated", bar.reason


def test_the_catalogue_lists_a_host_family() -> None:
    measures = toyhost_profile().measures
    assert [d.name for d in list_metrics(measures, family=TOYHOST_EXTRACTION_FAMILY.name)] == [FIELD_ACCURACY]


# --- graded_by decides whether a host family ranks ---------------------------------------------------


def test_a_judge_graded_host_family_is_described_and_never_ranked() -> None:
    measures = MeasureRegistry([_descriptor("table_feel_score", _TABLE_FEEL.name)], families=(_TABLE_FEEL,))

    assert not is_code_graded(measures.get("table_feel_score"), measures)  # type: ignore[arg-type]
    refused = resolve_bar_name("table_feel_score", rubric_dimensions={}, goal_state_checks=(), measures=measures)
    assert isinstance(refused, UnreadableBarName)


def test_a_code_graded_host_family_ranks_like_mechanical() -> None:
    family = MeasureFamily(name="tally", graded_by="code", description="A count code took.")
    measures = MeasureRegistry([_descriptor("tally_rate", family.name)], families=(family,))

    assert is_code_graded(measures.get("tally_rate"), measures)  # type: ignore[arg-type]


def test_the_engines_families_are_named_constants_and_three_are_code_graded() -> None:
    assert set(ENGINE_FAMILIES) == {"mechanical", "classifier", "goal_state", "rubric", "dual_axis", "composite"}
    assert CODE_GRADED_FAMILIES == {"mechanical", "classifier", "goal_state"}


# --- what a host may not declare ---------------------------------------------------------------------


def test_a_measure_naming_a_family_nobody_declared_is_refused() -> None:
    with pytest.raises(MeasureRegistrationError, match="names family 'extraction_grde', which neither"):
        MeasureRegistry([_descriptor("field_accuracy", "extraction_grde")], families=(TOYHOST_EXTRACTION_FAMILY,))


def test_a_host_family_reusing_an_engine_name_is_refused() -> None:
    with pytest.raises(MeasureRegistrationError, match="family mechanical is one of the engine's own"):
        MeasureRegistry([], families=(MeasureFamily(name="mechanical", graded_by="judge", description="x"),))


def test_a_family_declared_twice_is_refused() -> None:
    with pytest.raises(MeasureRegistrationError, match="family tally is declared twice"):
        MeasureRegistry(
            [],
            families=(
                MeasureFamily(name="tally", graded_by="code", description="x"),
                MeasureFamily(name="tally", graded_by="judge", description="y"),
            ),
        )


@pytest.mark.parametrize("name", ["", "Extraction", "extraction grade", "1st"])
def test_a_family_name_is_a_lowercase_identifier(name: str) -> None:
    with pytest.raises(ValidationError):
        MeasureFamily(name=name, graded_by="code", description="x")
    with pytest.raises(ValidationError):
        _descriptor("m", name)
