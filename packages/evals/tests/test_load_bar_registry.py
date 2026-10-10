"""A host's bars and pass thresholds, loaded from a declaration kept as data with ``load_bar_registry``.

The loader is strict because the failure it prevents is silent: a misspelt ``rationale`` or ``higher_is_better``
ignored by a lenient reader registers a standard other than the one its author wrote. So an unknown field, a
missing one and a value of the wrong type are each refused by where it sits in the declaration, and what the
registry itself refuses (a pair declared twice, a pass threshold outside 2-5) is still refused through it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.kernel.host import (
    DEFAULT_PASS_THRESHOLD,
    Bar,
    BarRegistrationError,
    PassThreshold,
    load_bar_registry,
)
from packages.evals.tests.fixtures.toyhost import toyhost_profile


def _bar(**overrides: Any) -> dict[str, Any]:
    declared: dict[str, Any] = {
        "behavior": "extract",
        "measure": "field_accuracy",
        "threshold": 0.92,
        "higher_is_better": True,
        "rationale": "the incumbent's lower bound, adopted after review",
    }
    return {**declared, **overrides}


def _pass_threshold(**overrides: Any) -> dict[str, Any]:
    declared: dict[str, Any] = {"behavior": "extract", "threshold": 4, "rationale": "only a good extraction is usable"}
    return {**declared, **overrides}


def test_a_declaration_loads_into_the_registry_code_would_build():
    registry = load_bar_registry({"bars": [_bar()], "pass_thresholds": [_pass_threshold()]})
    assert registry.bars == (
        Bar(
            behavior="extract",
            measure="field_accuracy",
            threshold=0.92,
            higher_is_better=True,
            rationale="the incumbent's lower bound, adopted after review",
        ),
    )
    assert registry.pass_thresholds == (
        PassThreshold(behavior="extract", threshold=4, rationale="only a good extraction is usable"),
    )
    assert registry.pass_threshold("extract") == 4


def test_an_empty_declaration_is_an_empty_registry():
    registry = load_bar_registry({})
    assert registry.bars == ()
    assert registry.pass_threshold("anything") == DEFAULT_PASS_THRESHOLD


def test_a_declaration_of_pass_thresholds_alone_loads():
    assert load_bar_registry({"pass_thresholds": [_pass_threshold()]}).bars == ()


def test_an_adopted_proposal_keeps_its_vacuous_flag():
    (bar,) = load_bar_registry({"bars": [_bar(vacuous_seed=True)]}).bars
    assert bar.vacuous_seed is True


def test_an_integer_threshold_reads_as_the_number_it_is():
    (bar,) = load_bar_registry({"bars": [_bar(threshold=1)]}).bars
    assert bar.threshold == 1.0


class TestTheShapeIsReadStrictly:
    """Each defect is named by its place in the declaration, and every one at once."""

    def test_an_unknown_top_level_field_is_refused(self):
        with pytest.raises(BarRegistrationError, match="thresholds: is not a field a bar declaration has"):
            load_bar_registry({"bars": [], "thresholds": []})

    def test_a_misspelt_field_on_a_bar_is_refused_rather_than_dropped(self):
        declared = _bar()
        declared["rationle"] = declared.pop("rationale")
        with pytest.raises(BarRegistrationError) as refused:
            load_bar_registry({"bars": [declared]})
        assert "bars.0.rationale: is required" in str(refused.value)
        assert "bars.0.rationle: is not a field a bar declaration has" in str(refused.value)

    def test_an_unknown_field_on_a_pass_threshold_is_refused(self):
        with pytest.raises(BarRegistrationError, match="pass_thresholds.0.scale: is not a field"):
            load_bar_registry({"pass_thresholds": [_pass_threshold(scale=5)]})

    @pytest.mark.parametrize(
        ("field", "value"),
        [("threshold", "0.92"), ("threshold", True), ("higher_is_better", "yes"), ("behavior", 7)],
    )
    def test_a_bar_value_of_the_wrong_type_is_refused_not_coerced(self, field: str, value: object):
        with pytest.raises(BarRegistrationError, match=rf"bars\.0\.{field}: .*got {value!r}"):
            load_bar_registry({"bars": [_bar(**{field: value})]})

    def test_a_threshold_that_is_not_finite_is_refused(self):
        with pytest.raises(BarRegistrationError, match=r"bars\.0\.threshold: input should be a finite number"):
            load_bar_registry({"bars": [_bar(threshold=float("inf"))]})

    @pytest.mark.parametrize("value", [4.0, True, "4"])
    def test_a_pass_threshold_that_is_not_a_whole_level_is_refused(self, value: object):
        with pytest.raises(
            BarRegistrationError, match=r"pass_thresholds\.0\.threshold: input should be a valid integer"
        ):
            load_bar_registry({"pass_thresholds": [_pass_threshold(threshold=value)]})

    def test_a_list_that_is_not_a_list_is_refused(self):
        with pytest.raises(BarRegistrationError, match="bars: input should be a valid list"):
            load_bar_registry({"bars": _bar()})

    def test_every_defect_is_named_at_once(self):
        with pytest.raises(BarRegistrationError) as refused:
            load_bar_registry({"bars": [_bar(threshold="x"), _bar(extra=1)], "pass_thresholds": [{}]})
        message = str(refused.value)
        for place in ("bars.0.threshold", "bars.1.extra", "pass_thresholds.0.behavior", "pass_thresholds.0.rationale"):
            assert place in message


class TestTheRegistrysOwnRefusalsStillHold:
    def test_a_pair_declared_twice_is_refused(self):
        with pytest.raises(BarRegistrationError, match="extract/field_accuracy is declared twice"):
            load_bar_registry({"bars": [_bar(), _bar(threshold=0.95)]})

    def test_a_blank_rationale_is_refused(self):
        with pytest.raises(BarRegistrationError, match="states no rationale"):
            load_bar_registry({"bars": [_bar(rationale="  ")]})

    def test_a_pass_threshold_of_one_is_refused(self):
        with pytest.raises(BarRegistrationError, match="outside 2-5"):
            load_bar_registry({"pass_thresholds": [_pass_threshold(threshold=1)]})


def test_a_loaded_registry_is_held_to_the_hosts_measures_like_one_built_in_code():
    """The profile checks a loaded registry against its measures exactly as it checks one built in code."""
    toy = toyhost_profile()
    (registered, *_rest) = toy.bars.bars
    declared = {
        "bars": [
            {
                "behavior": registered.behavior,
                "measure": registered.measure,
                "threshold": registered.threshold,
                "higher_is_better": not registered.higher_is_better,
                "rationale": registered.rationale,
            }
        ]
    }
    with pytest.raises(BarRegistrationError, match="the descriptor owns the fact"):
        replace(toy, bars=load_bar_registry(declared))
    agreeing = {"bars": [{**declared["bars"][0], "higher_is_better": registered.higher_is_better}]}
    assert replace(toy, bars=load_bar_registry(agreeing)).bars.bars == (replace(registered, vacuous_seed=False),)
