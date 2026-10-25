"""Figure keys: plan prose names figures by their place in the report input, and code writes the numbers.

Contracts held here (``threetears.evals.analysis.report.figure_keys``, GENERATOR.md §6.2):

- every key in the kit's Kairo round 2 plan resolves against r5's input, and renders the figures the reference page
  prints;
- a path is matched against the input's ids, never split at dots, so a dotted arm id or measure key resolves, a path
  that reads two ways is refused naming both, and one that names nothing is refused naming what could come next;
- each filter spells its figure as the grammar says, and refuses a figure it does not suit;
- the derived values (cost uplift, its range, a stratum's loss share, a candidate's flip margin) are computed from the
  input's point figures;
- D3: a digit typed outside a key and outside a name the input holds is found, with its offset;
- the planner prompt's section is written from the module's own families and filters.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.report_kit_support import KIT, chromium_unavailable, rendered
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import (
    figure_key_grammar,
    format_figure,
    keys_in,
    literal_numbers,
    parse_key,
    render,
    resolve,
)
from threetears.evals.analysis.report_input import ReportInput

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "report_plans"
R5_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"
R5_PAGE = KIT / "references" / "r5-kairo-round2.html"


def _input(name: str) -> ReportInput:
    return ReportInput.model_validate_json((FIXTURES / name / "input.json").read_text(encoding="utf-8"))


def _plan() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(R5_PLAN.read_text(encoding="utf-8"))
    return loaded


def _strings(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _r5_names() -> dict[str, str]:
    return {arm.id: arm.name or arm.id for arm in _input("r5").arms}


def _slot(key: str) -> str:
    return next(slot["text"] for slot in _plan()["decision"]["slots"] if slot["key"] == key)


def _rendered(text: str) -> str:
    return render(text, _input("r5"), _r5_names())


def _figure(path: str, filters: str = "", name: str = "r5") -> str:
    return format_figure(resolve(path, _input(name)), tuple(part for part in filters.split("|") if part))


def _small(**parts: Any) -> ReportInput:
    """A minimal valid input, with ``parts`` replacing or adding its top-level fields."""
    base: dict[str, Any] = {
        "schema": "report-input/1",
        "meta": {"campaign": "Small", "kind": "change"},
        "arms": [
            {"id": "base", "role": "control", "settings": {"model": "m-1"}},
            {"id": "a", "role": "challenger", "settings": {"model": "m-2"}},
            {"id": "a.b", "role": "challenger", "settings": {"model": "m-3"}},
        ],
        "aspects": [],
        "scoreboard": {},
        "measures": {},
        "judged": {},
        "contrasts": [],
        "caveats": {},
    }
    return ReportInput.model_validate({**base, **parts})


def _contrast(arm: str, measure: str, delta: float) -> dict[str, Any]:
    return {
        "arm": arm,
        "measure": measure,
        "kind": "trial",
        "verdict": "regressed",
        "delta": delta,
        "interval": [delta - 0.02, delta + 0.01],
    }


# --- the Kairo plan ------------------------------------------------------------------------------------------------


class TestTheKairoPlanResolves:
    def test_every_key_in_its_prose_resolves_and_renders(self):
        report_input = _input("r5")
        keys = [key for text in _strings(_plan()) for key in keys_in(text)]
        assert len(keys) > 30
        for key in keys:
            resolve(key.path, report_input)
        for text in _strings(_plan()):
            assert "{" not in _rendered(text)

    def test_every_claim_and_chart_reference_key_resolves(self):
        plan = _plan()
        report_input = _input("r5")
        claimed = [key for claim in plan["durable_claims"] for key in claim["keys"]]
        referenced = [ref["key"] for chart in [*plan["summary"], *plan["sections"]] for ref in _references(chart)]
        assert claimed and referenced
        for key in [*claimed, *referenced]:
            resolve(parse_key(key).path, report_input)

    def test_the_headline_prints_the_uplift_range_the_page_prints(self):
        assert "Both candidates cost 27–28% more per reply" in _rendered(_plan()["headline"])

    def test_the_lede_prints_the_drop_and_its_interval(self):
        lede = _rendered(_plan()["lede"])
        assert "down 7.6 points (95% interval -10.8 to -3.4)" in lede
        assert "inside the 15-point margin" in lede
        assert lede.startswith("qwen3.8-flash (A) is clearly worse")
        assert "1017 (B)'s drop of 3 points" in lede

    def test_the_recommendation_prints_the_controls_figures(self):
        text = _rendered(_slot("recommendation"))
        assert "the highest pass rate (85.3%)" in text
        assert "the highest in-character rate (91.5%)" in text
        assert "the lowest cost ($1.89 per 1,000 replies)" in text
        assert "(deepseek/deepseek-v4.1-flash)" in text

    def test_why_not_prints_each_candidates_uplift_and_contrast(self):
        text = _rendered(_slot("why_not"))
        assert "28% more per reply, in-character -6 points, pass rate -7.6." in text
        assert "27% more per reply, in-character -4.7 points." in text

    def test_watch_prints_the_reasoning_shares_in_whole_percents(self):
        assert "spent 55% of its output reasoning against September's 76%" in _rendered(_slot("watch"))

    def test_the_strata_headline_rounds_the_loss_share_to_ten(self):
        headline = next(section for section in _plan()["sections"] if section["id"] == "strata")["headline"]
        assert "carry about 50% of each candidate's loss" in _rendered(headline)

    def test_the_masthead_counts(self):
        counts = _rendered(_plan()["masthead"]["counts"])
        assert counts == "September vs 2 candidates · character adherence (v2) · 299 cases × 3 repeats"

    def test_the_figures_match_what_the_reference_page_prints(self):
        """The kit page computes its prose in its script; the figures the keys render appear in what it draws."""
        reason = chromium_unavailable()
        if reason is not None:
            pytest.skip(f"headless Chromium cannot launch: {reason}")
        text, errors = rendered(R5_PAGE, "() => document.body.innerText")
        assert not errors
        for figure in ("27–28%", "7.6", "-10.8 to -3.4", "85.3%", "91.5%", "$1.89", "55%", "76%", "about 50%"):
            assert figure in text, figure
        assert _figure("cost.uplift_range", "pct") in text
        assert _figure("reading.qwen3.8-flash.interval", "pts") in text
        assert _figure("pass_rate.September", "pct") in text


def _references(chart: dict[str, Any]) -> list[dict[str, Any]]:
    inner = chart.get("chart", chart)
    references: list[dict[str, Any]] = inner.get("reference", [])
    return references


# --- matching a path -----------------------------------------------------------------------------------------------


class TestAPathIsMatchedNotSplit:
    def test_an_arm_id_with_dots_resolves(self):
        assert _figure("reasoning_ratio.qwen3.8-flash", "pct") == "68%"
        assert _figure("reading.qwen3.8-flash.change", "pts") == "-7.6"

    def test_a_dimension_key_with_dots_resolves(self):
        assert _figure("judged.persona.holds_character.1017", "pct") == "86.6%"
        assert _figure("contrast.1017.persona.holds_character.delta", "pts") == "-4.7"

    def test_an_aspect_id_with_spaces_and_parentheses_resolves(self):
        assert _figure("aspects.character adherence (v2).margin") == "15"
        assert _figure("aspects.character adherence (v2).reading.1017.change", "pts") == "-3"

    def test_a_dotted_arm_beside_its_prefix_resolves_to_the_one_reading(self):
        report_input = _small(contrasts=[_contrast("a.b", "q", -0.05), _contrast("a", "b.r", -0.03)])
        assert format_figure(resolve("contrast.a.b.q.delta", report_input)) == "-5"
        assert format_figure(resolve("contrast.a.b.r.delta", report_input)) == "-3"

    def test_a_path_that_reads_two_ways_is_refused_naming_both(self):
        report_input = _small(contrasts=[_contrast("a.b", "x", -0.05), _contrast("a", "b.x", -0.03)])
        with pytest.raises(UnresolvableReference) as refused:
            resolve("contrast.a.b.x.delta", report_input)
        message = str(refused.value)
        assert "reads 2 ways" in message
        assert "<arm> `a` · <measure> `b.x`" in message
        assert "<arm> `a.b` · <measure> `x`" in message

    def test_an_unknown_arm_is_refused_naming_the_arms_that_read(self):
        with pytest.raises(UnresolvableReference) as refused:
            resolve("reading.qwen3.9.change", _input("r5"))
        message = str(refused.value)
        assert "after `reading.` comes `qwen3.8-flash`, `1017`" in message

    def test_an_unknown_statistic_is_refused_naming_the_statistics(self):
        with pytest.raises(UnresolvableReference) as refused:
            resolve("reading.1017.chnage", _input("r5"))
        assert "after `reading.1017.` comes `change`, `interval`" in str(refused.value)

    def test_an_unknown_family_is_refused_naming_how_a_key_begins(self):
        with pytest.raises(UnresolvableReference) as refused:
            resolve("contrast_families[0].disclosure", _input("r5"))
        message = str(refused.value)
        assert "a key begins with" in message and "`family`" in message and "`reading`" in message

    def test_a_contrast_names_only_that_arms_measures(self):
        with pytest.raises(UnresolvableReference) as refused:
            resolve("contrast.1017.passed.delta", _input("r5"))
        message = str(refused.value)
        assert "after `contrast.1017.` comes `production_replicating_cost`, `persona.holds_character`" in message
        assert "`passed`" not in message

    def test_a_trailing_dot_or_extra_segment_is_refused(self):
        for path in ("pass_rate.1017.", "pass_rate.1017.mean", "cases.all"):
            with pytest.raises(UnresolvableReference):
                resolve(path, _input("r5"))

    def test_an_arm_key_names_an_arm(self):
        assert resolve("arm:1017", _input("r5")).value == "1017"
        with pytest.raises(UnresolvableReference, match="names no arm"):
            resolve("arm:1018", _input("r5"))

    def test_a_figure_the_input_leaves_unstated_is_refused(self):
        with pytest.raises(UnresolvableReference, match="does not state"):
            resolve("meta.kind_means", _input("r1"))

    def test_the_asked_aspect_must_be_one(self):
        with pytest.raises(UnresolvableReference, match="no single one"):
            resolve("aspect.name", _input("r1"))


# --- filters -------------------------------------------------------------------------------------------------------


class TestFilters:
    def test_pct_reads_a_share_and_a_percent_alike(self):
        assert _figure("judged.persona.holds_character.September", "pct") == "91.5%"
        assert _figure("judged.persona.prod", "pct", name="r4") == "91.7%"

    def test_pts_is_signed_and_one_decimal_at_most(self):
        assert _figure("reading.qwen3.8-flash.change", "pts") == "-7.6"
        assert _figure("contrast.cor.facts.delta", "pts", name="r4") == "+1"
        assert _figure("contrast.f64.task.delta", "pts", name="r4") == "+0.2"

    def test_abs_drops_the_sign(self):
        assert _figure("reading.qwen3.8-flash.change", "abs|pts") == "7.6"
        assert _figure("reading.qwen3.8-flash.interval", "abs|pts") == "3.4 to 10.8"

    def test_usd_reads_dollars_and_cents(self):
        assert _figure("cost_per_1k.qwen3.8-flash", "usd") == "$2.43"
        assert _figure("scoreboard.September.test_cost_usd", "usd") == "$4.85"

    def test_usd_per_1k_states_its_ruler(self):
        assert _figure("scoreboard.September.reply_cost_usd", "usd_per_1k") == "$1.89 per 1,000 replies"
        assert _figure("cost_per_1k.September", "usd_per_1k") == "$1.89 per 1,000 replies"
        assert _figure("contrast.1017.production_replicating_cost.delta", "usd_per_1k") == "+$0.49 per 1,000 replies"

    def test_int_rounds_to_a_whole_number(self):
        assert _figure("flip_margin.qwen3.8-flash", "int") == "11"
        assert _figure("scoreboard.qwen3.8-flash.lost", "int") == "2"

    def test_pct10_rounds_to_the_nearest_ten(self):
        assert _figure("strata.off_lane.loss_share", "pct10") == "50%"

    def test_interval_joins_with_to(self):
        assert _figure("reading.1017.interval", "interval") == "-5.8 to +1.1"
        ci = _figure("measure.production_replicating_cost.September.ci", "interval|usd_per_1k")
        assert ci.startswith("$1.7") and ci.endswith(" per 1,000 replies to $2.03 per 1,000 replies")

    def test_an_unsigned_range_writes_its_unit_once(self):
        assert _figure("cost.uplift_range", "pct") == "27–28%"

    def test_text_reads_as_written(self):
        assert _figure("judge.model") == "openai/gpt-6-luna"
        assert _figure("late_criteria.set_at") == "25 Oct 2026"

    @pytest.mark.parametrize(
        ("path", "filters", "why"),
        [
            ("cost_per_1k.September", "pct", "cannot read as a percent"),
            ("contrast.1017.persona.holds_character.delta", "pct", "reads in points"),
            ("pass_rate.September", "usd", "cannot read as dollars"),
            ("pass_rate.September", "int", "cannot read as a whole number"),
            ("cost_per_1k.September", "pts", "cannot read in points"),
            ("reading.1017.change", "interval", "is one number"),
            ("judge.model", "pct", "takes no filter"),
        ],
    )
    def test_a_filter_that_does_not_suit_the_figure_is_refused(self, path, filters, why):
        with pytest.raises(UnresolvableReference, match=why):
            _figure(path, filters)

    @pytest.mark.parametrize(
        ("inner", "why"),
        [
            ("pass_rate.1017|percent", "names filter 'percent'"),
            ("pass_rate.1017|pct|pct", "repeats a filter"),
            ("pass_rate.1017|pct|pts", "two spellings"),
            ("arm:1017|pct", "takes no filter"),
            ("|pct", "has no path"),
        ],
    )
    def test_a_malformed_key_is_refused(self, inner, why):
        with pytest.raises(UnresolvableReference, match=why):
            parse_key(inner)


# --- derived values ------------------------------------------------------------------------------------------------


class TestDerivedValues:
    def test_cost_uplift_is_the_candidates_cost_over_the_controls(self):
        report_input = _input("r5")
        cost = report_input.measures["production_replicating_cost"].per_arm
        uplift = resolve("cost.uplift.qwen3.8-flash", report_input).value
        assert cost["qwen3.8-flash"].mean is not None and cost["September"].mean is not None
        assert uplift == pytest.approx(cost["qwen3.8-flash"].mean / cost["September"].mean - 1)

    def test_the_uplift_range_spans_the_candidates(self):
        low, high = resolve("cost.uplift_range", _input("r5")).value
        assert (low, high) == (
            pytest.approx(resolve("cost.uplift.1017", _input("r5")).value),
            pytest.approx(resolve("cost.uplift.qwen3.8-flash", _input("r5")).value),
        )

    def test_a_strata_loss_share_is_its_weighted_part_of_each_candidates_loss(self):
        report_input = _input("r5")
        per_arm = {
            arm: resolve(f"strata.off_lane.loss_share.{arm}", report_input).value for arm in ("qwen3.8-flash", "1017")
        }
        assert per_arm["qwen3.8-flash"] == pytest.approx(0.4756, abs=1e-4)
        assert per_arm["1017"] == pytest.approx(0.5281, abs=1e-4)
        assert resolve("strata.off_lane.loss_share", report_input).value == pytest.approx(per_arm["qwen3.8-flash"])

    def test_a_flip_margin_is_the_readings_worst_drop(self):
        assert _figure("flip_margin.qwen3.8-flash") == "10.8"
        assert _figure("flip_margin.1017") == "5.8"

    def test_the_uplift_needs_a_control(self):
        with pytest.raises(UnresolvableReference):
            resolve("cost.uplift_range", _input("r1"))


# --- rendering -----------------------------------------------------------------------------------------------------


class TestRender:
    def test_an_arm_is_named_by_its_short_name_and_tag(self):
        report_input = _input("r4")
        text = render("{arm:prod} against {arm:f64}", report_input, {"f64": "flash · 64k out"})
        assert text == "Production (P) against flash · 64k out (A)"

    def test_double_braces_are_the_memos_grammar_and_refused(self):
        with pytest.raises(UnresolvableReference, match="double brace"):
            keys_in("costs {{c1|cost|measure|mean}}")

    def test_a_stray_brace_is_refused(self):
        with pytest.raises(UnresolvableReference, match="opens or closes no figure key"):
            keys_in("costs {pass_rate.1017 more")

    def test_keys_carry_their_spans(self):
        text = "up {pass_rate.1017|pct} today"
        (key,) = keys_in(text)
        assert key.span is not None and text[key.span[0] : key.span[1]] == "{pass_rate.1017|pct}"
        assert key.filters == ("pct",)


# --- D3: no number typed outside a key -----------------------------------------------------------------------------


class TestLiteralNumbers:
    def test_a_dotted_arm_id_with_digits_passes(self):
        assert literal_numbers("qwen3.8-flash is worse, and qwen3.8-flash's cost is higher.", _input("r5")) == []

    def test_a_bare_all_digit_arm_id_fails(self):
        text = "1017 could be noise"
        assert literal_numbers(text, _input("r5")) == [(0, 4)]

    def test_the_arm_form_of_an_all_digit_id_passes(self):
        assert literal_numbers("{arm:1017} could be noise", _input("r5")) == []

    def test_a_typed_percent_fails(self):
        text = "a 95% interval"
        assert literal_numbers(text, _input("r5")) == [(2, 4)]

    def test_a_decimal_is_one_offence(self):
        assert literal_numbers("at 99.69 percent", _input("r5")) == [(3, 8)]

    def test_an_arabic_indic_digit_fails(self):
        text = "it costs ٣ more"
        assert literal_numbers(text, _input("r5")) == [(9, 10)]

    def test_digits_inside_keys_pass(self):
        text = "down {reading.qwen3.8-flash.change|abs|pts} points, {cost.uplift.1017|pct} more"
        assert literal_numbers(text, _input("r5")) == []

    def test_a_settings_display_value_passes(self):
        assert literal_numbers("f64 runs with 64k out and f16 with 16k.", _input("r4")) == []

    def test_a_name_must_be_bounded_on_both_sides(self):
        assert literal_numbers("164k out", _input("r4")) == [(0, 3)]
        assert literal_numbers("64kb out", _input("r4")) == [(0, 2)]

    def test_served_models_and_trial_ids_pass(self):
        text = "deepseek/deepseek-v4.1-flash-1017 failed T15 on oft-match-score"
        assert literal_numbers(text, _input("r5")) == []


# --- the grammar ---------------------------------------------------------------------------------------------------


class TestTheGrammar:
    def test_it_teaches_every_filter_with_a_rendered_example(self):
        grammar = figure_key_grammar()
        for name in ("pct", "pts", "abs", "usd", "int", "pct10", "interval", "usd_per_1k"):
            assert f"- `{name}`:" in grammar, name
        assert '`{pass_rate.base|pct}` reads "85.3%"' in grammar

    def test_it_teaches_every_family_the_design_names(self):
        grammar = figure_key_grammar()
        for form in (
            "meta.<field>",
            "control.<field>",
            "n_candidates",
            "aspect.<field>",
            "aspects.<aspect>.<field>",
            "reading.<arm>.<statistic>",
            "contrast.<arm>.<measure>.<statistic>",
            "judged.<dimension>.<arm>",
            "pass_rate.<arm>",
            "cost_per_1k.<arm>",
            "measure.<measure>.<arm>.<statistic>",
            "scoreboard.<arm>.<statistic>",
            "reasoning_ratio.<arm>",
            "cases",
            "repeats",
            "late_criteria.set_at",
            "family.<field>",
            "judge.<field>",
            "caveats.latency",
            "cost.uplift.<candidate>",
            "cost.uplift_range",
            "strata.<stratum>.loss_share",
            "flip_margin.<candidate>",
        ):
            assert f"- `{form}`:" in grammar, form

    def test_it_states_the_literal_number_rule(self):
        grammar = figure_key_grammar()
        assert "write `{arm:1017}`, not `1017`" in grammar
        assert "never split at dots" in grammar

    def test_its_worked_example_is_rendered(self):
        assert '"fast.v2 (A) is down 4.3 points (-7.1 to -1.4)."' in figure_key_grammar()
