"""Measurement-condition covariates + phase-timing folding.

The rule under test throughout is that an unmeasured covariate has **no key**. Most of
these cases exist to prove a specific way that rule could be violated — a sum that folds
None to 0, a ratio that reads "no split reported" as "did no thinking", a probe nobody
wired reading as "nothing else was running" — because each of those failures produces a
plausible number that a marginal would then treat as an observation.
"""

from __future__ import annotations

import pytest

from threetears.evals.contracts.covariates import (
    count_dropped_tool_calls,
    count_refused_tool_attaches,
    count_truncated_rounds,
    derive_covariates,
    fold_phase_timings,
)
from threetears.evals.contracts.models import RoleUsage


def _candidate(**kwargs) -> RoleUsage:
    return RoleUsage(role="candidate", model=kwargs.pop("model", "cand-1"), **kwargs)


# ---------------------------------------------------------------------------
# execution_mode — the concurrent-run probe
# ---------------------------------------------------------------------------


class TestExecutionMode:
    def test_one_running_run_is_serial(self):
        """The probe counts this run too, so 1 means nothing else was executing."""
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=1)
        assert covariates["execution_mode"] == "serial"

    def test_a_second_run_makes_the_cell_concurrent(self):
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=2)
        assert covariates["execution_mode"] == "concurrent"

    def test_an_unwired_probe_records_nothing_rather_than_serial(self):
        """ "Nobody looked" must not become the clean-conditions claim.

        This is the covariate's whole failure mode: a cell measured under unknown
        conditions that reports ``serial`` pools silently with genuinely-isolated cells,
        and the contamination the covariate exists to expose becomes invisible.
        """
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert "execution_mode" not in covariates


# ---------------------------------------------------------------------------
# context_tokens_in / reasoning_ratio — read off the candidate rows
# ---------------------------------------------------------------------------


class TestTokenCovariates:
    def test_context_tokens_sum_across_the_candidate_rows(self):
        """Rows are keyed by (role, model), so reading row[0] would undercount a cell
        whose candidate spent on more than one model."""
        covariates = derive_covariates(
            usage=[
                _candidate(model="a", prompt_tokens=100, completion_tokens=10),
                _candidate(model="b", prompt_tokens=50, completion_tokens=5),
            ],
            concurrent_eval_jobs=None,
        )
        assert covariates["context_tokens_in"] == 150

    def test_other_roles_do_not_leak_into_the_candidate_context(self):
        """The covariate is the context the CANDIDATE faced — a judge's prompt is
        measurement apparatus and would inflate it."""
        covariates = derive_covariates(
            usage=[
                _candidate(prompt_tokens=100, completion_tokens=10),
                RoleUsage(role="judge", model="j", prompt_tokens=900, completion_tokens=9),
            ],
            concurrent_eval_jobs=None,
        )
        assert covariates["context_tokens_in"] == 100

    def test_a_candidate_that_reported_no_prompt_tokens_records_nothing(self):
        covariates = derive_covariates(
            usage=[_candidate(prompt_tokens=None, completion_tokens=None)],
            concurrent_eval_jobs=None,
        )
        assert "context_tokens_in" not in covariates

    def test_reasoning_ratio_is_the_reasoning_share_of_generation(self):
        covariates = derive_covariates(
            usage=[_candidate(prompt_tokens=100, completion_tokens=80, reasoning_tokens=20)],
            concurrent_eval_jobs=None,
        )
        assert covariates["reasoning_ratio"] == pytest.approx(0.25)

    def test_an_unreported_split_leaves_the_ratio_absent_not_zero(self):
        """0.0 would assert the model did no thinking. The provider merely didn't say."""
        covariates = derive_covariates(
            usage=[_candidate(prompt_tokens=100, completion_tokens=80, reasoning_tokens=None)],
            concurrent_eval_jobs=None,
        )
        assert "reasoning_ratio" not in covariates

    def test_a_reported_zero_split_is_a_real_zero_ratio(self):
        """The mirror of the case above: a non-reasoning model reporting 0 IS an
        observation, and dropping it would lose the distinction the field exists for."""
        covariates = derive_covariates(
            usage=[_candidate(prompt_tokens=100, completion_tokens=80, reasoning_tokens=0)],
            concurrent_eval_jobs=None,
        )
        assert covariates["reasoning_ratio"] == 0.0

    def test_no_generation_means_no_ratio(self):
        """A candidate turn that errored before generating has no denominator; a
        ZeroDivisionError or a 0.0 would both be wrong."""
        covariates = derive_covariates(
            usage=[_candidate(prompt_tokens=100, completion_tokens=0, reasoning_tokens=0)],
            concurrent_eval_jobs=None,
        )
        assert "reasoning_ratio" not in covariates

    def test_a_result_whose_capture_attributed_no_roles_derives_an_empty_mapping(self):
        """``usage=[]`` — capture ran and found no roles — supports no covariate."""
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert covariates == {}


# ---------------------------------------------------------------------------
# dropped_tool_calls — the candidate's reach for a tool it was not given
# ---------------------------------------------------------------------------


def _candidate_turn(
    dropped: list[str] | None,
    refused: list[str] | None = None,
    truncated: int | None = None,
) -> dict:
    """A trace entry shaped like the one _record_candidate_turn appends.

    Every field is omitted rather than emptied when not supplied, because "the record does
    not carry this field" is a real population — a turn loop that does not stamp it — and it
    is the one the absent-versus-zero rule has to get right.
    """
    record: dict = {"turn_number": 0}
    if dropped is not None:
        record["dropped_tool_calls"] = dropped
    if refused is not None:
        record["refused_tool_attaches"] = refused
    if truncated is not None:
        record["truncated_rounds"] = truncated
    return {"turn": 0, "role": "candidate", "content": "…", "turn_record": record}


class TestDroppedToolCalls:
    def test_a_reach_for_a_bounded_away_tool_is_counted(self):
        """The defect: a candidate reaching past tools_allowed left no mark on the result."""
        trace = [_candidate_turn(["lookup"]), _candidate_turn([])]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            dropped_tool_calls=count_dropped_tool_calls(trace),
        )
        assert covariates["dropped_tool_calls"] == 1

    def test_a_candidate_that_never_reached_reports_a_measured_zero(self):
        """0 is the point of the key: silence must not read the same as never trying."""
        trace = [_candidate_turn([]), _candidate_turn([])]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            dropped_tool_calls=count_dropped_tool_calls(trace),
        )
        assert covariates["dropped_tool_calls"] == 0

    def test_no_candidate_turn_observed_yields_no_key(self):
        """A degraded cell that never ran a candidate turn measured nothing here."""
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert "dropped_tool_calls" not in covariates

    def test_counting_spans_every_candidate_turn_and_keeps_repeats(self):
        """Drain-delivery turns append through the same path; two reaches count as two."""
        trace = [_candidate_turn(["lookup", "lookup"]), _candidate_turn(["image_gen"])]
        assert count_dropped_tool_calls(trace) == 3

    def test_simulator_entries_and_unstamped_records_contribute_nothing(self):
        """Simulator turns carry no turn_record, and a turn loop that does not stamp the field has no list."""
        trace = [
            {"turn": 0, "role": "simulator", "content": "find something"},
            _candidate_turn(None),
            _candidate_turn(["lookup"]),
        ]
        assert count_dropped_tool_calls(trace) == 1

    def test_a_trace_with_no_candidate_turn_measured_nothing(self):
        """The load-bearing zero must not be published for a cell nobody watched.

        The defect: a cell whose first ``process_input`` raised breaks out of the turn loop
        before any record reaches the trace, and still builds a full ``EvalResult`` — so it
        recorded ``dropped_tool_calls=0``, "watched and clean", against the key's own
        docstring promising an ABSENT key there. An analyst pooling the covariate read
        crashed cells as well-behaved ones.
        """
        assert count_dropped_tool_calls([]) is None
        assert count_dropped_tool_calls([{"turn": 0, "role": "simulator", "content": "hi"}]) is None

    def test_the_absent_count_produces_no_key(self):
        """End to end: the sparse rule is what turns the None into silence."""
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            dropped_tool_calls=count_dropped_tool_calls([]),
        )
        assert "dropped_tool_calls" not in covariates


# ---------------------------------------------------------------------------
# refused_tool_attaches — the candidate's reach for a tool it was not given,
# on the ATTACH side, which the harness refuses rather than the client dropping
# ---------------------------------------------------------------------------


class TestRefusedToolAttaches:
    def test_a_refused_self_attach_is_counted(self):
        """The defect: a refusal reached the candidate and no read surface counted it."""
        trace = [_candidate_turn([], refused=["knowledge"]), _candidate_turn([], refused=[])]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            refused_tool_attaches=count_refused_tool_attaches(trace),
        )
        assert covariates["refused_tool_attaches"] == 1

    def test_repeats_count_because_asking_twice_is_a_different_observation(self):
        """The reported run asked for `knowledge` AND `web_search` and was refused both."""
        trace = [_candidate_turn([], refused=["knowledge", "web_search"]), _candidate_turn([], refused=["knowledge"])]
        assert count_refused_tool_attaches(trace) == 3

    def test_a_candidate_that_never_asked_reports_a_measured_zero(self):
        """0 is the point of the key: silence must not read the same as never asking."""
        trace = [_candidate_turn([], refused=[]), _candidate_turn([], refused=[])]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            refused_tool_attaches=count_refused_tool_attaches(trace),
        )
        assert covariates["refused_tool_attaches"] == 0

    def test_a_record_that_does_not_report_the_field_measured_nothing(self):
        """A turn loop that does not stamp the field measured nothing. Publishing 0 for it would
        assert an observation the harness never made — the same conflation the sibling key
        exists to end.
        """
        trace = [_candidate_turn(["lookup"]), _candidate_turn([])]
        assert count_refused_tool_attaches(trace) is None
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            refused_tool_attaches=count_refused_tool_attaches(trace),
        )
        assert "refused_tool_attaches" not in covariates

    def test_a_mixed_trace_counts_only_the_turns_that_reported(self):
        """One reporting turn is enough to make the cell measured; the silent one adds 0."""
        trace = [_candidate_turn([]), _candidate_turn([], refused=["web_search"])]
        assert count_refused_tool_attaches(trace) == 1

    def test_no_candidate_turn_at_all_yields_no_key(self):
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert "refused_tool_attaches" not in covariates

    def test_the_two_bound_covariates_are_independent(self):
        """A dropped CALL and a refused ATTACH are different reaches and count separately."""
        trace = [_candidate_turn(["lookup"], refused=["knowledge"])]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            dropped_tool_calls=count_dropped_tool_calls(trace),
            refused_tool_attaches=count_refused_tool_attaches(trace),
        )
        assert covariates["dropped_tool_calls"] == 1
        assert covariates["refused_tool_attaches"] == 1


# ---------------------------------------------------------------------------
# truncated_rounds — the PROVIDER's intervention, not the harness's: a round
# cut off at the output cap, which the loop exits on exactly as it would on
# a model that finished
# ---------------------------------------------------------------------------


class TestTruncatedRounds:
    def test_a_round_cut_at_the_output_cap_is_counted(self):
        """The defect: the cap ended the turn, the judge read the silence as the candidate's
        decision, and no eval read surface carried the finish_reason that said otherwise."""
        trace = [_candidate_turn([], truncated=1), _candidate_turn([], truncated=0)]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            truncated_rounds=count_truncated_rounds(trace),
        )
        assert covariates["truncated_rounds"] == 1

    def test_a_cell_none_of_whose_rounds_were_cut_reports_a_measured_zero(self):
        """0 is the point of the key: a model that chose silence must not read like one the
        cap silenced, and only a recorded zero lets the first claim be made."""
        trace = [_candidate_turn([], truncated=0), _candidate_turn([], truncated=0)]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            truncated_rounds=count_truncated_rounds(trace),
        )
        assert covariates["truncated_rounds"] == 0

    def test_counting_sums_over_every_turn_that_reported(self):
        trace = [_candidate_turn([], truncated=1), _candidate_turn([], truncated=1), _candidate_turn([], truncated=0)]
        assert count_truncated_rounds(trace) == 2

    def test_a_record_that_does_not_report_the_field_measured_nothing(self):
        """A turn loop that does not stamp the field was not watched for this; publishing 0
        for it would assert an observation the harness never made."""
        trace = [_candidate_turn(["lookup"]), _candidate_turn([], refused=[])]
        assert count_truncated_rounds(trace) is None
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            truncated_rounds=count_truncated_rounds(trace),
        )
        assert "truncated_rounds" not in covariates

    def test_a_mixed_trace_counts_only_the_turns_that_reported(self):
        trace = [_candidate_turn([]), _candidate_turn([], truncated=1)]
        assert count_truncated_rounds(trace) == 1

    def test_no_candidate_turn_at_all_yields_no_key(self):
        assert count_truncated_rounds([]) is None
        assert count_truncated_rounds([{"turn": 0, "role": "simulator", "content": "hi"}]) is None
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert "truncated_rounds" not in covariates

    def test_the_three_mechanical_counters_are_independent(self):
        """A dropped CALL, a refused ATTACH and a CUT round are three different events."""
        trace = [_candidate_turn(["lookup"], refused=["knowledge"], truncated=1)]
        covariates = derive_covariates(
            usage=[],
            concurrent_eval_jobs=None,
            dropped_tool_calls=count_dropped_tool_calls(trace),
            refused_tool_attaches=count_refused_tool_attaches(trace),
            truncated_rounds=count_truncated_rounds(trace),
        )
        assert covariates["dropped_tool_calls"] == 1
        assert covariates["refused_tool_attaches"] == 1
        assert covariates["truncated_rounds"] == 1


# ---------------------------------------------------------------------------
# fold_phase_timings — the delivery carrier
# ---------------------------------------------------------------------------


class TestFoldPhaseTimings:
    def test_phases_are_namespaced_by_their_source_tool(self):
        """Two async tools may both call a phase 'synthesis'; without the prefix they
        would silently sum into one meaningless number."""
        acc: dict[str, float] = {}
        fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": 120.0})
        fold_phase_timings(acc, source_tool="rss", timings={"synthesis": 5.0})
        assert acc == {"lookup_synthesis_ms": 120.0, "rss_synthesis_ms": 5.0}

    def test_repeat_deliveries_sum_per_phase(self):
        """A tool can re-fire within a cell; the covariate is wall-clock attributable
        to that phase across the whole cell, so the second window must not overwrite."""
        acc: dict[str, float] = {}
        fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": 100.0})
        fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": 50.5})
        assert acc == {"lookup_synthesis_ms": 150.5}

    def test_a_phase_reported_as_none_stays_unmeasured(self):
        acc: dict[str, float] = {}
        fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": None, "grounding": 10.0})
        assert acc == {"lookup_grounding_ms": 10.0}

    def test_a_delivery_with_no_timings_leaves_the_accumulator_untouched(self):
        acc: dict[str, float] = {"lookup_synthesis_ms": 7.0}
        fold_phase_timings(acc, source_tool="lookup", timings=None)
        assert acc == {"lookup_synthesis_ms": 7.0}

    def test_a_non_mapping_payload_is_dropped_loudly(self, caplog):
        """Metadata crosses an untyped process-internal boundary; a malformed carrier
        must not wedge the cell, but it must not vanish silently either."""
        acc: dict[str, float] = {}
        with caplog.at_level("WARNING"):
            fold_phase_timings(acc, source_tool="lookup", timings=[1, 2, 3])
        assert acc == {}
        assert "non-mapping" in caplog.text

    def test_an_unparseable_duration_is_dropped_loudly(self, caplog):
        acc: dict[str, float] = {}
        with caplog.at_level("WARNING"):
            fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": "fast"})
        assert acc == {}
        assert "not a number" in caplog.text

    def test_a_negative_duration_is_not_a_measurement(self, caplog):
        """Nothing takes negative time; folding it would silently subtract from a real
        phase total on the next delivery."""
        acc: dict[str, float] = {}
        with caplog.at_level("WARNING"):
            fold_phase_timings(acc, source_tool="lookup", timings={"synthesis": -5.0})
        assert acc == {}
        assert "negative" in caplog.text


# ---------------------------------------------------------------------------
# turns_ended_by_budget — the HOST's own bound: a turn production would have
# ended is ended in the cell, and the count is reported by the kind, since an
# ended turn leaves no record to count
# ---------------------------------------------------------------------------


class TestTurnsEndedByBudget:
    @pytest.mark.parametrize("count", [0, 2])
    def test_a_watched_count_is_written_zero_included(self, count):
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None, turns_ended_by_budget=count)
        assert covariates["turns_ended_by_budget"] == count

    def test_no_budget_watched_yields_no_key(self):
        covariates = derive_covariates(usage=[], concurrent_eval_jobs=None)
        assert "turns_ended_by_budget" not in covariates
