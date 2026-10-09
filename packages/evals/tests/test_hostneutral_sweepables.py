"""How the judge and the simulated user were asked, read through the engine's own core registry.

``judge_request_settings`` and ``simulator_request_settings`` belong to the shared core: each role's output cap and reasoning parameter is an apparatus input, pinned into its role, and
a run that never recorded one cannot be compared on it. This file reads them through
:data:`~threetears.evals.contracts.host.sweepables.SHARED_CORE` — the registry every host extends — over run
documents with no host vocabulary. No host profile is read, and nothing is imported from a
host adapter.

The last class pins the version pair those two moved. It uses the core's own
apparatus declarations, the set every host starts from, so no host's registrations are read.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.bundle import AnalysisContextBundle
from threetears.evals.analysis.cells import CELL_MODEL_VERSION
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.sweepables import CORE_SWEEPABLES, JUDGE_INPUTS, SHARED_CORE, SIMULATOR_INPUTS
from threetears.evals.contracts.models import ClientRequestSettings, EvalRun


_SUBJECT = SubjectSnapshot(subject_id="summarizer-3", subject_label="Summarizer, config 3", state=None)

ROLES = [("judge_request_settings", JUDGE_INPUTS), ("simulator_request_settings", SIMULATOR_INPUTS)]


def _run(**fields: Any) -> EvalRun:
    return EvalRun(
        apparatus_provenance="commissioned",
        id="run-1",
        scope_id="scope-a",
        subject_snapshot=_SUBJECT,
        candidate_model="writer-a",
        test_case_ids=["c-1"],
        **{"candidate_kind": "test-kind", "k_runs": 1, "rubric_scales": {}, **fields},
    )


def _read(name: str, run: EvalRun) -> Any:
    return SHARED_CORE.read_role_pins(run)[name]


@pytest.mark.parametrize(("name", "pins"), ROLES)
class TestHowARoleWasAskedIsAnApparatusInput:
    def test_it_is_a_core_apparatus_declaration_that_means_unrecorded_when_blank(
        self, name: str, pins: tuple[str, ...]
    ) -> None:
        (declared,) = [d for d in CORE_SWEEPABLES if d.name == name]
        assert declared.role == "apparatus"
        assert declared.indeterminate_when_blank is True

    def test_it_is_pinned_into_its_role(self, name: str, pins: tuple[str, ...]) -> None:
        assert name in pins

    def test_a_recorded_setting_is_read_as_a_json_level(self, name: str, pins: tuple[str, ...]) -> None:
        run = _run(**{name: ClientRequestSettings(max_tokens=4096, reasoning_max_tokens=1024)})
        assert _read(name, run) == {"max_tokens": 4096, "reasoning_max_tokens": 1024, "reasoning_effort": None}

    def test_a_recorded_effort_is_read_as_a_json_level(self, name: str, pins: tuple[str, ...]) -> None:
        run = _run(**{name: ClientRequestSettings(max_tokens=5120, reasoning_effort="minimal")})
        assert _read(name, run) == {"max_tokens": 5120, "reasoning_max_tokens": None, "reasoning_effort": "minimal"}

    def test_no_reasoning_parameter_is_a_recorded_level_not_a_blank(self, name: str, pins: tuple[str, ...]) -> None:
        run = _run(**{name: ClientRequestSettings(max_tokens=4096)})
        assert _read(name, run) == {"max_tokens": 4096, "reasoning_max_tokens": None, "reasoning_effort": None}

    def test_a_budget_and_an_effort_under_one_cap_differ(self, name: str, pins: tuple[str, ...]) -> None:
        """The simulator moved from a token budget to an effort level under the same 5120 cap; a campaign pooling
        runs from both sides must be told the role was asked differently."""
        budget = _read(name, _run(**{name: ClientRequestSettings(max_tokens=5120, reasoning_max_tokens=1024)}))
        effort = _read(name, _run(**{name: ClientRequestSettings(max_tokens=5120, reasoning_effort="minimal")}))

        assert SHARED_CORE.comparability(name, [budget, effort]) == "differs"

    def test_an_unstamped_run_reads_as_unrecorded(self, name: str, pins: tuple[str, ...]) -> None:
        assert _read(name, _run()) is None

    def test_two_runs_asked_differently_differ_and_alike_agree(self, name: str, pins: tuple[str, ...]) -> None:
        wide = _read(name, _run(**{name: ClientRequestSettings(max_tokens=8192, reasoning_max_tokens=4096)}))
        narrow = _read(name, _run(**{name: ClientRequestSettings(max_tokens=2048)}))

        assert SHARED_CORE.comparability(name, [wide, narrow]) == "differs"
        assert SHARED_CORE.comparability(name, [wide, dict(wide)]) == "same"

    def test_a_run_that_never_recorded_it_cannot_be_compared_on_it(self, name: str, pins: tuple[str, ...]) -> None:
        stamped = _read(name, _run(**{name: ClientRequestSettings(max_tokens=2048)}))

        assert SHARED_CORE.comparability(name, [stamped, None]) == "unknown"
        assert SHARED_CORE.comparability(name, [None, None]) == "unknown"


#: The core's apparatus declarations at each ``(schema_version, CELL_MODEL_VERSION)`` pair, oldest
#: first. Append a row for every bump; never edit one.
_CORE_V24 = frozenset({"judge_config_ids", "judge_dim_divergence", "judge_model", "max_cost_usd", "simulator_model"})
CORE_PINNED: tuple[tuple[tuple[int, int], frozenset[str]], ...] = (
    ((24, 5), _CORE_V24),
    ((25, 6), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 26 moved the writer's view of the bundle, not the apparatus partition.
    ((26, 6), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 27 keyed the design on arms, not the apparatus partition.
    ((27, 6), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 28/7 added a HOST state level; the core's apparatus set is unchanged.
    ((28, 7), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 29 typed an unresolved case set in the comparison lens, not the apparatus partition.
    ((29, 7), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 30/8 added two HOST world dimensions; the core's apparatus set is unchanged.
    ((30, 8), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 31/9: the three session world dimensions are a host's, so the core's apparatus set is unchanged.
    ((31, 9), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 32/10: provenance entered the apparatus class id and the reader fields joined the bundle; the
    # core's apparatus DECLARATIONS are unchanged.
    ((32, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 33/10: judge-versus-human agreement joined the bundle; the apparatus partition is unchanged.
    ((33, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 34/10: the corrected families of comparisons joined the bundle; the apparatus partition is unchanged.
    ((34, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 35/10: the time axis joined the bundle; the apparatus partition is unchanged.
    ((35, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 36/10: a date time axis states why it is not builds; the apparatus partition is unchanged.
    ((36, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 37/10: the judge's self-agreement and the evidence tiers it and calibration decide; the apparatus
    # partition is unchanged.
    ((37, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 38/10: a judge's tier is keyed by its config too, and its agreements count distinct results; the
    # apparatus partition is unchanged.
    ((38, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 39/10: a role's request settings may name a reasoning effort, so every recorded settings level gains
    # that key and the simulator's confound reads differently; the apparatus partition is unchanged.
    ((39, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 40/10: each cell carries its figures per stratum of its cases; the apparatus partition is unchanged.
    ((40, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 41/10: each coverage row checks the mechanism its lever declares it acts on, comparisons name an observed
    # mechanism that diverged between their levels, and each arm carries its reasoning share; the apparatus
    # partition is unchanged.
    ((41, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 42/10: a cost no result observed is no reading, and the bundle names the cells it went unmeasured in; the
    # apparatus partition is unchanged.
    ((42, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 43/10: a resolved surface folded into a fixed knob without a check is named as an `unverified_fold`
    # confound; the apparatus partition is unchanged.
    ((43, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 44/10: a cost or latency reading leaves out the results that delivered no turn, each cell counts its
    # candidate failures, and the bundle names the cells where no result delivered a turn; the apparatus
    # partition is unchanged.
    ((44, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 45/10: the frontier's pass^k is renamed `pass_hat_k`, estimated without bias and carried with its curve
    # and subject depth (#591); the apparatus partition is unchanged.
    ((45, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 46/10: guardrails (boundary judged dimensions and measures declared one) are decided apart from every
    # comparison family, and the readings no declared question asks about are labelled exploratory; the apparatus
    # partition is unchanged.
    ((46, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
    # 47/10: the declaration's `controls` is renamed `held_fixed`, and the bundle's `controls_reading` with it
    # (`held_fixed_reading`); the apparatus partition is unchanged.
    ((47, 10), _CORE_V24 | {"judge_request_settings", "simulator_request_settings"}),
)


class TestTheCoresApparatusSetIsPinnedToTheVersionsThatNameIt:
    """A core apparatus declaration joining moves every bundle and cell that applies it, so both versions move."""

    def test_the_current_versions_are_the_last_pinned_row(self) -> None:
        current = (AnalysisContextBundle.model_fields["schema_version"].default, CELL_MODEL_VERSION)
        assert CORE_PINNED[-1][0] == current

    def test_the_cores_apparatus_declarations_are_the_pinned_set(self) -> None:
        assert frozenset(d.name for d in CORE_SWEEPABLES if d.role == "apparatus") == CORE_PINNED[-1][1]

    @pytest.mark.parametrize("index", range(1, len(CORE_PINNED)))
    def test_a_set_change_moves_both_versions(self, index: int) -> None:
        (schema_before, cell_before), set_before = CORE_PINNED[index - 1]
        (schema_after, cell_after), set_after = CORE_PINNED[index]
        assert schema_after > schema_before
        if set_after != set_before:
            assert cell_after > cell_before
