"""How the judge and the simulated user were asked, read through the engine's own core registry.

``judge_request_settings`` and ``simulator_request_settings`` belong to the shared core: each role's output cap and reasoning parameter is an apparatus input, pinned into its role, and
a run that never recorded one cannot be compared on it. This file reads them through
:data:`~threetears.evals.kernel.host.sweepables.SHARED_CORE` — the registry every host extends — over run
documents with no host vocabulary. No host profile is read, and nothing is imported from a
host adapter.

The last class pins the version pair those two moved. It uses the core's own
apparatus declarations, the set every host starts from, so no host's registrations are read.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.cells import CELL_MODEL_VERSION
from threetears.evals.schema.subject import SubjectSnapshot
from threetears.evals.kernel.host.sweepables import CORE_SWEEPABLES, JUDGE_INPUTS, SHARED_CORE, SIMULATOR_INPUTS
from threetears.evals.schema.models import ClientRequestSettings, EvalRun


_SUBJECT = SubjectSnapshot(subject_id="summarizer-3", subject_label="Summarizer, config 3", state=None)

#: Each role's apparatus dimension, the run field it reads (what the role was ASKED for), and the role's pins.
ROLES = [
    ("judge_request_settings", "judge_asked_settings", JUDGE_INPUTS),
    ("simulator_request_settings", "simulator_asked_settings", SIMULATOR_INPUTS),
]


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


@pytest.mark.parametrize(("name", "field", "pins"), ROLES)
class TestHowARoleWasAskedIsAnApparatusInput:
    def test_it_is_a_core_apparatus_declaration_that_means_unrecorded_when_blank(
        self, name: str, field: str, pins: tuple[str, ...]
    ) -> None:
        (declared,) = [d for d in CORE_SWEEPABLES if d.name == name]
        assert declared.role == "apparatus"
        assert declared.indeterminate_when_blank is True

    def test_it_is_pinned_into_its_role(self, name: str, field: str, pins: tuple[str, ...]) -> None:
        assert name in pins

    def test_a_recorded_setting_is_read_as_a_json_level(self, name: str, field: str, pins: tuple[str, ...]) -> None:
        run = _run(**{field: ClientRequestSettings(max_tokens=4096, reasoning_max_tokens=1024)})
        assert _read(name, run) == {"max_tokens": 4096, "reasoning_max_tokens": 1024, "reasoning_effort": None}

    def test_a_recorded_effort_is_read_as_a_json_level(self, name: str, field: str, pins: tuple[str, ...]) -> None:
        run = _run(**{field: ClientRequestSettings(max_tokens=5120, reasoning_effort="minimal")})
        assert _read(name, run) == {"max_tokens": 5120, "reasoning_max_tokens": None, "reasoning_effort": "minimal"}

    def test_no_reasoning_parameter_is_a_recorded_level_not_a_blank(
        self, name: str, field: str, pins: tuple[str, ...]
    ) -> None:
        run = _run(**{field: ClientRequestSettings(max_tokens=4096)})
        assert _read(name, run) == {"max_tokens": 4096, "reasoning_max_tokens": None, "reasoning_effort": None}

    def test_a_budget_and_an_effort_under_one_cap_differ(self, name: str, field: str, pins: tuple[str, ...]) -> None:
        """The simulator moved from a token budget to an effort level under the same 5120 cap; a campaign pooling
        runs from both sides must be told the role was asked differently."""
        budget = _read(name, _run(**{field: ClientRequestSettings(max_tokens=5120, reasoning_max_tokens=1024)}))
        effort = _read(name, _run(**{field: ClientRequestSettings(max_tokens=5120, reasoning_effort="minimal")}))

        assert SHARED_CORE.comparability(name, [budget, effort]) == "differs"

    def test_an_unstamped_run_reads_as_unrecorded(self, name: str, field: str, pins: tuple[str, ...]) -> None:
        assert _read(name, _run()) is None

    def test_two_runs_asked_differently_differ_and_alike_agree(
        self, name: str, field: str, pins: tuple[str, ...]
    ) -> None:
        wide = _read(name, _run(**{field: ClientRequestSettings(max_tokens=8192, reasoning_max_tokens=4096)}))
        narrow = _read(name, _run(**{field: ClientRequestSettings(max_tokens=2048)}))

        assert SHARED_CORE.comparability(name, [wide, narrow]) == "differs"
        assert SHARED_CORE.comparability(name, [wide, dict(wide)]) == "same"

    def test_a_run_that_never_recorded_it_cannot_be_compared_on_it(
        self, name: str, field: str, pins: tuple[str, ...]
    ) -> None:
        stamped = _read(name, _run(**{field: ClientRequestSettings(max_tokens=2048)}))

        assert SHARED_CORE.comparability(name, [stamped, None]) == "unknown"
        assert SHARED_CORE.comparability(name, [None, None]) == "unknown"


#: The core's apparatus declarations at each ``(schema_version, CELL_MODEL_VERSION)`` pair, oldest
#: first. Append a row for every bump; never edit one.
_CORE_V24 = frozenset({"judge_config_ids", "judge_dim_divergence", "judge_model", "max_cost_usd", "simulator_model"})
CORE_PINNED: tuple[tuple[tuple[int, int], frozenset[str]], ...] = (
    # 50/12: the apparatus set as first pinned for the baseline — every core apparatus dimension a cell id digests,
    # the judge's request settings and temperature among them. Bundle shape 50 was the baseline until the control's
    # earlier cells joined the bundle; no release shipped it, and the row stays as the earlier end the 51 row is
    # checked against.
    ((50, 12), _CORE_V24 | {"judge_request_settings", "simulator_request_settings", "judge_temperature"}),
    # 51/12: the bundle gains `RealizedDesign.earlier_control` and `FrontierPoint.earlier_control`. The row repeats
    # the 50 set on purpose: the apparatus set and the cell model hold, and `test_a_set_change_moves_both_versions`
    # reads that as a shape-only bump (the bundle version moved, the cell model stayed). No release shipped it.
    ((51, 12), _CORE_V24 | {"judge_request_settings", "simulator_request_settings", "judge_temperature"}),
    # 52/12: the first public release's shape — the earlier control's disclosure says a marginal row counts its
    # cells at the control's level, a fingerprinted value moved over unchanged evidence. Another shape-only bump:
    # the set and the cell model hold. It is the last row, so it is also the one the current versions are checked
    # against.
    ((52, 12), _CORE_V24 | {"judge_request_settings", "simulator_request_settings", "judge_temperature"}),
    # 53/13: what the judge and the simulated user were actually SENT joins the apparatus beside what they were asked,
    # so cells pool only when both match; the set moved, so the cell model moves with the bundle.
    (
        (53, 13),
        _CORE_V24
        | {
            "judge_request_settings",
            "simulator_request_settings",
            "judge_temperature",
            "judge_sent_settings",
            "simulator_sent_settings",
        },
    ),
    # 54/13: the bundle counts each arm's exclusions by cause (a cut-off judge reply as `truncated`) and flags a
    # differential one, a question asked after the first run no longer confirms anything, and the bundle gains each
    # arm's census of its trials (`scoreboard`), the trials themselves and the build that assembled it. A shape-only
    # bump: the apparatus set and the cell model hold.
    (
        (54, 13),
        _CORE_V24
        | {
            "judge_request_settings",
            "simulator_request_settings",
            "judge_temperature",
            "judge_sent_settings",
            "simulator_sent_settings",
        },
    ),
    # 55/13: each family comparison carries its bounded interval, and the overall pass rate is compared as a reading
    # of its own in the family of each question whose decision rule has a rate clause. A shape-only bump: the
    # apparatus set and the cell model hold.
    (
        (55, 13),
        _CORE_V24
        | {
            "judge_request_settings",
            "simulator_request_settings",
            "judge_temperature",
            "judge_sent_settings",
            "simulator_sent_settings",
        },
    ),
    # 56/13: a question's decision rule is typed on the declared design where it was parsed from its words. A
    # shape-only bump: the apparatus set and the cell model hold.
    (
        (56, 13),
        _CORE_V24
        | {
            "judge_request_settings",
            "simulator_request_settings",
            "judge_temperature",
            "judge_sent_settings",
            "simulator_sent_settings",
        },
    ),
    # 57/13: a lever read off what ran that fewer than two results exercised is no level (`unexercised_factors`). A
    # shape-only bump: the apparatus set and the cell model hold.
    (
        (57, 13),
        _CORE_V24
        | {
            "judge_request_settings",
            "simulator_request_settings",
            "judge_temperature",
            "judge_sent_settings",
            "simulator_sent_settings",
        },
    ),
)


class TestTheCoresApparatusSetIsPinnedToTheVersionsThatNameIt:
    """A core apparatus declaration joining moves every bundle and cell that applies it, so both versions move."""

    def test_the_current_versions_are_the_last_pinned_row(self) -> None:
        current = (AnalysisContextBundle.model_fields["schema_version"].default, CELL_MODEL_VERSION)
        assert CORE_PINNED[-1][0] == current

    def test_the_cores_apparatus_declarations_are_the_pinned_set(self) -> None:
        assert frozenset(d.name for d in CORE_SWEEPABLES if d.role == "apparatus") == CORE_PINNED[-1][1]

    def test_a_set_change_moves_both_versions(self) -> None:
        """Every row after the baseline moves the bundle version, and the cell model too when the set moved."""
        for ((schema_before, cell_before), set_before), ((schema_after, cell_after), set_after) in zip(
            CORE_PINNED, CORE_PINNED[1:], strict=False
        ):
            assert schema_after > schema_before
            if set_after != set_before:
                assert cell_after > cell_before
