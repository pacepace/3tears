"""A role whose output must bind says so in its request settings, and the run records it (#686).

The generator sends its contract as a strict ``response_format`` and the judge a reasoning bound, and a
router whose default routing reaches a provider that ignores either returns free text (a paid repair, or a
lost generation) or an unbounded judge. ``ClientRequestSettings.strict_output`` is how the engine says
"route only to a provider honouring every parameter sent"; a host's client builder reads it from the
settings it is handed. Pinned here:

- the generator's settings and the judge's carry it, so a refactor that drops it fails;
- a host's builder reads it off the value it is handed, and it survives a store round trip;
- a stamp stored before the flag reads False, and its recorded apparatus level is unchanged, while a strict
  stamp reads as a different level, so two runs differing only in it are told apart.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.generator import (
    DEFAULT_ANALYSIS_GEN_ANSWER_BUDGET_TOKENS,
    DEFAULT_ANALYSIS_GEN_REASONING_BUDGET_TOKENS,
    analysis_gen_request_settings_for,
)
from threetears.evals.contracts.host.subject import SubjectSnapshot
from threetears.evals.contracts.host.sweepables import SHARED_CORE
from threetears.evals.contracts.models import ClientRequestSettings, EvalRun
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS

_SUBJECT = SubjectSnapshot(subject_id="summarizer-3", subject_label="Summarizer, config 3", state=None)


def _generator_settings() -> ClientRequestSettings:
    return analysis_gen_request_settings_for(
        answer_budget_tokens=DEFAULT_ANALYSIS_GEN_ANSWER_BUDGET_TOKENS,
        reasoning_budget_tokens=DEFAULT_ANALYSIS_GEN_REASONING_BUDGET_TOKENS,
    )


def _level(name: str, settings: ClientRequestSettings) -> Any:
    run = EvalRun(
        apparatus_provenance="commissioned",
        id="run-1",
        scope_id="scope-a",
        subject_snapshot=_SUBJECT,
        candidate_model="writer-a",
        test_case_ids=["c-1"],
        candidate_kind="test-kind",
        k_runs=1,
        rubric_scales={},
        **{name: settings},
    )
    return SHARED_CORE.read_role_pins(run)[name]


class TestTheRolesThatMustBindSayStrict:
    def test_the_generators_settings_ask_for_strict_routing(self) -> None:
        assert _generator_settings().strict_output is True

    def test_the_judges_settings_ask_for_strict_routing(self) -> None:
        assert JUDGE_REQUEST_SETTINGS.strict_output is True

    def test_a_hosts_builder_reads_it_off_the_settings_it_is_handed(self) -> None:
        applied: dict[str, Any] = {}

        def build(settings: ClientRequestSettings) -> None:
            # What a host's client factory does with it: its router's own require-parameters switch.
            applied["provider"] = {"require_parameters": settings.strict_output}

        build(_generator_settings())
        assert applied == {"provider": {"require_parameters": True}}

    def test_it_survives_a_store_round_trip(self) -> None:
        stored = _generator_settings().model_dump(mode="json")
        assert ClientRequestSettings.model_validate(stored).strict_output is True


@pytest.mark.parametrize("name", ["judge_request_settings", "simulator_request_settings"])
class TestTheRecordedLevel:
    def test_a_stamp_stored_before_the_flag_reads_not_strict_at_its_old_level(self, name: str) -> None:
        old = ClientRequestSettings.model_validate({"max_tokens": 4096, "reasoning_max_tokens": 1024})
        assert old.strict_output is False
        assert _level(name, old) == {"max_tokens": 4096, "reasoning_max_tokens": 1024, "reasoning_effort": None}

    def test_two_runs_differing_only_in_it_differ(self, name: str) -> None:
        loose = _level(name, ClientRequestSettings(max_tokens=4096, reasoning_max_tokens=1024))
        strict = _level(name, ClientRequestSettings(max_tokens=4096, reasoning_max_tokens=1024, strict_output=True))
        assert strict["strict_output"] is True
        assert SHARED_CORE.comparability(name, [loose, strict]) == "differs"
