"""The reporter kind's contract and the refusals every reporter launch makes, over a host's prompt and model hooks.

What a host supplies to launch the analysis reporter is only what it knows — which text a prompt preset names,
and which model a name resolves to. Every other refusal is the kind's and is held here: a template asking for
apparatus a memo writer cannot honour; a bank with no live case, or with a (campaign, memo) pair holding two;
a preset the registry does not hold, which must never fall back to the default and bill the incumbent under
the candidate's name; a model named by alias, which the run would record under a name whose target moves;
and the as-recorded replay over a case with no recorded memo to replay.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.analysis import (
    AS_RECORDED_MODEL,
    REPORTER_KIND,
    REPORTER_KIND_CONTRACT,
    ReporterCase,
    ReporterOverlays,
    live_reporter_cases,
    refuse_undeliverable_reporter_template,
    reporter_kind_contract,
)
from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.analysis.reporter_kind import reporter_case_payload
from threetears.evals.kernel import ANALYSIS_GEN_PROMPT, EvalStorage, ValidationFailedError
from threetears.evals.kernel.host import EvalHost, ProfileRegistrationError
from threetears.evals.ops import (
    REPORTER_UNHONOURED_LAUNCH_ARGUMENTS,
    ReporterModelRole,
    admit_reporter_launch,
    plan_reporter_arm,
)
from threetears.evals.run import LaunchGroup, LaunchRequest
from threetears.evals.schema.models import (
    CaseSet,
    ConversationSpec,
    EvalTemplate,
    EvalTestCase,
    RubricDim,
    VariationAxis,
    WorldSeed,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_INSTANT
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

SCOPE = "reporter-scope"
TEMPLATE_ID = "tpl-reporter"
GROUNDED = "reporter.groundedness"

#: The generator model the host resolves for a launch naming none.
GENERATOR_DEFAULT = "writer-large"
#: The judge role's default.
JUDGE_DEFAULT = "judge-large"
#: A name the host resolves to another model: an alias.
ALIAS = "writer-latest"


def _template(**fields: Any) -> EvalTemplate:
    declared: dict[str, Any] = {
        "id": TEMPLATE_ID,
        "scope_id": SCOPE,
        "name": "Memo reporter",
        "intent": "Judge the analysis memo against the evidence its writer read.",
        "candidate_kind": REPORTER_KIND,
        "rubric": [RubricDim(name=GROUNDED, description="Every claim traces to the evidence.", scale="ordinal")],
        "created_at": TOYHOST_INSTANT,
        "updated_at": TOYHOST_INSTANT,
    }
    return EvalTemplate(**(declared | fields))


def _case(*, campaign: str = "camp-1", memo: str | None = "memo-1", supersedes: tuple[str, ...] = ()) -> ReporterCase:
    message = None if memo is None else f"message for {memo}"
    return ReporterCase(
        bundle={"campaign_id": campaign},
        bundle_fingerprint="fp",
        bundle_assembled_at="2026-01-01T00:00:00+00:00",
        recorded_analysis_id=memo,
        recorded_memo=None if memo is None else f"memo of {memo}",
        writer_message=message,
        recorded_writer_message_digest=None if message is None else user_message_digest(message),
        supersedes=list(supersedes),
    )


def _stored(case_id: str, case: ReporterCase | None, *, created_at: str = "2026-01-01T00:00:00+00:00") -> EvalTestCase:
    return EvalTestCase(
        id=case_id,
        scope_id=SCOPE,
        template_id=TEMPLATE_ID,
        host_payload=reporter_case_payload(case) if case is not None else {},
        created_at=created_at,
    )


def _host(*cases: EvalTestCase) -> EvalHost:
    """The toy host, declaring the reporter kind and holding ``cases`` of the reporter template."""
    profile = toyhost_profile()
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(_template())
    for case in cases:
        storage.save_test_case(case)
    reporter = reporter_kind_contract(host_seats=["reviewer_pool"])
    return toyhost_host(profile=replace(profile, kinds=(*profile.kinds, reporter)), storage=storage)


class _Prompts:
    """A prompt registry holding a system default and the named presets — and nothing it does not hold."""

    def __init__(self, presets: dict[str, str] | None = None, *, default: str | None = "the default prompt") -> None:
        self.presets = presets or {}
        self.default = default
        self.asked: list[str | None] = []

    def resolve(self, preset: str | None) -> str | None:
        self.asked.append(preset)
        return self.default if preset is None else self.presets.get(preset)


def _resolve_model(role: ReporterModelRole, named: str | None) -> str:
    """The host's resolution: role defaults, an alias that resolves elsewhere, and a name it refuses."""
    if named is None:
        return GENERATOR_DEFAULT if role == "generator" else JUDGE_DEFAULT
    if named == "unknown-model":
        raise ValidationFailedError(f"no model named {named!r}")
    return "writer-2026" if named == ALIAS else named


def _request(
    *,
    template: EvalTemplate | None = None,
    model: str | None = None,
    preset: str | None = None,
    judge: str | None = None,
    case_set: tuple[CaseSet, tuple[EvalTestCase, ...]] | None = None,
) -> LaunchRequest:
    return LaunchRequest(
        template=template or _template(),
        kind=REPORTER_KIND,
        subject_id=ANALYSIS_GEN_PROMPT,
        candidate_model=model,
        k_runs=1,
        scope_id=SCOPE,
        n_variations=0,
        variation_model=None,
        judge_model=judge,
        judge_config_ids=None,
        simulator_model=None,
        cassette_mode="off",
        cassette_corpus_id=None,
        overlays=REPORTER_KIND_CONTRACT.validate_overlays({} if preset is None else {"prompt_preset": preset}),
        kind_spec=None,
        max_cost_usd=None,
        max_metered_calls=None,
        apparatus_settings={},
        generation_budget=None,
        arm_plan=None,
        arm_price=None,
        launch_group=LaunchGroup(candidate_models=[model or GENERATOR_DEFAULT]),
        settings=TOYHOST_LAUNCH_SETTINGS,
        case_set=case_set[0] if case_set is not None else None,
        case_set_cases=case_set[1] if case_set is not None else (),
    )


def _admit(host: EvalHost, request: LaunchRequest, prompts: _Prompts | None = None) -> Any:
    return admit_reporter_launch(
        request, storage=host.storage, prompts=prompts or _Prompts(), resolve_model=_resolve_model
    )


# =============================================================================
# The contract
# =============================================================================


class TestTheContract:
    def test_its_one_lever_is_the_prompt_preset(self):
        assert REPORTER_KIND_CONTRACT.kind == REPORTER_KIND
        assert REPORTER_KIND_CONTRACT.lever_names == ("analysis_reporter.prompt_preset",)
        assert REPORTER_KIND_CONTRACT.validate_overlays({}) == ReporterOverlays()

    def test_it_seats_the_judge_and_the_spend_ceiling_and_no_simulator(self):
        assert REPORTER_KIND_CONTRACT.seats == frozenset({"judge", "max_cost_usd"})

    def test_a_launch_turning_anything_but_the_preset_is_refused(self):
        with pytest.raises(ValidationFailedError, match="has no overlay temperature"):
            REPORTER_KIND_CONTRACT.validate_overlays({"temperature": 0.2})

    def test_a_blank_preset_is_refused(self):
        with pytest.raises(ValidationFailedError, match="prompt_preset"):
            REPORTER_KIND_CONTRACT.validate_overlays({"prompt_preset": ""})

    def test_a_host_whose_apparatus_a_reporter_run_records_seats_it(self):
        profile = toyhost_profile()
        with pytest.raises(ProfileRegistrationError, match="leaves 'reviewer_pool' unseated"):
            replace(profile, kinds=(*profile.kinds, REPORTER_KIND_CONTRACT))
        seated = reporter_kind_contract(host_seats=["reviewer_pool"])
        assert seated.seats == frozenset({"judge", "max_cost_usd", "reviewer_pool"})
        assert replace(profile, kinds=(*profile.kinds, seated)).kind_contract(REPORTER_KIND) == seated

    def test_it_declines_the_launch_arguments_a_memo_writer_cannot_honour(self):
        assert REPORTER_UNHONOURED_LAUNCH_ARGUMENTS == frozenset({"simulator_model", "cassette_mode", "n_variations"})


# =============================================================================
# The template
# =============================================================================


class TestAnUndeliverableTemplate:
    def test_a_rubric_and_nothing_else_is_deliverable(self):
        refuse_undeliverable_reporter_template(_template())

    @pytest.mark.parametrize(
        ("field", "value", "said"),
        [
            ("conversation", ConversationSpec.model_construct(), "a simulated conversation"),
            ("goal_state_checks", ["world.done == true"], "goal checks"),
            ("tools_allowed", ["search"], "tools"),
            ("tools_required", ["search"], "tools"),
            ("variation_axes", [VariationAxis(name="topic", generator="enum", values=["a"])], "variation axes"),
            ("world_seed", WorldSeed(namespaces={"crm": {"open": 1}}), "a world seed"),
        ],
    )
    def test_apparatus_a_memo_writer_cannot_honour_is_refused(self, field: str, value: Any, said: str):
        template = _template().model_copy(update={field: value})
        with pytest.raises(ValidationFailedError, match=f"declares {said}"):
            refuse_undeliverable_reporter_template(template)

    def test_no_rubric_is_refused(self):
        with pytest.raises(ValidationFailedError, match="no rubric"):
            refuse_undeliverable_reporter_template(_template(rubric=[]))

    def test_every_defect_is_named_at_once(self):
        template = _template(rubric=[]).model_copy(update={"tools_required": ["search"]})
        with pytest.raises(ValidationFailedError, match="tools .*; no rubric"):
            refuse_undeliverable_reporter_template(template)

    def test_an_unrestricted_tool_list_asks_for_no_tool(self):
        refuse_undeliverable_reporter_template(_template().model_copy(update={"tools_allowed": None}))


# =============================================================================
# The bank
# =============================================================================


class TestTheLiveBank:
    def test_the_live_cases_run_oldest_first_and_superseded_or_retired_ones_do_not(self):
        newer = _stored("c-b", _case(campaign="camp-2"), created_at="2026-02-01T00:00:00+00:00")
        older = _stored("c-a", _case(campaign="camp-3"), created_at="2026-01-01T00:00:00+00:00")
        replaced = _stored("c-old", _case())
        revision = _stored("c-new", _case(supersedes=("c-old",)), created_at="2026-03-01T00:00:00+00:00")
        retired = _stored("c-retired", _case(campaign="camp-4")).model_copy(update={"archived": True})
        host = _host(newer, older, replaced, revision, retired)
        live = live_reporter_cases(host.storage, _template(), SCOPE)
        assert [stored.id for stored, _case in live] == ["c-a", "c-b", "c-new"]

    def test_an_empty_bank_is_refused_saying_a_case_is_frozen_never_generated(self):
        with pytest.raises(ValidationFailedError, match="no live reporter case.*frozen from a campaign.*freeze one"):
            live_reporter_cases(_host().storage, _template(), SCOPE)

    def test_a_bank_of_only_retired_cases_is_empty(self):
        retired = _stored("c-1", _case()).model_copy(update={"archived": True})
        with pytest.raises(ValidationFailedError, match="no live reporter case"):
            live_reporter_cases(_host(retired).storage, _template(), SCOPE)

    def test_a_pair_holding_two_live_cases_is_refused_naming_both(self):
        host = _host(_stored("c-1", _case()), _stored("c-2", _case()))
        with pytest.raises(
            ValidationFailedError, match=r"campaign 'camp-1' with recorded memo 'memo-1' \(c-1, c-2\).*supersedes"
        ):
            live_reporter_cases(host.storage, _template(), SCOPE)

    def test_a_case_this_build_cannot_read_is_refused(self):
        unreadable = EvalTestCase(
            id="c-bad", scope_id=SCOPE, template_id=TEMPLATE_ID, host_payload={"reporter_case": {"bundle": "x"}}
        )
        with pytest.raises(ValidationFailedError, match="cannot read"):
            live_reporter_cases(_host(_stored("c-1", _case()), unreadable).storage, _template(), SCOPE)


# =============================================================================
# The launch
# =============================================================================


class TestAdmittingALaunch:
    def test_a_launch_naming_nothing_runs_the_default_prompt_on_the_role_defaults(self):
        host = _host(_stored("c-1", _case()))
        admitted = _admit(host, _request())
        assert (admitted.preset, admitted.prompt) == (None, "the default prompt")
        assert (admitted.candidate_model, admitted.judge_model) == (GENERATOR_DEFAULT, JUDGE_DEFAULT)
        assert [case.id for case in admitted.cases] == ["c-1"]
        assert admitted.generates

    def test_a_named_preset_resolves_to_its_own_text(self):
        prompts = _Prompts({"terse": "write tersely"})
        admitted = _admit(_host(_stored("c-1", _case())), _request(preset="terse"), prompts)
        assert (admitted.preset, admitted.prompt) == ("terse", "write tersely")
        assert prompts.asked == ["terse"]

    def test_a_preset_the_registry_does_not_hold_is_refused_never_resolved_to_the_default(self):
        with pytest.raises(ValidationFailedError, match="prompt_preset 'missing'.*author it first.*incumbent"):
            _admit(_host(_stored("c-1", _case())), _request(preset="missing"))

    def test_a_registry_with_no_default_is_refused(self):
        with pytest.raises(ValidationFailedError, match=f"no system default '{ANALYSIS_GEN_PROMPT}'.*SEED_PROMPTS"):
            _admit(_host(_stored("c-1", _case())), _request(), _Prompts(default=None))

    def test_a_prompt_with_no_text_is_refused(self):
        with pytest.raises(ValidationFailedError, match="resolves to no text"):
            _admit(_host(_stored("c-1", _case())), _request(), _Prompts(default="  "))

    def test_a_model_named_by_alias_is_refused_before_it_is_priced(self):
        with pytest.raises(ValidationFailedError, match=f"'{ALIAS}' is an alias of 'writer-2026'"):
            _admit(_host(_stored("c-1", _case())), _request(model=ALIAS))

    def test_a_model_the_host_refuses_is_refused(self):
        with pytest.raises(ValidationFailedError, match="no model named 'unknown-model'"):
            _admit(_host(_stored("c-1", _case())), _request(model="unknown-model"))

    def test_a_named_model_and_a_pinned_judge_are_the_ones_run(self):
        admitted = _admit(_host(_stored("c-1", _case())), _request(model="writer-2026", judge="judge-small"))
        assert (admitted.candidate_model, admitted.judge_model) == ("writer-2026", "judge-small")

    def test_the_judge_default_steps_off_a_candidate_model_to_the_alternate(self):
        """The engine's own pin rule applies: a role default that is a candidate yields to the alternate."""
        request = _request(model=JUDGE_DEFAULT)
        request = replace(
            request, settings=request.settings.model_copy(update={"judge_alternate_model": "judge-alternate"})
        )
        assert _admit(_host(_stored("c-1", _case())), request).judge_model == "judge-alternate"

    def test_the_replay_runs_with_no_generator(self):
        admitted = _admit(_host(_stored("c-1", _case())), _request(model=AS_RECORDED_MODEL))
        assert admitted.candidate_model == AS_RECORDED_MODEL
        assert not admitted.generates

    def test_the_replay_over_a_case_with_no_recorded_memo_is_refused(self):
        host = _host(_stored("c-1", _case()), _stored("c-2", _case(campaign="camp-2", memo=None)))
        with pytest.raises(ValidationFailedError, match=f"'{AS_RECORDED_MODEL}' replays.*case\\(s\\) c-2 pin none"):
            _admit(host, _request(model=AS_RECORDED_MODEL))

    def test_a_generator_runs_a_case_with_no_recorded_memo(self):
        host = _host(_stored("c-1", _case(memo=None)))
        assert [case.id for case in _admit(host, _request()).cases] == ["c-1"]

    def test_an_undeliverable_template_is_refused_first(self):
        prompts = _Prompts()
        with pytest.raises(ValidationFailedError, match="no rubric"):
            _admit(_host(), _request(template=_template(rubric=[])), prompts)
        assert prompts.asked == [], "the template is refused before the registry is read"

    def test_an_empty_bank_is_refused_at_admission(self):
        with pytest.raises(ValidationFailedError, match="no live reporter case"):
            _admit(_host(), _request())


class TestACaseSet:
    def _set(self, *cases: EvalTestCase) -> tuple[CaseSet, tuple[EvalTestCase, ...]]:
        case_set = CaseSet(
            scope_id=SCOPE,
            name="memos",
            version=1,
            template_id=TEMPLATE_ID,
            test_case_ids=[case.id for case in cases],
            created_at=TOYHOST_INSTANT,
        )
        return case_set, cases

    def test_a_named_set_runs_its_own_cases_in_its_order_whatever_the_bank_says(self):
        first, second = _stored("c-1", _case()), _stored("c-2", _case())
        host = _host(first, second)  # one pair, two live cases: the bank alone would be refused
        admitted = _admit(host, _request(case_set=self._set(second, first)))
        assert [case.id for case in admitted.cases] == ["c-2", "c-1"]

    def test_a_set_case_carrying_no_reporter_case_is_refused(self):
        other = _stored("c-other", None)
        with pytest.raises(ValidationFailedError, match="case set 'memos.*'c-other' \\(carries no reporter case\\)"):
            _admit(_host(other), _request(case_set=self._set(other)))

    def test_the_replay_over_a_set_holds_every_case_to_a_recorded_memo(self):
        bare = _stored("c-bare", _case(memo=None))
        with pytest.raises(ValidationFailedError, match="c-bare pin none"):
            _admit(_host(bare), _request(model=AS_RECORDED_MODEL, case_set=self._set(bare)))


class TestPlanningAnArm:
    def test_the_plan_is_the_admitted_cases_on_the_admitted_model_judged_on_the_memo(self):
        host = _host(_stored("c-1", _case()), _stored("c-2", _case(campaign="camp-2")))
        plan = plan_reporter_arm(host, _request(), prompts=_Prompts(), resolve_model=_resolve_model)
        assert (plan.case_count, plan.candidate_model, plan.simulator_model) == (2, GENERATOR_DEFAULT, None)
        assert plan.judge is not None and plan.judge.model == JUDGE_DEFAULT
        assert set(plan.judge.effective_judges) == {GROUNDED}

    def test_the_plan_makes_every_refusal_admission_makes(self):
        with pytest.raises(ValidationFailedError, match="no live reporter case"):
            plan_reporter_arm(_host(), _request(), prompts=_Prompts(), resolve_model=_resolve_model)
