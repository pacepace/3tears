"""A judged host for the jobs conformance kit: a quick callable kind, judged through the host's own client factory.

The toy host's kind is graded by code and calls no model, so a run of it would leave every spend action's job with
nothing to ask: the kit's positive control would call it vacuous, rightly. This host's kind is judged — through the
:class:`~threetears.evals.kernel.host.EvalHost`'s client factory, which the kit wraps — on a two-dimension rubric,
by a scripted judge answering every dimension 3 (inside the scale, so a temperature comparison has borderline cases).
The analysis writer answers with a payload the generator refuses: a generation's job still calls it, which is all the
kit asks of it. Harness support, not part of any fixture host.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.kernel import default_job_timeout
from threetears.evals.actions import Caller, ConformanceSample
from threetears.evals.kernel.host import CompletionClients
from threetears.evals.ops import AnalysisGeneration, OpsHost
from threetears.evals import Judge
from threetears.evals.quick.one_call import JUDGED_CALLABLE_KIND, callable_host, callable_kind
from threetears.evals.run import (
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    build_judge_service,
    launch_run,
)
from threetears.evals.schema import (
    DocumentStore,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    RubricDim,
    SubjectSnapshot,
)
from threetears.evals.testing import CompletionGuard, JobsConformanceHost
from packages.evals.tests.off_loop import off_loop

__all__ = ["CONFORMANCE_SCOPE", "ScriptedClients", "conformance_host"]

#: The scope every conformance case runs in.
CONFORMANCE_SCOPE = "conformance"
_TEMPLATE = "tmpl-judged"
_SUBJECT = "subject-judged"
_CANDIDATE = "candidate-a"
_JUDGE = "judge/first"
_SECOND = "judge/second"
#: Judged 1-5, so the judge's 3 is a borderline score.
_RUBRIC = [
    RubricDim(name="answer.faithful", description="Says only what the case supports.", scale="ordinal"),
    RubricDim(name="answer.clear", description="Reads plainly.", scale="ordinal"),
]


@dataclass(frozen=True)
class _Completion:
    """A completion as eval reads one."""

    content: str
    model: str
    served_model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    temperature: float | None = None


@dataclass
class _ScriptedClient:
    """One role's client: a judge scores every dimension it is asked 3; any other role answers ``{}``."""

    role: str
    model_name: str
    temperature: float | None = None

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        """The most one call costs."""
        return 0.01

    async def generate(self, *, system: str, user: str, response_format: Any = None, **_: Any) -> _Completion:
        """Answer as the role would: a judge's score of the one dimension it was asked, or an empty object."""
        content = "{}"
        if self.role == "judge" and (asked := re.search(r'the single key "(.+?)"', system)) is not None:
            content = json.dumps({"reasoning": "read it", "criteria_scores": {asked.group(1): 3}})
        return _Completion(
            content=content, model=self.model_name, served_model=self.model_name, temperature=self.temperature
        )

    async def aclose(self) -> None:
        """Nothing to release."""


@dataclass
class ScriptedClients:
    """The host's client factory: a scripted client per role and model, every one it built remembered."""

    built: list[_ScriptedClient] = field(default_factory=list)

    def __call__(self, role: Any, model: str | None, *, temperature: float | None = None) -> _ScriptedClient:
        """Build one client."""
        client = _ScriptedClient(str(role), model or f"{role}/default", temperature)
        self.built.append(client)
        return client


async def _candidate(case: Mapping[str, Any]) -> str:
    return f"the answer to {case['n']}"


def conformance_host(
    guard: CompletionGuard, clients: CompletionClients | None = None, *, store: DocumentStore | None = None
) -> JobsConformanceHost:
    """A fresh judged host for one conformance case, its client factory wrapped by ``guard``.

    :param guard: The case's guard.
    :ptype guard: CompletionGuard
    :param clients: The factory to wrap; ``None`` for a fresh :class:`ScriptedClients`.
    :ptype clients: CompletionClients | None
    :param store: The document store under the host; ``None`` for a fresh in-memory one.
    :ptype store: DocumentStore | None
    :return: The host, the caller and the sample launch every engine happy path builds on.
    :rtype: JobsConformanceHost
    """
    factory = guard.wrap(clients if clients is not None else ScriptedClients())
    eval_host = callable_host(clients=factory, store=store)
    judge = Judge(_ScriptedClient("judge", _JUDGE), _RUBRIC)
    template = EvalTemplate(
        id=_TEMPLATE,
        scope_id=CONFORMANCE_SCOPE,
        name="a judged callable",
        intent="Answer the case.",
        candidate_kind=JUDGED_CALLABLE_KIND,
        rubric=list(judge.dims),
    )
    cases = [
        EvalTestCase(
            id=f"{_TEMPLATE}-{n}",
            scope_id=CONFORMANCE_SCOPE,
            template_id=_TEMPLATE,
            variation_params={"n": str(n)},
            host_payload={"case": {"n": n}},
        )
        for n in range(2)
    ]
    eval_host.storage.save_template(template)
    for case in cases:
        eval_host.storage.save_test_case(case)
    kind = callable_kind(_candidate, judge=judge)

    async def launch(request: LaunchRequest) -> EvalRun:
        run_judge = off_loop(
            build_judge_service,
            eval_host,
            request.template,
            request.judge_model or _JUDGE,
            judged_artifact=JudgedArtifact.DOCUMENT,
        )
        subject = SubjectSnapshot(subject_id=request.subject_id, subject_label=request.subject_id, state={})
        return await launch_run(
            launching,
            request,
            KindWiring(kind_factory=lambda _cell: kind, subject=subject, test_cases=cases, judge=run_judge),
        )

    settings = LaunchSettings(
        max_launch_arms=2,
        max_admitted_runs=4,
        judge_concurrency=1,
        enforcement_enabled=False,
        max_cost_usd=1.0,
        max_metered_calls=None,
        max_out_of_run_cost_usd=1.0,
    )
    launching = LaunchHost(
        eval_host=eval_host,
        kinds={
            JUDGED_CALLABLE_KIND: LaunchableKind(
                launch=launch,
                resolve_subject=lambda _subject, _scope: None,
                unhonoured_launch_arguments=frozenset(
                    {"simulator_model", "judge_config_ids", "n_variations", "cassette_mode"}
                ),
            )
        },
        settings=lambda: settings,
        job_timeout_factory=default_job_timeout,
    )

    async def prompt() -> str:
        return "Write the analysis."

    host = OpsHost(
        launch=launching,
        generation=AnalysisGeneration(
            prompt_id="conformance", resolve_prompt=prompt, max_output_tokens=1000, budget_s=60.0
        ),
    )
    return JobsConformanceHost(
        host=host,
        caller=Caller(scope_id=CONFORMANCE_SCOPE, identity="agent:conformance"),
        sample=ConformanceSample(
            template_id=_TEMPLATE, subject_id=_SUBJECT, models=(_CANDIDATE,), second_judge_model=_SECOND
        ),
    )
