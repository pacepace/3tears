"""A campaign shaped like production's largest, for measuring the report and planning reads (W4b).

The campaign the hot-path oracle, the scaling test and ``scripts/profile_hot_paths.py`` all read: 41 runs of one
template over several arms (a control and its candidates), about 3,100 results at ``scale=1`` with two repeats per
case, each result about 7 KB stored, its bytes split across its fields as production's rows split theirs (rubric
scores with per-criterion reasoning the largest share, then usage, the transcript and outcome scores, the kind's
payload, the goal checks, the judge's reasoning, the host's measures and the judge identities, and the judge's request
settings repeated on every score). ``scale`` multiplies the cases each run measures, never the fields of a result, so
a doubling doubles the rows and nothing else.

Everything is stored through the real :class:`~threetears.evals.kernel.storage.EvalStorage` over the reference
:class:`~threetears.evals.storage.InMemoryDocumentStore`, so a read pays the hydration a host's store would make it
pay. Every value comes from a seeded generator and every instant is fixed, so two builds are the same corpus byte for
byte and the bundles assembled from them fingerprint alike.

The counting half (:data:`OPERATIONS`, :func:`counting_operations`) counts the operations the scaling test
bounds, by wrapping the production callables in place for the length of a ``with`` block: no source changes, and
the wrapped callable still runs.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any
from unittest import mock

from pydantic import BaseModel

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.campaign_bundle import CampaignBundle, build_campaign_bundle
from threetears.evals.analysis.reporting import ScoreRecord
from threetears.evals.kernel.declaration import AspectDesign, CampaignArm, ControlDeclaration, SweptAxis
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.identity import resolve_variant_identity
from threetears.evals.kernel.metrics import describe_measure
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.kernel.usage_capture import blended_cost_roles
from threetears.evals.schema import JudgedArtifact, JudgeEvidence, SweepableValue
from threetears.evals.schema import hashing
from threetears.evals.schema.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    ClientRequestSettings,
    EvalResult,
    EvalRun,
    GoalStateOutcome,
    RoleUsage,
    RubricDim,
    RubricScore,
    builtin_judge_config_id,
    fingerprint_criterion,
)
from threetears.evals.ops import report_read
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    declared,
    make_aspect_campaign,
    make_eval_run,
    make_eval_trace,
    make_subject,
    make_template,
    make_test_case,
)
from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE, courier_host, run_courier_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, generate, ops_fixture, settled

__all__ = [
    "ARM_MODELS",
    "CASES_PER_RUN",
    "HOT_PATH_INSTANT",
    "HOT_PATH_SCOPE",
    "K_RUNS",
    "N_RUNS",
    "HotPathCorpus",
    "OPERATIONS",
    "campaign_id_of",
    "counting_operations",
    "courierhost_capture",
    "courierhost_from",
    "dumped",
    "loaded",
    "toyhost_doc_capture",
    "toyhost_from",
    "field_byte_shares",
    "growth",
    "hot_path_corpus",
    "report_read_counts",
]

#: The scope every document of the corpus lives in.
HOT_PATH_SCOPE = "hot-path"

#: The instant the corpus was measured from: every run's creation and every result's scoring is an offset of it.
HOT_PATH_INSTANT = "2026-09-01T08:00:00+00:00"

#: The arms: the control first, then its candidates. One arm is one candidate model, as a model sweep's arms are.
ARM_MODELS = ("control-model", "candidate-a", "candidate-b", "candidate-c")

#: The runs, dealt to the arms in turn, so each group of four runs measures one slice of cases under every arm.
N_RUNS = 41

#: The cases each run measures at ``scale=1``: 41 runs x 38 cases x 2 repeats is 3,116 results.
CASES_PER_RUN = 38

#: The repeats per case.
K_RUNS = 2

#: The rubric criteria every result is judged on, beside the transcript and outcome scores.
CRITERIA = ("reply.accuracy", "reply.grounding")

#: The goal checks every result is held to.
GOAL_CHECKS = (
    "state.ticket.status == 'resolved'",
    "state.ticket.tags.length >= 1",
    "state.refund.amount <= state.order.total",
)

#: Each arm's underlying quality, on the 1-5 scale: the candidates straddle the control.
_ARM_QUALITY = (3.5, 3.7, 3.2, 3.9)

#: The judge model every result names, and what it was sent (repeated on every score, as production stores it).
_JUDGE_MODEL = "judge-model"
_JUDGE_SENT = ClientRequestSettings(
    max_tokens=10240, reasoning_effort="high", reasoning_max_tokens=None, strict_output=True
)

#: Words the reasoning text is drawn from: what a judge's reasoning reads like, enough that no two results repeat.
_WORDS = (
    "the reply addresses customer's question directly but omits refund window policy cited in "
    "ticket history agent confirms order number before acting and tone stays courteous throughout "
    "although second message repeats earlier apology grounding claims match retrieved article except "
    "shipping estimate which is not supported by any source provided response length reasonable"
).split()


@dataclass(frozen=True)
class HotPathCorpus:
    """One built corpus: the host whose store holds it, the campaign over it, and its size.

    :ivar host: The toy host over an in-memory store holding every run, result, trace, case and template.
    :ivar campaign: The stored campaign: one aspect over every run, or two (``two_aspects``), each over its own
        template's runs.
    :ivar scope_id: The scope it is stored in.
    :ivar template_ids: The templates the runs ran, in aspect order.
    :ivar n_results: How many results were stored.
    :ivar scale: The case multiplier it was built at.
    """

    host: EvalHost
    campaign: EvalCampaign
    scope_id: str
    template_ids: tuple[str, ...]
    n_results: int
    scale: int


def _instant(seconds: int) -> str:
    """:data:`HOT_PATH_INSTANT` plus ``seconds``, spelled as the corpus spells every instant."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    days, hours = divmod(8 + hours, 24)
    return f"2026-09-{1 + days:02d}T{hours:02d}:{minutes:02d}:{secs:02d}+00:00"


def _prose(rng: random.Random, length: int) -> str:
    """Seeded reasoning text of about ``length`` bytes."""
    words: list[str] = []
    size = 0
    while size < length:
        word = rng.choice(_WORDS)
        words.append(word)
        size += len(word) + 1
    return " ".join(words)[:length]


def _reasoning_length(rng: random.Random) -> int:
    """A per-criterion reasoning length whose median is about 277 bytes and 95th percentile about 741.

    Log-normal: ``exp(mu)`` is the median and ``mu + 1.645 sigma`` the 95th percentile's log.
    """
    mu = math.log(277)
    sigma = (math.log(741) - mu) / 1.645
    return max(40, int(rng.lognormvariate(mu, sigma)))


def _clamped_score(value: float) -> int:
    """A 1-5 ordinal score from a real-valued quality."""
    return max(1, min(5, round(value)))


def _score(rng: random.Random, dim: str, criterion: RubricDim | str, score: int, *, reasoning: int) -> RubricScore:
    """One judged score, carrying what production's carry: reasoning, the request settings, both fingerprints."""
    return RubricScore(
        dim=dim,
        scale="ordinal",
        axis="capability",
        score=score,
        reasoning=_prose(rng, reasoning),
        served_model=f"{_JUDGE_MODEL}-2026-08",
        judge_temperature=0.0,
        judge_sent_settings=_JUDGE_SENT,
        output_fingerprint=hashing.canonical_digest({"output": rng.random()}),
        criterion_fingerprint=fingerprint_criterion(criterion),
    )


def _usage(rng: random.Random, model: str) -> list[RoleUsage]:
    """The per-role usage rows a judged conversation records: the candidate's and the judge's."""
    rows = []
    for role, role_model, calls in (
        ("candidate", model, 6),
        ("judge", _JUDGE_MODEL, len(CRITERIA) + 2),
    ):
        prompt = rng.randint(4_000, 30_000)
        completion = rng.randint(300, 3_000)
        rows.append(
            RoleUsage(
                role=role,
                model=role_model,
                served_model=f"{role_model}-2026-08",
                prompt_tokens=prompt,
                completion_tokens=completion,
                reasoning_tokens=rng.randint(0, completion),
                cost_usd=round((prompt * 3 + completion * 15) / 1_000_000, 6),
                price_source="provider",
                call_count=calls,
                sent_settings=[_JUDGE_SENT],
            )
        )
    return rows


def _result(
    run: EvalRun, case_id: str, k: int, *, arm: int, case_effect: float, variant_key: str, identity_version: int
) -> EvalResult:
    """One result of ``run`` on ``case_id``, repeat ``k``: every field production fills, about 7 KB stored."""
    rng = random.Random(f"{run.id}|{case_id}|{k}")
    quality = _ARM_QUALITY[arm] + case_effect + rng.gauss(0.0, 0.6)
    roll = rng.random()
    candidate_error = "the candidate's turn timed out after 120s" if roll < 0.015 else None
    judge_error = "the judge's reply did not parse as the scoring schema" if 0.015 <= roll < 0.025 else None
    judged = candidate_error is None and judge_error is None
    rubric = (
        [
            _score(rng, dim.name, dim, _clamped_score(quality + rng.gauss(0.0, 0.4)), reasoning=_reasoning_length(rng))
            for dim in _template_rubric()
        ]
        if judged
        else []
    )
    passes = [rng.random() < 0.55 + 0.1 * (quality - 3.0) for _ in GOAL_CHECKS]
    usage = _usage(rng, run.candidate_model or "")
    roles = blended_cost_roles(None)
    return EvalResult(
        id=f"{run.id}--{case_id}--k{k}",
        scope_id=run.scope_id,
        eval_run_id=run.id,
        test_case_id=case_id,
        model=run.candidate_model or "",
        k_iteration=k,
        candidate_kind=run.candidate_kind,
        subject_id=run.subject_snapshot.subject_id if run.subject_snapshot is not None else None,
        has_trace=True,
        goal_state_outcomes=[
            GoalStateOutcome(
                expression=check,
                passed=passed,
                detail=f"{check.split(' ')[0]} read {rng.randint(0, 40)} against the bound",
            )
            for check, passed in zip(GOAL_CHECKS, passes, strict=True)
        ],
        rubric_scores=rubric,
        judge_reasoning=_prose(rng, 270) if judged else "",
        judge_model=_JUDGE_MODEL if judged else None,
        transcript_score=(
            _score(rng, TRANSCRIPT_DIM_ID, TRANSCRIPT_DIM_ID, _clamped_score(quality), reasoning=300)
            if judged
            else None
        ),
        outcome_score=(_score(rng, OUTCOME_DIM_ID, OUTCOME_DIM_ID, 1 + sum(passes), reasoning=250) if judged else None),
        judge_config_ids=(
            {dim: builtin_judge_config_id(f"score {dim}") for dim in (*CRITERIA, TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID)}
            if judged
            else {}
        ),
        cost_usd=math.fsum(row.cost_usd or 0.0 for row in usage if row.role in roles),
        cost_roles=list(roles),
        usage=usage,
        kind_payload={
            "ticket": {"id": f"T-{rng.randint(10_000, 99_999)}", "channel": "chat", "priority": rng.choice("abc")},
            "tool_calls": [{"tool": rng.choice(("lookup_order", "issue_refund", "tag_ticket")), "ok": True}] * 6,
            "notes": _prose(rng, 240),
        },
        world_events=[],
        covariates={"context_tokens_in": float(rng.randint(4_000, 30_000))},
        phase_timings={"candidate_s": round(rng.uniform(4.0, 40.0), 3), "judge_s": round(rng.uniform(2.0, 9.0), 3)},
        host_measures={
            "field_accuracy": round(min(1.0, max(0.0, quality / 5.0 + rng.gauss(0.0, 0.05))), 4),
            "manual_review_rate": round(rng.random() * 0.3, 4),
            "fields_stripped": float(rng.randint(0, 3)),
            "cost_per_document_usd": round(rng.uniform(0.01, 0.06), 5),
            "p95_extract_ms": round(rng.uniform(800.0, 4000.0), 1),
            "field_count_error": float(rng.randint(0, 2)),
        },
        variant_key=variant_key,
        identity_version=identity_version,
        termination="completed",
        turns_delivered=rng.randint(3, 12),
        candidate_error=candidate_error,
        judge_error=judge_error,
        scored_at=_instant(rng.randint(0, 6 * 3600)),
    )


def _template_rubric() -> list[RubricDim]:
    """The template's declared criteria, each judged by the built-in prompt."""
    return [
        declared(RubricDim(name=dim, description=f"How well the reply does on {dim}", scale="ordinal"))
        for dim in CRITERIA
    ]


def _store_runs(
    storage: EvalStorage,
    host: EvalHost,
    *,
    scope_id: str,
    template_id: str,
    run_indices: range,
    scale: int,
) -> tuple[list[str], int, str]:
    """Store the runs at ``run_indices`` of ``template_id``, each with its results, traces and cases.

    :return: The run ids in order, how many results were stored, and the control arm's variant key.
    """
    cases_per_run = CASES_PER_RUN * scale
    subject = make_subject("support-agent-7", "Support agent, config 7")
    run_ids: list[str] = []
    stored = 0
    control_key = ""
    saved_cases: set[str] = set()
    for index in run_indices:
        arm = index % len(ARM_MODELS)
        group = index // len(ARM_MODELS)
        case_ids = [f"{template_id}--case-{group * cases_per_run + n:04d}" for n in range(cases_per_run)]
        for case_id in case_ids:
            if case_id not in saved_cases:
                saved_cases.add(case_id)
                storage.save_test_case(
                    make_test_case(
                        id=case_id,
                        scope_id=scope_id,
                        template_id=template_id,
                        variation_params={"topic": f"topic-{int(case_id[-4:]) % 7}"},
                        stratum=("refund", "shipping", "account")[int(case_id[-4:]) % 3],
                    )
                )
        run = make_eval_run(
            id=f"{template_id}--run-{index:02d}",
            scope_id=scope_id,
            template_id=template_id,
            subject_snapshot=subject,
            candidate_model=ARM_MODELS[arm],
            judge_model=_JUDGE_MODEL,
            simulator_model="simulator-model",
            k_runs=K_RUNS,
            test_case_ids=case_ids,
            status="completed",
            created_at=_instant(index * 600),
        )
        storage.save_eval_run(run)
        run_ids.append(run.id)
        identity = resolve_variant_identity(run=run, profile=host.profile)
        if arm == 0:
            control_key = identity.variant_key
        for case_id in case_ids:
            case_effect = random.Random(case_id).gauss(0.0, 0.5)
            for k in range(1, K_RUNS + 1):
                result = _result(
                    run,
                    case_id,
                    k,
                    arm=arm,
                    case_effect=case_effect,
                    variant_key=identity.variant_key,
                    identity_version=identity.identity_version,
                )
                trace = make_eval_trace(
                    result_id=result.id,
                    scope_id=scope_id,
                    eval_run_id=run.id,
                    judge_evidence=JudgeEvidence(case_material=f"case {case_id}", artifact="the reply as judged"),
                    judged_artifact=JudgedArtifact.TRANSCRIPT,
                )
                storage.save_eval_result(result, trace)
                stored += 1
    return run_ids, stored, control_key


def hot_path_corpus(
    *,
    scale: int = 1,
    two_aspects: bool = False,
    scope_id: str = HOT_PATH_SCOPE,
    template_id: str = "tpl-support",
    store: InMemoryDocumentStore | None = None,
) -> HotPathCorpus:
    """Build the corpus at ``scale`` (1, 2 or 4: the cases each run measures), stored and ready to read.

    :param scale: The case multiplier: 1 is about 3,100 results, 2 about 6,200, 4 about 12,500.
    :ptype scale: int
    :param two_aspects: Split the runs over two templates, one aspect each, so assembly runs once per aspect.
    :ptype two_aspects: bool
    :param scope_id: The scope to store it in.
    :ptype scope_id: str
    :param template_id: The (first) template its runs ran: a launch planned on that template reads them as its history.
    :ptype template_id: str
    :param store: The store to write it into, beside what it holds; None for a fresh one.
    :ptype store: InMemoryDocumentStore | None
    :return: The corpus.
    :rtype: HotPathCorpus
    """
    storage = EvalStorage(store if store is not None else InMemoryDocumentStore())
    host = toyhost_host(storage=storage)
    template_ids = (template_id, "tpl-escalation") if two_aspects else (template_id,)
    for each_template in template_ids:
        storage.save_template(
            make_template(
                id=each_template,
                scope_id=scope_id,
                name=each_template,
                rubric=_template_rubric(),
                goal_state_checks=list(GOAL_CHECKS),
            )
        )
    split = (N_RUNS + 1) // 2 if two_aspects else N_RUNS
    ranges = (range(0, split), range(split, N_RUNS)) if two_aspects else (range(0, N_RUNS),)
    design = AspectDesign(
        axes=[
            SweptAxis(
                axis_id=CANDIDATE_MODEL_LEVER, values=[SweepableValue.of(model, display=model) for model in ARM_MODELS]
            )
        ],
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
        merit_axes=[],
    )
    aspects = []
    total = 0
    control_key = ""
    for each_template, indices in zip(template_ids, ranges, strict=True):
        run_ids, stored, key = _store_runs(
            storage, host, scope_id=scope_id, template_id=each_template, run_indices=indices, scale=scale
        )
        total += stored
        control_key = control_key or key
        aspects.append((each_template, run_ids))
    first = make_aspect_campaign(
        label=aspects[0][0],
        template_id=aspects[0][0],
        run_ids=aspects[0][1],
        design=design,
        question="does any candidate answer better than the control, at what cost?",
        asked_at=HOT_PATH_INSTANT,
        control=CampaignArm(label="control", role="control", variant_key=control_key, named_at=HOT_PATH_INSTANT),
        id="hot-path-campaign",
        scope_id=scope_id,
        name="support model sweep",
        subject_id="support-agent-7",
        created_by="test:hot-path",
        created_at=HOT_PATH_INSTANT,
    )
    campaign = first
    if two_aspects:
        second = first.aspects[0].model_copy(
            update={"label": aspects[1][0], "template_id": aspects[1][0], "run_ids": aspects[1][1]}
        )
        campaign = first.model_copy(update={"aspects": [*first.aspects, second]})
        campaign = EvalCampaign.model_validate(campaign.model_dump())
    storage.save_campaign(campaign)
    return HotPathCorpus(
        host=host, campaign=campaign, scope_id=scope_id, template_ids=template_ids, n_results=total, scale=scale
    )


def field_byte_shares(corpus: HotPathCorpus) -> tuple[float, dict[str, float]]:
    """The stored results' mean size in bytes and each top-level field's share of their bytes, largest first.

    The check that the corpus is shaped like production's rows: compare the shares against the plan's.

    :param corpus: A built corpus.
    :ptype corpus: HotPathCorpus
    :return: ``(mean bytes per result, {field: share})``.
    :rtype: tuple[float, dict[str, float]]
    """
    totals: Counter[str] = Counter()
    count = 0
    whole = 0
    for run_id in corpus.campaign.aspects[0].run_ids[:4]:
        for result in corpus.host.storage.query_eval_results_by_run(run_id, corpus.scope_id):
            document = result.to_dict()
            whole += len(hashing.canonical_json(document))
            count += 1
            for name, value in document.items():
                totals[name] += len(hashing.canonical_json({name: value}))
    return whole / count, {name: size / whole for name, size in totals.most_common()}


# --- captured stores ------------------------------------------------------------------------------------------------


def dumped(store: InMemoryDocumentStore, scope_id: str) -> list[dict[str, Any]]:
    """Every document ``store`` holds in ``scope_id``, in the order it was written (the order a listing reads)."""
    return [document for (scope, _doc_id), document in store.documents.items() if scope == scope_id]


def loaded(documents: Sequence[Mapping[str, Any]]) -> InMemoryDocumentStore:
    """A fresh store holding ``documents``, written in the order given."""
    store = InMemoryDocumentStore()
    for document in documents:
        store.upsert(dict(document))
    return store


async def courierhost_capture() -> list[dict[str, Any]]:
    """The courier's planner bake-off, run through the real trial loop and stored: every document of its scope.

    A run mints its ids and reads the clock, so the corpus is captured once and read back from the capture: the
    captured campaign is the courier's campaign, and every read of it is the same read.
    """
    store = InMemoryDocumentStore()
    await run_courier_campaign(replace(courier_host(), storage=EvalStorage(store)))
    return dumped(store, COURIER_SCOPE)


async def toyhost_doc_capture() -> list[dict[str, Any]]:
    """The toy campaign with its report written and stored by ``analysis_generate`` (a ``doc`` analysis): every document
    of its scope.

    Captured once for the reason :func:`courierhost_capture` is: a generation mints its analysis's id and dates its
    made line, which the page prints.
    """
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    (job,) = (await generate(fixture.host, fixture.campaign.id)).jobs
    status = await settled(fixture.host, job.job_id)
    if (status.state, status.status) != ("completed", "stored"):
        raise AssertionError(f"the toy campaign's report was not stored: {status}")
    return dumped(store, TOYHOST_SCOPE)


def courierhost_from(documents: Sequence[Mapping[str, Any]]) -> EvalHost:
    """The courier host over a store holding ``documents`` (:func:`courierhost_capture`)."""
    return replace(courier_host(), storage=EvalStorage(loaded(documents)))


def toyhost_from(documents: Sequence[Mapping[str, Any]]) -> EvalHost:
    """The toy host over a store holding ``documents`` (:func:`toyhost_doc_capture`)."""
    return toyhost_host(storage=EvalStorage(loaded(documents)))


def campaign_id_of(documents: Sequence[Mapping[str, Any]]) -> str:
    """The id of the one campaign ``documents`` hold."""
    (campaign_id,) = [str(document["id"]) for document in documents if document.get("doc_type") == "eval_campaign"]
    return campaign_id


# --- counting ------------------------------------------------------------------------------------------------------

#: The operations :func:`counting_operations` counts, in the order a table lists them:
#:
#: - ``score_records``: ``ScoreRecord`` constructions;
#: - ``result_validations``: ``EvalResult`` validations (each hydration of a stored row, and each construction);
#: - ``model_copies``: ``model_copy`` calls on any model, and ``result_model_copies`` those on an ``EvalResult``;
#: - ``fingerprints``: bundle ``fingerprint()`` calls, aspect and campaign bundles together;
#: - ``canonical``: ``canonical_json`` and ``canonical_digest`` calls, wherever they are reached from (a digest's own
#:   call to ``canonical_json`` counts too);
#: - ``describe_measure``: measure descriptions (``kernel.metrics.describe_measure``);
#: - ``model_inits``: every pydantic model built through its constructor, of any class.
OPERATIONS = (
    "score_records",
    "result_validations",
    "model_copies",
    "result_model_copies",
    "fingerprints",
    "canonical",
    "describe_measure",
    "model_inits",
)


class _CountingValidator:
    """A model's validator, counting each validation before handing it to the real one."""

    def __init__(self, inner: Any, bump: Callable[[], None]) -> None:
        self.inner = inner
        self.bump = bump

    def validate_python(self, *args: Any, **kwargs: Any) -> Any:
        self.bump()
        return self.inner.validate_python(*args, **kwargs)

    def validate_json(self, *args: Any, **kwargs: Any) -> Any:
        self.bump()
        return self.inner.validate_json(*args, **kwargs)

    def validate_strings(self, *args: Any, **kwargs: Any) -> Any:
        self.bump()
        return self.inner.validate_strings(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _wrapped(function: Callable[..., Any], bump: Callable[[], None]) -> Callable[..., Any]:
    """``function``, counting each call."""

    def counted(*args: Any, **kwargs: Any) -> Any:
        bump()
        return function(*args, **kwargs)

    return counted


def _bindings(function: Callable[..., Any]) -> Iterator[tuple[object, str]]:
    """Every ``(module, name)`` under ``threetears.evals`` that binds ``function``: where a caller looks it up."""
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("threetears.evals"):
            continue
        for attribute, value in list(vars(module).items()):
            if value is function:
                yield module, attribute


@contextmanager
def counting_operations() -> Iterator[Counter[str]]:
    """Count the operations in :data:`OPERATIONS` run inside the block, by name.

    Validations are counted at each model's validator, which every construction and every ``model_validate`` goes
    through; ``model_copy`` and constructions at :class:`pydantic.BaseModel`; the fingerprints at both bundle
    classes; and the module functions at every module that imported them. Each wrapped callable still runs.
    """
    counts: Counter[str] = Counter({name: 0 for name in OPERATIONS})

    def bumping(name: str) -> Callable[[], None]:
        def bump() -> None:
            counts[name] += 1

        return bump

    copy = BaseModel.model_copy
    construct = BaseModel.__init__

    def counted_copy(self: BaseModel, *args: Any, **kwargs: Any) -> Any:
        counts["model_copies"] += 1
        if isinstance(self, EvalResult):
            counts["result_model_copies"] += 1
        return copy(self, *args, **kwargs)

    def counted_init(self: BaseModel, /, **data: Any) -> None:
        counts["model_inits"] += 1
        construct(self, **data)

    with ExitStack() as stack:
        for model, name in ((ScoreRecord, "score_records"), (EvalResult, "result_validations")):
            validator = _CountingValidator(model.__pydantic_validator__, bumping(name))
            stack.enter_context(mock.patch.object(model, "__pydantic_validator__", validator))
        stack.enter_context(mock.patch.object(BaseModel, "model_copy", counted_copy))
        stack.enter_context(mock.patch.object(BaseModel, "__init__", counted_init))
        for bundle_class in (AnalysisContextBundle, CampaignBundle):
            counted_fingerprint = _wrapped(bundle_class.fingerprint, bumping("fingerprints"))
            stack.enter_context(mock.patch.object(bundle_class, "fingerprint", counted_fingerprint))
        for function, name in (
            (hashing.canonical_json, "canonical"),
            (hashing.canonical_digest, "canonical"),
            (describe_measure, "describe_measure"),
        ):
            counted = _wrapped(function, bumping(name))
            for module, attribute in list(_bindings(function)):
                stack.enter_context(mock.patch.object(module, attribute, counted))
        yield counts


def report_read_counts(corpus: HotPathCorpus) -> dict[str, dict[str, int]]:
    """The operations one code-only ``report_read`` of ``corpus`` (html, ``trials="table"``) counts, and its assembly's.

    One read, counted once: ``assembly`` is the count as the campaign bundle is built (the moment
    :func:`~threetears.evals.analysis.campaign_bundle.build_campaign_bundle` returns, the last step of assembly), and
    ``report_read`` the whole read's, the page and its serialization included.

    :param corpus: A built corpus.
    :ptype corpus: HotPathCorpus
    :return: ``{"assembly": {operation: count}, "report_read": {operation: count}}``.
    :rtype: dict[str, dict[str, int]]
    """
    assembled: dict[str, int] = {}
    with counting_operations() as counts, ExitStack() as stack:

        def marked(*args: Any, **kwargs: Any) -> CampaignBundle:
            built = build_campaign_bundle(*args, **kwargs)
            assembled.update(counts)
            return built

        for module, attribute in list(_bindings(build_campaign_bundle)):
            stack.enter_context(mock.patch.object(module, attribute, marked))
        with mock.patch.object(analysis_service, "utc_now_iso", lambda: HOT_PATH_INSTANT):
            report_read(corpus.host, corpus.campaign.id, corpus.scope_id, format="html")
    return {"assembly": assembled, "report_read": dict(counts)}


def growth(counts: Mapping[int, Mapping[str, int]], name: str) -> list[float]:
    """Each doubling's growth ratio of ``name``: ``counts[2s][name] / counts[s][name]`` for each scale ``s``.

    A count that is zero at the smaller scale grows by 1.0 when it stays zero, and without bound when it does not.
    """
    scales = sorted(counts)
    ratios = []
    for lower, upper in zip(scales, scales[1:], strict=False):
        before, after = counts[lower][name], counts[upper][name]
        ratios.append(after / before if before else (math.inf if after else 1.0))
    return ratios
