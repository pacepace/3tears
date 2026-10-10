"""Quick's building blocks inside a host of your own: ``@measure`` and ``callable_kind``, mixed with hand-written parts.

``callable_host`` is one host shape: its store, the shared-core sweepables and one measure per scorer. A host that
needs its own store or its own declarations used to drop to the contract and write by hand what quick generated.
Here a host built by hand — its own SQLite store, a hand-written measure beside an ``@measure`` one, a lever of its
own on the callable kind — launches the quick kind through the engine's own ``start_run``.

Mutations that turn this file red: a ``Measure`` that stops being callable or loses its descriptor; a descriptor
built apart from the decorated function's name and docstring; ``callable_kind`` ignoring a ``Measure``'s range;
``callable_host`` writing its own descriptor over a ``Measure``'s.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.contracts import EvalTemplate, EvalTestCase, MetricDescriptor
from threetears.evals.contracts.host import (
    SHARED_CORE,
    EvalHost,
    HostProfile,
    MeasureRegistry,
    SubjectSnapshot,
    default_cell_timeout,
)
from threetears.evals.contracts.storage import EvalStorage
from threetears.evals.quick import (
    CALLABLE_KIND,
    Measure,
    callable_host,
    callable_kind,
    callable_kind_contracts,
    measure,
)
from threetears.evals.run import (
    KindWiring,
    LaunchableKind,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    default_job_timeout,
    launch_run,
    list_results,
    start_run,
)
from threetears.evals.storage import SqliteDocumentStore


@measure(higher_is_better=True, merit_axis="quality", value_range=(0.0, 1.0), population="scored")
def exact(case: Mapping[str, Any], answer: str) -> float:
    """Whether the answer is the one the case wants."""
    return 1.0 if answer == case["want"] else 0.0


#: A measure written by hand, computed by a plain scorer: the two kinds of declaration mix in one registry.
ANSWER_LENGTH = MetricDescriptor(
    name="answer_length",
    reader_name="Answer length",
    data_type="numeric",
    family="mechanical",
    transferability_class="mechanical",
    attribution_scope="end_to_end",
    description="How many characters the answer ran to.",
    higher_is_better=None,
    diagnostic=True,
    unit="characters",
    population="all_observed",
)


def answer_length(case: Mapping[str, Any], answer: str) -> float:
    return float(len(answer))


async def candidate(case: Mapping[str, Any]) -> str:
    return str(case["want"]) if case["n"] % 2 == 0 else "nope"


def test_a_measure_is_its_function_carrying_its_declaration() -> None:
    assert isinstance(exact, Measure) and exact({"want": "a"}, "a") == 1.0
    assert exact.__name__ == exact.descriptor.name == "exact"
    assert exact.descriptor.description == "Whether the answer is the one the case wants."
    assert exact.descriptor.reader_name == "Exact score" and exact.descriptor.value_range == (0.0, 1.0)


def test_a_measure_is_refused_where_it_is_declared() -> None:
    with pytest.raises(ValueError, match="no name to key"):
        measure(higher_is_better=True)(lambda case, answer: 1.0)
    with pytest.raises(ValueError, match="needs a description"):
        measure(name="unsaid", higher_is_better=True)(lambda case, answer: 1.0)
    with pytest.raises(ValueError, match="guardrail"):

        @measure(higher_is_better=True, merit_axis="quality", guardrail=True)
        def leaks(case: Mapping[str, Any], answer: str) -> float:
            """Whether it leaked."""
            return 0.0


def test_callable_host_registers_a_measure_s_own_descriptor_and_refuses_a_second_one() -> None:
    assert callable_host([exact]).profile.measures.get("exact") == exact.descriptor
    with pytest.raises(ValueError, match="on its own @measure"):
        callable_host([exact], margins={"exact": 0.1})


async def test_a_host_of_your_own_launches_the_quick_kind_through_start_run(tmp_path: Path) -> None:
    contract, _judged = callable_kind_contracts(levers=("prompt",))
    with SqliteDocumentStore(tmp_path / "mine.sqlite") as store:
        host = EvalHost(
            profile=HostProfile(
                host_id="mine",
                host_sweepables=SHARED_CORE,
                measures=MeasureRegistry([exact.descriptor, ANSWER_LENGTH]),
                kinds=(contract,),
            ),
            storage=EvalStorage(store),
            failure_describer=lambda failure: None,
            trace_sink=None,
            blocking_executor=None,
            cell_timeout=default_cell_timeout,
            clients=None,
        )
        template = EvalTemplate(
            id="t", scope_id="mine", name="four cases", intent="Answer.", candidate_kind=CALLABLE_KIND
        )
        cases = [
            EvalTestCase(
                id=f"t-{n}",
                scope_id="mine",
                template_id="t",
                variation_params={"n": str(n)},
                host_payload={"case": {"n": n, "want": "yes"}},
            )
            for n in range(4)
        ]
        host.storage.save_template(template)
        for case in cases:
            host.storage.save_test_case(case)
        kind = callable_kind(candidate, [exact, answer_length])

        async def launch(request: LaunchRequest) -> Any:
            subject = SubjectSnapshot(subject_id=request.subject_id, subject_label=request.subject_id, state={})
            wiring = KindWiring(kind_factory=lambda _cell: kind, subject=subject, test_cases=cases, judge=None)
            return await launch_run(launching, request, wiring)

        settings = LaunchSettings(
            max_launch_arms=1,
            max_admitted_runs=1,
            judge_concurrency=1,
            enforcement_enabled=False,
            max_cost_usd=1.0,
            max_metered_calls=None,
            max_out_of_run_cost_usd=1.0,
        )
        launching = LaunchHost(
            eval_host=host,
            kinds={
                CALLABLE_KIND: LaunchableKind(
                    launch=launch,
                    unhonoured_launch_arguments=frozenset(
                        {"simulator_model", "judge_config_ids", "n_variations", "judge_model", "cassette_mode"}
                    ),
                )
            },
            settings=lambda: settings,
            job_timeout_factory=default_job_timeout,
        )
        (run,) = await start_run(
            launching,
            template_id="t",
            subject_id="mine",
            models=["m"],
            k_runs=1,
            scope_id="mine",
            overlays={"prompt": "v2"},
        )
        await launching.job_manager.wait_for([run.id])
        results = sorted(list_results(host.storage, run.id, "mine"), key=lambda result: result.test_case_id)
    assert [result.host_measures for result in results] == [
        {"exact": 1.0, "answer_length": 3.0},
        {"exact": 0.0, "answer_length": 4.0},
        {"exact": 1.0, "answer_length": 3.0},
        {"exact": 0.0, "answer_length": 4.0},
    ]
