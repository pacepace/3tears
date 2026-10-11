"""A subject the host does not hold is refused at admission — by the estimate and the launch alike, before any pricing.

``launch_estimate`` priced a launch naming subject "Vera", which the host did not hold, while ``run_launch`` with
the same arguments was refused ("subject 'Vera' not found") — by the host's launcher, inside the launch's job,
after every arm had been priced. The estimate promises every refusal the launch makes before pricing, so the
subject is resolved where both admit a launch: through the host's :attr:`~threetears.evals.run.LaunchHost.resolve_subject`,
which answers whether it holds the subject and, when it does not, how its caller finds one it does.
"""

from __future__ import annotations

import dataclasses

import pytest

from threetears.evals.kernel import NotFoundError
from threetears.evals.ops import LaunchArguments, OpsHost, launch_estimate, run_launch
from threetears.evals.run import UnknownSubject
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import RUN_MODELS, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture


def _arguments(subject_id: str) -> LaunchArguments:
    return LaunchArguments(template_id=toyhost_template().id, subject_id=subject_id, models=list(RUN_MODELS))


async def test_the_estimate_refuses_a_subject_the_host_does_not_hold() -> None:
    fixture = ops_fixture()
    with pytest.raises(NotFoundError, match="subject 'Vera' not found") as refused:
        await launch_estimate(fixture.host, _arguments("Vera"), TOYHOST_SCOPE)
    assert refused.value.remedy, "the host says how to find a subject it holds"


async def test_the_launch_refuses_it_at_admission_before_any_arm_is_priced() -> None:
    fixture = ops_fixture()
    priced: list[object] = []
    pricer = fixture.host.launch.launch_pricer
    assert pricer is not None

    def counting(*args: object, **kwargs: object) -> object:
        priced.append(args)
        return pricer(*args, **kwargs)  # type: ignore[arg-type]

    host = OpsHost(launch=dataclasses.replace(fixture.host.launch, launch_pricer=counting))  # type: ignore[arg-type]
    with pytest.raises(NotFoundError, match="subject 'Vera' not found"):
        await run_launch(host, _arguments("Vera"), TOYHOST_SCOPE)
    assert priced == []


async def test_a_subject_the_host_holds_is_admitted() -> None:
    fixture = ops_fixture()
    estimate = await launch_estimate(fixture.host, _arguments(TOYHOST_SUBJECT.subject_id), TOYHOST_SCOPE)
    assert estimate.subject_id == TOYHOST_SUBJECT.subject_id


def test_an_unknown_subject_names_how_to_find_one() -> None:
    with pytest.raises(ValueError, match="remedy"):
        UnknownSubject(remedy=" ")
