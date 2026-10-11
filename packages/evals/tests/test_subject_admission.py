"""A subject the host does not hold is refused at admission — by the estimate and the launch alike, before any pricing.

``launch_estimate`` priced a launch naming subject "Vera", which the host did not hold, while ``run_launch`` with
the same arguments was refused ("subject 'Vera' not found") — by the host's launcher, inside the launch's job,
after every arm had been priced. The estimate promises every refusal the launch makes before pricing, so the
subject is resolved where both admit a launch, and answered whether it is held and, when it is not, how its caller
finds one that is.

It is asked of the template's KIND (:attr:`~threetears.evals.run.LaunchableKind.resolve_subject`), not of the host:
one subject id is a live entity to one kind, a fixed prompt key to another and a free label to the judge kind. A
host-wide resolver, told only the id and the scope, could either refuse every judge launch or admit any string for
every kind.
"""

from __future__ import annotations

import dataclasses

import pytest

from threetears.evals.kernel import JUDGE_KIND, JUDGE_KIND_CONTRACT, NotFoundError, ValidationFailedError
from threetears.evals.ops import LaunchArguments, OpsHost, launch_estimate, run_launch
from threetears.evals.run import (
    LaunchHost,
    UnknownSubject,
    launchable_judge_kind,
    quote_launch,
    start_universal_battery,
)
from threetears.evals.schema import EvalTemplate
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import RUN_MODELS, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture

#: A judge arm's subject: a label, which no host holds and the judge kind admits.
_JUDGE_LABEL = "the matrix judge"


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


def _two_kind_host() -> LaunchHost:
    """The toy host launching its own kind and the judge kind, with a template of each stored."""
    profile = toyhost_profile()
    toy, _ = toyhost_launch_host(
        profile=dataclasses.replace(
            profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
        )
    )
    holder: list[LaunchHost] = []
    host = dataclasses.replace(toy, kinds={**toy.kinds, JUDGE_KIND: launchable_judge_kind(lambda: holder[0])})
    holder.append(host)
    host.eval_host.storage.save_template(toyhost_template())
    host.eval_host.storage.save_template(
        EvalTemplate(
            id="judge-tpl",
            scope_id=TOYHOST_SCOPE,
            name="a judge",
            intent="Score as a person would.",
            candidate_kind=JUDGE_KIND,
        )
    )
    return host


async def _quoted(host: LaunchHost, template_id: str, subject_id: str) -> None:
    await quote_launch(
        host, template_id=template_id, subject_id=subject_id, models=[RUN_MODELS[0]], scope_id=TOYHOST_SCOPE
    )


async def test_one_host_asks_each_kind_what_its_subject_is() -> None:
    """One id, two kinds on one host: the toy kind holds no subject by that name, and the judge kind takes any label."""
    host = _two_kind_host()
    with pytest.raises(NotFoundError, match=f"subject '{_JUDGE_LABEL}' not found; the toy host holds subjects"):
        await _quoted(host, toyhost_template().id, _JUDGE_LABEL)
    # Admitted: the launch goes on to its next refusal, a judge template with no case frozen into it.
    with pytest.raises(ValidationFailedError, match="holds no judge case to run"):
        await _quoted(host, "judge-tpl", _JUDGE_LABEL)
    with pytest.raises(NotFoundError, match="subject '.*' not found; name the judge being measured"):
        await _quoted(host, "judge-tpl", "  ")


async def test_the_kinds_resolver_decides_not_one_the_host_holds() -> None:
    fixture = ops_fixture()
    [(name, kind)] = fixture.host.launch.kinds.items()
    admitting = dataclasses.replace(kind, resolve_subject=lambda _subject, _scope: None)
    host = OpsHost(launch=dataclasses.replace(fixture.host.launch, kinds={name: admitting}))
    estimate = await launch_estimate(host, _arguments("Vera"), TOYHOST_SCOPE)
    assert estimate.subject_id == "Vera"


async def test_a_battery_asks_each_kind_it_launches_before_any_arm_is_priced() -> None:
    fixture = ops_fixture()
    priced: list[object] = []
    pricer = fixture.host.launch.launch_pricer
    assert pricer is not None

    def counting(*args: object, **kwargs: object) -> object:
        priced.append(args)
        return pricer(*args, **kwargs)  # type: ignore[arg-type]

    host = dataclasses.replace(fixture.host.launch, launch_pricer=counting)  # type: ignore[arg-type]
    host.eval_host.storage.save_template(toyhost_template().model_copy(update={"id": "universal", "universal": True}))

    async def no_preflight(_subject_id: str, _models: list[str]) -> object:  # pragma: no cover - never reached
        raise AssertionError("the subject is refused before the pre-flight")

    with pytest.raises(NotFoundError, match="subject 'Vera' not found"):
        await start_universal_battery(
            host,
            "Vera",
            scope_id=TOYHOST_SCOPE,
            models=[RUN_MODELS[0]],
            preflight=no_preflight,  # type: ignore[arg-type]
        )
    assert priced == []
