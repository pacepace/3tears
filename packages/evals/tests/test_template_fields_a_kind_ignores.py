"""A template carrying a field its kind never acts on is refused at the dispatch, naming the field.

``launchable_judge_kind`` admitted a judge template carrying a world seed, which the judge kind never applies, and the
launch ran as if the template had said nothing — the same silence a launch argument a kind cannot honour used to
meet, before ``unhonoured_launch_arguments`` refused it. So a kind's registry entry declares the template fields it
ignores (``LaunchableKind.ignored_template_fields``), the dispatch refuses a template stating any of them, and the
package's own launchables declare theirs.

Every path that admits a template refuses the same template before it is stored, through the dispatch's own
refusal (``refuse_ignored_template_fields``) over the kind's entry in the host's registry (``TemplateChecks.kinds``):
``create_template``, ``update_template`` and the boot seed. A host's own checks no longer call it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from threetears.evals.kernel import JUDGE_KIND, JUDGE_KIND_CONTRACT, ValidationFailedError, default_job_timeout
from threetears.evals.kernel.host import EvalHost
from threetears.evals.run import (
    TEMPLATE_FIELDS,
    LaunchableKind,
    LaunchHost,
    SeedCorpus,
    create_template,
    launchable_judge_kind,
    quote_launch,
    seed_eval_definitions,
    update_template,
)
from threetears.evals.run.authoring import refuse_ignored_template_fields
from threetears.evals.schema import EvalTemplate, Precondition, WorldSeed
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.host import TOYHOST_TEMPLATE_CHECKS, toyhost_host, with_registry
from packages.evals.tests.off_loop import off_loop

_SCOPE = "judge-template-fields"


def _judge_host() -> LaunchHost:
    profile = toyhost_profile()
    eval_host = dataclasses.replace(
        toyhost_host(),
        # Held to every seat the toy host declares, so its profile stays sound with a second kind beside its own.
        profile=dataclasses.replace(
            profile, kinds=(*profile.kinds, dataclasses.replace(JUDGE_KIND_CONTRACT, seats=None))
        ),
    )
    holder: list[LaunchHost] = []
    launch_host = LaunchHost(
        eval_host=with_registry(eval_host, {JUDGE_KIND: launchable_judge_kind(lambda: holder[0])}),
        settings=lambda: TOYHOST_LAUNCH_SETTINGS,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda _run: {},
    )
    holder.append(launch_host)
    return launch_host


def _judge_template(**fields: Any) -> EvalTemplate:
    return EvalTemplate(
        id="judge-tpl",
        scope_id=_SCOPE,
        name="a judge",
        intent="Score as a person would.",
        candidate_kind=JUDGE_KIND,
        **fields,
    )


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        pytest.param({"world_seed": WorldSeed(namespaces={"inbox": {"unread": 3}})}, "world_seed", id="a-world-seed"),
        pytest.param({"goal_state_checks": ["state.inbox.unread == 0"]}, "goal_state_checks", id="goal-checks"),
        pytest.param(
            {"preconditions": [Precondition(expression="state.inbox.unread == 3", presumes="the inbox starts full")]},
            "preconditions",
            id="preconditions",
        ),
    ],
)
async def test_a_judge_template_stating_what_the_judge_kind_ignores_is_refused(
    fields: dict[str, Any], named: str
) -> None:
    host = _judge_host()
    off_loop(host.eval_host.storage.save_template, _judge_template(**fields))
    with pytest.raises(ValidationFailedError, match=named):
        await quote_launch(host, template_id="judge-tpl", subject_id="judge", models=["judge/model"], scope_id=_SCOPE)


def test_the_judge_kind_declares_every_field_but_what_it_reads() -> None:
    entry = launchable_judge_kind(lambda: _judge_host())
    assert set(TEMPLATE_FIELDS) - entry.ignored_template_fields == set(), "a judge template is a kind and an intent"


def test_a_kind_declaring_a_name_no_template_has_is_refused() -> None:
    with pytest.raises(ValueError, match="not template fields"):
        LaunchableKind(
            launch=lambda _request: None,  # type: ignore[arg-type,return-value]
            resolve_subject=lambda _subject, _scope: None,
            ignored_template_fields=frozenset({"world_sed"}),
        )


def test_a_hosts_authoring_check_refuses_what_the_dispatch_would_through_the_same_call() -> None:
    entry = launchable_judge_kind(lambda: _judge_host())
    refuse_ignored_template_fields(_judge_template(), entry)
    with pytest.raises(ValidationFailedError, match="never acts on its goal_state_checks, world_seed") as refused:
        refuse_ignored_template_fields(
            _judge_template(
                world_seed=WorldSeed(namespaces={"inbox": {"unread": 3}}), goal_state_checks=["state.done == 1"]
            ),
            entry,
        )
    assert refused.value.remedy


# --- authoring: the package refuses it on every path, from the host's registry --------------------------------------

_SEED = WorldSeed(namespaces={"inbox": {"unread": 3}})


def _authoring_host() -> EvalHost:
    """The judge host's own EvalHost: its template checks carry the registry it launches with."""
    return _judge_host().eval_host


def _host_call(template: EvalTemplate) -> ValidationFailedError:
    """What a host's own check raised over the same template, by calling the dispatch's refusal itself."""
    with pytest.raises(ValidationFailedError) as refused:
        refuse_ignored_template_fields(template, launchable_judge_kind(lambda: _judge_host()))
    return refused.value


def test_create_template_refuses_a_field_the_kind_ignores_with_the_host_calls_message() -> None:
    host = _authoring_host()
    definition = {"name": "a judge", "intent": "Score as a person would.", "candidate_kind": JUDGE_KIND}

    with pytest.raises(ValidationFailedError) as refused:
        create_template(host, {**definition, "world_seed": _SEED}, scope_id=_SCOPE)

    template_id = refused.value.message.split("'")[1]
    expected = _host_call(EvalTemplate(id=template_id, scope_id=_SCOPE, **definition, world_seed=_SEED))
    assert (refused.value.message, refused.value.remedy) == (expected.message, expected.remedy)
    assert host.storage.load_template_by_name("a judge", _SCOPE) is None, "nothing is stored"
    assert create_template(host, definition, scope_id=_SCOPE).candidate_kind == JUDGE_KIND


def test_update_template_refuses_a_field_the_kind_ignores_with_the_host_calls_message() -> None:
    host = _authoring_host()
    host.storage.save_template(_judge_template())

    with pytest.raises(ValidationFailedError) as refused:
        update_template(host, "judge-tpl", _SCOPE, {"world_seed": _SEED})

    expected = _host_call(_judge_template(world_seed=_SEED))
    assert (refused.value.message, refused.value.remedy) == (expected.message, expected.remedy)
    stored = host.storage.load_template("judge-tpl", _SCOPE)
    assert stored is not None and stored.world_seed == _judge_template().world_seed, "nothing is written"


def test_an_update_reads_the_merged_template_and_archiving_alone_is_never_refused() -> None:
    host = _authoring_host()
    # Written past authoring, as a template stored before the rule reached it was.
    host.storage.save_template(_judge_template(world_seed=_SEED))

    with pytest.raises(ValidationFailedError, match="never acts on its world_seed"):
        update_template(host, "judge-tpl", _SCOPE, {"intent": "Score as a careful person would."})
    assert update_template(host, "judge-tpl", _SCOPE, {"archived": True}).archived


def test_the_boot_seed_refuses_a_field_the_kind_ignores() -> None:
    host = _authoring_host()
    with pytest.raises(ValidationFailedError, match="never acts on its world_seed"):
        seed_eval_definitions(host, SeedCorpus(scope_id=_SCOPE, templates=(_judge_template(world_seed=_SEED),)))


def test_a_kind_the_registry_does_not_carry_is_refused_nothing_at_authoring() -> None:
    # The toy host's reading checks carry no registry: its launch refuses the kind by name instead.
    host = dataclasses.replace(_authoring_host(), template_checks=TOYHOST_TEMPLATE_CHECKS)
    stored = create_template(
        host,
        {"name": "a judge", "intent": "Score as a person would.", "candidate_kind": JUDGE_KIND, "world_seed": _SEED},
        scope_id=_SCOPE,
    )
    assert stored.world_seed == _SEED
