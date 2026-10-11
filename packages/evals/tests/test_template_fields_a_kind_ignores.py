"""A template carrying a field its kind never acts on is refused at the dispatch, naming the field.

``launchable_judge_kind`` admitted a judge template carrying a world seed, which the judge kind never applies, and the
launch ran as if the template had said nothing — the same silence a launch argument a kind cannot honour used to
meet, before ``unhonoured_launch_arguments`` refused it. So a kind's registry entry declares the template fields it
ignores (``LaunchableKind.ignored_template_fields``), the dispatch refuses a template stating any of them, and the
package's own launchables declare theirs.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from threetears.evals.kernel import JUDGE_KIND, JUDGE_KIND_CONTRACT, ValidationFailedError, default_job_timeout
from threetears.evals.run import (
    TEMPLATE_FIELDS,
    LaunchableKind,
    LaunchHost,
    launchable_judge_kind,
    quote_launch,
)
from threetears.evals.schema import EvalTemplate, Precondition, WorldSeed
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

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
        eval_host=eval_host,
        kinds={JUDGE_KIND: launchable_judge_kind(lambda: holder[0])},
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
    host.eval_host.storage.save_template(_judge_template(**fields))
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

