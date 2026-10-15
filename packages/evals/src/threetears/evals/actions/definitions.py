"""The two catalogue actions that keep a host's seeded definitions current: read the drift, and reset.

The engine does not know where a host's seed corpus lives or what its host checks are, so it ships these actions
as a factory the host calls with both and hands to :func:`~threetears.evals.actions.eval_catalogue`:
``definitions_drift`` (read) and ``definitions_reset`` (write), over
:func:`~threetears.evals.run.definition_drift` and :func:`~threetears.evals.run.reset_definitions`. Neither runs at
boot: a reset is an operator's, or a release step's, deliberate act.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from pydantic import Field

from threetears.evals.actions.catalogue import Action, Caller
from threetears.evals.ops.host import OpsHost
from threetears.evals.run import DefinitionDrift, DefinitionReset, SeedCorpus, definition_drift, reset_definitions
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import EvalTemplate

#: The help index's group for the two actions.
DEFINITIONS_WORKFLOW = "Keep the seeded definitions current"


class DefinitionsDriftParams(EvalBaseModel):
    """``definitions_drift`` takes nothing: it reads the caller's scope against the host's corpus."""


class DefinitionsResetParams(EvalBaseModel):
    """What ``definitions_reset`` writes, and whether it may write over what may be an operator's."""

    definitions: list[str] = Field(
        min_length=1,
        description="The definitions to reset, by the keys definitions_drift names them, such as "
        "'eval_template/<name>' or 'judge_case/<judge template>/<case>'.",
    )
    discard_customization: bool = Field(
        default=False,
        description="Write over a definition reading customized or unknown (possibly an operator's); refused without "
        "it. What it supersedes stays stored, archived and readable by id.",
    )


def definition_drift_actions(
    corpus: Callable[[str], SeedCorpus],
    *,
    require_known_tools_allowed: Callable[[Sequence[str] | None], None],
    refuse_undeclared_world_seed: Callable[[EvalTemplate], None],
    refuse_undeliverable_template: Callable[[EvalTemplate], None],
    seeded_digests: Callable[[str], Mapping[str, str]] | None = None,
) -> tuple[Action, Action]:
    """``definitions_drift`` and ``definitions_reset``, for the host's catalogue.

    Args:
        corpus: The host's seed corpus for a scope (``load_seed_corpus(its directory, scope_id)``).
        require_known_tools_allowed: The host's tool-catalog check, as ``create_template`` takes it.
        refuse_undeclared_world_seed: The host's seed walk, as ``create_template`` takes it.
        refuse_undeliverable_template: The host's kind-capability check, as ``create_template`` takes it.
        seeded_digests: What the host seeded in a scope before ``.30``, by key (``corpus_digests`` over that
            corpus); ``None`` when it supplies none.

    Returns:
        The two actions.
    """

    def supplied(scope_id: str) -> Mapping[str, str] | None:
        return None if seeded_digests is None else seeded_digests(scope_id)

    async def drift(host: OpsHost, caller: Caller, params: DefinitionsDriftParams) -> DefinitionDrift:
        del params
        scope_id = caller.scope_id
        return await run_blocking(
            host.eval_host.blocking_executor,
            lambda: definition_drift(host.eval_host, corpus(scope_id), seeded_digests=supplied(scope_id)),
        )

    async def reset(host: OpsHost, caller: Caller, params: DefinitionsResetParams) -> DefinitionReset:
        scope_id = caller.scope_id
        return await run_blocking(
            host.eval_host.blocking_executor,
            lambda: reset_definitions(
                host.eval_host,
                corpus(scope_id),
                params.definitions,
                require_known_tools_allowed=require_known_tools_allowed,
                refuse_undeclared_world_seed=refuse_undeclared_world_seed,
                refuse_undeliverable_template=refuse_undeliverable_template,
                discard_customization=params.discard_customization,
                seeded_digests=supplied(scope_id),
            ),
        )

    return (
        Action(
            name="definitions_drift",
            summary="Read each seeded definition against the seed corpus: current, stale, customized, unknown or absent.",
            workflow=DEFINITIONS_WORKFLOW,
            permission="read",
            params=DefinitionsDriftParams,
            result=DefinitionDrift,
            handler=drift,
            render=lambda result: result.render(),
            example={},
            detail=(
                "The boot seed fills empty slots only, so a corpus edit reaches a scope that already holds the "
                "definition only through definitions_reset. A stale definition is one the seed wrote and nobody "
                "edited; a customized one changed since; an unknown one has no record of what was seeded (a row "
                "seeded before .30); an absent one the next boot seeds. Writes nothing."
            ),
        ),
        Action(
            name="definitions_reset",
            summary="Write the seed corpus's content for named definitions as new revisions; customized or unknown "
            "ones need discard_customization.",
            workflow=DEFINITIONS_WORKFLOW,
            permission="write",
            params=DefinitionsResetParams,
            result=DefinitionReset,
            handler=reset,
            render=lambda result: result.render(),
            example={"definitions": ["eval_template/support_triage"]},
            detail=(
                "Never in place: a template's new content is its next revision, the one it supersedes archived and "
                "readable by id, so results stay with the content they ran against and two revisions never pool. A "
                "judge case resets with its judge template, whose new revision holds the corpus's whole battery. "
                "All or nothing: a refusal writes nothing and names every refused definition."
            ),
        ),
    )


__all__ = ["DEFINITIONS_WORKFLOW", "DefinitionsDriftParams", "DefinitionsResetParams", "definition_drift_actions"]
