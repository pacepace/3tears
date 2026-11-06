"""The output contract a rubric proposal is written for, below both the proposer and its shipped default prompts.

The default prompts end with the contract's line (:mod:`~threetears.evals.gen.prompts.proposer`,
:mod:`~threetears.evals.gen.prompts.boundary_gen`), and the proposer
(:mod:`~threetears.evals.gen.proposers`) sends the defaults and refuses a stored prompt without the line, so the
contract lives here, where all three import it.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.prompts.contract import OutputContract
from threetears.evals.schema.models import SCALE_LEVELS, RubricProposal

__all__ = ["PROPOSAL_RESPONSE_FORMAT", "RUBRIC_PROPOSAL_CONTRACT"]

#: Every level any scale answers: what a drafted dimension's scoring guide may describe.
_GUIDE_LEVELS: tuple[str, ...] = tuple(dict.fromkeys(level for levels in SCALE_LEVELS.values() for level in levels))


def _proposal_schema() -> dict[str, Any]:
    """:class:`~threetears.evals.schema.models.RubricProposal`'s JSON schema, its one map field closed.

    A strict schema admits no object with arbitrary keys, and ``RubricDim.scoring_guide`` maps a scale's levels to
    descriptors. Which levels depends on the dimension's own ``scale``, which a strict schema cannot condition on,
    so the guide is sent as an object holding every level any scale has, each a descriptor or ``null``: the
    unwritten levels are dropped before the draft is validated (by
    :func:`~threetears.evals.gen.proposers.propose_draft`), and the model still refuses a guide describing a level its dimension's scale does not have.

    A dimension's ``judge`` declaration is left out: which judge scores a dimension is the operator's to declare when
    the draft becomes a template (``create_template`` refuses a dimension that declares none), never a proposer's to
    guess, and leaving it out keeps the contract every stored proposer prompt was written against.
    """
    schema = RubricProposal.model_json_schema()
    del schema["$defs"]["RubricDim"]["properties"]["judge"]
    schema["$defs"]["RubricDim"]["properties"]["scoring_guide"] = {
        "type": "object",
        "description": "Each level of the dimension's scale mapped to what earns it; null for a level not described "
        "and for every level its scale does not have.",
        "properties": {level: {"anyOf": [{"type": "string"}, {"type": "null"}]} for level in _GUIDE_LEVELS},
    }
    return schema


#: What a proposal asks for: the draft model as a strict schema, so a provider that enforces it cannot write a
#: draft the model refuses — a dimension with no ``scale``, say — whatever the stored prompt says.
PROPOSAL_RESPONSE_FORMAT: dict[str, Any] = strict_response_format("rubric_proposal", _proposal_schema())

#: The output contract both proposer prompts are written for: the draft's schema as the call sends it, and a
#: revision for the rule it is refused by that the schema cannot state (a dimension name's ``<context>.<dim>``
#: namespacing, :data:`~threetears.evals.schema.models.DimName`). Bump the revision when such a rule changes. Each
#: shipped default ends with its line, and :func:`~threetears.evals.gen.proposers.plan_draft` refuses a stored prompt
#: that does not (:mod:`threetears.evals.kernel.prompts.contract`); ``tests/test_prompt_contract.py`` pins its fingerprint, so a
#: change that moves it — and so refuses every stored proposer prompt until it is reset — is made on purpose.
RUBRIC_PROPOSAL_CONTRACT = OutputContract.of(
    "rubric_proposal", PROPOSAL_RESPONSE_FORMAT["json_schema"]["schema"], revision=1
)
