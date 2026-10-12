"""Whether a host's stored copy of a package prompt predates the package's own default.

Every catalogue prompt is store-master (:mod:`threetears.evals.kernel.prompts.catalogue`): a host seeds an empty
slot from the shipped default once and never overwrites a stored copy, so an operator can tune it with no deploy.
The cost is that a copy seeded before the package changed its default keeps being sent: when the change was one the
package's own validation depends on (``RubricDim.scale`` becoming required, say), every paid call that copy steers
can fail. Nothing in the store says so; this does, before the host pays for it.

**A digest names a default's content.** :func:`seed_digest` is the canonical digest of a seed's content: a text
seed's string, or a template's every field but its revision number (a promotion bumps that with the content it
changes, and the content is what the model reads). A host records the digest of what it seeded when it seeds a
slot, and compares later with :func:`prompt_drift`.

**A tuned prompt is not drift.** A stored copy that differs from the shipped default is either a default the
package has moved past or an operator's own wording, and only the record of what was seeded tells them apart: a copy
that still matches what was seeded is the old default (``stale``), and one that has changed since is the operator's
(``customized``). A host with no record, or whose prompt is deliberately its own, says so with ``customized=True``
and is never told it is stale.

**But a tuned prompt can still be written for a reply shape the engine no longer reads.** So each reading also
says which output contract the stored copy was written for (:mod:`threetears.evals.kernel.prompts.contract`),
against the one the shipped default states: ``current``, ``outdated``, ``undeclared``, or ``unbound`` for a prompt
with no contract. A copy that reads ``outdated`` or ``undeclared`` is refused when it is used, whether it is stale or
customized, and :attr:`PromptDrift.refused` says so here first, with a remedy, so an operator can see it before a
call is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from threetears.evals.kernel.prompts.contract import (
    REFUSED_CONTRACT_STATES,
    ContractState,
    OutputContract,
    contract_state,
    declared_contracts,
    require_reset,
    template_text,
)
from threetears.evals.kernel.prompts.seed import KIND_TEMPLATE, SeedPrompt, SeedTemplate
from threetears.evals.schema.hashing import canonical_digest

#: How a stored copy reads against the shipped default: the default itself, a default the package has moved past,
#: or the host's own.
DriftState = Literal["current", "stale", "customized"]


def seed_digest(seed: SeedTemplate | str) -> str:
    """The canonical digest of a seed's content: what a host records when it seeds a slot from it.

    Args:
        seed: A text seed's string, or a sectioned template — a shipped constant, or a stored copy rebuilt with
            :meth:`SeedTemplate.from_dict`.

    Returns:
        The full sha256 hex digest; equal for equal content whatever the template's revision number.
    """
    if isinstance(seed, str):
        return canonical_digest({"text": seed})
    payload = seed.to_dict()
    del payload["version"]
    return canonical_digest({"template": payload})


@dataclass(frozen=True)
class PromptDrift:
    """How one stored prompt reads against the package's shipped default.

    Attributes:
        prompt_type: The catalogue id of the slot.
        state: ``current``, ``stale`` or ``customized`` (the module docstring).
        shipped_digest: The digest of the package's default.
        stored_digest: The digest of the stored copy.
        remedy: What to do about a stale copy, or one a call refuses for its contract; empty otherwise.
        contract: Which output contract the stored copy was written for, against the one the shipped default states:
            ``current``, ``outdated``, ``undeclared``, or ``unbound`` for a prompt with no contract.
        expected_contract: The contract line the shipped default carries; empty for an unbound prompt.
    """

    prompt_type: str
    state: DriftState
    shipped_digest: str
    stored_digest: str
    remedy: str = ""
    contract: ContractState = "unbound"
    expected_contract: str = ""

    @property
    def stale(self) -> bool:
        """Whether the stored copy is a default the package has moved past."""
        return self.state == "stale"

    @property
    def refused(self) -> bool:
        """Whether a call steered by the stored copy is refused for the contract it was written for, edited or not."""
        return self.contract in REFUSED_CONTRACT_STATES


def _text(seed: SeedTemplate | str) -> str:
    return seed if isinstance(seed, str) else template_text(seed.sections)


def _expected_contract(prompt_type: str, shipped: SeedTemplate | str) -> OutputContract | None:
    """The contract the shipped default states, or ``None`` when it states none; one, since the engine checks one."""
    declared = declared_contracts(_text(shipped))
    if len(set(declared)) > 1:
        raise ValueError(
            f"the shipped {prompt_type!r} default declares {len(set(declared))} output contracts; a default is written "
            "for the one contract its reply is read by"
        )
    return declared[0] if declared else None


def prompt_drift(
    entry: SeedPrompt,
    *,
    shipped: SeedTemplate | str,
    stored: SeedTemplate | str,
    seeded_digest: str | None,
    reset: str,
    customized: bool = False,
) -> PromptDrift:
    """Read a stored copy of a catalogue prompt against the package's shipped default.

    Args:
        entry: The catalogue entry the slot was seeded from.
        shipped: The package's default, resolved from ``entry.seed_ref`` by import.
        stored: The copy the host's registry holds now, in the entry's kind: a string for a text seed, a
            :class:`SeedTemplate` (rebuilt from its payload) for a template seed.
        seeded_digest: :func:`seed_digest` of what the host seeded the slot with, as it recorded it then; ``None``
            when it recorded nothing.
        reset: How an operator of this host resets the slot to the shipped default — an action or a command,
            such as ``"call prompt_reset with prompt_type='eval_proposer'"``. The registry is the host's, so the
            engine cannot name it; the remedy does.
        customized: Whether the host holds the stored copy as deliberately its own — then it is never stale.

    Returns:
        The reading, with the output contract the stored copy was written for read against the one the shipped
        default states.

    Raises:
        TypeError: ``shipped`` or ``stored`` is not of the entry's kind.
        ValueError: ``shipped`` declares more than one output contract, or ``reset`` is blank.
    """
    reset = require_reset(reset)
    wanted: type = SeedTemplate if entry.kind == KIND_TEMPLATE else str
    for name, value in (("shipped", shipped), ("stored", stored)):
        if not isinstance(value, wanted):
            raise TypeError(
                f"prompt {entry.prompt_type!r} is a {entry.kind} seed, so its {name} copy is a {wanted.__name__}, "
                f"not a {type(value).__name__}"
            )
    shipped_digest, stored_digest = seed_digest(shipped), seed_digest(stored)
    if stored_digest == shipped_digest:
        state: DriftState = "current"
    elif customized or (seeded_digest is not None and stored_digest != seeded_digest):
        state = "customized"
    else:
        state = "stale"
    expected = _expected_contract(entry.prompt_type, shipped)
    contract = contract_state(_text(stored), expected)
    remedy = ""
    if state == "stale":
        remedy = (
            f"reset the {entry.prompt_type!r} slot to the shipped default ({reset}); or, if the stored copy is "
            "deliberately your own, keep it as customized"
        )
    if contract in REFUSED_CONTRACT_STATES and expected is not None:
        written = "for another output contract" if contract == "outdated" else "with no output contract stated"
        remedy = (
            f"{remedy + '. ' if remedy else ''}The stored {entry.prompt_type!r} copy was written {written}, so every "
            f"call it steers is refused until it is reset to the shipped default ({reset}), or, to keep an edit, its "
            "output instructions are rewritten for the current reply shape and its contract line made to read "
            f"'{expected.line}'"
        )
    return PromptDrift(
        prompt_type=entry.prompt_type,
        state=state,
        shipped_digest=shipped_digest,
        stored_digest=stored_digest,
        remedy=remedy,
        contract=contract,
        expected_contract="" if expected is None else expected.line,
    )


__all__ = ["DriftState", "PromptDrift", "prompt_drift", "seed_digest"]
