"""Which output contract a stored prompt was written for, so a copy written for an older one is refused unpaid.

Every catalogue prompt is store-master (:mod:`threetears.evals.kernel.prompts.catalogue`), so a copy seeded before the
package changed what its reply is read into keeps being sent. The content digest
(:mod:`threetears.evals.kernel.prompts.drift`) sees that a copy differs from the shipped default, but not whether the
difference matters: an operator's tuned wording differs by design, and a copy seeded before ``RubricDim.scale``
became required differs fatally. A host met the second and read it as the first: every rubric draft failed after its
call was paid for, with a parse error that named a field rather than the stale prompt.

**A prompt states the contract it was written for, in its own text.** A prompt whose reply the engine reads into a
typed model ends with a contract line, :attr:`OutputContract.line`: ``eval-output-contract: <name> <fingerprint>``.
The fingerprint is the digest of the reply's schema as the call sends it, with its annotations (``description``,
``title``, ``examples``, ``default``) dropped, so rewording a field's documentation moves nothing and a structural
change (a field made required, a level added, a type changed) moves it; and of a hand-bumped ``revision``, for a
rule the reply is refused by that the schema cannot carry (a dimension name's namespacing). The shipped default
carries the current line because it is built from the same contract, so the line costs no store field and no host
change: it travels wherever the prompt's text does, and an operator sees it where they edit the prompt.

**The call checks it before it is planned.** :func:`require_output_contract` reads the rendered prompt and refuses,
naming the prompt and the remedy, a prompt that declares another fingerprint (``outdated``) or none
(``undeclared`` — a copy seeded before contracts were stated). A copy that declares the current one passes,
however else an operator has edited it: the line is the operator's statement that the edit was written for this
contract, and the strict ``response_format`` the call sends, with the validation of its reply, still stands behind
it. A prompt with no output contract (the analysis generator's, whose shape is sent as a schema and whose grammar
is appended from code) is ``unbound`` and never refused.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from threetears.evals.kernel.errors import ConflictError
from threetears.evals.kernel.prompts.seed import SeedSection
from threetears.evals.schema.hashing import canonical_digest

#: How a stored copy reads against the output contract the engine reads its reply by: written for it, written for
#: an earlier one, declaring none, or a prompt with no contract to be written for.
ContractState = Literal["current", "outdated", "undeclared", "unbound"]

#: The contract states a call refuses: the copy was written for another contract, or says nothing of one.
REFUSED_CONTRACT_STATES: frozenset[ContractState] = frozenset({"outdated", "undeclared"})

#: The words a contract line starts with. Hyphenated so no field or action name can be mistaken for it.
CONTRACT_MARKER = "eval-output-contract:"

#: Hex digits of the digest a line carries: enough that two contracts never share one, few enough to read.
FINGERPRINT_LENGTH = 16

#: Schema keywords that document a field rather than shape it, dropped before the digest.
_ANNOTATIONS = frozenset({"description", "title", "examples", "default"})

#: Keywords whose value maps names to subschemas, so its keys are field or definition names and never keywords.
_NAMED_SUBSCHEMAS = frozenset({"properties", "$defs", "definitions", "patternProperties"})

_LINE = re.compile(rf"^[ \t]*{re.escape(CONTRACT_MARKER)}[ \t]*(\S+)[ \t]+([0-9a-f]+)[ \t]*$", re.MULTILINE)


def _structure(node: Any, *, names: bool = False) -> Any:
    """``node`` with every annotation keyword dropped, recursively; a name map keeps its keys whatever they spell."""
    if isinstance(node, list):
        return [_structure(item) for item in node]
    if not isinstance(node, dict):
        return node
    if names:
        return {key: _structure(value) for key, value in node.items()}
    return {
        key: _structure(value, names=key in _NAMED_SUBSCHEMAS) for key, value in node.items() if key not in _ANNOTATIONS
    }


@dataclass(frozen=True)
class OutputContract:
    """The contract a prompt's reply is read by: its name and the fingerprint of its shape.

    :ivar name: The reply's name, as the call's ``response_format`` carries it (``rubric_proposal``).
    :ivar fingerprint: :data:`FINGERPRINT_LENGTH` hex digits of the digest of the reply's structure and revision.
    """

    name: str
    fingerprint: str

    @classmethod
    def of(cls, name: str, schema: Mapping[str, Any], *, revision: int) -> OutputContract:
        """The contract a reply of ``schema``'s shape is read by.

        :param name: The reply's name.
        :ptype name: str
        :param schema: The JSON schema the call sends for the reply; its annotations do not count.
        :ptype schema: Mapping[str, Any]
        :param revision: Bumped by hand when a rule the reply is refused by changes outside the schema, so the prompts
            that teach the old rule are refused with the rest.
        :ptype revision: int
        :return: The contract.
        :rtype: OutputContract
        """
        digest = canonical_digest({"name": name, "revision": revision, "schema": _structure(dict(schema))})
        return cls(name=name, fingerprint=digest[:FINGERPRINT_LENGTH])

    @property
    def line(self) -> str:
        """The line a prompt written for this contract carries: what the shipped default ends with."""
        return f"{CONTRACT_MARKER} {self.name} {self.fingerprint}"


def declared_contracts(text: str) -> tuple[OutputContract, ...]:
    """Every contract line ``text`` carries, in order.

    :param text: A prompt: rendered, a text seed's string, or a template's sections joined.
    :ptype text: str
    :return: The contracts its lines declare; empty when it carries none.
    :rtype: tuple[OutputContract, ...]
    """
    return tuple(OutputContract(name=name, fingerprint=digest) for name, digest in _LINE.findall(text))


def contract_state(text: str, contract: OutputContract | None) -> ContractState:
    """How ``text`` reads against ``contract``.

    :param text: The prompt as it is stored or rendered.
    :ptype text: str
    :param contract: The contract the engine reads the prompt's reply by, or ``None`` for a prompt that has none.
    :ptype contract: OutputContract | None
    :return: ``unbound`` when ``contract`` is ``None``; ``current`` when every line ``text`` carries declares
        ``contract`` and it carries one; ``undeclared`` when it carries none; ``outdated`` otherwise — a line naming
        another fingerprint, or another contract, beside or instead of the current one.
    :rtype: ContractState
    """
    if contract is None:
        return "unbound"
    declared = declared_contracts(text)
    if not declared:
        return "undeclared"
    return "current" if all(found == contract for found in declared) else "outdated"


def template_text(sections: tuple[SeedSection, ...]) -> str:
    """A sectioned template's text as its contract is read from: every section's literal body, in order.

    :param sections: A :class:`~threetears.evals.kernel.prompts.seed.SeedTemplate`'s sections.
    :ptype sections: tuple[SeedSection, ...]
    :return: The bodies, one per line.
    :rtype: str
    """
    return "\n".join(section.content_template for section in sections)


def _contract_remedy(prompt_type: str, contract: OutputContract, reset: str) -> str:
    """What an operator does about a stored copy written for another contract, or for none: the host's own reset."""
    return (
        f"reset the stored '{prompt_type}' prompt to the default this package ships ({reset}); or, to keep an edited "
        "copy, rewrite its output instructions for the current reply shape and make its contract line read "
        f"'{contract.line}'"
    )


class PromptContractError(ConflictError):
    """A stored prompt was written for another output contract than the one its reply is read by (status 409).

    Raised before the call is planned, so nothing is priced or paid for. A conflict because the stored copy, not the
    request, is what has to change, and the remedy names how.

    :ivar prompt_type: The catalogue id of the prompt that was refused.
    :ivar state: ``outdated`` or ``undeclared``.
    :ivar expected: The contract the engine reads the reply by.
    :ivar declared: The contracts the prompt's lines declare; empty when undeclared.
    """

    def __init__(
        self,
        prompt_type: str,
        *,
        state: ContractState,
        expected: OutputContract,
        declared: tuple[OutputContract, ...],
        reset: str,
    ) -> None:
        """Build the refusal of ``prompt_type``'s stored copy.

        :param prompt_type: The catalogue id of the prompt.
        :ptype prompt_type: str
        :param state: How the copy read: ``outdated`` or ``undeclared``.
        :ptype state: ContractState
        :param expected: The contract the engine reads the reply by.
        :ptype expected: OutputContract
        :param declared: The contracts the copy's lines declare.
        :ptype declared: tuple[OutputContract, ...]
        :param reset: How an operator of this host resets the slot to the shipped default.
        :ptype reset: str
        """
        self.prompt_type = prompt_type
        self.state = state
        self.expected = expected
        self.declared = declared
        if declared:
            found = ", ".join(f"{entry.name} {entry.fingerprint}" for entry in declared)
            said = (
                f"was written for output contract {found}, and its reply is now read as "
                f"{expected.name} {expected.fingerprint}"
            )
        else:
            said = (
                f"states no output contract, so it predates contract lines and may teach a reply shape the "
                f"engine no longer reads; its reply is read as {expected.name} {expected.fingerprint}"
            )
        super().__init__(
            f"the stored {prompt_type!r} prompt {said}. Refused before the call, so nothing was paid for",
            remedy=_contract_remedy(prompt_type, expected, require_reset(reset)),
        )


def require_reset(reset: str) -> str:
    """``reset``, refused when blank: a remedy that names no way to reset the slot leaves the operator nothing to do.

    :param reset: The host's words for resetting a prompt slot.
    :ptype reset: str
    :return: ``reset``, stripped.
    :rtype: str
    :raises ValueError: ``reset`` is blank.
    """
    if not reset.strip():
        raise ValueError(
            "reset is blank: name how an operator of this host resets a prompt slot to the shipped default (an "
            "action or a command), since the prompt registry is the host's and the engine cannot name it"
        )
    return reset.strip()


def require_output_contract(prompt_type: str, text: str, contract: OutputContract, *, reset: str) -> None:
    """Refuse a prompt that was not written for ``contract``.

    :param prompt_type: The catalogue id of the prompt, named in the refusal.
    :ptype prompt_type: str
    :param text: The prompt as the call would send it.
    :ptype text: str
    :param contract: The contract the call reads its reply by.
    :ptype contract: OutputContract
    :param reset: How an operator of this host resets the slot to the shipped default, named in the remedy
        (an action or a command).
    :ptype reset: str
    :raises PromptContractError: ``text`` declares another contract, or none.
    :raises ValueError: ``reset`` is blank.
    """
    require_reset(reset)
    state = contract_state(text, contract)
    if state in REFUSED_CONTRACT_STATES:
        raise PromptContractError(
            prompt_type, state=state, expected=contract, declared=declared_contracts(text), reset=reset
        )


__all__ = [
    "CONTRACT_MARKER",
    "REFUSED_CONTRACT_STATES",
    "ContractState",
    "OutputContract",
    "PromptContractError",
    "contract_state",
    "declared_contracts",
    "require_output_contract",
    "template_text",
]
