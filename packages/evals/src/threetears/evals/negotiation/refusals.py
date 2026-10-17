"""A provider refusing a request for a setting it carried: which setting, in its own words, and what that teaches.

A :class:`SettingRefusal` says which setting a provider refused (or that a router could not place an enforced
request), in the provider's own words, and what that teaches about the model (:meth:`SettingRefusal.teach`). A
reader makes one from what a client saw of a failed request (:mod:`threetears.evals.negotiation.readers` ships one
per common wire), and the negotiation loop decides whether and how to answer it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from threetears.evals.negotiation.settings import SettingFacts

__all__ = ["RefusedSetting", "SettingRefusal"]

#: What a provider refused:
#:
#: * ``temperature``: a temperature at all, or, when :attr:`SettingRefusal.beside_reasoning`, beside reasoning;
#: * ``effort_word``: one reasoning effort word (:attr:`SettingRefusal.word`), the others perhaps taken;
#: * ``effort``: an effort parameter at all;
#: * ``budget``: a reasoning token budget (Anthropic's thinking enabled);
#: * ``adaptive``: Anthropic's adaptive thinking;
#: * ``disable``: switching reasoning off (Anthropic's thinking disabled);
#: * ``format``: a strict reply schema;
#: * ``max_tokens``: an output cap above the model's own, which the provider named (:attr:`SettingRefusal.limit`);
#: * ``enforcement``: a router placing the request under ``require_parameters``, which names no parameter. The loop
#:   answers it in a fixed order: a temperature the request carried goes first, keeping enforcement, and only when
#:   the request carried none, or the resend without it found no endpoint either, enforcement itself, for that call
#:   alone. It is never learned beyond the call: a 404 can be an outage, and a client must not stay unenforced for
#:   the rest of its life.
RefusedSetting = Literal[
    "temperature", "effort_word", "effort", "budget", "adaptive", "disable", "format", "max_tokens", "enforcement"
]


@dataclass(frozen=True, slots=True)
class SettingRefusal:
    """A provider refusing a request for one setting it carried, in the provider's own words.

    :ivar setting: What was refused (:data:`RefusedSetting`).
    :ivar message: The provider's own message, for the log; never the whole body.
    :ivar word: For ``effort_word``, the word refused, or ``None`` for the one the request carried.
    :ivar accepted: For ``effort_word``, the words the provider listed as taken, when it listed them.
    :ivar limit: For ``max_tokens``, the model's own output cap, as the provider named it.
    :ivar beside_reasoning: For ``temperature``, that the refused request reasoned (an effort word other than
        ``none``, or a budget), so what it teaches holds beside reasoning only. Set by
        :func:`~threetears.evals.negotiation.negotiate` from the refused request; a reader leaves it ``False``.
    """

    setting: RefusedSetting
    message: str
    word: str | None = None
    accepted: tuple[str, ...] | None = None
    limit: int | None = None
    beside_reasoning: bool = False

    def teach(self, facts: SettingFacts) -> SettingFacts:
        """``facts`` with what this refusal says added.

        :param facts: What was known of the model.
        :ptype facts: SettingFacts
        :return: What is known once the refusal is.
        :rtype: SettingFacts
        """
        if self.setting == "temperature":
            if self.beside_reasoning:
                return replace(facts, temperature_beside_reasoning=False)
            return replace(facts, accepts_temperature=False)
        if self.setting == "effort_word":
            refused = facts.refused_efforts | ({self.word} if self.word else set())
            efforts = self.accepted if self.accepted is not None else facts.reasoning_efforts
            return replace(facts, refused_efforts=frozenset(refused), reasoning_efforts=efforts)
        if self.setting == "effort":
            return replace(facts, reasoning_efforts=())
        if self.setting == "budget":
            return replace(facts, reasoning_budget=False)
        if self.setting == "adaptive":
            return replace(facts, adaptive_reasoning=False)
        if self.setting == "disable":
            return replace(facts, reasoning_disable=False)
        if self.setting == "format":
            return replace(facts, structured_output=False)
        if self.setting == "max_tokens":
            if self.limit is None or (facts.max_output_tokens is not None and facts.max_output_tokens <= self.limit):
                return facts
            return replace(facts, max_output_tokens=self.limit)
        return replace(facts, enforcement=False)

    @property
    def label(self) -> str:
        """The refused setting, as a log names it."""
        return {
            "temperature": "a temperature beside reasoning" if self.beside_reasoning else "a temperature",
            "effort_word": f"reasoning effort {self.word!r}",
            "effort": "an effort parameter",
            "budget": "a thinking budget",
            "adaptive": "adaptive thinking",
            "disable": "thinking disabled",
            "format": "a strict reply format",
            "max_tokens": f"max_tokens above {self.limit}",
            "enforcement": "require_parameters",
        }[self.setting]
