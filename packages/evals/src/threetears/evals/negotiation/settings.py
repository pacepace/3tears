"""What to send a model for what a client was asked: the closest settings it is known to accept.

A client is asked for request settings (an output cap, a temperature, a reasoning word or budget, and per call whether
the reply format is enforced) and sends them to one model, which may not take all of them. :func:`closest_settings`
chooses what the request carries instead, from what is known of the model (:class:`SettingFacts`), and says why each
setting moved; :class:`WireSettings` is both the asked and the sent shape, and :meth:`WireSettings.reported` the sent
one as a completion's ``sent_settings``. Pure: nothing here sends, learns or logs. The loop that sends, reads a
refusal and learns from it is :mod:`threetears.evals.negotiation.loop`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final, Literal

from threetears.evals.schema import ClientRequestSettings

__all__ = [
    "ANTHROPIC_MIN_THINKING_BUDGET",
    "BUDGET_SHARES",
    "EFFORT_LADDER",
    "ProviderFamily",
    "SettingFacts",
    "WireSettings",
    "anthropic_thinking",
    "closest_settings",
    "wire_settings_for",
]

#: The request shape a client speaks, which decides how a setting goes on the wire: ``anthropic`` (Anthropic's
#: Messages API: thinking budgets, adaptive thinking with effort words, no temperature beside thinking), ``openai``
#: (OpenAI's ``reasoning_effort``, no budget) and ``router`` (a router's chat completions: ``reasoning.effort`` with
#: all seven words or a ``reasoning.max_tokens`` budget, never both, and enforcement as
#: ``provider.require_parameters``, under which a router that finds no endpoint taking every parameter refuses the
#: request). A client of an OpenAI-compatible gateway speaks ``openai``.
ProviderFamily = Literal["anthropic", "openai", "router"]

#: The engine's seven reasoning words (:data:`~threetears.evals.schema.REASONING_EFFORTS`) as a ladder, lowest
#: first: what "closest" is measured on. ``none`` is the off side of the line, every other word the on side.
EFFORT_LADDER: Final[tuple[str, ...]] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

#: The share of the output cap each effort word stands for: the ``router`` family's documented ratios for turning an
#: effort into a reasoning budget on a model that takes budgets, so a budget sent as its word comes back as about the
#: budget asked (the judge's 8192 of 10240 is 0.80, ``high``, and ``high`` is 0.80 of the cap again).
#: Read both ways: a budget asked of a model that takes words only is sent as the word whose share is nearest its
#: own (ties go to the lower word, so the boundaries 0.15, 0.35, 0.65 and 0.875 all go lower), and a word asked of a
#: model that takes budgets only is sent as that share of the cap. ``max`` has no share of its own above ``xhigh``'s
#: and is sent as ``xhigh``'s budget; ``none`` is no reasoning, never a budget.
BUDGET_SHARES: Final[Mapping[str, Fraction]] = {
    "minimal": Fraction("0.10"),
    "low": Fraction("0.20"),
    "medium": Fraction("0.50"),
    "high": Fraction("0.80"),
    "xhigh": Fraction("0.95"),
    "max": Fraction("0.95"),
}

#: Anthropic's floor on a thinking budget, in tokens: a budget below it is the caller's error, and a word sent as a
#: budget is never below it.
ANTHROPIC_MIN_THINKING_BUDGET: Final = 1024

#: The effort words each family's API names, which is what a model of it is sent from when nothing narrower is
#: known of the model. Anthropic's ``none`` is thinking disabled, and its others ride adaptive thinking as
#: ``output_config.effort``; OpenAI's ``reasoning_effort`` has no ``max``; a router types all seven.
_FAMILY_EFFORTS: Final[Mapping[ProviderFamily, tuple[str, ...]]] = {
    "anthropic": ("none", "low", "medium", "high", "xhigh", "max"),
    "openai": ("none", "minimal", "low", "medium", "high", "xhigh"),
    "router": EFFORT_LADDER,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class WireSettings:
    """Request settings as a client is asked for them, or as one request carries them.

    Beside :class:`~threetears.evals.schema.ClientRequestSettings` because a client can be built with no output cap
    and with a temperature, which that shape does not hold. :func:`closest_settings` takes the asked value and
    returns the sent one, which is what a client builds its request from; :meth:`reported` is the sent value as a
    completion's ``sent_settings``, and :attr:`temperature` its ``temperature``.

    Attributes:
        max_tokens: The output cap, or ``None`` for the provider's default (none is sent).
        temperature: The sampling temperature, or ``None`` for none.
        reasoning_effort: An effort word on :data:`EFFORT_LADDER`, or ``None``.
        reasoning_max_tokens: A reasoning budget in tokens, or ``None``; never beside ``reasoning_effort``.
        strict_output: Whether the request is enforced: a strict reply schema a direct provider refuses rather than
            ignores, or a router's ``require_parameters``.
        unsent_reasoning_max_tokens: On sent settings, the reasoning budget asked that the request could not carry
            as a budget (the wire takes effort words only, or the model refused a budget), so it went as
            ``reasoning_effort`` or as no reasoning parameter; ``None`` when no budget was asked or it went as one.
            Never beside ``reasoning_max_tokens``, and never set on asked settings.
    """

    max_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    reasoning_max_tokens: int | None = None
    strict_output: bool = False
    unsent_reasoning_max_tokens: int | None = None

    def reported(self) -> ClientRequestSettings | None:
        """These settings as a completion's ``sent_settings``, or ``None`` when they name no output cap.

        Returns:
            The settings value; the temperature is not part of it (a completion reports it as ``temperature``).
        """
        if self.max_tokens is None:
            return None
        return ClientRequestSettings(
            max_tokens=self.max_tokens,
            reasoning_max_tokens=self.reasoning_max_tokens,
            reasoning_effort=self.reasoning_effort,
            strict_output=self.strict_output,
            unsent_reasoning_max_tokens=self.unsent_reasoning_max_tokens,
        )


def wire_settings_for(settings: ClientRequestSettings, temperature: float | None) -> WireSettings:
    """What a client asks for when it is asked for ``settings`` at ``temperature``: the engine's values, as wire ones.

    A host's own client is built from the request settings the engine stamps on a run for a role
    (:data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS`, :data:`~threetears.evals.run.SIMULATOR_REQUEST_SETTINGS`)
    and the temperature the engine names on each request for a client; this builds what
    :func:`~threetears.evals.negotiation.negotiate` takes as ``asked`` from them, so a host never hand-copies the
    engine's constants and never drifts from what the run records as asked. ``strict_output`` is the role's asked
    enforcement; a client whose calls differ (one without a reply format) replaces it per call.

    Args:
        settings: The request settings the client is asked for.
        temperature: The temperature it is asked for, or ``None`` for none.

    Returns:
        The asked settings, every field as ``settings`` holds it.
    """
    return WireSettings(
        max_tokens=settings.max_tokens,
        temperature=temperature,
        reasoning_effort=settings.reasoning_effort,
        reasoning_max_tokens=settings.reasoning_max_tokens,
        strict_output=settings.strict_output,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SettingFacts:
    """What is known of the request settings one model accepts: each fact ``None`` while nothing says.

    A caller fills what it knows before any request (a registry's capability fields of the same names, a router's
    catalogue through :func:`~threetears.evals.negotiation.facts_from_router_catalogue`, its own configuration), and
    a refusal the provider answered fills the rest for as long as the caller's
    :class:`~threetears.evals.negotiation.LearnedSettings` lives: the process, or the client factory that keeps it.
    Unknown is sent as asked.

    Attributes:
        accepts_temperature: Whether a temperature is taken.
        temperature_beside_reasoning: Whether a temperature is taken beside reasoning (an effort word other than
            ``none``, or a budget). A temperature refused only while the request reasoned is learned here, not as
            ``accepts_temperature``, so a call that does not reason still sends one.
        reasoning_efforts: The provider-native effort words the model takes; ``()`` when it takes no effort
            parameter at all. Anthropic's off switch is not a word here but ``reasoning_disable``.
        refused_efforts: Words the provider refused for this model, whatever ``reasoning_efforts`` says.
        reasoning_budget: Whether a reasoning token budget is taken (Anthropic's ``thinking.budget_tokens``, a
            router's ``reasoning.max_tokens``).
        adaptive_reasoning: Whether Anthropic's adaptive thinking is taken, which its effort words ride on.
        reasoning_disable: Whether reasoning can be switched off (Anthropic's thinking disabled, or a provider's
            ``none``).
        structured_output: Whether a strict ``json_schema`` reply format is taken.
        enforcement: Whether a router finds an endpoint for the model's requests under ``require_parameters``.
        max_output_tokens: The model's own output cap, or ``None`` when unknown.
    """

    accepts_temperature: bool | None = None
    temperature_beside_reasoning: bool | None = None
    reasoning_efforts: tuple[str, ...] | None = None
    refused_efforts: frozenset[str] = frozenset()
    reasoning_budget: bool | None = None
    adaptive_reasoning: bool | None = None
    reasoning_disable: bool | None = None
    structured_output: bool | None = None
    enforcement: bool | None = None
    max_output_tokens: int | None = None

    def over(self, lower: SettingFacts) -> SettingFacts:
        """These facts, each unknown one taken from ``lower``; refused words from both.

        Args:
            lower: The facts of a source below this one.

        Returns:
            The combined facts.
        """
        return SettingFacts(
            accepts_temperature=_first(self.accepts_temperature, lower.accepts_temperature),
            temperature_beside_reasoning=_first(self.temperature_beside_reasoning, lower.temperature_beside_reasoning),
            reasoning_efforts=self.reasoning_efforts if self.reasoning_efforts is not None else lower.reasoning_efforts,
            refused_efforts=self.refused_efforts | lower.refused_efforts,
            reasoning_budget=_first(self.reasoning_budget, lower.reasoning_budget),
            adaptive_reasoning=_first(self.adaptive_reasoning, lower.adaptive_reasoning),
            reasoning_disable=_first(self.reasoning_disable, lower.reasoning_disable),
            structured_output=_first(self.structured_output, lower.structured_output),
            enforcement=_first(self.enforcement, lower.enforcement),
            max_output_tokens=self.max_output_tokens if self.max_output_tokens is not None else lower.max_output_tokens,
        )


def _first(known: bool | None, fallback: bool | None) -> bool | None:
    """``known`` unless it is unknown, then ``fallback``."""
    return known if known is not None else fallback


def _rung(word: str) -> int:
    """Where ``word`` stands on :data:`EFFORT_LADDER`."""
    return EFFORT_LADDER.index(word)


def _closest_word(word: str, on: Sequence[str], off: bool) -> str | None:
    """The accepted word closest to ``word``, on its own side of the on/off line.

    ``none`` asked of a model that cannot switch reasoning off crosses the line to the lowest word that reasons. A
    reasoning word asked of a model that takes none never crosses to ``none``: no reasoning parameter is sent, which
    is what such a model does by default, rather than an off switch nobody asked for.

    Args:
        word: The word asked for, on :data:`EFFORT_LADDER`.
        on: The accepted words that reason (every one but ``none``).
        off: Whether reasoning can be switched off (``none`` is accepted).

    Returns:
        The word to send, the lower of two equally close; ``None`` for no reasoning parameter.
    """
    if word == "none":
        if off:
            return "none"
        return min(on, key=_rung) if on else None
    if not on:
        return None
    return min(on, key=lambda candidate: (abs(_rung(candidate) - _rung(word)), _rung(candidate)))


def _word_for_share(budget: int, cap: int) -> str:
    """The effort word whose share of the cap (:data:`BUDGET_SHARES`) is nearest ``budget / cap``; ties go lower."""
    share = Fraction(budget, cap)
    return min(
        (word for word in BUDGET_SHARES if word != "max"),
        key=lambda word: (abs(BUDGET_SHARES[word] - share), _rung(word)),
    )


def _takes_budget(family: ProviderFamily, facts: SettingFacts) -> bool:
    """Whether a reasoning budget goes as a budget: Anthropic's ``thinking.budget_tokens``, a router's
    ``reasoning.max_tokens``; never OpenAI's, whose request has no budget parameter. A router model whose catalogue
    row lists no ``reasoning`` (or that refused every effort) takes no reasoning parameter at all."""
    if facts.reasoning_budget is False:
        return False
    return family == "anthropic" or (family == "router" and facts.reasoning_efforts != ())


def _budget_for_word(word: str, cap: int) -> int | None:
    """A thinking budget for ``word``: its share of the cap, at least Anthropic's floor and below the cap, if any is."""
    if cap <= ANTHROPIC_MIN_THINKING_BUDGET:
        return None
    share = BUDGET_SHARES.get(word, BUDGET_SHARES["minimal"])
    return min(max(round(share * cap), ANTHROPIC_MIN_THINKING_BUDGET), cap - 1)


def _vocabulary(family: ProviderFamily, facts: SettingFacts) -> tuple[list[str], bool]:
    """The words the model is sent from: those that reason, lowest first, and whether reasoning can be switched off."""
    words = facts.reasoning_efforts if facts.reasoning_efforts is not None else _FAMILY_EFFORTS[family]
    on = [word for word in EFFORT_LADDER if word != "none" and word in words and word not in facts.refused_efforts]
    if family == "anthropic":
        # Anthropic's off switch is thinking disabled, which no effort list names.
        off = facts.reasoning_disable is not False and "none" not in facts.refused_efforts
    else:
        off = "none" in words and "none" not in facts.refused_efforts and facts.reasoning_disable is not False
    return on, off


def _closest_reasoning(
    family: ProviderFamily,
    asked: WireSettings,
    cap: int | None,
    facts: SettingFacts,
    notes: list[str],
) -> tuple[str | None, int | None]:
    """The reasoning word and budget to send: one, the other, or neither."""
    takes_budget = _takes_budget(family, facts)
    effort, budget = asked.reasoning_effort, asked.reasoning_max_tokens
    if budget is not None:
        if takes_budget:
            if cap is None or budget < cap:
                return None, budget
            if cap - 1 >= (ANTHROPIC_MIN_THINKING_BUDGET if family == "anthropic" else 1):
                notes.append(f"reasoning budget {budget} sent as {cap - 1}, below the model's output cap {cap}")
                return None, cap - 1
        if asked.max_tokens is None:
            if takes_budget:
                return None, budget
            notes.append(f"reasoning budget {budget} sent as no reasoning: there is no max_tokens to take its share of")
            return None, None
        effort = _word_for_share(budget, asked.max_tokens)
        why = "the model takes no reasoning budget" if not takes_budget else f"the model's output cap is {cap}"
        notes.append(
            f"reasoning budget {budget} of max_tokens {asked.max_tokens} (share "
            f"{float(Fraction(budget, asked.max_tokens)):.2f}) taken as effort {effort!r}: {why}"
        )
    if effort is None:
        return None, None
    on, off = _vocabulary(family, facts)
    if family == "anthropic" and not on and takes_budget and effort != "none":
        if cap is None:
            notes.append(
                f"effort {effort!r} sent as no reasoning: the model takes budgets only, and no max_tokens sizes one"
            )
            return None, None
        sent_budget = _budget_for_word(effort, cap)
        notes.append(
            f"effort {effort!r} sent as a thinking budget of {sent_budget} ({float(BUDGET_SHARES[effort]):.2f} of "
            f"max_tokens {cap}): the model takes budgets, not effort words"
            if sent_budget is not None
            else f"effort {effort!r} sent as no reasoning: the model takes budgets only, and max_tokens {cap} leaves "
            "no room for one"
        )
        return None, sent_budget
    word = _closest_word(effort, on, off)
    if word is None:
        notes.append(f"effort {effort!r} sent as no reasoning parameter: the model takes none")
    elif word != effort:
        notes.append(f"effort {effort!r} sent as {word!r}, the closest word the model takes")
    return word, None


def closest_settings(
    asked: WireSettings, facts: SettingFacts, provider_family: ProviderFamily
) -> tuple[WireSettings, tuple[str, ...]]:
    """What to send for ``asked``: the closest settings ``facts`` say the model accepts, and why each one moved.

    Pure: the same inputs give the same answer, and anything ``facts`` leave unknown is sent as asked, for the
    provider to refuse and the caller to learn from (:func:`~threetears.evals.negotiation.negotiate`). The rules,
    per setting:

    * **Output cap**: above the model's known cap (``max_output_tokens``), that cap.
    * **Effort word**: the accepted word closest on :data:`EFFORT_LADDER`, on the same side of the on/off line
      (``none`` is off), crossing it only for ``none`` asked of a model that cannot switch reasoning off, which gets
      its lowest word; a tie goes to the lower word, which never spends more than was asked. The words a model is
      sent from are its declared ones, else its family's (``max`` on ``openai`` is ``xhigh``, ``minimal`` on
      ``anthropic`` is ``low``), less any it refused. A model that takes no effort word is sent none, except an
      Anthropic model that takes a budget, which is sent the word's share of the cap (:data:`BUDGET_SHARES`), at
      least :data:`ANTHROPIC_MIN_THINKING_BUDGET` and below the cap.
    * **Budget**: as asked where the model takes one: on Anthropic as ``thinking.budget_tokens``, on a router as
      ``reasoning.max_tokens`` (lowered below a cap the clamp moved under it). Otherwise the effort word its share of
      ``max_tokens`` maps to (:data:`BUDGET_SHARES`), then as a word, and the sent settings name the budget that
      went unsent (``unsent_reasoning_max_tokens``). OpenAI's chat completions take no budget, so there a budget is
      always a word. A budget not below the asked ``max_tokens`` is the caller's error, refused naming both: it
      leaves the answer no room, and is never clamped.
    * **Temperature**: dropped when the model takes none; beside reasoning (an effort word other than ``none``, or a
      budget) when it takes none there (``temperature_beside_reasoning``); and on Anthropic beside any thinking (a
      budget, or an effort word over adaptive thinking), which the API refuses.
    * **Strict output**: kept unless the model takes no strict schema, or, on a router, no endpoint serves its
      requests under ``require_parameters``; the request then carries the format unenforced.

    Args:
        asked: What the client was asked for, ``strict_output`` being the call's own.
        facts: What is known of the model.
        provider_family: The request shape the client speaks.

    Returns:
        The settings the request carries, and one sentence per setting that moved, for a log.

    Raises:
        ValueError: ``asked`` carries a reasoning budget not below its ``max_tokens``, naming both.
    """
    budget_asked = asked.reasoning_max_tokens
    if budget_asked is not None and asked.max_tokens is not None and budget_asked >= asked.max_tokens:
        raise ValueError(
            f"a reasoning budget ({budget_asked}) not below max_tokens ({asked.max_tokens}) leaves the answer no "
            "room: ask for a max_tokens above the budget"
        )
    notes: list[str] = []
    cap = asked.max_tokens
    if cap is not None and facts.max_output_tokens is not None and cap > facts.max_output_tokens:
        notes.append(f"max_tokens {cap} sent as {facts.max_output_tokens}, the model's output cap")
        cap = facts.max_output_tokens
    effort, budget = _closest_reasoning(provider_family, asked, cap, facts, notes)
    temperature = asked.temperature
    if temperature is not None and facts.accepts_temperature is False:
        notes.append(f"temperature {temperature} sent as none: the model takes no temperature")
        temperature = None
    elif temperature is not None and facts.temperature_beside_reasoning is False and reasons(effort, budget):
        notes.append(f"temperature {temperature} sent as none: the model takes no temperature beside reasoning")
        temperature = None
    elif temperature is not None and provider_family == "anthropic" and _anthropic_thinks(effort, budget, facts):
        notes.append(f"temperature {temperature} sent as none: Anthropic takes no temperature beside thinking")
        temperature = None
    strict = asked.strict_output
    if strict and facts.structured_output is False:
        notes.append("the reply format sent unenforced: the model takes no strict schema")
        strict = False
    elif strict and provider_family == "router" and facts.enforcement is False:
        notes.append("the reply format sent without require_parameters: no endpoint serves the model under it")
        strict = False
    sent = WireSettings(
        max_tokens=cap,
        temperature=temperature,
        reasoning_effort=effort,
        reasoning_max_tokens=budget,
        strict_output=strict,
        # A budget asked and not sent as one is named, never left for a reader to infer from the word beside it.
        unsent_reasoning_max_tokens=budget_asked if budget_asked is not None and budget is None else None,
    )
    return sent, tuple(notes)


def reasons(effort: str | None, budget: int | None) -> bool:
    """Whether a request with this reasoning word and budget reasons: a budget, or a word other than ``none``.

    Args:
        effort: The effort word the request carries, or ``None``.
        budget: The reasoning budget it carries, or ``None``.

    Returns:
        Whether it reasons.
    """
    return budget is not None or effort not in (None, "none")


def _anthropic_thinks(effort: str | None, budget: int | None, facts: SettingFacts) -> bool:
    """Whether an Anthropic request with this reasoning carries thinking: a budget, or a word over adaptive thinking."""
    return budget is not None or (effort not in (None, "none") and facts.adaptive_reasoning is not False)


def anthropic_thinking(sent: WireSettings, facts: SettingFacts) -> dict[str, str | int] | None:
    """The ``thinking`` an Anthropic request carries for ``sent``, or ``None`` for none.

    A budget is thinking enabled with that budget; ``none`` is thinking disabled; any other word rides adaptive
    thinking, unless the model is known not to take it, when the word goes as ``output_config.effort`` alone. The
    word itself is always the request's ``output_config.effort`` (except ``none``), which a client sets beside this.

    Args:
        sent: What the request carries (:func:`closest_settings`' answer).
        facts: What was known of the model when it was chosen.

    Returns:
        The request's ``thinking`` object, or ``None`` when the request carries none.
    """
    if sent.reasoning_max_tokens is not None:
        return {"type": "enabled", "budget_tokens": sent.reasoning_max_tokens}
    if sent.reasoning_effort == "none":
        return {"type": "disabled"}
    if sent.reasoning_effort is not None and facts.adaptive_reasoning is not False:
        return {"type": "adaptive"}
    return None
