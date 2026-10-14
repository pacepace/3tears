"""The negotiation loop: send the closest settings, and on a refusal learn it and send again, never failing for a lack.

:func:`negotiate` is the one implementation of "a model that lacks a request feature never fails the request" (owner
ruling, 2026-10-13), for the shipped client and for a host's own client alike. The caller brings the sending
(``send``, its own HTTP or SDK call) and the reading of a failure (``read_refusal``, usually one of the shipped
readers over the status and body it saw); the loop chooses what to send (:func:`closest_settings`), decides whether a
refusal is one to answer, in what order, and how often, keeps what a served request proved
(:class:`LearnedSettings`, for the store's life: the process, or the factory that keeps it) and says what the served
request carried (:class:`Negotiated`), which is what the caller's completion reports.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from threetears.evals.negotiation.refusals import SettingRefusal
from threetears.evals.negotiation.settings import (
    ProviderFamily,
    SettingFacts,
    WireSettings,
    closest_settings,
    reasons,
)
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.schema.logs import get_logger

__all__ = ["MAX_RESENDS", "LearnedSettings", "Negotiated", "negotiate"]

log = get_logger(__name__)

#: The most times one call is resent for a refused setting: one per setting a client can drop or move (the
#: temperature, the reasoning parameter, the reply format, enforcement). A refused send is not billed, so a call's
#: price ceiling is one request's, but it is not free in time: each refused send can take as long as the caller's own
#: timeout for one request before it is refused, so one :func:`negotiate` can take up to ``MAX_RESENDS + 1`` request
#: timeouts, and a wall-clock ceiling a caller holds around it allows ``MAX_RESENDS`` × that timeout past the one
#: served request's.
MAX_RESENDS: Final = 4

#: The refusals of a reasoning effort: at most two are answered in one call (a word, then any word or the parameter),
#: and the second always as the parameter, so a call never walks the effort ladder into a failure.
_EFFORT_REFUSALS: Final = frozenset({"effort_word", "effort"})


class LearnedSettings:
    """What providers refused, per provider and model, kept for as long as the callers that share it live.

    One store is shared by every client that should not meet a refusal another already met: the shipped client keeps
    one per :class:`~threetears.evals.providers.Catalogue` (every model sharing it), and a host's own client
    keeps one per process or per client factory, as it chooses. **A learned fact lasts for the store's life**, which
    is the process, or the factory that keeps it: nothing expires it. Not persisted: a provider may start accepting
    a setting, and each completion already records what its request carried. A refusal wins over what the caller
    declared of the model, and a contradiction is logged as the declaration's error.

    Only what a served request proved is kept (:func:`negotiate` decides which refusals those are), and never a
    router's refusal to place an enforced request (``enforcement``): a 404 can be an outage, so each call starts
    enforced and drops enforcement for that call alone. A temperature refused while the request reasoned is kept as
    a refusal beside reasoning (:attr:`SettingFacts.temperature_beside_reasoning`), so a call that does not reason
    still sends one.

    Safe under concurrent calls, on one event loop or across threads: each fact is read and added under a lock, and
    calls refused together each resend; the first to be served records the fact, and it is logged once.
    """

    def __init__(
        self,
        *,
        client: str = "eval.negotiation",
        declared_by: str = "declared",
        contradiction: str = "a declared fact the provider contradicts; correct where it was declared",
    ) -> None:
        """Know nothing yet; name the client and what declared its facts, for the log.

        Args:
            client: The event name a learned fact's WARNING begins with, naming the client that learned it (the
                shipped client's is ``eval.providers``).
            declared_by: What the declared facts are called in the log (``registry=unknown``, ``declared=refuses``).
            contradiction: What the log says when a refusal contradicts a declared fact: where to correct it.
        """
        self._client = client
        self._declared_by = declared_by
        self._contradiction = contradiction
        self._facts: dict[tuple[str, str], SettingFacts] = {}
        self._lock = threading.Lock()

    def facts(self, provider: str, model: str) -> SettingFacts:
        """What has been learned of ``model`` on ``provider``; every fact unknown until a refusal.

        Args:
            provider: The provider as the caller names it (:func:`negotiate`'s ``provider``).
            model: The model id.

        Returns:
            The learned facts.
        """
        with self._lock:
            return self._facts.get((provider, model), SettingFacts())

    def learn(
        self,
        provider: str,
        model: str,
        refusals: Sequence[SettingRefusal],
        *,
        declared: SettingFacts,
        sent: WireSettings,
    ) -> None:
        """Keep what each refusal of a call says, now that the request without those settings was served.

        Logs each fact not already known once, at WARNING, with the provider's message (at most 500 characters; never
        a whole body), what the declared facts said of it, and what the served request carried in its place. An
        ``enforcement`` refusal is never kept, whoever hands it here: enforcement is dropped per call.

        Args:
            provider: The provider as the caller names it.
            model: The model id.
            refusals: The call's refusals the served request proved, in the order met.
            declared: What the caller declared of the model, for the log.
            sent: What the served request carried.
        """
        for refusal in refusals:
            if refusal.setting == "enforcement":
                continue
            with self._lock:
                known = self._facts.get((provider, model), SettingFacts())
                taught = refusal.teach(known)
                if taught == known:
                    continue
                self._facts[(provider, model)] = taught
            log.warning(
                "%s model refuses %s provider=%s model=%s %s=%s; the request was resent as %s and is sent so for the "
                "rest of this store's life: %s",
                self._client,
                refusal.label,
                provider,
                model,
                self._declared_by,
                self._said(refusal, declared),
                _settings_text(sent),
                refusal.message[:500],
            )

    def _said(self, refusal: SettingRefusal, declared: SettingFacts) -> str:
        """What the declared facts said of the refused setting: ``unknown``, ``refuses``, or a contradiction."""
        temperature = declared.accepts_temperature
        if refusal.beside_reasoning and declared.temperature_beside_reasoning is not None:
            temperature = declared.temperature_beside_reasoning
        cap = declared.max_output_tokens
        said = {
            "temperature": temperature,
            "budget": declared.reasoning_budget,
            "adaptive": declared.adaptive_reasoning,
            "disable": declared.reasoning_disable,
            "format": declared.structured_output,
            "effort": None if declared.reasoning_efforts is None else bool(declared.reasoning_efforts),
            "effort_word": None if declared.reasoning_efforts is None else refusal.word in declared.reasoning_efforts,
            "max_tokens": None if cap is None or refusal.limit is None else cap > refusal.limit,
            "enforcement": declared.enforcement,
        }[refusal.setting]
        if said is None:
            return "unknown"
        return f"accepts ({self._contradiction})" if said else "refuses"


def _settings_text(sent: WireSettings) -> str:
    """``sent`` as one line for the log."""
    reasoning = (
        f"budget {sent.reasoning_max_tokens}"
        if sent.reasoning_max_tokens is not None
        else f"effort {sent.reasoning_effort!r}"
        if sent.reasoning_effort is not None
        else "no reasoning parameter"
    )
    temperature = "no temperature" if sent.temperature is None else f"temperature {sent.temperature}"
    strict = "enforced" if sent.strict_output else "unenforced"
    return f"max_tokens={sent.max_tokens}, {reasoning}, {temperature}, {strict}"


@dataclass(frozen=True, slots=True, kw_only=True)
class Negotiated[R]:
    """A served request: the caller's reply, and what the request that produced it carried.

    Attributes:
        reply: What the caller's ``send`` returned for the served request.
        wire: The settings the served request was built from.
        notes: Why each setting of ``wire`` moved from what was asked, one sentence each, for a log.
        refusals: The refusals the served request answered, in the order met: what it was sent without, or moved
            for. A temperature dropped for a router's 404 and then put back, since dropping it alone did not place
            the request, is not among them. Each but ``enforcement`` is learned, except a temperature dropped for a
            404 when a later refusal was answered before the request was served (it proved nothing).
    """

    reply: R
    wire: WireSettings
    notes: tuple[str, ...]
    refusals: tuple[SettingRefusal, ...]

    @property
    def sent_settings(self) -> ClientRequestSettings | None:
        """What the served request carried, as the completion's ``sent_settings``: ``None`` when it named no cap."""
        return self.wire.reported()

    @property
    def temperature(self) -> float | None:
        """The temperature the served request carried, as the completion's ``temperature``: ``None`` for none."""
        return self.wire.temperature


async def negotiate[R](
    asked: WireSettings,
    *,
    provider_family: ProviderFamily,
    model: str,
    provider: str | None = None,
    send: Callable[[WireSettings, SettingFacts], Awaitable[R]],
    read_refusal: Callable[[Exception], SettingRefusal | None],
    learned: LearnedSettings,
    declared: SettingFacts | None = None,
) -> Negotiated[R]:
    """Send the closest settings the model is known to take; on a recognised refusal, learn it and send again.

    **What is sent.** :func:`closest_settings` over what is known of the model: what ``learned`` holds, over what the
    caller ``declared``, re-read at every attempt (another client sharing ``learned`` may have learned since).
    ``send`` builds its request from the settings it is handed (and, where its request shape depends on more than
    the settings, from the facts it is handed beside them: Anthropic's thinking, :func:`anthropic_thinking`).

    **What is answered.** When ``send`` raises, ``read_refusal`` reads the exception. A refusal is answered only
    when it is of a setting the refused request carried, and not one this call already answered, with a resend at
    the closest settings the model then takes, at most once per setting and :data:`MAX_RESENDS` times per call. A
    request the provider refused generated nothing, so a resend is not a hidden retry, and the call's price is one
    request's; its time is not: each refused send can take up to ``send``'s own timeout, so a wall-clock ceiling
    around this call allows ``MAX_RESENDS`` × that timeout past the served request's. In particular:

    * **A router's 404 under enforcement** (``enforcement``, naming nothing) is answered in a fixed order, whatever
      the facts say: a temperature the request carried goes first, keeping enforcement (a catalogue listing
      ``temperature`` says only that SOME endpoint takes one, never that one endpoint takes it beside the request's
      other parameters). If the resend without it is refused the same way, the temperature is put back and
      enforcement goes, for this call alone.
    * **An effort word refused** with the list of words the model takes is learned with that list, and the request
      resent once at the closest of them. Refused with no list, it is resent once at the closest word left. A second
      effort refusal in the call, of any word or of the parameter, is read as the parameter: the model is learned to
      take no effort word, and the request goes with none (on an Anthropic model that takes a budget, as a budget),
      so a call never walks the ladder into a failure.
    * **A temperature refused while the request reasoned** (an effort word other than ``none``, or a budget) is
      answered and learned as a refusal beside reasoning only: a call that does not reason still sends one.

    **What is learned.** Once a request is served, what it proved is kept in ``learned`` for the store's life (the
    process, or the factory that keeps it) and logged once at WARNING: every refusal it answered, except
    enforcement, which is never kept (a 404 can be an outage, and a client must not stay unenforced for life), and
    except a temperature dropped for a 404, which is kept only when the request that dropped it alone was the one
    served.

    **What propagates.** Anything ``read_refusal`` does not read as a refusal, a refusal of a setting the request
    did not carry or the call already answered, one that would teach nothing new, and the one after the
    :data:`MAX_RESENDS`-th resend, is raised as ``send`` raised it, and nothing is learned from the call.

    Args:
        asked: What the client was asked for, ``strict_output`` being this call's own enforcement.
        provider_family: The request shape the client speaks.
        model: The model id, as ``learned`` keys it.
        provider: The provider as the caller names it, which ``learned`` keys the model under and the log names;
            ``None`` for the family itself.
        send: The caller's request: given the settings to send and the facts they were chosen under, it sends one
            request and returns its reply, or raises what its transport raised.
        read_refusal: The caller's reading of what ``send`` raised: a :class:`SettingRefusal`, or ``None`` for anything
            else. Usually one of the shipped readers over the status and body the exception carries.
        learned: The store of learned facts, shared as the caller chooses.
        declared: What the caller knows of the model before any request (a registry's capabilities, a router's
            catalogue through :func:`~threetears.evals.negotiation.facts_from_router_catalogue`), below what was
            learned; ``None`` for nothing.

    Returns:
        The reply, with the settings the served request carried.

    Raises:
        Exception: What ``send`` raised, unchanged, for a failure the loop does not answer.
    """
    known = declared if declared is not None else SettingFacts()
    key = provider if provider is not None else provider_family
    refusals: list[SettingRefusal] = []
    resends = 0
    guess: SettingRefusal | None = None
    while True:
        facts = learned.facts(key, model).over(known)
        for refusal in refusals:
            facts = refusal.teach(facts)
        wire, notes = closest_settings(asked, facts, provider_family)
        try:
            reply = await send(wire, facts)
        except Exception as exc:
            read = read_refusal(exc)
            answer = None if resends >= MAX_RESENDS else _answerable(read, wire, facts, provider_family, refusals)
            if answer is None:
                raise
            resends += 1
            if answer.setting == "enforcement" and guess is not None:
                # Dropping the temperature alone did not place the request: it was not the temperature. Put it back.
                refusals = [answered for answered in refusals if answered is not guess]
                guess = None
            elif read is not None and read.setting == "enforcement" and answer.setting == "temperature":
                guess = answer
            refusals.append(answer)
            continue
        proved = [refusal for refusal in refusals if refusal is not guess or refusal is refusals[-1]]
        if proved:
            learned.learn(key, model, proved, declared=known, sent=wire)
        return Negotiated(reply=reply, wire=wire, notes=notes, refusals=tuple(refusals))


def _answerable(
    refusal: SettingRefusal | None,
    wire: WireSettings,
    facts: SettingFacts,
    family: ProviderFamily,
    answered: Sequence[SettingRefusal],
) -> SettingRefusal | None:
    """The refusal to answer with a resend, resolved to the setting it is of, or ``None`` to raise what was raised.

    Args:
        refusal: What the caller's reader made of the failure.
        wire: What the refused request carried.
        facts: What was known of the model when it was chosen.
        family: The request shape.
        answered: The refusals this call answered that still stand, in order.
    """
    if refusal is None:
        return None
    if refusal.setting == "enforcement":
        if not wire.strict_output:
            return None
        if wire.temperature is not None and not any(done.setting == "temperature" for done in answered):
            refusal = SettingRefusal("temperature", refusal.message)
    if refusal.setting == "effort_word":
        if any(done.setting in _EFFORT_REFUSALS for done in answered):
            refusal = SettingRefusal("effort", refusal.message)
        elif refusal.word is None:
            refusal = replace(refusal, word=wire.reasoning_effort)
    if refusal.setting == "temperature" and reasons(wire.reasoning_effort, wire.reasoning_max_tokens):
        refusal = replace(refusal, beside_reasoning=True)
    if not _carried(refusal, wire, facts, family):
        return None
    if any(_same(done, refusal) for done in answered):
        return None
    if refusal.teach(facts) == facts:
        return None
    return refusal


def _same(done: SettingRefusal, refusal: SettingRefusal) -> bool:
    """Whether ``refusal`` is of a setting the call already answered: the same setting, and for a word, that word.

    A temperature refused beside reasoning and one refused without it are two settings: the first is put back when
    reasoning goes, and its refusal then is a new fact.
    """
    return (done.setting, done.word, done.beside_reasoning) == (refusal.setting, refusal.word, refusal.beside_reasoning)


def _carried(refusal: SettingRefusal, wire: WireSettings, facts: SettingFacts, family: ProviderFamily) -> bool:
    """Whether the request built from ``wire`` under ``facts`` carried the setting ``refusal`` is of."""
    effort = wire.reasoning_effort
    # Anthropic's `none` is thinking disabled, not an effort parameter.
    reasoning = effort is not None and not (family == "anthropic" and effort == "none")
    match refusal.setting:
        case "temperature":
            return wire.temperature is not None
        case "effort_word":
            return reasoning and refusal.word == effort
        case "effort":
            return reasoning
        case "budget":
            return wire.reasoning_max_tokens is not None
        case "adaptive":
            return family == "anthropic" and effort not in (None, "none") and facts.adaptive_reasoning is not False
        case "disable":
            return effort == "none"
        case "max_tokens":
            return wire.max_tokens is not None and refusal.limit is not None and refusal.limit < wire.max_tokens
        case "format" | "enforcement":
            return wire.strict_output
