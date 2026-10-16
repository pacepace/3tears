"""The refusal readers for the common wires: a status and a JSON body in, a :class:`SettingRefusal` or ``None`` out.

A reader turns what a client saw of a failed request, its HTTP status and its JSON body, into a refusal, or ``None``
for anything else, which the negotiation loop then raises unchanged. Each takes a status and a body, never an
exception type, so it serves a client of raw HTTP and an SDK's error alike (an SDK keeps both on its exception:
``exc.status_code`` and ``exc.body``):

* :func:`read_anthropic_refusal`: Anthropic's ``invalid_request_error`` saying a parameter is not taken, listing
  the effort words it takes, or naming the model's own output cap.
* :func:`read_openai_refusal`: OpenAI's ``unsupported_parameter`` / ``unsupported_value`` naming it in ``param``, and
  an output cap above the model's own, naming it.
* :func:`read_router_refusal`: OpenRouter's (the ``router`` family's) 404 when ``require_parameters`` leaves no
  endpoint, and a 400 it relays from an upstream, read by that upstream's own shape.

Each reads structured fields first, and a message only for the parameter it names and a word saying it is not taken,
never as a whole sentence; a value out of range (a budget below Anthropic's floor) says nothing is unsupported and is
the caller's error, never read as a refusal. Two value refusals are the model's, not the caller's, and are read: an
effort word outside the list the provider gives of those it takes, and an output cap above the model's own, which
the provider names. This is the one module of the negotiation that reads a provider's error shape.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Final

from threetears.evals.negotiation.refusals import RefusedSetting, SettingRefusal
from threetears.evals.negotiation.settings import EFFORT_LADDER

__all__ = ["read_anthropic_refusal", "read_openai_refusal", "read_router_refusal"]

#: Words a provider's refusal uses when a parameter is not taken at all, as against a value out of range (Anthropic's
#: range error), which is the caller's to fix and is never resent. Anthropic's refusal names the parameter in its
#: message only ("`temperature` is deprecated for this model."), so the message is read for the parameter's name and
#: one of these words, never matched as a whole sentence.
_NOT_TAKEN_WORDS: Final = ("deprecated", "not supported", "unsupported", "does not support")

#: OpenAI's error codes for a parameter the model does not take (``Unsupported parameter: 'temperature' …``) and for
#: a value of it the model does not take (``Unsupported value: 'temperature' does not support 0.2 …``); either
#: arrives with ``param`` naming the parameter.
_OPENAI_NOT_TAKEN_CODES: Final = frozenset({"unsupported_parameter", "unsupported_value"})

#: An OpenAI ``unsupported_value`` message's list of what the model does take ("Supported values are: 'low',
#: 'medium', and 'high'."), read only to fill the accepted words of a refusal already read off ``code``.
_SUPPORTED_VALUES: Final = re.compile(r"supported values are:(?P<values>[^.]*)", re.IGNORECASE)

#: Anthropic's validation error for an effort word the model does not take, which lists the ones it does, seen live on
#: ``claude-haiku-5-5``: ``output_config.effort: Input should be 'low', 'medium', 'high', 'xhigh' or 'max'``. Read only
#: for the effort field, and only for the quoted words of the effort ladder it lists; the same "Input should be" on
#: any other field (``thinking.budget_tokens: Input should be greater than or equal to 1024``) is the caller's range
#: error and is not read.
_ANTHROPIC_EFFORT_LIST: Final = re.compile(r"\beffort: input should be (?P<values>.+)", re.IGNORECASE)

#: Anthropic's refusal of an output cap above the model's own, which it names: ``max_tokens: 64001 > 64000, which is
#: the maximum allowed number of output tokens for claude-...``.
_ANTHROPIC_CAP: Final = re.compile(r"\bmax_tokens: \d+ > (?P<limit>\d+), which is the maximum allowed", re.IGNORECASE)

#: OpenAI's refusal of an output cap above the model's own, arriving with ``param`` naming the cap: ``max_tokens is
#: too large: 200000. This model supports at most 128000 completion tokens, whereas you provided 200000.``
_OPENAI_CAP: Final = re.compile(r"supports at most (?P<limit>\d+) (?:completion|output) tokens", re.IGNORECASE)

#: The parameter names in an Anthropic message that say which setting it refuses, by setting. The one named first is
#: the refused one, since the message goes on to name what to send instead (``"thinking.type.enabled" is not
#: supported for this model. Use "thinking.type.adaptive" …``).
_ANTHROPIC_TOKENS: Final[Mapping[RefusedSetting, tuple[str, ...]]] = {
    "temperature": ("temperature",),
    "budget": ("thinking.type.enabled", "budget_tokens"),
    "adaptive": ("thinking.type.adaptive",),
    "disable": ("thinking.type.disabled",),
    "effort": ("effort",),
    "format": ("output_config.format", "structured output"),
}


def _mapping(value: object) -> Mapping[str, object]:
    """``value`` when it is a mapping, else an empty one, so a nested read of an absent field is ``None``."""
    return value if isinstance(value, Mapping) else {}


def _parsed(body: object) -> object:
    """``body`` as JSON: parsed from text or bytes (an SDK keeps the raw body), or as given; ``None`` if unreadable.

    A body that is not JSON is not a shape any reader knows, so the error it came with is raised unchanged.
    """
    if isinstance(body, bytes | bytearray):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str):
        try:
            return json.loads(body)
        except ValueError:
            # NOSILENT: a body that is not JSON is no shape a reader knows; None makes the reader answer "no refusal",
            # so the negotiation raises the provider's original error unchanged rather than losing it.
            return None
    return body


def _error(body: object) -> Mapping[str, object]:
    """The error object of a provider's body: its ``error`` member, or the body itself (an SDK kept the inner one)."""
    outer = _mapping(body)
    inner = outer.get("error")
    return inner if isinstance(inner, Mapping) else outer


def _message(error: Mapping[str, object]) -> str | None:
    """The provider's own message in an error object, or ``None`` when it carries none."""
    message = error.get("message")
    return message if isinstance(message, str) and message else None


def _listed_words(listed: str) -> tuple[str, ...]:
    """The effort words ``listed`` quotes (``'low'``, ``"low"`` or a backticked one), in ladder order."""
    return tuple(
        word for word in EFFORT_LADDER if any(quoted in listed for quoted in (f"'{word}'", f'"{word}"', f"`{word}`"))
    )


def _quoted_word(said: str) -> str | None:
    """The effort word the message quotes first (``'xhigh'``, ``"xhigh"`` or a backticked one), or ``None``."""
    found = [
        (said.find(quoted), word)
        for word in EFFORT_LADDER
        for quoted in (f"'{word}'", f'"{word}"', f"`{word}`")
        if quoted in said
    ]
    return min(found)[1] if found else None


def read_anthropic_refusal(status: int, body: object) -> SettingRefusal | None:
    """Whether Anthropic's answer refuses a request for a setting it carried, which, and in what words.

    A 400 ``invalid_request_error`` whose message says a parameter is not taken ("deprecated", "not supported",
    "unsupported", "does not support") and names it: ``temperature``; ``thinking.type.enabled`` or ``budget_tokens``
    (a budget); ``thinking.type.adaptive``; ``thinking.type.disabled``; ``effort`` (one word when the message quotes
    it, else the parameter); ``output_config.format`` or ``structured output``. The one named first is the refused
    one, since the message goes on to name what to send instead. Also two value refusals that are the model's: an
    effort word refused with the list of those taken (``output_config.effort: Input should be 'low', 'medium',
    'high', 'xhigh' or 'max'``, read as ``effort_word`` with that list as ``accepted`` and the word left to the loop,
    which fills the one the request carried), and an output cap above the model's own (``max_tokens: 64001 > 64000,
    which is the maximum allowed ...``, read as ``max_tokens`` with that ``limit``). Anything else (a range error, the
    credit notice Anthropic also sends as a 400) is ``None``.

    :param status: The HTTP status (``anthropic.APIStatusError.status_code``).
    :ptype status: int
    :param body: The JSON body, parsed or as text (``anthropic.APIStatusError.body``, the whole
        ``{"type": "error", "error": {...}}``).
    :ptype body: object
    :return: The refusal, or ``None``.
    :rtype: SettingRefusal | None
    """
    if status != 400:
        return None
    error = _error(_parsed(body))
    message = _message(error)
    if error.get("type") != "invalid_request_error" or message is None:
        return None
    said = message.lower()
    if (listed := _ANTHROPIC_EFFORT_LIST.search(message)) is not None and (accepted := _listed_words(listed["values"])):
        return SettingRefusal("effort_word", message, accepted=accepted)
    if (capped := _ANTHROPIC_CAP.search(message)) is not None:
        return SettingRefusal("max_tokens", message, limit=int(capped["limit"]))
    if not any(word in said for word in _NOT_TAKEN_WORDS):
        return None
    named: list[tuple[int, SettingRefusal]] = []
    for setting, tokens in _ANTHROPIC_TOKENS.items():
        found = [said.find(token) for token in tokens if token in said]
        if not found:
            continue
        if setting == "effort":
            word = _quoted_word(said)
            named.append(
                (
                    min(found),
                    SettingRefusal("effort_word", message, word=word) if word else SettingRefusal("effort", message),
                )
            )
        else:
            named.append((min(found), SettingRefusal(setting, message)))
    return min(named, key=lambda found: found[0])[1] if named else None


def read_openai_refusal(status: int, body: object) -> SettingRefusal | None:
    """Whether OpenAI's answer refuses a request for a setting it carried, which, and in what words.

    A 400 whose ``code`` is ``unsupported_parameter`` or ``unsupported_value`` and whose ``param`` names
    ``temperature`` or ``reasoning_effort`` (an ``unsupported_value`` there is the word the request carried, and its
    message's "Supported values are" list fills the words the model takes), or whose ``param`` is
    ``response_format`` with such a code or a message saying the format is not supported. And a ``param`` naming the
    output cap (``max_tokens`` or ``max_completion_tokens``) whose message says the model "supports at most" a number
    of tokens, read as ``max_tokens`` with that ``limit``. Read by ``param`` and ``code``, never by the message, except
    that list and that number.

    :param status: The HTTP status (``openai.APIStatusError.status_code``).
    :ptype status: int
    :param body: The JSON body, parsed or as text: ``{"error": {...}}`` as it arrives, or the inner error object an
        SDK keeps (``openai.APIStatusError.body``).
    :ptype body: object
    :return: The refusal, or ``None``.
    :rtype: SettingRefusal | None
    """
    if status != 400:
        return None
    error = _error(_parsed(body))
    param, code = error.get("param"), error.get("code")
    message = _message(error) or f"{code}: {param}"
    if param == "temperature" and code in _OPENAI_NOT_TAKEN_CODES:
        return SettingRefusal("temperature", message)
    if param == "reasoning_effort" and code in _OPENAI_NOT_TAKEN_CODES:
        if code == "unsupported_parameter":
            return SettingRefusal("effort", message)
        listed = _SUPPORTED_VALUES.search(message)
        accepted = _listed_words(listed.group("values")) if listed else ()
        return SettingRefusal("effort_word", message, accepted=accepted or None)
    if param == "response_format" and (code in _OPENAI_NOT_TAKEN_CODES or "not supported" in message.lower()):
        return SettingRefusal("format", message)
    if param in ("max_tokens", "max_completion_tokens") and (capped := _OPENAI_CAP.search(message)) is not None:
        return SettingRefusal("max_tokens", message, limit=int(capped["limit"]))
    return None


def read_router_refusal(status: int, body: object) -> SettingRefusal | None:
    """Whether OpenRouter's answer refuses a request for a setting it carried, which, and in what words.

    The ``router`` family's reader. OpenRouter answers
    ``{"error": {"code": <status>, "message": ..., "metadata": {...}}}``. Without
    ``require_parameters`` it drops a parameter the routed upstream does not take, so no refusal arrives; under it,
    an upstream that cannot take one is excluded, and when none is left it answers 404 ("No endpoints found that can
    handle the requested parameters."), naming no parameter: that is an ``enforcement`` refusal, which the loop
    answers in its fixed order (the temperature first, then enforcement) and only for a request that carried
    enforcement. A 400 an upstream returned is relayed with that upstream's own body as text in
    ``error.metadata.raw`` (and its name in ``metadata.provider_name``); it is read by its own shape, OpenAI's
    (``param`` and ``code``) or else Anthropic's. Anything else, OpenRouter's own 400s included, is ``None``.

    :param status: The HTTP status (``openrouter.errors.OpenRouterError.status_code`` on the SDK).
    :ptype status: int
    :param body: The JSON body, parsed or as text (``OpenRouterError.body`` on the SDK is the text).
    :ptype body: object
    :return: The refusal, or ``None``.
    :rtype: SettingRefusal | None
    """
    error = _error(_parsed(body))
    message = _message(error)
    if message is None:
        return None
    if status == 404:
        return SettingRefusal("enforcement", message)
    if status != 400:
        return None
    upstream = _parsed(_mapping(error.get("metadata")).get("raw"))
    if not isinstance(upstream, Mapping):
        return None
    if "param" in _error(upstream):
        return read_openai_refusal(400, upstream)
    return read_anthropic_refusal(400, upstream)
